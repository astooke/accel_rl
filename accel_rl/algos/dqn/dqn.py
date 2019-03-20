

import theano.tensor as T
from lasagne.updates import rmsprop

from accel_rl.algos.base import RLAlgorithm
from accel_rl.util.quick_args import save_args
from accel_rl.optimizers.single.dqn_optimizer import DqnOptimizer
from accel_rl.algos.dqn.replay_buffers.uniform import UniformReplayBuffer
from accel_rl.algos.dqn.replay_buffers.prioritized import PrioritizedReplayBuffer


class DQN(RLAlgorithm):

    def __init__(
            self,
            discount=0.99,
            batch_size=32,
            min_steps_learn=int(5e4),
            delta_clip=1,
            replay_size=int(1e6),
            training_intensity=8,  # avg number of training uses per datum
            target_update_steps=int(1e4),
            reward_horizon=1,
            OptimizerCls=None,
            optimizer_args=None,
            eps_greedy_args=None,
            double_dqn=False,
            dueling_dqn=False,  # Just a shortcut for optimizer args
            prioritized_replay=False,
            priority_args=None,
            ):
        save_args(vars(), underscore=False)

        opt_args, eps_args, pri_args = self._get_default_sub_args()

        if optimizer_args is not None:
            opt_args.update(optimizer_args)
        if OptimizerCls is None:
            OptimizerCls = DqnOptimizer
        self.optimizer = OptimizerCls(**opt_args)

        if eps_greedy_args is not None:
            eps_args.update(eps_greedy_args)
        self._eps_initial = eps_args["initial"]
        self._eps_final = eps_args["final"]
        self._eps_eval = eps_args["eval"]
        self._eps_anneal_steps = eps_args["anneal_steps"]

        if prioritized_replay:
            if priority_args is not None:
                pri_args.update(priority_args)
            self._priority_beta_initial = pri_args["beta_initial"]
            self._priority_beta_final = pri_args["beta_final"]
            self._priority_beta_anneal_steps = pri_args["beta_anneal_steps"]
            self._priority_args = dict(
                alpha=pri_args["alpha"],
                beta_initial=pri_args["beta_initial"],
                default_priority=pri_args["default_priority"],
            )

        self.need_extra_obs = False  # (for the sampler; should clean this up)

    def _get_default_sub_args(self):
        opt_args = dict(
            learning_rate=2.5e-4,
            update_method=rmsprop,
            grad_norm_clip=10 if self.dueling_dqn else None,
            update_method_args=dict(rho=0.95, epsilon=1e-6),
            scale_conv_grads=self.dueling_dqn,
        )
        eps_greedy_args = dict(  # original DQN values
            initial=1.,
            final=0.1,
            eval=0.05,
            anneal_steps=int(1e6),
        )
        d_clip = self.delta_clip
        priority_args = dict(
            alpha=0.6,
            beta_initial=0.4,
            beta_final=1.,
            beta_anneal_steps=50e6,
            default_priority=d_clip if d_clip is not None else 1.,
        )
        return opt_args, eps_greedy_args, priority_args

    def initialize(self, policy, env_spec, sample_size, horizon, mid_batch_reset):
        """params 'sample_size' and 'horizon' refer to the sampler only."""
        assert self.training_intensity * sample_size % self.batch_size == 0
        self._updates_per_optimize = int(
            (self.training_intensity * sample_size) // self.batch_size)
        # e.g. standard DQN: (8 * 4) // 32 = 1 updates_per_optimize
        print("from sample_bs: ", sample_size, " and opt_bs: ", self.batch_size,
            " and training intensity: ", self.training_intensity, " computed ",
            self._updates_per_optimize, " updates_per_optimize")

        self._eps_anneal_itr = max(1, self._eps_anneal_steps // sample_size)
        self._target_update_itr = max(1, self.target_update_steps // sample_size)
        self._min_itr_learn = self.min_steps_learn // sample_size
        if self.prioritized_replay:
            self._priority_beta_anneal_itr = \
                max(1, self._priority_beta_anneal_steps // sample_size)

        if not mid_batch_reset:
            raise NotImplementedError
        if int(policy.recurrent):
            raise NotImplementedError

        input_list, loss, priority_expr = self.build_loss(env_spec, policy)

        self.optimizer.initialize(
            inputs=input_list,
            loss=loss,
            target=policy,
            priority_expr=priority_expr,
        )

        replay_args = dict(
            env_spec=env_spec,
            size=self.replay_size,
            reward_horizon=self.reward_horizon,
            sampling_horizon=horizon,
            n_environments=sample_size // horizon,
            discount=self.discount,
            reward_dtype=loss.dtype,
        )
        if self.prioritized_replay:
            replay_args.update(self._priority_args)
            ReplayCls = PrioritizedReplayBuffer
        else:
            ReplayCls = UniformReplayBuffer
        self.replay_buffer = ReplayCls(**replay_args)

        self.policy = policy

    def build_loss(self, env_spec, policy):
        obs = env_spec.observation_space.new_tensor_variable('obs', extra_dims=1)
        next_obs = env_spec.observation_space.new_tensor_variable('next_obs', extra_dims=1)
        act = env_spec.action_space.new_tensor_variable('act', extra_dims=1)
        ret = T.vector('disc_n_return')
        term = T.bvector('terminal')
        if self.prioritized_replay:
            isw = T.vector('importance_sample_weights')

        if self.double_dqn:
            next_a = policy.actions_sym(next_obs)
            next_q = policy.target_q_at_a_sym(next_obs, next_a)
        else:
            next_q = policy.target_max_q_sym(next_obs)

        disc_next_q = (self.discount ** self.reward_horizon) * next_q
        y = ret + (1 - term) * disc_next_q
        q = policy.q_at_a_sym(obs, act)
        d = y - q
        losses = 0.5 * d ** 2
        if self.delta_clip is not None:
            # Huber loss:
            b = self.delta_clip * (abs(d) - self.delta_clip / 2)
            losses = T.switch(abs(d) <= self.delta_clip, losses, b)
        if self.prioritized_replay:
            losses = isw * losses
        loss = T.mean(losses)

        td_abs_errors = T.clip(abs(d), 0, self.delta_clip)

        input_list = [obs, next_obs, act, ret, term]
        if self.prioritized_replay:
            input_list.append(isw)

        return input_list, loss, td_abs_errors

    def optimize_policy(self, itr, samples_data):
        self.replay_buffer.append_data(samples_data)
        if itr < self._min_itr_learn:
            return None, dict()
        priorities = list()
        losses = list()
        for _ in range(self._updates_per_optimize):
            opt_minibatch = self.replay_buffer.sample_batch(self.batch_size)
            priority, loss = self.optimizer.optimize(opt_minibatch)
            if self.prioritized_replay:
                self.replay_buffer.update_batch_priorities(priority)
            priorities.extend(priority[::8])  # (downsample for stats)
            losses.append(loss)
        if itr % self._target_update_itr == 0:
            self.policy.update_target()
        self.update_epsilon(itr)
        if self.prioritized_replay:
            self.update_priority_beta(itr)
        opt_info = dict(Priority=priorities, Loss=losses)

        return opt_minibatch, opt_info

    def update_epsilon(self, itr):
        prog = min(1, itr / self._eps_anneal_itr)
        new_eps = prog * self._eps_final + (1 - prog) * self._eps_initial
        self.policy.set_epsilon(new_eps)

    def update_priority_beta(self, itr):
        prog = min(1, itr / self._priority_beta_anneal_itr)
        new_beta = prog * self._priority_beta_final + \
            (1 - prog) * self._priority_beta_initial
        self.replay_buffer.set_beta(new_beta)

    def set_n_itr(self, n_itr):
        self.n_itr = n_itr

    def prep_eval(self, itr):
        if itr > 0:
            self._prev_eps = self.policy.get_epsilon()
            self.policy.set_epsilon(self._eps_eval)

    def post_eval(self, itr):
        if itr > 0:
            self.policy.set_epsilon(self._prev_eps)

    @property
    def opt_info_keys(self):
        keys = ["Priority", "Loss"]
        return keys
