

from accel_rl.algos.dqn.dqn import DQN
from accel_rl.optimizers.single.dqn_optimizer import DqnOptimizer


class TwoLearnerDqn(DQN):

    def __init__(
            self,
            two_batch_size,
            two_optimizer_args=None,
            layerwise_stats=False,
            **kwargs
            ):
        super().__init__(layerwise_stats=layerwise_stats, **kwargs)
        if layerwise_stats:
            raise NotImplementedError
        self.two_batch_size = two_batch_size
        two_opt_args, _, _ = self._get_default_sub_args()
        if two_optimizer_args is not None:
            two_opt_args.update(two_optimizer_args)
        self.two_optimizer = DqnOptimizer(**two_opt_args)

    def initialize(self, policy, env_spec, sample_size, horizon, mid_batch_reset,
            two_policy):
        super().initialize(policy, env_spec, sample_size, horizon, mid_batch_reset)
        twobs = self.two_batch_size
        bs = self.batch_size
        ti = self.training_intensity
        if twobs == bs:
            self._two_updates_per_optimize = self._updates_per_optimize
        elif twobs > bs:
            if ti * sample_size < twobs:
                assert twobs % (ti * sample_size) == 0
                self._optimizes_per_two_update = int(twobs // (ti * sample_size))
            else:
                assert (ti * sample_size) % twobs == 0
                self._two_updates_per_optimize = int((ti * sample_size) // twobs)
        else:
            assert (ti * sample_size) % twobs == 0
            self._two_updates_per_optimize = int((ti * sample_size) // twobs)

        if hasattr(self, "_two_updates_per_optimize"):
            print("\n Had batch_size: ", bs, " two_batch_size: ", twobs,
                "\n\tcomputed ", self._two_updates_per_optimize, " two_updates_per_optimize")
        else:
            print("\n Had batch_size: ", bs, " two_batch_size: ", twobs,
                "\n\tcomputed ", self._optimizes_per_two_update, ", optimizes_per_two_update")

        input_list, loss, priority_expr = self.build_loss(env_spec, two_policy)

        self.two_optimizer.initialize(
            inputs=input_list,
            loss=loss,
            target=two_policy,
            priority_expr=priority_expr,
        )

        two_policy.set_param_values(policy.get_param_values())
        two_policy.update_target()
        two_policy.set_epsilon(self._eps_eval)
        self.two_policy = two_policy

    def optimize_policy(self, itr, samples_data):
        opt_minibatch, opt_info = super().optimize_policy(itr, samples_data)
        if itr < self._min_itr_learn:
            return None, dict()

        priorities = list()
        losses = list()
        grad_norms = list()
        step_norms = list()

        if hasattr(self, "_optimizes_per_two_update"):
            if itr % self._optimizes_per_two_update == 0:
                two_minibatch = self.replay_buffer.sample_batch(self.two_batch_size)
                two_outputs = self.two_optimizer.optimize(two_minibatch)
                pr, lo, gn, sn = two_outputs
                priorities.extend(pr[::8])
                losses.append(lo)
                grad_norms.append(gn)
                step_norms.append(sn)
        elif hasattr(self, "_two_updates_per_optimize"):
            for _ in range(self._two_updates_per_optimize):
                two_minibatch = self.replay_buffer.sample_batch(self.two_batch_size)
                two_outputs = self.two_optimizer.optimize(two_minibatch)
                pr, lo, gn, sn = two_outputs
                priorities.extend(pr[::8])
                losses.append(lo)
                grad_norms.append(gn)
                step_norms.append(sn)
        else:
            raise KeyError("Couldn't find how many two updates to do.")

        if itr % self._target_update_itr == 0:
            self.two_policy.update_target()

        two_info = dict(
            twoPriority=priorities,
            twoLoss=losses,
            twoGradNorm=grad_norms,
            twoStepNorm=step_norms,
        )
        opt_info.update(two_info)

        return opt_minibatch, opt_info

    @property
    def opt_info_keys(self):
        keys = super().opt_info_keys
        keys += ["twoPriority", "twoLoss", "twoGradNorm", "twoStepNorm"]
        return keys

