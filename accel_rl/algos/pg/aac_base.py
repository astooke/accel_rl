
import numpy as np
import theano
import theano.tensor as T

from accel_rl.util.quick_args import save_args
from accel_rl.buffers.batch import buffer_with_segs_view
from accel_rl.algos.base import RLAlgorithm
from accel_rl.algos.pg.util import gen_adv_est, discount_returns, zero_after_reset, \
    valids_mean, update_valids

LR_SCHEDULES = ["linear"]


class AdvActorCriticBase(RLAlgorithm):

    def __init__(
            self,
            discount,
            gae_lambda,
            v_loss_coeff=1,
            ent_loss_coeff=0.01,
            standardize_adv=False,
            lr_schedule=None,
            ):
        if lr_schedule is not None and lr_schedule not in LR_SCHEDULES:
            raise ValueError("Unrecognized lr_schedule: {}, should be None "
                "(for constant) or in: {}".format(lr_schedule, LR_SCHEDULES))
        save_args(vars(), underscore=False)
        self.need_extra_obs = True  # (signal sent to the sampler)

    def initialize(self, policy, env_spec, sample_size, horizon, mid_batch_reset):
        if mid_batch_reset and policy.recurrent:
            raise NotImplementedError

        obs = env_spec.observation_space.new_tensor_variable('obs', extra_dims=1)
        act = env_spec.action_space.new_tensor_variable('act', extra_dims=1)
        adv = T.vector('adv')
        ret = T.vector('ret')
        old_value = T.vector('old_value')

        dist = policy.distribution
        old_dist_info = {k: T.matrix('old_%s' % k) for k in dist.dist_info_keys}
        self._dist_info_keys = dist.dist_info_keys
        state_info = {k: T.matrix(k) for k in policy.state_info_keys}
        self._state_info_keys = policy.state_info_keys
        new_dist_info = policy.dist_info_sym(obs, state_info_vars=state_info)
        new_value = policy.value_sym(obs, state_info_vars=state_info)

        self._lr_mult = theano.shared(np.array(1., dtype=theano.config.floatX),
            name='lr_mult')

        if mid_batch_reset and not policy.recurrent:
            self._use_valids = False
            valids = None  # will be ignored inside valids_mean()
        else:
            self._use_valids = True
            valids = T.bvector('valids')  # dtype int8

        v_err = (new_value - ret) ** 2
        v_loss = self.v_loss_coeff * valids_mean(v_err, valids)
        ent = policy.distribution.entropy_sym(new_dist_info)
        ent_loss = - self.ent_loss_coeff * valids_mean(ent, valids)
        pi_loss = \
            self.pi_loss(policy, act, adv, old_dist_info, new_dist_info, valids)
        losses = (pi_loss, v_loss, ent_loss)

        pi_kl = valids_mean(dist.kl_sym(old_dist_info, new_dist_info), valids)
        v_kl = valids_mean((new_value - old_value) ** 2, valids)
        constraints = (pi_kl, v_kl)

        input_list = [obs, act, adv, ret, old_value]
        old_dist_info_list = [old_dist_info[k] for k in dist.dist_info_keys]
        state_info_list = [state_info[k] for k in policy.state_info_keys]
        input_list += old_dist_info_list + state_info_list

        opt_examples = dict(advantages=np.array(1, dtype=adv.dtype),
                            returns=np.array(1, dtype=ret.dtype),)
        if self._use_valids:
            input_list.append(valids)
            opt_examples["valids"] = np.array(1, dtype=np.int8)

        self.optimizer.initialize(
            inputs=input_list,
            losses=losses,
            constraints=constraints,
            target=policy,
            lr_mult=self._lr_mult,
        )

        self._opt_buf = buffer_with_segs_view(opt_examples, sample_size, horizon,
            shared=False)
        self._batch_size = sample_size
        self._mid_batch_reset = mid_batch_reset
        self._horizon = horizon

        self.policy = policy

    def set_n_itr(self, n_itr):
        self.n_itr = n_itr

    def optimize_policy(self, itr, samples_data):
        opt_data = self.process_samples(itr, samples_data)
        opt_input_values = self.prep_opt_inputs(itr, samples_data, opt_data)
        _, grad_norm = self.optimizer.optimize(opt_input_values)
        return opt_data, dict(GradNorm=grad_norm)

    def process_samples(self, itr, samples_data):
        gam, lam = (self.discount, self.gae_lambda)
        # NOTE: recurrent policy should automatically use latest hidden state
        # without updating in policy.value()
        last_values = self.policy.value(samples_data["extra_observations"])
        opt_buf = self._opt_buf

        if lam == 1:  # (discount returns is faster than GAE)
            for lv, path, opt in \
                    zip(last_values, samples_data.segs_view, opt_buf.segs_view):
                r, d, v = (path["rewards"], path["dones"], path["agent_infos"]["value"])
                ret, adv = (opt["returns"], opt["advantages"])
                discount_returns(r, d, lv, gam, ret_dest=ret)
                adv[:] = ret - v
        else:
            for lv, path, opt in \
                    zip(last_values, samples_data.segs_view, opt_buf.segs_view):
                r, d, v = (path["rewards"], path["dones"], path["agent_infos"]["value"])
                ret, adv = (opt["returns"], opt["advantages"])
                gen_adv_est(r, v, d, lv, gam, lam, adv_dest=adv, ret_dest=ret)

        if self._use_valids:
            for path, opt in zip(samples_data.segs_view, opt_buf.segs_view):
                update_valids(path, opt["valids"])
                adv, ret = (opt["advantages"], opt["returns"])
                v = path["agent_infos"]["value"]
                zero_after_reset(adv, ret, v, path)  # not necessary but maybe helpful

        if self.standardize_adv:
            all_adv = opt_buf["advantages"]
            if not self._use_valids:
                all_adv[:] = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-6)
            else:
                valids_idxs = opt_buf["valids"].nonzero()
                adv = all_adv[valids_idxs]
                all_adv[valids_idxs] = (adv - adv.mean()) / (adv.std() + 1e-6)

        return opt_buf

    def prep_opt_inputs(self, itr, samples_data, opt_data):
        agent_infos = samples_data["agent_infos"]
        opt_input_values = (
            samples_data["observations"],
            samples_data["actions"],
            opt_data["advantages"],
            opt_data["returns"],
            agent_infos["value"],
        )
        opt_input_values += tuple([agent_infos[k] for k in self._dist_info_keys])
        if self._state_info_keys:
            # Only feed in the initial (previous) hidden states
            all_prev_state_values = [agent_infos[k] for k in self._state_info_keys]
            init_state_values = [s[::self._horizon] for s in all_prev_state_values]
            opt_input_values += tuple(init_state_values)
        if self._use_valids:
            opt_input_values += (opt_data["valids"],)

        if self.lr_schedule == "linear":
            new_lr = np.array(max((self.n_itr - itr) / self.n_itr, 0.),
                dtype=theano.config.floatX)
            self._lr_mult.set_value(new_lr)

        return opt_input_values

    def pi_loss(self, act, adv, old_dist_info, new_dist_info, valids):
        raise NotImplementedError

    @property
    def opt_info_keys(self):
        return ["GradNorm"]
