
"""
For Sampling Speed Test
"""

from accel_rl.algos.pg.a2c import A2C

import theano.tensor as T


class A2C_NoOpt(A2C):

    def initialize(self, policy, env_spec, batch_size, horizon, mid_batch_reset):
        if int(policy.recurrent):
            raise NotImplementedError

        obs = env_spec.observation_space.new_tensor_variable('obs', extra_dims=1)
        act = env_spec.action_space.new_tensor_variable('act', extra_dims=1)
        adv = T.vector('adv')
        ret = T.vector('ret')

        dist = policy.distribution
        self._dist_info_keys = dist.dist_info_keys
        old_dist_info = {k: T.matrix('old_%s' % k) for k in dist.dist_info_keys}
        new_dist_info = policy.dist_info_sym(obs)

        old_value = T.vector('old_value')
        new_value = policy.value_sym(obs)

        pi_loss = self.pi_loss(policy, act, adv, old_dist_info, new_dist_info)
        v_loss = self.v_loss_coeff * T.mean((new_value - ret) ** 2)
        ent = policy.distribution.entropy_sym(new_dist_info)
        ent_loss = - self.ent_loss_coeff * T.mean(ent)
        losses = (pi_loss, v_loss, ent_loss)

        pi_kl = T.mean(dist.kl_sym(old_dist_info, new_dist_info))
        v_kl = T.mean((new_value - old_value) ** 2)
        constraints = (pi_kl, v_kl)

        input_list = [obs, act, adv, ret, old_value, *old_dist_info.values()]

        # self._lr_mult = theano.shared(np.array(1, dtype=theano.config.floatX),
        #     name='lr_mult')

        # NOTE: turn this off for sampling speed tests; don't need to compile.
        # self.optimizer.initialize(
        #     inputs=input_list,
        #     losses=losses,
        #     constraints=constraints,
        #     target=policy,
        #     lr_mult=self._lr_mult,
        # )

        # opt_examples = dict(advantages=np.array(1, dtype=adv.dtype),
        #                     returns=np.array(1, dtype=ret.dtype))
        # self._opt_buf = buffer_with_segs_view(opt_examples, batch_size, horizon,
        #     shared=False)
        self._batch_size = batch_size
        self._mid_batch_reset = mid_batch_reset
        self._horizon = horizon

        self.policy = policy
