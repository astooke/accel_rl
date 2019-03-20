
import numpy as np
import theano
import theano.tensor as T

from accel_rl.algos.adv_actor_critic import AdvActorCritic
from accel_rl.optimizers.single.fo_stats_optimizer import FOStatsOptimizer
from accel_rl.optimizers.update_methods_stats import rmsprop
from accel_rl.buffers.batch import buffer_with_segs_view, combine_distinct_buffers


class StatsA2C(AdvActorCritic):

    def __init__(
            self,
            OptimizerCls=FOStatsOptimizer,
            optimizer_args=None,
            discount=0.99,
            gae_lambda=1,
            v_loss_coeff=0.25,
            grad_diversity_itrs=10,
            **kwargs
            ):
        opt_args = dict(
            num_slices=1,
            learning_rate=7e-4,
            update_method=rmsprop,
            update_method_args=dict(),
            grad_norm_clip=None,
        )
        if optimizer_args is not None:
            opt_args.update(optimizer_args)
        self.optimizer = OptimizerCls(**opt_args)
        super().__init__(
            discount=discount,
            gae_lambda=gae_lambda,
            v_loss_coeff=v_loss_coeff,
            **kwargs)
        self.layerwise_stats = True
        self.grad_diversity_itrs = grad_diversity_itrs

    def initialize(self, policy, env_spec, sample_size, horizon, mid_batch_reset):
        if int(policy.recurrent):
            raise NotImplementedError
        assert mid_batch_reset

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

        self._lr_mult = theano.shared(np.array(1., dtype=theano.config.floatX),
            name='lr_mult')

        # Do NOT take the mean (or sum) of the losses, keep them separate, for
        # gradient diversity.
        v_errors = (new_value - ret) ** 2
        v_losses = self.v_loss_coeff * v_errors
        ents = policy.distribution.entropy_sym(new_dist_info)
        ent_losses = - self.ent_loss_coeff * ents
        logli = policy.distribution.log_likelihood_sym(act, new_dist_info)
        pi_losses = - logli * adv
        losses = pi_losses + v_losses + ent_losses  # still a vector

        pi_kl = T.mean(dist.kl_sym(old_dist_info, new_dist_info))
        v_kl = T.mean((new_value - old_value) ** 2)
        constraints = (pi_kl, v_kl)

        input_list = [obs, act, adv, ret, old_value, *old_dist_info.values()]

        opt_examples = dict(advantages=np.array(1, dtype=adv.dtype),
                            returns=np.array(1, dtype=ret.dtype),)

        self.optimizer.initialize(
            inputs=input_list,
            losses=losses,
            constraints=constraints,
            target=policy,
            lr_mult=self._lr_mult,
            batch_size=sample_size,
        )

        self._opt_buf = buffer_with_segs_view(opt_examples, sample_size, horizon,
            shared=False)
        self._batch_size = sample_size
        self._mid_batch_reset = mid_batch_reset
        self._horizon = horizon

        self.policy = policy

    def optimize_policy(self, itr, samples_data):
        opt_data = self.process_samples(itr, samples_data)
        opt_input_values = self.prep_opt_inputs(itr, samples_data, opt_data)
        opt_stats = self.optimizer.optimize(opt_input_values)
        if self.layerwise_stats:
            gn, sn, pgn, psn = opt_stats
        else:
            gn, sn = opt_stats
        opt_info = dict(
            GradNorm=gn,
            StepNorm=sn,
        )
        if self.layerwise_stats:
            for stat_list, stat_name in zip(
                    [pgn, psn], ["GradNorm", "StepNorm"]):
                for stat, name in zip(stat_list, self.policy.param_short_names):
                    opt_info[name + "_" + stat_name] = stat
        if itr % self.grad_diversity_itrs == 0:
            grad_diversity, batch_size_bound, lyr_grad_diversities = \
                self.optimizer.grad_diversity(opt_input_values)
            opt_info["GradDvrsty"] = grad_diversity
            opt_info["BatchSizeBound"] = batch_size_bound
            for dvrsty, name in zip(lyr_grad_diversities, self.policy.param_short_names):
                opt_info[name + "_GradDvrsty"] = dvrsty
        return opt_data, opt_info

    @property
    def opt_info_keys(self):
        keys = ["GradNorm", "StepNorm", "GradDvrsty", "BatchSizeBound"]
        if True:  # self.layerwise_stats:
            _keys = keys[:3]
            for name in self.policy.param_short_names:
                for k in _keys:
                    keys += [name + "_" + k]
        return keys
