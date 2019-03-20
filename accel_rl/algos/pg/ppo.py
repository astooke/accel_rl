import theano.tensor as T
import lasagne.updates

from accel_rl.algos.pg.aac_base import AdvActorCriticBase
from accel_rl.algos.pg.util import valids_mean
from accel_rl.optimizers.single.ppo_optimizer import PpoOptimizer
from accel_rl.optimizers.sync.sync_ppo_optimizer import SyncPpoOptimizer
from accel_rl.optimizers.async.sync_ppo_optimizer import AsyncPpoOptimizer


class BasePPO(AdvActorCriticBase):

    def __init__(
            self,
            OptimizerCls,
            optimizer_args=None,
            discount=0.99,
            gae_lambda=0.95,
            clip_param=0.2,
            **kwargs
            ):
        default_optimizer_args = dict(
            num_slices=1,
            learning_rate=1e-3,
            epochs=4,
            minibatch_size=64 * 8,
            update_method=lasagne.updates.adam,
            update_method_args=dict(epsilon=1e-5),
            grad_norm_clip=None,
            shuffle=True,
        )
        if optimizer_args is None:
            optimizer_args = default_optimizer_args
        else:
            for k, v in default_optimizer_args.items():
                if k not in optimizer_args:
                    optimizer_args[k] = v
        self.optimizer = OptimizerCls(**optimizer_args)
        self.clip_param = clip_param
        super().__init__(discount=discount, gae_lambda=gae_lambda, **kwargs)

    def pi_loss(self, policy, act, adv, old_dist_info, new_dist_info, valids):
        ratio = policy.distribution.likelihood_ratio_sym(
            act, old_dist_info, new_dist_info)
        surr_1 = ratio * adv
        clip_param = self.clip_param * self._lr_mult
        clipped_ratio = T.clip(ratio, 1. - clip_param, 1. + clip_param)
        surr_2 = clipped_ratio * adv
        surr = T.minimum(surr_1, surr_2)
        pi_loss = - valids_mean(surr, valids)
        return pi_loss


class PPO(BasePPO):

    """Single GPU"""

    def __init__(self, OptimizerCls=PpoOptimizer, **kwargs):
        super().__init__(OptimizerCls=OptimizerCls, **kwargs)


class mPPO(BasePPO):

    """Multi-GPU Synchronous"""

    def __init__(self, OptimizerCls=SyncPpoOptimizer, **kwargs):
        super().__init__(OptimizerCls=OptimizerCls, **kwargs)


class mAPPO(BasePPO):

    """Multi-GPU Asynchronous"""

    def __init__(self, OptimizerCls=AsyncPpoOptimizer, **kwargs):
        super().__init__(OptimizerCls=OptimizerCls, **kwargs)
