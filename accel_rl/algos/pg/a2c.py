

import lasagne.updates

from accel_rl.algos.pg.aac_base import AdvActorCriticBase
from accel_rl.algos.pg.util import valids_mean

from accel_rl.optimizers.single.a2c_optimizer import A2cOptimizer
from accel_rl.optimizers.sync.sync_a2c_optimizer import SyncA2cOptimizer
from accel_rl.optimizers.async.async_a2c_optimizer import AsyncA2cOptimizer


class BaseA2C(AdvActorCriticBase):

    def __init__(
            self,
            OptimizerCls,
            optimizer_args=None,
            discount=0.99,
            gae_lambda=1,
            v_loss_coeff=0.25,
            **kwargs
            ):
        default_optimizer_args = dict(
            learning_rate=7e-4,
            update_method=lasagne.updates.rmsprop,
            update_method_args=dict(),
            grad_norm_clip=0.5,
        )
        if optimizer_args is None:
            optimizer_args = default_optimizer_args
        else:
            for k, v in default_optimizer_args.items():
                if k not in optimizer_args:
                    optimizer_args[k] = v
        self.optimizer = OptimizerCls(**optimizer_args)
        super().__init__(
            discount=discount,
            gae_lambda=gae_lambda,
            v_loss_coeff=v_loss_coeff,
            **kwargs)

    def pi_loss(self, policy, act, adv, old_dist_info, new_dist_info, valids):
        logli = policy.distribution.log_likelihood_sym(act, new_dist_info)
        pi_loss = - valids_mean(logli * adv, valids)
        return pi_loss


class A2C(BaseA2C):

    """ Single GPU """

    def __init__(self, OptimizerCls=A2cOptimizer, **kwargs):
        super().__init__(OptimizerCls=OptimizerCls, **kwargs)


class mA2C(BaseA2C):

    """ Multi-GPU Synchronous"""

    def __init__(self, OptimizerCls=SyncA2cOptimizer, **kwargs):
        super().__init__(OptimizerCls=OptimizerCls, **kwargs)


class mA3C(BaseA2C):

    """ Multi-GPU Asynchronous"""

    def __init__(
            self,
            OptimizerCls=AsyncA2cOptimizer,
            optimizer_args=None,
            **kwargs
            ):
        default_optimizer_args = dict(
            update_method_name="rmsprop",
            n_update_chunks=3,
        )
        if optimizer_args is None:
            optimizer_args = default_optimizer_args
        else:
            for k, v in default_optimizer_args.items():
                if k not in optimizer_args:
                    optimizer_args[k] = v
        super().__init__(
            OptimizerCls=OptimizerCls,
            optimizer_args=optimizer_args,
            **kwargs
        )
