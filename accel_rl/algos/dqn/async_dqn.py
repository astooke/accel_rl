
from accel_rl.algos.dqn.dqn import DQN
from accel_rl.algos.dqn.cat_dqn import CategoricalDQN
from accel_rl.algos.dqn.eps_rainbow import EpsRainbow

from accel_rl.optimizers.async.dqn_optimizer import AsyncDqnOptimizer


class AsyncDQN(DQN):

    def __init__(self, OptimizerCls=AsyncDqnOptimizer, **kwargs):
        super().__init__(OptimizerCls=OptimizerCls, **kwargs)

    def _get_default_sub_args(self):
        opt_args, eps_args, pri_args = super()._get_default_sub_args()
        opt_args.pop("update_method")
        opt_args["update_method_name"] = "rmsprop"
        return opt_args, eps_args, pri_args


class AsyncCategoricalDQN(CategoricalDQN):

    def __init__(self, OptimizerCls=AsyncDqnOptimizer, **kwargs):
        super().__init__(OptimizerCls=OptimizerCls, **kwargs)

    def _get_default_sub_args(self):
        opt_args, eps_args, pri_args = super()._get_default_sub_args()
        opt_args.pop("update_method")
        opt_args["update_method_name"] = "adam"
        return opt_args, eps_args, pri_args


class AsyncEpsRainbow(EpsRainbow):

    def __init__(self, OptimizerCls=AsyncDqnOptimizer, **kwargs):
        super().__init__(OptimizerCls=OptimizerCls, **kwargs)

    def _get_default_sub_args(self):
        opt_args, eps_args, pri_args = super()._get_default_sub_args()
        opt_args.pop("update_method")
        opt_args["update_method_name"] = "adam"
        return opt_args, eps_args, pri_args
