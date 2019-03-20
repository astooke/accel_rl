
from accel_rl.algos.dqn.dqn import DQN
from accel_rl.algos.dqn.cat_dqn import CategoricalDQN
from accel_rl.algos.dqn.eps_rainbow import EpsRainbow

from accel_rl.optimizers.sync.dqn_optimizer import SyncDqnOptimizer


class SyncDQN(DQN):

    def __init__(self, OptimizerCls=SyncDqnOptimizer, **kwargs):
        super().__init__(OptimizerCls=OptimizerCls, **kwargs)


class SyncCategoricalDQN(CategoricalDQN):

    def __init__(self, OptimizerCls=SyncDqnOptimizer, **kwargs):
        super().__init__(OptimizerCls=OptimizerCls, **kwargs)


class SyncEpsRainbow(EpsRainbow):

    def __init__(self, OptimizerCls=SyncDqnOptimizer, **kwargs):
        super().__init__(OptimizerCls=OptimizerCls, **kwargs)
