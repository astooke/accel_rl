
"""
This class provides the interface to the algorithm, so the algorithm doesn't
touch the underlying parallel sampler.
"""

import numpy as np
from accel_rl.util.quick_args import save_args


class Sampler(object):

    def initialize(self, **kwargs):
        raise NotImplementedError

    def policy_init(self, policy):
        raise NotImplementedError

    def obtain_samples(self, itr):
        raise NotImplementedError

    def shutdown_worker(self):
        raise NotImplementedError

    @property
    def alternating(self):
        return False


class BaseMbSampler(Sampler):

    def __init__(
            self,
            EnvCls,
            env_args,
            horizon,
            n_parallel=1,
            envs_per=1,
            max_path_length=np.inf,
            mid_batch_reset=True,
            max_decorrelation_steps=2000,
            profile_pathname=None,
            ):

        save_args(vars(), underscore=False)
        self.common_kwargs = vars(self).copy()
        self.common_kwargs.pop("n_parallel")

    @property
    def total_n_envs(self):
        return self._total_n_envs

