
import numpy as np
import theano

from accel_rl.optimizers.base import BaseOptimizer


class BaseSyncOptimizer(BaseOptimizer):

    def init_comm(self, gpu_comm, rank, n_gpu):
        self._gpu_comm = gpu_comm
        self._n_gpu = n_gpu
        self._rank = rank
        if hasattr(self, "_avg_factor_var"):
            self._avg_factor_var.set_value(
                np.array(1. / n_gpu, dtype=theano.config.floatX))

    @property
    def parallelism_tag(self):
        return "synchronous"

    def _share_grad(self):
        g = self._shared_grad.container.data
        self._gpu_comm.all_reduce(src=g, op="sum", dest=g)
