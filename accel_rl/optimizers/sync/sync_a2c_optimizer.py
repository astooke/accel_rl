
import theano

from rllab.misc import ext
from accel_rl.optimizers.sync.base import BaseSyncOptimizer

from accel_rl.optimizers.util import \
    apply_grad_norm_clip, flat_shared_grad, avg_grads_from_flat


class SyncA2cOptimizer(BaseSyncOptimizer):

    def initialize(self, inputs, losses, constraints, target,
            givens=None, lr_mult=1):
        self._target = target
        loss = sum(losses)
        params = target.get_params(trainable=True)
        gradients = theano.grad(loss, wrt=params, disconnected_inputs='ignore')

        # Phase 1: Compute gradient and save to GPU vector
        flat_grad, shared_grad, flat_update = flat_shared_grad(target, gradients)
        self._shared_grad = shared_grad

        # Phase 2: All-reduce gradient in-place in shared_grad, then reshape
        gradients, avg_factor_var = avg_grads_from_flat(shared_grad, params)
        self._avg_factor_var = avg_factor_var  # (set later as 1 / n_gpu)

        # Phase 3: Apply combined gradient locally
        gradients, grad_norm = apply_grad_norm_clip(gradients, self._grad_norm_clip)
        lr = self._learning_rate * lr_mult  # (lr_mult can be shared variable)
        updates = self._update_method(gradients, params, learning_rate=lr)

        self._f_grad = ext.compile_function(
            inputs=inputs,
            outputs=loss,
            updates=[flat_update],
            givens=givens,
            log_name="gradient",
        )
        self._f_update = ext.compile_function(
            inputs=[],
            outputs=grad_norm,
            updates=updates,
            log_name="update",
        )

    def optimize(self, inputs):
        local_loss = self._compute_grad(inputs)
        self._share_grad()
        grad_norm = self._do_update()
        return local_loss, grad_norm

    # Separated methods for better profiling. #################################

    def _compute_grad(self, inputs):
        return self._f_grad(*inputs)

    def _do_update(self):
        return self._f_update()
