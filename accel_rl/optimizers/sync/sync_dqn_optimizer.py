
import theano

from rllab.misc import ext

from accel_rl.optimizers.sync.base import BaseSyncOptimizer
from accel_rl.optimizers.util import apply_grad_norm_clip, \
    scale_conv_gradients, flat_shared_grad, avg_grads_from_flat


class SyncDqnOptimizer(BaseSyncOptimizer):

    def initialize(self, inputs, loss, target, priority_expr,
            givens=None, lr_mult=1):
        self._target = target
        params = target.get_params(trainable=True)
        gradients = theano.grad(loss, wrt=params, disconnected_inputs="ignore")

        if self._scale_conv_grads:
            gradients = scale_conv_gradients(params, gradients,
                scale=2 ** (-1 / 2))

        # Compute gradient and save to GPU vector.
        flat_grad, shared_grad, flat_update = flat_shared_grad(target, gradients)
        self._shared_grad = shared_grad

        # All-reduce gradient in-place in shared_grad, then reshape
        gradients, avg_factor_var = avg_grads_from_flat(shared_grad, params)
        self._avg_factor_var = avg_factor_var

        gradients, grad_norm = apply_grad_norm_clip(gradients, self._grad_norm_clip)
        lr = self._learning_rate * lr_mult  # (lr_mult can be shared variable)
        updates = self._update_method(gradients, params, learning_rate=lr)

        self._f_gradient = ext.compile_function(
            inputs=inputs,
            outputs=[priority_expr, loss],
            updates=[flat_update],
            givens=givens,
            log_name="gradient",
        )

        self._f_update = ext.compile_function(
            inputs=[],
            updates=updates,
            log_name="update",
        )

    def optimize(self, inputs):
        priority, loss = self._compute_grad(inputs)
        self._share_grad()
        self._update_params()
        return priority, loss

    # Separated methods for better profiling. #################################

    def _compute_grad(self, inputs):
        return self._f_gradient(*inputs)

    def _update_params(self):
        self._f_update()
