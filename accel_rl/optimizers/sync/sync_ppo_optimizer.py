
import theano

from rllab.misc import ext

from accel_rl.optimizers.sync.base import BaseSyncOptimizer
from accel_rl.optimizers.util import make_shared_inputs, flat_shared_grad, \
    avg_grads_from_flat, apply_grad_norm_clip, iterate_mb_idxs


class SyncPpoOptimizer(BaseSyncOptimizer):

    def initialize(self, inputs, losses, constraints, target,
            givens=None, lr_mult=1):
        self._target = target
        loss = sum(losses)
        params = target.get_params(trainable=True)
        gradients = theano.grad(loss, wrt=params, disconnected_inputs='ignore')

        # Phase 0: Load data onto GPU (and shuffle indexes there)
        load_updates, givens, grad_inputs = make_shared_inputs(inputs, self._shuffle)

        # Phase 1: Compute gradient and save to GPU vector
        flat_grad, shared_grad, flat_update = flat_shared_grad(target, gradients)
        self._shared_grad = shared_grad

        # Phase 2: All-reduce gradient in-place in shared_grad, then reshape
        gradients, avg_factor_var = avg_grads_from_flat(shared_grad, params)
        self._avg_factor_var = avg_factor_var  # (set later as 1 / n_gpu)

        # Phase 3: Apply combined gradients locally
        gradients, grad_norm = apply_grad_norm_clip(gradients, self._grad_norm_clip)
        lr = self._learning_rate * lr_mult  # (lr_mult can be shared variable)
        updates = self._update_method(gradients, params, learning_rate=lr)

        self._f_load = ext.compile_function(
            inputs=inputs,
            updates=load_updates,
            log_name="load",
        )
        self._f_gradient = ext.compile_function(
            inputs=grad_inputs,
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
        self._load_data(inputs)
        return self._do_updates(len(inputs[0]))

    # Separated methods for better profiling. #################################

    def _do_updates(self, data_length):
        local_losses, grad_norms = (list(), list())
        batch_size, shuffle = (self._minibatch_size, self._shuffle)
        for i in range(self._epochs):
            for mb_idxs in iterate_mb_idxs(batch_size, data_length, shuffle):
                loss = self._compute_grad(mb_idxs)
                self._share_grad()
                grad_norm = self._do_one_update()
                local_losses.append(loss)
                grad_norms.append(grad_norm)
        return local_losses, grad_norms

    def _compute_grad(self, mb_idxs):
        # if shuffle, idxs will be a single list within a tuple
        return self._f_gradient(*mb_idxs)

    def _do_one_update(self):
        return self._f_update()
