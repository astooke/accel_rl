
import theano

from rllab.misc import ext

from accel_rl.optimizers.async.base import BaseAsyncOptimizer
from accel_rl.optimizers.util import \
    apply_grad_norm_clip, flat_shared_grad, copy_params_from_flat
from accel_rl.optimizers.async.chunked_updates import \
    chunked_updates, whole_update
from accel_rl.optimizers.async.chunked_updates import \
    WHOLE_UPDATE_NAMES, CHUNKED_UPDATE_NAMES


class AsyncA2cOptimizer(BaseAsyncOptimizer):

    def __init__(
            self,
            learning_rate,
            update_method_name,
            update_method_args=None,
            n_update_chunks=1,
            grad_norm_clip=None,
            ):
        self._learning_rate = learning_rate
        self._grad_norm_clip = grad_norm_clip
        if n_update_chunks == 1 and update_method_name not in WHOLE_UPDATE_NAMES:
            raise ValueError("update method '{}' not available for NON-chunked "
                "updates, choose from: {}".format(update_method_name, WHOLE_UPDATE_NAMES))
        elif update_method_name not in CHUNKED_UPDATE_NAMES:
            raise ValueError("update method '{}' not available, for CHUNKED "
                "updates, choose from: {}".format(update_method_name, CHUNKED_UPDATE_NAMES))
        self._update_method_name = update_method_name
        self._update_method_args = update_method_args
        self.n_update_chunks = n_update_chunks
        if n_update_chunks == 1:
            self._push_update = self._single_lock_push
        else:
            self._push_update = self._cycle_locks_push

    def initialize(self, inputs, losses, constraints, target, lr_mult=1):
        self._target = target
        loss = sum(losses)
        params = target.get_params(trainable=True)
        gradients = theano.grad(loss, wrt=params, disconnected_inputs='ignore')

        gradients, grad_norm = apply_grad_norm_clip(gradients, self._grad_norm_clip)

        # Phase 1: Compute gradient and save to GPU vector
        flat_grad, shared_grad, flat_update = flat_shared_grad(target, gradients)

        # Phase 2: apply gradient chunks to update central params; e.g. rmsprop
        lr = self._learning_rate * lr_mult
        if self.n_update_chunks > 1:
            updates_args = (shared_grad, lr, self.n_update_chunks, self._update_method_args)
            chunk_inputs, outputs_list, updates, idxs = \
                chunked_updates(self._update_method_name, updates_args)
            self._chunk_idxs = idxs
        else:
            whole_inputs, whole_outputs, whole_updates = \
                whole_update(self._update_method_name, shared_grad, lr, self._update_method_args)

        # Phase 3: copy new param values from shared_grad to params
        copy_updates = copy_params_from_flat(params, shared_grad)

        # Phase 1
        self._f_gradient = ext.compile_function(
            inputs=inputs,
            outputs=[loss, grad_norm],
            updates=[flat_update],
            log_name="gradient",
        )

        # Phase 2
        if self.n_update_chunks > 1:
            f_update_chunks = list()
            for i, (outputs, update) in enumerate(zip(outputs_list, updates)):
                f_update_chunks.append(ext.compile_function(
                    inputs=chunk_inputs,
                    outputs=outputs,
                    updates=[update],
                    log_name="update_chunk_{}".format(i))
                )
            self._f_update_chunks = f_update_chunks
        else:
            self._f_update = ext.compile_function(
                inputs=whole_inputs,
                outputs=whole_outputs,
                updates=whole_updates,
                log_name="update",
            )

        # Phase 3
        self._f_copy = ext.compile_function(
            inputs=[],
            updates=copy_updates,
            log_name="copy_params",
        )

    def optimize(self, inputs):
        loss, grad_norm = self._compute_grad(inputs)
        self._push_update()
        self._f_copy()
        return loss, grad_norm

    # Separated methods for better profiling. #################################

    def _compute_grad(self, inputs):
        return self._f_gradient(*inputs)
