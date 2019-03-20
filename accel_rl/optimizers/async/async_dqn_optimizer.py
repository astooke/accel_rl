
import theano
# import theano.tensor as T

from rllab.misc import ext

from accel_rl.optimizers.async.chunked_updates import chunked_updates

from accel_rl.util.quick_args import save_args
from accel_rl.optimizers.async.base import BaseAsyncOptimizer

from accel_rl.optimizers.util import apply_grad_norm_clip, flat_shared_grad, \
    copy_params_from_flat, scale_conv_gradients


class AsyncDqnOptimizer(BaseAsyncOptimizer):

    def __init__(
            self,
            learning_rate,
            update_method_name,
            update_method_args,
            grad_norm_clip=None,
            scale_conv_grads=False,
            n_update_chunks=3,
            ):
        assert update_method_name in ["rmsprop", "adam"]
        save_args(vars(), underscore=True)
        self.n_update_chunks = n_update_chunks
        if n_update_chunks == 1:
            self._push_update = self._single_lock_push
        else:
            self._push_update = self._cycle_locks_push

    def initialize(self, inputs, loss, target, priority_expr,
            givens=None, lr_mult=1):
        self._target = target
        params = target.get_params(trainable=True)
        gradients = theano.grad(loss, wrt=params, disconnected_inputs="ignore")

        if self._scale_conv_grads:
            gradients = scale_conv_gradients(params, gradients,
                scale=2 ** (-1 / 2))

        gradients, grad_norm = apply_grad_norm_clip(gradients, self._grad_norm_clip)

        # Phase 1: Compute gradient and save to GPU vector
        flat_grad, shared_grad, flat_update = flat_shared_grad(target, gradients)

        # Phase 2: apply gradient chunks to central params
        lr = self._learning_rate * lr_mult
        updates_args = (shared_grad, lr, self._n_update_chunks, self._update_method_args)
        chunk_inputs, outputs_list, updates, idxs = \
            chunked_updates(self._update_method_name, updates_args)
        self._chunk_idxs = idxs

        # Phase 3: copy new param values from sared_grad to params
        copy_updates = copy_params_from_flat(params, shared_grad)

        if self._update_method_name == "adam":
            copy_updates.append(updates.pop())  # (Move the t update)

        # Phase 1
        self._f_gradient = ext.compile_function(
            inputs=inputs,
            outputs=[priority_expr, loss, grad_norm],
            updates=[flat_update],
            givens=givens,
            log_name="gradient",
        )

        # Phase 2
        f_update_chunks = list()
        for i, (outputs, update) in enumerate(zip(outputs_list, updates)):
                f_update_chunks.append(ext.compile_function(
                    inputs=chunk_inputs,
                    outputs=outputs,
                    updates=[update],
                    log_name="update_chunk_{}".format(i))
                )
        self._f_update_chunks = f_update_chunks

        # Phase 3
        self._f_copy = ext.compile_function(
            inputs=[],
            updates=copy_updates,
            log_name="copy_params",
        )

    def optimize(self, inputs):
        priority, loss = self._compute_grad(inputs)
        self._push_update()
        self._f_copy()
        return priority, loss

    def _compute_grad(self, inputs):
        return self._f_gradient(*inputs)
