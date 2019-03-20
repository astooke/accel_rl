
from functools import partial
import theano

from rllab.misc import ext
from accel_rl.optimizers.base import BaseOptimizer
from accel_rl.optimizers.util import \
    apply_grad_norm_clip, iterate_mb_idxs, make_shared_inputs


class PpoOptimizer(BaseOptimizer):

    def __init__(
            self,
            learning_rate,
            update_method,  # e.g. lasagne.updates.rmsprop
            update_method_args,
            epochs,
            minibatch_size,
            grad_norm_clip=None,
            shuffle=True,
            ):
        self._learning_rate = learning_rate
        self._update_method = partial(update_method, **update_method_args)
        self._epochs = epochs
        self._minibatch_size = minibatch_size
        self._shuffle = shuffle
        self._grad_norm_clip = grad_norm_clip

    def initialize(self, inputs, losses, constraints, target,
            givens=None, lr_mult=1):
        self._target = target
        loss = sum(losses)
        params = target.get_params(trainable=True)
        gradients = theano.grad(loss, wrt=params, disconnected_inputs='ignore')

        gradients, grad_norm = apply_grad_norm_clip(gradients, self._grad_norm_clip)
        lr = self._learning_rate * lr_mult  # (lr_mult can be shared variable)
        updates = self._update_method(gradients, params, learning_rate=lr)

        # Prepare to load data onto GPU (and shuffle indexes there).
        load_updates, givens, opt_inputs = make_shared_inputs(inputs, self._shuffle)

        self._f_load = ext.compile_function(
            inputs=inputs,
            updates=load_updates,
            log_name="load",
        )
        self._f_opt = ext.compile_function(
            inputs=opt_inputs,
            outputs=[loss, grad_norm],
            updates=updates,
            givens=givens,
            log_name="grad_and_update",
        )

    def optimize(self, inputs):
        self._load_data(inputs)
        return self._do_updates(len(inputs[0]))

    # Separated methods for better profiling. #################################

    def _load_data(self, inputs):
        self._f_load(*inputs)

    def _do_updates(self, data_length):
        losses, grad_norms = (list() for _ in range(2))
        batch_size, shuffle = (self._minibatch_size, self._shuffle)
        for i in range(self._epochs):
            for idxs in iterate_mb_idxs(batch_size, data_length, shuffle):
                # if shuffle, idxs will be a list within a tuple
                loss, grad_norm = self._f_opt(*idxs)
                losses.append(loss)
                grad_norms.append(grad_norm)
        return losses, grad_norms
