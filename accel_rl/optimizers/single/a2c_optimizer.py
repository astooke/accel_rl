

from functools import partial
import theano

from rllab.misc import ext
from accel_rl.optimizers.base import BaseOptimizer
from accel_rl.optimizers.util import apply_grad_norm_clip


class A2cOptimizer(BaseOptimizer):

    def __init__(
            self,
            learning_rate,
            update_method,  # e.g. lasagne.updates.rmsprop
            update_method_args=None,
            grad_norm_clip=None,
            ):
        self._learning_rate = learning_rate
        if update_method_args is None:
            update_method_args = dict()
        self._update_method = partial(update_method, **update_method_args)
        self._grad_norm_clip = grad_norm_clip

    def initialize(self, inputs, losses, constraints, target,
            givens=None, lr_mult=1):
        self._target = target
        loss = sum(losses)
        params = target.get_params(trainable=True)
        gradients = theano.grad(loss, wrt=params, disconnected_inputs='ignore')

        gradients, grad_norm = apply_grad_norm_clip(gradients, self._grad_norm_clip)
        lr = self._learning_rate * lr_mult  # (lr_mult can be theano shared variable)
        updates = self._update_method(gradients, params, learning_rate=lr)

        self._f_opt = ext.compile_function(
            inputs=inputs,
            outputs=[loss, grad_norm],
            updates=updates,
            givens=givens,
            log_name="grad_and_update",
        )

    def optimize(self, inputs):
        return self._f_opt(*inputs)

    @property
    def parallelism_tag(self):
        return "single"
