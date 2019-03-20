
""" UNDER CONSTRUCTION """


from functools import partial

import theano
import theano.tensor as T

from rllab.misc import ext
from accel_rl.optimizers.util import apply_grad_norm_clip, scale_conv_gradients


class DqnOptimizer(object):

    def __init__(
            self,
            learning_rate,
            update_method,
            update_method_args=None,
            grad_norm_clip=None,
            scale_conv_grads=False,
            layerwise_stats=False,
            ):
        self._learning_rate = learning_rate
        if update_method_args is None:
            update_method_args = dict()
        self._update_method = partial(update_method, **update_method_args)
        self._grad_norm_clip = grad_norm_clip
        self._scale_conv_grads = scale_conv_grads
        self._layerwise_stats = layerwise_stats

    def initialize(self, inputs, loss, target, priority_expr,
            givens=None, lr_mult=1):
        self._target = target
        params = target.get_params(trainable=True)
        gradients = theano.grad(loss, wrt=params, disconnected_inputs="ignore")

        if self._scale_conv_grads:
            gradients = scale_conv_gradients(params, gradients,
                scale=2 ** (-1 / 2))
        # if self._layerwise_stats:
        #     self._n_params = len(params)
        #     param_grad_norms = [T.sqrt(T.sum(g ** 2)) for g in gradients]

        gradients, grad_norm = apply_grad_norm_clip(gradients, self._grad_norm_clip)
        lr = self._learning_rate * lr_mult  # (lr_mult can be shared variable)
        updates, steps = self._update_method(gradients, params, learning_rate=lr)

        # step_norm = T.sqrt(sum(T.sum(s ** 2) for s in steps))
        # if self._layerwise_stats:
        #     param_step_norms = [T.sqrt(T.sum(s ** 2)) for s in steps]

        # outputs = [priority_expr, loss, grad_norm, step_norm]
        outputs = [priority_expr, loss]
        # if self._layerwise_stats:
        #     outputs += param_grad_norms + param_step_norms

        self._f_opt = ext.compile_function(
            inputs=inputs,
            outputs=outputs,
            updates=updates,
            givens=givens,
            log_name="grad_and_update",
        )

    def optimize(self, inputs):
        outputs = self._f_opt(*inputs)
        priority, loss = outputs
        # priority, loss, grad_norm, step_norm = outputs[:4]
        # ret = (priority, loss, grad_norm, step_norm)
        # if self._layerwise_stats:  # (group grads together, steps together)
        #     param_grad_norms = outputs[4:4 + self._n_params]
        #     param_step_norms = outputs[4 + self._n_params:]
        #     ret += (param_grad_norms, param_step_norms)
        return priority, loss, 0., 0.

    @property
    def parallelism_tag(self):
        return "single"
