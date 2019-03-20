
import numpy as np
from functools import partial

import theano
import theano.tensor as T

from rllab.misc import ext
from accel_rl.optimizers.base import BaseOptimizer
from accel_rl.optimizers.util import apply_grad_norm_clip

# import ipdb


class A2cStatsOptimizer(BaseOptimizer):

    def __init__(
            self,
            learning_rate,
            update_method,  # e.g. accel_rl.optimizers.update_methods_stats.rmsprop
            update_method_args=None,
            grad_norm_clip=None,
            layerwise_stats=True,
            gradient_diversity=True,
            **kwargs
            ):
        super().__init__(**kwargs)
        self._learning_rate = learning_rate
        if update_method_args is None:
            update_method_args = dict()
        self._update_method = partial(update_method, **update_method_args)
        self._grad_norm_clip = grad_norm_clip
        self._layerwise_stats = True
        self._gradient_diversity = True

    def initialize(self, inputs, losses, constraints, target, batch_size, givens=None, lr_mult=1):
        self._target = target
        params = target.get_params(trainable=True)
        loss = T.mean(losses)
        gradients = theano.grad(loss, wrt=params, disconnected_inputs='ignore')

        # sep_grads = [theano.grad(losses[d], wrt=params, disconnected_inputs='ignore') for d in range(batch_size)]
        # lyr_sep_grads = [list() for _ in range(len(params))]
        # for d_grads in sep_grads:
        #     for d_grad, lyr_sep_grad in zip(d_grads, lyr_sep_grads):
        #         lyr_sep_grad.append(d_grad)
        # lyr_sep_grads = [T.stack(lsg) for lsg in lyr_sep_grads]

        # lyr_sep_grd_sqnorms = [T.sum(lsg ** 2) for lsg in lyr_sep_grads]
        # lyr_cmb_grd_sqnorms = [T.sum(T.sum(lsg, axis=0) ** 2) for lsg in lyr_sep_grads]
        # lyr_grad_diversities = [sep / cmb for sep, cmb in
        #     zip(lyr_sep_grd_sqnorms, lyr_cmb_grd_sqnorms)]
        # tot_sep_grd_sqnorms = T.sum(lyr_sep_grd_sqnorms)
        # tot_cmb_grd_sqnorms = T.sum(lyr_cmb_grd_sqnorms)
        # grad_diversity = tot_sep_grd_sqnorms / tot_cmb_grd_sqnorms
        # batch_size_bound = batch_size * grad_diversity

        # # Jacobians broke inside Theano.
        # # jacobians = theano.gradient.jacobian(losses, wrt=params, disconnected_inputs='ignore')

        # # lyr_sep_grd_sqnorms = [T.sum(j ** 2) for j in jacobians]
        # # lyr_cmb_grd_sqnorms = [T.sum(T.sum(j, axis=0) ** 2) for j in jacobians]
        # # lyr_grad_diversities = [sep / cmb for sep, cmb in
        # #     zip(lyr_sep_grd_sqnorms, lyr_cmb_grd_sqnorms)]
        # # tot_sep_grd_sqnorms = T.sum(lyr_sep_grd_sqnorms)
        # # tot_cmb_grd_sqnorms = T.sum(lyr_cmb_grd_sqnorms)
        # # grad_diversity = tot_sep_grd_sqnorms / tot_cmb_grd_sqnorms
        # # n = inputs[0].shape[0]
        # # batch_size_bound = n * grad_diversity

        # check_cmb_grd_sqnorms = sum(T.sum(g ** 2) for g in gradients) * batch_size

        if self._layerwise_stats:
            self._n_params = len(params)
            param_grad_norms = [T.sqrt(T.sum(g ** 2)) for g in gradients]

        gradients, grad_norm = apply_grad_norm_clip(gradients, self._grad_norm_clip)
        lr = self._learning_rate * lr_mult  # (lr_mult can be theano shared variable)
        updates, steps = self._update_method(gradients, params, learning_rate=lr)

        step_norm = T.sqrt(sum(T.sum(s ** 2) for s in steps))
        if self._layerwise_stats:
            param_step_norms = [T.sqrt(T.sum(s ** 2)) for s in steps]

        outputs = [grad_norm, step_norm]
        # grad_div_outputs = [grad_diversity, batch_size_bound]
        if self._layerwise_stats:
            outputs += param_grad_norms + param_step_norms
            # grad_div_outputs += lyr_grad_diversities
        # ipdb.set_trace()
        self._opt_fun["optimize"] = ext.compile_function(
            inputs=inputs,
            outputs=outputs,
            updates=updates,
            givens=givens,
            log_name="grad_and_update",
        )

        self._opt_fun["gradient"] = ext.compile_function(
            inputs=inputs,
            outputs=gradients,
            log_name="gradients_only",
        )

    def optimize(self, inputs):
        outputs = self._opt_fun["optimize"](*inputs)
        grad_norm, step_norm = outputs[:2]
        param_grad_norms = outputs[2:2 + self._n_params]
        param_step_norms = outputs[2 + self._n_params:]
        return grad_norm, step_norm, param_grad_norms, param_step_norms

    def grad_diversity(self, inputs):
        n = len(inputs[0])
        datum_0 = tuple(np.expand_dims(inp[0], axis=0) for inp in inputs)
        cum_grads = self._opt_fun["gradient"](*datum_0)
        cum_grads = [np.asarray(cg) for cg in cum_grads]
        # ipdb.set_trace()
        lyr_sep_grd_sqnorms = [np.sum(dg ** 2) for dg in cum_grads]
        for i in range(1, n):
            datum = tuple(np.expand_dims(inp[i], axis=0) for inp in inputs)
            datum_grads = self._opt_fun["gradient"](*datum)
            datum_grads = [np.asarray(dg) for dg in datum_grads]
            for lyr, cum, dg in zip(lyr_sep_grd_sqnorms, cum_grads, datum_grads):
                lyr += np.sum(dg ** 2)
                cum += dg
        lyr_cmb_grd_sqnorms = [np.sum(cg ** 2) for cg in cum_grads]
        lyr_grad_diversities = [sep / cmb for sep, cmb in
            zip(lyr_sep_grd_sqnorms, lyr_cmb_grd_sqnorms)]
        grad_diversity = np.sum(lyr_sep_grd_sqnorms) / np.sum(lyr_cmb_grd_sqnorms)
        batch_size_bound = n * grad_diversity

        # Yup looks good:
        # check_grads = self._opt_fun["gradient"](*inputs)
        # check_grads = [n * np.asarray(cg) for cg in check_grads]
        # for i, (check, cum) in enumerate(zip(check_grads, cum_grads)):
        #     if not np.allclose(check, cum):
        #         abs_diff = abs(check - cum)
        #         sum_abs_diff = np.sum(abs_diff)
        #         max_abs_diff = np.max(abs_diff)
        #         print("had bad check at {}, sum_abs_diff: {}, max_abs_diff: {}".format(
        #             i, sum_abs_diff, max_abs_diff))

        return grad_diversity, batch_size_bound, lyr_grad_diversities

