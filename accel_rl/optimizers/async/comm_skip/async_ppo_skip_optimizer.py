
from rllab.misc import ext

from accel_rl.optimizers.async.base import BaseAsyncOptimizer
from accel_rl.optimizers.util import \
    apply_grad_norm_clip, iterate_mb_idxs, make_shared_inputs
# from accel_rl.optimizers.async.chunked_updates import chunked_updates
# from accel_rl.optimizers.async.periodic_sync import *
from accel_rl.optimizers.async.epoch_adam_util import *


class AsyncPpoSkipOptimizer(BaseAsyncOptimizer):

    """This optimizer is specific to Adam update rule, and allows multiple
    local gradient update steps between syncs to the central parameters."""

    def __init__(
            self,
            learning_rate,
            epochs,
            minibatch_size,
            update_method_name="adam",
            update_method_args=None,
            grad_norm_clip=0.5,
            shuffle=True,
            sync_period=1,   # (i.e. if sync_period > 1, some syncs "skipped")
            # n_update_chunks=1,
            ):
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._minibatch_size = minibatch_size
        if update_method_args is None:
            update_method_args = dict()
        self._update_method_args = update_method_args
        assert update_method_name == "adam"
        self._update_method_name = update_method_name
        self._shuffle = shuffle
        self._grad_norm_clip = grad_norm_clip
        self._sync_period = sync_period
        self.n_update_chunks = 3  # so that three update locks will be made

    def initialize(self, inputs, losses, constraints, target,
            givens=None, lr_mult=1):
        self._target = target
        loss = sum(losses)
        params = target.get_params(trainable=True)
        gradients = theano.grad(loss, wrt=params, disconnected_inputs='ignore')

        gradients, grad_norm = apply_grad_norm_clip(gradients, self._grad_norm_clip)

        # Phase 0: load data onto GPU (and shuffle indexes there).
        load_updates, givens, opt_inputs = make_shared_inputs(inputs, self._shuffle)

        # Phase 1: apply gradient update to local params; accum changes
        lr = self._learning_rate * lr_mult
        beta1 = self._update_method_args.get("beta1", 0.9)
        beta2 = self._update_method_args.get("beta2", 0.999)

        grad_updates, clear_updates, adam_m_vars, adam_v_vars, t_var, accums = \
            adam_accum_updates(gradients, params, lr, beta1, beta2)

        self._f_grad_local_update = ext.compile_function(
            inputs=opt_inputs,
            outputs=[loss, grad_norm],
            updates=grad_updates,
            givens=givens,
            log_name="gradient_local_update",
        )
        self._f_load = ext.compile_function(
            inputs=inputs,
            updates=load_updates,
            log_name="load",
        )
        self._f_clear_accum = ext.compile_function(
            inputs=[],
            updates=clear_updates,
            log_name="clear_accum",
        )

        # Phase intermixed: whatever the central params update scheme is.
        central_flat_var = adam_make_flat_var()

        step_accums, g_accums, g2_accums = accums
        param_sync_updates, new_flat_params = \
            apply_accums_params(params, step_accums, central_flat_var)
        m_sync_updates, new_flat_m = apply_accums_adam(adam_m_vars, g_accums,
            central_flat_var, beta1, self._sync_period)
        v_sync_updates, new_flat_v = apply_accums_adam(adam_v_vars, g2_accums,
            central_flat_var, beta2, self._sync_period)

        f_sync_params = ext.compile_function(
            inputs=[central_flat_var],
            updates=param_sync_updates,
            outputs=new_flat_params,
            log_name="sync_params",
        )
        f_sync_m = ext.compile_function(
            inputs=[central_flat_var],
            updates=m_sync_updates,
            outputs=new_flat_m,
            log_name="sync_adam_m",
        )
        f_sync_v = ext.compile_function(
            inputs=[central_flat_var],
            updates=v_sync_updates,
            outputs=new_flat_v,
            log_name="sync_adam_v",
        )
        self._f_syncs = [f_sync_params, f_sync_m, f_sync_v]

        pull_updates_params = pull_to_local(params, central_flat_var)
        pull_updates_adam_m = pull_to_local(adam_m_vars, central_flat_var)
        if self._global_t:
            pull_updates_adam_v = pull_to_local(adam_v_vars + [t_var], central_flat_var)
        else:
            pull_updates_adam_v = pull_to_local(adam_v_vars, central_flat_var)
        f_pull_params = ext.compile_function(
            inputs=[central_flat_var],
            updates=pull_updates_params,
            log_name="pull_params",
        )
        f_pull_adam_m = ext.compile_function(
            inputs=[central_flat_var],
            updates=pull_updates_adam_m,
            log_name="pull_adam_m",
        )
        f_pull_adam_v = ext.compile_function(
            inputs=[central_flat_var],
            updates=pull_updates_adam_v,
            log_name="pull_adam_v",
        )
        self._f_pulls = [f_pull_params, f_pull_adam_m, f_pull_adam_v]

    def optimize(self, inputs):
        self._load_data(inputs)  # (separate methods for profiling)
        return self._make_updates(len(inputs[0]))

    # Separated methods for better profiling. #################################

    def _load_data(self, inputs):
        self._f_load(*inputs)

    def _make_updates(self, data_length):
        losses, grad_norms = (list() for _ in range(2))
        batch_size, shuffle = (self._minibatch_size, self._shuffle)

        self._f_clear_accum()
        self._pull_vals_rw()
        j = 0
        for i in range(self._epochs):
            for idxs in iterate_mb_idxs(batch_size, data_length, shuffle):
                j += 1
                loss, grad_norm = self._compute_grad(idxs)
                losses.append(loss)
                grad_norms.append(grad_norm)
                if j % self._sync_period == 0:
                    self._sync_vals_rw()
                    self._opt_fun["clear_accum"]()

        return losses, grad_norms

    def _pull_vals_rw(self):
        for i, (lock, func, val) in enumerate(
                zip(self._rwlocks, self._f_pulls, self._shmem_arrays)):
            lock.reader_acquire()
            func(val)
            lock.reader_release()

    def _sync_vals_rw(self):
        for lock, func, val in \
                zip(self._rwlocks, self._f_syncs, self._shmem_arrays):
            lock.writer_acquire()
            val[:] = func(val)
            lock.writer_release()

    def _compute_grad(self, idxs):
        return self._f_grad_local_update(*idxs)
