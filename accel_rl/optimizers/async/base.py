

from accel_rl.optimizers.base import BaseOptimizer
from accel_rl.optimizers.async.chunked_updates import get_init_arrays
from accel_rl.buffers.shmemarray import NpShmemArray, get_random_tag


TAG = "async_opt_shared_"


class BaseAsyncOptimizer(BaseOptimizer):

    def init_comm(self, rank, n_runners, par_objs):
        self._rank = rank
        self._n_runners = n_runners
        self._update_locks = par_objs["update_locks"]
        if "rwlocks" in par_objs:
            self._rwlocks = par_objs["rwlocks"]
        self._shmem_arrays = self.make_shmem_arrays(rank, par_objs)

    def make_shmem_arrays(self, rank, par_objs):
        init_arrays = get_init_arrays(self._target, self._update_method_name)
        shmem_arrays = list()
        if rank == 0:
            tag = TAG + get_random_tag()
            par_objs["dict"]["shmem_tag"] = tag
            for i, arr in enumerate(init_arrays):
                tag_i = tag + "_" + str(i)
                sh_arr = NpShmemArray(arr.shape, arr.dtype, tag_i, create=True)
                sh_arr[:] = arr
                shmem_arrays.append(sh_arr)
            par_objs["barrier"].wait()
        else:
            par_objs["barrier"].wait()
            tag = par_objs["dict"]["shmem_tag"]
            for i, arr in enumerate(init_arrays):
                tag_i = tag + "_" + str(i)
                sh_arr = NpShmemArray(arr.shape, arr.dtype, tag_i, create=False)
                shmem_arrays.append(sh_arr)
        return shmem_arrays

    @property
    def parallelism_tag(self):
        return "asynchronous"

    @property
    def central_shared_params(self):
        return self._shmem_arrays[0]

    @property
    def central_params_lock(self):
        # later make it so multiple can read at once.
        return self._update_locks[0]

    @property
    def central_params_rwlock(self):
        return self._rwlocks[0]

    def _single_lock_push(self):
        shmems = self._shmem_arrays
        with self._update_locks[0]:
            new_vals = self._f_update(*shmems)
            for s, new_val in zip(shmems, new_vals):
                s[:] = new_val

    def _cycle_locks_push(self):
        locks = list(self._update_locks)
        lock_ids = list(range(len(locks)))
        removals = list()
        shmems = self._shmem_arrays

        # Cycle through the locks, acquire as available, until all are done.
        while locks:
            for rem_i, (lock_id, lock) in enumerate(zip(lock_ids, locks)):
                if lock.acquire(block=True, timeout=0.001):
                    start, stop = self._chunk_idxs[lock_id]
                    func = self._f_update_chunks[lock_id]
                    new_vals = apply_update_func(func, shmems, start, stop)
                    write_vals(new_vals, shmems, start, stop)
                    lock.release()
                    removals.append(rem_i)
            if removals:
                locks[:] = [k for i, k in enumerate(locks) if i not in removals]
                lock_ids[:] = [k for i, k in enumerate(lock_ids) if i not in removals]
                removals.clear()

        # Previous code: loop once in order, blocking at lock acquire.
        # for (start, stop), lock, func in zip(idxs, locks, funcs):
        #     with lock:
        #         new_vals = func(*(s[start:stop] for s in shmems))
        #         for s, new_val in zip(shmems, new_vals):
        #             s[start:stop] = new_val


# Separated functions for better profiling ##################################


def apply_update_func(func, shmems, start, stop):
    return func(*(s[start:stop] for s in shmems))


def write_vals(new_vals, shmems, start, stop):
    for s, new_val in zip(shmems, new_vals):
        s[start:stop] = new_val
