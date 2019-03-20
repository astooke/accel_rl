
import multiprocessing as mp
import time

from accel_rl.runners.accel_rl_base import AccelRLBase
from accel_rl.util.misc import struct, make_seed
from accel_rl.util.quick_args import save_args


class MultiGpuRLBase(AccelRLBase):

    def startup(self):
        self.launch_workers()
        n_itr = super().startup()
        self.init_comm(n_itr, self._log_interval_itrs)
        self.par_objs.barrier.wait()
        self._start_time = self._last_time = time.time()  # (overwrite)
        return n_itr

    def launch_workers(self):
        n_runners = len(self.affinities)
        if self.seed is None:
            self.seed = make_seed()
        worker_kwargs_list = [dict(
            algo=self.algo,
            policy=self.policy,
            sampler=self.sampler,
            seed=self.seed + 100 * rank,
            affinities=self.affinities[rank],
            rank=rank,
            n_runners=n_runners,
            )
            for rank in range(1, n_runners)]

        par_objs, WorkerCls = self.build_par_objs(n_runners)

        workers = [WorkerCls(**w_kwargs) for w_kwargs in worker_kwargs_list]
        self.worker_procs = \
            [mp.Process(target=w.train, args=(par_objs,)) for w in workers]
        for w in self.worker_procs:
            w.start()
        self.par_objs = par_objs
        self.rank = 0
        self.affinities = self.affinities[0]
        self.n_runners = n_runners

    def init_comm(self, n_itr):
        raise NotImplementedError

    def build_par_objs(self, n_runners):
        barrier = mp.Barrier(n_runners)
        mgr = mp.Manager()
        par_dict = mgr.dict()
        traj_infos_queue = mp.Queue()
        par_objs = struct(
            barrier=barrier,
            dict=par_dict,
            traj_infos_queue=traj_infos_queue,
        )
        return par_objs

    def get_n_itr(self, sample_size):
        return super().get_n_itr(sample_size * self.n_runners)

    @property
    def parallelism_tag(self):
        return "undef"  # (i.e. do not use)


class MultiGpuWorkerBase(AccelRLBase):

    def __init__(
            self,
            algo,
            policy,
            sampler,
            seed,
            affinities,
            rank,
            n_runners,
            use_gpu=True,
            ):
        save_args(vars(), underscore=False)

    def startup(self):
        super().startup(master=False)
        n_itr, log_interval_itrs = self.init_comm()
        self._log_interval_itrs = log_interval_itrs
        self.algo.set_n_itr(n_itr)
        self.par_objs.barrier.wait()
        return n_itr

    def shutdown(self):
        self.sampler.shutdown()

    def train(self, par_objs):
        self.par_objs = par_objs
        super().train()

    def init_comm(self):
        raise NotImplementedError


###############################################################################
# synchronous
###############################################################################


class SyncBase(MultiGpuRLBase):

    def init_comm(self, n_itr, log_interval_itrs):
        import theano.gpuarray
        from pygpu import collectives as gpu_coll
        gpu_ctx = theano.gpuarray.get_context(None)
        clique_id = gpu_coll.GpuCommCliqueId(gpu_ctx)
        self.par_objs.dict["gpu_comm_id"] = clique_id.comm_id
        self.par_objs.dict["n_itr"] = n_itr
        self.par_objs.dict["log_interval_itrs"] = log_interval_itrs
        self.par_objs.dict["initial_param_values"] = self.policy.get_param_values()
        self.par_objs.barrier.wait()
        gpu_comm = gpu_coll.GpuComm(clique_id, self.n_runners, self.rank)
        self.algo.optimizer.init_comm(gpu_comm, self.rank, self.n_runners)

    def shutdown(self):
        super().shutdown()
        for w in self.worker_procs:
            w.join()

    @property
    def parallelism_tag(self):
        return "synchronous"


class SyncWorkerBase(MultiGpuWorkerBase):

    def init_comm(self):
        import theano.gpuarray
        from pygpu import collectives as gpu_coll
        gpu_ctx = theano.gpuarray.get_context(None)
        clique_id = gpu_coll.GpuCommCliqueId(gpu_ctx)
        self.par_objs.barrier.wait()
        initial_param_values = self.par_objs.dict["initial_param_values"]
        self.policy.set_param_values(initial_param_values)
        clique_id.comm_id = self.par_objs.dict["gpu_comm_id"]
        gpu_comm = gpu_coll.GpuComm(clique_id, self.n_runners, self.rank)
        self.algo.optimizer.init_comm(gpu_comm, self.rank, self.n_runners)
        n_itr = self.par_objs.dict["n_itr"]
        log_interval_itrs = self.par_objs.dict["log_interval_itrs"]
        return n_itr, log_interval_itrs

    @property
    def parallelism_tag(self):
        return "synchronous"


###############################################################################
# Asynchronous
###############################################################################


class AsyncBase(MultiGpuRLBase):

    def build_par_objs(self, n_runners):
        par_objs = super().build_par_objs(n_runners)
        par_objs["update_locks"] = \
            [mp.Lock() for _ in range(self.algo.optimizer.n_update_chunks)]
        par_objs["workers_done"] = [mp.Semaphore(0) for _ in range(n_runners - 1)]
        return par_objs

    def init_comm(self, n_itr, log_interval_itrs):
        self.par_objs.dict["n_itr"] = n_itr
        self.par_objs.dict["log_interval_itrs"] = log_interval_itrs
        self.par_objs.dict["initial_param_values"] = self.policy.get_param_values()
        self.par_objs.barrier.wait()
        self.algo.optimizer.init_comm(self.rank, self.n_runners, self.par_objs)

    def shutdown(self):
        super().shutdown()
        for s in self.par_objs.workers_done:
            s.acquire()
        while self.par_objs.traj_infos_queue.qsize():
            self.par_objs.traj_infos_queue.get()
        for w in self.worker_procs:
            w.join()

    @property
    def parallelism_tag(self):
        return "asynchronous"


class AsyncWorkerBase(MultiGpuWorkerBase):

    def init_comm(self):
        self.par_objs.barrier.wait()
        n_itr = self.par_objs.dict["n_itr"]
        log_interval_itrs = self.par_objs.dict["log_interval_itrs"]
        self.policy.set_param_values(
            self.par_objs.dict["initial_param_values"])
        self.algo.optimizer.init_comm(self.rank, self.n_runners, self.par_objs)
        return n_itr, log_interval_itrs

    def shutdown(self):
        super().shutdown()
        self.par_objs.workers_done[self.rank - 1].release()

    @property
    def parallelism_tag(self):
        return "asynchronous"


###############################################################################
# Logging
###############################################################################


class OnlineLog(object):

    def store_diagnostics(self, itr, samples_data, opt_data, traj_infos, opt_infos):
        while self.par_objs.traj_infos_queue.qsize():
            traj_infos.append(self.par_objs.traj_infos_queue.get())
        super().store_diagnostics(itr, samples_data, opt_data, traj_infos, opt_infos)


class OnlineLogWorker(object):

    def store_diagnostics(self, itr, samples_data, opt_data, traj_infos, opt_infos):
        for traj_info in traj_infos:
            self.par_objs.traj_infos_queue.put(traj_info)

    def log_diagnostics(self, *args, **kwargs):
        pass


class EvalLog(object):

    def log_diagnostics(self, *args, **kwargs):
        super().log_diagnostics(*args, **kwargs)
        self.par_objs.barrier.wait()


class EvalLogWorker(object):

    def store_diagnostics(self, *args, **kwargs):
        pass

    def log_diagnostics(self, *args, **kwargs):
        self.par_objs.barrier.wait()

    def eval_policy(self, *args, **kwargs):
        return None, None

