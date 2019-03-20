

from accel_rl.runners.multigpu_rl_base import *
from accel_rl.runners.accel_rl import AccelRL, AccelRLEval


class AccelRLSync(SyncBase, OnlineLog, AccelRL):

    def build_par_objs(self, n_runners):
        par_objs = super().build_par_objs(n_runners)
        return par_objs, SyncWorker


class SyncWorker(SyncWorkerBase, OnlineLogWorker, AccelRL):
    pass


class AccelRLAsync(AsyncBase, OnlineLog, AccelRL):

    def build_par_objs(self, n_runners):
        par_objs = super().build_par_objs(n_runners)
        return par_objs, AsyncWorker


class AsyncWorker(AsyncWorkerBase, OnlineLogWorker, AccelRL):
    pass


class AccelRLEvalSync(SyncBase, EvalLog, AccelRLEval):

    def build_par_objs(self, n_runners):
        par_objs = super().build_par_objs(n_runners)
        return par_objs, SyncEvalWorker


class SyncEvalWorker(SyncWorkerBase, EvalLogWorker, AccelRLEval):
    pass


class AccelRLEvalAsync(AsyncBase, EvalLog, AccelRLEval):

    def build_par_objs(self, n_runners):
        par_objs = super().build_par_objs(n_runners)
        return par_objs, AsyncEvalWorker


class AsyncEvalWorker(AsyncWorkerBase, EvalLogWorker, AccelRLEval):
    pass
