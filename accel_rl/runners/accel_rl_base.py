
import psutil
import time
import numpy as np

from rllab.misc.ext import set_seed
from rllab.misc import logger

from accel_rl.util.quick_args import save_args
from accel_rl.util.misc import make_seed, nbytes_unit
from accel_rl.runners.base import Runner
from accel_rl.runners.prog_bar import ProgBarCounter


class AccelRLBase(Runner):

    def __init__(
            self,
            algo,
            policy,
            sampler,
            n_steps,
            seed=None,
            affinities=None,
            use_gpu=True,
            ):
        n_steps = int(n_steps)
        save_args(vars(), underscore=False)
        if affinities is None:
            self.affinities = dict()
        if algo.optimizer.parallelism_tag != self.parallelism_tag:
            raise TypeError("Had mismatched parallelism between Runner ({}) "
                "and Optimizer: {}".format(self.parallelism_tag,
                algo.optimizer.parallelism_tag))

    def startup(self, master=True):
        if self.seed is None:
            self.seed = make_seed()
        set_seed(self.seed)
        env_spec, sample_size, horizon, mid_batch_reset = self.sampler.initialize(
            seed=self.seed + 1,
            affinities=self.affinities,
            discount=getattr(self.algo, "discount", None),
            need_extra_obs=self.algo.need_extra_obs,
        )
        self.init_policy(env_spec)
        self.algo.initialize(
            policy=self.policy,
            env_spec=env_spec,
            sample_size=sample_size,
            horizon=horizon,
            mid_batch_reset=mid_batch_reset,
        )
        self.sampler.policy_init(self.policy)
        if master:
            n_itr = self.get_n_itr(sample_size)
            self.algo.set_n_itr(n_itr)
            self.init_logging()
            return n_itr

    def init_policy(self, env_spec):
        if self.use_gpu:
            import theano.gpuarray
            theano.gpuarray.use("cuda" + str(self.affinities.get("gpu", "")))
        kwargs = {} if not self.policy.recurrent else \
            dict(alternating_sampler=self.sampler.alternating)
        self.policy.initialize(env_spec, **kwargs)
        flat_params = self.policy.get_param_values(trainable=True)
        logger.log("Policy trainable params -- number: {:,}   size: {:,.1f} "
            "{}".format(flat_params.size, *nbytes_unit(flat_params.nbytes)))
        p = psutil.Process()
        p.cpu_affinity(self.affinities.get("gpu_cpus", p.cpu_affinity()))

    def get_n_itr(self, sample_size):
        self._sample_size = sample_size
        self._log_interval_itrs = max(self._log_steps // sample_size, 1)
        n_itr = max(self.n_steps // sample_size, 1)
        itr_rem = n_itr % self._log_interval_itrs
        if itr_rem <= self._log_interval_itrs / 2.:
            n_itr -= itr_rem
        else:
            n_itr += (self._log_interval_itrs - itr_rem)
        assert n_itr % self._log_interval_itrs == 0
        n_itr += 1
        self._n_itr = n_itr
        logger.log("Iterations to run: {}".format(n_itr))
        return n_itr

    def init_logging(self):
        self._opt_infos = {k: list() for k in self.algo.opt_info_keys}
        self._initial_param_vector = self.policy.get_param_values()
        self._layerwise_stats = getattr(self.algo, "layerwise_stats", False)
        if self._layerwise_stats:
            self._param_names = self.policy.param_short_names
            self._params = self.policy.get_params()
            self._init_params_values = [p.get_value() for p in self._params]
        self._start_time = self._last_time = time.time()
        self.pbar = ProgBarCounter(self._log_interval_itrs)

    def shutdown(self):
        self.finish_logging()
        self.sampler.shutdown()

    def finish_logging(self):
        logger.log('Training complete.')
        self.pbar.stop()

    def get_itr_snapshot(self, itr):
        return dict(
            itr=itr,
            cum_samples=itr * self._sample_size,
            policy_param_values=self.policy.get_param_values(),
        )

    def save_itr_snapshot(self, itr):
        logger.log("saving snapshot...")
        params = self.get_itr_snapshot(itr)
        # params["algo"] = self.algo
        logger.save_itr_params(itr, params)
        logger.log("saved")

    def _log_infos(self, traj_infos=None):
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            for k in traj_infos[0]:
                if not k.startswith("_"):
                    logger.record_tabular_misc_stat(k,
                        [info[k] for info in traj_infos])

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)

        if self._layerwise_stats:
            for name, param, init_val in zip(
                    self._param_names, self._params, self._init_params_values):
                new_val = param.get_value()
                diff = new_val - init_val
                logger.record_tabular(name + "_Norm", np.sqrt(np.sum(new_val ** 2)))
                logger.record_tabular(name + "_NormFromInit", np.sqrt(np.sum(diff ** 2)))
        new_param_vector = self.policy.get_param_values()
        logger.record_tabular("ParamsNorm", np.sqrt(np.sum(new_param_vector ** 2)))
        params_diff = new_param_vector - self._initial_param_vector
        logger.record_tabular("NormFromInit", np.sqrt(np.sum(params_diff ** 2)))

    @property
    def parallelism_tag(self):
        return "single"
