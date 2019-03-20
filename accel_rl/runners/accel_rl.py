
import time
import numpy as np
from collections import deque

from rllab.misc import logger

from accel_rl.runners.accel_rl_base import AccelRLBase
from accel_rl.runners.prog_bar import ProgBarCounter


class AccelRL(AccelRLBase):
    """Runs RL; tracks performance online using learning trajctories"""

    def __init__(
            self,
            log_interval_steps=1e5,
            log_traj_window=100,
            log_ema_steps=None,
            **kwargs
            ):
        super().__init__(**kwargs)
        self._log_steps = int(log_interval_steps)
        self._log_traj_window = int(log_traj_window)
        self._log_ema_steps = int(log_interval_steps) if log_ema_steps is None \
            else int(log_ema_steps)

    def train(self):
        n_itr = self.startup()
        for itr in range(n_itr):
            with logger.prefix('itr #%d | ' % itr):
                samples_data, traj_infos = self.sampler.obtain_samples(itr)
                opt_data, opt_infos = self.algo.optimize_policy(itr, samples_data)
                self.store_diagnostics(itr, samples_data, opt_data, traj_infos, opt_infos)
                if (itr + 1) % self._log_interval_itrs == 0:
                    self.log_diagnostics(itr)
        self.shutdown()

    def init_logging(self):
        self._traj_infos = deque(maxlen=self._log_traj_window)
        self._cum_completed_steps = 0
        self._cum_completed_trajs = 0
        self._new_completed_trajs = 0

        self._log_entropy = hasattr(self.policy, "distribution")
        if self._log_entropy:
            self._entropy_ema = 1.
            self._perplexity_ema = 1.
            self._ema_a = 1 - (0.01) ** (self._log_ema_steps / self._sample_size)


        logger.log('optimizing over {} iterations'.format(self._log_interval_itrs))
        super().init_logging()

    def store_diagnostics(self, itr, samples_data, opt_data, traj_infos, opt_infos):
        self._cum_completed_trajs += len(traj_infos)
        self._new_completed_trajs += len(traj_infos)
        for traj_info in traj_infos:
            self._cum_completed_steps += traj_info["Length"]
            self._traj_infos.append(traj_info)

        for k, v in opt_infos.items():
            self._opt_infos[k].extend(v if isinstance(v, list) else [v])

        if self._log_entropy:
            entropies = self.policy.distribution.entropy(samples_data.agent_infos)
            entropy = np.mean(entropies)
            perplexity = np.mean(np.exp(entropies))
            self._entropy_ema = \
                self._ema_a * entropy + (1 - self._ema_a) * self._entropy_ema
            self._perplexity_ema = \
                self._ema_a * perplexity + (1 - self._ema_a) * self._perplexity_ema

        self.pbar.update((itr + 1) % self._log_interval_itrs)

    def log_diagnostics(self, itr):
        self.pbar.stop()
        self.save_itr_snapshot(itr)

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('CumCompletedTrajs', self._cum_completed_trajs)
        logger.record_tabular('CumCompletedSteps', self._cum_completed_steps)
        logger.record_tabular('CumTotalSteps', (itr + 1) * self._sample_size)
        logger.record_tabular('NewCompletedTrajs', self._new_completed_trajs)
        logger.record_tabular('StepsInTrajWindow',
            sum(info["Length"] for info in self._traj_infos))

        if self._log_entropy:
            logger.record_tabular('Entropy', self._entropy_ema)
            logger.record_tabular('Perplexity', self._perplexity_ema)

        self._log_infos()

        new_time = time.time()
        samples_per_second = \
            (self._log_interval_itrs * self._sample_size) / (new_time - self._last_time)
        logger.record_tabular('CumTime (s)', new_time - self._start_time)
        logger.record_tabular('SamplesPerSecond', samples_per_second)
        self._last_time = new_time
        logger.dump_tabular(with_prefix=False)

        self._new_completed_trajs = 0
        if itr < self._n_itr - 1:
            logger.log('optimizing over {} iterations'.format(self._log_interval_itrs))
            self.pbar = ProgBarCounter(self._log_interval_itrs)


class AccelRLEval(AccelRLBase):
    """Runs RL; tracks learning performance offline using evaluation trajectories"""

    def __init__(self, eval_interval_steps=1e6, **kwargs):
        super().__init__(**kwargs)
        self._log_steps = int(eval_interval_steps)

    def train(self):
        n_itr = self.startup()
        for itr in range(n_itr):
            with logger.prefix('itr #%d | ' % itr):
                if itr % self._log_interval_itrs == 0:
                    eval_traj_infos, eval_time = self.eval_policy(itr)
                    self.log_diagnostics(itr, eval_traj_infos, eval_time)
                samples_data, traj_infos = self.sampler.obtain_samples(itr)
                opt_data, opt_infos = self.algo.optimize_policy(itr, samples_data)
                self.store_diagnostics(itr, samples_data, opt_data, traj_infos, opt_infos)
        self.shutdown()

    def init_logging(self):
        self._cum_train_time = 0
        self._cum_eval_time = 0
        self._cum_total_time = 0
        super().init_logging()

    def eval_policy(self, itr):
        self.pbar.stop()
        logger.log("evaluating policy...")
        eval_start_time = time.time()
        self.algo.prep_eval(itr)
        traj_infos = self.sampler.evaluate_policy(itr)
        self.algo.post_eval(itr)
        eval_end_time = time.time()
        logger.log("evaluation run complete")
        eval_time = eval_end_time - eval_start_time
        return traj_infos, eval_time

    def log_diagnostics(self, itr, eval_traj_infos, eval_time):
        self.save_itr_snapshot(itr)
        if not eval_traj_infos:
            logger.log("ERROR: had no complete trajectories in eval.")
        steps_in_eval = sum([info["Length"] for info in eval_traj_infos])
        logger.record_tabular('Iteration', itr)
        logger.record_tabular('CumCompletedSteps', itr * self._sample_size)
        logger.record_tabular('StepsInEval', steps_in_eval)
        logger.record_tabular('TrajsInEval', len(eval_traj_infos))

        self._log_infos(eval_traj_infos)

        new_time = time.time()
        log_interval_time = new_time - self._last_time
        new_train_time = log_interval_time - eval_time
        self._cum_train_time += new_train_time
        self._cum_eval_time += eval_time
        self._cum_total_time += log_interval_time
        self._last_time = new_time
        train_speed = float('nan') if itr == 0 else \
            self._log_interval_itrs * self._sample_size / new_train_time

        logger.record_tabular('CumTrainTime', self._cum_train_time)
        logger.record_tabular('CumEvalTime', self._cum_eval_time)
        logger.record_tabular('CumTotalTime', self._cum_total_time)
        logger.record_tabular('SamplesPerSecond', train_speed)

        logger.dump_tabular(with_prefix=False)

        logger.log('optimizing over {} iterations'.format(self._log_interval_itrs))
        self.pbar = ProgBarCounter(self._log_interval_itrs)

    def store_diagnostics(self, itr, samples_data, opt_data, traj_infos, opt_infos):
        for k, v in opt_infos.items():
            self._opt_infos[k].extend(v if isinstance(v, list) else [v])
        self.pbar.update((itr + 1) % self._log_interval_itrs)
