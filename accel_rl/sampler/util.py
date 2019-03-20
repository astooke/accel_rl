

import time
import psutil
from rllab.misc import logger, ext

from accel_rl.util.misc import struct


def profiling_process(target, profile_pathname, **kwargs):
    import cProfile
    import os
    try:
        directory, filename = profile_pathname.rsplit('/', 1)
        os.chdir(directory)
    except ValueError:
        filename = profile_pathname
    filename += "_sim_{}.prof".format(kwargs["unique_ID"])
    cProfile.runctx('target(**kwargs)', locals(), globals(), filename)


def get_random_fraction():
    return float(str(time.time())[-3:]) / 1000  # grabs microseconds or so


def start_envs(envs, max_decorrelation_steps, max_path_length, discount,
        unique_ID=0):
    """calls reset() on every env"""
    observations = list()
    traj_infos = [TrajInfo(discount=discount) for _ in range(len(envs))]
    if max_decorrelation_steps == 0:
        for env in envs:
            observations.append(env.reset())
    else:
        if unique_ID == 0:
            logger.log("MbSampler: Decorrelating envs, max steps: {}".format(
                max_decorrelation_steps))
        for i, env in enumerate(envs):
            n_steps = int(get_random_fraction() * max_decorrelation_steps)
            if hasattr(env.action_space, "sample_n"):
                actions = env.action_space.sample_n(n_steps)
            else:
                actions = [env.action_space.sample() for _ in range(n_steps)]
            o = env.reset()
            num_resets = 0
            traj_info = traj_infos[i]
            for a in actions:
                o, r, d, env_info = env.step(a)
                traj_info.step(r, env_info)
                if traj_info.Length > max_path_length or \
                        (d and env_info.get("need_reset", True)):
                    o = env.reset()
                    num_resets += 1
                    traj_info = traj_infos[i] = TrajInfo(discount=discount)

            observations.append(o)
    return observations, traj_infos


def initialize_worker(group, rank, seed, cpu):
    log_str = "MbSampler rank: {} initialized".format(rank)
    try:
        p = psutil.Process()
        p.cpu_affinity([cpu])
        log_str += ", CPU Affinity: {}".format(p.cpu_affinity())
    except AttributeError:
        pass
    if seed is not None:
        ext.set_seed(seed)
        time.sleep(0.3)  # (so the printing from set_seed is not intermixed)
        log_str += ", Seed: {}".format(seed)
    logger.log(log_str)


class TrajInfo(struct):
    """
    Because it inits as a struct, this has the methods of a dictionary,
    e.g. the attributes can be iterated through by traj_info.items()

    All attributes not starting with underscore "_" will be logged.

    (if you want to log something else, write your own ;)  )
    """

    def __init__(self, discount=1, **kwargs):
        super().__init__(**kwargs)
        self.Length = 0
        self.Return = 0
        self.RawReturn = 0
        self.NonzeroRewards = 0
        self.DiscountedReturn = 0
        self._discount = discount
        self._cur_discount = 1

    def step(self, r, env_info):
        self.Length += 1
        self.Return += r
        self.RawReturn += env_info.get("raw_reward", r)
        self.NonzeroRewards += r != 0
        self.DiscountedReturn += self._cur_discount * r
        self._cur_discount *= self._discount
