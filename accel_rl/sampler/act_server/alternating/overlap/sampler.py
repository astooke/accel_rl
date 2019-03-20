

import multiprocessing as mp
import ctypes

from rllab.misc import logger

from accel_rl.util.misc import nbytes_unit, struct
from accel_rl.sampler.act_server.alternating.overlap.worker import sampling_process
from accel_rl.sampler.util import profiling_process

from accel_rl.sampler.act_server.buffers import build_env_buffer, \
    build_step_buffer, build_policy_buffer, view_worker_segs_bufs
from accel_rl.buffers.batch import buffer_length, combine_distinct_buffers, \
    count_buffer_size

from accel_rl.sampler.base import BaseMbSampler


class ActsrvAltOvrlpSampler(BaseMbSampler):

    """
    Provides GPU-based parallel sampling in minibatches.
    Workers are organized into two alternating groups--one simulates while the
      other waits for actions to be served.
    The two groups may both be simulating at the same time (overlap)
      (e.g. using hyperthreads, this helps for Atari, probably not for Mujoco)
    """

    # TODO: fix up all the places with eval_envs_per

    ###########################################################################
    #   API to the Runner
    ###########################################################################

    def __init__(self, n_parallel=1, envs_per=1, **kwargs):
        super().__init__(n_parallel=n_parallel, envs_per=envs_per, **kwargs)
        self._total_n_envs = n_parallel * envs_per * 2

    def initialize(self, seed, affinities, discount=1, need_extra_obs=False,
            worker_process_target=None):
        env = self.EnvCls(**self.env_args)  # instantiate one for examples
        eval_envs_per = getattr(self, "eval_envs_per", None)
        par_objs = build_par_objs(env, self.n_parallel, self.envs_per,
            self.horizon, need_extra_obs, eval_envs_per)
        self.common_kwargs["discount"] = discount
        worker_kwargs_list = assemble_worker_kwargs(self.n_parallel, par_objs,
            seed, affinities, self.common_kwargs)

        if worker_process_target is None:
            worker_process_target = sampling_process
        if self.profile_pathname is None:
            target = worker_process_target
            for w_kwargs in worker_kwargs_list:
                w_kwargs.pop("profile_pathname")
        else:
            target = profiling_process
            for w_kwargs in worker_kwargs_list:
                w_kwargs["target"] = worker_process_target

        workers = [mp.Process(target=target, args=(), kwargs=w_kwargs)
            for w_kwargs in worker_kwargs_list]
        for w in workers:
            w.start()

        if eval_envs_per is None:
            self.ctrl, self.sync, self.envs_buf, self.step_bufs, \
                self.traj_infos_queue = par_objs
        else:
            self.ctrl, self.sync, self.envs_buf, self.step_bufs, \
                self.traj_infos_queue, self.eval_step_bufs = par_objs
        self.workers = workers

        self.sample_size = 2 * self.n_parallel * self.envs_per * self.horizon
        assert self.sample_size == buffer_length(self.envs_buf)
        self.env_spec = env.spec
        self.seed = seed
        self.need_extra_obs = need_extra_obs
        return env.spec, self.sample_size, self.horizon, self.mid_batch_reset

    def policy_init(self, policy):
        self.policy = policy
        policy_buf = build_policy_buffer(self.env_spec, policy, self.sample_size,
            self.horizon)
        if self.need_extra_obs:
            self._pol_groups_bufs, self._extra_obs_groups = \
                view_serve_groups(policy_buf, self.envs_buf.extra_observations)
        else:
            self._pol_groups_bufs = view_serve_groups(policy_buf)
        self.samples_buf = combine_distinct_buffers(self.envs_buf, policy_buf)
        policy.reset(n_batch=self.n_parallel * self.envs_per)  # half of total
        self.ctrl.barrier_out.wait()  # (wait for workers to finish start_envs)
        logger.log("MbSampler -- total_n_envs: {}".format(self.total_n_envs))
        logger.log("MbSampler -- batch buffer size: {:,.1f} {}".format(
            *nbytes_unit(count_buffer_size(self.samples_buf))))

    def obtain_samples(self, itr):
        self.ctrl.barrier_in.wait()
        self.serve_actions(itr)
        self.ctrl.barrier_out.wait()
        traj_infos = list()
        while self.traj_infos_queue.qsize():
            traj_infos.append(self.traj_infos_queue.get())
        return self.samples_buf, traj_infos

    def shutdown(self):
        self.ctrl.quit.value = True
        self.ctrl.barrier_in.wait()
        for w in self.workers:
            w.join()

    @property
    def alternating(self):
        return True

    ###########################################################################
    #   Helper methods
    ###########################################################################

    def serve_actions(self, itr):
        policy = self.policy
        horizon = self.horizon
        step_blockers = self.sync.step_blockers
        act_waiters = self.sync.act_waiters
        pol_buf_pair = self._pol_groups_bufs
        step_bufs = self.step_bufs
        if self.need_extra_obs:
            extra_obs_pair = self._extra_obs_groups
        for s in range(horizon):
            for j in range(2):
                step_buf = step_bufs[j]
                pol_buf = pol_buf_pair[j]
                for b in step_blockers[j]:
                    b.acquire()
                if any(step_buf.reset):
                    for idx in [i for i, r in enumerate(step_buf.reset) if r]:
                        policy.reset_one(idx=idx)  # for recurrence
                    step_buf.reset[:] = False
                acts, agent_infos = policy.get_actions(step_buf.obs)
                step_buf.act[:] = acts  # used in sampling
                for w in act_waiters[j]:
                    w.release()
                pol_buf.actions[s::horizon] = acts  # used in optimizing
                for k, v in agent_infos.items():
                    pol_buf.agent_infos[k][s::horizon] = v

        for j in range(2):
            for b in step_blockers[j]:
                b.acquire()
            if self.need_extra_obs:
                extra_obs_pair[j][:] = step_bufs[j].obs


def assemble_worker_kwargs(n, par_objs, seed, affinities, common_kwargs):
    if len(par_objs) == 5:
        ctrl, sync, envs_buf, step_bufs, traj_infos_queue = par_objs
        eval_step_bufs = None
    else:
        ctrl, sync, envs_buf, step_bufs, traj_infos_queue, eval_step_bufs = par_objs
    worker_segs_lists = view_worker_segs_bufs(envs_buf.segs_view, 2 * n)
    cpu_list = affinities.get("sim_cpus", list(range(2 * n)))
    common_par_objs = (ctrl, traj_infos_queue)
    worker_kwargs_list = list()
    i = 0
    for group in range(2):
        for rank in range(n):
            w_sync = struct(step_blocker=sync.step_blockers[group][rank],
                            act_waiter=sync.act_waiters[group][rank])
            w_segs_buf = worker_segs_lists[i]
            w_step_buf = step_bufs[group]
            w_par_objs = (w_sync, w_segs_buf, w_step_buf)
            if eval_step_bufs is not None:
                w_eval_step_buf = eval_step_bufs[group]
                w_par_objs += (w_eval_step_buf,)
            w_kwargs = dict(
                group=group,
                rank=rank,
                unique_ID=i,
                seed=seed + i,
                cpu=cpu_list[i],
                par_objs=common_par_objs + w_par_objs,
                **common_kwargs
                )
            worker_kwargs_list.append(w_kwargs)
            i += 1
    return worker_kwargs_list


def build_par_objs(env, n, envs_per, horizon, extra_obs, eval_envs_per=None):
    ctrl = struct(
        quit=mp.RawValue(ctypes.c_bool, False),
        do_eval=mp.RawValue(ctypes.c_bool, False),
        barrier_in=mp.Barrier(2 * n + 1),
        barrier_out=mp.Barrier(2 * n + 1),
    )
    sync = struct(
        step_blockers=[[mp.Semaphore(0) for _ in range(n)] for _ in range(2)],
        act_waiters=[[mp.Semaphore(0) for _ in range(n)] for _ in range(2)],
    )
    envs_buf = build_env_buffer(env, 2 * n * envs_per * horizon, horizon, extra_obs)
    step_bufs = [build_step_buffer(env.spec, n * envs_per) for _ in range(2)]
    traj_infos_queue = mp.Queue()
    ret = (ctrl, sync, envs_buf, step_bufs, traj_infos_queue)
    if eval_envs_per is not None:
        eval_step_bufs = [build_step_buffer(env.spec, n * eval_envs_per) for _ in range(2)]
        ret += (eval_step_bufs,)
    return ret


def view_serve_groups(policy_buf, extra_observations=None):
    half = buffer_length(policy_buf) // 2
    pol_group_0 = struct(
        actions=policy_buf.actions[:half],
        agent_infos={k: v[:half] for k, v in policy_buf.agent_infos.items()},
    )
    pol_group_1 = struct(
        actions=policy_buf.actions[half:],
        agent_infos={k: v[half:] for k, v in policy_buf.agent_infos.items()},
    )
    if extra_observations is not None:
        half_obs = len(extra_observations) // 2
        extra_obs_0 = extra_observations[:half_obs]
        extra_obs_1 = extra_observations[half_obs:]
        return (pol_group_0, pol_group_1), (extra_obs_0, extra_obs_1)
    return (pol_group_0, pol_group_1)
