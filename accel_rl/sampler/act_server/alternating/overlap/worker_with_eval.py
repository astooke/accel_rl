
from accel_rl.util.quick_args import save_args
from accel_rl.sampler.util import start_envs, initialize_worker, TrajInfo


class Collector(object):

    def __init__(
            self,
            rank,
            envs,
            eval_envs,
            sync,
            segs_buf,
            step_buf,
            eval_step_buf,
            horizon,
            eval_horizon,
            max_path_length,
            discount,
            ):
        save_args(vars(), underscore=False)
        self.n_envs = len(envs)
        self.n_eval_envs = len(eval_envs)


class ResetCollector(Collector):

    def collect(self, traj_infos):
        step_buf = self.step_buf
        step_blocker = self.sync.step_blocker
        act_waiter = self.sync.act_waiter
        step_blocker.release()  # (previous step_buf obs already written)
        for i in range(self.n_envs):
            sb_idx = self.n_envs * self.rank + i
            self.segs_buf[i].observations[0] = step_buf.obs[sb_idx]

        completed_infos = list()
        for s in range(self.horizon):
            act_waiter.acquire()
            for i, (env, seg, traj_info) in \
                    enumerate(zip(self.envs, self.segs_buf, traj_infos)):
                sb_idx = self.n_envs * self.rank + i
                o, r, d, env_info = env.step(step_buf.act[sb_idx])
                traj_info.step(r, env_info)
                over_length = traj_info.Length >= self.max_path_length
                if over_length or (d and env_info.get("need_reset", True)):
                    d = True
                    o = env.reset()
                    if over_length and "need_reset" in env_info:
                        env_info["need_reset"] = True
                    completed_infos.append(traj_info)
                    traj_infos[i] = TrajInfo(self.discount)
                step_buf.obs[sb_idx] = o
                if s < self.horizon - 1:
                    seg.observations[s + 1] = o
                seg.rewards[s] = r
                seg.dones[s] = d
                if env_info:
                    for k, v in env_info.items():
                        seg.env_infos[k][s] = v
            step_blocker.release()
        return traj_infos, completed_infos

    def collect_eval(self):
        step_buf = self.eval_step_buf
        step_blocker = self.sync.step_blocker
        act_waiter = self.sync.act_waiter

        observations = [env.reset() for env in self.eval_envs]
        for i, o in enumerate(observations):
            sb_idx = self.n_eval_envs * self.rank + i
            step_buf.obs[sb_idx] = o
        step_blocker.release()

        traj_infos = [TrajInfo(self.discount) for _ in range(self.n_eval_envs)]
        completed_infos = list()
        for _ in range(self.eval_horizon):
            act_waiter.acquire()
            for i, (env, traj_info) in enumerate(zip(self.eval_envs, traj_infos)):
                sb_idx = self.n_eval_envs * self.rank + i
                o, r, d, env_info = env.step(step_buf.act[sb_idx])
                traj_info.step(r, env_info)
                over_length = traj_info.Length >= self.max_path_length
                if over_length or (d and env_info.get("need_reset", True)):
                    d = True
                    o = env.reset()
                    if over_length and "need_reset" in env_info:
                        env_info["need_reset"] = True
                    completed_infos.append(traj_info)
                    traj_infos[i] = TrajInfo(self.discount)
                step_buf.obs[sb_idx] = o
            step_blocker.release()
        print("inside collect_eval, had {} completed_infos".format(len(completed_infos)))
        if not completed_infos:
            traj_lengths = [traj.Length for traj in traj_infos]
            print("inside collect_eval, had traj lengths: ", traj_lengths)
            print("inside collect_eval, eval_horizon: ", self.eval_horizon)
        return completed_infos


class NonResetCollector(Collector):

    def collect(self, traj_infos):
        step_buf = self.step_buf
        step_blocker = self.sync.step_blocker
        act_waiter = self.sync.act_waiter
        step_blocker.release()  # (previous step_buf obs already written)
        for i in range(self.n_envs):
            sb_idx = self.n_envs * self.rank + i
            self.segs_buf[i].observations[0] = step_buf.obs[sb_idx]

        completed_infos = list()
        need_reset = [False] * self.n_envs
        for s in range(self.horizon):
            act_waiter.acquire()
            for i, (env, seg, traj_info) in \
                    enumerate(zip(self.envs, self.segs_buf, traj_infos)):
                if not need_reset[i]:
                    sb_idx = self.n_envs * self.rank + i
                    o, r, d, env_info = env.step(step_buf.act[sb_idx])
                    traj_info.step(r, env_info)
                    over_length = traj_info.Length >= self.max_path_length
                    if over_length or (d and env_info.get("need_reset", True)):
                        # Allows to continue for episodic lives:
                        # env reports "need_reset" = False, and auto-resets obs
                        d = True
                        need_reset[i] = True
                        if over_length and "need_reset" in env_info:
                            env_info["need_reset"] = True
                        completed_infos.append(traj_info)
                        traj_infos[i] = TrajInfo(self.discount)
                    else:
                        step_buf.obs[sb_idx] = o
                        if s < self.horizon - 1:
                            seg.observations[s + 1] = o
                    seg.rewards[s] = r
                    seg.dones[s] = d
                    if env_info:
                        for k, v in env_info.items():
                            seg.env_infos[k][s] = v
            step_blocker.release()
        self.need_reset = need_reset
        return traj_infos, completed_infos

    def reset_needed_envs(self):
        for i, need in enumerate(self.need_reset):
            if need:  # (need to reset env)
                o = self.envs[i].reset()
                sb_idx = self.n_envs * self.rank + i
                self.step_buf.obs[sb_idx] = o

    def collect_eval(self):
        step_buf = self.eval_step_buf
        step_blocker = self.sync.step_blocker
        act_waiter = self.sync.act_waiter

        observations = [env.reset() for env in self.eval_envs]
        for i, o in enumerate(observations):
            sb_idx = self.n_eval_envs * self.rank + i
            step_buf.obs[sb_idx] = o
        step_blocker.release()

        traj_infos = [TrajInfo(self.discount) for _ in range(self.n_eval_envs)]
        completed_infos = list()
        for _ in range(self.eval_horizon):
            act_waiter.acquire()
            for i, (env, traj_info) in enumerate(zip(self.eval_envs, traj_infos)):
                sb_idx = self.n_eval_envs * self.rank + i
                o, r, d, env_info = env.step(step_buf.act[sb_idx])
                traj_info.step(r, env_info)
                over_length = traj_info.Length >= self.max_path_length
                if over_length or (d and env_info.get("need_reset", True)):
                    d = True
                    o = env.reset()
                    if over_length and "need_reset" in env_info:
                        env_info["need_reset"] = True
                    completed_infos.append(traj_info)
                    traj_infos[i] = TrajInfo(self.discount)
                step_buf.obs[sb_idx] = o
            step_blocker.release()
        print("inside collect_eval, had {} completed_infos".format(len(completed_infos)))
        if not completed_infos:
            traj_lengths = [traj.Length for traj in traj_infos]
            print("inside collect_eval, had traj lengths: ", traj_lengths)
            print("inside collect_eval, eval_horizon: ", self.eval_horizon)
        return completed_infos


def sampling_process(group, rank, unique_ID, EnvCls, env_args, envs_per,
        horizon, eval_horizon, max_path_length, par_objs, seed, cpu,
        discount, mid_batch_reset, max_decorrelation_steps, eval_envs_per):

    # if not mid_batch_reset:
    #     raise NotImplementedError

    ctrl, infos_queue, sync, segs_buf, step_buf, eval_step_buf = par_objs
    initialize_worker(group, rank, seed, cpu)
    envs = [EnvCls(**env_args) for _ in range(envs_per)]
    eval_envs = [EnvCls(**env_args) for _ in range(eval_envs_per)]

    CollectorCls = ResetCollector if mid_batch_reset else NonResetCollector
    collector = CollectorCls(
        rank=rank,
        envs=envs,
        eval_envs=eval_envs,
        sync=sync,
        segs_buf=segs_buf,
        step_buf=step_buf,
        eval_step_buf=eval_step_buf,
        horizon=horizon,
        eval_horizon=eval_horizon,
        max_path_length=max_path_length,
        discount=discount,
    )

    observations, traj_infos = start_envs(envs, max_decorrelation_steps,
        max_path_length, discount, unique_ID)
    for i, o in enumerate(observations):
        sb_idx = envs_per * rank + i
        step_buf.obs[sb_idx] = o
    ctrl.barrier_out.wait()  # So the master can know when envs are started

    while True:
        ctrl.barrier_in.wait()
        if ctrl.quit.value:
            break
        elif ctrl.do_eval.value:
            completed_infos = collector.collect_eval()
            print("worker has {} completed infos".format(len(completed_infos)))
        else:
            traj_infos, completed_infos = collector.collect(traj_infos)

        for info in completed_infos:
            infos_queue.put(info)
        ctrl.barrier_out.wait()

        if not ctrl.do_eval.value and not mid_batch_reset:
            collector.reset_needed_envs()


