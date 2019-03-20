
from accel_rl.sampler.act_server.alternating.overlap.sampler import ActsrvAltOvrlpSampler
from accel_rl.sampler.act_server.alternating.overlap.worker_with_eval import sampling_process


class AAOEvalSampler(ActsrvAltOvrlpSampler):

    def __init__(self, eval_steps, eval_envs_per, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_envs_per = eval_envs_per
        self._total_n_eval_envs = eval_envs_per * self.n_parallel * 2
        self.eval_horizon = eval_steps // self._total_n_eval_envs
        self.common_kwargs["eval_horizon"] = self.eval_horizon
        self.common_kwargs["eval_envs_per"] = eval_envs_per

    def initialize(self, *args, **kwargs):
        return super().initialize(*args, worker_process_target=sampling_process,
            **kwargs)

    def evaluate_policy(self, itr):
        self.ctrl.do_eval.value = True
        prev_step_buf_obs = [step_buf.obs.copy() for step_buf in self.step_bufs]
        self.ctrl.barrier_in.wait()
        self.serve_actions_eval(itr)
        self.ctrl.barrier_out.wait()
        traj_infos = list()
        while self.traj_infos_queue.qsize():
            traj_infos.append(self.traj_infos_queue.get())
        self.ctrl.do_eval.value = False
        for step_buf, prev_obs in zip(self.step_bufs, prev_step_buf_obs):
            step_buf.obs[:] = prev_obs
        return traj_infos

    def serve_actions_eval(self, itr):
        """Does not store any state-action-agent info"""
        policy = self.policy
        horizon = self.eval_horizon
        step_blockers = self.sync.step_blockers
        act_waiters = self.sync.act_waiters
        step_bufs = self.eval_step_bufs

        for _ in range(horizon):
            for j in range(2):
                step_buf = step_bufs[j]
                for b in step_blockers[j]:
                    b.acquire()
                acts, agent_infos = policy.get_actions(step_buf.obs)
                step_buf.act[:] = acts
                for w in act_waiters[j]:
                    w.release()

        for j in range(2):
            for b in step_blockers[j]:
                b.acquire()
