
from accel_rl.sampler.act_server.alternating.overlap.sampler import \
    ActsrvAltOvrlpSampler


class ActsrvAltOvrlpPollSampler(ActsrvAltOvrlpSampler):

    def __init__(self, poll_horizon, **kwargs):
        super().__init__(**kwargs)
        self._poll_horizon = poll_horizon

    def poll_init(self, central_shared_params, params_lock, params_rwlock=None):
        self._central_shared_params = central_shared_params
        self._params_lock = params_lock
        self._params_rwlock = params_rwlock

    def serve_actions(self, itr):
        policy = self.policy
        horizon = self.horizon
        step_blockers = self.sync.step_blockers
        act_waiters = self.sync.act_waiters
        pol_buf_pair = self._pol_groups_bufs
        extra_obs_pair = self._extra_obs_groups
        step_bufs = self.step_bufs

        params = self._central_shared_params
        lock = self._params_lock
        rwlock = self._params_rwlock
        poll_horizon = self._poll_horizon

        for s in range(horizon):
            if (s + 1) % poll_horizon == 0:
                if rwlock is not None:
                    rwlock.reader_acquire()
                    policy.set_param_values(params, trainable=True)
                    rwlock.reader_release()
                else:
                    with lock:
                        policy.set_param_values(params, trainable=True)
            for j in range(2):
                step_buf = step_bufs[j]
                pol_buf = pol_buf_pair[j]
                for b in step_blockers[j]:
                    b.acquire()
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
            extra_obs_pair[j][:] = step_bufs[j].obs
