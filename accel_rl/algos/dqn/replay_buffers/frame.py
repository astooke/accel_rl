
import numpy as np

from accel_rl.buffers.batch import build_array


class FrameReplayBuffer(object):
    """
    Stores unique frames only (reduced memory footprint).
    Assumes the observation is composed of consective frames, so that
    two consecutive observations would be, e.g.:
    obs0: [frame0, frame1, frame2]
    obs1: [frame1, frame2, frame3]  (where higher number is newer)

    All buffers are environment-wise.

    Frame-history following a reset will be blank (zeros).

    Should be subclassed depending on sampling method (e.g. uniform,
    proportional)
    """

    def __init__(
            self,
            env_spec,
            size,
            reward_horizon,
            sampling_horizon,
            n_environments,
            discount,
            reward_dtype="float32",
            ):

        sampling_size = sampling_horizon * n_environments
        n_chunks = -(-size // sampling_size)  # (ceiling division)
        replay_size = n_chunks * sampling_size
        self.env_replay_size = env_replay_size = replay_size // n_environments
        assert n_environments * env_replay_size == replay_size
        self.n_environments = n_environments

        env_buf_args = dict(
            env_spec=env_spec,
            size=env_replay_size,
            reward_horizon=reward_horizon,
            discount=discount,
            reward_dtype=reward_dtype,
        )
        self.env_bufs = [EnvBuffer(**env_buf_args) for _ in range(n_environments)]

        self.num_img_obs = self.env_bufs[0].num_img_obs
        self.reward_horizon = reward_horizon
        self.sampling_horizon = sampling_horizon
        self.discount = discount

        self.idx = 0  # (where to write the next state)

    def append_data(self, samples_data):
        for env_buf, path in zip(self.env_bufs, samples_data["segs_view"]):
            env_buf.write_samples(path, self.idx)
        self.idx = (self.idx + self.sampling_horizon) % self.env_replay_size

    def sample_batch(self, batch_size):
        raise NotImplementedError

    ###########################################################################
    #  Helper methods                                                         #
    ###########################################################################

    def extract_batch(self, env_idxs, step_idxs):
        next_step_idxs = (step_idxs + self.reward_horizon) % self.env_replay_size
        observations = self.extract_observations(env_idxs, step_idxs)
        next_observations = self.extract_observations(env_idxs, next_step_idxs)
        actions = np.array([self.env_bufs[e].acts[i]
            for e, i in zip(env_idxs, step_idxs)])
        returns = np.array([self.env_bufs[e].returns[i]
            for e, i in zip(env_idxs, step_idxs)])
        terminals = np.array([self.env_bufs[e].terminals[i]
            for e, i in zip(env_idxs, step_idxs)])
        return observations, next_observations, actions, returns, terminals

    def extract_observations(self, env_idxs, step_idxs):
        n = self.num_img_obs
        observations = np.stack([self.env_bufs[e].frames[i:i + n]
            for e, i in zip(env_idxs, step_idxs)])
        blanks = [self.env_bufs[e].n_blanks[i]
            for e, i in zip(env_idxs, step_idxs)]
        for j, b in enumerate(blanks):
            if b:
                observations[j][:b] = 0  # (after reset, some blank frames)
        return observations


class EnvBuffer(object):
    """
    Handles the data for one environment instance.
    """

    def __init__(
            self,
            env_spec,
            size,
            reward_horizon,
            discount,
            reward_dtype,
            ):
        self.reward_horizon = reward_horizon
        self.discount = discount

        obs = env_spec.observation_space.sample()
        act = env_spec.action_space.sample()
        self.num_img_obs = n = obs.shape[0]
        # (redundant frame storage for easier idx wrapping)
        n_frames = size + n - 1

        self.frames = build_array(obs[0], n_frames)
        self.acts = build_array(act, size)
        self.n_blanks = np.zeros(n_frames, dtype=np.uint8)
        self.terminals = np.zeros(size, dtype=np.bool)
        self.rewards = np.zeros(size, dtype=reward_dtype)
        self.returns = np.zeros(size, dtype=reward_dtype)

    def write_samples(self, path_data, idx):
        """
        idx: idx of next state to be written
        """
        n = self.num_img_obs
        h_s = len(path_data["rewards"])  # sampling_horizon
        h_r = self.reward_horizon
        f_idx = idx + (n - 1)  # (frame_idx)
        r_idx = idx - (h_r - 1)  # (return_idx: negative wraps automatically)

        # If just wrapped, copy relevant ending values to beginning
        # (Assumes env_size is multiple of sampling horizon, will get idx == 0)
        # (Do AFTER wrapping next_idx but BEFORE in-processing new samples)
        if idx == 0:
            self.frames[:n - 1] = self.frames[-(n - 1):]
            self.n_blanks[:n - 1] = self.n_blanks[-(n - 1):]

        self.acts[idx:idx + h_s] = path_data["actions"]
        self.rewards[idx:idx + h_s] = path_data["rewards"]
        self.terminals[idx:idx + h_s] = terms = path_data["dones"]
        for t, obs in enumerate(path_data["observations"]):
            self.frames[f_idx + t] = obs[-1]  # copy only last (newest) frame
        for t, term in enumerate(terms):
            if term:
                # populate the following states with n_blanks = [3, 2, 1]
                self.n_blanks[idx + t + 1:idx + t + n] = np.arange(n - 1, 0, -1)
            elif self.n_blanks[idx + t + 1] and \
                    self.n_blanks[idx + t + 1] >= self.n_blanks[idx + t]:
                # this non-zero n_blanks is leftover: clear it
                # (have to do one at a time because wrapping caused a
                # problem where a 3 could get over-written, with 2's and 1's
                # left dangling, when checking env_blanks[idx + t + 1] == 3)
                self.n_blanks[idx + t + 1] = 0
        for t in range(h_s):
            ret = self.rewards[r_idx + t]
            if not self.terminals[r_idx + t]:
                for i in range(1, h_r):
                    ret += (self.discount ** i) * self.rewards[r_idx + t + i]
                    if self.terminals[r_idx + t + i]:
                        # NOTE: careful not to reverse the range(h_s) for-loop
                        # when checking and setting terminal status here.
                        # Mark terminal within (h_r - 1) steps of actual end
                        self.terminals[r_idx + t] = True
                        break
            self.returns[r_idx + t] = ret
