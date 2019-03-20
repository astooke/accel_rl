
import numpy as np

from accel_rl.algos.dqn.replay_buffers.frame import FrameReplayBuffer


class UniformReplayBuffer(FrameReplayBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffer_full = False

    def append_data(self, samples_data):
        super().append_data(samples_data)
        if self.idx == 0:  # (have wrapped--will always land at 0)
            self._buffer_full = True

    def sample_batch(self, batch_size):
        env_idxs, step_idxs = self.sample_idxs(batch_size)
        batch_data = self.extract_batch(env_idxs, step_idxs)
        return batch_data

    ###########################################################################
    #  Helper methods                                                         #
    ###########################################################################

    def sample_idxs(self, batch_size):
        """Uniform sampling, with replacement for speed (may be duplicates)."""
        n = self.num_img_obs
        idx = self.idx
        h_r = self.reward_horizon
        env_idxs = np.random.randint(
            low=0,
            high=self.n_environments,
            size=batch_size,
        )
        if self._buffer_full:
            # (n - 1) states have invalid observations due to frame overlap
            # reward_horizon states will have invalid next state
            step_idx_high = self.env_replay_size - (n - 1) - h_r
        else:
            step_idx_high = idx - h_r
        step_idxs = np.random.randint(
            low=0,
            high=step_idx_high,
            size=batch_size,
        )
        # shift idxs up to valid values as needed
        if idx <= h_r:
            # (invalids at the beginning due to reward_horizon)
            step_idxs += n - 1 + idx
        elif idx >= self.env_replay_size - (n - 1):
            # (possible invalids at beginning due to frame buffer wrap)
            step_idxs += (n - 1 + idx) % self.env_replay_size
        else:
            # (invalids in the middle, starting reward_horizon behind current)
            idxs_to_shift = np.where(step_idxs >= idx - h_r)[0]
            step_idxs[idxs_to_shift] += (n - 1) + h_r
        return env_idxs, step_idxs
