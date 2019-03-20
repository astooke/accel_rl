
from accel_rl.algos.dqn.replay_buffers.frame import FrameReplayBuffer
from accel_rl.algos.dqn.replay_buffers.sum_tree import PartedSumTree


class PrioritizedReplayBuffer(FrameReplayBuffer):

    def __init__(self, alpha, beta_initial, default_priority, **kwargs):
        super().__init__(**kwargs)
        self.priority_tree = PartedSumTree(
            part_size=self.env_replay_size,
            num_parts=self.n_environments,
            zeros_forward=self.num_img_obs,
            zeros_backward=self.reward_horizon,
            default_value=default_priority ** alpha,
            n_advance=self.sampling_horizon,
        )

        self.alpha = alpha
        self.beta = beta_initial
        self.default_probability = default_priority ** alpha

    def set_beta(self, value):
        self.beta = value

    def append_data(self, samples_data):
        super().append_data(samples_data)
        self.priority_tree.advance()

    def sample_batch(self, batch_size):
        env_idxs, step_idxs, probs = self.priority_tree.sample_n(batch_size)
        batch_data = self.extract_batch(env_idxs, step_idxs)
        is_weights = (1. / probs) ** self.beta  # (don't need any normalization)
        is_weights /= max(is_weights)  # (...because normalizing here)
        return batch_data + (is_weights,)

    def update_batch_priorities(self, priorities):
        self.priority_tree.update_last_samples(priorities ** self.alpha)
