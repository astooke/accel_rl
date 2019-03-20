
import numpy as np


class PartedSumTree(object):
    """
    One sum tree, but the leaves are considered divided into groups/parts

    All the operations are vectorized, for speed.
    """

    def __init__(
            self,
            part_size,
            num_parts,
            zeros_forward,
            zeros_backward,
            default_value,
            n_advance,
            ):
        self.part_size = part_size
        self.num_parts = num_parts
        self.zeros_forward = zeros_forward
        self.zeros_backward = zeros_backward
        self.default_value = default_value
        self.n_leaves = part_size * num_parts
        self.tree_level = int(np.ceil(np.log2(self.n_leaves + 1)) + 1)
        self.tree_size = 2 ** self.tree_level - 1
        self.tree = np.zeros(self.tree_size)
        self.t_l_shift = 2 ** (self.tree_level - 1) - 1  # tree_idx to leaf_idx
        self.n_advance = n_advance
        assert part_size % n_advance == 0
        self.step_cursor = 0
        self.step_idxs_vector = np.arange(part_size)
        self.set_initial_values()
        self.n_ons = n_advance * num_parts
        self.on_diffs = self.default_value * np.ones(self.n_ons)

    def set_initial_values(self):
        """ Just for wrapped turn-on during very first advance(s) """
        leaf_idxs = list()
        for p in range(self.num_parts):
            last_leaf_idx = (p + 1) * self.part_size - 1
            for i in range(self.zeros_backward):
                leaf_idxs.append(last_leaf_idx - i)
        diffs = -self.default_value * np.ones(len(leaf_idxs))
        tree_idxs = np.array(leaf_idxs) + self.t_l_shift
        self.reconstruct(tree_idxs, diffs)

    def reconstruct(self, tree_idxs, diffs):
        for _ in range(self.tree_level):
            np.add.at(self.tree, tree_idxs, diffs)
            tree_idxs = (tree_idxs - 1) // 2

    def advance(self):
        cursor = self.step_cursor
        on = cursor - self.zeros_backward
        off = cursor - self.part_size + self.zeros_forward
        ons = self.step_idxs_vector[list(range(on, on + self.n_advance))]
        offs = self.step_idxs_vector[list(range(off, off + self.n_advance))]
        ons = [ons + p * self.part_size for p in range(self.num_parts)]
        offs = [offs + p * self.part_size for p in range(self.num_parts)]
        on_off_idxs = np.concatenate(ons + offs) + self.t_l_shift
        off_diffs = -self.tree[on_off_idxs[self.n_ons:]]
        on_off_diffs = np.concatenate([self.on_diffs, off_diffs])
        self.reconstruct(on_off_idxs, on_off_diffs)
        self.step_cursor = (cursor + self.n_advance) % self.part_size

    def update_last_samples(self, new_values):
        self.reconstruct(self.last_tree_idxs, new_values - self.last_probs)

    def sample_n(self, n):
        """ Get n unique samples """
        tree_idxs = np.unique(self.find(np.random.rand(int(1.05 * n))))
        i = 0
        while len(tree_idxs) < n:
            i += 1
            if i > 100:
                raise RuntimeError("After 100 tries, unable to get unique idxs")
            new_idxs = self.find(np.random.rand(2 * (n - len(tree_idxs))))
            tree_idxs = np.unique(np.concatenate([tree_idxs, new_idxs]))
        self.last_tree_idxs = tree_idxs = tree_idxs[:n]
        self.last_probs = probs = self.tree[tree_idxs]
        env_idxs, step_idxs = \
            np.divmod(tree_idxs - self.t_l_shift, self.part_size)
        return env_idxs, step_idxs, probs

    def find(self, random_values):
        """ Random values: numpy array of floats in range [0, 1] """
        random_values *= self.tree[0]
        tree_idxs = np.zeros(len(random_values), dtype=np.int32)
        for _ in range(self.tree_level - 1):
            tree_idxs = 2 * tree_idxs + 1
            left_values = self.tree[tree_idxs]
            where_right = np.where(random_values > left_values)[0]
            tree_idxs[where_right] += 1
            random_values[where_right] -= left_values[where_right]
        return tree_idxs

    def print_tree(self):
        for k in range(1, self.tree_level + 1):
            for j in range(2 ** (k - 1) - 1, 2 ** k - 1):
                print(self.tree[j], end=' ')
            print()
