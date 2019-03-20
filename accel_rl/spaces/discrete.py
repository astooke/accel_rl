from rllab.spaces.base import Space
import numpy as np
from rllab.misc import special
from rllab.misc import ext


class Discrete(Space):
    """
    {0,1,...,n-1}
    """

    def __init__(self, n):
        self._n = n
        if n <= 2 ** 8:
            self._dtype = 'uint8'
        elif n <= 2 ** 16:
            self._dtype = 'uint16'
        else:
            self._dtype = 'uint32'
        self._items_arr = np.arange(n).astype(self.dtype)

    @property
    def n(self):
        return self._n

    @property
    def dtype(self):
        return self._dtype

    def sample(self):
        return np.random.randint(self.n, dtype=self.dtype)

    def sample_n(self, n):
        return np.random.randint(low=0, high=self.n, size=n, dtype=self.dtype)

    def contains(self, x):
        x = np.asarray(x)
        return x.shape == () and x.dtype.kind == 'i' and x >= 0 and x < self.n

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        if not isinstance(other, Discrete):
            return False
        return self.n == other.n

    def flatten(self, x):
        return special.to_onehot(x, self.n)

    def unflatten(self, x):
        return special.from_onehot(x)

    def flatten_n(self, x):
        return special.to_onehot_n(x, self.n)

    def unflatten_n(self, x):
        return special.from_onehot_n(x)

    @property
    def flat_dim(self):
        return self.n

    def weighted_sample(self, weights):
        return special.weighted_sample(weights, self._items_arr)

    def weighted_sample_n(self, weights_matrix):
        return special.weighted_sample_n(weights_matrix, self._items_arr)

    @property
    def default_value(self):
        return 0

    def new_tensor_variable(self, name, extra_dims):
        return ext.new_tensor(
            name=name,
            ndim=extra_dims,
            dtype=self.dtype,
        )

    def __hash__(self):
        return hash(self.n)
