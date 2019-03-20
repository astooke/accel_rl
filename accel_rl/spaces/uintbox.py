
from rllab.spaces.base import Space
import numpy as np
from rllab.misc import ext


class UintBox(Space):
    """
    A box in +Z^n.
    I.e., each coordinate is bounded.
    dtype is unsigned-integer

    NO MORE FLATTENING.
    """

    def __init__(self, shape, low=0, high=None, bits=8):
        """
        Must provide shape.

        Same limits (low, high) are applied to every element.

        Data type is unsigned integer ( e.g., for Atari pixels)
        """
        assert bits in [8, 16, 32, 64]
        self.dtype = "uint" + str(bits)
        max_val = 2 ** bits - 1
        if high is None:
            high = max_val
        low = np.asarray(low, dtype=self.dtype)
        high = np.asarray(high, dtype=self.dtype)
        assert low.shape == () and high.shape == ()
        assert low >= 0
        assert high > low
        assert high <= max_val
        self.low = low
        self.high = high
        self._shape = tuple(shape)

    def sample(self):
        return np.random.randint(low=self.low, high=self.high, size=self._shape, dtype=self.dtype)

    def sample_n(self, n):
        return np.random.randint(low=self.low, high=self.high, size=(n,) + self._shape, dtype=self.dtype)

    def contains(self, x):
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()

    @property
    def shape(self):
        return self._shape

    @property
    def flat_dim(self):
        return int(np.prod(self._shape))

    @property
    def bounds(self):
        return self.low, self.high

    # def flatten(self, x):
    #     return np.asarray(x).reshape(-1)  # (no copy, as flatten() does)

    # def unflatten(self, x):
    #     return np.asarray(x).reshape(self.shape)

    # def flatten_n(self, xs):
    #     xs = np.asarray(xs)
    #     return xs.reshape((xs.shape[0], -1))

    # def unflatten_n(self, xs):
    #     xs = np.asarray(xs)
    #     return xs.reshape((xs.shape[0],) + self.shape)

    @property
    def default_value(self):
        return (self.low + self.high) // 2

    def __repr__(self):
        return "Uint{}Box".format(self.dtype[4:]) + str(self.shape)

    def __eq__(self, other):
        return isinstance(other, UintBox) and np.allclose(self.low, other.low) and \
            np.allclose(self.high, other.high) and self.dtype == other.dtype

    def __hash__(self):
        return hash((self.low, self.high))

    def new_tensor_variable(self, name, extra_dims):
        return ext.new_tensor(
            name=name,
            ndim=extra_dims + len(self._shape),
            dtype=self.dtype
        )
