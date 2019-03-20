
import numpy as np
import theano.tensor as T

from lasagne.layers import DenseLayer
from lasagne.init import Constant
from lasagne.random import get_rng
from lasagne.utils import floatX


def f(x):
    return T.sgn(x) * T.sqrt(abs(x))


class NoisyDenseLayer(DenseLayer):
    """
    lasagne.layers.NoisyDenseLayer(incoming, num_units, rng, W_sigma=None,
    b_sigma=None, factorized=True, common_noise=False, **kwargs)
    A subclass of lasagne.layers.DenseLayer.
    A fully connected layer with noise like in NoisyNets:
    https://arxiv.org/abs/1706.10295.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    num_units : int
        The number of units of the layer
    rng : a :class:'RandomStreams' instance
        Random number generator from which to make all random variables (by
        making this external to the layer, can use the same one for multiple
        layers, and control state elsewhere).
    W_sigma : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the noise weights.
        These should be a matrix with shape ``(num_inputs, num_units)``.
        See :func:`lasagne.utils.create_param` for more information.
    b_sigma : Theano shared variable, expression, numpy array, callable or
        ``None``
        Initial value, expression or initializer for the noise biases. If set to
        ``None``, the layer will have no noise biases. Otherwise, biases should
        be a 1D array with shape ``(num_units,)``. See
        :func:`lasagne.utils.create_param` for more information.
    factorized : bool
        If True, use factorized noise generation (reduces number of values to
        generate), else generate full independent noise (options (a) and (b)
        in Section 3 of NoisyNets).
    common_noise : bool
        If True, sample one set of noise values to use in computing the output.
        If False, samples a separate set of noise values for each element along
        dim-0 of the input (i.e. treating them as separate data points).  (In
        the NoisyNets paper, Noisy-A3C roughly corresponds to the True setting,
        and Noisy-DQN roughly corresponds to the False setting.)
    num_leading_axes : int
        Whenever `common_noise` is False, this is restricted to 1 in the
        current implementation (until needed otherwise).
    """
    def __init__(self, incoming, num_units, rng, factorized=True,
                 common_noise=False, sigma_0=0.4, use_mu_init=True, **kwargs):

        super(NoisyDenseLayer, self).__init__(incoming, num_units, **kwargs)
        if not common_noise and self.num_leading_axes != 1:
            raise NotImplementedError("Test use of theano.tensor.batched_dot")
        num_inputs = int(np.prod(self.input_shape[self.num_leading_axes:]))

        if use_mu_init:  # (override earlier W and b values, using num_inputs)
            val = np.sqrt(1 / num_inputs) if factorized else \
                np.sqrt(3 / num_inputs)
            for param in [self.W, self.b]:
                param.set_value(floatX(get_rng().uniform(
                    -val, val, param.get_value(borrow=True).shape)))

        # NOTE: paper quotes sigma_0 = 0.017 in case of not factorized
        sigma_0 = sigma_0 / np.sqrt(num_inputs) if factorized else sigma_0
        W_sigma = b_sigma = Constant(sigma_0)

        self.W_sigma = self.add_param(W_sigma, (num_inputs, num_units),
            name="W_sigma")
        if self.b is None:
            self.b_sigma = None
        else:
            self.b_sigma = self.add_param(b_sigma, (num_units,),
                name="b_sigma", regularizable=False)

        if common_noise:
            if factorized:
                self.eps_i = eps_i = rng.normal((num_inputs,))
                self.eps_j = eps_j = rng.normal((num_units,))
                self.W_epsilon = T.outer(f(eps_i), f(eps_j))
                self.b_epsilon = f(eps_j)
            else:
                self.W_epsilon = rng.normal((num_inputs, num_units))
                self.b_epsilon = rng.normal((num_units,))
        else:
            self.num_inputs = num_inputs
            self.num_units = num_units
            self.W_epsilon = None  # Must build later, when have input length
            self.b_epsilon = None
            self.eps_is, self.eps_js = list(), list()
            self.W_epsilons, self.b_epsilons = list(), list()

        self.rng = rng
        self.common_noise = common_noise
        self.factorized = factorized
        self.use_mu_init = use_mu_init

    def get_output_for(self, input, **kwargs):
        num_leading_axes = self.num_leading_axes
        if num_leading_axes < 0:
            num_leading_axes += input.ndim
        if input.ndim > num_leading_axes + 1:
            # flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
            input = input.flatten(num_leading_axes + 1)

        if self.common_noise:
            activation = T.dot(input, self.W + self.W_sigma * self.W_epsilon)
        else:
            num_inputs, num_units = self.num_inputs, self.num_units
            num_data = input.shape[0]  # assume this
            rng = self.rng
            if self.factorized:
                self.eps_i = eps_i = rng.normal((num_data, num_inputs))
                self.eps_j = eps_j = rng.normal((num_data, num_units))
                self.W_epsilon = f(eps_i.dimshuffle(0, 1, 'x')) * \
                    f(eps_j.dimshuffle(0, 'x', 1))
                self.b_epsilon = f(eps_j)
                self.eps_is.append(eps_i)
                self.eps_js.append(eps_j)
            else:
                self.W_epsilon = rng.normal((num_data, num_inputs, num_units))
                self.b_epsilon = rng.normal((num_data, num_units))
            # Note: although have saved e.g. self.W_epsilon here, this is not
            # necessarily persistent within the layer, i.e. each call to
            # get_output_for overrides, but the previous graph should still
            # exist.  But save here in case want to access later.
            self.W_epsilons.append(self.W_epsilon)
            self.b_epsilons.append(self.b_epsilon)

            activation = T.dot(input, self.W)
            # Can't use theano.tensor.batched_dot because it breaks the gradient
            activation += T.batched_dot(input, self.W_sigma * self.W_epsilon)
            # input_bc = T.shape_padright(input)
            # activation += T.sum(input_bc * (self.W_sigma * self.W_epsilon),
            #     axis=1)
        if self.b is not None:
            activation += self.b
        if self.b_sigma is not None:
            activation += self.b_sigma * self.b_epsilon
        return self.nonlinearity(activation)
