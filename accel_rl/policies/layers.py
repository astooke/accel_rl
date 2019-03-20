

import numpy as np
import theano
import theano.tensor as T
import lasagne.layers as L
import lasagne.init as LI
import lasagne.nonlinearities as LN


class NormCInit(LI.Initializer):

    def __init__(self, std=1.0):
        self.std = std

    def sample(self, shape):  # from OpenAI Baselines
        out = np.random.randn(*shape).astype(theano.config.floatX)
        out *= self.std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return out


class ScalarFixedScaleLayer(L.Layer):
    """
    A layer which divides input by a fixed scalar
    (multiplies by reciprocal of scale)
    """

    def __init__(self, incoming, scale=1., name='scale'):
        super().__init__(incoming, name=name)
        self.scale = self.add_param(
            spec=T.constant(1. / scale),
            shape=(),
            name='scale',
            regularizable=False,
            trainable=False,
        )

    def get_output_for(self, input, **kwargs):
        pattern = ['x' for _ in range(input.ndim)]
        return input * self.scale.dimshuffle(*pattern)
        # not sure why dimshuffle instead of just input * self.scale


class RecurrentLayer(L.MergeLayer):
    """
    Vanilla recurrent layer that supports vectorized operation, meaning it can
    hold multiple independent copies of the hidden state.

    h(t) = s(x(t)W_xh + h(t-1)W_hh + b)

    param incomings should be: (x_input_layer, prev_hidden_input_layer)
    """

    def __init__(self, incomings, num_units,
            nonlinearity=LN.tanh,
            name=None,
            W_xh=LI.GlorotUniform(),
            W_hh=LI.GlorotUniform(),
            b=LI.Constant(0.),
            h0=LI.Constant(0.),
            # h0_trainable=False,
            ):

        super().__init__(incomings, name=name)

        input_shape = self.input_shapes[0][1:]
        input_dim = np.int(np.prod(input_shape))
        # NOTE: for now, not set up to train this...probably never need to,
        # but should make it noisy.
        self.h0 = self.add_param(h0, (num_units,), name="h0",
            trainable=False, regularizable=False)

        self.W_xh = self.add_param(W_xh, (input_dim, num_units), name="W_xh")
        self.W_hh = self.add_param(W_hh, (num_units, num_units), name="W_hh")
        self.b = self.add_param(b, (num_units,), name="b", regularizable=False)

        self.num_units = num_units
        self.nonlinearity = nonlinearity

    def step(self, x, hprev):
        h = self.nonlinearity(x.dot(self.W_xh) + hprev.dot(self.W_hh) + self.b)
        return h

    def get_output_shape_for(self, input_shapes):
        n_batch_x_step = input_shapes[0][0]
        assert input_shapes[0][0] == input_shapes[1][0]
        return n_batch_x_step, self.num_units

    def get_output_for(self, inputs, step_or_train="train", **kwargs):
        xs, hprevs = inputs  # (hprevs is all initial states, one timestep)
        n_batch = hprevs.shape[0]
        if step_or_train == "train":
            # Assume data is concatenated trajectories of same length
            n_step = xs.shape[0] // n_batch  # NOTE: Inferred number of time steps
            xs = T.reshape(xs, (n_batch, n_step, -1))  # (flatten the rest)
            xs = xs.dimshuffle(1, 0, 2)  # put time dimension first
            hs, _ = theano.scan(fn=self.step, sequences=[xs], outputs_info=hprevs)
            hs = hs.dimshuffle(1, 0, 2)  # put batch dimension first
            hs = hs.reshape((n_batch * n_step, -1))  # get rid of time dimension
        elif step_or_train == "step":
            n_batch = hprevs.shape[0]
            xs = xs.reshape((n_batch, -1))  # (flatten the rest)
            hs = self.step(xs, hprevs)
        else:
            raise ValueError("Unrecognized step_or_train: {}".format(step_or_train))
        return hs


class GruLayer(L.MergeLayer):
    """
    GRU recurrent layer that supports vectorized operation, meaning it can
    hold multiple independent copies of the hidden state.

    A gated recurrent unit implements the following update mechanism:
    Reset gate:        r(t) = f_r(x(t) @ W_xr + h(t-1) @ W_hr + b_r)
    Update gate:       u(t) = f_u(x(t) @ W_xu + h(t-1) @ W_hu + b_u)
    Cell gate:         c(t) = f_c(x(t) @ W_xc + r(t) * (h(t-1) @ W_hc) + b_c)
    New hidden state:  h(t) = (1 - u(t)) * h(t-1) + u_t * c(t)
    Note that the reset, update, and cell vectors must have the same dimension as the hidden state

    param incomings should be: (x_input_layer, prev_hidden_input_layer)
    """

    def __init__(self, incomings, num_units,
            nonlinearity=LN.tanh,
            gate_nonlinearity=LN.sigmoid,
            name=None,
            W=LI.GlorotUniform(),
            b=LI.Constant(0.),
            h0=LI.Constant(0.),
            # h0_trainable=False,
            ):

        super().__init__(incomings, name=name)

        input_shape = self.input_shapes[0][1:]
        input_dim = np.int(np.prod(input_shape))
        # NOTE: for now, not set up to train this...probably never need to,
        # but should make it noisy.
        self.h0 = self.add_param(h0, (num_units,), name="h0",
                                 trainable=False, regularizable=False)

        self.num_units = num_units
        self.nonlinearity = nonlinearity
        self.gate_nonlinearity = gate_nonlinearity

        self.W_xh = self.add_param(W, (input_dim, num_units), name="W_xh")
        self.W_hh = self.add_param(W, (num_units, num_units), name="W_hh")
        self.b = self.add_param(b, (num_units,), name="b", regularizable=False)

        self.W_xr = self.add_param(W, (input_dim, num_units), name="W_xr")
        self.W_hr = self.add_param(W, (num_units, num_units), name="W_hr")
        self.b_r = self.add_param(b, (num_units,), name="b_r", regularizable=False)
        # Weights for the update gate
        self.W_xu = self.add_param(W, (input_dim, num_units), name="W_xu")
        self.W_hu = self.add_param(W, (num_units, num_units), name="W_hu")
        self.b_u = self.add_param(b, (num_units,), name="b_u", regularizable=False)
        # Weights for the cell gate
        self.W_xc = self.add_param(W, (input_dim, num_units), name="W_xc")
        self.W_hc = self.add_param(W, (num_units, num_units), name="W_hc")
        self.b_c = self.add_param(b, (num_units,), name="b_c", regularizable=False)

    def step(self, x, hprev):
        r = self.gate_nonlinearity(x.dot(self.W_xr) + hprev.dot(self.W_hr) + self.b_r)
        u = self.gate_nonlinearity(x.dot(self.W_xu) + hprev.dot(self.W_hu) + self.b_u)
        c = self.nonlinearity(x.dot(self.W_xc) + r * (hprev.dot(self.W_hc)) + self.b_c)
        h = (1 - u) * hprev + u * c
        return h

    def get_output_shape_for(self, input_shapes):
        n_batch_x_step = input_shapes[0][0]
        assert input_shapes[0][0] == input_shapes[1][0]
        return n_batch_x_step, self.num_units

    def get_output_for(self, inputs, step_or_train="train", **kwargs):
        xs, hprevs = inputs  # (hprevs is all initial states, one timestep)
        n_batch = hprevs.shape[0]
        if step_or_train == "train":
            # Assume data is concatenated trajectories of same length
            n_step = xs.shape[0] // n_batch  # NOTE: Inferred number of time steps
            xs = T.reshape(xs, (n_batch, n_step, -1))  # (flatten the rest)
            xs = xs.dimshuffle(1, 0, 2)  # put time dimension first
            hs, _ = theano.scan(fn=self.step, sequences=[xs], outputs_info=hprevs)
            hs = hs.dimshuffle(1, 0, 2)  # put batch dimension first
            hs = hs.reshape((n_batch * n_step, -1))  # get rid of time dimension
        elif step_or_train == "step":
            n_batch = hprevs.shape[0]
            xs = xs.reshape((n_batch, -1))  # (flatten the rest)
            hs = self.step(xs, hprevs)
        else:
            raise ValueError("Unrecognized step_or_train: {}".format(step_or_train))
        return hs


class LstmLayer(L.MergeLayer):
    """
    GRU recurrent layer that supports vectorized operation, meaning it can
    hold multiple independent copies of the hidden state.

    A gated recurrent unit implements the following update mechanism:
    Forget gate:       f(t) = g_f(x(t) @ W_xf + h(t-1) @ W_hf + b_f)
    Input gate:        i(t) = g_i(x(t) @ W_xi + h(t-1) @ W_hi + b_i)
    Cell gate:         c(t) = g_c(x(t) @ W_xc + h(t-1) @ W_hc + b_c)
    Output Gate:       o(t) = g_o(x(t) @ W_xo + h(t-1) @ W_ho + b_o)
    New cell:          C(t) = f(t) * C(t-1) + i(t) * c(t)
    New hidden state:  h(t) = o(t) * g_h(C(t))
    Note that the forger, input, cell and output vectors must have the same dimension as the hidden state

    param incomings should be: (x_input_layer, prev_hidden_input_layer)
    """

    def __init__(self, incomings, num_units,
            nonlinearity=LN.tanh,
            gate_nonlinearity=LN.sigmoid,
            name=None,
            W=LI.GlorotUniform(),
            b=LI.Constant(0.),
            h0=LI.Constant(0.),
            c0=LI.Constant(0.),
            # h0_trainable=False,
            ):

        super().__init__(incomings, name=name)

        input_shape = self.input_shapes[0][1:]
        input_dim = np.int(np.prod(input_shape))
        # NOTE: for now, not set up to train this...probably never need to,
        # but should make it noisy.
        self.h0 = self.add_param(h0, (num_units,), name="h0",
                                 trainable=False, regularizable=False)
        self.c0 = self.add_param(c0, (num_units,), name="c0",
                                 trainable=False, regularizable=False)

        self.num_units = num_units
        self.nonlinearity = nonlinearity
        self.gate_nonlinearity = gate_nonlinearity

        # Weights for the output gate
        self.W_xo = self.add_param(W, (input_dim, num_units), name="W_xo")
        self.W_ho = self.add_param(W, (num_units, num_units), name="W_ho")
        self.b_o = self.add_param(b, (num_units,), name="b_o", regularizable=False)
        # Weights for the forget gate
        self.W_xf = self.add_param(W, (input_dim, num_units), name="W_xf")
        self.W_hf = self.add_param(W, (num_units, num_units), name="W_hf")
        self.b_f = self.add_param(b, (num_units,), name="b_f", regularizable=False)
        # Weights for the input gate
        self.W_xi = self.add_param(W, (input_dim, num_units), name="W_xi")
        self.W_hi = self.add_param(W, (num_units, num_units), name="W_hi")
        self.b_i = self.add_param(b, (num_units,), name="b_i", regularizable=False)
        # Weights for the cell state
        self.W_xc = self.add_param(W, (input_dim, num_units), name="W_xc")
        self.W_hc = self.add_param(W, (num_units, num_units), name="W_hc")
        self.b_c = self.add_param(b, (num_units,), name="b_c", regularizable=False)

    def step(self, x, hprev, cprev):
        f = self.gate_nonlinearity(x.dot(self.W_xf) + hprev.dot(self.W_hf) + self.b_f)
        i = self.gate_nonlinearity(x.dot(self.W_xi) + hprev.dot(self.W_hi) + self.b_i)
        _c = self.nonlinearity(x.dot(self.W_xc) + hprev.dot(self.W_hc) + self.b_c)
        c = f * cprev + i * _c
        o = self.gate_nonlinearity(x.dot(self.W_xo) + hprev.dot(self.W_ho) + self.b_o)
        h = o * self.nonlinearity(c)
        return h, c

    def get_output_shape_for(self, input_shapes):
        n_batch_x_step = input_shapes[0][0]
        assert input_shapes[0][0] == input_shapes[1][0] == input_shapes[2][0]
        return [(n_batch_x_step, self.num_units)] * 2  # (outputs [h, c])

    def get_output_for(self, inputs, step_or_train="train", **kwargs):
        xs, hprevs, cprevs = inputs  # (hprevs and cprevs are all initial states, one timestep)
        n_batch = hprevs.shape[0]
        if step_or_train == "train":
            # Assume data is concatenated trajectories of same length
            n_step = xs.shape[0] // n_batch  # NOTE: Inferred number of time steps
            xs = T.reshape(xs, (n_batch, n_step, -1))  # (flatten the rest)
            xs = xs.dimshuffle(1, 0, 2)  # put time dimension first
            outputs, _ = theano.scan(fn=self.step, sequences=[xs], outputs_info=[hprevs, cprevs])
            hs, cs = outputs
            hs = hs.dimshuffle(1, 0, 2)  # put batch dimension first
            cs = cs.dimshuffle(1, 0, 2)
            hs = hs.reshape((n_batch * n_step, -1))  # get rid of time dimension
            cs = cs.reshape((n_batch * n_step, -1))
        elif step_or_train == "step":
            n_batch = hprevs.shape[0]
            xs = xs.reshape((n_batch, -1))  # (flatten the rest)
            hs, cs = self.step(xs, hprevs, cprevs)
        else:
            raise ValueError("Unrecognized step_or_train: {}".format(step_or_train))
        return [hs, cs]


class FastLstmLayer(L.MergeLayer):

    """
    Same functionality as LSTM layer, but internal optimizations:
    -precompute inputs
    -combined W params
    -possibly unroll_scan...nope this was same speed but slow compile
    """

    def __init__(self, incomings, num_units,
            nonlinearity=LN.tanh,
            gate_nonlinearity=LN.sigmoid,
            name=None,
            W=LI.Orthogonal(1.0),
            b=LI.Constant(0.),
            h0=LI.Constant(0.),
            c0=LI.Constant(0.),
            grad_clipping=0.,
            # h0_trainable=False,
            ):

        super().__init__(incomings, name=name)

        input_shape = self.input_shapes[0][1:]
        input_dim = np.int(np.prod(input_shape))
        self.h0 = self.add_param(h0, (num_units,), name="h0",
                                 trainable=False, regularizable=False)
        self.c0 = self.add_param(c0, (num_units,), name="c0",
                                 trainable=False, regularizable=False)

        self.num_units = num_units
        self.nonlinearity = nonlinearity
        self.gate_nonlinearity = gate_nonlinearity
        self.grad_clipping = grad_clipping

        # Weights for all gates.
        self.W_x = self.add_param(W, (input_dim, num_units * 4), name="W_x")
        self.W_h = self.add_param(W, (num_units, num_units * 4), name="W_h")
        self.b = self.add_param(b, (num_units * 4,), name="b", regularizable=False)

    def step(self, xWb, hprev, cprev):
        # xWb is pre-computed with bias.
        fico = xWb + hprev.dot(self.W_h)
        if self.grad_clipping:
            gc = self.grad_clipping
            fico = theano.gradient.grad_clip(fico, -gc, gc)
        f, i, _c, o = [self.slice_w(fico, n) for n in range(4)]
        f = self.gate_nonlinearity(f)
        i = self.gate_nonlinearity(i)
        _c = self.nonlinearity(_c)
        o = self.gate_nonlinearity(o)
        c = f * cprev + i * _c
        h = o * self.nonlinearity(c)
        return h, c

    def slice_w(self, x, n):
        # From Lasagne.
        s = x[:, n * self.num_units:(n + 1) * self.num_units]
        if self.num_units == 1:
            s = T.addbroadcast(s, 1)
        return s

    def get_output_shape_for(self, input_shapes):
        n_batch_x_step = input_shapes[0][0]
        assert input_shapes[0][0] == input_shapes[1][0] == input_shapes[2][0]
        return [(n_batch_x_step, self.num_units)] * 2  # (outputs [h, c])

    def get_output_for(self, inputs, step_or_train="train", **kwargs):
        xs, hprevs, cprevs = inputs  # (hprevs and cprevs are all initial states, one timestep)
        n_batch = hprevs.shape[0]
        if step_or_train == "train":
            # Assume data is concatenated trajectories of same length
            n_step = xs.shape[0] // n_batch  # NOTE: Inferred number of time steps
            # Precompute inputs (like Lasagne):
            xs = T.reshape(xs, (xs.shape[0], -1))  # flatten then rest
            xWbs = xs.dot(self.W_x) + self.b
            xWbs = T.reshape(xWbs, (n_batch, n_step, -1))  # separate trajectories
            xWbs = xWbs.dimshuffle(1, 0, 2)  # put time dimension first
            (hs, cs), _ = theano.scan(
                fn=self.step,
                sequences=[xWbs],
                outputs_info=[hprevs, cprevs],
            )
            hs = hs.dimshuffle(1, 0, 2)  # restore batch dimension first
            cs = cs.dimshuffle(1, 0, 2)
            hs = hs.reshape((n_batch * n_step, -1))  # re-concat trajectories
            cs = cs.reshape((n_batch * n_step, -1))
        elif step_or_train == "step":
            xs = xs.reshape((n_batch, -1))  # (flatten the rest)
            xWbs = xs.dot(self.W_x) + self.b
            hs, cs = self.step(xWbs, hprevs, cprevs)
        else:
            raise ValueError("Unrecognized step_or_train: {}".format(step_or_train))
        return [hs, cs]


class SelectOutputLayer(L.Layer):
    """
    Select one output from a layer which has multiple outputs in a list.
    """

    def __init__(self, incoming, idx=0, name='Select'):
        self.input_layer = incoming
        input_shapes = incoming.output_shape
        assert isinstance(input_shapes, list)  # A list of shapes.
        assert idx >= 0 and idx < len(input_shapes)
        self._idx = idx

        self.name = name
        self.params = OrderedDict()
        self.get_output_kwargs = []

        for shape in input_shapes:
            if any(d is not None and d <= 0 for d in shape):
                raise ValueError((
                    "Cannot create Layer with a non-positive input_shape "
                    "dimension. input_shape=%r, self.name=%r") % (
                        input_shapes, self.name))
        self.input_shape = input_shapes

    def get_output_shape_for(self, input_shape):
        return input_shape[self._idx]

    def get_output_for(self, input, **kwargs):
        return input[self._idx]

