
"""TEMPORARY: delete soon if pg_cnn_rnn works """


import theano.tensor as T
import lasagne.layers as L
import lasagne.init as LI
import lasagne.nonlinearities as LN

from accel_rl.policies.layers import \
    NormCInit, ScalarFixedScaleLayer, RecurrentLayer


class PgCnnRnn(object):
    """
    Policy gradient Convolutional net with Recurrent layers on top.
    Outputs: Pi and V.
    """
    def __init__(
            self,
            input_shape,
            output_dim,
            hidden_sizes,
            conv_filters, conv_filter_sizes, conv_strides, conv_pads,
            hidden_nonlinearity=LN.tanh,
            output_pi_nonlinearity=LN.softmax,
            conv_W_init=LI.GlorotUniform(),
            hidden_W_init=NormCInit(1.0),
            output_pi_W_init=NormCInit(0.01),  # (like in OpenAI baselines)
            output_v_W_init=NormCInit(1.0),
            all_b_init=LI.Constant(0.),
            hidden_h_init=LI.Constant(0.),
            hidden_init_trainable=False,
            pixel_scale=255.,
            input_var=None,
            name=None,
            ):

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        if input_var is None:
            # NOTE: decide whether to provide frame history in each obs
            input_var = T.tensor5('input', dtype='uint8')
            step_input_var = T.tensor4('step_input', dtype='uint8')
        assert input_var.ndim == 5
        assert len(input_shape) == 3

        # First, build the TRAINING layers. ###################################
        l_in = L.InputLayer(shape=(None, None) + input_shape, input_var=input_var)
        x = ScalarFixedScaleLayer(l_in, scale=pixel_scale, name='pixel_scale')
        # Reshape leading dimensions for convolution (n_batch x n_step)
        n_batch, n_step = x.shape[0], x.shape[1]
        x = L.reshape(x, (n_batch * n_step,) + input_shape)
        ls_conv = list()
        for i, (n_filters, size, stride, pad) in enumerate(zip(
                conv_filters, conv_filter_sizes, conv_strides, conv_pads)):
            x = L.Conv2DLayer(
                x,
                num_filters=n_filters,
                filter_size=size,
                stride=(stride, stride),
                pad=pad,
                nonlinearity=hidden_nonlinearity,
                W=conv_W_init,
                b=all_b_init,
                name="%sconv_hidden_%d" % (prefix, i),
            )
            ls_conv.append(x)
        l_conv_out = x
        # Reshape leading dimensions for recurrence, might as well flatten trailing
        x = L.reshape(x, [n_batch, n_step, -1])
        ls_recurrent = list()
        ls_prev_hidden = list()
        for i, hidden_size in enumerate(hidden_sizes):
            l_prev_hidden = L.InputLayer(shape=(None,) + hidden_size)
            x = RecurrentLayer(
                incomings=(x, l_prev_hidden),
                num_units=hidden_size,
                nonlinearity=hidden_nonlinearity,
                name="%shidden_recurrent_%d" % (prefix, i),
                W_xh=hidden_W_init,
                W_hh=hidden_W_init,
                b=all_b_init,
                h0=hidden_h_init,
                h0_trainable=hidden_init_trainable,
            )
            ls_prev_hidden.append(l_prev_hidden)
            ls_recurrent.append(x)
        l_out_pi = L.DenseLayer(
            x,
            num_units=output_dim,
            nonlinearity=output_pi_nonlinearity,
            name="%soutput_pi" % (prefix,),
            W=output_pi_W_init,
            b=all_b_init,
        )
        l_out_pi = L.reshape(l_out_pi, [n_batch, n_step, output_dim])
        l_out_v = L.DenseLayer(
            x,
            num_units=1,
            nonlinearity=LN.linear,
            name="%soutput_v" % (prefix,),
            W=output_v_W_init,
            b=all_b_init,
        )
        l_out_v = L.reshape(l_out_v, [n_batch, n_step])

        # Second, repeat for one-step INFERENCE layers (no time dimension) ####
        # (yes this is annoying)
        # Needed because input has different n_dim, and different reshaping
        # throughout.
        l_step_in = L.InputLayer(shape=(None,) + input_shape, input_var=step_input_var)
        x_step = ScalarFixedScaleLayer(l_step_in, scale=pixel_scale, name='step_pixel_scale')
        for i, (n_filters, size, stride, pad) in enumerate(zip(
                conv_filters, conv_filter_sizes, conv_strides, conv_pads)):
            x_step = L.Conv2DLayer(
                x_step,
                num_filters=n_filters,
                filter_size=size,
                stride=(stride, stride),
                pad=pad,
                nonlinearity=hidden_nonlinearity,
                W=ls_conv[i].W,  # must use the same parameter variable
                b=ls_conv[i].b,
                name="%sconv_hidden_step_%d" % (prefix, i),
            )
        l_step_conv_out = x_step
        ls_step_hidden = list()
        for i, hidden_size in enumerate(hidden_sizes):
            x_step = RecurrentLayer(
                incomings=(x_step, ls_prev_hidden[i]),
                num_units=hidden_size,
                nonlinearity=hidden_nonlinearity,
                name="%shidden_step_%d" % (prefix, i),
                W_xh=ls_recurrent[i].W_xh,
                W_hh=ls_recurrent[i].W_hh,
                b=ls_recurrent[i].b,
                h0=ls_recurrent[i].h0,
                h0_trainable=hidden_init_trainable,
            )
            ls_step_hidden.append(x_step)

        l_step_pi = L.DenseLayer(
            x_step,
            num_units=output_dim,
            nonlinearity=output_pi_nonlinearity,
            W=l_out_pi.W,
            b=l_out_pi.b,
        )

        l_step_v = L.DenseLayer(
            x_step,
            num_units=1,
            nonlinearity=LN.linear,
            W=l_out_v.W,
            b=l_out_v.b,
        )
        l_step_v = L.reshape(l_step_v, [-1])

        self._l_out_pi = l_out_pi
        self._l_out_v = l_out_v
        self._l_in = l_in
        self._l_conv_out = l_conv_out
        self._ls_recurrent = ls_recurrent
        self._ls_prev_hidden = ls_prev_hidden

        self._l_step_in = l_step_in
        self._ls_step_hidden = ls_step_hidden
        self._l_step_pi = l_step_pi
        self._l_step_v = l_step_v
        self._l_step_conv_out = l_step_conv_out

    @property
    def input_layer(self):
        return self._l_in

    @property
    def input_var(self):
        return self._l_in._input_var

    @property
    def output_layer_pi(self):
        return self._l_out_pi

    @property
    def output_layer_v(self):
        return self._l_out_v

    @property
    def output_layers(self):
        return [self._l_out_pi, self._l_out_v]

    @property
    def conv_output_layer(self):
        return self._l_conv_out

    @property
    def prev_hidden_layers(self):
        return self._ls_prev_hidden

    @property
    def step_input_layer(self):
        return self._l_step_input

    @property
    def step_hidden_layers(self):
        return self._ls_step_hidden

    @property
    def step_output_layer_pi(self):
        return self._l_step_pi

    @property
    def step_output_layer_v(self):
        return self._l_step_v

    @property
    def step_output_layers(self):
        return [self._l_step_pi, self._l_step_v]

    @property
    def hid_init_params(self):
        return [layer.h0 for layer in self._ls_recurrent]

    def get_params(self, **tags):
        return L.get_all_params(self.output_layers, **tags)
