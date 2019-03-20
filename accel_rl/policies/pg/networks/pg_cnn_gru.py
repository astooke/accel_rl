
import theano.tensor as T
import lasagne.layers as L
import lasagne.init as LI
import lasagne.nonlinearities as LN

from accel_rl.policies.base import BaseNetwork
from accel_rl.policies.layers import \
    NormCInit, ScalarFixedScaleLayer, GruLayer


class PgCnnGru(BaseNetwork):
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
            conv_nonlinearity=LN.rectify,
            hidden_nonlinearity=LN.tanh,
            output_pi_nonlinearity=LN.softmax,
            conv_W_init=LI.GlorotUniform(),
            hidden_W_init=NormCInit(1.0),
            output_pi_W_init=NormCInit(0.01),  # (like in OpenAI baselines)
            output_v_W_init=NormCInit(1.0),
            all_b_init=LI.Constant(0.),
            hidden_h_init=LI.Constant(0.),
            # hidden_init_trainable=False,
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
            # NOTE: could still leave this alone and just have 1 channel
            input_var = T.tensor4('step_input', dtype='uint8')
        assert input_var.ndim == 4
        assert len(input_shape) == 3

        # Input will be shape: (n_batch * n_step, *input_shape)
        # (avoid requiring the extra dimension for n_step, it changes too much
        # other code--instead, just reshape by n_batch, n_step automatically
        # in recurrent layer (n_step either inferred or as input))
        l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var)
        x = ScalarFixedScaleLayer(l_in, scale=pixel_scale, name='pixel_scale')
        ls_conv = list()
        for i, (n_filters, size, stride, pad) in enumerate(zip(
                conv_filters, conv_filter_sizes, conv_strides, conv_pads)):
            x = L.Conv2DLayer(
                x,
                num_filters=n_filters,
                filter_size=size,
                stride=(stride, stride),
                pad=pad,
                nonlinearity=conv_nonlinearity,
                W=conv_W_init,
                b=all_b_init,
                name="%sconv_hidden_%d" % (prefix, i),
            )
            ls_conv.append(x)
        l_conv_out = x
        ls_recurrent = list()
        ls_prev_hidden = list()
        for i, hidden_size in enumerate(hidden_sizes):
            #FIXME: I think it's a bit weird do more than one update... (because x is the output of a conv
            # layer, and then it's the next_hidden state of the previous GRU)
            l_prev_hidden = L.InputLayer(
                shape=(None, hidden_size),
                name="%shprev_%d_input" % (prefix, i),
            )
            x = GruLayer(
                incomings=(x, l_prev_hidden),
                num_units=hidden_size,
                nonlinearity=hidden_nonlinearity,
                name="%shidden_gru_%d" % (prefix, i),
                W=hidden_W_init,
                b=all_b_init,
                h0=hidden_h_init,
                # h0_trainable=hidden_init_trainable,
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
        l_out_v = L.DenseLayer(
            x,
            num_units=1,
            nonlinearity=LN.linear,
            name="%soutput_v" % (prefix,),
            W=output_v_W_init,
            b=all_b_init,
        )
        l_out_v = L.reshape(l_out_v, [-1])

        self._l_out_pi = l_out_pi
        self._l_out_v = l_out_v
        self._l_in = l_in
        self._l_conv_out = l_conv_out
        self._ls_recurrent = ls_recurrent
        self._ls_prev_hidden = ls_prev_hidden

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
    def recurrent_layers(self):
        return self._ls_recurrent

    @property
    def hid_init_params(self):
        return [layer.h0 for layer in self._ls_recurrent]

    @property
    def n_hidden_state_params(self):
        return len(self.hid_init_parms)

    def get_params(self, **tags):
        return L.get_all_params(self.output_layers, **tags)
