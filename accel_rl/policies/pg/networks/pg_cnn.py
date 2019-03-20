
import theano.tensor as T
import lasagne.layers as L
import lasagne.init as LI
import lasagne.nonlinearities as LN

from accel_rl.policies.base import BaseNetwork
from accel_rl.policies.layers import NormCInit, ScalarFixedScaleLayer


class PgCnn(BaseNetwork):
    """
    Convolutional net with fully connected layers on top.
    Outputs: Pi and V.
    """

    def __init__(
            self,
            input_shape,
            output_dim,
            hidden_sizes,
            conv_filters, conv_filter_sizes, conv_strides, conv_pads,
            hidden_nonlinearity=LN.rectify,
            output_pi_nonlinearity=LN.softmax,
            conv_W_init=LI.GlorotUniform(),
            hidden_W_init=NormCInit(1.0),
            output_pi_W_init=NormCInit(0.01),  # (like in OpenAI baselines)
            output_v_W_init=NormCInit(1.0),
            all_b_init=LI.Constant(0.),
            pixel_scale=255.,
            input_var=None,
            name=None,
            ):

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        if input_var is None:
            input_var = T.tensor4('input', dtype='uint8')
        assert input_var.ndim == 4
        assert len(input_shape) == 3

        l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var)
        x = ScalarFixedScaleLayer(l_in, scale=pixel_scale, name='pixel_scale')
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
        l_conv_out = x
        for i, hidden_size in enumerate(hidden_sizes):
            x = L.DenseLayer(
                x,
                num_units=hidden_size,
                nonlinearity=hidden_nonlinearity,
                name="%shidden_%d" % (prefix, i),
                W=hidden_W_init,
                b=all_b_init,
            )
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
        self._input_var = input_var
        self._l_conv_out = l_conv_out

    @property
    def input_layer(self):
        return self._l_in

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
    def input_var(self):
        return self._input_var

    @property
    def conv_output_layer(self):
        return self._l_conv_out

    def get_params(self, **tags):
        return L.get_all_params(self.output_layers, **tags)
