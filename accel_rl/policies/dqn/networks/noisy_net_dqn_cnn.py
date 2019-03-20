
import theano.tensor as T
import lasagne.layers as L
import lasagne.init as LI
import lasagne.nonlinearities as LN

from accel_rl.policies.base import BaseNetwork
from accel_rl.policies.layers import NormCInit, ScalarFixedScaleLayer
from accel_rl.policies.dqn.layers.noisy_layer import NoisyDenseLayer


class NoisyNetDqnCnn(BaseNetwork):
    """
    Standard convolutional net with fully connected layers on top.
    """

    def __init__(
            self,
            input_shape,
            output_dim,
            hidden_sizes,
            conv_filters, conv_filter_sizes, conv_strides, conv_pads,
            hidden_nonlinearity=LN.rectify,
            conv_W_init=LI.GlorotUniform(),
            hidden_W_init=NormCInit(1.0),
            output_W_init=NormCInit(0.01),  # (like in OpenAI baselines)
            all_b_init=LI.Constant(0.),
            pixel_scale=255.,
            input_var=None,
            name=None,
            rng=None,
            factorized=True,
            common_noise=False,
            use_mu_init=True,
            sigma_0=0.4,
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
            x = NoisyDenseLayer(
                x,
                num_units=hidden_size,
                rng=rng,
                factorized=factorized,
                common_noise=common_noise,
                use_mu_init=use_mu_init,
                sigma_0=sigma_0,
                nonlinearity=hidden_nonlinearity,
                name="%shidden_%d" % (prefix, i),
                W=hidden_W_init,
                b=all_b_init,
            )

        l_out = NoisyDenseLayer(
            x,
            num_units=output_dim,
            rng=rng,
            factorized=factorized,
            common_noise=common_noise,
            use_mu_init=use_mu_init,
            sigma_0=sigma_0,
            nonlinearity=LN.linear,
            name="%soutput_q" % (prefix,),
            W=output_W_init,
            b=all_b_init,
        )

        # out_b_init = None if shared_last_bias else all_b_init
        # l_out = L.DenseLayer(
        #     x,
        #     num_units=output_dim,
        #     nonlinearity=LN.linear,
        #     name="%soutput_q" % (prefix,),
        #     W=output_W_init,
        #     b=out_b_init,
        # )
        # if shared_last_bias:
        #     l_out = L.BiasLayer(
        #         l_out,
        #         b=all_b_init,
        #         shared_axes=(0, 1),  # makes a single scalar value
        #         name="%sshared_output_bias" % (prefix,),
        #     )

        self._l_in = l_in
        self._l_out = l_out
        self._input_var = input_var
        self._l_conv_out = l_conv_out

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def output_layers(self):
        return [self._l_out]

    @property
    def input_var(self):
        return self._input_var

    @property
    def conv_output_layer(self):
        return self._l_conv_out

    def get_params(self, **tags):
        return L.get_all_params(self._l_out, **tags)

