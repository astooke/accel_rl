
import theano.tensor as T
import lasagne.layers as L
import lasagne.init as LI
import lasagne.nonlinearities as LN

from accel_rl.policies.base import BaseNetwork
from accel_rl.policies.layers import NormCInit, ScalarFixedScaleLayer
from accel_rl.policies.dqn.layers.dueling_merge_layer import DuelingMergeLayer


class CatDqnCnn(BaseNetwork):

    def __init__(
            self,
            input_shape,
            n_actions,
            n_atoms,
            hidden_sizes,
            conv_filters, conv_filter_sizes, conv_strides, conv_pads,
            hidden_nonlinearity=LN.rectify,
            conv_W_init=LI.GlorotUniform(),
            hidden_W_init=NormCInit(1.0),
            output_W_init=NormCInit(0.01),  # (like in OpenAI baselines)
            all_b_init=LI.Constant(0.),
            pixel_scale=255.,
            input_var=None,
            dueling=False,
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
        x = L.DenseLayer(
            x,
            num_units=n_actions * n_atoms,
            nonlinearity=LN.linear,
            name="%saction_atoms" % (prefix,),
            W=output_W_init,
            b=all_b_init,
        )
        if dueling:
            v = l_conv_out
            for i, hidden_size in enumerate(hidden_sizes):
                v = L.DenseLayer(
                    v,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="%shidden_Val_%d" % (prefix, i),
                    W=hidden_W_init,
                    b=all_b_init,
                )
            v = L.DenseLayer(
                v,
                num_units=n_atoms,
                nonlinearity=LN.linear,
                name="%sVal" % (prefix,),
                W=output_W_init,
                b=all_b_init,
            )
            v = L.ReshapeLayer(v, shape=(-1, 1, n_atoms))
            x = L.ReshapeLayer(x, shape=(-1, n_actions, n_atoms))
            x = DuelingMergeLayer(incomings=(v, x), name="DuelMerge")
        x = L.ReshapeLayer(x, shape=(-1, n_atoms))
        x = L.NonlinearityLayer(x, LN.softmax)  # softmax accepts only 2D tensor
        l_out = L.ReshapeLayer(x, shape=(-1, n_actions, n_atoms))

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

