
import numpy as np
import lasagne.layers as L
import lasagne.nonlinearities as LN

from rllab.misc import ext
from accel_rl.policies.base import BaseNNPolicy
from accel_rl.distributions.categorical import Categorical
from accel_rl.spaces.discrete import Discrete
from accel_rl.util.quick_args import save_args, retrieve_args
from accel_rl.policies.pg.networks.pg_cnn import PgCnn
from accel_rl.policies.util import shorten_param_name


class AtariCnnPolicy(BaseNNPolicy):

    def __init__(
            self,
            conv_filters, conv_filter_sizes, conv_strides, conv_pads,
            hidden_sizes=[],
            hidden_nonlinearity=LN.rectify,
            output_pi_nonlinearity=LN.softmax,
            pixel_scale=255.,
            **kwargs
            ):
        """
        The policy consists of several convolution layers followed by fc layers and softmax
        :param env_spec: A spec for the mdp.
        :param conv_filters, conv_filter_sizes, conv_strides, conv_pads: specify the convolutional layers. See rllab.core.network.ConvNetwork for details.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param prob_network: manually specified network for this policy, other network params
        are ignored
        :param feature_layer_index: index of the feature layer. Default -2 means the last layer before fc-softmax
        :param eps: mixture weight on uniform distribution; useful to force exploration
        :return:
        """
        save_args(vars())
        super().__init__(**kwargs)

    def initialize(self, env_spec, **kwargs):
        # Wait to do this until GPU is initialized.
        assert isinstance(env_spec.action_space, Discrete)
        s = retrieve_args(self)

        network = PgCnn(
            input_shape=env_spec.observation_space.shape,
            output_dim=env_spec.action_space.n,
            conv_filters=s.conv_filters,
            conv_filter_sizes=s.conv_filter_sizes,
            conv_strides=s.conv_strides,
            conv_pads=s.conv_pads,
            hidden_sizes=s.hidden_sizes,
            hidden_nonlinearity=s.hidden_nonlinearity,
            output_pi_nonlinearity=s.output_pi_nonlinearity,
            pixel_scale=s.pixel_scale,
            name="atari_cnn",
        )

        self._l_obs = network.input_layer
        input_var = network.input_layer.input_var

        prob, value = L.get_output(network.output_layers)

        self._f_prob = ext.compile_function([input_var], prob)
        self._f_value = ext.compile_function([input_var], value)
        self._f_prob_value = ext.compile_function([input_var], [prob, value])

        self._dist = Categorical(env_spec.action_space.n)
        self._network = network
        super().initialize(env_spec, network=network, **kwargs)
        self.param_short_names = \
            [shorten_param_name(p.name) for p in network.get_params(trainable=True)]

    @property
    def vectorized(self):
        return True

    def dist_info_sym(self, obs_var, state_info_vars=None):
        inputs = {self._l_obs: obs_var}
        probs = L.get_output(self._network.output_layer_pi, inputs)
        return dict(prob=probs)

    def value_sym(self, obs_var, state_info_vars=None):
        inputs = {self._l_obs: obs_var}
        return L.get_output(self._network.output_layer_v, inputs)

    def dist_info(self, observations, state_infos=None):
        probs = self._f_prob(observations)
        return dict(prob=probs)

    def value(self, observations, state_infos=None):
        return self._f_value(observations)

    def dist_info_value(self, observations, state_infos=None):
        probs, values = self._f_prob_value(observations)
        return dict(prob=probs, value=values)

    def get_action(self, observation, deterministic=False):
        probs, values = self._f_prob_value(observation[None])
        prob, value = (probs[0], values[0])
        if deterministic:
            action = np.argmax(prob)
        else:
            action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob, value=value)

    def get_actions(self, observations):
        probs, values = self._f_prob_value(observations)
        actions = self.action_space.weighted_sample_n(probs)
        return actions, dict(prob=probs, value=values)

    @property
    def distribution(self):
        return self._dist

    @property
    def state_info_keys(self):
        return []
