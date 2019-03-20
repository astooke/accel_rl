
import numpy as np
import lasagne.layers as L
import lasagne.nonlinearities as LN

from rllab.misc import ext

from accel_rl.policies.base import BaseRecurrentPolicy
from accel_rl.distributions.categorical import Categorical
from accel_rl.spaces.discrete import Discrete
from accel_rl.util.quick_args import save_args, retrieve_args
from accel_rl.policies.pg.networks.pg_cnn_gru import PgCnnGru
from accel_rl.policies.util import shorten_param_name


class AtariGruPolicy(BaseRecurrentPolicy):

    def __init__(
            self,
            conv_filters, conv_filter_sizes, conv_strides, conv_pads,
            conv_nonlinearity=LN.rectify,
            hidden_sizes=[32, 32],
            hidden_nonlinearity=LN.tanh,
            output_pi_nonlinearity=LN.softmax,
            pixel_scale=255.,
            alternating_sampler=False,
            **kwargs
            ):
        """
        The policy consists of several convolution layers followed by recurrent
        layers and softmax
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

    def initialize(self, env_spec, alternating_sampler=False):
        # Wait to do this until GPU is initialized.
        assert isinstance(env_spec.action_space, Discrete)
        s = retrieve_args(self)

        network = PgCnnGru(
            input_shape=env_spec.observation_space.shape,
            output_dim=env_spec.action_space.n,
            conv_filters=s.conv_filters,
            conv_filter_sizes=s.conv_filter_sizes,
            conv_strides=s.conv_strides,
            conv_pads=s.conv_pads,
            conv_nonlinearity=s.conv_nonlinearity,
            hidden_sizes=s.hidden_sizes,
            hidden_nonlinearity=s.hidden_nonlinearity,
            output_pi_nonlinearity=s.output_pi_nonlinearity,
            pixel_scale=s.pixel_scale,
            name="atari_cnn_gru",
        )

        self._l_obs = network.input_layer

        input_var = network.input_layer.input_var
        prev_hidden_vars = [lay.input_var for lay in network.prev_hidden_layers]
        prob, value = L.get_output(network.output_layers,
            step_or_train="step")
        hidden_vars = L.get_output(network.recurrent_layers,
            step_or_train="step")

        self._f_act = ext.compile_function(
            inputs=[input_var] + prev_hidden_vars,
            outputs=[prob, value] + hidden_vars,
        )
        self._f_prob_value = ext.compile_function(
            inputs=[input_var] + prev_hidden_vars,
            outputs=[prob, value],
        )
        self._f_prob = ext.compile_function(
            inputs=[input_var] + prev_hidden_vars,
            outputs=prob,
        )
        self._f_value = ext.compile_function(
            inputs=[input_var] + prev_hidden_vars,
            outputs=value,
        )
        self._f_hidden = ext.compile_function(
            inputs=[input_var] + prev_hidden_vars,
            outputs=hidden_vars,
        )

        self._dist = Categorical(env_spec.action_space.n)

        super().initialize(env_spec, network, alternating_sampler)
        self.param_short_names = \
            [shorten_param_name(p.name) for p in network.get_params(trainable=True)]
        self._network = network
        self._hprev_keys = ["hprev_{}".format(i)
            for i in range(len(network.recurrent_layers))]
        self.hid_init_params = self._network.hid_init_params

    @property
    def vectorized(self):
        return True

    def dist_info_sym(self, obs_var, state_info_vars=None):
        inputs = {self._l_obs: obs_var}
        prev_vars = [state_info_vars[k] for k in self._hprev_keys]
        for k, v in zip(self._network._ls_prev_hidden, prev_vars):
            inputs[k] = v
        probs = L.get_output(self._network.output_layer_pi, inputs,
            step_or_train="train")
        return dict(prob=probs)

    def value_sym(self, obs_var, state_info_vars=None):
        inputs = {self._l_obs: obs_var}
        prev_vars = [state_info_vars[k] for k in self._hprev_keys]
        for k, v in zip(self._network._ls_prev_hidden, prev_vars):
            inputs[k] = v
        return L.get_output(self._network.output_layer_v, inputs,
            step_or_train="train")

    def dist_info(self, observations, state_infos=None):
        """Do NOT update the internal state in these methods"""
        state_infos = self.get_state_info() if state_infos is None else state_infos
        probs = self._f_prob(observations, *state_infos)
        return dict(prob=probs)

    def value(self, observations, state_infos=None):
        state_infos = self.get_state_info() if state_infos is None else state_infos
        return self._f_value(observations, *state_infos)

    def dist_info_value(self, observations, state_infos=None):
        state_infos = self.get_state_info() if state_infos is None else state_infos
        probs, values = self._f_prob_value(observations, *state_infos)
        return dict(prob=probs, value=values)

    def next_hiddens(self, observations, state_infos=None):
        state_infos = self.get_state_info() if state_infos is None else state_infos
        hiddens = self._f_hidden(observations, *state_infos)
        return {k: h for k, h in zip(self._hprev_keys, hiddens)}

    def get_action(self, observation, deterministic=False):
        """DO update the internal state when getting action(s)"""
        # NOTE: must reset previously with n_batch = 1.
        prev_hiddens = self.get_prev_hiddens()
        probs, values, *hiddens = self._f_act(observation[None], *prev_hiddens)
        prob, value = (probs[0], values[0])
        if deterministic:
            action = np.argmax(prob)
        else:
            action = self.action_space.weighted_sample(prob)
        agent_info = {k: h[0] for k, h in zip(self._hprev_keys, prev_hiddens)}
        agent_info.update(dict(prob=prob, value=value))
        self.advance_hiddens(hiddens)
        return action, agent_info

    def get_actions(self, observations):
        # NOTE: must reset previously with n_batch = len(observations)
        prev_hiddens = self.get_prev_hiddens()
        probs, values, *hiddens = self._f_act(observations, *prev_hiddens)
        actions = self.action_space.weighted_sample_n(probs)
        agent_info = {k: h for k, h in zip(self._hprev_keys, prev_hiddens)}
        agent_info.update(dict(prob=probs, value=values))
        self.advance_hiddens(hiddens)
        return actions, agent_info

    @property
    def distribution(self):
        return self._dist

    @property
    def state_info_keys(self):
        return self._hprev_keys
