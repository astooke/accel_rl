
"""UNDER CONSTRUCTION"""

import numpy as np
import theano.tensor as T
import lasagne.layers as L
import lasagne.nonlinearities as LN

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from rllab.misc import ext

from accel_rl.policies.base import BaseNNPolicy
from accel_rl.util.quick_args import save_args, retrieve_args
from accel_rl.spaces.discrete import Discrete
# from accel_rl.policies.networks.atari_cnn_dqn import AtariCnn
from accel_rl.policies.dqn.networks.atari_cnn_noisy_net_dqn import AtariNoisyNetCnn


class AtariNoisyNetDqnPolicy(BaseNNPolicy):

    def __init__(
            self,
            conv_filters, conv_filter_sizes, conv_strides, conv_pads,
            hidden_sizes=[],
            hidden_nonlinearity=LN.rectify,
            pixel_scale=255.,
            epsilon=1,
            # shared_last_bias=False,
            factorized=True,
            common_noise=False,
            sigma_0=0.4,
            use_mu_init=True,
            **kwargs
            ):
        save_args(vars())
        super().__init__(**kwargs)

    def initialize(self, env_spec, **kwargs):
        # Wait to do this until GPU is initialized
        assert isinstance(env_spec.action_space, Discrete)
        s = retrieve_args(self)

        rng = RandomStreams(np.random.randint(1, 123456))

        policy_network = AtariNoisyNetCnn(
            input_shape=env_spec.observation_space.shape,
            output_dim=env_spec.action_space.n,
            conv_filters=s.conv_filters,
            conv_filter_sizes=s.conv_filter_sizes,
            conv_strides=s.conv_strides,
            conv_pads=s.conv_pads,
            hidden_sizes=s.hidden_sizes,
            hidden_nonlinearity=s.hidden_nonlinearity,
            pixel_scale=s.pixel_scale,
            rng=rng,
            factorized=s.factorized,
            common_noise=s.common_noise,
            sigma_0=s.sigma_0,
            use_mu_init=s.use_mu_init,
            # shared_last_bias=s.shared_last_bias,
            name="policy_network",
        )

        target_network = AtariNoisyNetCnn(
            input_shape=env_spec.observation_space.shape,
            output_dim=env_spec.action_space.n,
            conv_filters=s.conv_filters,
            conv_filter_sizes=s.conv_filter_sizes,
            conv_strides=s.conv_strides,
            conv_pads=s.conv_pads,
            hidden_sizes=s.hidden_sizes,
            hidden_nonlinearity=s.hidden_nonlinearity,
            pixel_scale=s.pixel_scale,
            rng=rng,
            factorized=s.factorized,
            common_noise=s.common_noise,
            sigma_0=s.sigma_0,
            use_mu_init=s.use_mu_init,
            # shared_last_bias=s.shared_last_bias,
            name="target_network",
        )

        self._l_obs = policy_network.input_layer
        self._l_target_obs = target_network.input_layer
        self._policy_network = policy_network
        self._target_network = target_network

        self._rng = rng  # NOTE: later, can use this to control noise sampling

        input_var = policy_network.input_layer.input_var
        q = L.get_output(policy_network.output_layer)
        self._f_q = ext.compile_function([input_var], q)
        a = T.argmax(q, axis=1)
        self._f_a = ext.compile_function([input_var], a)
        target_input_var = target_network.input_layer.input_var
        target_q = L.get_output(target_network.output_layer)
        self._f_target_q = ext.compile_function([target_input_var], target_q)

        policy_params = policy_network.get_params(trainable=True)
        target_params = target_network.get_params(trainable=True)
        updates = [(t, p) for p, t in zip(policy_params, target_params)]
        self._f_update_target = ext.compile_function(inputs=[], updates=updates)
        self._f_update_target()

        super().initialize(env_spec, network=policy_network, **kwargs)
        self._epsilon = 0.

    def q_sym(self, obs_var):
        inputs = {self._l_obs: obs_var}
        return L.get_output(self._policy_network.output_layer, inputs)

    def q_a_sym(self, obs_var, act_var):
        inputs = {self._l_obs: obs_var}
        qs = L.get_output(self._policy_network.output_layer, inputs)
        return qs[T.arange(T.shape(act_var)[0]), act_var]

    def target_q_sym(self, obs_var):
        inputs = {self._l_target_obs: obs_var}
        return L.get_output(self._target_network.output_layer, inputs)

    def target_q_a_sym(self, obs_var, act_var):
        inputs = {self._l_target_obs: obs_var}
        qs = L.get_output(self._target_network.output_layer, inputs)
        return qs[T.arange(T.shape(act_var)[0]), act_var]

    def q(self, observations):
        return self._f_q(observations)

    def target_q(self, observations):
        return self._f_target_q(observations)

    def update_target(self):
        self._f_update_target()

    def get_action(self, observation):
        actions = self._f_a(observation[None])
        return actions[0], dict()

    def get_actions(self, observations):
        actions = self._f_a(observations)
        return actions, dict()

    def get_epsilon(self):
        return 0.

    def set_epsilon(self, value):
        pass
