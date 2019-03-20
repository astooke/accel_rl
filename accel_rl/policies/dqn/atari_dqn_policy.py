
import numpy as np
import theano.tensor as T
import lasagne.layers as L
import lasagne.nonlinearities as LN

from rllab.misc import ext

from accel_rl.policies.base import BaseNNPolicy
from accel_rl.util.quick_args import save_args, retrieve_args
from accel_rl.spaces.discrete import Discrete
from accel_rl.policies.dqn.networks.dqn_cnn import DqnCnn
from accel_rl.policies.util import shorten_param_name


class AtariDqnPolicy(BaseNNPolicy):

    def __init__(
            self,
            conv_filters, conv_filter_sizes, conv_strides, conv_pads,
            hidden_sizes=[],
            hidden_nonlinearity=LN.rectify,
            pixel_scale=255.,
            epsilon=1,
            dueling=False,
            shared_last_bias=False,
            **kwargs
            ):
        save_args(vars())
        super().__init__(**kwargs)

    def initialize(self, env_spec, **kwargs):
        # Wait to do this until GPU is initialized
        assert isinstance(env_spec.action_space, Discrete)
        s = retrieve_args(self)

        network_args = dict(
            input_shape=env_spec.observation_space.shape,
            output_dim=env_spec.action_space.n,
            conv_filters=s.conv_filters,
            conv_filter_sizes=s.conv_filter_sizes,
            conv_strides=s.conv_strides,
            conv_pads=s.conv_pads,
            hidden_sizes=s.hidden_sizes,
            hidden_nonlinearity=s.hidden_nonlinearity,
            pixel_scale=s.pixel_scale,
            dueling=s.dueling,
            shared_last_bias=s.shared_last_bias,
        )
        policy_network = DqnCnn(name="policy", **network_args)
        target_network = DqnCnn(name="target", **network_args)

        self._l_obs = policy_network.input_layer
        self._l_target_obs = target_network.input_layer
        self._policy_network = policy_network
        self._target_network = target_network

        input_var = policy_network.input_layer.input_var
        q = L.get_output(policy_network.output_layer)
        self._f_q = ext.compile_function([input_var], q)
        greedy_action = T.argmax(q, axis=1)
        self._f_a = ext.compile_function([input_var], greedy_action)

        target_input_var = target_network.input_layer.input_var
        target_q = L.get_output(target_network.output_layer)
        self._f_target_q = ext.compile_function([target_input_var], target_q)

        policy_params = policy_network.get_params(trainable=True)
        target_params = target_network.get_params(trainable=True)
        updates = [(t, p) for p, t in zip(policy_params, target_params)]
        self._f_update_target = ext.compile_function(inputs=[], updates=updates)
        self._f_update_target()

        super().initialize(env_spec, network=policy_network, **kwargs)
        self._epsilon = s.epsilon
        self.param_short_names = \
            [shorten_param_name(p.name) for p in policy_params]

    def actions_sym(self, obs_var=None):
        inputs = {} if obs_var is None else {self._l_obs: obs_var}
        qs = L.get_output(self._policy_network.output_layer, inputs)
        return T.argmax(qs, axis=1)

    def target_q_at_a_sym(self, obs_var, act_var):
        inputs = {self._l_target_obs: obs_var}
        qs = L.get_output(self._target_network.output_layer, inputs)
        return qs[T.arange(T.shape(act_var)[0]), act_var]

    def target_max_q_sym(self, obs_var):
        inputs = {self._l_target_obs: obs_var}
        qs = L.get_output(self._target_network.output_layer, inputs)
        return T.max(qs, axis=1)

    def q_sym(self, obs_var):
        inputs = {self._l_obs: obs_var}
        return L.get_output(self._policy_network.output_layer, inputs)

    def q_at_a_sym(self, obs_var, act_var):
        inputs = {self._l_obs: obs_var}
        qs = L.get_output(self._policy_network.output_layer, inputs)
        return qs[T.arange(T.shape(act_var)[0]), act_var]

    def target_q_sym(self, obs_var):
        inputs = {self._l_target_obs: obs_var}
        return L.get_output(self._target_network.output_layer, inputs)

    def q(self, observations):
        return self._f_q(observations)

    def target_q(self, observations):
        return self._f_target_q(observations)

    def update_target(self):
        self._f_update_target()

    def get_action(self, observation, deterministic=False):
        if deterministic or (np.random.rand() > self._epsilon):
            action = self._f_a(observation[None])[0]
        else:
            action = self.action_space.sample()
        return action, dict()  # (dict to be consistent with agent_info)

    def get_actions(self, observations, deterministic=False):
        actions = self._f_a(observations)
        if not deterministic:
            random_idx = np.where(np.random.rand(len(actions)) < self._epsilon)[0]
            actions[random_idx] = self.action_space.sample_n(len(random_idx))
        return actions, dict()  # (dict to be consistent with agent_infos)

    def get_epsilon(self):
        return self._epsilon

    def set_epsilon(self, value):
        self._epsilon = value
