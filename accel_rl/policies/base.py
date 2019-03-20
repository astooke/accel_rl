
import numpy as np

from rllab.core.lasagne_powered import LasagnePowered
from rllab.policies.base import StochasticPolicy


class BaseNNPolicy(StochasticPolicy, LasagnePowered):

    def __init__(self, initial_param_values=None):
        self.initial_param_values = initial_param_values

    def initialize(self, env_spec, network):
        """Call this in subclass's initialize, after building the network"""
        StochasticPolicy.__init__(self, env_spec)
        LasagnePowered.__init__(self, network.output_layers)
        if self.initial_param_values is not None:
            print("\nSetting initial NN param values")
            print("param values before loading initial values: ",
                self.get_param_values()[:5])
            self.set_param_values(self.initial_param_values)
            print("param values after loading: ",
                self.get_param_values()[:5])

    def reset(self, n_batch):
        pass

    def reset_one(self, idx):
        pass


class BaseRecurrentPolicy(BaseNNPolicy):

    """
    All recurrent state is held on the CPU.
    Accomodates alternating sampler by holding pair of states.
    """

    def initialize(self, env_spec, network, alternating_sampler=False):
        super().initialize(env_spec, network)
        self.alternating_sampler = alternating_sampler
        self._prev_hiddens = None
        if alternating_sampler:
            self._prev_hiddens_pair = None
            self._j = 0

    @property
    def recurrent(self):
        return True

    def reset(self, n_batch):
        """ Must set the inference batch size here using n_batch,
        (no effect on training)."""
        full_len = n_batch * (1 + int(self.alternating_sampler))
        hiddens = list()
        for p in self._network.hid_init_params:
            val = p.get_value()
            hiddens.append(np.tile(val, (full_len, 1)))
        if self.alternating_sampler:
            hiddens_0, hiddens_1 = list(), list()
            for h in hiddens:
                hiddens_0.append(h[:n_batch])
                hiddens_1.append(h[n_batch:])
            self._prev_hiddens_pair = [hiddens_0, hiddens_1]
            self._j = 0
        else:
            self._prev_hiddens = hiddens

    def reset_one(self, idx):
        """Use when an environment instance resets"""
        for hid, p in zip(self.get_prev_hiddens(), self.hid_init_params):
            hid[idx] = p.get_value()

    def advance_hiddens(self, new_hiddens):
        if self.alternating_sampler:
            self._prev_hiddens_pair[self._j] = new_hiddens
            self._j ^= 1
        else:
            self._prev_hiddens = new_hiddens

    def get_prev_hiddens(self):
        """If alternating, return only one of the pair"""
        if self.alternating_sampler:
            return self._prev_hiddens_pair[self._j]
        else:
            return self._prev_hiddens

    def get_state_info(self):
        """If alternating, return both together as one"""
        if self.alternating_sampler:
            return [np.concatenate([h0, h1]) for h0, h1 in zip(*self._prev_hiddens_pair)]
        else:
            return self._prev_hiddens


class BaseNetwork(object):

    @property
    def input_layer(self):
        pass

    @property
    def output_layers(self):
        return []
