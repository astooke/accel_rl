
import numpy as np
import theano
import theano.tensor as T
from lasagne.updates import adam

from accel_rl.algos.dqn.dqn import DQN


class CategoricalDQN(DQN):

    def __init__(self, V_min=-10, V_max=10, **kwargs):
        self.V_min = V_min
        self.V_max = V_max
        super().__init__(**kwargs)

    def _get_default_sub_args(self):
        opt_args = dict(
            learning_rate=2.5e-4,
            update_method=adam,
            grad_norm_clip=10 if self.dueling_dqn else None,
            update_method_args=dict(epsilon=0.01 / self.batch_size),
            scale_conv_grads=self.dueling_dqn,
        )
        eps_greedy_args = dict(  # original DQN values
            initial=1.,
            final=0.01,
            eval=0.001,
            anneal_steps=int(1e6),
        )
        priority_args = dict(
            alpha=0.6,
            beta_initial=0.4,
            beta_final=1.,
            beta_anneal_steps=50e6,
            default_priority=1.,
        )
        return opt_args, eps_greedy_args, priority_args

    def build_loss(self, env_spec, policy):
        obs = env_spec.observation_space.new_tensor_variable('obs', extra_dims=1)
        next_obs = env_spec.observation_space.new_tensor_variable('next_obs', extra_dims=1)
        act = env_spec.action_space.new_tensor_variable('act', extra_dims=1)
        ret = T.vector('disc_n_return')
        term = T.bvector('terminal')
        if self.prioritized_replay:
            isw = T.vector('importance_sample_weights')

        z_np = np.linspace(self.V_min, self.V_max, policy.n_atoms,
            dtype=theano.config.floatX)
        z = theano.shared(z_np)
        z_contracted = theano.shared((self.discount ** self.reward_horizon) * z_np)
        policy.incorporate_z(z)  # (policy sets n_atoms, but algo sets vmin,vmax)
        delta_z = (self.V_max - self.V_min) / (policy.n_atoms - 1)

        # Yeah this is difficult to read and know if it's right.
        # (tested it vs numpy loop and numpy vectorized form in another script)
        z_contracted_bc = z_contracted.dimshuffle('x', 0)  # (bc: broadcast)
        z_cntrct_term = (1 - term.dimshuffle(0, 'x')) * z_contracted_bc
        # z_cntrct_term is 2D tensor, with contracted z-values repeated for
        # each data point (each row), and zero'd wherever terminal is True
        ret_bc = ret.dimshuffle(0, 'x')
        z_next = T.clip(ret_bc + z_cntrct_term, self.V_min, self.V_max)
        # each row (data entry) in z_next had all z_values shifted by
        # corresponding return
        # must compare every pair of base z atom with next z atom
        z_next_bc = z_next.dimshuffle(0, 1, 'x')
        z_bc = z.dimshuffle('x', 'x', 0)
        abs_diff_on_delta = abs(z_next_bc - z_bc) / delta_z
        projection_coeffs = T.clip(1 - abs_diff_on_delta, 0, 1)  # (mostly 0's)
        # projection coefficients is a 3-D tensor.
        # dim-0: independent data entries (gets scanned/looped over in batched_dot)
        # dim-1: corresponds to z_next atoms (gets summed over in batched_dot)
        # dim-2: corresponds to base z atoms (becomes dim-1 after batched_dot)

        if self.double_dqn:
            next_act = policy.actions_sym(next_obs)
            next_Z = policy.target_Z_at_a_sym(next_obs, next_act)
        else:
            next_Z = policy.target_max_Z_sym(next_obs)
        # lower case z refers to the domain of atoms,
        # capital Z refers to the probabilities for given state and action
        # projected_next_Z = T.batched_dot(next_Z, projection_coeffs)
        # NOTE: use of batched_dot somehow breaks the gradient (Theano 0.9);
        # so, do the broadcasting and summing manually (until Theano 1.0)
        next_Z_bc = T.shape_padright(next_Z)
        next_Z_x_coeff = projection_coeffs * next_Z_bc
        projected_next_Z = next_Z_x_coeff.sum(axis=1)

        predicted_Z = policy.Z_at_a_sym(obs, act)
        predicted_Z = T.clip(predicted_Z, 1e-6, 1)  # (NaN-guard)
        losses = -T.sum(projected_next_Z * T.log(predicted_Z), axis=1)  # CrossEnt

        if self.prioritized_replay:
            losses = isw * losses
        loss = T.mean(losses)

        projected_next_Z = T.clip(projected_next_Z, 1e-6, 1)  # (NaN-guard)
        KL_divs = T.sum(
            projected_next_Z * T.log(projected_next_Z / predicted_Z),
            axis=1,
        )
        KL_divs = T.clip(KL_divs, 1e-6, 1e6)  # avoid < 0 from NaN-guard

        input_list = [obs, next_obs, act, ret, term]
        if self.prioritized_replay:
            input_list.append(isw)

        return input_list, loss, KL_divs
