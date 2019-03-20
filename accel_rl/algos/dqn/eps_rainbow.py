

from lasagne.updates import adam
from accel_rl.algos.dqn.cat_dqn import CategoricalDQN


class EpsRainbow(CategoricalDQN):
    """
    Rainbow minus NoisyNets (i.e. using epsilon-greedy exploration)
    (NoisyNets are slow)
    """

    def __init__(
            self,
            reward_horizon=3,
            double_dqn=True,
            dueling_dqn=True,
            prioritized_replay=True,
            target_update_steps=int(8e3),  # (32,000 frames)
            min_steps_learn=int(2e4),  # (80,000 frames)
            **kwargs
            ):
        super().__init__(
            reward_horizon=reward_horizon,
            double_dqn=double_dqn,
            dueling_dqn=dueling_dqn,
            prioritized_replay=prioritized_replay,
            target_update_steps=target_update_steps,
            min_steps_learn=min_steps_learn,
            **kwargs
        )

    def _get_default_sub_args(self):
        opt_args = dict(
            learning_rate=6.25e-5,
            update_method=adam,
            grad_norm_clip=10,
            update_method_args=dict(epsilon=0.005 / self.batch_size),
            scale_conv_grads=self.dueling_dqn,
        )
        eps_greedy_args = dict(
            initial=1.,
            final=0.01,
            eval=0.001,
            anneal_steps=int(62.5e3),  # (250,000 frames)
        )
        priority_args = dict(
            alpha=0.5,
            beta_initial=0.4,
            beta_final=1.,
            beta_anneal_steps=50e6,
            default_priority=1.,
        )
        return opt_args, eps_greedy_args, priority_args
