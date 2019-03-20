

from accel_rl.algos.dqn import DQN
from accel_rl.optimizers.single.dqn_stats_optimizer import DqnStatsOptimizer
from accel_rl.optimizers.update_methods_stats import rmsprop


class DQNStats(DQN):

    def __init__(
            self,
            OptimizerCls=DqnStatsOptimizer,
            layerwise_stats=False,
            **kwargs):
        super.__init__(OptimizerCls=OptimizerCls, **kwargs)
        self._layerwise_stats = layerwise_stats

    def _get_default_sub_args(self):
        opt_args, eps_greedy_args, priority_args = super()._get_default_sub_args()
        opt_args["update_method"] = rmsprop
        return opt_args, eps_greedy_args, priority_args

    def optimize_policy(self, itr, samples_data):
        self.replay_buffer.append_data(samples_data)
        if itr < self._min_itr_learn:
            return None, dict()
        priorities = list()
        losses = list()
        grad_norms = list()
        step_norms = list()
        if self.layerwise_stats:
            param_grad_norms_list = list()
            param_step_norms_list = list()
        for _ in range(self._updates_per_optimize):
            opt_minibatch = self.replay_buffer.sample_batch(self.batch_size)
            opt_outputs = self.optimizer.optimize(opt_minibatch)
            if self.layerwise_stats:
                pr, lo, gn, sn, pgn, psn = opt_outputs
                param_grad_norms_list.append(pgn)
                param_step_norms_list.append(psn)
            else:
                pr, lo, gn, sn = opt_outputs
            if self.prioritized_replay:
                self.replay_buffer.update_batch_priorities(pr)
            priorities.extend(pr[::8])  # (downsample for stats)
            losses.append(lo)
            grad_norms.append(gn)
            step_norms.append(sn)
        if itr % self._target_update_itr == 0:
            self.policy.update_target()
        self.update_epsilon(itr)
        if self.prioritized_replay:
            self.update_priority_beta(itr)
        opt_info = dict(
            Priority=priorities,
            Loss=losses,
            GradNorm=grad_norms,
            StepNorm=step_norms,
        )
        if self.layerwise_stats:
            opt_info.update(self._reorg_layerwise_stat(
                "_GradNorm", param_grad_norms_list))
            opt_info.update(self._reorg_layerwise_stat(
                "_StepNorm", param_step_norms_list))

        return opt_minibatch, opt_info

    def _reorg_layerwise_stat(self, stat_name, layerwise_steps_list):
        keys = [name + stat_name for name in self.policy.param_short_names]
        layerwise_dict = {k: list() for k in keys}
        for step in layerwise_steps_list:
            for key, layer_stat in zip(keys, step):
                layerwise_dict[key].append(layer_stat)
        return layerwise_dict

    @property
    def opt_info_keys(self):
        keys = ["Priority", "Loss", "GradNorm", "StepNorm"]
        if self.layerwise_stats:
            for name in self.policy.param_short_names:
                keys += [name + "_GradNorm", name + "_StepNorm"]
        return keys
