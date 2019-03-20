
import sys

from accel_rl.runners.accel_rl import AccelRLEval
from accel_rl.algos.dqn.dqn import DQN
from accel_rl.policies.dqn.atari_dqn_policy import AtariDqnPolicy
from accel_rl.policies.atari_cnn_specs import cnn_specs
from accel_rl.sampler.act_server.alternating.overlap.sampler_with_eval import \
    AAOEvalSampler
from accel_rl.envs.atari_env import AtariEnv

from accel_rl.scripts.launching.affinities import get_affinities
from accel_rl.util.logging import logger_context


def build_and_run(run_slot_affinities_code, log_dir, game, game_run_ID,
        learning_rate, batch_size):
    affinities = get_affinities(run_slot_affinities_code)
    assert isinstance(affinities, dict)  # should not be a list

    learning_rate = float(learning_rate)  # (convert from string input)
    batch_size = int(batch_size)

    training_intensity = 8  # (Atari papers always keep at 8)
    sampling_horizon = 4
    min_n_envs = batch_size // (training_intensity * sampling_horizon)
    n_envs = max(32, min_n_envs)

    env_args = dict(
        game=game,
        clip_reward=True,
        max_start_noops=30,
        episodic_lives=True,
    )

    n_sim_cores = len(affinities["sim_cores"])
    assert n_envs % (n_sim_cores * 2) == 0
    envs_per = n_envs // (n_sim_cores * 2)

    sampler = AAOEvalSampler(
        EnvCls=AtariEnv,
        env_args=env_args,
        horizon=sampling_horizon,
        eval_steps=int(125e3),
        n_parallel=len(affinities["sim_cores"]),
        envs_per=envs_per,
        eval_envs_per=1,
        mid_batch_reset=True,
        max_path_length=int(27e3),
    )
    assert sampler.total_n_envs == n_envs

    algo = DQN(
        batch_size=batch_size,
        training_intensity=training_intensity,
        optimizer_args=dict(learning_rate=learning_rate),
    )

    cnn_spec = 1
    cnn_args = cnn_specs[cnn_spec]
    policy = AtariDqnPolicy(**cnn_args)
    print("cnn_args: ", cnn_args)

    runner = AccelRLEval(
        algo=algo,
        policy=policy,
        sampler=sampler,
        n_steps=10e6,
        eval_interval_steps=1e6,
        affinities=affinities,
        seed=None,
        use_gpu=True,
    )

    log_params = dict(
        exp='basic_dqn',
        cnn_spec=cnn_spec,
        n_envs=n_envs,
        sampling_horizon=sampling_horizon,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_intensity=training_intensity,
    )
    with logger_context(log_dir, game, game_run_ID, log_params):
        runner.train()


if __name__ == "__main__":
    build_and_run(*sys.argv[1:])
