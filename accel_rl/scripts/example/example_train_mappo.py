
import sys

from accel_rl.runners.multigpu_rl import AccelRLAsync
from accel_rl.algos.pg.ppo import mAPPO
from accel_rl.policies.pg.atari_cnn_policy import AtariCnnPolicy
from accel_rl.policies.atari_cnn_specs import cnn_specs
from accel_rl.sampler.act_server.alternating.overlap.sampler import \
    ActsrvAltOvrlpSampler
from accel_rl.envs.atari_env import AtariEnv

from accel_rl.scripts.launching.affinities import get_affinities
from accel_rl.util.logging import logger_context


def build_and_run(run_slot_affinities_code, log_dir, game, game_run_ID,
        learning_rate, n_envs):
    affinities = get_affinities(run_slot_affinities_code)
    assert isinstance(affinities, list)  # should be a list

    learning_rate = float(learning_rate)
    n_envs = int(n_envs)  # (this is per learner)

    env_args = dict(
        game=game,
        clip_reward=True,
        max_start_noops=30,
        episodic_lives=True,
    )

    n_sim_cores = len(affinities[0]["sim_cores"])
    assert n_envs % (n_sim_cores * 2) == 0
    envs_per = n_envs // (n_sim_cores * 2)

    sampler = ActsrvAltOvrlpSampler(
        EnvCls=AtariEnv,
        env_args=env_args,
        horizon=5,
        n_parallel=len(affinities["sim_cores"]),
        envs_per=envs_per,
        mid_batch_reset=True,
        max_path_length=int(27e3),
    )
    assert sampler.total_n_envs == n_envs

    algo = mAPPO(
        discount=0.99,
        gae_lambda=1,
        optimizer_args=dict(
            learning_rate=learning_rate,
            n_update_chunks=3,
        ),
    )

    cnn_spec = 0
    cnn_args = cnn_specs[cnn_spec]
    policy = AtariCnnPolicy(**cnn_args)
    print("cnn_args: ", cnn_args)

    runner = AccelRLAsync(
        algo=algo,
        policy=policy,
        sampler=sampler,
        n_steps=10e6,
        log_interval_steps=1e5,
        affinities=affinities,
        seed=None,
        use_gpu=True,
    )

    log_params = dict(
        exp='basic_mappo',
        cnn_spec=cnn_spec,
        n_envs=n_envs,
        learning_rate=learning_rate,
    )
    with logger_context(log_dir, game, game_run_ID, log_params):
        runner.train()


if __name__ == "__main__":
    build_and_run(*sys.argv[1:])
