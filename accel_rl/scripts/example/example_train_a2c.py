
import sys

from accel_rl.runners.accel_rl import AccelRL
from accel_rl.algos.pg.a2c import A2C
from accel_rl.policies.pg.atari_cnn_policy import AtariCnnPolicy
from accel_rl.policies.pg.atari_lstm_policy import AtariLstmPolicy
from accel_rl.policies.pg.atari_gru_policy import AtariGruPolicy
from accel_rl.policies.atari_cnn_specs import cnn_specs
from accel_rl.sampler.act_server.alternating.overlap.sampler import \
    ActsrvAltOvrlpSampler
from accel_rl.envs.atari_env import AtariEnv

from accel_rl.scripts.launching.affinities import get_affinities
from accel_rl.util.logging import logger_context


def build_and_run(run_slot_affinities_code, log_dir, game, game_run_ID,
        learning_rate, policy_type):
    affinities = get_affinities(run_slot_affinities_code)
    assert isinstance(affinities, dict)  # should not be a list

    learning_rate = float(learning_rate)  # (convert from string input)
    n_envs = 64
    if policy_type == 'lstm':
        policy = AtariLstmPolicy
    elif policy_type == 'gru':
        policy = AtariGruPolicy
    elif policy_type == 'ff':
        policy = AtariCnnPolicy

    env_args = dict(
        game=game,
        clip_reward=True,
        max_start_noops=30,
        episodic_lives=True,
        num_img_obs=1,
    )

    n_sim_cores = len(affinities["sim_cores"])
    assert n_envs % (n_sim_cores * 2) == 0
    envs_per = n_envs // (n_sim_cores * 2)

    sampler = ActsrvAltOvrlpSampler(
        EnvCls=AtariEnv,
        env_args=env_args,
        horizon=5,
        n_parallel=len(affinities["sim_cores"]),
        envs_per=envs_per,
        mid_batch_reset=False,
        max_path_length=int(27e3),
    )
    assert sampler.total_n_envs == n_envs

    algo = A2C(
        discount=0.99,
        gae_lambda=1,
        optimizer_args=dict(learning_rate=learning_rate),
    )

    cnn_spec = 0
    cnn_args = cnn_specs[cnn_spec]
    policy = policy(**cnn_args)
    print("cnn_args: ", cnn_args)

    runner = AccelRL(
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
        exp='basic_a2c',
        cnn_spec=cnn_spec,
        n_envs=n_envs,
        learning_rate=learning_rate,
        policy_type=policy_type
    )
    with logger_context(log_dir, game, game_run_ID, log_params):
        runner.train()


if __name__ == "__main__":
    build_and_run(*sys.argv[1:])
