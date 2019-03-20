
import numpy as np

from accel_rl.buffers.batch import buffer_with_segs_view, batch_buffer


def build_env_buffer(env, batch_size, horizon, need_extra_obs):
    env.reset()
    observation, reward, done, env_info = env.step(env.spec.action_space.sample())
    check_dict_level(env_info)
    examples = dict(
        observations=observation,
        rewards=reward,
        dones=done,
        env_infos=env_info,
        )
    env_buffer = buffer_with_segs_view(examples, batch_size, horizon, shared=True)
    if need_extra_obs:
        env_buffer.extra_observations = \
            batch_buffer(observation, batch_size // horizon, shared=False)
    return env_buffer


def build_step_buffer(env_spec, n_envs):
    examples = dict(
        obs=env_spec.observation_space.sample(),
        act=env_spec.action_space.sample(),
        reset=False,  # for recurrence
    )
    return batch_buffer(examples, n_envs, shared=True)


def build_policy_buffer(env_spec, policy, batch_size, horizon):
    policy.reset(n_batch=1)
    action, agent_info = policy.get_action(env_spec.observation_space.sample())
    check_dict_level(agent_info)
    examples = dict(actions=action, agent_infos=agent_info)
    return buffer_with_segs_view(examples, batch_size, horizon, shared=False)


def view_worker_segs_bufs(segs_list, n):
    assert len(segs_list) % n == 0
    segs_per = len(segs_list) // n
    worker_segs_lists = list()
    i = 0
    for _ in range(n):
        worker_segs_lists.append(segs_list[i:i + segs_per])
        i += segs_per
    return worker_segs_lists


def check_dict_level(info):
    for k, v in info.items():
        v = np.asarray(v)
        if v.dtype == "object":
            raise TypeError("Unsupported infos data type under key: {}\n"
                "Sampler does not permit nested dictionaries, values must be "
                "able to cast under np.asarray() and not result in "
                "dtype=='object')".format(k))


