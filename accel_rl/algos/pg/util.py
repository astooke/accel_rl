
import numpy as np
import theano.tensor as T


def gen_adv_est(rewards, values, dones, last_value, discount, gae_lambda,
        adv_dest=None, ret_dest=None):
    not_done = 1 - dones
    vpred = np.append(values, last_value)

    advantages = adv_dest if adv_dest is not None else \
        np.zeros(values.shape, dtype=values.dtype)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + discount * vpred[t + 1] * not_done[t] - vpred[t]
        advantages[t] = lastgaelam = \
            delta + discount * gae_lambda * not_done[t] * lastgaelam

    returns = ret_dest if ret_dest is not None else \
        np.zeros(values.shape, dtype=values.dtype)
    returns[:] = advantages + values

    return advantages, returns


def discount_returns(rewards, dones, last_value, discount, ret_dest=None):
    returns = ret_dest if ret_dest is not None else \
        np.zeros(rewards.shape, dtype=rewards.dtype)
    ret = last_value
    for t in reversed(range(len(rewards))):
        if dones[t]:
            ret = rewards[t]
        else:
            ret *= discount
            ret += rewards[t]
        returns[t] = ret
    return returns


def zero_after_reset(advantages, returns, values, path):
    need_reset = path["env_infos"].get("need_reset", path["dones"])
    if any(need_reset):
        t_invalid = np.min(np.where(need_reset)) + 1
        advantages[t_invalid:] = 0
        returns[t_invalid:] = 0
        values[t_invalid:] = 0


def valids_mean(expression, valids=None):
    if valids is None:
        return T.mean(expression)
    else:
        return T.sum(valids * expression) * (1. / T.sum(valids))


def update_valids(path, path_valids):
    need_reset = path["env_infos"].get("need_reset", path["dones"])
    if any(need_reset):
        t_invalid = np.min(np.where(need_reset)) + 1
        path_valids[:t_invalid] = 1
        path_valids[t_invalid:] = 0
    else:
        path_valids[:] = 1
