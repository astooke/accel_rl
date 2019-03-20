
import numpy as np
import theano
import theano.tensor as T
from rllab.misc import logger

CHUNKED_UPDATE_NAMES = ["rmsprop", "adam"]  # (list of all that have been implemented)
WHOLE_UPDATE_NAMES = CHUNKED_UPDATE_NAMES + []


def get_init_arrays(target, update_method_name):
    """
    Returns a list of numpy arrays of the exact shape needed for centrally held
    parameters and update method values, in the order to be fed into and
    returned from the update function, and with correct initial values.
    """
    if update_method_name == "rmsprop":
        param_values = target.get_param_values(trainable=True)
        accum_values = param_values.copy()
        accum_values.fill(0)
        return [param_values, accum_values]  # (one for params, one for accums)
    elif update_method_name == "adam":
        param_values = target.get_param_values(trainable=True)
        m_values = param_values.copy()
        m_values.fill(0)
        v_values = param_values.copy()
        v_values.fill(0)
        return [param_values, m_values, v_values]
    else:
        raise NotImplementedError


def chunked_updates(update_method_name, updates_args):
    if update_method_name == "rmsprop":
        return rmsp_chunks(*updates_args)
    elif update_method_name == "adam":
        return adam_chunks(*updates_args)
    else:
        raise NotImplementedError


def whole_update(update_method_name, updates_args):
    if update_method_name == "rmsprop":
        return rmsp_whole(*updates_args)
    else:
        raise NotImplementedError


def rmsp_whole(shared_grad, learning_rate, rmsp_settings):
    raise NotImplementedError


def rmsp_chunks(shared_grad, learning_rate, n_chunks, rmsp_settings):
    rho = rmsp_settings.get("rho", 0.9)
    epsilon = rmsp_settings.get("epsilon", 1e-6)
    n_params = shared_grad.get_value(borrow=True).size
    p_per_chunk = (n_params // n_chunks) + 1
    logger.log("Using chunked RMSProp with settings -- rho: {}  epsilon: "
        "{}  num_chunks: {}".format(rho, epsilon, n_chunks))

    old_p = T.vector('old_parameters')
    old_a = T.vector('old_accum')

    def rmsp_chunk(start):
        stop = start + p_per_chunk
        n_a = rho * old_a + (1. - rho) * shared_grad[start:stop] ** 2
        n_p = old_p - \
            learning_rate * shared_grad[start:stop] / T.sqrt(n_a + epsilon)
        update = (shared_grad, T.set_subtensor(shared_grad[start:stop], n_p))
        return n_p, n_a, update

    outputs_lists, rmsp_updates, idxs = (list() for _ in range(3))
    for i in range(0, n_params, p_per_chunk):
        new_p, new_a, chunk_update = rmsp_chunk(i)
        outputs_lists.append([new_p, new_a])
        rmsp_updates.append(chunk_update)
        idxs.append((i, i + p_per_chunk))
    inputs = (old_p, old_a)

    return inputs, outputs_lists, rmsp_updates, idxs


def adam_chunks(shared_grad, learning_rate, n_chunks, adam_settings):
    beta1 = adam_settings.get("beta1", 0.9)
    beta2 = adam_settings.get("beta2", 0.999)
    epsilon = adam_settings.get("epsilon", 1e-8)

    n_params = shared_grad.get_value(borrow=True).size
    p_per_chunk = (n_params // n_chunks) + 1
    logger.log("Using chunked Adam with settings -- beta1: {}, beta2: {}"
        " epsilon: {}  num_chunks: {}".format(beta1, beta2, epsilon, n_chunks))

    old_p = T.vector('old_parameters')
    old_m = T.vector('old_m')
    old_v = T.vector('old_v')

    t_prev = theano.shared(np.array(0., dtype=theano.config.floatX))
    t = t_prev + 1
    one = T.constant(1)
    a_t = learning_rate * T.sqrt(one - beta2 ** t) / (one - beta1 ** t)

    def adam_chunk(start):
        stop = start + p_per_chunk
        grad_chunk = shared_grad[start:stop]
        m_t = beta1 * old_m + (one - beta1) * grad_chunk
        v_t = beta2 * old_v + (one - beta2) * grad_chunk ** 2
        step = a_t * m_t / (T.sqrt(v_t) + epsilon)
        p_t = old_p - step
        update = (shared_grad, T.set_subtensor(grad_chunk, p_t))
        return p_t, m_t, v_t, update

    outputs_lists, adam_updates, idxs = list(), list(), list()
    for i in range(0, n_params, p_per_chunk):
        new_p, new_m, new_v, chunk_update = adam_chunk(i)
        outputs_lists.append([new_p, new_m, new_v])
        adam_updates.append(chunk_update)
        idxs.append((i, i + p_per_chunk))
    inputs = [old_p, old_m, old_v]
    adam_updates.append((t_prev, t))  # extra update at end; don't forget it!
    return inputs, outputs_lists, adam_updates, idxs
