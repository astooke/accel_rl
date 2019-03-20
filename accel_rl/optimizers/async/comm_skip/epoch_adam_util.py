import theano
import theano.tensor as T
import numpy as np


def adam_accum_updates(gradients, params, learning_rate,
        beta1=0.9, beta2=0.999, epsilon=1e-5):
    t_prev = theano.shared(np.array(0., dtype=theano.config.floatX))
    one = T.constant(1)

    t = t_prev + 1
    a_t = learning_rate * T.sqrt(one - beta2 ** t) / (one - beta1 ** t)

    m_vars = list()
    v_vars = list()
    accum_step_vars = list()
    accum_g_vars = list()
    accum_g2_vars = list()

    grad_updates = list()
    clear_updates = list()

    for p, g in zip(params, gradients):
        value = p.get_value(borrow=True)
        m = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                          broadcastable=p.broadcastable)
        v = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                          broadcastable=p.broadcastable)
        accum_step = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                broadcastable=p.broadcastable)
        accum_g = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                broadcastable=p.broadcastable)
        accum_g2 = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=p.broadcastable)

        m_t = beta1 * m + (one - beta1) * g
        v_t = beta2 * v + (one - beta2) * g ** 2
        step = - a_t * m_t / (T.sqrt(v_t) + epsilon)
        p_t = p + step
        g_t = beta1 * accum_g + g
        g2_t = beta2 * accum_g2 + g ** 2
        step_t = accum_step + step

        grad_updates += [(m, m_t), (v, v_t), (p, p_t)]
        grad_updates += [(accum_step, step_t), (accum_g, g_t), (accum_g2, g2_t)]

        m_vars.append(m)
        v_vars.append(v)
        accum_step_vars.append(accum_step)
        accum_g_vars.append(accum_g)
        accum_g2_vars.append(accum_g2)

        clear_updates.append((accum_g, accum_g.fill(0.)))
        clear_updates.append((accum_g2, accum_g2.fill(0.)))
        clear_updates.append((accum_step, accum_step.fill(0.)))

    grad_updates.append((t_prev, t))
    accum_vars = [accum_step_vars, accum_g_vars, accum_g2_vars]

    return grad_updates, clear_updates, m_vars, v_vars, t_prev, accum_vars


###############################################################################
# separated updates
###############################################################################


def apply_accums_params(shared_vars, accums, old_flat_val):
    updates = list()
    new_vals = list()
    i = 0
    for var, acc in zip(shared_vars, accums):
        val = var.get_value(borrow=True)
        extract = T.reshape(old_flat_val[i:i + val.size], val.shape)
        new_val = extract + acc
        updates.append((var, new_val))
        new_vals.append(new_val)
        i += val.size
    new_flat_val = T.concatenate([T.flatten(nv) for nv in new_vals])
    return updates, new_flat_val


def apply_accums_adam(shared_vars, accums, old_flat_val, beta, n_step):
    one = T.constant(1.)
    updates = list()
    new_vals = list()
    i = 0
    for var, acc in zip(shared_vars, accums):
        val = var.get_value(borrow=True)
        extract = T.reshape(old_flat_val[i:i + val.size], val.shape)
        new_val = extract * \
            T.cast(beta ** n_step, theano.config.floatX) + (one - beta) * acc
        updates.append((var, new_val))
        new_vals.append(new_val)
        i += val.size
    new_flat_val = T.concatenate([T.flatten(nv) for nv in new_vals])
    return updates, new_flat_val


def pull_to_local(shared_vars, central_flat_val):
    pull_updates = list()
    i = 0
    for var in shared_vars:
        val = var.get_value(borrow=True)
        extract = T.reshape(central_flat_val[i:i + val.size], val.shape)
        pull_updates.append((var, extract))
        i += val.size
    return pull_updates


def adam_make_flat_var():
    return T.vector("flat_var")
