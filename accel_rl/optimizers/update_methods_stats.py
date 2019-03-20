
import numpy as np
import theano
import theano.tensor as T
import lasagne.updates as LU
import lasagne.utils
from collections import OrderedDict

# import ipdb

def rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    """
    Exact copy from Lasagne updates, except also return expressions for
    the update step of each param.
    """
    grads = LU.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    steps = list()
    one = T.constant(1)
    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
            broadcastable=param.broadcastable)
        accu_new = rho * accu + (one - rho) * grad ** 2
        updates[accu] = accu_new
        # updates[param] = param - (learning_rate * grad /
        #                           T.sqrt(accu_new + epsilon))
        step = learning_rate * grad / T.sqrt(accu_new + epsilon)
        updates[param] = param - step
        steps.append(step)

    return updates, steps


def rmsprop_flat(flat_grad, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
    updates = OrderedDict()
    steps = list()
    one = T.constant(1)
    # ipdb.set_trace()
    param_size = sum([p.get_value(borrow=True).size for p in params])
    accu = theano.shared(np.zeros(param_size, dtype=flat_grad.dtype))
    accu_new = rho * accu + (one - rho) * flat_grad ** 2
    updates[accu] = accu_new
    step = learning_rate * flat_grad / T.sqrt(accu_new + epsilon)
    steps.append(step)
    i = 0
    for p in params:
        val = p.get_value(borrow=True)
        extract = step[i:i + val.size]
        updates[p] = p - T.reshape(extract, val.shape)
        i += val.size
    return updates, steps


def adam(loss_or_grads, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    """
    Exact copy from Lasagne updates, except also return expressions for the
    update step of each param.
    """
    all_grads = LU.get_or_compute_grads(loss_or_grads, params)
    t_prev = theano.shared(lasagne.utils.floatX(0.))
    updates = OrderedDict()
    steps = list()
    one = T.constant(1)
    t = t_prev + 1
    a_t = learning_rate * T.sqrt(one - beta2 ** t) / (one - beta1 ** t)

    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1 * m_prev + (one - beta1) * g_t
        v_t = beta2 * v_prev + (one - beta2) * g_t ** 2
        step = a_t * m_t / (T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

        steps.append(step)

    updates[t_prev] = t
    return updates, steps


def adam_flat(flat_grad, params, learning_rate=0.001, beta1=0.9,
        beta2=0.999, epsilon=1e-8):
    t_prev = theano.shared(lasagne.util.floatX(0.))
    updates = OrderedDict()
    steps = list()
    one = T.constant(1)
    t = t_prev + 1
    a_t = learning_rate * T.sqrt(one - beta2 ** t) / (one - beta1 ** t)

    m_prev = theano.shared(np.zeros(flat_grad.shape, dtype=flat_grad.dtype))
    v_prev = theano.shared(np.zeros(flat_grad.shape, dtype=flat_grad.dtype))

    m_t = beta1 * m_prev + (one - beta1) * flat_grad
    v_t = beta2 * v_prev + (one - beta2) * flat_grad ** 2
    step = a_t * m_t / (T.sqrt(v_t) + epsilon)

    updates[m_prev] = m_t
    updates[v_prev] = v_t
    steps.append(step)

    i = 0
    for p in params:
        val = p.get_value(borrow=True)
        extract = step[i:i + val.size]
        updates[p] = p - T.reshape(extract, val.shape)
        i += val.size

    updates[t_prev] = t
    return updates, steps
