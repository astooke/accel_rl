
import numpy as np
import theano
import theano.tensor as T
import lasagne.updates as LU


def iterate_mb_idxs(batch_size, data_length, shuffle=False):
    if shuffle:
        indices = np.arange(data_length)
        np.random.shuffle(indices)
    for start_idx in range(0, data_length - batch_size + 1, batch_size):
        if shuffle:
            # NOTE: tuple is hack until theano slice works (need update)
            batch = (indices[start_idx:start_idx + batch_size], )
        else:
            batch = (start_idx, start_idx + batch_size)
        yield batch


def iterate_traj_idxs(batch_size, data_length, horizon=1, shuffle=False):
    assert data_length % horizon == 0
    assert batch_size % horizon == 0
    assert data_length % batch_size == 0
    indices = np.arange(data_length).reshape(-1, horizon)
    traj_per_batch = batch_size // horizon
    trajectories = np.arange(data_length // horizon)
    if shuffle:
        np.random.shuffle(trajectories)
    for mb_idx in range(0, len(trajectories), traj_per_batch):
        trajs = trajectories[mb_idx:mb_idx + traj_per_batch]
        yield (indices[trajs].reshape(-1), trajs)


def flat_shared_grad(target, gradients):
    param_values = target.get_param_values(trainable=True)
    shared_grad = theano.shared(param_values, name='shared_grad')
    flat_gradient = T.concatenate([T.flatten(g) for g in gradients])
    return flat_gradient, shared_grad, (shared_grad, flat_gradient)


def reshape_grad(flat_grad, params):
    shaped_grads = list()
    i = 0
    for p in params:
        val = p.get_value(borrow=True)
        shaped_grads.append(T.reshape(flat_grad[i:i + val.size], val.shape))
        i += val.size
    return shaped_grads


def copy_params_from_flat(params, flat_var):
    updates = list()
    i = 0
    for p in params:
        val = p.get_value(borrow=True)
        extract = T.reshape(flat_var[i:i + val.size], val.shape)
        updates.append((p, extract))
        i += val.size
    return updates


def avg_grads_from_flat(flat_grad, params):
    shaped_grads = reshape_grad(flat_grad, params)
    avg_factor = theano.shared(np.array(1., dtype=theano.config.floatX))
    avg_gradients = [avg_factor * g for g in shaped_grads]
    return avg_gradients, avg_factor


def apply_grad_norm_clip(gradients, clip=None):
    if clip is None:
        _, norm = LU.total_norm_constraint(gradients, 1, return_norm=True)
    else:
        gradients, norm = LU.total_norm_constraint(gradients, clip,
            return_norm=True)
    return gradients, norm


def make_shared_inputs(inputs, shuffle):
    input_shareds = [theano.shared(
        np.zeros([1] * inp.ndim, dtype=inp.dtype),
        broadcastable=inp.broadcastable,
        )
        for inp in inputs]
    load_updates = [(s, inp) for inp, s in zip(inputs, input_shareds)]
    if shuffle:
        idxs = T.ivector('idxs')
        givens = [(inp, s[idxs]) for inp, s in zip(inputs, input_shareds)]
        opt_inputs = [idxs]
    else:
        start = T.iscalar('start')  # later, make this a slice (need theano update)
        stop = T.iscalar('stop')
        givens = [(inp, s[start:stop]) for inp, s in zip(inputs, input_shareds)]
        opt_inputs = [start, stop]
    return load_updates, givens, opt_inputs


def make_shared_inputs_recur(inputs, init_state_inputs):
    # Just use the same indexing mechanicsm regardless of whether shuffling.
    input_shareds = [theano.shared(
        np.zeros([1] * inp.ndim, dtype=inp.dtype),
        broadcastable=inp.broadcastable,
        )
        for inp in inputs]
    init_state_shareds = [theano.shared(
        np.zeros([1] * init_state.ndim, dtype=init_state.dtype),
        broadcastable=init_state.broadcastable,
        )
        for init_state in init_state_inputs]
    all_in = inputs + init_state_inputs
    all_shared = input_shareds + init_state_shareds
    load_updates = [(s, inp) for inp, s in zip(all_in, all_shared)]
    idxs = T.ivector('idxs')
    state_idxs = T.ivector('idxs')
    givens = [(inp, s[idxs]) for inp, s in zip(inputs, input_shareds)]
    givens += [(state, s[state_idxs]) for state, s in
        zip(init_state_inputs, init_state_shareds)]
    opt_inputs = [idxs, state_idxs]
    return load_updates, givens, opt_inputs


def scale_conv_gradients(params, gradients, scale):
    for i, (p, g) in enumerate(zip(params, gradients)):
        if "conv" in p.name:
            gradients[i] = scale * g
    return gradients
