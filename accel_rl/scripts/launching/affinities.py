
""""
CPU & GPU Affinities

(Same scheme relating CPU cores to GPUs regardless of number of GPUs per run)
"""

# abbreviations for affinity code (so it's readable)
N_GPU = "gpu"
CTX_PER_GPU = "cxg"
CTX_PER_RUN = "cxr"
N_CPU_CORES = "cpu"
HT_OFFSET = "hto"
N_SOCKET = "skt"
RUN_SLOT = "slt"


###############################################################################
# Main API
###############################################################################


def encode_affinity_params(n_gpu, ctx_per_gpu, ctx_per_run, n_cpu_cores,
        ht_offset=None, n_socket=None, run_slot=None):
    """Use in the run script; specifies computer configuration"""
    affinities_code = "{}{}_{}{}_{}{}_{}{}".format(n_gpu, N_GPU, ctx_per_gpu,
        CTX_PER_GPU, ctx_per_run, CTX_PER_RUN, n_cpu_cores, N_CPU_CORES)
    if ht_offset is not None:
        affinities_code += "_{}{}".format(ht_offset, HT_OFFSET)
    if n_socket is not None:
        affinities_code += "_{}{}".format(n_socket, N_SOCKET)
    if run_slot is not None:
        assert run_slot <= ctx_per_run // (n_gpu * ctx_per_gpu)
        affinities_code = "{}{}_".format(run_slot, RUN_SLOT) + affinities_code
    return affinities_code


def prepend_run_slot_code(run_slot, affinities_code):
    """Use in the launch manager when assigning run slot"""
    return "{}{}_".format(run_slot, RUN_SLOT) + affinities_code


def get_affinities(run_slot_affinities_code):
    """Use in the inidividual experiment script; pass output to Runner"""
    run_slot_str, aff_code = run_slot_affinities_code.split("_", 1)
    assert run_slot_str[-3:] == RUN_SLOT
    run_slot = int(run_slot_str[:-3])
    return build_affinities(run_slot, **decode_affinity_params(aff_code))


###############################################################################
# Helpers / Extended API
###############################################################################

def build_affinities(run_slot, n_gpu, ctx_per_gpu, ctx_per_run, n_cpu_cores,
        ht_offset=None, n_socket=1):
    """
    This one concentrates the gpu_cores at the low end of each socket.
    """
    n_ctx = n_gpu * ctx_per_gpu
    n_run_slots = n_ctx // ctx_per_run
    assert run_slot < n_run_slots
    if ht_offset is None:
        ht_offset = n_cpu_cores
    cpu_per_gpu = n_cpu_cores // n_gpu
    sim_cpu_per_gpu = cpu_per_gpu - 1

    n_sim_cores = n_cpu_cores - n_gpu
    cores_per_ctx = n_sim_cores // n_ctx

    assert n_gpu >= n_socket
    assert n_gpu % n_socket == 0
    gpu_per_socket = n_gpu // n_socket
    assert n_cpu_cores % n_socket == 0
    cpu_per_socket = n_cpu_cores // n_socket

    affinities_by_ctx = list()
    contexts = list(range(ctx_per_run))
    contexts = [i + run_slot * ctx_per_run for i in contexts]
    for ctx in contexts:
        gpu = ctx // ctx_per_gpu
        socket = gpu // gpu_per_socket
        gpu_in_skt = gpu % gpu_per_socket
        gpu_core = gpu_in_skt + socket * cpu_per_socket
        ctx_in_gpu = ctx % ctx_per_gpu

        sim_cores = list(range(cores_per_ctx))
        sim_cores = [i + socket * cpu_per_socket for i in sim_cores]
        sim_cores = [i + gpu_per_socket for i in sim_cores]
        sim_cores = [i + gpu_in_skt * sim_cpu_per_gpu for i in sim_cores]
        sim_cores = [i + ctx_in_gpu * cores_per_ctx for i in sim_cores]

        gpu_cores = [gpu_core]
        if ctx_per_gpu > 1:
            gpu_cores += sim_cores
        gpu_cpus = hyperthreads(gpu_cores, ht_offset)
        sim_cpus = hyperthreads(sim_cores, ht_offset)
        aff = dict(gpu=gpu, gpu_cpus=gpu_cpus, sim_cpus=sim_cpus,
            sim_cores=sim_cores, gpu_cores=gpu_cores)
        affinities_by_ctx.append(aff)
    if len(affinities_by_ctx) == 1:
        affinities_by_ctx = affinities_by_ctx[0]
    return affinities_by_ctx


# def build_affinities(run_slot, n_gpu, ctx_per_gpu, ctx_per_run, n_cpu_cores,
#         ht_offset=None):
#     """
#     This one spreads out the gpu_cores, interspersed among sim_cores.
#     """
#     n_ctx = n_gpu * ctx_per_gpu
#     n_run_slots = n_ctx // ctx_per_run
#     assert run_slot < n_run_slots
#     if ht_offset is None:
#         ht_offset = n_cpu_cores
#     cpu_per_gpu = n_cpu_cores // n_gpu

#     n_sim_cores = n_cpu_cores - n_gpu
#     cores_per_ctx = n_sim_cores // n_ctx

#     affinities_by_ctx = list()
#     for ctx in range(run_slot * ctx_per_run, (run_slot + 1) * ctx_per_run):
#         ctx_in_gpu = ctx % ctx_per_gpu
#         gpu = ctx // ctx_per_gpu
#         gpu_core = cpu_per_gpu * gpu
#         sim_cores = [gpu_core + 1 + i for i in
#             range(ctx_in_gpu * cores_per_ctx, (ctx_in_gpu + 1) * cores_per_ctx)]
#         gpu_cores = [gpu_core]
#         if ctx_per_gpu > 1:
#             gpu_cores += sim_cores
#         gpu_cpus = hyperthreads(gpu_cores, ht_offset)
#         sim_cpus = hyperthreads(sim_cores, ht_offset)
#         aff = dict(gpu=gpu, gpu_cpus=gpu_cpus, sim_cpus=sim_cpus,
#             sim_cores=sim_cores, gpu_cores=gpu_cores)
#         affinities_by_ctx.append(aff)
#     if len(affinities_by_ctx) == 1:
#         affinities_by_ctx = affinities_by_ctx[0]
#     return affinities_by_ctx


def build_all_affinities(n_gpu, ctx_per_gpu, ctx_per_run, n_cpu_cores,
        ht_offset=None, n_socket=1):
    n_ctx = n_gpu * ctx_per_gpu
    all_affinities = list()
    for run_slot in range(n_ctx // ctx_per_run):
        affinities_within_run = build_affinities(run_slot, n_gpu,
            ctx_per_gpu, ctx_per_run, n_cpu_cores, ht_offset, n_socket)
        all_affinities.append(affinities_within_run)
    return all_affinities


def hyperthreads(cpu, ht_offset):
    # (e.g. in DGX proc and proc + 40 are two hyperthreads of same physical core)
    # returns tuples for input to psutil.Process.cpu_affinity()
    if isinstance(cpu, list):
        return tuple(cpu + [c + ht_offset for c in cpu])
    return (cpu, cpu + ht_offset)


def decode_affinity_params(affinities_code):
    args_strs = affinities_code.split("_")
    aff_kwargs = dict()
    for a_str in args_strs:
        abbrev = a_str[-3:]
        value = int(a_str[:-3])
        if abbrev == N_GPU:
            aff_kwargs["n_gpu"] = value
        elif abbrev == CTX_PER_GPU:
            aff_kwargs["ctx_per_gpu"] = value
        elif abbrev == CTX_PER_RUN:
            aff_kwargs["ctx_per_run"] = value
        elif abbrev == N_CPU_CORES:
            aff_kwargs["n_cpu_cores"] = value
        elif abbrev == HT_OFFSET:
            aff_kwargs["ht_offset"] = value
        elif abbrev == N_SOCKET:
            aff_kwargs["n_socket"] = value
        else:
            raise ValueError("Unrecognized affinity code abbreviation: ", abbrev)
    return aff_kwargs
