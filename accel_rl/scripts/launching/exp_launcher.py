
"""
LAUNCHING SINGLE- or MULTI-GPU EXPERIMENTS (side-by-side)

(affinities passed as arguments)
"""

import os
import subprocess
import time

from accel_rl.util.logging import make_log_dir
from accel_rl.scripts.launching.affinities import \
    decode_affinity_params, prepend_run_slot_code


def launch_GPU_exp(script, run_slot, run_slot_affinities_code,
                   log_dir, game, game_run_ID,
                   *other_args):
    last_args = " ".join([str(a) for a in (game_run_ID,) + other_args])
    run_env = os.environ.copy()
    compdirflag = "compiledir=~/.theano/compiledir_accel_rl_run_slot_{},".format(
        run_slot)
    run_env["THEANO_FLAGS"] = compdirflag + run_env.get("THEANO_FLAGS", "")
    call_str = "python {} {} {} {} {}".format(
        script, run_slot_affinities_code, log_dir, game, last_args)
    print("\ncall string:\n", call_str)
    call_list = call_str.split(" ")
    p = subprocess.Popen(call_list, env=run_env)
    return p


# (to be imported into the run script which sets up the variants and puts them
# into args_list)
def run_GPU_exps(script, affinities_code, args_list):
    aff_params = decode_affinity_params(affinities_code)
    n_gpu, ctx_per_gpu, ctx_per_run = \
        (aff_params[k] for k in ["n_gpu", "ctx_per_gpu", "ctx_per_run"])
    n_run_slots = (n_gpu * ctx_per_gpu) // ctx_per_run
    procs = [None] * n_run_slots
    for args in args_list:
        launched = False
        while not launched:
            for run_slot, p in enumerate(procs):
                if p is None or p.poll() is not None:
                    run_slot_affinities_code = prepend_run_slot_code(run_slot,
                        affinities_code)
                    procs[run_slot] = launch_GPU_exp(script, run_slot,
                        run_slot_affinities_code, *args)
                    launched = True
                    break
            if not launched:
                time.sleep(10)
    for p in procs:
        if p is not None:
            p.wait()  # (don't return until they are all done)


def build_args_list(exp_name, sub_names, games, runs_per_game, exp_args_list):
    args_list = list()
    for sub_name, exp_args in zip(sub_names, exp_args_list):
        log_dir = make_log_dir(exp_name, sub_name)
        for game in games:
            for run_ID in range(runs_per_game):
                args = (log_dir, game, run_ID) + exp_args
                args_list.append(args)
    return args_list


def chunk_list(full_list, n_chunks):
    chunk_size = (len(full_list) // n_chunks)
    chunks = list()
    for i in range(0, len(full_list), chunk_size):
        chunks.append(full_list[i:i + chunk_size])
    if len(chunks) > n_chunks:
        extra = chunks.pop()
        for ex, ch in zip(extra, chunks):
            ch.append(ex)
    return chunks


def log_exp_list(exp_name, sub_names, exp_args_list):
    log_dir = make_log_dir(exp_name)
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, "experiments.txt")
    with open(filename, "w") as f:
        [f.write(name + " " + " ".join([str(a) for a in exp_args]) + "\n")
        for name, exp_args in zip(sub_names, exp_args_list)]
