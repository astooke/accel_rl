from accel_rl.scripts.launching.exp_launcher import \
    run_GPU_exps, build_args_list, chunk_list, log_exp_list

import sys
n_computers = int(sys.argv[1])
computer_ID = int(sys.argv[2])

script = "accel_rl/scripts/example/example_train_cat_dqn.py"

# n_gpu, ctx_per_gpu, ctx_per_run, n_cpu_cores, n_sockets, (ht_offset)
affinity_code = "2gpu_1cxg_1cxr_6cpu"  # (write according to computer specs)
# (ctx_per_gpu (cxg) will run multiple independent learners per GPU; can be faster)
# (ctx_per_run (cxr) must match the desired number of parallel learners)
# (n_cpu_cores (cpu) -- these will be divided evenly among workers, with one
# core reserved per learner for driving GPU.)
# (n_sockets (skt) is optional that attempts to have CPUs driving GPUs with
# same PCIe affinity)
# hyperthread_offset (hto) is optional to use different than provided cpu count

exp_name = "basic_cat_dqn"

runs_per_game = 2

games = ["alien", "beam_rider", "space_invaders", "zaxxon"]

# Must write out each setting explicitly: since len(learning_rates)==2, there
# will be 2 settings used (len(batch_sizes) must equal len(learning_rates))
learning_rates = [2.5e-4, 7.5e-4]
batch_sizes = [512, 512]

exps = list(zip(learning_rates, batch_sizes))  # same order as train script

sub_names = ["dqn_{}lr_{}bs".format(*exp) for exp in exps]

args_list = build_args_list(exp_name, sub_names, games, runs_per_game, exps)
log_exp_list(exp_name, sub_names, exps)

if n_computers > 1:
    args_chunks = chunk_list(args_list, n_computers)
    my_args_list = args_chunks[computer_ID]
else:
    my_args_list = args_list


run_GPU_exps(script, affinity_code, my_args_list)

# (Within the same file here, could set up multiple different experiments that
# would each call run_GPU_exps...the next will not start until the previous has
# completely finished.  Under current setup, all experiments running together
# must use the same script and affinity code.)
