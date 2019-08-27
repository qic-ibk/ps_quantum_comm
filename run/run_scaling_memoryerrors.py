from __future__ import division, print_function
import os, sys; sys.path.insert(0, os.path.abspath("."))
from run.aux_scaling_memoryerrors import run_scaling_fids, run_scaling_distances
import numpy as np
import itertools as it

num_processes = 64
num_agents = 128
num_trials = 10000
target_fid = 0.9
result_path = "results/scaling_memoryerrors/raw/"

# p_gates = np.arange(0.98, 0.001, 1.001)  # takes about 1 day per data point
p_gates = [0.99]
memory_alphas = [1.0 / 1, 1.0 / 0.5]  # one second/half a second


setups = [{"allowed_block_lengths": [], "start_fids": list(it.product(np.arange(0.6, 1.00, 0.05), repeat=2))},
          {"allowed_block_lengths": [2], "start_fids": list(it.product(np.arange(0.6, 1.00, 0.1), repeat=3))},
          {"allowed_block_lengths": [2, 3], "start_fids": list(it.product(np.arange(0.6, 1.00, 0.1), repeat=4))},
          # {"allowed_block_lengths": [2, 3, 4], "start_fids": [(0.7,) * 8, (0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6), (0.95, 0.9, 0.6, 0.9, 0.95, 0.95, 0.9, 0.6)]}
          {"allowed_block_lengths": [2, 3, 4], "start_fids": [(0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6)]}
          ]

# setups = [{"start_fids": list(it.product(np.arange(0.6, 1.00, 0.05), repeat=2)), "allowed_block_lengths": []}]
# test_setup = {"distances": [(0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6)], "allowed_block_lengths": [2, 3, 4]}

if __name__ == "__main__":
    for p in p_gates:
        for alpha in memory_alphas:
            p_path = result_path + "p_gates" + str(int(p * 1000)) + "_alpha" + str(int(alpha * 10)) + "/"
            for setup in setups:
                run_scaling_fids(num_processes=num_processes, num_agents=num_agents, num_trials=num_trials,
                                 start_fids=setup["start_fids"], allowed_block_lengths=setup["allowed_block_lengths"],
                                 p_gates=p, memory_alpha=alpha, target_fid=target_fid, result_path=p_path)
        # run_scaling_distances(num_processes=num_processes, num_agents=num_agents, num_trials=num_trials,
        #                       distances=test_setup["distances"], allowed_block_lengths=test_setup["allowed_block_lengths"],
        #                       p_gates=p, memory_alpha=memory_alpha, target_fid=target_fid, result_path=p_path)
