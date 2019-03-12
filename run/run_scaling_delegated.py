from __future__ import division, print_function
import os, sys; sys.path.insert(0, os.path.abspath("."))
from run.aux_scaling_delegated import run_scaling_delegated
import numpy as np
import itertools as it

num_processes = 64
num_agents = 128
num_trials = 10000
target_fid = 0.9
result_path = "results/scaling_delegated/"

p_gates = [0.986, 0.987, 0.988, 0.989, 0.991, 0.992, 0.993, 0.994, 0.996, 0.997, 0.998, 0.999, 0.981, 0.982, 0.983, 0.984]  # [0.98, 0.985, 0.99, 0.995, 1.0]

setups = [{"repeater_length": 2, "allowed_block_lengths": [], "start_fids": list(it.product(np.arange(0.6, 1.00, 0.05), repeat=2))},
          {"repeater_length": 3, "allowed_block_lengths": [2], "start_fids": list(it.product(np.arange(0.6, 1.00, 0.1), repeat=3))},
          {"repeater_length": 4, "allowed_block_lengths": [2, 3], "start_fids": list(it.product(np.arange(0.6, 1.00, 0.1), repeat=4))},
          {"repeater_length": 8, "allowed_block_lengths": [2, 3, 4], "start_fids": [(0.7,) * 8, (0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6), (0.95, 0.9, 0.6, 0.9, 0.95, 0.95, 0.9, 0.6)]}
          ]

if __name__ == "__main__":
    for p in p_gates:
        p_path = result_path + "p_gates" + str(int(p * 1000)) + "/"
        for setup in setups:
            run_scaling_delegated(num_processes=num_processes, num_agents=num_agents, num_trials=num_trials,
                                  repeater_length=setup["repeater_length"], allowed_block_lengths=setup["allowed_block_lengths"],
                                  p_gates=p, target_fid=target_fid, start_fids=setup["start_fids"], result_path=p_path)
