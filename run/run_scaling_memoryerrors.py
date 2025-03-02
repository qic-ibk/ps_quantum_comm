"""
Copyright 2020 Julius Wallnöfer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from __future__ import division, print_function
import os, sys; sys.path.insert(0, os.path.abspath("."))
from run.aux_scaling_memoryerrors import run_scaling_fids, run_scaling_distances, SolutionCollection, assert_dir
import numpy as np
import itertools as it
from warnings import warn

num_processes = 64
num_agents = 128
num_trials = 10000
target_fid = 0.9
result_path = "results/scaling_memoryerrors/raw/"

# p_gates = np.arange(0.98, 0.001, 1.001)  # takes about 1 day per data point
p_gates = [0.99]
# memory_alphas = [1.0 / 1, 1.0 / 0.5]  # one second/half a second
# memory_alphas = [1.0 / 0.1, 1.0 / 0.01, 1.0 / 0.001]
# memory_alphas = [1. / 0.09, 1. / 0.08, 1. / 0.07, 1. / 0.06, 1. / 0.05, 1. / 0.04, 1. / 0.03, 1. / 0.02]
memory_alphas = [26, 27, 28, 29, 30, 31, 32]


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
            p_path = result_path + "p_gates" + str(int(p * 1000)) + "_alpha" + str(int(alpha)) + "/"
            for setup in setups:
                run_scaling_fids(num_processes=num_processes, num_agents=num_agents, num_trials=num_trials,
                                 start_fids=setup["start_fids"], allowed_block_lengths=setup["allowed_block_lengths"],
                                 p_gates=p, memory_alpha=alpha, target_fid=target_fid, result_path=p_path)
        # run_scaling_distances(num_processes=num_processes, num_agents=num_agents, num_trials=num_trials,
        #                       distances=test_setup["distances"], allowed_block_lengths=test_setup["allowed_block_lengths"],
        #                       p_gates=p, memory_alpha=memory_alpha, target_fid=target_fid, result_path=p_path)


# # # Different length distribution run
#
# # total_length = 20000  # 20km
# num_possible_stations = 7
# # positions = [x * total_length / (num_possible_stations + 1) for x in range(num_possible_stations + 2)]  # equidistant
# alpha = 1 / 0.1
# p = 0.99
# target_fid = 0.9
# position_path = "results/scaling_memoryerrors_positions/raw/" + "p_gates" + str(int(p * 1000)) + "_alpha" + str(int(alpha)) + "/"
# sc_path = "results/scaling_memoryerrors/raw/" + "p_gates" + str(int(p * 1000)) + "_alpha" + str(int(alpha)) + "/"
# assert_dir(position_path)
#
# positions = [0, 6823, 23590, 30413, 37237, 48475, 55298, 62122, 78889]  # corresponds to initial fidelities [(0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6)]
# positions = [x * 20000 / 78889 for x in positions]  # shorten distance to 20km while maintaining the distance ratios
#
#
# def distances_from_station_numbers(numbers):
#     aux1 = [0] + list(numbers)
#     aux2 = list(numbers) + [num_possible_stations + 1]
#     distances = [positions[j] - positions[i] for i, j in zip(aux1, aux2)]
#     return distances
#
#
# # take the solution collection from the above results
# sc = SolutionCollection()
# try:
#     sc.load(sc_path + "/solution_collection.pickle")
# except IOError:
#     # warn("SolutionCollection not found - creating new one.")
#     warn("Expected existing SolutionCollection not found.")
#     raise
# sc.save(position_path + "solution_collection.pickle")
#
#
# # one additional station
# distances = [distances_from_station_numbers([i]) for i in range(1, num_possible_stations + 1)]
# run_scaling_distances(num_processes=num_processes, num_agents=num_agents, num_trials=num_trials,
#                       distances=distances, allowed_block_lengths=[],
#                       p_gates=p, memory_alpha=alpha, target_fid=target_fid, result_path=position_path)
#
# # two additional stations
# distances = [distances_from_station_numbers([i, j]) for i in range(1, num_possible_stations) for j in range(i + 1, num_possible_stations + 1)]
# run_scaling_distances(num_processes=num_processes, num_agents=num_agents, num_trials=num_trials,
#                       distances=distances, allowed_block_lengths=[2],
#                       p_gates=p, memory_alpha=alpha, target_fid=target_fid, result_path=position_path)
#
# # three additional stations
# distances = [distances_from_station_numbers([i, j, k]) for i in range(1, num_possible_stations - 1) for j in range(i + 1, num_possible_stations) for k in range(j + 1, num_possible_stations + 1)]
# run_scaling_distances(num_processes=num_processes, num_agents=num_agents, num_trials=num_trials,
#                       distances=distances, allowed_block_lengths=[2, 3],
#                       p_gates=p, memory_alpha=alpha, target_fid=target_fid, result_path=position_path)
