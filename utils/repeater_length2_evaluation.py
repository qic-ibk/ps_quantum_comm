"""
Copyright 2020 Julius Wallnöfer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pickle
import matplotlib.pyplot as plt
from run.aux_scaling_delegated import naive_constant

start_fids = [(0.75,) * 2]
results_path = "results/scaling_delegated/p_gates990/length2_27/"  # 27 is the index corresponding to the (0.75, 0.75) starting fidelities
output_path = "results/repeater_length2/plot_ready/"
p_gates = 0.990


def assert_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


assert_dir(output_path)
with open(output_path + "info.txt", "w") as f:
    info_lines = ["p=" + str(p_gates), "starting_fidelities: " + str(start_fids)]
    info_lines = "\n".join(info_lines)
    f.writelines(info_lines)

for i, start_fid in enumerate(start_fids):
    print(start_fids[i])
    path = results_path
    with open(path + "best_history.pickle", "rb") as f:  # best history is actually quite useless like this - we need actions instead of action_indices
        history = pickle.load(f)
    action_history = [action for _, _, action in history]
    b = [str(k) for k in action_history]
    b = "\n".join(b)
    print(b)
    with open(path + "block_action.pickle", "rb") as f:
        block_action = pickle.load(f)
    a = [str(k) for k in block_action["actions"]]
    a = "\n".join(a)
    # print(a)
    resources = np.load(path + "best_resources.npy")
    const = naive_constant(repeater_length=len(start_fid), start_fid=start_fid, target_fid=0.9, p_gates=p_gates)
    with open(output_path + "naive_constant.txt", "w") as f:
        f.write(str(const))
    # resources = resources[:1500]
    plt.scatter(np.arange(1, len(resources) + 1), resources, s=20)
    plt.yscale("log")
    plt.axhline(y=const, color='r')
    # plt.title("starting fids: " + str(start_fid) + " ; best solution found")
    plt.ylabel("resources used")
    plt.xlabel("trial number")
    plt.savefig(output_path + "best_learning_curve.png")
    plt.show()
    np.savetxt(output_path + "best_learning_curve.txt", resources)
    resource_list = np.loadtxt(path + "resource_list.txt")
    try:
        # resource_list = resource_list[np.logical_not(np.isnan(resource_list))]
        # print(dict(zip(*np.unique(resource_list, return_counts=True))))
        # print(resource_list)
        plt.hist(resource_list, bins=50)
    except IndexError:
        plt.axvline(x=resource_list[0])
    plt.axvline(x=const, color='r')
    plt.xscale("log")
    # ax = plt.gca()
    # ax.ticklabel_format(axis="x", style="sci")
    plt.ylabel("number of agents")
    plt.xlabel("resources of found solution")
    # plt.title("starting fids = " + str(start_fid))
    plt.savefig(output_path + "resource_histogram.png")
    plt.show()
    np.savetxt(output_path + "resource_histogram.txt", resource_list)
