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

start_fids = [(0.7,) * 8, (0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6), (0.95, 0.9, 0.6, 0.9, 0.95, 0.95, 0.9, 0.6)]
ps = np.linspace(0.98, 1.0, num=21)
results_path = "results/scaling_delegated/raw/"
output_path = "results/scaling_delegated/plot_ready/"


def assert_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


assert_dir(output_path)
with open(output_path + "info.txt", "w") as f:
    f.write("starting fidelities: " + str(start_fids) + "\n")
    f.write("learning curves are plotted for p=0.99")
np.savetxt(output_path + "ps.txt", ps)

for i, start_fid in enumerate(start_fids):
    print(start_fid)
    compare = np.zeros(len(ps))
    for j, p in enumerate(ps):
        print("p=%.3f" % p)
        p_path = results_path + "p_gates" + str(int(p * 1000)) + "/"
        p_gates = p
        path = p_path + "length8_%d/" % i
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
        const = naive_constant(repeater_length=8, start_fid=start_fid, target_fid=0.9, p_gates=p_gates)
        # resources = resources[:1000]
        plt.scatter(np.arange(1, len(resources) + 1), resources, s=20)
        plt.yscale("log")
        plt.axhline(y=const, color='r')
        # plt.title("starting fids: " + str(start_fid) + " ; best solution found")
        plt.ylabel("resources used")
        plt.xlabel("trial number")
        if p == 0.99:
            plt.savefig(output_path + "best_resources_p99_%d.png" % i)
            np.savetxt(output_path + "best_resources_p99_%d.txt" % i, resources)
            with open(output_path + "const_%d.txt" % i, "w") as f:
                f.write(str(const))
        # plt.show()
        plt.cla()
        resource_list = np.loadtxt(path + "resource_list.txt")
        try:
            # resource_list = resource_list[np.logical_not(np.isnan(resource_list))]
            # print(dict(zip(*np.unique(resource_list, return_counts=True))))
            # print(resource_list)
            plt.hist(resource_list, bins=300)
        except IndexError:
            plt.axvline(x=resource_list[0])
        plt.axvline(x=const, color='r')
        plt.xscale("log")
        # ax = plt.gca()
        # ax.ticklabel_format(axis="x", style="sci")
        plt.ylabel("number of agents")
        plt.xlabel("resources of found solution")
        # plt.title("starting fids = " + str(start_fid))
        # plt.show()
        plt.cla()
        min_resource = min(resource_list)
        compare[j] = min_resource / const

    plt.scatter(ps, compare)
    # plt.title("starting fids: " + str(start_fid))
    plt.xlabel("gate error parameter p")
    plt.ylabel("resources relative to symmetric guess")
    plt.grid()
    plt.savefig(output_path + "comparison_by_error_%d.png" % i)
    np.savetxt(output_path + "comparison_by_error_%d.txt" % i, compare)
    plt.show()
