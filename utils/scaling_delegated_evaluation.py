import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pickle
import matplotlib.pyplot as plt
from run.run_scaling_delegated import naive_constant

start_fids = [(0.7,) * 8, (0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6), (0.95, 0.9, 0.6, 0.9, 0.95, 0.95, 0.9, 0.6)]
results_path = "results/scaling_delegated/p_gates98/"
p_gates = 0.98


for i, start_fid in enumerate(start_fids):
    print(start_fids[i])
    path = results_path + "length8_%d/" % i
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
    const = naive_constant(repeater_length=8, start_fid=start_fid, target_fid=0.9, p=p_gates)
    plt.scatter(np.arange(1, len(resources) + 1), resources, s=20)
    plt.yscale("log")
    plt.axhline(y=const, color='r')
    plt.title("starting fids: " + str(start_fid) + " ; best solution found")
    plt.ylabel("resources used")
    plt.xlabel("trial number")
    plt.show()
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
    plt.title("starting fids = " + str(start_fid))
    plt.show()
