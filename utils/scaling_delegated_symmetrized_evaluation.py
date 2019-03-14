import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pickle
import matplotlib.pyplot as plt
from run.run_scaling_delegated_symmetrized import naive_constant

start_fid = 0.75
start_fid_index = 3
results_path = "results/scaling_delegated_symmetrized/p_gates99/"
lengths = [2, 4, 8, 16, 32, 64]
p_gates = 0.99


compare = np.zeros(len(lengths))
for j, length in enumerate(lengths):
    path = results_path + "length%d_%d/" % (length, start_fid_index)
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
    const = naive_constant(repeater_length=length, start_fid=start_fid, target_fid=0.9, p=p_gates)
    # resources = resources[:1000]
    plt.scatter(np.arange(1, len(resources) + 1), resources, s=20)
    plt.yscale("log")
    plt.axhline(y=const, color='r')
    plt.title("starting fids: " + str(start_fid) + " ; best solution found ; " + "length = " + str(length))
    plt.ylabel("resources used")
    plt.xlabel("trial number")
    plt.savefig(path + "best_resources.png")
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
    plt.title("starting fids = " + str(start_fid) + " ; length = " + str(length))
    plt.show()
    min_resource = min(resource_list)
    compare[j] = min_resource / const

plt.scatter(np.log2(lengths), compare)
plt.title("starting fids: " + str(start_fid))
plt.xlabel("distance")
plt.ylabel("resources relative to working fidelity strategy")
# plt.xscale("log")
plt.xticks(np.log2(lengths), ["2^1", "2^2", "2^3", "2^4", "2^5", "2^6"])
plt.grid()
plt.savefig(results_path + "comparison_by_length.png")
plt.show()