import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pickle
import matplotlib.pyplot as plt
from run.run_scaling_delegated import naive_constant

start_fids = [(0.7,) * 8, (0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6)]

for i, start_fid in enumerate(start_fids):
    path = "results/scaling_delegated/length8_%d/" % i
    with open(path + "best_history.pickle", "rb") as f:  # best history is actually quite useless like this - we need actions instead of action_indices
        history = pickle.load(f)
    # print(history)
    with open(path + "block_action.pickle", "rb") as f:
        block_action = pickle.load(f)
    a = [str(k) for k in block_action["actions"]]
    a = "\n".join(a)
    print(start_fids[i])
    print(a)
    resources = np.load(path + "best_resources.npy")
    const = naive_constant(repeater_length=8, start_fid=start_fid, target_fid=0.9)
    plt.scatter(np.arange(1, len(resources) + 1), resources, s=20)
    plt.yscale("log")
    plt.axhline(y=const, color='r')
    plt.title("starting fids: " + str(start_fid) + " ; best solution found")
    plt.ylabel("resources used")
    plt.xlabel("trial number")
    plt.show()
    resource_list = np.loadtxt(path + "resource_list.txt")
    plt.hist(resource_list, bins=30)
    plt.axvline(x=const, color='r')
    ax = plt.gca()
    ax.ticklabel_format(axis="x", style="sci")
    plt.ylabel("number of agents")
    plt.xlabel("resources of found solution")
    plt.title("starting fids = " + str(start_fid))
    plt.show()


exit()
for i in range(2, 9):
    print(i)
    with open("results/best_history_%d.pickle" % i, "rb") as f:
        history = pickle.load(f)
    # print(history)
    with open("results/block_action_%d.pickle" % i, "rb") as f:
        block_action = pickle.load(f)
    a = [str(k) for k in block_action["actions"]]
    a = "\n".join(a)
    print(a)
    resources = np.load("results/best_resources_%d.npy" % i)
    aux += [resources[-1]]
    plt.scatter(np.arange(1, len(resources) + 1), resources, s=20)
    plt.yscale("log")
    plt.axhline(y=naive_constants[i], color="r")
    # if naive_constants[i] != found_constants[i]:
    #     plt.axhline(y=found_constants[i], color="b")
    plt.title("repeater length: " + str(i) + " ; best solution found")
    plt.ylabel("resources used")
    plt.xlabel("trial number")
    plt.show()

print(aux)


for i in range(2, 9):
    a = np.loadtxt("results/resource_list_%d.txt" % i)
    plt.hist(a, bins=30)
    plt.ylabel("number of agents")
    plt.xlabel("resources of found solution")
    plt.title("repeater_length = " + str(i))
    plt.show()