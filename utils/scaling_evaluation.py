import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pickle
import matplotlib.pyplot as plt


# reward_constants = [0, 0, 142, 764, 1309, 4427, 7001, 9328, 11997]  # for q=0.75
naive_constants = [0, 0, 55.530075111293556, 298.8066811742171, 511.3369612524592, 1729.5926924526652, 2735.132853627021, 3644.117803177942, 4686.472277498218]  # for q=0.8
found_constants = [0, 0, 55.530075111293556, 183.5589298854627, 502.57222565976156, 989.9185924406921, 1812.4840860508734, 2822.031940256553, 4177.7422676456845]

aux = [0, 0]

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
