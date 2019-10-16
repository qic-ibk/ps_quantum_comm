import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pickle
import matplotlib.pyplot as plt
from run.aux_scaling_memoryerrors import naive_constant, SolutionCollection
from run.aux_scaling_delegated import naive_constant as naive_constant_nomemory

# start_fids = [(0.7,) * 8, (0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6), (0.95, 0.9, 0.6, 0.9, 0.95, 0.95, 0.9, 0.6)]
# ps = np.linspace(0.98, 1.0, num=21)
results_path = "results/scaling_memoryerrors_positions/raw/"
output_path = "results/scaling_memoryerrors_positions/plot_ready/"


def assert_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


assert_dir(output_path)

num_possible_stations = 7
# first: cases with one additional station

resource_list = []
for i in range(num_possible_stations):
    resources = np.load(results_path + "p_gates990_alpha10/length2_" + str(i) + "/best_resources.npy")
    resource_list += [resources[-1]]
    # with open(results_path + "p_gates990_alpha10/length2_" + str(i) + "/" + "best_history.pickle", "rb") as f:  # best history is actually quite useless like this - we need actions instead of action_indices
    #     history = pickle.load(f)
    # print(history)

plt.scatter(np.arange(1, num_possible_stations + 1), resource_list)
plt.yscale("log")
plt.grid()
plt.xlabel("Position of intermediate station")
plt.ylabel("Resources used")
plt.savefig(output_path + "one_station.png")
plt.show()

resource_list = []
# then: cases with two additional stations
for i in range(21):
    resources = np.load(results_path + "p_gates990_alpha10/length3_" + str(i) + "/best_resources.npy")
    resource_list += [resources[-1]]
    # if i == 12:
    #     with open(results_path + "p_gates990_alpha10/length3_" + str(i) + "/" + "best_history.pickle", "rb") as f:  # best history is actually quite useless like this - we need actions instead of action_indices
    #         history = pickle.load(f)
    #     hist = [(i, j) for i, j, _ in history]
    #     sc = SolutionCollection()
    #     sc.load("results/scaling_memoryerrors/raw/" + "p_gates" + str(int(0.99 * 1000)) + "_alpha" + str(int(10)) + "/" + "solution_collection.pickle")
    #     deleg = sc.get_block_action((0.863, 0.757))
    #
    #     print("\n".join(str(d) for d in deleg))
    #     print("%%===================%%")
    #     print("\n".join(str(h) for h in history))


plt.scatter(np.arange(2, num_possible_stations + 1), resource_list[0:6], c="C0", label="pos1 = 1")
plt.scatter(np.arange(3, num_possible_stations + 1), resource_list[6:11], c="C1", label="pos1 = 2")
plt.scatter(np.arange(4, num_possible_stations + 1), resource_list[11:15], c="C2", label="pos1 = 3")
plt.scatter(np.arange(5, num_possible_stations + 1), resource_list[15:18], c="C3", label="pos1 = 4")
plt.scatter(np.arange(6, num_possible_stations + 1), resource_list[18:20], c="C4", label="pos1 = 5")
plt.scatter([7], [resources[-1]], c="C5", label="pos1 = 6")
plt.legend()
plt.yscale("log")
plt.grid()
plt.xlabel("Position of second station")
plt.ylabel("Resources used")
plt.savefig(output_path + "two_stations.png")
plt.show()

positions_list = [(i, j) for i in range(1, num_possible_stations) for j in range(i + 1, num_possible_stations + 1)]
aux_list = zip(positions_list, resource_list)
mytable = np.zeros((num_possible_stations - 1, num_possible_stations - 1), dtype=np.float)
for position, resource in aux_list:
    mytable[position[0] - 1, position[1] - 2] = resource
np.savetxt(output_path + "2position_table.txt", mytable, fmt="%.2e")

# finally: 3 additional stations
resource_list = []
for i in range(35):
    resources = np.load(results_path + "p_gates990_alpha10/length4_" + str(i) + "/best_resources.npy")
    resource_list += [resources[-1]]

positions_list = [(i, j, k) for i in range(1, num_possible_stations - 1) for j in range(i + 1, num_possible_stations) for k in range(j + 1, num_possible_stations + 1)]

aux_list = zip(positions_list, resource_list)
sorted_list = sorted(aux_list, key=lambda x: x[1])
# print("\n".join(str(x) for x in sorted_list))
with open(output_path + "three_stations_sorted.txt", "w") as f:
    f.write("\n".join(str(x) for x in sorted_list))
