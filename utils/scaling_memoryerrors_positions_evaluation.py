import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pickle
import matplotlib.pyplot as plt
from run.aux_scaling_memoryerrors import naive_constant
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
    best_history = pickle.load
    with open(results_path + "p_gates990_alpha10/length2_" + str(i) + "/" + "best_history.pickle", "rb") as f:  # best history is actually quite useless like this - we need actions instead of action_indices
        history = pickle.load(f)
    print(history)

plt.scatter(np.arange(1, num_possible_stations + 1), resource_list)
plt.show()


exit()


# with open(output_path + "info.txt", "w") as f:
#     f.write("starting fidelities: " + str(start_fids) + "\n")
#     f.write("learning curves are plotted for p=0.99")
# np.savetxt(output_path + "ps.txt", ps)





# with open(path + "p_gates990_alpha10/length8_0") as f:

resources = np.load(results_path + "p_gates990_alpha10/length8_0/best_resources.npy")
resources_alpha2 = np.load(results_path + "p_gates990_alpha20/length8_0/best_resources.npy")
resources_nomemory = np.load("results/scaling_delegated/raw/" + "p_gates990/length8_1/best_resources.npy")
const = naive_constant(start_fid=(0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6), target_fid=0.9, p_gates=0.99, memory_alpha=1)
const_alpha2 = naive_constant(start_fid=(0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6), target_fid=0.9, p_gates=0.99, memory_alpha=2)
const_nomemory = naive_constant_nomemory(repeater_length=8, start_fid=(0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6), target_fid=0.9, p_gates=0.99)
plt.scatter(np.arange(1, len(resources) + 1), resources, s=20, color="b", label="1s")
plt.scatter(np.arange(1, len(resources) + 1), resources_alpha2, s=20, color="r", label="0.5s")
plt.scatter(np.arange(1, len(resources) + 1), resources_nomemory, s=20, color="g", label="inf")
plt.yscale("log")
plt.axhline(y=const, color='b')
plt.axhline(y=const_alpha2, color="r")
plt.axhline(y=const_nomemory, color='g')
# plt.axhline(y=1, color='r')
plt.title("different memory times for: " + str((0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6)))
plt.ylabel("resources used")
plt.xlabel("trial number")
plt.legend()
plt.show()


# for i, start_fid in enumerate(start_fids):
#     print(start_fid)
#     compare = np.zeros(len(ps))
#     for j, p in enumerate(ps):
#         print("p=%.3f" % p)
#         p_path = results_path + "p_gates" + str(int(p * 1000)) + "/"
#         p_gates = p
#         path = p_path + "length8_%d/" % i
#         with open(path + "best_history.pickle", "rb") as f:  # best history is actually quite useless like this - we need actions instead of action_indices
#             history = pickle.load(f)
#         action_history = [action for _, _, action in history]
#         b = [str(k) for k in action_history]
#         b = "\n".join(b)
#         print(b)
#         with open(path + "block_action.pickle", "rb") as f:
#             block_action = pickle.load(f)
#         a = [str(k) for k in block_action["actions"]]
#         a = "\n".join(a)
#         # print(a)
#         resources = np.load(path + "best_resources.npy")
#         const = naive_constant(repeater_length=8, start_fid=start_fid, target_fid=0.9, p_gates=p_gates)
#         # resources = resources[:1000]
#         plt.scatter(np.arange(1, len(resources) + 1), resources, s=20)
#         plt.yscale("log")
#         plt.axhline(y=const, color='r')
#         # plt.title("starting fids: " + str(start_fid) + " ; best solution found")
#         plt.ylabel("resources used")
#         plt.xlabel("trial number")
#         if p == 0.99:
#             plt.savefig(output_path + "best_resources_p99_%d.png" % i)
#             np.savetxt(output_path + "best_resources_p99_%d.txt" % i, resources)
#             with open(output_path + "const_%d.txt" % i, "w") as f:
#                 f.write(str(const))
#         # plt.show()
#         plt.cla()
#         resource_list = np.loadtxt(path + "resource_list.txt")
#         try:
#             # resource_list = resource_list[np.logical_not(np.isnan(resource_list))]
#             # print(dict(zip(*np.unique(resource_list, return_counts=True))))
#             # print(resource_list)
#             plt.hist(resource_list, bins=300)
#         except IndexError:
#             plt.axvline(x=resource_list[0])
#         plt.axvline(x=const, color='r')
#         plt.xscale("log")
#         # ax = plt.gca()
#         # ax.ticklabel_format(axis="x", style="sci")
#         plt.ylabel("number of agents")
#         plt.xlabel("resources of found solution")
#         # plt.title("starting fids = " + str(start_fid))
#         # plt.show()
#         plt.cla()
#         min_resource = min(resource_list)
#         compare[j] = min_resource / const
#
#     plt.scatter(ps, compare)
#     # plt.title("starting fids: " + str(start_fid))
#     plt.xlabel("gate error parameter p")
#     plt.ylabel("resources relative to symmetric guess")
#     plt.grid()
#     plt.savefig(output_path + "comparison_by_error_%d.png" % i)
#     np.savetxt(output_path + "comparison_by_error_%d.txt" % i, compare)
#     plt.show()
