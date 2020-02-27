import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pickle
import matplotlib.pyplot as plt
from run.aux_scaling_memoryerrors import naive_constant
from run.aux_scaling_delegated import naive_constant as naive_constant_nomemory

# start_fids = [(0.7,) * 8, (0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6), (0.95, 0.9, 0.6, 0.9, 0.95, 0.95, 0.9, 0.6)]
# ps = np.linspace(0.98, 1.0, num=21)
results_path = "results/scaling_memoryerrors/raw/"
output_path = "results/scaling_memoryerrors/plot_ready/"


def assert_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


assert_dir(output_path)
# with open(output_path + "info.txt", "w") as f:
#     f.write("starting fidelities: " + str(start_fids) + "\n")
#     f.write("learning curves are plotted for p=0.99")
# np.savetxt(output_path + "ps.txt", ps)

alphas = [1, 2, 10]  # , 100, 1000]
colors = ["C0", "C1", "C2", "C3"]  # , "C4", "C5"]


# resource_list = []
# const_list = []
for i, alpha in enumerate(alphas):
    resources = np.load(results_path + "p_gates990_alpha%d/length8_0/best_resources.npy" % alpha)
    np.savetxt(output_path + "best_resources_alpha%d.txt" % alpha, resources)
    # resource_list += [resources[-1]]
    plt.scatter(np.arange(1, len(resources) + 1), resources, s=20, color=colors[i], label="τ = 1/%d s" % alpha)
    const = naive_constant(start_fid=(0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6), target_fid=0.9, p_gates=0.99, memory_alpha=alpha)
    # const_list += [const]
    plt.axhline(y=const, color=colors[i])
# also plot no memory errors
resources = np.load("results/scaling_delegated/raw/" + "p_gates990/length8_1/best_resources.npy")
np.savetxt(output_path + "best_resources_alpha0.txt", resources)
plt.scatter(np.arange(1, len(resources) + 1), resources, s=20, color=colors[-1], label="τ = ∞")
const_nomemory = naive_constant_nomemory(repeater_length=8, start_fid=(0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6), target_fid=0.9, p_gates=0.99)
plt.axhline(y=const_nomemory, color=colors[-1])

# print(naive_constant(start_fid=(0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6), target_fid=0.9, p_gates=0.99, memory_alpha=30))

plt.title("different memory times for: " + str((0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6)))
plt.ylabel("resources used")
plt.xlabel("trial number")
plt.yscale("log")
plt.ylim((10**9, 10**14))
plt.legend()
plt.savefig(output_path + "differing_alphas.png")
plt.show()


# ### now do the plot of many alphas
alphas = [1, 2, 11, 12, 14, 16, 20, 25, 26, 27, 28, 29, 30]
colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C0", "C1", "C2", "C3"]

resource_list = []
for i, alpha in enumerate(alphas):
    resources = np.load(results_path + "p_gates990_alpha%d/length8_0/best_resources.npy" % alpha)
    resource_list += [resources[-1]]
# also plot no memory errors
resources = np.load("results/scaling_delegated/raw/" + "p_gates990/length8_1/best_resources.npy")


plt.title("required resources for different memory times")
plt.ylabel("resources used")
plt.xlabel("memory quality 1/τ")
plt.yscale("log")
plt.scatter([0] + alphas, [resources[-1]] + resource_list)
# plt.scatter([0] + alphas, [const_nomemory] + const_list)
np.savetxt(output_path + "x_alphas.txt", [0] + alphas, fmt="%-d")
np.savetxt(output_path + "y_resources.txt", [resources[-1]] + resource_list)
plt.grid()
plt.savefig(output_path + "resources_per_alpha.png")
plt.show()
