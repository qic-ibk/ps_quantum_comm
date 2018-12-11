from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import os

num_agents = 100
sparsity = 10
result_path = "results/epp/raw/"


def assert_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    assert_dir(result_path)
    reward_curves = [np.load(result_path + "reward_curve_%d.npy" % i) for i in range(num_agents)]
    average_curve = np.sum(reward_curves, axis=0) / num_agents
    plt.plot(np.arange(1, len(average_curve) * sparsity + 1, sparsity), average_curve)
    plt.xlabel("Number of trials")
    plt.ylabel("Average reward.")
    plt.show()
    found_rewards = [reward_curve[-1] for reward_curve in reward_curves]
    plt.hist(found_rewards, bins=30)
    plt.xlabel("Reward")
    plt.ylabel("Number of agents")
    plt.show()
