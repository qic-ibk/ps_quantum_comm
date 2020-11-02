"""
Copyright 2020 Julius Wallnöfer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import os

num_agents = 100
# sparsity = 10
result_path = "results/epp_modified_depolarized/raw/"
plot_path = "results/epp_modified_depolarized/plot_ready/"


def assert_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    assert_dir(result_path)
    assert_dir(plot_path)
    reward_curves = [np.load(result_path + "reward_curve_%d.npy" % i) for i in range(num_agents)]
    # average_curve = np.sum(reward_curves, axis=0) / num_agents
    # np.savetxt(plot_path + "average_reward_sparsity_%d.txt" % sparsity, average_curve, fmt="%.6f")
    # plt.plot(np.arange(1, len(average_curve) * sparsity + 1, sparsity), average_curve)
    # plt.xlabel("Number of trials")
    # plt.ylabel("Average reward")
    # plt.savefig(plot_path + "average_reward.png")
    # plt.show()
    found_rewards = [reward_curve[-1] for reward_curve in reward_curves]
    non_zero = np.nonzero(found_rewards)[0]
    for reward_index in non_zero:
        print("Reward: " + str(found_rewards[reward_index]))
        with open(result_path + "last_trial_history_%d.txt" % reward_index, "r") as f:
            for line in f:
                print(line, end="")
    np.savetxt(plot_path + "found_rewards.txt", found_rewards, fmt="%.6f")
    plt.hist(found_rewards, bins=np.arange(-0.005, 1.015, 0.01))
    plt.xlabel("Reward")
    plt.ylabel("Number of agents")
    plt.savefig(plot_path + "found_rewards_hist.png")
    plt.show()
