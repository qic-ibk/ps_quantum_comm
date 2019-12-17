from __future__ import division, print_function
import os, sys; sys.path.insert(0, os.path.abspath("."))
from time import time
# from environments.teleportation_variant_1 import TaskEnvironment as TeleportationVariantEnv
from environments.teleportation_variant_2 import TaskEnvironment as TeleportationVariantEnv
from agents.ps_agent_changing_actions import ChangingActionsPSAgent
from general_interaction import Interaction
from multiprocessing import Pool
import numpy as np
import os
import matplotlib.pyplot as plt


# eta = 0.3
# n_trials = 10**6

# env = TeleportationVariantEnv()
# agent = ChangingActionsPSAgent(env.n_actions, ps_gamma=0, ps_eta=eta, policy_type="softmax", ps_alpha=1, brain_type="dense", reset_glow=True)
# interaction = Interaction(agent=agent, environment=env)
# res = interaction.single_learning_life(n_trials=n_trials, max_steps_per_trial=50, verbose_trial_count=True, live_display=True)
# print(res)
# plt.ioff()
# plt.show()

num_processes = 64  # change according to cluster computer you choose
num_agents = 500
# etas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
etas = [0.3]
n_trials = 5 * 10**6
result_path = "results/teleportation/variant2/raw/"
sparsity = 50


def run_teleportation(i, eta, label_multiplicator=10, sparsity=20):
    np.random.seed()
    env = TeleportationVariantEnv()
    agent = ChangingActionsPSAgent(env.n_actions, ps_gamma=0, ps_eta=eta, policy_type="softmax", ps_alpha=1, brain_type="dense", reset_glow=True)
    interaction = Interaction(agent=agent, environment=env)
    res = interaction.single_learning_life(n_trials=n_trials, max_steps_per_trial=50)
    learning_curve = res["learning_curve"]
    success_list = np.ones(len(learning_curve), dtype=np.int)
    success_list[learning_curve == 0] = 1
    learning_curve[learning_curve == 0] = 10**-4
    step_curve = learning_curve**-1
    if sparsity != 1:
        step_curve = step_curve[0::sparsity]
        success_list = success_list[0::sparsity]
    # np.savetxt("results/eta_%d/step_curve_%d.txt" % (eta * label_multiplicator, i), step_curve, fmt="%.5f")
    np.save(result_path + "eta_%d/step_curve_%d.npy" % (eta * label_multiplicator, i), step_curve)
    np.savetxt(result_path + "eta_%d/success_list_%d.txt" % (eta * label_multiplicator, i), success_list, fmt="%-d")


def get_label_multiplicator(eta):
    return 10**(len(str(eta)) - 2)


class RunCallable(object):  # this solution is necessary because only top-level objects can be pickled
    def __init__(self, eta):
        self.eta = eta

    def __call__(self, i):
        return run_teleportation(i, self.eta, label_multiplicator=get_label_multiplicator(self.eta), sparsity=sparsity)


# def callback_error(result):
#     print('error', result)


if __name__ == "__main__":
    start_time = time()
    p = Pool(processes=num_processes)
    for eta in etas:
        if not os.path.exists(result_path + "eta_%d/" % (eta * get_label_multiplicator(eta))):
            os.makedirs(result_path + "eta_%d/" % (eta * get_label_multiplicator(eta)))
        my_callable = RunCallable(eta)
        p.map_async(my_callable, np.arange(num_agents))
    p.close()
    p.join()
    print("The whole script took %.2f minutes." % ((time() - start_time) / 60))
