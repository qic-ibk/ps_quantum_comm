from __future__ import division, print_function
from time import time
# from environments.teleportation_universal_env import TaskEnvironment as TeleportationUniversalEnv
from environments.teleportation_env import TaskEnvironment as TeleportationEnv
from agents.ps_agent_flexible_percepts import FlexiblePerceptsPSAgent
from general_interaction import Interaction
from multiprocessing import Pool
import numpy as np


def run_teleportation(i, eta, label_multiplicator=10):
    np.random.seed()
    env = TeleportationEnv()
    agent = FlexiblePerceptsPSAgent(env.n_actions, ps_gamma=0, ps_eta=eta, policy_type="softmax", ps_alpha=1, brain_type="dense")
    interaction = Interaction(agent_type="FlexiblePerceptsPSAgent", agent=agent, environment=env)
    res = interaction.single_learning_life(n_trials=60000, max_steps_per_trial=50)
    learning_curve = res["learning_curve"]
    learning_curve[learning_curve == 0] = 10000
    step_curve = learning_curve**-1
    np.savetxt("results/eta_%d/step_curve_%d.txt" % (eta * label_multiplicator, i), step_curve, fmt="%.5f")


def run01(i):
    run_teleportation(i, 0.1)


def run015(i):
    run_teleportation(i, 0.15, 100)


def run02(i):
    run_teleportation(i, 0.2)


def run025(i):
    run_teleportation(i, 0.25, 100)


def run03(i):
    run_teleportation(i, 0.3)


def run035(i):
    run_teleportation(i, 0.35, 100)


def run04(i):
    run_teleportation(i, 0.4)


def run045(i):
    run_teleportation(i, 0.45, 100)


def run05(i):
    run_teleportation(i, 0.5)


if __name__ == "__main__":
    start_time = time()
    p = Pool(processes=64)
    p.map_async(run015, np.arange(128))
    p.map_async(run01, np.arange(128))
    p.map_async(run02, np.arange(128))
    p.map_async(run03, np.arange(128))
    p.map_async(run04, np.arange(128))
    p.map_async(run05, np.arange(128))
    # p.map_async(run025, np.arange(128))
    # p.map_async(run035, np.arange(128))
    # p.map_async(run045, np.arange(128))
    p.close()
    p.join()
    print("The whole script took %.2f minutes." % ((time() - start_time) / 60))
