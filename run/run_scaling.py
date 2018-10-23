"""Run the scaling environment using blocks of different scales."""
import os, sys; sys.path.insert(0, os.path.abspath("."))
import environments.scaling_repeater_env as sr
from environments.scaling_repeater_env import TaskEnvironment as Env
from agents.ps_agent_changing_actions import ChangingActionsPSAgent
from general_interaction import Interaction
import numpy as np
from time import time
# import matplotlib.pyplot as plt
from multiprocessing import Pool
import pickle

num_processes = 64
# reward_constants = [0, 0, 8686, 23626, 78887, 261237, 226018, 404088, 712699]  # for default q=0.57
reward_constants = [55.530075111293556, 298.8066811742171, 511.3369612524592, 1729.5926924526652, 2735.132853627021, 3644.117803177942, 4686.472277498218]  # for q=0.8
# reward_constants = [0, 0, 142, 764, 1309, 4427, 7001, 9328, 11997]  # for q=0.75
collected_action = []
num_agents = 128
q_initial = 0.8
eta = 0


def setup_interaction(repeater_length, collection):
    # print("No. of collected actions:" + str(len(collection)))
    env = Env(length=repeater_length, composite_actions=collection, q=q_initial, reward_constant=reward_constants[repeater_length], reward_exponent=2)
    agent = ChangingActionsPSAgent(env.n_base_actions, 0, eta, "softmax", 1, "dense", reset_glow=True)
    interaction = Interaction(agent=agent, environment=env)
    return interaction


def run(aux):
    np.random.seed()
    repeater_length = aux[0]
    collection = aux[1]
    interaction = setup_interaction(repeater_length, collection)
    res = interaction.single_learning_life(10000, 500, True, env_statistics={"resources": interaction.env.get_resources})
    return interaction, res


for repeater_length in range(2, 9):
    start_time = time()
    aux = [(repeater_length, collected_action) for i in range(num_agents)]
    p = Pool(processes=num_processes)
    interactions, res_list = zip(*p.map(run, aux))
    resource_list = [res["resources"][-1] for res in res_list]
    p.close()
    p.join()
    np.savetxt("results/resource_list.txt", resource_list)
    exit()
    try:
        min_index = np.nanargmin(resource_list)  # because unsuccessful agents will return NaN
    except ValueError:  # if all values are NaN
        print("No solutions were found for repeater length %d" % repeater_length)
        continue
    min_resource = resource_list[min_index]
    if min_resource < reward_constants[repeater_length]:
        print("Repeater length %d found a solution with %.2f resources instead of %.2f" % (repeater_length, min_resource, reward_constants[repeater_length]))
    # add best block to next iteration
    best_env = interactions[min_index].env
    best_history = res_list[min_index]["last_trial_history"]
    block_action = best_env.composite_action_from_history(best_history)
    collected_action += [block_action]
    # plot best resource curve
    best_resources = res_list[min_index]["resources"]
    np.save("results/best_resources_%d.npy" % repeater_length, best_resources)
    with open("results/block_action_%d.pickle" % repeater_length, "wb") as f:
        pickle.dump(block_action, f)
    with open("results/best_history_%d.pickle" % repeater_length, "wb") as f:
        pickle.dump(best_history, f)
    print("repeater length %d took %.2f minutes" % (repeater_length, (time() - start_time) / 60))
    # plt.scatter(np.arange(1, len(best_resources) + 1), best_resources, s=20)
    # plt.yscale("log")
    # plt.axhline(y=reward_constants[repeater_length], color="r")
    # plt.title("repeater length:" + str(repeater_length))
    # plt.show()


exit()


def generate_constant(repeater_length, q_noise, working_fidelity, target_fidelity):
    env = Env(length=repeater_length, q=q_noise, target_fid=target_fidelity)
    while len(env.state) > 1:
        # purify all pairs to working fidelity
        for i, pair in enumerate(env.state):
            # print(i, "in purify to working")
            action_index = env.action_list.index(sr._Action(sr.ACTION_PURIFY, pair.stations))
            while env.state[i].fid < working_fidelity:
                env.move(action_index)
        # ent_swap the shortest distance
        stations = []
        distances = []
        for i, pair in enumerate(env.state[:-1]):
            # print(i, "in distance evaluation")
            stations += [pair.right_station]
            other_pair = env.state[i + 1]
            distances += [other_pair.right_station - pair.left_station]
        station_index = np.argmin(distances)
        station = stations[station_index]
        action_index = env.action_list.index(sr._Action(sr.ACTION_SWAP, station))
        env.move(action_index)
    while env.state[0].fid < target_fidelity:
        # print("final purification")
        action_index = env.action_list.index(sr._Action(sr.ACTION_PURIFY, env.state[0].stations))
        env.move(action_index)
    return env.get_resources()


auxlist = []
for i in range(2, 9):
    ffs = np.arange(0.85, 0.999, 0.001)
    y = [generate_constant(i, q_initial, j, 0.9) for j in ffs]
    print(i, np.min(y))
    auxlist += [np.min(y)]
    # plt.plot(ffs, y)
    # plt.title(str(i))
    # plt.show()
print(auxlist)
