"""Run the scaling environment using blocks of different scales."""
import os, sys; sys.path.insert(0, os.path.abspath("."))
import environments.scaling_repeater_delegated_env as sr
from environments.scaling_repeater_delegated_env import TaskEnvironment as Env
from agents.ps_agent_changing_actions import ChangingActionsPSAgent
from general_interaction import Interaction
import numpy as np
from time import time
# import matplotlib.pyplot as plt
from multiprocessing import Pool
import pickle
import traceback
from warnings import warn
import itertools as it

num_processes = 48
num_agents = 128
repeater_length = 8
# allowed_block_lengths = [i for i in range(2, repeater_length)]
allowed_block_lengths = [2, 3, 4]
p_gates = 1.0
eta = 0
target_fid = 0.9
result_path = "results/scaling_delegated/"


def assert_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def merge_collections(*args):
    # ideally, we would choose the BEST solution so far
    my_dict = {}
    for collection in args:
        my_dict.update(collection.solution_dict)
    return SolutionCollection(initial_dict=my_dict)


def generate_constant(repeater_length, start_fid, working_fidelity, target_fidelity):
    env = Env(length=repeater_length, start_fid=start_fid, target_fid=target_fidelity, p=p_gates)
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


def naive_constant(repeater_length, start_fid, target_fid):
    working_fids = np.arange(0.85, 0.999, 0.001)
    my_constant = np.min([generate_constant(repeater_length, start_fid, wf, target_fid) for wf in working_fids])
    return my_constant


def distance(first, second):
    return sum([abs(x - y) for x, y in zip(first, second)])


def all_smaller(first, second):
    return all([x < y for x, y in zip(first, second)])


def resources_from_block_action(start_fid, action_sequence):
    rep_length = len(start_fid)
    env = Env(length=rep_length, start_fid=start_fid, target_fid=target_fid, p=p_gates)
    for action in action_sequence:
        action_index = env.action_list.index(action)
        env.move(action_index)
    return env.get_resources


class SolutionCollection(object):
    def __init__(self, initial_dict={}):
        self.solution_dict = initial_dict

    def get_block_action(self, fid_list):
        dict_key = tuple((int(fid * 100) for fid in fid_list))  # getting actions is rounded down
        try:
            return self.solution_dict[dict_key]  # raises KeyError if the solution is not present
        except KeyError:
            smaller_keys = filter(lambda x: len(x) == len(dict_key) and all_smaller(x, dict_key), self.solution_dict)
            try:
                dict_key = min(smaller_keys, key=lambda x: distance(x, dict_key))
            except ValueError as e:
                return None
                # my_str = "\n"
                # my_str += "fid_list: " + str(fid_list) + "\n"
                # my_str += "smaller_keys: " + str(fid_list) + "\n"
                # my_str += "dict_key: " + str(dict_key) + "\n"
                # my_str += "all keys: " + str(self.solution_dict.keys()) + "\n"
                # raise type(e)(e.message + my_str)
            return self.solution_dict[dict_key]

    def add_block_action(self, fid_list, action_list):
        key = tuple((int(fid * 100) for fid in fid_list))
        if key in self.solution_dict:
            old_action_list = self.solution_dict[key]
            if resources_from_block_action(fid_list, action_list) >= resources_from_block_action(fid_list, old_action_list):
                return
            else:  # if it uses fewer results, overwrite solution
                warn("Found a better solution for " + str(key))
        self.solution_dict[key] = action_list

    def save(self, destination):
        with open(destination, "wb") as f:
            pickle.dump(self.solution_dict, f)

    def load(self, origin):
        with open(origin, "rb") as f:
            self.solution_dict = pickle.load(f)


def setup_interaction(repeater_length, solution_collection, start_fid):
    # print("No. of collected actions:" + str(len(collection)))
    reward_constant = naive_constant(repeater_length, start_fid, target_fid)
    env = Env(length=repeater_length, available_block_lengths=allowed_block_lengths, start_fid=start_fid, target_fid=target_fid, p=p_gates, reward_constant=reward_constant, reward_exponent=2, delegated_solutions=solution_collection)
    agent = ChangingActionsPSAgent(env.n_base_actions, 0, eta, "softmax", 1, "dense", reset_glow=True)
    interaction = Interaction(agent=agent, environment=env)
    return interaction


def run(aux):
    try:
        np.random.seed()
        repeater_length = aux[0]
        solution_collection = aux[1]
        q_initial = aux[2]
        interaction = setup_interaction(repeater_length, solution_collection, q_initial)
        res = interaction.single_learning_life(10000, 500, True, env_statistics={"resources": interaction.env.get_resources})
        return interaction, res
    except Exception:
        print("Exception occured in child process")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    start_time = time()
    assert_dir(result_path)
    sc = SolutionCollection()
    # sc.save(result_path + "/solution_collection.pickle")  # to reset everything
    # exit()
    sc.load(result_path + "/solution_collection.pickle")
    # fids = np.arange(0.55, 1.00, 0.05)
    # fids = np.arange(0.6, 1.00, 0.10)
    # start_fids = it.product(fids, repeat=repeater_length
    start_fids = [(0.7,) * 8, (0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6)]
    for i, start_fid in enumerate(start_fids):
        config_path = result_path + "length%d_%d/" % (repeater_length, i)
        assert_dir(config_path)
        print(start_fid)
        aux = [(repeater_length, sc, start_fid) for i in range(num_agents)]
        p = Pool(processes=num_processes)
        interactions, res_list = zip(*p.map(run, aux))
        p.close()
        p.join()
        resource_list = [res["resources"][-1] for res in res_list]
        np.savetxt(config_path + "resource_list.txt", resource_list)
        try:
            min_index = np.nanargmin(resource_list)  # because unsuccessful agents will return NaN
        except ValueError:  # if all values are NaN
            print("No solutions were found for repeater length %d and initial fidelities " % repeater_length + str(start_fid))
            continue
        min_resource = resource_list[min_index]
        # add best block to next iteration
        best_env = interactions[min_index].env
        best_history = res_list[min_index]["last_trial_history"]
        block_action = best_env.composite_action_from_history(best_history)
        action_sequence = block_action["actions"]
        sc.add_block_action(fid_list=start_fid, action_list=action_sequence)  # save for later use
        best_resources = res_list[min_index]["resources"]
        np.save(config_path + "best_resources.npy", best_resources)
        with open(config_path + "block_action.pickle", "wb") as f:
            pickle.dump(block_action, f)
        with open(config_path + "best_history.pickle", "wb") as f:
            pickle.dump(best_history, f)

    sc.save(result_path + "/solution_collection.pickle")
    print("Repeater length %d took %.2f minutes." % (repeater_length, (time() - start_time) / 60.0))

# # plot best resource curve
# best_resources = res_list[min_index]["resources"]
# np.save("results/best_resources_%d.npy" % repeater_length, best_resources)
# with open("results/block_action_%d.pickle" % repeater_length, "wb") as f:
#     pickle.dump(block_action, f)
# with open("results/best_history_%d.pickle" % repeater_length, "wb") as f:
#     pickle.dump(best_history, f)
# print("repeater length %d took %.2f minutes" % (repeater_length, (time() - start_time) / 60))
# # plt.scatter(np.arange(1, len(best_resources) + 1), best_resources, s=20)
# # plt.yscale("log")
# # plt.axhline(y=reward_constants[repeater_length], color="r")
# # plt.title("repeater length:" + str(repeater_length))
# # plt.show()


# exit()
#
#



# auxlist = []
# for i in range(2, 9):
#     ffs = np.arange(0.85, 0.999, 0.001)
#     y = [generate_constant(i, q_initial, j, 0.9) for j in ffs]
#     print(i, np.min(y))
#     auxlist += [np.min(y)]
#     # plt.plot(ffs, y)
#     # plt.title(str(i))
#     # plt.show()
# print(auxlist)
