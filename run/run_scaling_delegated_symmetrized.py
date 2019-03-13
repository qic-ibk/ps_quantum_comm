"""Run the scaling environment using blocks of different scales."""
import os, sys; sys.path.insert(0, os.path.abspath("."))
import environments.scaling_repeater_delegated_symmetrized as sr
from environments.scaling_repeater_delegated_symmetrized import TaskEnvironment as Env
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
num_trials = 10000
repeater_length = 2
allowed_block_lengths = []
p_gates = 0.99
eta = 0
target_fid = 0.9
result_path = "results/scaling_delegated_symmetrized/p_gates99/"


def assert_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def merge_collections(*args):
    # ideally, we would choose the BEST solution so far
    my_dict = {}
    for collection in args:
        my_dict.update(collection.solution_dict)
    return SolutionCollection(initial_dict=my_dict)


def generate_constant(repeater_length, start_fid, working_fidelity, target_fidelity, p=p_gates):
    env = Env(length=repeater_length, start_fid=start_fid, target_fid=target_fidelity, p=p)
    while len(env.state) > 1:
        # purify all pairs to working fidelity
        action_index = env.action_list.index(sr._Action(sr.ACTION_PURIFY))
        while env.state[0].fid < working_fidelity:
            fid_before = env.state[0].fid
            env.move(action_index)
            if env.state[0].fid <= fid_before:
                return np.nan
        action_index = env.action_list.index(sr._Action(sr.ACTION_SWAP))
        env.move(action_index)
    while env.state[0].fid < target_fidelity:
        action_index = env.action_list.index(sr._Action(sr.ACTION_PURIFY))
        fid_before = env.state[0].fid
        env.move(action_index)
        if env.state[0].fid <= fid_before:
            return np.nan
    return env.get_resources()


def naive_constant(repeater_length, start_fid, target_fid, p=p_gates):
    working_fids = np.arange(0.85, 0.999, 0.001)
    my_constant = np.nanmin([generate_constant(repeater_length, start_fid, wf, target_fid, p) for wf in working_fids])
    return my_constant


def resources_from_block_action(start_fid, block_size, action_sequence):
    rep_length = block_size
    env = Env(length=rep_length, start_fid=start_fid, target_fid=target_fid, p=p_gates)
    for action in action_sequence:
        action_index = env.action_list.index(action)
        env.move(action_index)
    return env.get_resources()


class SolutionCollection(object):
    def __init__(self, initial_dict={}):
        self.solution_dict = initial_dict

    def get_block_action(self, fid, block_size):
        dict_key = (int(fid * 100), block_size)  # getting actions is rounded down
        try:
            return self.solution_dict[dict_key]  # raises KeyError if the solution is not present
        except KeyError:
            smaller_keys = filter(lambda x: x[1] == block_size and x[0] <= dict_key[0], self.solution_dict)
            try:
                dict_key = max(smaller_keys, key=lambda x: x[0])
            except ValueError:
                return None
                # my_str = "\n"
                # my_str += "fid_list: " + str(fid_list) + "\n"
                # my_str += "smaller_keys: " + str(fid_list) + "\n"
                # my_str += "dict_key: " + str(dict_key) + "\n"
                # my_str += "all keys: " + str(self.solution_dict.keys()) + "\n"
                # raise type(e)(e.message + my_str)
            return self.solution_dict[dict_key]

    def add_block_action(self, fid, block_size, action_list):
        key = (int(fid * 100), block_size)
        if key in self.solution_dict:
            old_action_list = self.solution_dict[key]
            if resources_from_block_action(fid, block_size, action_list) >= resources_from_block_action(fid, block_size, old_action_list):
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
    if np.isnan(reward_constant):
        reward_constant = np.finfo(np.float32).max  # this will give the duty of determining the constant solely to
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
        res = interaction.single_learning_life(num_trials, 500, True, env_statistics={"resources": interaction.env.get_resources})
        return interaction, res
    except Exception:
        print("Exception occured in child process")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    start_time = time()
    assert_dir(result_path)
    sc = SolutionCollection()
    try:
        sc.load(result_path + "/solution_collection.pickle")
    except IOError:
        warn("SolutionCollection not found - creating new one.")
    start_fids = np.arange(0.6, 1.00, 0.05)
    # start_fids = np.arange(0.6, 1.00, 0.10)
    # start_fids = it.product(fids, repeat=repeater_length)
    # start_fids = [(0.7,) * 8, (0.8, 0.6, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6)]
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
        sc.add_block_action(fid=start_fid, block_size=repeater_length, action_list=action_sequence)  # save for later use
        best_resources = res_list[min_index]["resources"]
        np.save(config_path + "best_resources.npy", best_resources)
        with open(config_path + "block_action.pickle", "wb") as f:
            pickle.dump(block_action, f)
        # add action also before saving, because action index is not very informative
        best_history = [(observation, action_index, best_env.action_list[action_index]) for observation, action_index in best_history]
        with open(config_path + "best_history.pickle", "wb") as f:
            pickle.dump(best_history, f)

    sc.save(result_path + "/solution_collection.pickle")
    print("Repeater length %d took %.2f minutes." % (repeater_length, (time() - start_time) / 60.0))
