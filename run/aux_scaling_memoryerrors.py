"""Run the scaling environment using blocks of different scales."""
import os, sys; sys.path.insert(0, os.path.abspath("."))
import environments.scaling_repeater_delegated_memoryerrors_env as sr
from environments.scaling_repeater_delegated_memoryerrors_env import TaskEnvironment as Env
from agents.ps_agent_changing_actions import ChangingActionsPSAgent
from general_interaction import Interaction
import numpy as np
from time import time
# import matplotlib.pyplot as plt
from multiprocessing import Pool
import pickle
import traceback
from warnings import warn


def assert_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def merge_collections(*args):
    # ideally, we would choose the BEST solution so far
    my_dict = {}
    for collection in args:
        my_dict.update(collection.solution_dict)
    return SolutionCollection(initial_dict=my_dict)


def generate_constant(repeater_length, start_fid, working_fidelity, target_fidelity, p_gates):
    env = Env(length=repeater_length, start_fid=start_fid, target_fid=target_fidelity, p=p_gates)
    while len(env.state) > 1:
        # purify all pairs to working fidelity
        for i, pair in enumerate(env.state):
            action_index = env.action_list.index(sr._Action(sr.ACTION_PURIFY, pair.stations))
            while env.state[i].fid < working_fidelity:
                fid_before = env.state[i].fid
                env.move(action_index)
                if env.state[i].fid <= fid_before:
                    return np.nan
        # ent_swap the shortest distance
        stations = []
        distances = []
        for i, pair in enumerate(env.state[:-1]):
            stations += [pair.right_station]
            other_pair = env.state[i + 1]
            distances += [other_pair.right_station - pair.left_station]
        station_index = np.argmin(distances)
        station = stations[station_index]
        action_index = env.action_list.index(sr._Action(sr.ACTION_SWAP, station))
        env.move(action_index)
    # final purification
    while env.state[0].fid < target_fidelity:
        action_index = env.action_list.index(sr._Action(sr.ACTION_PURIFY, env.state[0].stations))
        fid_before = env.state[0].fid
        env.move(action_index)
        if env.state[0].fid <= fid_before:
            return np.nan
    return env.get_resources()


def naive_constant(repeater_length, start_fid, target_fid, p_gates):
    working_fids = np.arange(0.85, 0.999, 0.001)
    my_constant = np.nanmin([generate_constant(repeater_length, start_fid, wf, target_fid, p_gates) for wf in working_fids])
    return my_constant


def distance(first, second):
    return sum([abs(x - y) for x, y in zip(first, second)])


def all_smaller(first, second):
    return all([x <= y for x, y in zip(first, second)])


def resources_from_block_action(start_fid, action_sequence, target_fid, p_gates):
    rep_length = len(start_fid)
    env = Env(length=rep_length, start_fid=start_fid, target_fid=target_fid, p=p_gates)
    for action in action_sequence:
        action_index = env.action_list.index(action)
        env.move(action_index)
    return env.get_resources()


class SolutionCollection(object):
    def __init__(self, initial_dict={}):
        self.solution_dict = dict(initial_dict)  # make a copy, because we will do in-place operations

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

    def add_block_action(self, fid_list, action_list, target_fid, p_gates):
        key = tuple((int(fid * 100) for fid in fid_list))
        if key in self.solution_dict:
            old_action_list = self.solution_dict[key]
            if resources_from_block_action(fid_list, action_list, target_fid, p_gates) >= resources_from_block_action(fid_list, old_action_list, target_fid, p_gates):
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


def setup_interaction_fids(start_fid, solution_collection, target_fid, allowed_block_lengths, p_gates, memory_alpha):
    repeater_length = len(start_fid)
    # reward_constant = naive_constant(repeater_length, start_fid, target_fid, p_gates)
    reward_constant = np.nan
    if np.isnan(reward_constant):
        reward_constant = np.finfo(np.float32).max  # this will give the duty of determining the constant solely to the environemnt
    env = Env(start_fid=start_fid, available_block_lengths=allowed_block_lengths, target_fid=target_fid, p=p_gates, alpha=memory_alpha, reward_constant=reward_constant, reward_exponent=2, delegated_solutions=solution_collection)
    agent = ChangingActionsPSAgent(n_actions=env.n_base_actions, ps_gamma=0, ps_eta=0, policy_type="softmax", ps_alpha=1, brain_type="dense", reset_glow=True)
    interaction = Interaction(agent=agent, environment=env)
    return interaction


def setup_interaction_distances(repeater_distances, solution_collection, target_fid, allowed_block_lengths, p_gates, memory_alpha):
    repeater_length = len(repeater_distances)
    # reward_constant = naive_constant(repeater_length, start_fid, target_fid, p_gates)
    reward_constant = np.nan
    if np.isnan(reward_constant):
        reward_constant = np.finfo(np.float32).max  # this will give the duty of determining the constant solely to the environemnt
    env = Env(repeater_distances=repeater_distances, available_block_lengths=allowed_block_lengths, target_fid=target_fid, p=p_gates, alpha=memory_alpha, reward_constant=reward_constant, reward_exponent=2, delegated_solutions=solution_collection)
    agent = ChangingActionsPSAgent(n_actions=env.n_base_actions, ps_gamma=0, ps_eta=0, policy_type="softmax", ps_alpha=1, brain_type="dense", reset_glow=True)
    interaction = Interaction(agent=agent, environment=env)
    return interaction


def run_fids(aux):
    try:
        np.random.seed()
        start_fid = aux[0]
        solution_collection = aux[1]
        target_fid = aux[2]
        num_trials = aux[3]
        allowed_block_lengths = aux[4]
        p_gates = aux[5]
        memory_alpha = aux[6]
        interaction = setup_interaction_fids(start_fid, solution_collection, target_fid, allowed_block_lengths, p_gates, memory_alpha)
        res = interaction.single_learning_life(num_trials, 500, True, env_statistics={"resources": interaction.env.get_resources})
        return interaction, res
    except Exception:
        print("Exception occured in child process")
        traceback.print_exc()
        raise


def run_distances(aux):
    try:
        np.random.seed()
        repeater_distances = aux[0]
        solution_collection = aux[1]
        target_fid = aux[2]
        num_trials = aux[3]
        allowed_block_lengths = aux[4]
        p_gates = aux[5]
        memory_alpha = aux[6]
        interaction = setup_interaction_distances(repeater_distances, solution_collection, target_fid, allowed_block_lengths, p_gates, memory_alpha)
        res = interaction.single_learning_life(num_trials, 500, True, env_statistics={"resources": interaction.env.get_resources})
        return interaction, res
    except Exception:
        print("Exception occured in child process")
        traceback.print_exc()
        raise


def run_scaling_fids(num_processes, num_agents, num_trials, start_fids, allowed_block_lengths, p_gates, memory_alpha, target_fid, result_path):
    start_time = time()
    assert_dir(result_path)
    sc = SolutionCollection()
    try:
        sc.load(result_path + "/solution_collection.pickle")
    except IOError:
        warn("SolutionCollection not found - creating new one.")
    for i, start_fid in enumerate(start_fids):
        repeater_length = len(start_fid)
        config_path = result_path + "length%d_%d/" % (repeater_length, i)
        assert_dir(config_path)
        # print(start_fid)
        aux = [(start_fid, sc, target_fid, num_trials, allowed_block_lengths, p_gates, memory_alpha) for i in range(num_agents)]
        p = Pool(processes=num_processes)
        interactions, res_list = zip(*p.map(run_fids, aux))
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
        sc.add_block_action(fid_list=start_fid, action_list=action_sequence, target_fid=target_fid, p_gates=p_gates)  # save for later use
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


def run_scaling_distances(num_processes, num_agents, num_trials, distances, allowed_block_lengths, p_gates, memory_alpha, target_fid, result_path):
    start_time = time()
    assert_dir(result_path)
    sc = SolutionCollection()
    try:
        sc.load(result_path + "/solution_collection.pickle")
    except IOError:
        warn("SolutionCollection not found - creating new one.")
    for i, repeater_distances in enumerate(distances):
        repeater_length = len(repeater_distances)
        config_path = result_path + "length%d_%d/" % (repeater_length, i)
        assert_dir(config_path)
        # print(start_fid)
        aux = [(repeater_distances, sc, target_fid, num_trials, allowed_block_lengths, p_gates, memory_alpha) for i in range(num_agents)]
        p = Pool(processes=num_processes)
        interactions, res_list = zip(*p.map(run_distances, aux))
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
        # action_sequence = block_action["actions"]
        # sc.add_block_action(fid_list=start_fid, action_list=action_sequence, target_fid=target_fid, p_gates=p_gates)  # save for later use
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
