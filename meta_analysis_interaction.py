"""
"""
from __future__ import print_function, division
from copy import deepcopy
import numpy as np
from environments.libraries import matrix as mat
from environments.epp_env import EPPEnv


def fidelity(rho):
    fid = np.dot(np.dot(mat.H(mat.phiplus), rho), mat.phiplus)
    return float(fid[0, 0])


class PartialMultiverseTrial(object):
    """
    """
    def __init__(self, agent, environment):
        self.agent = agent
        self.env = environment
        self.history_of_branch = []


class MetaAnalysisInteraction(object):
    """
    """
    def __init__(self, agent, environment):
        self.primary_agent = agent
        self.primary_env = environment
        self.partial_trial_list = []

    def run_branch_until_finished(self, partial_multiverse_trial, observation, reward, episode_finished, info):
        agent = partial_multiverse_trial.agent
        env = partial_multiverse_trial.env
        episode_finished = episode_finished
        while not episode_finished:
            action = agent.deliberate_and_learn(observation, reward, episode_finished, info)
            partial_multiverse_trial.history_of_branch += [(list(observation), action)]  # important to create copy of list, because environment may modify the observation in-place
            split, split_actions = env.is_splitting_action(action)
            if split is True:
                new_agent = deepcopy(agent)
                new_env = deepcopy(env)
                new_branch_action = split_actions[1]
                new_observation, new_reward, new_episode_finished, new_info = new_env.move(new_branch_action)
                new_partial_trial = PartialMultiverseTrial(new_agent, new_env)
                self.partial_trial_list += [new_partial_trial]
                self.run_branch_until_finished(new_partial_trial, new_observation, new_reward, new_episode_finished, new_info)  # haha, recursion!
            branch_action = split_actions[0]
            observation, reward, episode_finished, info = env.move(branch_action)
        return  # something

    def multiverse_reward(self, partial_trial_list, depolarize=False):
        # this method probably should belong to the environment instead, but it would be a bit weird with the environment creating new env objects
        accepted_branches = filter(lambda x: x.env.percept_now[-1] == 14, partial_trial_list)
        env_list = [branch.env for branch in accepted_branches]
        if env_list == []:  # if no branches were accepted, give no reward
            return 0
        accepted_actions_lists = [env.percept_now for env in env_list]
        probability = np.sum([env.branch_probability for env in env_list])
        new_state = np.sum([env.branch_probability * env.get_pair_state() for env in env_list], axis=0)
        new_state = new_state / np.trace(new_state)
        if depolarize is True:
            fid = fidelity(new_state)
            pp = (4 * fid - 1) / 3
            new_state = np.dot(mat.phiplus, mat.H(mat.phiplus))
            new_state = mat.wnoise(new_state, 0, pp)
        for i in range(1, 1):
            # if probability == 0:
            #     print(i, probability, new_state)
            input_state = mat.tensor(new_state, new_state)
            input_state = mat.reorder(input_state, [0, 2, 1, 3])
            for env, action_list in zip(env_list, accepted_actions_lists):
                env.reset(input_state=input_state)
                for action in action_list:
                    env.move(action)
            probability *= np.sum([env.branch_probability for env in env_list])
            new_state = np.sum([env.branch_probability * env.get_pair_state() for env in env_list], axis=0)
            new_state = new_state / np.trace(new_state)
            if depolarize is True:
                fid = fidelity(new_state)
                pp = (4 * fid - 1) / 3
                new_state = np.dot(mat.phiplus, mat.H(mat.phiplus))
                new_state = mat.wnoise(new_state, 0, pp)
        my_env = env_list[0]
        my_env.reset()
        initial_fidelity = fidelity(mat.ptrace(my_env.state, [1, 3]))
        # # if (fidelity(new_state) - initial_fidelity) > 0:
        # #     return 10
        # # else:
        # #     return 0
        # reward = probability / 10**-12 * max(fidelity(new_state) - initial_fidelity, 0)
        reward = probability * max(fidelity(new_state) - initial_fidelity, 0) / 0.6634 / (1 - 0.8154959300572808)
        return reward

    def merge_reward(self, reward):
        # note: this has weird behavior when glow is active because of the different length of solutions
        self.primary_agent._learning(reward)
        self.primary_agent.brain.reset_glow()
        for partial_trial in self.partial_trial_list[1:]:
            branch_history = partial_trial.history_of_branch
            # we need percepts instead of observations - this also makes sure that new percepts are all added correctly
            # we also need to recover primitive actions from the history
            # also, turns out, history_since_last_reward needs it in (action, observation) order... not the other way around
            # why, oh why?
            branch_history = [(self.primary_env.action_from_index(action), self.primary_agent._percept_preprocess(observation)) for observation, action in branch_history]
            self.primary_agent.history_since_last_reward = branch_history
            self.primary_agent._learning(reward)
            self.primary_agent.brain.reset_glow()

    def single_learning_life(self, n_trials, verbose_trial_count=False):
        reward_curve = np.zeros(n_trials)
        res = {}
        for i_trial in range(n_trials):
            reward = 0  # inside the loop because learning is handled manually here
            episode_finished = 0  # same
            info = {}
            self.partial_trial_list = []
            if verbose_trial_count and i_trial % 100 == 0:
                print("Interaction is now starting trial %d of %d." % (i_trial, n_trials))
            setup = self.primary_env.reset()
            if not isinstance(setup, tuple):
                observation = setup
            else:
                observation = setup[0]
                info.update(setup[1])
            partial_trial = PartialMultiverseTrial(self.primary_agent, self.primary_env)
            self.partial_trial_list += [partial_trial]
            self.run_branch_until_finished(partial_trial, observation, reward, episode_finished, info)
            # now the evaluating and updating part starts
            reward = self.multiverse_reward(self.partial_trial_list, depolarize=True)
            self.merge_reward(reward)
            reward_curve[i_trial] = reward
        res["reward_curve"] = reward_curve
        return res
