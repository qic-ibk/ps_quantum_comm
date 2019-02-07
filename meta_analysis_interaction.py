"""
"""
from __future__ import print_function, division
from copy import deepcopy
import numpy as np
from environments.libraries import matrix as mat
from environments.epp_env import EPPEnv
from memory_profiler import profile


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

    # @profile
    def run_branch_until_finished(self, partial_multiverse_trial, observation, reward, episode_finished, info, file=None):
        agent = partial_multiverse_trial.agent
        env = partial_multiverse_trial.env
        episode_finished = episode_finished
        while not episode_finished:
            action = agent.deliberate_and_learn(observation, reward, episode_finished, info)
            # action = int(input(repr((observation, reward, episode_finished, info)) + "\n Action: "))  # debugging
            partial_multiverse_trial.history_of_branch += [(list(observation), action)]  # important to create copy of list, because environment may modify the observation in-place
            if file is not None:
                file.write(str(partial_multiverse_trial.history_of_branch[-1]) + "\n")
            split, split_actions = env.is_splitting_action(action)
            if split is True:
                if file is not None:
                    file.write("Branching off!" + "\n")
                new_agent = agent  # we don't need the very slow deepcopy here, because we do not use the built-in learning features of the agent
                new_env = deepcopy(env)
                new_branch_action = split_actions[1]
                new_observation, new_reward, new_episode_finished, new_info = new_env.move(new_branch_action)
                new_partial_trial = PartialMultiverseTrial(new_agent, new_env)
                self.partial_trial_list += [new_partial_trial]
                self.run_branch_until_finished(new_partial_trial, new_observation, new_reward, new_episode_finished, new_info, file=file)  # haha, recursion!
                if file is not None:
                    file.write(str("Back to initial branch." + "\n"))
            branch_action = split_actions[0]
            observation, reward, episode_finished, info = env.move(branch_action)
        return  # something

    # @profile(precision=4)
    def merge_reward(self, reward):
        # note: this has weird behavior when glow is active because of the different length of solutions
        self.primary_agent.history_since_last_reward = []  # just to make it explicit
        for partial_trial in self.partial_trial_list:
            branch_history = partial_trial.history_of_branch
            # we need percepts instead of observations - this also makes sure that new percepts are all added correctly
            # we also need to recover primitive actions from the history
            # also, turns out, history_since_last_reward needs it in (action, observation) order... not the other way around
            # why, oh why?
            branch_history = [(self.primary_env.action_from_index(action), self.primary_agent._percept_preprocess(observation)) for observation, action in branch_history]
            self.primary_agent.history_since_last_reward = branch_history
            self.primary_agent._learning(reward)
            self.primary_agent.brain.reset_glow()

    # @profile(precision=4)
    def single_learning_life(self, n_trials, verbose_trial_count=False, last_history_file=None):
        reward_curve = np.zeros(n_trials)
        res = {}
        for i_trial in range(n_trials):
            reward = 0  # inside the loop because learning is handled manually here
            episode_finished = 0  # same
            info = {}
            self.partial_trial_list = []
            if verbose_trial_count and i_trial % 1000 == 0:
                print("Interaction is now starting trial %d of %d." % (i_trial, n_trials))
            setup = self.primary_env.reset()
            if not isinstance(setup, tuple):
                observation = setup
            else:
                observation = setup[0]
                info.update(setup[1])
            partial_trial = PartialMultiverseTrial(self.primary_agent, self.primary_env)
            self.partial_trial_list += [partial_trial]
            if i_trial == n_trials - 1 and last_history_file is not None:
                with open(last_history_file, "w") as f:
                    self.run_branch_until_finished(partial_trial, observation, reward, episode_finished, info, file=f)
            else:
                self.run_branch_until_finished(partial_trial, observation, reward, episode_finished, info, file=None)
            # now the evaluating and updating part starts
            reward = self.primary_env.multiverse_reward(self.partial_trial_list, depolarize=False, recurrence_steps=10)
            self.merge_reward(reward)
            reward_curve[i_trial] = reward
        res["reward_curve"] = reward_curve
        return res
