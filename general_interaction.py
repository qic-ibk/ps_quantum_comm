from __future__ import division, print_function
import numpy as np


class Interaction(object):

    def __init__(self, agent_type, agent, environment):
        self.agent_type = agent_type
        self.agent = agent
        self.env = environment

    def single_learning_life(self, n_trials, max_steps_per_trial, return_last_trial_history=False, reset_glow=False):  # Note: reset_glow as an option here feels weird - should be an agent option
        learning_curve = np.zeros(n_trials)
        reward = 0
        info = {}
        res = {}
        for i_trial in range(n_trials):
            reward_trial = 0
            if i_trial % 500 == 0:
                print("Interaction is now starting trial %d of %d." % (i_trial, n_trials))
            setup = self.env.reset()
            if not isinstance(setup, tuple):
                observation = setup
            else:
                observation, info = setup
            if i_trial == n_trials - 1:
                last_trial_history = []
            for t in range(max_steps_per_trial):
                old_observation = observation
                observation, reward, done, action, info = self.single_interaction_step_PS(observation, reward, info)
                if reset_glow:
                    info["reset_glow"] = done
                reward_trial += float(reward)
                if return_last_trial_history and i_trial == n_trials - 1:
                    last_trial_history += [(old_observation, action)]
                if done:
                    learning_curve[i_trial] = reward_trial / (t + 1)
                    break
        res["learning_curve"] = learning_curve
        if return_last_trial_history:
            res["last_trial_history"] = last_trial_history
        return res

    def single_interaction_step_PS(self, observation, reward, info):
        action = self.agent.deliberate_and_learn(observation, reward, info)
        res = self.env.move(action)
        if len(res) == 3:  # backwards compatibility
            res += ({},)
        observation, reward, done, info = res
        return observation, reward, done, action, info
