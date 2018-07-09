from __future__ import division, print_function
import numpy as np


class Interaction(object):

    def __init__(self, agent_type, agent, environment):
        self.agent_type = agent_type
        self.agent = agent
        self.env = environment

    def single_learning_life(self, n_trials, max_steps_per_trial):
        learning_curve = np.zeros(n_trials)
        reward = 0
        info = {}
        for i_trial in range(n_trials):
            reward_trial = 0
            if hasattr(self.env, "tracks_time") and self.env.tracks_time is True:
                observation, time_now = self.env.reset()
                info = {"time_now": time_now}
            else:
                observation = self.env.reset()
            if i_trial == n_trials - 1:
                last_trial_history = []
            for t in range(max_steps_per_trial):
                old_observation = observation
                observation, reward, done, action, info = self.single_interaction_step_PS(observation, reward, info)
                reward_trial += float(reward)
                if i_trial == n_trials - 1:
                    last_trial_history += [(old_observation, action)]
                if done:
                    learning_curve[i_trial] = reward_trial / (t + 1)
                    break
        return learning_curve, last_trial_history

    def single_interaction_step_PS(self, observation, reward, info):
        action = self.agent.deliberate_and_learn(observation, reward, info)
        if hasattr(self.env, "tracks_time") and self.env.tracks_time is True:
            observation, reward, done, time_now = self.env.move(action)
            info = {"time_now": time_now}
        else:
            observation, reward, done = self.env.move(action)
        return observation, reward, done, action, info
