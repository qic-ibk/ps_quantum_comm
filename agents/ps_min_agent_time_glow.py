"""
Copyright 2018 Alexey Melnikov and Katja Ried.
Copyright 2020 Julius Wallnöfer, Alexey Melnikov
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

Original code for basic PS agent written by Alexey Melnikov and Katja Ried.
Modifications by Julius Wallnöfer:
    * Modifications to basic PS agent as outlined there.
    * Modifications to make glow time-based.
"""


from __future__ import division, print_function
from time import time
from .ps_min_agent import BasicPSAgent


class TimeGlowAgent(BasicPSAgent):
    """An agent that uses time-based glow mechanism instead of steps.

    The time value has to be provided by the environment. We ended up not
    using this variant of the PS agent in the final results for the paper.
    """
    def __init__(self, *args, **kwargs):
        if kwargs["brain_type"] == "clip":
            raise ValueError("%s is not a supported brain_type" % kwargs["brain_type"])
        BasicPSAgent.__init__(self, *args, **kwargs)
        self.time_before = 0
        self.history_since_last_time_step = []

    def _update_g_matrix(self, time_now):
        if self.time_before == time_now:
            self.history_since_last_reward += [self.history_since_last_time_step]
        else:
            self.history_since_last_reward += [self.history_since_last_time_step[:-1], self.history_since_last_time_step[-1:]]
        n = len(self.history_since_last_reward)
        self.g_matrix = (1 - self.ps_eta)**n * self.g_matrix
        for i, sub_history in enumerate(self.history_since_last_reward):
            for action, percept in sub_history:
                self.g_matrix[action, percept] = (1 - self.ps_eta)**(n - 1 - i)

    def _learning(self, reward_now, time_now):  # learning and forgetting
        if self.ps_gamma != 0:
            self.brain.decay(self.ps_gamma)
        if reward_now != 0:
            self._update_g_matrix(time_now)  # needs to handle updating itself instead of relying on brain
            self.history_since_last_time_step = []
            self.history_since_last_reward = []
            self.brain.update_h_matrix(reward_now)

    def deliberate_and_learn(self, observation, reward, episode_finished, info):
        if (time() - self.agent_wait_time) / 60 > 5:
            self.agent_wait_time = time()
            print('Please wait... I am deliberating')

        if "time_now" not in info:
            raise TypeError("Agent with time aware glow needs 'time_now' info from environment")
        time_now = info["time_now"]

        self._learning(reward, time_now)
        if episode_finished and self.reset_glow:
            self.history_since_last_reward = []
            self.history_since_last_time_step = []
            self.brain.reset_glow()
        percept_now = self._percept_preprocess(observation)
        action = self._policy(percept_now)
        if time_now != self.time_before:
            self.history_since_last_reward += [self.history_since_last_time_step[:-1]]
            self.history_since_last_time_step = self.history_since_last_time_step[-1:]
            self.time_before = time_now
        self.history_since_last_time_step += [(action, percept_now)]
        return action

#
#     def percept_preprocess(self, observation):  # preparing for creating a percept
#         percept = observation[0]
#         for i_sum in range(1, observation.size):
#             product = 1
#             for i_product in range(i_sum):
#                 product = product * self.n_percepts_multi[i_product]
#             percept += product * observation[i_sum]
#         return percept
#
#     def update_g_matrix(self, time_now=None):
#         if self.config["time_aware_glow"] is True:
#             if self.time_before == time_now:
#                 self.history_since_last_reward += [self.history_since_last_time_step]
#             else:
#                 self.history_since_last_reward += [self.history_since_last_time_step[:-1], self.history_since_last_time_step[-1:]]
#             n = len(self.history_since_last_reward)
#             self.g_matrix = (1 - self.ps_eta)**n * self.g_matrix
#             for i, sub_history in enumerate(self.history_since_last_reward):
#                 for action, percept in sub_history:
#                     self.g_matrix[action, percept] = (1 - self.ps_eta)**(n - 1 - i)
#             self.history_since_last_time_step = []
#             self.history_since_last_reward = []
#         else:
#             n = len(self.history_since_last_reward)
#             self.g_matrix = (1 - self.ps_eta)**n * self.g_matrix
#             for i, [action, percept] in enumerate(self.history_since_last_reward):
#                 self.g_matrix[action, percept] = (1 - self.ps_eta)**(n - 1 - i)
#             self.history_since_last_reward = []
#
#     def policy(self, observation, time_now=None):  # action selection
#         if (time() - self.agent_wait_time) / 60 > 5:
#             self.agent_wait_time = time()
#             print('Please wait... I am deliberating')
#         percept_now = self.percept_preprocess(observation)
#         if self.policy_type == 'softmax':
#             h_vector_now = self.ps_alpha * self.h_matrix.get_h_vector(percept_now)
#             h_vector_now_mod = h_vector_now - np.max(h_vector_now)
#             p_vector_now = np.exp(h_vector_now_mod) / np.sum(np.exp(h_vector_now_mod))
#         elif self.policy_type == 'standard':
#             h_vector_now = self.h_matrix.get_h_vector(percept_now)
#             p_vector_now = h_vector_now / np.sum(h_vector_now)
#         action = np.random.choice(np.arange(self.n_actions), p=p_vector_now)
# #        # internally update the g-matrix
# #        self.g_matrix = (1 - self.ps_eta) * self.g_matrix
# #        self.g_matrix[action, percept_now] = 1
#         if self.config["time_aware_glow"] is True:
#             if time_now != self.time_before:
#                 self.history_since_last_reward += [self.history_since_last_time_step[:-1]]
#                 self.history_since_last_time_step = self.history_since_last_time_step[-1:]
#                 self.time_before = time_now
#             self.history_since_last_time_step += [(action, percept_now)]
#         else:
#             self.history_since_last_reward += [(action, percept_now)]
#         return action
#
#     def learning(self, reward_now, time_now=None):  # learning and forgetting
#             if self.ps_gamma != 0:
#                 self.h_matrix.decay(self.ps_gamma)
#             if reward_now != 0:
#                 self.update_g_matrix(time_now)
#                 self.h_matrix += self.g_matrix * reward_now  # h- and g-matrices have in general different sparsity
