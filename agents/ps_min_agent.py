from __future__ import division, print_function
import numpy as np
from time import time


class BasicPSAgent(object):
    """PS agent implementation.

    parameters:
    n_actions - number of available actions, constant
    n_percepts - number of percepts, constant
    ps_gamma, ps_eta - constants
    policy_type - 'standard' or 'softmax'
    ps_alpha - constant

    """

    def __init__(self, n_actions, n_percepts_multi, ps_gamma, ps_eta, policy_type, ps_alpha, matrix_type="dense"):
        self.agent_wait_time = time()
        self.n_actions = n_actions
        self.n_percepts_multi = n_percepts_multi
        self.n_percepts = np.prod(n_percepts_multi)
        self.policy_type = policy_type

        self.ps_gamma = ps_gamma
        self.ps_eta = ps_eta
        self.ps_alpha = ps_alpha

        if matrix_type == "dense":
            from brains.dense_brain import DenseBrain
            self.brain = DenseBrain(self.n_actions, self.n_percepts)
        elif matrix_type == "sparse":
            from brains.sparse_brain import SparseBrain
            self.brain = SparseBrain(self.n_actions, self.n_percepts)
        else:
            raise ValueError("%s is not a supported matrix_type" % matrix_type)

        self.history_since_last_reward = []

    def percept_preprocess(self, observation):  # preparing for creating a percept
        percept = observation[0]
        for i_sum in range(1, observation.size):
            product = 1
            for i_product in range(i_sum):
                product = product * self.n_percepts_multi[i_product]
            percept += product * observation[i_sum]
        return percept

    def policy(self, observation):  # action selection
        if (time() - self.agent_wait_time) / 60 > 5:
            self.agent_wait_time = time()
            print('Please wait... I am deliberating')
        percept_now = self.percept_preprocess(observation)
        if self.policy_type == 'softmax':
            h_vector_now = self.ps_alpha * self.brain.get_h_vector(percept_now)
            h_vector_now_mod = h_vector_now - np.max(h_vector_now)
            p_vector_now = np.exp(h_vector_now_mod) / np.sum(np.exp(h_vector_now_mod))
        elif self.policy_type == 'standard':
            h_vector_now = self.brain.get_h_vector(percept_now)
            p_vector_now = h_vector_now / np.sum(h_vector_now)
        action = np.random.choice(np.arange(self.n_actions), p=p_vector_now)
#        # internally update the g-matrix
#        self.g_matrix = (1 - self.ps_eta) * self.g_matrix
#        self.g_matrix[action, percept_now] = 1
        self.history_since_last_reward += [(action, percept_now)]
        return action

    def learning(self, reward_now, time_now=None):  # learning and forgetting
            if self.ps_gamma != 0:
                self.brain.decay(self.ps_gamma)
            if reward_now != 0:
                self.brain.update_g_matrix(self.ps_eta, self.history_since_last_reward)
                self.history_since_last_reward = []
                self.brain.h_matrix += self.brain.g_matrix * reward_now  # h- and g-matrices have in general different sparsity
