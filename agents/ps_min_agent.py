from __future__ import division, print_function
import numpy as np


class BasicPSAgent(object):
    """ PS agent implementation """
    """ parameters: """
    """ n_actions - number of available actions, constant """
    """ n_percepts - number of percepts, constant """
    """ ps_gamma, ps_eta - constants """
    """ policy_type - 'standard' or 'softmax' """
    """ ps_alpha - constant """
    
    def __init__(self, n_actions, n_percepts_multi, ps_gamma, ps_eta, policy_type, ps_alpha, time_glow=False, **userconfig):
        self.n_actions = n_actions
        self.n_percepts_multi = n_percepts_multi
        self.n_percepts = np.prod(n_percepts_multi)
        self.policy_type = policy_type
        self.config = {
            "ps_gamma" : ps_gamma,
            "ps_eta" : ps_eta,
            "ps_alpha" : ps_alpha,
            "time_aware_glow" : time_glow}
        self.config.update(userconfig)
        self.h_matrix = np.ones((self.n_actions, self.n_percepts))
        self.g_matrix = np.zeros((self.n_actions, self.n_percepts))
        self.history_since_last_reward = []
        if time_glow == True: # prepare some properties the time dependant glow will need
            self.time_before = -1
            self.history_since_last_time_step = []
        
    def percept_preprocess(self, observation): # preparing for creating a percept
        return self.mapping_to_1d(observation)
        
    def mapping_to_1d(self, observation):
        percept = observation[0]
        for i_sum in range(1, observation.size):
            product = 1
            for i_product in range(i_sum):
                product = product * self.n_percepts_multi[i_product]
            percept += product * observation[i_sum]
        return percept

    def update_g_matrix(self):
        if self.config["time_aware_glow"] == True:
            self.history_since_last_reward += [self.history_since_last_time_step]
            n = len(self.history_since_last_reward)
            self.g_matrix = (1-self.config["ps_eta"])**n * self.g_matrix
            for i, sub_history in enumerate(self.history_since_last_reward):
                for action,percept in sub_history:
                    self.g_matrix[action,percept] = (1-self.config["ps_eta"])**(n - 1 - i)
            self.history_since_last_time_step = []
            self.history_since_last_reward = []
        else:
            n = len(self.history_since_last_reward)
            self.g_matrix = (1-self.config["ps_eta"])**n * self.g_matrix
            for i,[action,percept] in enumerate(self.history_since_last_reward):
                self.g_matrix[action,percept] = (1-self.config["ps_eta"])**(n - 1 - i)
            self.history_since_last_reward = []
    
    def policy(self, percept_now, time_now = None): # policy + g-matrix update
        config = self.config
        # action selection
        if self.policy_type == 'standard':
            h_vector_now = self.h_matrix[:, percept_now]
            p_vector_now = h_vector_now / np.sum(h_vector_now)
        if self.policy_type == 'softmax':
            h_vector_now = self.config["ps_alpha"] * self.h_matrix[:, percept_now]
            h_vector_now_mod = h_vector_now - np.max(h_vector_now)
            p_vector_now = np.exp(h_vector_now_mod) / np.sum(np.exp(h_vector_now_mod))   
        action = np.random.choice(np.arange(self.n_actions), 1, p=p_vector_now)[0]
#        # internally update the g-matrix
#        self.g_matrix = (1 - self.config["ps_eta"]) * self.g_matrix 
#        self.g_matrix[action, percept_now] = 1
        if self.config["time_aware_glow"] == True:
            if time_now != self.time_before:
                self.history_since_last_reward += [self.history_since_last_time_step]
                self.history_since_last_time_step = []
                self.time_before = time_now
            self.history_since_last_time_step += [(action,percept_now)]
        else:
            self.history_since_last_reward += [(action,percept_now)]
        return action
        
    def learning(self, reward_now):
        config = self.config
        if config["ps_gamma"] == 0 and reward_now == 0:
            pass
        else:
            self.update_g_matrix()       
            self.h_matrix = (1 - self.config["ps_gamma"]) * self.h_matrix + self.config["ps_gamma"] * np.ones((self.n_actions, self.n_percepts)) + reward_now * self.g_matrix
        
    def h_matrix_output(self):
        return self.h_matrix
            
    def g_matrix_output(self):
        return self.g_matrix