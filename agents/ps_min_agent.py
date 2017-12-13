from __future__ import division, print_function
import numpy as np
from scipy.sparse import lil_matrix  # for sparse matrices


class DenseHMatrix(np.ndarray):
    def __new__(cls, shape, dtype=np.float):
        return np.ones(shape, dtype=dtype).view(cls)

    def __array_finalize__(self, obj):
        pass

    def get_h_vector(self, percept):
        return self[:, percept]

    def decay(self, gamma):
        self = (1. - gamma) * self + gamma * np.ones(self.shape, dtype=self.dtype)


class DenseGMatrix(np.ndarray):
    def __new__(cls, shape, dtype=np.float):
        return np.zeros(shape, dtype=dtype).view(cls)

    def __array_finalize__(self, obj):
        pass


class CustomSparseMatrix(lil_matrix):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __add__(self, other):  # because lil_matrix.__add__ doesn't preserve sparsity type by default
        return self.__class__(self.tocsc().__add__(other))

    def __radd__(self, other):
        return self.__class__(self.tocsc().__radd__(other))


class SparseHMatrix(CustomSparseMatrix):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_h_vector(self, percept):
        if np.sum(self[:, percept]) == 0:  # if percept is new - create it
            self[:, percept] = 1
        return self[:, percept].toarray().flatten()

    def decay(self, gamma):
        aux = self.tocsc()
        aux.data -= gamma * (aux.data - 1.)  # h = (1-gamma)*h + gamma*1 matrix
        self = SparseHMatrix(aux)


class SparseGMatrix(CustomSparseMatrix):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BasicPSAgent(object):
    """ PS agent implementation """
    """ parameters: """
    """ n_actions - number of available actions, constant """
    """ n_percepts - number of percepts, constant """
    """ ps_gamma, ps_eta - constants """
    """ policy_type - 'standard' or 'softmax' """
    """ ps_alpha - constant """

    def __init__(self, n_actions, n_percepts_multi, ps_gamma, ps_eta, policy_type, ps_alpha, time_glow=False, matrix_type="dense"):
        self.n_actions = n_actions
        self.n_percepts_multi = n_percepts_multi
        self.n_percepts = np.prod(n_percepts_multi)
        self.policy_type = policy_type

        self.ps_gamma = ps_gamma
        self.ps_eta = ps_eta
        self.ps_alpha = ps_alpha
        self.config = {
            "time_aware_glow": time_glow}

        if matrix_type == "dense":
            self.h_matrix = DenseHMatrix((self.n_actions, self.n_percepts), dtype=np.float32)
            self.g_matrix = DenseGMatrix((self.n_actions, self.n_percepts), dtype=np.float32)
        elif matrix_type == "sparse":
            self.h_matrix = SparseHMatrix((self.n_actions, self.n_percepts), dtype=np.float32)
            self.g_matrix = SparseGMatrix((self.n_actions, self.n_percepts), dtype=np.float32)
        else:
            raise ValueError("%s is not a supported matrix_type" % matrix_type)

        self.history_since_last_reward = []
        if time_glow is True:  # prepare some properties the time dependant glow will need
            self.time_before = 0
            self.history_since_last_time_step = []

    def percept_preprocess(self, observation):  # preparing for creating a percept
        percept = observation[0]
        for i_sum in range(1, observation.size):
            product = 1
            for i_product in range(i_sum):
                product = product * self.n_percepts_multi[i_product]
            percept += product * observation[i_sum]
        return percept

    def update_g_matrix(self, time_now=None):
        if self.config["time_aware_glow"] is True:
            if self.time_before == time_now:
                self.history_since_last_reward += [self.history_since_last_time_step]
            else:
                self.history_since_last_reward += [self.history_since_last_time_step[:-1], self.history_since_last_time_step[-1:]]
            n = len(self.history_since_last_reward)
            self.g_matrix = (1 - self.ps_eta)**n * self.g_matrix
            for i, sub_history in enumerate(self.history_since_last_reward):
                for action, percept in sub_history:
                    self.g_matrix[action, percept] = (1 - self.ps_eta)**(n - 1 - i)
            self.history_since_last_time_step = []
            self.history_since_last_reward = []
        else:
            n = len(self.history_since_last_reward)
            self.g_matrix = (1 - self.ps_eta)**n * self.g_matrix
            for i, [action, percept] in enumerate(self.history_since_last_reward):
                self.g_matrix[action, percept] = (1 - self.ps_eta)**(n - 1 - i)
            self.history_since_last_reward = []

    def policy(self, observation, time_now=None):  # action selection
        percept_now = self.percept_preprocess(observation)
        if self.policy_type == 'softmax':
            h_vector_now = self.ps_alpha * self.h_matrix.get_h_vector(percept_now)
            h_vector_now_mod = h_vector_now - np.max(h_vector_now)
            p_vector_now = np.exp(h_vector_now_mod) / np.sum(np.exp(h_vector_now_mod))
        elif self.policy_type == 'standard':
            h_vector_now = self.h_matrix.get_h_vector(percept_now)
            p_vector_now = h_vector_now / np.sum(h_vector_now)
        action = np.random.choice(np.arange(self.n_actions), p=p_vector_now)
#        # internally update the g-matrix
#        self.g_matrix = (1 - self.ps_eta) * self.g_matrix
#        self.g_matrix[action, percept_now] = 1
        if self.config["time_aware_glow"] is True:
            if time_now != self.time_before:
                self.history_since_last_reward += [self.history_since_last_time_step[:-1]]
                self.history_since_last_time_step = self.history_since_last_time_step[-1:]
                self.time_before = time_now
            self.history_since_last_time_step += [(action, percept_now)]
        else:
            self.history_since_last_reward += [(action, percept_now)]
        return action

    def learning(self, reward_now, time_now=None):  # learning and forgetting
            if self.ps_gamma != 0:
                self.h_matrix.decay(self.ps_gamma)
            if reward_now != 0:
                self.update_g_matrix(time_now)
                self.h_matrix += self.g_matrix * reward_now  # h- and g-matrices have in general different sparsity
