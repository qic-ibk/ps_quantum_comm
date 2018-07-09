"""A two-layer clip network with h and g matrices stored as numpy arrays."""

import numpy as np


class _DenseHMatrix(np.ndarray):
    def __new__(cls, shape, dtype=np.float):
        return np.ones(shape, dtype=dtype).view(cls)

    def __array_finalize__(self, obj):
        pass

    def get_h_vector(self, percept):
        return self[:, percept]

    def decay(self, gamma):
        self = (1. - gamma) * self + gamma * np.ones(self.shape, dtype=self.dtype)


class _DenseGMatrix(np.ndarray):
    def __new__(cls, shape, dtype=np.float):
        return np.zeros(shape, dtype=dtype).view(cls)

    def __array_finalize__(self, obj):
        pass


class DenseBrain(object):
    """A two-layer clip network with h and g matrices stored as numpy arrays."""

    def __init__(self, n_actions, n_percepts):
        self.h_matrix = _DenseHMatrix((n_actions, n_percepts), dtype=np.float32)
        self.g_matrix = _DenseGMatrix((n_actions, n_percepts), dtype=np.float32)

    def decay(self, gamma):
        self.h_matrix.decay(gamma)

    def get_h_vector(self, percept):
        return self.h_matrix.get_h_vector(percept)

    def update_g_matrix(self, eta, history_since_last_reward):
        if not isinstance(history_since_last_reward[0], tuple):
            history_since_last_reward = [history_since_last_reward]  # if it is only a tuple, make it a list of tuples anyway
        n = len(history_since_last_reward)
        self.g_matrix = (1 - eta)**n * self.g_matrix
        for i, [action, percept] in enumerate(history_since_last_reward):
            self.g_matrix[action, percept] = (1 - eta)**(n - 1 - i)
