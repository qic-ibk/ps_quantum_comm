"""A two-layer clip network with h and g matrices stored as numpy arrays."""

import numpy as np


class _DenseHMatrix(np.ndarray):
    def __new__(cls, shape, dtype=np.float):
        return np.ones(shape, dtype=dtype).view(cls)

    def __array_finalize__(self, obj):
        pass

    def get_h_vector(self, percept):
        return self[:, percept]


class _DenseGMatrix(np.ndarray):
    def __new__(cls, shape, dtype=np.float):
        return np.zeros(shape, dtype=dtype).view(cls)

    def __array_finalize__(self, obj):
        pass


class DenseBrain(object):
    """A two-layer clip network with h and g matrices stored as numpy arrays."""

    def __init__(self, n_actions, n_percepts, blocksize=1024):
        self.h_matrix = _DenseHMatrix((n_actions, n_percepts), dtype=np.float32)
        self.g_matrix = _DenseGMatrix((n_actions, n_percepts), dtype=np.float32)
        self.blocksize = blocksize
        self.percept_buffer = blocksize - 1

    def decay(self, gamma):
        self.h_matrix = (1. - gamma) * self.h_matrix + gamma * np.ones(self.h_matrix.shape, dtype=self.h_matrix.dtype)

    def get_h_vector(self, percept):
        return self.h_matrix.get_h_vector(percept)

    def update_g_matrix(self, eta, history_since_last_reward):
        if not isinstance(history_since_last_reward[0], tuple):
            history_since_last_reward = [history_since_last_reward]  # if it is only a tuple, make it a list of tuples anyway
        n = len(history_since_last_reward)
        self.g_matrix = (1 - eta)**n * self.g_matrix
        for i, [action, percept] in enumerate(history_since_last_reward):
            self.g_matrix[action, percept] = (1 - eta)**(n - 1 - i)

    def update_h_matrix(self, reward):
        self.h_matrix += self.g_matrix * reward

    def reset_glow(self):
        self.g_matrix = _DenseGMatrix(self.g_matrix.shape, dtype=np.float32)

    def add_percept(self):
        self.percept_buffer += 1
        if self.percept_buffer >= self.blocksize:
            # there must be a more elegant way to do this
            self.h_matrix = np.hstack([self.h_matrix, np.ones((self.h_matrix.shape[0], self.blocksize), dtype=self.h_matrix.dtype)]).view(self.h_matrix.__class__)
            self.g_matrix = np.hstack([self.g_matrix, np.zeros((self.g_matrix.shape[0], self.blocksize), dtype=self.g_matrix.dtype)]).view(self.g_matrix.__class__)
            self.percept_buffer = 0

    def add_action(self):
        self.h_matrix = np.vstack([self.h_matrix, np.ones((1, self.h_matrix.shape[1]), dtype=self.h_matrix.dtype)]).view(self.h_matrix.__class__)
        self.g_matrix = np.vstack([self.g_matrix, np.zeros((1, self.g_matrix.shape[1]), dtype=self.g_matrix.dtype)]).view(self.g_matrix.__class__)
