"""A two-layer clip network with h and g matrices stored as sparse matrices."""

from scipy.sparse import lil_matrix
import numpy as np


class _CustomSparseMatrix(lil_matrix):
    def __init__(self, *args, **kwargs):
        lil_matrix.__init__(self, *args, **kwargs)
        # super().__init__(*args, **kwargs)  # only works properly in python3

    def __add__(self, other):  # because lil_matrix.__add__ doesn't preserve sparsity type by default
        return self.__class__(self.tocsc().__add__(other))

    def __radd__(self, other):
        return self.__class__(self.tocsc().__radd__(other))


class _SparseHMatrix(_CustomSparseMatrix):
    def __init__(self, *args, **kwargs):
        _CustomSparseMatrix.__init__(self, *args, **kwargs)
        # super().__init__(*args, **kwargs)  # only works properly in python3

    def get_h_vector(self, percept):
        if np.sum(self[:, percept]) == 0:  # if percept is new - create it
            self[:, percept] = 1
        return self[:, percept].toarray().flatten()

    def decay(self, gamma):
        aux = self.tocsc()
        aux.data -= gamma * (aux.data - 1.)  # h = (1-gamma)*h + gamma*1 matrix
        self = _SparseHMatrix(aux)


class _SparseGMatrix(_CustomSparseMatrix):
    def __init__(self, *args, **kwargs):
        _CustomSparseMatrix.__init__(self, *args, **kwargs)
        # super().__init__(*args, **kwargs)  # only works properly in python3


class SparseBrain(object):
    """A two-layer clip network with h and g matrices stored as sparse matrices."""

    def __init__(self, n_actions, n_percepts):
        self.h_matrix = _SparseHMatrix((n_actions, n_percepts), dtype=np.float32)
        self.g_matrix = _SparseGMatrix((n_actions, n_percepts), dtype=np.float32)

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
