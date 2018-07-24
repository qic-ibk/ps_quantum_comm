"""A two-layer clip network with h and g matrices stored as sparse matrices."""

from scipy.sparse import lil_matrix, hstack
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
        aux = self.h_matrix.tocsc()
        aux.data -= gamma * (aux.data - 1.)  # h = (1-gamma)*h + gamma*1 matrix
        self.h_matrix = _SparseHMatrix(aux)

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
        self.h_matrix += self.g_matrix * reward  # h- and g-matrices have in general different sparsity

    def add_percept(self):
        # also here, there must be a more elegant way to do this
        self.h_matrix = _SparseHMatrix(hstack([self.h_matrix, lil_matrix((self.h_matrix.shape[0], 1), dtype=self.h_matrix.dtype)], format="lil"))
        self.g_matrix = _SparseGMatrix(hstack([self.g_matrix, lil_matrix((self.g_matrix.shape[0], 1), dtype=self.g_matrix.dtype)], format="lil"))
