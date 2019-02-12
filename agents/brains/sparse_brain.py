"""A two-layer clip network with h and g matrices stored as sparse matrices."""

from scipy.sparse import lil_matrix, hstack, vstack
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


class _SparseGMatrix(_CustomSparseMatrix):
    def __init__(self, *args, **kwargs):
        _CustomSparseMatrix.__init__(self, *args, **kwargs)
        # super().__init__(*args, **kwargs)  # only works properly in python3


class SparseBrain(object):
    """A two-layer clip network with h and g matrices stored as sparse matrices."""

    def __init__(self, n_actions, n_percepts, mode="h_zero", blocksize=1024):
        self.h_matrix = _SparseHMatrix((n_actions, n_percepts), dtype=np.float32)
        self.g_matrix = _SparseGMatrix((n_actions, n_percepts), dtype=np.float32)
        self.mode = mode  # h_zero mode will save h-1 instead of h. use only if all edges exist
        self.blocksize = blocksize
        self.percept_buffer = blocksize - 1

    def decay(self, gamma):
        if self.mode == "h_zero":
            aux = self.h_matrix.tocsc()  # not sure if this is necessary in this case
            aux.data *= (1 - gamma)
            self.h_matrix = _SparseHMatrix(aux)
        else:
            aux = self.h_matrix.tocsc()
            aux.data -= gamma * (aux.data - 1.)  # h = (1-gamma)*h + gamma*1 matrix
            self.h_matrix = _SparseHMatrix(aux)

    def get_h_vector(self, percept):
        try:
            if self.mode == "h_zero":
                return self.h_matrix[:, percept].toarray().flatten() + 1
            else:
                if np.sum(self.h_matrix[:, percept]) == 0:  # if percept is new - create it
                    self.h_matrix[:, percept] = 1
                return self.h_matrix[:, percept].toarray().flatten()
        except IndexError:
            return np.ones(self.h_matrix.shape[0], dtype=self.h_matrix.dtype)

    def update_g_matrix(self, eta, history_since_last_reward):
        if not isinstance(history_since_last_reward[0], tuple):
            history_since_last_reward = [history_since_last_reward]  # if it is only a tuple, make it a list of tuples anyway
        n = len(history_since_last_reward)
        self.g_matrix = (1 - eta)**n * self.g_matrix
        for i, [action, percept] in enumerate(history_since_last_reward):
            self.g_matrix[action, percept] = (1 - eta)**(n - 1 - i)

    def update_h_matrix(self, reward):
        self.h_matrix += self.g_matrix * reward  # h- and g-matrices have in general different sparsity

    def reset_glow(self):
        self.g_matrix = _SparseGMatrix(self.g_matrix.shape, dtype=np.float32)

    def add_percept(self):
        self.percept_buffer += 1
        if self.percept_buffer >= self.blocksize:
            # also here, there must be a more elegant way to do this
            self.h_matrix = _SparseHMatrix(hstack([self.h_matrix, lil_matrix((self.h_matrix.shape[0], self.blocksize), dtype=self.h_matrix.dtype)], format="lil"))
            self.g_matrix = _SparseGMatrix(hstack([self.g_matrix, lil_matrix((self.g_matrix.shape[0], self.blocksize), dtype=self.g_matrix.dtype)], format="lil"))
            self.percept_buffer = 0

    def add_action(self):
        if self.mode != "h_zero":
            raise RuntimeError("add_action method is not supported without mode 'h_zero'")
        self.h_matrix = _SparseHMatrix(vstack([self.h_matrix, lil_matrix((1, self.h_matrix.shape[1]), dtype=self.h_matrix.dtype)], format="lil"))
        self.g_matrix = _SparseGMatrix(vstack([self.g_matrix, lil_matrix((1, self.g_matrix.shape[1]), dtype=self.g_matrix.dtype)], format="lil"))
