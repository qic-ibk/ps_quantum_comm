import numpy as np
from .libraries import matrix as mat

H_0 = 0
P_0 = 1
H_1 = 2
P_1 = 3
H_2 = 4
P_2 = 5
CNOT_01 = 6
MEASURE_0 = 7
MEASURE_1 = 8
MEASURE_2 = 9
OUTCOME_PLUS_0 = 10
OUTCOME_MINUS_0 = 11
OUTCOME_PLUS_1 = 12
OUTCOME_MINUS_1 = 13
OUTCOME_PLUS_2 = 14
OUTCOME_MINUS_2 = 15


Ha = mat.Ha
P = np.array([[1, 0], [0, 1j]], dtype=np.complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
Id = np.eye(2)

phiplus = mat.phiplus


def tensor(*args):
    res = np.array([[1]])
    for i in args:
        res = np.kron(res, i)
    return res


def H(rho):
    return rho.conj().T


# def apply(rho, U):
#     return np.dot(np.dot(U, rho), H(U))


h0 = tensor(Id, Ha, Id, Id)
p0 = tensor(Id, P, Id, Id)
h1 = tensor(Id, Id, Ha, Id)
p1 = tensor(Id, Id, P, Id)
h2 = tensor(Id, Id, Id, Ha)
p2 = tensor(Id, Id, Id, P)
cnot01 = tensor(Id, CNOT, Id)

proj_plus_0 = tensor(Id, mat.Pz0, Id, Id)
proj_minus_0 = tensor(Id, mat.Pz1, Id, Id)
proj_plus_1 = tensor(Id, Id, mat.Pz0, Id)
proj_minus_1 = tensor(Id, Id, mat.Pz1, Id)
proj_plus_2 = tensor(Id, Id, Id, mat.Pz0)
proj_minus_2 = tensor(Id, Id, Id, mat.Pz1)


# def _random_pure_state():
#     # pick a random point on the bloch sphere
#     phi = 2 * np.pi * np.random.random()
#     theta = np.arccos(2 * np.random.random() - 1)
#     return np.cos(theta / 2) * mat.z0 + np.exp(1j * phi) * np.sin(theta / 2) * mat.z1


def _norm(psi):
    return np.sqrt(np.sum(np.abs(psi)**2))


def _normalize(psi):
    return psi / _norm(psi)


def _measure(psi, i):
    if i == 0:
        aux1 = np.dot(proj_plus_0, psi)
        aux2 = np.dot(proj_minus_0, psi)
    elif i == 1:
        aux1 = np.dot(proj_plus_1, psi)
        aux2 = np.dot(proj_minus_1, psi)
    elif i == 2:
        aux1 = np.dot(proj_plus_2, psi)
        aux2 = np.dot(proj_minus_2, psi)
    p_plus = _norm(aux1)**2
    p_minus = _norm(aux2)**2
    if np.random.random() < p_plus:
        state = _normalize(aux1)
        outcome = 0
    else:
        state = _normalize(aux2)
        outcome = 1
    return state, outcome


class TaskEnvironment(object):
    """

    """
    def __init__(self, **userconfig):
        self.n_actions = 10
        # self.n_percepts  # not applicable here
        self.target = phiplus
        self.target_rho = np.dot(self.target, mat.H(self.target))
        self.state = tensor(self.target, phiplus)
        self.percept_now = []

    def reset(self):
        self.target = phiplus
        self.target_rho = np.dot(self.target, mat.H(self.target))
        self.state = tensor(self.target, phiplus)
        self.percept_now = []
        return self.percept_now

    def _check_success(self):
        aux = np.dot(self.state, mat.H(self.state))
        aux = mat.ptrace(aux, [1, 2])  # note that 1, 2 in this notation corresponds to qubits 0 and 1
        return np.allclose(aux, self.target_rho)

    def move(self, action):
        if action in range(7):
            self.percept_now += [action]

        if action == H_0:
            self.state = np.dot(h0, self.state)
        elif action == P_0:
            self.state = np.dot(p0, self.state)
        elif action == H_1:
            self.state = np.dot(h1, self.state)
        elif action == P_1:
            self.state = np.dot(p1, self.state)
        elif action == H_2:
            self.state = np.dot(h2, self.state)
        elif action == P_2:
            self.state = np.dot(p2, self.state)
        elif action == CNOT_01:
            self.state = np.dot(cnot01, self.state)
        elif action == MEASURE_0:
            self.state, outcome = _measure(self.state, 0)
            if outcome == 0:
                self.percept_now += [OUTCOME_PLUS_0]
            else:
                self.percept_now += [OUTCOME_MINUS_0]
        elif action == MEASURE_1:
            self.state, outcome = _measure(self.state, 1)
            if outcome == 0:
                self.percept_now += [OUTCOME_PLUS_1]
            else:
                self.percept_now += [OUTCOME_MINUS_1]
        elif action == MEASURE_2:
            self.state, outcome = _measure(self.state, 2)
            if outcome == 0:
                self.percept_now += [OUTCOME_PLUS_2]
            else:
                self.percept_now += [OUTCOME_MINUS_2]

        if self._check_success():
            reward = 1
            episode_finished = 1
        else:
            reward = 0
            episode_finished = 0

        return self.percept_now, reward, episode_finished
