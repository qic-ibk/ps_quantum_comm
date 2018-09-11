import numpy as np
from .libraries import matrix as mat

H_0 = 0
T_0 = 1
H_1 = 2
T_1 = 3
H_2 = 4
T_2 = 5
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

# actions involving certain qubits
ACTIONS_Q0 = [0, 1, 6, 7]
ACTIONS_Q1 = [2, 3, 6, 8]
ACTIONS_Q2 = [4, 5, 9]


Ha = mat.Ha
T = np.array([[1, 0], [0, 1 / np.sqrt(2) * (1 + 1j)]], dtype=np.complex)
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


h0 = tensor(Ha, Id, Id)
t0 = tensor(T, Id, Id)
h1 = tensor(Id, Ha, Id)
t1 = tensor(Id, T, Id)
h2 = tensor(Id, Id, Ha)
t2 = tensor(Id, Id, T)
cnot01 = tensor(CNOT, Id)

proj_plus_0 = tensor(mat.Pz0, Id, Id)
proj_minus_0 = tensor(mat.Pz1, Id, Id)
proj_plus_1 = tensor(Id, mat.Pz0, Id)
proj_minus_1 = tensor(Id, mat.Pz1, Id)
proj_plus_2 = tensor(Id, Id, mat.Pz0)
proj_minus_2 = tensor(Id, Id, mat.Pz1)


def _random_pure_state():
    # pick a random point on the bloch sphere
    phi = 2 * np.pi * np.random.random()
    theta = np.arccos(2 * np.random.random() - 1)
    return np.cos(theta / 2) * mat.z0 + np.exp(1j * phi) * np.sin(theta / 2) * mat.z1


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
        self.target = _random_pure_state()
        self.target_rho = np.dot(self.target, mat.H(self.target))
        self.state = tensor(self.target, phiplus)
        self.percept_now = []
        self.available_actions = [i for i in range(self.n_actions)]

    def reset(self):
        self.target = _random_pure_state()
        self.target_rho = np.dot(self.target, mat.H(self.target))
        self.state = tensor(self.target, phiplus)
        self.percept_now = []
        self.available_actions = [i for i in range(self.n_actions)]
        return self.percept_now, {"available_actions": self.available_actions}

    def _check_success(self):
        aux = np.dot(self.state, mat.H(self.state))
        aux = mat.ptrace(aux, [0, 1])
        return np.allclose(aux, self.target_rho)

    def _remove_actions(self, actions):
        for action in actions:
            try:
                self.available_actions.remove(action)
            except ValueError:
                continue

    def move(self, action):
        if action in range(7):
            self.percept_now += [action]

        if action == H_0:
            self.state = np.dot(h0, self.state)
        elif action == T_0:
            self.state = np.dot(t0, self.state)
        elif action == H_1:
            self.state = np.dot(h1, self.state)
        elif action == T_1:
            self.state = np.dot(t1, self.state)
        elif action == H_2:
            self.state = np.dot(h2, self.state)
        elif action == T_2:
            self.state = np.dot(t2, self.state)
        elif action == CNOT_01:
            self.state = np.dot(cnot01, self.state)
        elif action == MEASURE_0:
            self.state, outcome = _measure(self.state, 0)
            if outcome == 0:
                self.percept_now += [OUTCOME_PLUS_0]
            else:
                self.percept_now += [OUTCOME_MINUS_0]
            self._remove_actions(ACTIONS_Q0)
        elif action == MEASURE_1:
            self.state, outcome = _measure(self.state, 1)
            if outcome == 0:
                self.percept_now += [OUTCOME_PLUS_1]
            else:
                self.percept_now += [OUTCOME_MINUS_1]
            self._remove_actions(ACTIONS_Q1)
        elif action == MEASURE_2:
            self.state, outcome = _measure(self.state, 2)
            if outcome == 0:
                self.percept_now += [OUTCOME_PLUS_2]
            else:
                self.percept_now += [OUTCOME_MINUS_2]
            self._remove_actions(ACTIONS_Q2)
        else:
            raise ValueError("TaskEnvironment does not support action %s" % repr(action))

        if self._check_success():
            reward = 1
            episode_finished = 1
        else:
            reward = 0
            episode_finished = 0

        if not self.available_actions:  # if no actions remain, episode is over
            episode_finished = 1

        return self.percept_now, reward, episode_finished, {"available_actions": self.available_actions}
