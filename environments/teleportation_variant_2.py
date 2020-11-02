"""
Copyright 2020 Julius Wallnöfer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import numpy as np
from .libraries import matrix as mat

H_0 = 0
T_0 = 1
H_1 = 2
T_1 = 3
H_2 = 4
T_2 = 5
CNOT_01 = 6
CNOT_02 = 7
CNOT_12 = 8
SEND_1 = 9
SEND_2 = 10
MEASURE_0 = 11
MEASURE_1 = 12
MEASURE_2 = 13
OUTCOME_PLUS_0 = 14
OUTCOME_MINUS_0 = 15
OUTCOME_PLUS_1 = 16
OUTCOME_MINUS_1 = 17
OUTCOME_PLUS_2 = 18
OUTCOME_MINUS_2 = 19


# actions involving certain qubits
ACTIONS_Q0 = [H_0, T_0, CNOT_01, CNOT_02, MEASURE_0]
ACTIONS_Q1 = [H_1, T_1, CNOT_01, CNOT_12, SEND_1, MEASURE_1]
ACTIONS_Q2 = [H_2, T_2, CNOT_02, CNOT_12, SEND_2, MEASURE_2]


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


h0 = tensor(Id, Ha, Id, Id)
t0 = tensor(Id, T, Id, Id)
h1 = tensor(Id, Id, Ha, Id)
t1 = tensor(Id, Id, T, Id)
h2 = tensor(Id, Id, Id, Ha)
t2 = tensor(Id, Id, Id, T)
cnot01 = tensor(Id, CNOT, Id)
cnot02 = mat.reorder(cnot01, [0, 1, 3, 2])
cnot12 = tensor(Id, Id, CNOT)

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
        self.n_actions = 14
        # self.n_percepts  # not applicable here
        self.target = phiplus
        self.target_rho = np.dot(self.target, mat.H(self.target))
        self.state = tensor(self.target, mat.z0, mat.z0)
        self.qubit_locations = [-1, 0, 0]
        self.percept_now = []
        self.available_actions = [i for i in range(self.n_actions)]
        self._remove_actions(ACTIONS_Q0)

    def reset(self):
        self.target = phiplus
        self.target_rho = np.dot(self.target, mat.H(self.target))
        self.state = tensor(self.target, mat.z0, mat.z0)
        self.percept_now = []
        self.qubit_locations = [-1, 0, 0]
        self.available_actions = [i for i in range(self.n_actions)]
        self._remove_actions(ACTIONS_Q0)
        return self.percept_now, {"available_actions": self.available_actions}

    def _check_success(self):
        rho = np.dot(self.state, mat.H(self.state))
        aux1 = mat.ptrace(rho, [1, 3])
        aux2 = mat.ptrace(rho, [1, 2])  # note that 1, 2 in this notation corresponds to qubits 0 and 1
        if self.qubit_locations[1] == 1 and np.allclose(aux1, self.target_rho):
            return True
        elif self.qubit_locations[2] == 1 and np.allclose(aux2, self.target_rho):
            return True
        else:
            return False

    def _add_actions(self, actions):
        for action in actions:
            if action not in self.available_actions:
                self.available_actions += [action]
        self.available_actions.sort()

    def _remove_actions(self, actions):
        for action in actions:
            try:
                self.available_actions.remove(action)
            except ValueError:
                continue

    def _adjust_cnots(self):
        if self.qubit_locations[0] == self.qubit_locations[1] and H_0 in self.available_actions and H_1 in self.available_actions:
            self._add_actions([CNOT_01])
        else:
            self._remove_actions([CNOT_01])
        if self.qubit_locations[0] == self.qubit_locations[2] and H_0 in self.available_actions and H_2 in self.available_actions:
            self._add_actions([CNOT_02])
        else:
            self._remove_actions([CNOT_02])
        if self.qubit_locations[1] == self.qubit_locations[2] and H_1 in self.available_actions and H_2 in self.available_actions:
            self._add_actions([CNOT_12])
        else:
            self._remove_actions([CNOT_12])

    def move(self, action):
        if action in range(11):
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
        elif action == CNOT_02:
            self.state = np.dot(cnot02, self.state)
        elif action == CNOT_12:
            self.state = np.dot(cnot12, self.state)
        elif action == SEND_1:
            self.qubit_locations[1] = 1
            self._remove_actions([SEND_1, SEND_2])
            # qubit 0 gets added in
            self.qubit_locations[0] = 0
            self._add_actions(ACTIONS_Q0)
            self._adjust_cnots()
        elif action == SEND_2:
            self.qubit_locations[2] = 1
            self._remove_actions([SEND_1, SEND_2])
            # qubit 0 gets added in
            self.qubit_locations[0] = 0
            self._add_actions(ACTIONS_Q0)
            self._adjust_cnots()
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
