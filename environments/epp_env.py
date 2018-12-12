"""
"""

from __future__ import print_function, division
from .abstract_environment import AbstractEnvironment
import numpy as np
from .libraries import matrix as mat
from warnings import warn

np.seterr(divide="raise")


# NOTE: T is the x-basis variant here
H_0 = 0
T_0 = 1
H_1 = 2
T_1 = 3
H_2 = 4
T_2 = 5
H_3 = 6
T_3 = 7
CNOT_01 = 8
CNOT_23 = 9
MEASURE_0 = 10
MEASURE_1 = 11
MEASURE_2 = 12
MEASURE_3 = 13
ACCEPT = 14
REJECT = 15
OUTCOME_PLUS_0 = 16
OUTCOME_MINUS_0 = 17
OUTCOME_PLUS_1 = 18
OUTCOME_MINUS_1 = 19
OUTCOME_PLUS_2 = 20
OUTCOME_MINUS_2 = 21
OUTCOME_PLUS_3 = 22
OUTCOME_MINUS_3 = 23

# actions involving certain qubits
ACTIONS_Q0 = [0, 1, 8, 10]
ACTIONS_Q1 = [2, 3, 8, 11]
ACTIONS_Q2 = [4, 5, 9, 12]
ACTIONS_Q3 = [6, 7, 9, 13]

Ha = mat.Ha
T = np.array([[1, 0], [0, 1 / np.sqrt(2) * (1 + 1j)]], dtype=np.complex)
# carfeful: T is the x-variant
T = np.dot(np.dot(Ha, T), Ha)
# even more simplified: only clifford - ugly hack
T = np.dot(T, T)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
Id = np.eye(2)

phiplus = mat.phiplus
psiplus = mat.psiplus
phiminus = mat.phiminus
psiminus = mat.psiminus


def tensor(*args):
    res = np.array([[1]])
    for i in args:
        res = np.kron(res, i)
    return res


def H(rho):
    return rho.conj().T


h0 = tensor(Ha, Id, Id, Id)
t0 = tensor(T, Id, Id, Id)
h1 = tensor(Id, Ha, Id, Id)
t1 = tensor(Id, T, Id, Id)
h2 = tensor(Id, Id, Ha, Id)
t2 = tensor(Id, Id, T, Id)
h3 = tensor(Id, Id, Id, Ha)
t3 = tensor(Id, Id, Id, T)
cnot01 = tensor(CNOT, Id, Id)
cnot23 = tensor(Id, Id, CNOT)


proj_plus_0 = tensor(mat.Pz0, Id, Id, Id)
proj_minus_0 = tensor(mat.Pz1, Id, Id, Id)
proj_plus_1 = tensor(Id, mat.Pz0, Id, Id)
proj_minus_1 = tensor(Id, mat.Pz1, Id, Id)
proj_plus_2 = tensor(Id, Id, mat.Pz0, Id)
proj_minus_2 = tensor(Id, Id, mat.Pz1, Id)
proj_plus_3 = tensor(Id, Id, Id, mat.Pz0)
proj_minus_3 = tensor(Id, Id, Id, mat.Pz1)


def fidelity(rho):
    fid = np.dot(np.dot(mat.H(mat.phiplus), rho), mat.phiplus)
    return float(fid[0, 0])


def bbpssw_step(input_fid):
    f = input_fid
    p_suc = f**2 + 2 * f * (1 - f) / 3 + 5 * (1 - f)**2 / 9
    output_fid = (f**2 + (1 - f)**2 / 9) / p_suc
    return output_fid, p_suc


def proj(phi):
    return np.dot(phi, H(phi))


def dejmps_step(input_state):
    lambda_00 = np.dot(np.dot(H(phiplus), input_state), phiplus)[0, 0]
    lambda_01 = np.dot(np.dot(H(psiplus), input_state), psiplus)[0, 0]
    lambda_10 = np.dot(np.dot(H(phiminus), input_state), phiminus)[0, 0]
    lambda_11 = np.dot(np.dot(H(psiminus), input_state), psiminus)[0, 0]
    p_suc = (lambda_00 + lambda_11)**2 + (lambda_01 + lambda_10)**2
    output_state = ((lambda_00**2 + lambda_11**2) * proj(phiplus) +
                    2 * lambda_00 * lambda_11 * proj(phiminus) +
                    (lambda_01**2 + lambda_10**2) * proj(psiplus) +
                    2 * lambda_01 * lambda_10 * proj(psiminus))
    return output_state, p_suc


def get_constant(input_state, depolarize, recurrence_steps):
    input_fid = fidelity(input_state)
    p_suc = 1.0
    if depolarize:
        fid = input_fid
        for i in range(recurrence_steps):
            fid, p_step = bbpssw_step(fid)
            p_suc *= p_step
    else:
        state = input_state
        for i in range(recurrence_steps):
            state, p_step = dejmps_step(state)
            p_suc *= p_step
        fid = fidelity(state)
    const = p_suc**(1 / recurrence_steps) * (fid - input_fid)
    return float(const)


class EPPEnv(AbstractEnvironment):
    """
    """
    n_actions = 16

    def __init__(self, q_intial_noise=0.8):
        self.q = q_intial_noise

        aux = np.dot(phiplus, H(phiplus))
        self.state = mat.reorder(tensor(aux, aux), [0, 2, 1, 3])
        self.state = mat.wnoise_all(self.state, self.q)
        self.percept_now = []
        self.active_qubits = [0, 1, 2, 3]
        self.available_actions = [i for i in range(14)]
        self.branch_probability = 1

    def multiverse_reward(self, partial_trial_list, depolarize=True, recurrence_steps=1):
        accepted_branches = filter(lambda x: x.env.percept_now[-1] == 14, partial_trial_list)
        env_list = [branch.env for branch in accepted_branches]
        if env_list == []:  # if no branches were accepted, give no reward
            return 0
        accepted_actions_lists = [env.percept_now for env in env_list]
        probability = np.sum([env.branch_probability for env in env_list])
        new_state = np.sum([env.branch_probability * env.get_pair_state() for env in env_list], axis=0)
        new_state = new_state / np.trace(new_state)
        if depolarize is True:
            fid = fidelity(new_state)
            pp = (4 * fid - 1) / 3
            new_state = np.dot(mat.phiplus, mat.H(mat.phiplus))
            new_state = mat.wnoise(new_state, 0, pp)
        for i in range(1, recurrence_steps):
            input_state = mat.tensor(new_state, new_state)
            input_state = mat.reorder(input_state, [0, 2, 1, 3])
            for env, action_list in zip(env_list, accepted_actions_lists):
                env.reset(input_state=input_state)
                for action in action_list:
                    env.move(action)
            probability *= np.sum([env.branch_probability for env in env_list])
            new_state = np.sum([env.branch_probability * env.get_pair_state() for env in env_list], axis=0)
            new_state = new_state / np.trace(new_state)
            if depolarize is True:
                fid = fidelity(new_state)
                pp = (4 * fid - 1) / 3
                new_state = np.dot(mat.phiplus, mat.H(mat.phiplus))
                new_state = mat.wnoise(new_state, 0, pp)
        my_env = EPPEnv()
        my_env.reset()
        initial_fidelity = fidelity(mat.ptrace(my_env.state, [1, 3]))
        delta_f = (fidelity(new_state) - initial_fidelity)
        if delta_f < 10**-15:
            reward = 0
        else:
            # this whole constant calculation really, really shouldn't happen every step
            aux = np.dot(phiplus, H(phiplus))
            start_state = mat.wnoise_all(aux, self.q)
            const = get_constant(start_state, depolarize=depolarize, recurrence_steps=recurrence_steps)
            reward = probability * delta_f / const
            # print(probability, (fidelity(new_state) - initial_fidelity))
            # print(accepted_actions_lists)
        return reward

    def reset(self, input_state=None):  # makes it easy to initialize with different starting states for the meta-analysis
        if input_state is not None:
            self.state = input_state
        else:
            aux = np.dot(phiplus, H(phiplus))
            self.state = mat.reorder(tensor(aux, aux), [0, 2, 1, 3])
            self.state = mat.wnoise_all(self.state, self.q)
        self.percept_now = []
        self.active_qubits = [0, 1, 2, 3]
        self.available_actions = [i for i in range(14)]
        self.branch_probability = 1
        return self.percept_now, {"available_actions": self.available_actions}

    def action_from_index(self, index):
        if index == OUTCOME_PLUS_0 or index == OUTCOME_MINUS_0:
            return MEASURE_0
        elif index == OUTCOME_PLUS_1 or index == OUTCOME_MINUS_1:
            return MEASURE_1
        elif index == OUTCOME_PLUS_2 or index == OUTCOME_MINUS_2:
            return MEASURE_2
        elif index == OUTCOME_PLUS_3 or index == OUTCOME_MINUS_3:
            return MEASURE_3
        else:
            return index

    def get_pair_state(self):
        if len(self.active_qubits) != 2:
            raise ValueError("State cannot be reduced to a pair yet. This should never happen.")
        aux = [0, 1, 2, 3]
        aux.remove(self.active_qubits[0])
        aux.remove(self.active_qubits[1])
        return mat.ptrace(self.state, aux)

    def is_splitting_action(self, action):
        if action == MEASURE_0:
            return True, [OUTCOME_PLUS_0, OUTCOME_MINUS_0]
        elif action == MEASURE_1:
            return True, [OUTCOME_PLUS_1, OUTCOME_MINUS_1]
        elif action == MEASURE_2:
            return True, [OUTCOME_PLUS_2, OUTCOME_MINUS_2]
        elif action == MEASURE_3:
            return True, [OUTCOME_PLUS_3, OUTCOME_MINUS_3]
        else:
            return False, [action]

    def _remove_actions(self, actions):
        for action in actions:
            try:
                self.available_actions.remove(action)
            except ValueError:
                continue

    def _apply(self, operator):
        # print(self.state)
        # print(operator)
        return np.dot(np.dot(operator, self.state), H(operator))

    def _measure_random(self, qubit):
        if qubit == 0:
            aux1 = self._apply(proj_plus_0)
            aux2 = self._apply(proj_minus_0)
        elif qubit == 1:
            aux1 = self._apply(proj_plus_1)
            aux2 = self._apply(proj_minus_1)
        elif qubit == 2:
            aux1 = self._apply(proj_plus_2)
            aux2 = self._apply(proj_minus_2)
        elif qubit == 3:
            aux1 = self._apply(proj_plus_3)
            aux2 = self._apply(proj_minus_3)
        p_plus = float(np.trace(aux1))
        p_minus = float(np.trace(aux2))
        if np.random.random() < p_plus:
            state = aux1 / p_plus
            outcome = 0
            self.branch_probability *= p_plus
        else:
            state = aux2 / p_minus
            outcome = 1
            self.branch_probability *= p_minus
        return state, outcome

    def _measure(self, projector):
        state = self._apply(projector)
        probability = float(np.trace(state))
        # print(probability)
        self.state = state / probability
        self.branch_probability *= probability
        return

    def _update_actions(self, qubit):
        if qubit == 0:
            self._remove_actions(ACTIONS_Q0)
            self._remove_actions([MEASURE_1])
        elif qubit == 1:
            self._remove_actions(ACTIONS_Q1)
            self._remove_actions([MEASURE_0])
        elif qubit == 2:
            self._remove_actions(ACTIONS_Q2)
            self._remove_actions([MEASURE_3])
        elif qubit == 3:
            self._remove_actions(ACTIONS_Q3)
            self._remove_actions([MEASURE_2])
        try:
            self.active_qubits.remove(qubit)
        except ValueError:
            pass

        if len(self.active_qubits) == 2 and ACCEPT not in self.available_actions:
            self.available_actions += [ACCEPT, REJECT]

    def move(self, action):
        episode_finished = 0
        if action in [MEASURE_0, MEASURE_1, MEASURE_2, MEASURE_3]:
            warn("It is not expected to call the move method with a measure action. Meta-analysis of deterministic branches is the expected operation mode. This will now output random measurement result.")
        else:
            self.percept_now += [action]

        if action == H_0:
            self.state = self._apply(h0)
        elif action == T_0:
            self.state = self._apply(t0)
        elif action == H_1:
            self.state = self._apply(h1)
        elif action == T_1:
            self.state = self._apply(t1)
        elif action == H_2:
            self.state = self._apply(h2)
        elif action == T_2:
            self.state = self._apply(t2)
        elif action == H_3:
            self.state = self._apply(h3)
        elif action == T_3:
            self.state = self._apply(t3)
        elif action == CNOT_01:
            self.state = self._apply(cnot01)
        elif action == CNOT_23:
            self.state = self._apply(cnot23)
        elif action == MEASURE_0:
            self.state, outcome = self._measure_random(self.state, 0)
            if outcome == 0:
                self.percept_now += [OUTCOME_PLUS_0]
            else:
                self.percept_now += [OUTCOME_MINUS_0]
            self._update_actions(0)
        elif action == MEASURE_1:
            self.state, outcome = self._measure_random(self.state, 1)
            if outcome == 0:
                self.percept_now += [OUTCOME_PLUS_1]
            else:
                self.percept_now += [OUTCOME_MINUS_1]
            self._update_actions(1)
        elif action == MEASURE_2:
            self.state, outcome = self._measure_random(self.state, 2)
            if outcome == 0:
                self.percept_now += [OUTCOME_PLUS_2]
            else:
                self.percept_now += [OUTCOME_MINUS_2]
            self._update_actions(2)
        elif action == MEASURE_3:
            self.state, outcome = self._measure_random(self.state, 3)
            if outcome == 0:
                self.percept_now += [OUTCOME_PLUS_3]
            else:
                self.percept_now += [OUTCOME_MINUS_3]
            self._update_actions(0)
        elif action == ACCEPT:
            episode_finished = 1
        elif action == REJECT:
            episode_finished = 1
        elif action == OUTCOME_PLUS_0:
            self._measure(proj_plus_0)
            self._update_actions(0)
        elif action == OUTCOME_MINUS_0:
            self._measure(proj_minus_0)
            self._update_actions(0)
        elif action == OUTCOME_PLUS_1:
            self._measure(proj_plus_1)
            self._update_actions(1)
        elif action == OUTCOME_MINUS_1:
            self._measure(proj_minus_1)
            self._update_actions(1)
        elif action == OUTCOME_PLUS_2:
            self._measure(proj_plus_2)
            self._update_actions(2)
        elif action == OUTCOME_MINUS_2:
            self._measure(proj_minus_2)
            self._update_actions(2)
        elif action == OUTCOME_PLUS_3:
            self._measure(proj_plus_3)
            self._update_actions(3)
        elif action == OUTCOME_MINUS_3:
            self._measure(proj_minus_3)
            self._update_actions(3)

        # Note: reward is always zero as reward is calculated from meta_analysis
        return self.percept_now, 0, episode_finished, {"available_actions": self.available_actions}
