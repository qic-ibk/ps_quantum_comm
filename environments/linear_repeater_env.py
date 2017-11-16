# -*- coding: utf-8 -*-
"""
@author: julius

issue list:
indexing system is a mess
"""

from __future__ import division, print_function
import numpy as np

SEND_Q0_LEFT = 0
SEND_Q0_RIGHT = 1
SEND_Q1_LEFT = 2
SEND_Q1_RIGHT = 3
SEND_Q2_LEFT = 4
SEND_Q2_RIGHT = 5
SEND_Q3_LEFT = 6
SEND_Q3_RIGHT = 7
BELL_01 = 8
BELL_02 = 9
BELL_03 = 10
BELL_12 = 11
BELL_13 = 12
BELL_23 = 13
PURIFY_01 = 14
PURIFY_02 = 15
PURIFY_03 = 16
PURIFY_12 = 17
PURIFY_13 = 18
PURIFY_23 = 19


q = 0.57
index_array = np.array([[-1,0,1,2],[0,-1,3,4],[1,3,-1,5],[2,4,5,-1]])
qubits_from_index = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
list_of_pairwise_indices = [[0,1,2],[0,3,4],[1,3,5],[2,4,5]] #01,02,03,12,13,23


def fidelity_to_percept(fid): #discretization
    if fid < 0.5:
        return 0
    elif fid < 0.6:
        return 1
    elif fid < 0.7:
        return 2
    elif fid < 0.8:
        return 3
    elif fid < 0.9:
        return 4
    elif fid < 1.0:
        return 5
    else:
        return 6

def check_success(state):
    check = False
    for i in range(6):
        if state.fid[i] >= 0.6:
            qubit1, qubit2 = qubits_from_index[i]
            if (state.pos[qubit1] == 0 and state.pos[qubit2] == 2) or (state.pos[qubit1] == 2 and state.pos[qubit2] == 0):
                check = True
                break
    return check

class EnvState:
    def __init__(self):
        self.pos = np.array([0,0,2,2],dtype=int)
        self.fid = np.array([0,0,0,0,0,0],dtype=np.float64)
    
    def observation(self):
        fidpercept = np.array([fidelity_to_percept(f) for f in self.fid],dtype=int)
        return np.hstack((self.pos,fidpercept))
    
    def noise(self,qubit,q):
        for index in list_of_pairwise_indices[qubit]:
            f = self.fid[index]
            if f != 0: #still not sure if I'm a fan of using 0 as "no entanglement"
                qq = (4*f - 1)/3
                self.fid[index] = q*qq + (1-q*qq)/4
    
    def bell(self,qubit1,qubit2): #the entanglement swapping part of this suffers massively from the wonky indexing system
        if self.pos[qubit1] == self.pos[qubit2]:
            entswap_qubit1 = -1
            entswap_qubit2 = -1
            for index in list_of_pairwise_indices[qubit1]:
                if self.fid[index] != 0:
                    aux = qubits_from_index[index]
                    entswap_qubit1 = aux[aux != qubit1][0]
                    entswap_fid1 = self.fid[index]
                    break
            for index in list_of_pairwise_indices[qubit2]:
                if self.fid[index] != 0:
                    aux = qubits_from_index[index]
                    entswap_qubit2 = aux[aux != qubit2][0]
                    entswap_fid2 = self.fid[index]
            if entswap_qubit1 != -1 and entswap_qubit2 != -1:
                q1 = (4*entswap_fid1 - 1)/3
                q2 = (4*entswap_fid2 - 1)/3
                self.fid[index_array[entswap_qubit1,entswap_qubit2]] = q1*q2 + (1-q1*q2)/4
            del_list = list_of_pairwise_indices[qubit1] + list_of_pairwise_indices[qubit2]
            for index in del_list:
                self.fid[index] = 0
            self.fid[index_array[qubit1,qubit2]] = 1
    
    def purify(self,qubit1,qubit2):
        index = index_array[qubit1,qubit2]
        f = self.fid[index]
        if f != 0:
            self.fid[index]= (f**2 + (1-f)**2/9)/(f**2 + 2*f*(1-f)/3 + 5*(1-f)**2/9)
    


class TaskEnvironment(object):
    '''
    
    '''
    
    def __init__(self, **userconfig):
        self.n_actions = 20
        self.n_percepts = np.array([3]*4 + [7]*6) #positions and pair-wise coarse-grained fidelity
        self.state = EnvState()

    def actions(self): #why is this a method?
        return self.n_actions
        
    def percepts(self):
        return self.n_percepts
    
    def reset(self):
        self.state = EnvState()
        return self.state.observation()
    
    def move(self,action): 
        if action == SEND_Q0_LEFT:
            if self.state.pos[0] != 0:
                self.state.pos[0] -= 1
                self.state.noise(0,q)
        elif action == SEND_Q0_RIGHT:
            if self.state.pos[0] != 2:
                self.state.pos[0] += 1
                self.state.noise(0,q)
        elif action == SEND_Q1_LEFT:
            if self.state.pos[1] != 0:
                self.state.pos[1] -= 1
                self.state.noise(1,q)
        elif action == SEND_Q1_RIGHT:
            if self.state.pos[1] != 2:
                self.state.pos[1] += 1
                self.state.noise(1,q)
        elif action == SEND_Q2_LEFT:
            if self.state.pos[2] != 0:
                self.state.pos[2] -= 1
                self.state.noise(2,q)
        elif action == SEND_Q2_RIGHT:
            if self.state.pos[2] != 2:
                self.state.pos[2] += 1
                self.state.noise(2,q)
        elif action == SEND_Q3_LEFT:
            if self.state.pos[3] != 0:
                self.state.pos[3] -= 1
                self.state.noise(3,q)
        elif action == SEND_Q3_RIGHT:
            if self.state.pos[3] != 2:
                self.state.pos[3] += 1
                self.state.noise(3,q)
        elif action == BELL_01:
            self.state.bell(0,1)
        elif action == BELL_02:
            self.state.bell(0,2)
        elif action == BELL_03:
            self.state.bell(0,3)
        elif action == BELL_12:
            self.state.bell(1,2)
        elif action == BELL_13:
            self.state.bell(1,3)
        elif action == BELL_23:
            self.state.bell(2,3)
        elif action == PURIFY_01:
            self.state.purify(0,1)
        elif action == PURIFY_02:
            self.state.purify(0,2)
        elif action == PURIFY_03:
            self.state.purify(0,3)
        elif action == PURIFY_12:
            self.state.purify(1,2)
        elif action == PURIFY_13:
            self.state.purify(1,3)
        elif action == PURIFY_23:
            self.state.purify(2,3)
        
        if check_success(self.state):
            reward = 1
            episode_finished = 1
        else:
            reward = 0
            episode_finished = 0
        return self.state.observation(), reward, episode_finished
        