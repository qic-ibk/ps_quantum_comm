import numpy as np
import qutip as qt

class TaskEnvironment(object):
	"""Quantum Network Environment implementation"""
	
	def __init__(self, n_qubits, line_length, **userconfig):
		# local operations
		self.n_actions = 4 #0 purify 1-2 #1 purify 2-3 #2 purify 1-3 #3 merge 2
		self.reward_value = 1
		self.max_actions = 5 # max number of actions per trial
		
		self.n_percepts = np.full(self.max_actions, self.n_actions+1, np.int32) # sequence of actions, dimensions
		self.next_state = np.full(self.max_actions, 0, np.int32)
		#self.initial_cost = 0
		self.quantum_state = np.full(4, 0, np.int32) # toy description [pur_12, pur_23, pur_13, merge_2]
		self.fidelity_state = [0.75, 0.75, 0] # toy description fidelity of [pur_12, pur_23, pur_13]
		#self.target_fidelity = 0.95
		
	def actions(self):
		return self.n_actions
		
	def percepts(self):
		return self.n_percepts
		
	def reset(self):
		self.next_state = np.full(self.max_actions, 0, np.int32)
		self.quantum_state = np.full(4, 0, np.int32)
		return self.next_state
		
	def move(self, action):
		reward = 0
		episode_finished = 0
		action_position = np.count_nonzero(self.next_state)
		if action_position < self.max_actions:
			self.next_state[action_position] = action+1 # next state is previous state + new action
			if action==0:
				self.quantum_state[0] = 1
			elif action==1:
				self.quantum_state[1] = 1
			elif (action==2) and (self.quantum_state[0]==1) and (self.quantum_state[1]==1) and (self.quantum_state[3]==1):
				self.quantum_state[2] = 1
				reward = self.reward_value
				episode_finished = 1
			elif (action==3) and (self.quantum_state[0]==1) and (self.quantum_state[1]==1):
				self.quantum_state[3] = 1
		else:
			episode_finished = 1
			
		return self.next_state, reward, episode_finished
		