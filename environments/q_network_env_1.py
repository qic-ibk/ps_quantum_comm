import numpy as np
import qutip as qt

class TaskEnvironment(object):
	"""Quantum Network Environment implementation"""
	
	def __init__(self, n_qubits, line_length, **userconfig):
		# local operations
		self.n_moves_per_qubit = 2 # 0 - left # 1 - right
		self.n_operations_per_qubit = 2 # 2 - X gate # 3 - H gate
		self.n_actions_per_qubit = self.n_moves_per_qubit + self.n_operations_per_qubit
		self.n_qubits = n_qubits
		# global operations
		self.n_positions = line_length
		self.n_cnots = self.n_positions
		
		self.n_actions = self.n_actions_per_qubit * self.n_qubits + self.n_cnots
		self.reward_value = 1
		self.n_position_percepts = np.full(self.n_qubits, self.n_positions, np.int) # position of the qubits
		self.n_q_state_percepts = np.array([2]) # state of qubits 0 - don't satisfy and 1 - satisfy
		self.n_percepts = np.concatenate((self.n_position_percepts, self.n_q_state_percepts), axis=0)
		self.initial_percept = np.array([0, 0, 0])
		
		#self.target_q_state = (qt.basis(4, 0) + qt.basis(4, 3))/np.sqrt(2)
		#print('target state', self.target_q_state)
		
	def actions(self):
		return self.n_actions
		
	def percepts(self):
		return self.n_percepts
		
	def reset(self):
		self.initial_percept = np.array([0, 0, 0]) # strange to do it
		self.next_state = self.initial_percept
		self.Quantum_State = qt.tensor(np.full(self.n_qubits, qt.basis(2, 0)))
		#print(self.Quantum_State)
		return self.next_state
		
	def analyzer_two_particles(self, state):
		target_distanse1 = state - np.array([0, self.n_positions-1])
		target_distanse2 = state - np.array([self.n_positions-1, 0])
		if (not any(target_distanse1)) or (not any(target_distanse2)):
			reward = self.reward_value
			episode_finished = 1
		else:
			reward = 0
			episode_finished = 0
		return [reward, episode_finished]
		
	def analyzer_Bell_pair(self, position_state, quantum_state, concurrence_rho):
		target_distanse1 = position_state - np.array([0, self.n_positions-1, position_state[2]])
		target_distanse2 = position_state - np.array([self.n_positions-1, 0, position_state[2]])
		if ((not any(target_distanse1)) or (not any(target_distanse2))) and (concurrence_rho > 0.9): #  and (quantum_state == self.target_q_state)
#			print(position_state)
#			print(quantum_state)
			reward = self.reward_value
			episode_finished = 1
		else:
			reward = 0
			episode_finished = 0
		return [reward, episode_finished]
		
	def move(self, action):
		if action < (self.n_actions-self.n_cnots): # local operations
			action_map = divmod(action, self.n_qubits) # (division, remainder) = (action, qubit)
			if action_map[0] < self.n_moves_per_qubit: # moving qubits around
				self.next_state[action_map[1]] += 2*action_map[0] - 1 # 'terminal state' is the next state
				# "walls"
				if (self.next_state[action_map[1]] < 0) or (self.next_state[action_map[1]] > self.n_positions-1):
					self.next_state[action_map[1]] -= 2*action_map[0] - 1
			else: # apply local rotations
				local_gate_list = np.full(self.n_qubits, qt.qeye(self.n_qubits))
				if action_map[0] == self.n_moves_per_qubit:
					local_gate_list[action_map[1]] = 1.j*qt.rx(np.pi) # X on a selected qubit
				elif action_map[0] == (self.n_moves_per_qubit+1):
					local_gate_list[action_map[1]] = qt.snot() # H on a selected qubit
				local_gate = qt.tensor(local_gate_list)
				self.Quantum_State = local_gate * self.Quantum_State
		else: # apply global operations
			cnot_position = action - (self.n_actions-self.n_cnots)
			if len(np.where( self.next_state == cnot_position )[0])==2: # apply a cnot
				self.Quantum_State = qt.cnot() * self.Quantum_State
				
		# concurence
		rho_state = self.Quantum_State * self.Quantum_State.dag()
		concurrence_rho = qt.concurrence(rho_state)
		
		if concurrence_rho > 0.9:
			self.next_state[self.n_qubits] = 1
		else:
			self.next_state[self.n_qubits] = 0
			
		# call analyzer
		reward, episode_finished = self.analyzer_Bell_pair(self.next_state, self.Quantum_State, concurrence_rho)
		return [self.next_state, reward, episode_finished]
		