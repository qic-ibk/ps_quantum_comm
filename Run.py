#import platform
#print('Python version:', platform.python_version())

import os, inspect # for current directory
import numpy as np
import sys # for importing agents and environments
sys.path.insert(0, 'agents')
sys.path.insert(0, 'environments')

# performance evaluation
n_agents = 10
n_trials = 10
max_steps_per_trial = 1000000

# Choose an environment
environment_here = 'JW_GridWorld' # don't forget to set ps_eta = 1 for Invasion_Game, Driver_Game (they don't have temporal correlations)
# Choose an agent
agent_here = 'PS-basic'

# Import an environment
if environment_here == 'Driver_Game': 
	from driver_game_env import *
elif environment_here == 'Invasion_Game': 
	from invasion_game_env import *
elif environment_here == 'Neverending_Color': 
	from neverending_color_env import *
elif environment_here == 'Quantum_Networks_1': 
	from q_network_env_1 import *
elif environment_here == 'Quantum_Networks_2': 
	from q_network_env_2 import *
elif environment_here == 'JW_GridWorld':
    from JW_GridWorld_env import *

# Import an agent
if agent_here == 'PS-basic': 
	from ps_min_agent import * # import the basic PS agent
	
# Import interaction class
from general_interaction import *

# parameter to 1D encoding
n_param_scan = 1
average_param_performance = np.zeros(n_param_scan)
for i_param_scan in range(n_param_scan):
	
	ps_eta = i_param_scan * 0.001
	
	average_learning_curve = np.zeros(n_trials)
	for i_agent in range(n_agents): 
		# Inialize an environmemnt	
		if environment_here in ('Driver_Game', 'Invasion_Game', 'JW_GridWorld'): 
			env = TaskEnvironment()
		elif environment_here == 'Neverending_Color': 
			env = TaskEnvironment(2, n_trials, 1) # n_actions, n_trials, reward_value
		elif environment_here == 'Quantum_Networks_1': 
			env = TaskEnvironment(2, 2) # n_qubits, line_length
		elif environment_here == 'Quantum_Networks_2': 
			env = TaskEnvironment(2, 2) # n_qubits, line_length
		# Inialize an agent
		if agent_here == 'PS-basic': 
			agent = BasicPSAgent(env.actions(), env.percepts(), 0, 0.2, 'standard', 1) 
			# n_actions, n_percepts_multi, ps_gamma, ps_eta, policy_type ('standard' or 'softmax'), ps_alpha
			
		interaction = Interaction(agent_here, agent, env, False)
		learning_curve = interaction.single_learning_life(n_trials, max_steps_per_trial)
		average_learning_curve += learning_curve/n_agents	
	print('Total average reward per action\n', average_learning_curve)
	print('Number of steps to the goal (only makes sense when the goal is always reached in each trial, e.g. grid world or q. networks)\n', 1/(average_learning_curve + pow(10,-10)) ) # pow(10,-10) to awoid division by 0
	average_param_performance[i_param_scan] = average_learning_curve[n_trials-1]

# Saving files
current_file_directory = os.path.dirname(os.path.abspath(__file__))
if n_agents == 1:
	np.savetxt(current_file_directory+'/results'+'/h_matrix', agent.h_matrix_output(), fmt='%.2f', delimiter=',')
	np.savetxt(current_file_directory+'/results'+'/g_matrix', agent.g_matrix_output(), fmt='%.3f', delimiter=',')
	np.savetxt(current_file_directory+'/results'+'/learning_curve', average_learning_curve, fmt='%.10f', delimiter=',')
else:
	np.savetxt(current_file_directory+'/results'+'/param_performance', average_param_performance, fmt='%.10f', delimiter=',')
