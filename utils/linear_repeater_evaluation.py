# -*- coding: utf-8 -*-
"""
Copyright 2020 Julius Wallnöfer

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import json
import linear_repeater_env as lr

def load_array(path_to_file):
    with open(path_to_file,"r") as f:
        return np.loadtxt(f)

def load_history(path_to_file):
    with open(path_to_file,"r") as f:
        return json.load(f)

n_agents = 100


all_learning_curves = np.vstack([load_array(str(i) + "/learning_curve.txt") for i in range(n_agents)])

n_trials = all_learning_curves.shape[1]

average_learning_curve = np.sum(all_learning_curves,axis=0)/n_agents
sigma_learning_curve = np.sqrt( np.sum(all_learning_curves**2, axis = 0)/n_agents - average_learning_curve**2) # matrix - vector subtracts row-wise
sigma_steps = 1/average_learning_curve**2 * sigma_learning_curve #propagation of error

plt.errorbar(np.arange(1,n_trials + 1), 1/(average_learning_curve + 10**(-10)), yerr = sigma_steps, fmt = 'o')
plt.ylim((0,10000))
plt.show(block=True)
plt.scatter(np.arange(1,n_trials + 1), 1/(average_learning_curve + 10**(-10)))
plt.ylim((0,10000))
plt.show(block= True)
#plt.plot(np.arange(1,n_trials + 1), 1/(average_learning_curve + 10**(-10)) )
#plt.show()

#for i in range(n_agents):
#    plt.scatter(np.arange(1,all_learning_curves.shape[1] + 1), 1/(all_learning_curves[i] + 10**(-10)))
#
#plt.show()


##step count histogram of solutions
#solutions_steps = [len(load_history(str(i) + "/last_trial_history.txt")) for i in range(n_agents)]
#solutions_min = np.min(solutions_steps)
#solutions_max = np.max(solutions_steps)
#
#plt.hist(solutions_steps, bins = [ solutions_min - 0.5 + x for x in range(solutions_max - solutions_min + 2)])
#plt.xlabel("number of steps in last trial")
#plt.ylabel("number of agents")
#plt.title("amount of steps in solution")
#plt.show(block = True)

solutions = [load_history(str(i) + "/last_trial_history.txt") for i in range(n_agents)]
print(np.argmin([len(x) for x in solutions]))

def detect_ent_swap(history):
    env = lr.TaskEnvironment()
    env.reset()
    for percept,action in history:
        if env.ent_swap_detection(action):
            return True
        env.move(action)
    return False

yes_ent_swap = []
no_ent_swap = []
for solution in solutions:
    if(detect_ent_swap(solution)):
        yes_ent_swap += [solution]
    else:
        no_ent_swap += [solution]

yes_steps = [len(i) for i in yes_ent_swap]
no_steps = [len(i) for i in no_ent_swap]

#print(len(yes_steps), len(no_steps))

solutions_min = np.min(yes_steps + no_steps)
solutions_max = np.max(yes_steps + no_steps)

plt.hist((yes_steps, no_steps), histtype= 'barstacked', bins = [ solutions_min - 0.5 + x for x in range(solutions_max - solutions_min + 2)], label=('entanglement swappping','no entanglement swapping'), color = ('red','blue'))
plt.xlabel("number of steps in last trial")
plt.ylabel("number of agents")
plt.title("eta = 0.05")
plt.legend(loc='upper right')
plt.show(block = True)



#print(np.argpartition(np.array(solutions_steps),3))
