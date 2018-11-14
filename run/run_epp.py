from __future__ import division, print_function
import os, sys; sys.path.insert(0, os.path.abspath("."))
from environments.epp_env import EPPEnv
from meta_analysis_interaction import MetaAnalysisInteraction
from agents.ps_agent_changing_actions import ChangingActionsPSAgent
import matplotlib.pyplot as plt
import numpy as np

eta = 0
n_agents = 1
n_trials = 500

results = []
for i_agent in range(n_agents):
    env = EPPEnv()
    agent = ChangingActionsPSAgent(env.n_actions, ps_gamma=0, ps_eta=eta, policy_type="softmax", ps_alpha=1, brain_type="dense")
    interaction = MetaAnalysisInteraction(agent, env)
    res = interaction.single_learning_life(n_trials, True)
    results += [res]
# print(res["reward_curve"])
for res in results:
    reward_curve = res["reward_curve"] + 10**-30
    plt.scatter(np.arange(len(reward_curve)), reward_curve)
    plt.ylim(10**-30, np.max(reward_curve))
    plt.yscale("log")
    plt.show()
