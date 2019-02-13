from __future__ import division, print_function
import os, sys; sys.path.insert(0, os.path.abspath("."))
from environments.epp_env import EPPEnv
from meta_analysis_interaction import MetaAnalysisInteraction
from agents.ps_agent_changing_actions import ChangingActionsPSAgent
from multiprocessing import Pool
from time import time
import numpy as np
import os
import traceback


def assert_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


num_processes = 48  # change according to cluster computer you choose
num_agents = 100
n_trials = 100000
eta = 0
result_path = "results/epp_modified/raw/"


def run_epp(i, sparsity=10):
    np.random.seed()
    env = EPPEnv()
    agent = ChangingActionsPSAgent(env.n_actions, ps_gamma=0, ps_eta=eta, policy_type="softmax", ps_alpha=1, brain_type="dense")  # glow reset is handled by MetaAnalysis Interaction
    interaction = MetaAnalysisInteraction(agent, env)
    last_history_file = result_path + "last_trial_history_%d.txt" % i
    res = interaction.single_learning_life(n_trials, verbose_trial_count=False, last_history_file=last_history_file)
    reward_curve = res["reward_curve"]
    if sparsity != 1:
        reward_curve = reward_curve[::sparsity]
    np.save(result_path + "reward_curve_%d.npy" % i, reward_curve)
    print(str(i) + "; n_percepts: " + str(len(agent.percept_dict)))


class RunCallable(object):  # this solution is necessary because only top-level objects can be pickled
    def __init__(self):
        pass

    def __call__(self, i):
        try:
            return run_epp(i)
        except Exception:
            print("Exception occured in child process")
            traceback.print_exc()  # so exception in child process gets output on older python versions
            raise


if __name__ == "__main__":
    start_time = time()
    p = Pool(processes=num_processes)
    assert_dir(result_path)
    my_callable = RunCallable()
    p.map_async(my_callable, np.arange(num_agents))
    p.close()
    p.join()
    print("The whole script took %.2f minutes." % ((time() - start_time) / 60))
