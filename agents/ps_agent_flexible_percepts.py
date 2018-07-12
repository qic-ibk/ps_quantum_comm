"""
"""
import numpy as np
from .ps_min_agent import BasicPSAgent
from time import time


class FlexiblePerceptsPSAgent(BasicPSAgent):
    """PS Agent without predefined percepts
    """
    def __init__(self, n_actions, ps_gamma, ps_eta, policy_type, ps_alpha, matrix_type="dense"):
        self.agent_wait_time = time()
        self.n_actions = n_actions
        self.n_percepts = 0  # no initial preprogrammed percepts
        self.policy_type = policy_type

        self.ps_gamma = ps_gamma
        self.ps_eta = ps_eta
        self.ps_alpha = ps_alpha

        if matrix_type == "dense":
            from .brains.dense_brain import DenseBrain
            self.brain = DenseBrain(self.n_actions, n_percepts=0)
        elif matrix_type == "sparse":
            from .brains.sparse_brain import SparseBrain
            self.brain = SparseBrain(self.n_actions, n_percepts=0)
        else:
            raise ValueError("%s is not a supported matrix_type" % matrix_type)

        self.history_since_last_reward = []
        self.percept_dict = {}

    def _percept_preprocess(self, observation):
        # dictionary keys must be immutable
        if type(observation) in [str, int, bool, float, np.float, tuple]:
            dict_key = observation
        elif type(observation) == list:
            dict_key = tuple(observation)
        elif type(observation) == np.ndarray:
            dict_key = tuple(observation.flatten())
        else:
            raise TypeError('Observation is of a type not supported as dictionary key. You may be able to add a way of handling this type.')

        if dict_key not in self.percept_dict.keys():
            # add new percept
            self.percept_dict[dict_key] = self.n_percepts
            self.n_percepts += 1
            self.brain.add_percept()
        return self.percept_dict[dict_key]
