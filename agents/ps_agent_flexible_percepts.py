"""
Copyright 2018 Alexey Melnikov and Katja Ried.
Copyright 2020 Julius Wallnöfer
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

Original code written by Katja Ried, implementing ideas from

'Projective simulation for artificial intelligence'
Hans J. Briegel & Gemma De las Cuevas
Scientific Reports 2, Article number: 400 (2012) doi:10.1038/srep00400

and

'Projective Simulation for Classical Learning Agents: A Comprehensive Investigation'
Julian Mautner, Adi Makmal, Daniel Manzano, Markus Tiersch & Hans J. Briegel
New Generation Computing, Volume 33, Issue 1, pp 69-114 (2015) doi:10.1007/s00354-015-0102-0

Modifications by Julius Wallnöfer:
    * Modifications to basic PS agent as outlined there.
    * Rewrite this flexible agent as subclass of the BasicPSAgent
"""
import numpy as np
from .ps_min_agent import BasicPSAgent
from time import time


class FlexiblePerceptsPSAgent(BasicPSAgent):
    """PS Agent without predefined percepts
    """
    def __init__(self, n_actions, ps_gamma, ps_eta, policy_type, ps_alpha, brain_type="dense", reset_glow=False):
        BasicPSAgent.__init__(self, n_actions, [0], ps_gamma, ps_eta, policy_type, ps_alpha, brain_type, reset_glow)
        self.percept_dict = {}
        self.temporary_percepts = {}

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

        if dict_key not in self.percept_dict:
            if dict_key not in self.temporary_percepts:
                self.temporary_percepts[dict_key] = len(self.percept_dict) + len(self.temporary_percepts)
            return self.temporary_percepts[dict_key]
            # # add new percept
            # self.percept_dict[dict_key] = self.n_percepts
            # self.n_percepts += 1
            # self.brain.add_percept()
        else:
            return self.percept_dict[dict_key]

    def _learning(self, reward_now):  # learning and forgetting
            if self.ps_gamma != 0:
                self.brain.decay(self.ps_gamma)
            if reward_now != 0:
                for new_percept in self.temporary_percepts:
                    self.n_percepts += 1
                    self.brain.add_percept()
                self.percept_dict.update(self.temporary_percepts)
                self.temporary_percepts = {}
                self.brain.update_g_matrix(self.ps_eta, self.history_since_last_reward)
                self.history_since_last_reward = []
                self.brain.update_h_matrix(reward_now)

    def deliberate_and_learn(self, observation, reward, episode_finished, info={}):  # this variant does nothing with info
        """Learn according to reward, then select and return next action.

        Parameters
        ----------
        observation : int or list of ints
            The observation provided by the environment in the same format
            as `n_percepts_multi` at init.
        reward : float
            The reward for the previously selected action.
        episode_finished : int
            1 if this is a new trial, else 0
        info : dict
            Dictionary of additional information passed to the agent.
            This basic variant does nothing with it.

        Returns
        -------
        int
            The action selected.

        """
        self._learning(reward)
        if episode_finished and self.reset_glow:
            self.history_since_last_reward = []
            self.brain.reset_glow()
            self.temporary_percepts = {}  # discard percepts that will not be rewarded
        percept_now = self._percept_preprocess(observation)
        action = self._policy(percept_now)
        self.history_since_last_reward += [(action, percept_now)]
        return action
