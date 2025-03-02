"""Includes the BasicPSAgent class.

Copyright 2018 Alexey Melnikov and Katja Ried.
Copyright 2020 Julius Wallnöfer, Alexey Melnikov
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

Original code written by Alexey Melnikov and Katja Ried, implementing ideas from

'Projective simulation for artificial intelligence'
Hans J. Briegel & Gemma De las Cuevas
Scientific Reports 2, Article number: 400 (2012) doi:10.1038/srep00400

and

'Projective Simulation for Classical Learning Agents: A Comprehensive Investigation'
Julian Mautner, Adi Makmal, Daniel Manzano, Markus Tiersch & Hans J. Briegel
New Generation Computing, Volume 33, Issue 1, pp 69-114 (2015) doi:10.1007/s00354-015-0102-0

Modifications by Julius Wallnöfer:
    * separate the storage mechanism of h- and glow-values ("brains") of the PS agent from the agent logic
    * add checks so the update operations on h- and glow-values only gets performed when necessary
    * option to reset glow matrix after every trial (essentially, if the agent is aware that trials are separate from each other)
    * minor performance improvements
"""

from __future__ import division, print_function
import numpy as np
from time import time


class BasicPSAgent(object):
    """Base class for PS agents. Includes only basic features.

    Parameters
    ----------
    n_actions : int
        Number of possible actions.
    n_percepts_multi : int or list of ints
        Number of percepts, possibly for different inputs.
    ps_gamma : float
        The forgetting parameter gamma.
    ps_eta : float
        The glow parameter eta.
    policy_type : str
        Selects with policy to use for converting h-values to probabilities.
        "standard" for standard policy or "softmax" for softmax policy.
    ps_alpha : float
        Parameter alpha for the softmax policy. Does nothing if
        `policy_type` is "standard".
    brain_type : str
        Which representation of the clip network to use. Supported types are
        "dense" for dense matrix
        "sparse" for sparse matrix
        "clip" for object oriented clip network
        (the default is "dense")
    reset_glow : bool
        If True, the glow matrix will be reset after every trial,
        i.e. when the deliberate_and_learn method receives episode_finished
        (the default is False).

    Attributes
    ----------
    agent_wait_time : float
        The initialisation time.
    n_percepts : type
        Number of possible percepts.
    brain : Brain
        A Brain object that represents the clip network.
    history_since_last_reward : list of tuples
        Collection of (observation, action) pairs since the last reward was
        received. This is only used for more efficient update of the glow and
        should not be interpreted as an actual memory.
    n_actions
    n_percepts_multi
    policy_type
    ps_gamma
    ps_eta
    ps_alpha
    reset_glow
    """

    def __init__(self, n_actions, n_percepts_multi, ps_gamma, ps_eta, policy_type, ps_alpha, brain_type="dense", reset_glow=False):
        self.agent_wait_time = time()
        self.n_actions = n_actions
        self.n_percepts_multi = n_percepts_multi
        self.n_percepts = np.prod(n_percepts_multi)
        self.policy_type = policy_type

        self.ps_gamma = ps_gamma
        self.ps_eta = ps_eta
        self.ps_alpha = ps_alpha

        if brain_type == "dense":
            from .brains.dense_brain import DenseBrain
            self.brain = DenseBrain(self.n_actions, self.n_percepts)
        elif brain_type == "sparse":
            from .brains.sparse_brain import SparseBrain
            self.brain = SparseBrain(self.n_actions, self.n_percepts)
        elif brain_type == "clip":
            from .brains.clip_brain import ClipBrain
            self.brain = ClipBrain(self.n_actions, self.n_percepts)
        else:
            raise ValueError("%s is not a supported brain_type" % brain_type)

        self.history_since_last_reward = []
        self.reset_glow = reset_glow

    def _percept_preprocess(self, observation):  # preparing for creating a percept
        percept = observation[0]
        for i_sum in range(1, observation.size):
            product = 1
            for i_product in range(i_sum):
                product = product * self.n_percepts_multi[i_product]
            percept += product * observation[i_sum]
        return percept

    def _policy(self, percept_now):  # action selection
        if self.policy_type == 'softmax':
            h_vector_now = self.ps_alpha * self.brain.get_h_vector(percept_now)
            h_vector_now_mod = h_vector_now - np.max(h_vector_now)
            p_vector_now = np.exp(h_vector_now_mod) / np.sum(np.exp(h_vector_now_mod))
        elif self.policy_type == 'standard':
            h_vector_now = self.brain.get_h_vector(percept_now)
            p_vector_now = h_vector_now / np.sum(h_vector_now)
        action = np.random.choice(np.arange(self.n_actions), p=p_vector_now)
#        # internally update the g-matrix
#        self.g_matrix = (1 - self.ps_eta) * self.g_matrix
#        self.g_matrix[action, percept_now] = 1
        return action

    def _learning(self, reward_now):  # learning and forgetting
            if self.ps_gamma != 0:
                self.brain.decay(self.ps_gamma)
            if reward_now != 0:
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
        # if (time() - self.agent_wait_time) / 60 > 5:
        #     self.agent_wait_time = time()
        #     print('Please wait... I am deliberating')

        self._learning(reward)
        if episode_finished and self.reset_glow:
            self.history_since_last_reward = []
            self.brain.reset_glow()
        percept_now = self._percept_preprocess(observation)
        action = self._policy(percept_now)
        self.history_since_last_reward += [(action, percept_now)]
        return action
