"""
Copyright 2018 Alexey Melnikov and Katja Ried.
Copyright 2020 Julius Wallnöfer, Alexey Melnikov
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

Original code for basic PS agent written by Alexey Melnikov and Katja Ried.
The ChangingActionsPSAgent written by Julius Wallnöfer, it modifies the basic
PS agent in the following ways:
    * Modifications to basic PS agent and flexible PS anent as outlined in their respective files.
    * Allows the available actions to change at every interaction step.
"""
import numpy as np
from warnings import warn
from .ps_agent_flexible_percepts import FlexiblePerceptsPSAgent


class ChangingActionsPSAgent(FlexiblePerceptsPSAgent):
    """
    """
    def __init__(self, n_actions, ps_gamma, ps_eta, policy_type, ps_alpha, brain_type="dense", reset_glow=False):
        FlexiblePerceptsPSAgent.__init__(self, n_actions, ps_gamma, ps_eta, policy_type, ps_alpha, brain_type, reset_glow)
        self.available_actions = [i for i in range(self.n_actions)]  # environments that do not provide action info, will have all actions available

    def _policy(self, percept_now):
        if self.policy_type == 'softmax':
            h_vector_now = self.ps_alpha * self.brain.get_h_vector(percept_now)
            h_vector_now_mod = h_vector_now[self.available_actions]  # only consider available actions
            h_vector_now_mod = h_vector_now_mod - np.max(h_vector_now_mod)
            p_vector_now = np.exp(h_vector_now_mod) / np.sum(np.exp(h_vector_now_mod))
        elif self.policy_type == 'standard':
            h_vector_now = self.brain.get_h_vector(percept_now)
            h_vector_now_mod = h_vector_now[self.available_actions]  # only consider available actions
            p_vector_now = h_vector_now_mod / np.sum(h_vector_now_mod)
        try:
            action = np.random.choice(self.available_actions, p=p_vector_now)
            return action
        except ValueError:
            return None

    def deliberate_and_learn(self, observation, reward, episode_finished, info):
        if "available_actions" in info:
            self.available_actions = info["available_actions"]
        else:
            warn("ChangingActionsPSAgent did not receive list of available actions.")

        if np.max(self.available_actions) >= self.n_actions:
            for i in range(self.n_actions, np.max(self.available_actions) + 1):
                self.n_actions += 1
                self.brain.add_action()

        return FlexiblePerceptsPSAgent.deliberate_and_learn(self, observation, reward, episode_finished, info)
