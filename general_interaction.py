"""
Copyright 2018 Alexey Melnikov and Katja Ried.
Copyright 2020 Julius Wallnöfer
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

Original simple interaction written by Katja Ried
Modification to collect more data, live display and to allow the use of the
ChangingActionPSAgent by Julius Wallnöfer
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt


class Interaction(object):

    def __init__(self, agent, environment):
        self.agent = agent
        self.env = environment

    def single_learning_life(self, n_trials, max_steps_per_trial, return_last_trial_history=True, env_statistics={}, verbose_trial_count=False, live_display=False):
        """Run consecutive trials between agent and environment.

        Parameters
        ----------
        n_trials : int
            number of trials to be executed
        max_steps_per_trial : int
            limits the maximum number of steps per trial
        return_last_trial_history : bool
            whether to include the last trial in the output (default: True)
        env_statistics : dict
            keys are used as keys for the `res` dict
            values are callables, usually environment methods
        verbose_trial_count : bool
            If True, prints a reminder every 1000 trials. (default: False)
        live_display : bool
            If True, displays the step curve as it becomes available.
            Not recommended for performance or more than one agent at a time.
            (default: False)

        Returns
        -------
        res : dict
            result dictionary that contains the following statistics

            - `"learning_curve"` : np.ndarray
                the learning curve, i.e. reward per step of each trial
            - `"last_trial_history"` : list of tuples
                key is only present if `return_last_trial_history` is True
                tuples are of the form (observation, action)
            - other keys as specified by `env_statistics` : list
        """
        learning_curve = np.zeros(n_trials)
        step_curve = np.zeros(n_trials, dtype=np.int)
        reward = 0
        episode_finished = 0
        info = {}
        res = {}
        if live_display is True:
            plt.ion()
        for stat in env_statistics:
            res[stat] = []
        for i_trial in range(n_trials):
            reward_trial = 0
            if verbose_trial_count and i_trial % 1000 == 0:
                print("Interaction is now starting trial %d of %d." % (i_trial, n_trials))
            setup = self.env.reset()
            if not isinstance(setup, tuple):
                observation = setup
            else:
                observation = setup[0]
                info.update(setup[1])
            if i_trial == n_trials - 1:
                last_trial_history = []
            for i_step in range(1, max_steps_per_trial + 1):
                old_observation = observation
                observation, reward, episode_finished, action, info = self.single_interaction_step_PS(observation, reward, episode_finished, info)
                reward_trial += float(reward)
                if return_last_trial_history and i_trial == n_trials - 1:
                    last_trial_history += [(old_observation, action)]
                if episode_finished:
                    break
            learning_curve[i_trial] = reward_trial / i_step
            step_curve[i_trial] = i_step
            for stat in env_statistics:
                res[stat] += [env_statistics[stat]()]
            if live_display is True and i_trial % 50 == 0:
                plt.cla()
                plt.yscale("log")
                # plt.ylim(-1, 51)
                # aux = np.copy(step_curve)
                # aux[learning_curve == 0] = 50
                # aux[i_trial:] = 0
                aux = learning_curve
                plt.scatter(np.arange(1, n_trials + 1)[::1], aux[::1])
                plt.pause(0.5)
        res["learning_curve"] = learning_curve
        res["step_curve"] = step_curve
        if return_last_trial_history:
            res["last_trial_history"] = last_trial_history
        return res

    def single_interaction_step_PS(self, observation, reward, episode_finished, info):
        action = self.agent.deliberate_and_learn(observation, reward, episode_finished, info)
        res = self.env.move(action)
        if len(res) == 3:  # backwards compatibility
            res += ({},)
        observation, reward, episode_finished, info = res
        return observation, reward, episode_finished, action, info
