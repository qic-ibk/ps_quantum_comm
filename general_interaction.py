from __future__ import division, print_function
import numpy as np


class Interaction(object):

    def __init__(self, agent, environment):
        self.agent = agent
        self.env = environment

    def single_learning_life(self, n_trials, max_steps_per_trial, return_last_trial_history=False, env_statistics={}):
        """Run consecutive trials between agent and environment.

        Parameters
        ----------
        n_trials : int
            number of trials to be executed
        max_steps_per_trial : int
            limits the maximum number of steps per trial
        return_last_trial_history : bool
            whether to include the last trial in the output (default: False)
        env_statistics : dict
            keys are used as keys for the `res` dict
            values are callables, usually environment methods

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
        reward = 0
        done = 0
        info = {}
        res = {}
        for stat in env_statistics:
            res[stat] = []
        for i_trial in range(n_trials):
            reward_trial = 0
            if i_trial % 1000 == 0:
                print("Interaction is now starting trial %d of %d." % (i_trial, n_trials))
            setup = self.env.reset()
            if not isinstance(setup, tuple):
                observation = setup
            else:
                observation = setup[0]
                info.update(setup[1])
            if i_trial == n_trials - 1:
                last_trial_history = []
            for t in range(max_steps_per_trial):
                old_observation = observation
                observation, reward, done, action, info = self.single_interaction_step_PS(observation, reward, done, info)
                reward_trial += float(reward)
                if return_last_trial_history and i_trial == n_trials - 1:
                    last_trial_history += [(old_observation, action)]
                if done:
                    learning_curve[i_trial] = reward_trial / (t + 1)
                    for stat in env_statistics:
                        res[stat] += [env_statistics[stat]()]
                    break
        res["learning_curve"] = learning_curve
        if return_last_trial_history:
            res["last_trial_history"] = last_trial_history
        return res

    def single_interaction_step_PS(self, observation, reward, done, info):
        action = self.agent.deliberate_and_learn(observation, reward, done, info)
        res = self.env.move(action)
        if len(res) == 3:  # backwards compatibility
            res += ({},)
        observation, reward, done, info = res
        return observation, reward, done, action, info
