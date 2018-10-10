from __future__ import division, print_function
from general_interaction import Interaction as GeneralInteraction


class Interaction(GeneralInteraction):
    """A simple interaction class with minimal options.

    Attributes
    ----------
    agent : Agent
    environment: Environment
    """
    def __init__(self, agent, environment):
        GeneralInteraction.__init__(self, agent, environment)

    def single_learning_life(self, n_trials, max_steps_per_trial):
        """Run consecutive trials between agent and environment.

        Parameters
        ----------
        n_trials : int
            number of trials to be executed
        max_steps_per_trial : int
            limits the maximum number of steps per trial
        """
        reward = 0
        done = 0
        info = {}
        for i_trial in range(1, n_trials + 1):
            setup = self.env.reset()
            if not isinstance(setup, tuple):
                observation = setup
            else:
                observation = setup[0]
                info.update(setup[1])
            for i_step in range(1, max_steps_per_trial + 1):
                observation, reward, done, action, info = self.single_interaction_step_PS(observation, reward, done, info)
                if done:
                    print("trial %d: %d steps" % (i_trial, i_step))
                    break
