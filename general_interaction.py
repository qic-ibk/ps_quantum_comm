from __future__ import division, print_function
import numpy as np

class Interaction(object):
    
    def __init__(self, agent_type, agent, environment, video_bool, **userconfig):
        self.agent_type = agent_type
        self.agent = agent
        self.env = environment
        self.video_bool = video_bool
        
    def single_learning_life(self, n_trials, max_steps_per_trial):
        learning_curve = np.zeros(n_trials)
        for i_trial in range(n_trials):
#            print(i_trial)
            reward_trial = 0
            observation = self.env.reset()
            if i_trial == n_trials - 1:
                last_trial_history = []
            for t in range(max_steps_per_trial):
                old_observation = observation
                if self.agent_type in ('PS', 'PS-basic'):
                    observation, reward, done, action = self.single_interaction_step_PS(observation)
                reward_trial += float(reward)
                if i_trial == n_trials - 1:
                    last_trial_history += [(old_observation, action)]
                if done:
                    learning_curve[i_trial] = reward_trial/(t+1)
                    break
        return learning_curve, last_trial_history
        
    def single_interaction_step_PS(self, observation):
        percept_now = self.agent.percept_preprocess(observation)
        action = self.agent.policy(percept_now)
#        print("Agent policy step finished.")
        #print(observation, percept_now, action) #, end=" "
        observation, reward, done = self.env.move(action)
#        print("Environment move finished.")
        #print(reward)
        self.agent.learning(reward)
#        print("Agent learning step finished.")
        #print(observation, reward, done) # print history
        return observation, reward, done, action
