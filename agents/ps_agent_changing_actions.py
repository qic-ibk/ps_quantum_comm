"""
"""

from .ps_agent_flexible_percepts import FlexiblePerceptsPSAgent


class ChangingActionsPSAgent(FlexiblePerceptsPSAgent):
    """
    """
    def __init__(self, n_actions, ps_gamma, ps_eta, policy_type, ps_alpha, brain_type="dense"):
        FlexiblePerceptsPSAgent.__init__(self, n_actions, ps_gamma, ps_eta, policy_type, ps_alpha, brain_type="dense")
        self.available_actions = [i for i in range(self.n_actions)]  # environments that do not provide action info, will have all actions available

    def _policy(self, percept_now):
        if self.policy_type == 'softmax':
            h_vector_now = self.ps_alpha * self.brain.get_h_vector(percept_now)
            h_vector_now_mod = h_vector_now - np.max(h_vector_now)
            h_vector_now_mod = h_vector_now_mod[self.available_actions]  # only consider available actions
            p_vector_now = np.exp(h_vector_now_mod) / np.sum(np.exp(h_vector_now_mod))
        elif self.policy_type == 'standard':
            h_vector_now = self.brain.get_h_vector(percept_now)
            h_vector_now_mod = h_vector_now[self.available_actions]  # only consider available actions
            p_vector_now = h_vector_now_mod / np.sum(h_vector_now_mod)
        action = np.random.choice(np.arange(self.n_actions), p=p_vector_now)
        return action

    def deliberate_and_learn(self, observation, reward, info):
        if "available_actions" in info:
            self.available_actions = info["available_actions"]
        return FlexiblePerceptsPSAgent.deliberate_and_learn(self, observation, reward, info)
