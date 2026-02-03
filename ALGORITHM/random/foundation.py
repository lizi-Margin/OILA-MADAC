import numpy as np
from config import GlobalConfig

class AlgorithmConfig:
    preserve = ''

class RandomController(object):
    def __init__(self, n_agent, n_thread, mcv=None, team=None):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.mcv = mcv
        self.n_action = GlobalConfig.ScenarioConfig.n_actions
        if isinstance(self.n_action, (int, float)):
            self.n_action = (self.n_action,)

    def interact_with_env(self, StateRecall):
        obs = StateRecall['Latest-Obs']
        P = StateRecall['ENV-PAUSE']
        active_thread_obs = obs[~P]
        # actions = np.random.randint(low=0,high=self.n_action, size=(self.n_thread, self.n_agent, 1))
        actions = np.random.randint(low=(0,)*len(self.n_action),high=self.n_action, size=(self.n_thread, self.n_agent, len(self.n_action)))
        StateRecall['_hook_'] = None
        return actions, StateRecall 

