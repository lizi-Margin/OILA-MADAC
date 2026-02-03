from config import GlobalConfig
from uhtk.UTIL.colorful import *

class AlgorithmBase():
    def __init__(self, n_agent, n_thread, mcv=None, team=None):
        self.n_thread = n_thread
        self.n_agent = n_agent
        self.team = team
        self.ScenarioConfig = GlobalConfig.ScenarioConfig
        self.mcv = mcv

    def interact_with_env(self, team_intel):
        raise NotImplementedError
