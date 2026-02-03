# please register this ScenarioConfig into MISSION/env_router.py
from typing import Union

class ScenarioConfig(object):  #CONF_SYSTEM#
    # <Part 1> Needed by the hmp core #
    AGENT_ID_EACH_TEAM = [[0, 1], [2, 3]]

    TEAM_NAMES = ['ALGORITHM.None->None', 'ALGORITHM.None->None']


    # <Part 2> Needed by env itself #
    MaxEpisodeStep = 100
    render = False

    # <Part 3> Needed by ALGORITHM #
    EntityOriented = False
    obs_shape: Union[tuple, list] = [10,]  # use list because jsonc does not support tuple, if you don't want to change this, tuple is also okay

    n_actions = 2

    AvailActProvided = False  # info['Avail-Act']

def make_env(env_name, rank):
    return BaseEnv(rank)

class BaseEnv(object):
    def __init__(self, rank) -> None:
        ############################# Deprecated, use ScenarioConfig to pass obs/action space is more flexible
        self.observation_space = None
        self.action_space = None
        #############################
        self.rank = rank

    def step(self, act):
        # obs: a np.ndarray with shape (n_agent, ...)
        # reward: a np.ndarray with shape (n_agent, 1)
        # done: a np.ndarray (.all() method will be used)
        # info: a dict
        raise NotImplementedError
        return (obs, reward, done, info)

    def reset(self):
        # obs: a Tensor with shape (n_agent, ...)
        # info: a dict
        raise NotImplementedError
        return obs, info






#########################################################################################################################################
TOLERANCE = 20
def check_ScenarioConfig(SC: ScenarioConfig):
    global TOLERANCE
    if TOLERANCE <= 0: return

    assert hasattr(SC, "AGENT_ID_EACH_TEAM")
    assert hasattr(SC, "TEAM_NAMES")
    assert hasattr(SC, "MaxEpisodeStep")
    assert hasattr(SC, "render")
    assert hasattr(SC, "EntityOriented")
    assert hasattr(SC, "obs_shape")
    assert hasattr(SC, "n_actions")
    assert hasattr(SC, "AvailActProvided")

    assert len(SC.AGENT_ID_EACH_TEAM) == len(SC.TEAM_NAMES), "Multiple defintions of N_TEAM"
    for t in SC.AGENT_ID_EACH_TEAM:
        for a in t:
            assert (isinstance(a, int) and a >= 0), "AGENT_ID_EACH_TEAM should be like [[0, 1], [2, 3]]"
            # 空数也不行, but we don't need too much check
    
    n_teams = len(ScenarioConfig.AGENT_ID_EACH_TEAM)
    if hasattr(SC, "interested_team"):
        assert SC.interested_team in range(0, n_teams)
        assert SC.interested_team == 0, 'if not so, errors may occur'
    
    TOLERANCE -= 1

def get_N_AGENT_EACH_TEAM(ScenarioConfig: ScenarioConfig):
    check_ScenarioConfig(ScenarioConfig)
    return [len(ids) for ids in ScenarioConfig.AGENT_ID_EACH_TEAM]

def get_N_TEAM(ScenarioConfig: ScenarioConfig):
    check_ScenarioConfig(ScenarioConfig)
    assert isinstance(ScenarioConfig.AGENT_ID_EACH_TEAM[0], (list, tuple))
    return len(ScenarioConfig.AGENT_ID_EACH_TEAM)
#########################################################################################################################################