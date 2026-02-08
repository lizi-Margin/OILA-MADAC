import numpy as np
from uhtk.UTIL.colorful import *
from uhtk.UTIL.tensor_ops import my_view, __hash__
from config import GlobalConfig as cfg



class ShellEnvWrapper(object):
    def __init__(self, n_agent, n_thread, mcv, RL_functional, 
                                          alg_config, ScenarioConfig):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.RL_functional = RL_functional
        self.raw_ob_shape = alg_config.rawob_shape
        self.action_spec = alg_config.action_spec or [ScenarioConfig.n_actions]
        self.action_branch_dim = len(self.action_spec)

        # whether to use avail_act to block forbiden actions
        self.AvailActProvided = False
        if hasattr(ScenarioConfig, 'AvailActProvided'):
            self.AvailActProvided = ScenarioConfig.AvailActProvided 


    def interact_with_env(self, State_Recall):
        assert State_Recall is not None
        act = np.zeros(
            shape=(self.n_thread, self.n_agent, self.action_branch_dim),
            dtype=int
        ) - 1  # 初始化全部为 -1

        ENV_PAUSE = State_Recall['ENV-PAUSE']
        obs = State_Recall['Latest-Obs']
        obs_feed = obs[~ENV_PAUSE]
        obs_feed_in = self.shape_attention_obs(obs_feed)

        assert obs.shape[0] == self.n_thread

        I_State_Recall = {
            'Latest-Obs':obs_feed_in, 
            'ENV-PAUSE':ENV_PAUSE, 
            'Latest-Info':State_Recall['Latest-Info'][~ENV_PAUSE],
        }

        if self.AvailActProvided:
            avail_act = np.array([info['Avail-Act'] for info in np.array(State_Recall['Latest-Info'][~ENV_PAUSE], dtype=object)])
            I_State_Recall.update({'Avail-Act':avail_act})
        else:
            raise NotImplementedError("Go modify the madac_sampler, remove avail_act from the req_dict")

        act_active, internal_recall = self.RL_functional.interact_with_env_genuine(I_State_Recall)

        act[~ENV_PAUSE] = act_active
        actions_list = act
        
        # register callback hook
        State_Recall['_hook_'] = internal_recall['_hook_']
        assert State_Recall['_hook_'] is not None
        return actions_list, State_Recall 

    def shape_attention_obs(self, obs_feed: np.ndarray):
        #  input might be (n_thread, n_agent, n_entity, basic_dim), or (n_thread, n_agent, n_entity*basic_dim)
        # both can be converted to (n_thread, n_agent, n_entity, basic_dim)
        obs_feed = obs_feed.astype(np.float32)
        obs_feed = my_view(obs_feed,[0, 0, -1,] + list(iter(self.raw_ob_shape)))

        # turning all zero padding to NaN, used for normalization
        obs_feed[(obs_feed==0).all(-1)] = np.nan

        return obs_feed
