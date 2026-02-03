# multi-team mapper
import importlib
import numpy as np
from copy import copy

class MTM:
    def __init__(self, mcv):
        from config import GlobalConfig as cfg
        self.n_thread = cfg.num_threads

        from MISSION.example import check_ScenarioConfig
        SC = cfg.ScenarioConfig
        check_ScenarioConfig(SC)
        self.n_team = len(SC.AGENT_ID_EACH_TEAM)
        self.team_member_list = SC.AGENT_ID_EACH_TEAM
        self.team_names = SC.TEAM_NAMES

        # spaces = envs.get_spaces()

        # init algorithm instances
        self.algo_foundations = []
        for t, team_spec in enumerate(self.team_names):
            module_name, cls_name = team_spec.split('->')
            import time
            t0 = time.time()
            module = importlib.import_module(module_name)
            print(f"imported {module_name} {cls_name} in {time.time()-t0:.4f}s")
            cls = getattr(module, cls_name)
            n_agent = len(self.team_member_list[t])
            
            t1 = time.time()
            print(f"start init {cls_name}")
            self.algo_foundations.append(cls(n_agent=n_agent, n_thread=self.n_thread, mcv=mcv, team=t))
            print(f"init {cls_name} in {time.time()-t1:.4f}s")

    def act(self, runner_info: dict):
        actions_list = []
        for t_name, t_members, algo_fdn, t_index in zip(self.team_names, self.team_member_list, self.algo_foundations, range(self.n_team)):
            t_intel = self._grab_t_intel(runner_info, t_members, t_name, t_index)
            act, t_intel = algo_fdn.interact_with_env(t_intel)

            act = np.swapaxes(act, 0, 1) 
            assert act.shape[0]==len(t_members), ('number of actions differs number of agents, Try to switch mt_act_order!')

            ########################BUG only consistently ascending t_memebers mapping is supported
            actions_list.extend(act)
            ########################BUG
            if t_intel is None: continue
            # process internal states loop back, featured with keys that startswith and endswith '_'
            for key in t_intel:
                if key.startswith('_') and key.endswith('_'): 
                    self._update_runner(runner_info, runner_info['ENV-PAUSE'], t_name, key, t_intel[key])

        # assemble into (n_thread, n_agent, ...)
        actions_list = np.swapaxes(np.array(actions_list, dtype=np.float32), 0, 1)

        # handle paused threads if align_episode
        ENV_PAUSE = runner_info['ENV-PAUSE']
        if ENV_PAUSE.any(): actions_list[ENV_PAUSE,:] = np.nan
        return actions_list, runner_info

    def _grab_t_intel(self, runner_info, t_members, t_name, t_index):
        obs    = runner_info['Latest-Obs'][:, t_members]
        reward = runner_info['Latest-Reward'][:, t_members]
        if isinstance(runner_info['Latest-Info'][0], dict):  # which is team info
            info   = runner_info['Latest-Info']
        elif isinstance(runner_info['Latest-Info'][0], (list, np.ndarray)):
            # wich is a list or np.darray of shape (n_threads, n_agents,) ?
            raise NotImplementedError("Agent wise info is not supported.")
        else: raise TypeError("Wrong info type, pls check the environment.step/reset.")
        t_intel = {
            "Team_Name": t_name,
            "Latest-Obs": obs,
            "Latest-Reward": reward,
            "Latest-Info": info,
            "Env-Suffered-Reset": runner_info['Env-Suffered-Reset'],
            "ENV-PAUSE": runner_info['ENV-PAUSE'],
            "Current-Step": runner_info['Current-Step']
        }

        for key in runner_info:
            if not (t_name in key): continue
            # otherwise t_name in key
            s_key = key.replace(t_name, '')
            assert s_key.startswith('_') and s_key.endswith('_')
            t_intel[s_key] = runner_info[key]
            if (s_key != '_hook_'): continue
            # otherwise deal with _hook_
            if t_intel['_hook_'] is not None:
                t_intel['_hook_'](copy(t_intel))
                runner_info[key] = None
                t_intel['_hook_'] = None
            # remove _hook_ key
            t_intel.pop('_hook_')
        return t_intel
    
    def _update_runner(self, runner_info:dict, ENV_PAUSE, t_name, key, content):
        u_key = t_name+key
        if (u_key in runner_info) and hasattr(content, '__len__') and \
                len(content)==self.n_thread and ENV_PAUSE.any():
            runner_info[u_key][~ENV_PAUSE] = content[~ENV_PAUSE]
            return
        runner_info[u_key] = content
        return
