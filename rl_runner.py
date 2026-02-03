"""
    Author: Fu Qingxu,CASIA
    Description: HMP task runner, coordinates environments and algorithms
"""

import time, os
import numpy as np
from uhtk.UTIL.colorful import *
from uhtk.mcv_log_manager import LogManager, get_a_logger
from config import GlobalConfig as cfg
from MISSION.env_router import make_parallel_envs
from MISSION.example import get_N_AGENT_EACH_TEAM, get_N_TEAM
from ALGORITHM.mt_mapper import MTM
class Runner(object):
    def __init__(self, envs):
        self.envs = envs
        print("init logger")
        self.mcv = get_a_logger(cfg.logdir)                # multiagent silent logging bridge active
        print("init platform controller")
        self.platform_controller = MTM(self.mcv)  # block infomation access between teams
        print("init info_runner")
        self.info_runner = {}                                       # dict of realtime obs, reward, reward, info et.al.
        self.n_agent  = sum(get_N_AGENT_EACH_TEAM(cfg.ScenarioConfig))
        self.n_team  = len(get_N_AGENT_EACH_TEAM(cfg.ScenarioConfig))
        
        self.n_thread = cfg.num_threads
        self.total_step_cnt = 0
        self.total_episode_cnt = 0
        self.max_n_episode = cfg.max_n_episode

        self.report_interval = cfg.report_reward_interval
        self.top_rewards = None
        self.sync_terminal_width()
        
    def sync_terminal_width(self):
        try:
            self.terminal_width =  os.get_terminal_size().columns
        except (OSError, AttributeError):
            self.terminal_width = 80  # fallback to a default value

    # -------------------------------------------------------------------------
    # ------------------------------ Major Loop -------------------------------
    # -------------------------------------------------------------------------
    def run(self):
        self.init_runner()
        tic = time.time()
        while (self.total_episode_cnt < self.max_n_episode):
            actions_list, self.info_runner = self.platform_controller.act(self.info_runner)
            obs, reward, done, info = self.envs.step(actions_list)
            self.info_runner = self.update_runner(done, obs, reward, info)
            toc=time.time(); dt = toc-tic; tic = toc
            print('\r [task runner]: FPS %d, episode steping %s       '%(
                self.get_fps(dt), self.heartbeat(style=0)), end='', flush=True
            )

    def init_runner(self):
        # self.info_runner['Test-Flag'] = False  # not testing mode for rl methods

        # to monitor
        self.info_runner['Recent-Reward-Sum'] = []
        self.info_runner['Recent-Team-Ranking'] = []
        for i in range(self.n_team + 5):  # see how we plot top-rank ratio
            self.info_runner[f't{i}_win_cnt_avg'] = []
        self.info_runner[f'draw_cnt_avg'] = []

        # obs, info, reward
        obs_info = self.envs.reset() # assumes only the first time reset is manual
        (self.info_runner['Latest-Obs'], self.info_runner['Latest-Info']) = \
            obs_info if isinstance(obs_info, tuple) else (obs_info, None)
        self.info_runner['Latest-Reward']      = np.zeros(shape=(self.n_thread, self.n_agent))
        self.info_runner['Latest-Reward-Sum']  = np.zeros(shape=(self.n_thread, self.n_agent))

        # to control
        self.info_runner['Env-Suffered-Reset'] = np.array([True for _ in range(self.n_thread)])
        self.info_runner['ENV-PAUSE']          = np.array([False for _ in range(self.n_thread)])
        self.info_runner['Current-Step']   = np.array([0 for _ in range(self.n_thread)])
        self.info_runner['Thread-Episode-Cnt'] = np.array([0 for _ in range(self.n_thread)])
        

    def update_runner(self, done, obs, reward, info):
        P = self.info_runner['ENV-PAUSE']
        R = ~P

        assert info is not None
        if self.info_runner['Latest-Info'] is None: self.info_runner['Latest-Info'] = info

        self.info_runner['Latest-Obs'][R] = obs[R]
        self.info_runner['Latest-Info'][R] = info[R]
        self.info_runner['Latest-Reward'][R] = reward[R]    # note, reward shape: (thread, n-team\n-agent)
        self.info_runner['Latest-Reward-Sum'][R] += reward[R]
        self.info_runner['Current-Step'][R] += 1

        for i in range(self.n_thread):
            self.info_runner['Env-Suffered-Reset'][i] = done[i].all()
            # if the environment has not been reset, do nothing
            if P[i] or (not self.info_runner['Env-Suffered-Reset'][i]): continue
            # otherwise, the environment just been reset
            self.total_step_cnt += self.info_runner['Current-Step'][i]
            self.total_episode_cnt += 1
            self.info_runner['Thread-Episode-Cnt'][i] += 1

            self.info_runner['Recent-Reward-Sum'].append(self.info_runner['Latest-Reward-Sum'][i].copy())
            self.info_runner['Latest-Reward-Sum'][i] = 0
            self.info_runner['Current-Step'][i] = 0
            self.info_runner['ENV-PAUSE'][i] = True

            term_info = self.info_runner['Latest-Info'][i]
            if 'team_ranking' in term_info: 
                self.info_runner['Recent-Team-Ranking'].append(term_info['team_ranking'].copy())

            if self.total_episode_cnt % self.report_interval == 0: 
                self._report(self.info_runner)  # monitor rewards for some specific agents
                self.info_runner['Recent-Reward-Sum'] = []
                self.info_runner['Recent-Team-Ranking'] = []

        # all threads haulted, finished and Aligned, then restart all thread
        if self.info_runner['ENV-PAUSE'].all():  self.info_runner['ENV-PAUSE'][:] = False
        return self.info_runner





    def _report(self, info_runner):
        # (1). record mean reward
        self.mcv.rec(self.total_episode_cnt, 'time')
        prefix = 'train '
        recent_rewards = np.stack(info_runner['Recent-Reward-Sum'])
        mean_reward_each_team = []
        for interested_team in range(self.n_team):
            tean_agent_uid = cfg.ScenarioConfig.AGENT_ID_EACH_TEAM[interested_team]
            mean_reward_each_team.append(recent_rewards[:, tean_agent_uid].mean().copy())

        for team in range(self.n_team):
            self.mcv.rec(mean_reward_each_team[team], f'{prefix}reward of=team-{team}')

        # (2).reflesh historical top reward
        if self.top_rewards is None: self.top_rewards = mean_reward_each_team
        top_rewards_list_pointer = self.top_rewards

        for team in range(self.n_team):
            if mean_reward_each_team[team] > top_rewards_list_pointer[team]:
                top_rewards_list_pointer[team] = mean_reward_each_team[team]
            self.mcv.rec(top_rewards_list_pointer[team], f'{prefix}top reward of=team-{team}')

        # (3).record winning rate (single-team) or record winning rate (multi-team)
        # for team in range(self.n_team):
        teams_ranking = info_runner['Recent-Team-Ranking']
        n_team = len(teams_ranking[0])  # sometimes, the baseline team is managed by the env, so the real team num may larger than self.n_team
        # if len(teams_ranking) != self.n_team:
        #     print(f'Warning: n_team is {self.n_team}, but {n_team} teams are ranked')
        win_rate_each_team = [0]*n_team
        draw_rate = 0
        if len(teams_ranking)>0:
            for team in range(n_team):
                rank_itr_team = np.array(teams_ranking)[:, team]
                win_rate = (rank_itr_team==0).mean()  # 0 means rank first
                win_rate_each_team[team] = win_rate
                self.mcv.rec(win_rate, f'{prefix}top-rank ratio of=team-{team}')
            rank_itr = np.array(teams_ranking)
            draw_rate = (rank_itr==-1).mean()  # -1 means draw
            self.mcv.rec(draw_rate, f'{prefix}draw ratio')
        # else:
        #     team = 0; assert self.n_team == 1, "There is only one team"
        #     win_rate_each_team[team] = np.array(info_runner['Recent-Win']).mean()
        #     win_rate = np.array(info_runner['Recent-Win']).mean()
        #     self.mcv.rec(win_rate, f'{prefix}win rate of=team-{team}')

        for i in range(n_team):
            self.info_runner[f't{i}_win_cnt_avg'].append(win_rate_each_team[i])
            ti = np.array(self.info_runner[f't{i}_win_cnt_avg']).mean()
            self.mcv.rec(ti, f'{prefix}acc win ratio of=team-{i}')
        self.info_runner['draw_cnt_avg'].append(draw_rate)
        avg_draw_rate = np.array(self.info_runner['draw_cnt_avg']).mean()
        self.mcv.rec(avg_draw_rate, f'{prefix}acc draw ratio')
            

        # plot the figure
        self.mcv.rec_show()
        print_info = [f'\r[task runner]: Finished episode {self.total_episode_cnt}, frame {self.total_step_cnt}.']
        for team in range(self.n_team): 
            print_info.append(' | team-%d: win rate: %.3f, recent reward %.3f'%(team, win_rate_each_team[team], mean_reward_each_team[team]))
        print_info.append(' | draw rate: %.3f'%(draw_rate))
        print靛(''.join(print_info))
            
        return win_rate_each_team, mean_reward_each_team

    def heartbeat(self, style=0, beat=None):
        # default ⠁⠈⠐⠠⢀⡀⠄⠂
        if style==0: sym = ['⠁','⠈','⠐','⠠','⢀','⡀','⠄','⠂',]
        elif style==1: sym = ['◐ ','◓ ','◑ ','◒ ']
        elif style==2: sym = ['▁','▂','▃','▄','▅','▆','▇','█']
        elif style==3: sym = ['💐','🌷','🌸','🌹','🌺','🌻','🌼',]
        elif style==4: sym = ['😅','🤣','🙃','🫨','😶','😶','😶','😶']  # better terminal like kitty is recommended
        elif style==5: sym = ['😶','🙂','☺️','🤗','☺️','🙂']  # better terminal like kitty is recommended
        elif style==6: sym = ['😶','😶','🙂','🙂','☺️','☺️','🤗','🤗','☺️','☺️','🙂','🙂']  # better terminal like kitty is recommended
        elif style==6: sym = [
            '🕛','🕧','🕐','🕜','🕑','🕝','🕒','🕞',
            '🕓','🕟','🕔','🕠','🕕','🕡','🕖','🕢',
            '🕗','🕣','🕘','🕤','🕙','🕥','🕚','🕦']
        self.sync_terminal_width()
        if beat is None: beat = self.info_runner['Current-Step']
        beat = beat % len(sym)
        beat = beat[:int(max(self.terminal_width - 50, 1))]   # [task runner]: FPS 1499, episode steping  ........
        beat.astype(int)
        beat = [sym[t] for t in beat]
        return ''.join(beat)

    
    def get_fps(self, dt):
        dt = max(dt, 1e-6)
        n_thread_active = np.logical_not(self.info_runner['ENV-PAUSE']).sum()
        new_fps = int(n_thread_active/dt)
        if not hasattr(self, 'fps_smooth'):
            self.fps_smooth = new_fps
        else:
            self.fps_smooth = self.fps_smooth*0.98 + new_fps*0.02
        return int(self.fps_smooth)