import numpy as np
from uhtk.UTIL.colorful import *

class GlobalConfig(object):
    logdir = './RESULT/default/'
    seed = np.random.randint(0, 100000)                 # seed for numpy and pytorch
    machine_info = 'auto load, do not change this var!'
    cfg_ready = False  # DO NOT change! automatically set

    runner = 'rl_runner'

    # for RL & RL runner
    env_name = 'default'                                # which environment, see ./MISSION/env_router.py
    num_threads = 32                                    # run N parallel envs, a 'env' is refered to as a 'thread'
    fold = 1                                            # A 'linux process' can handle multiple envs ('thread'), run N parallel envs, on (N//fold) processes
    max_n_episode = int(2e5)                            # max number of episodes
    report_reward_interval = 32                         # reporting interval
    ScenarioConfig = 'This ScenarioConfig var will be automatically linked to task configuration later in ./MISSION/env_router.py'

    
