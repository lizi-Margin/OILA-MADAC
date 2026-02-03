import os, threading, atexit
import time, numpy as np, cv2, gymnasium.spaces as spaces, copy, torch, random, importlib
import multiprocessing as mp
from uhtk.imitation.utils import safe_load_traj_pool, safe_dump_traj_pool,get_container_from_traj_pool
from uhtk.print_pack import *
from uhtk.mcv_log_manager import get_a_logger
from config import GlobalConfig as cfg
from uhtk.UTIL.kill_process import kill_process_and_its_children

from typing import Union, List, TYPE_CHECKING
from data_loader import data_loader_process

class ILRunnerConfig:
    alg_name = 'ALGORITHM.imitation_bc.foundation->BehaviorCloningFoundation'
    traj_dir: Union[str, List[str]] = 'traj-Grabber-tick=0.1-limit=200-pure'
    dataset_dir = 'G:HMP_IL'
    N_LOAD = 2000
    traj_per_LOAD = 2
    traj_reuse = 1
    queue_size = 2

class Runner(object):
    def __init__(self):
        alg_name = ILRunnerConfig.alg_name
        traj_dir = ILRunnerConfig.traj_dir
        self.N_LOAD = ILRunnerConfig.N_LOAD
        self.traj_per_LOAD = ILRunnerConfig.traj_per_LOAD
        self.traj_reuse = ILRunnerConfig.traj_reuse
        self.queue_size = ILRunnerConfig.queue_size
        self.dataset_dir = ILRunnerConfig.dataset_dir

        self.mcv = get_a_logger(cfg.logdir) 
        
        
        # init algorithm instances
        module_name, cls_name = alg_name.split('->')
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        self.algo_foundation = cls(mcv=self.mcv)

        self.traj_dir = traj_dir

    
    def run(self):
        if self.traj_dir == '*':
            self.traj_dir = os.listdir(self.dataset_dir)
        self.train_on(self.traj_dir, self.dataset_dir)
    
    ############################################
    # train_on([
    #     'traj-Grabber-tick=0.1-limit=200-pure',
    # ])
    def train_on(self, traj_dir, dataset_dir):
        if isinstance(traj_dir, str): traj_dir = [traj_dir]
        # torch.rand(1)
        queue = mp.Queue(maxsize=0)
        loader = mp.Process(target=data_loader_process, args=(traj_dir, dataset_dir, self.traj_per_LOAD, queue, self.queue_size), daemon=False)
        atexit.register(loader.terminate)
        # loader = threading.Thread(target=data_loader_process, args=(traj_dir, dataset_dir, self.traj_per_LOAD, queue, self.queue_size), daemon=True)
        loader.start()

        try:
            for i in range(self.N_LOAD):
                decoration = "_" * 20
                # datas, dir_name = queue.get()  # wait
                # print(decoration + f" train N_LOAD={i} starts, traj_dir={dir_name} " + decoration)

                # for j in range(self.traj_per_LOAD * self.traj_reuse):
                    # data = copy.copy(datas[j % self.traj_per_LOAD])
                    # print_dict(data)
                    # try:
                    #     self.algo_foundation.train_on_data(data)
                    # # except torch.OutOfMemoryError:
                    # except RuntimeError as e:
                    #     if "out of memory" not in str(e).lower():
                    #         raise e
                    #     continue

                print_yellow("start wait for data")
                data, dir_name = queue.get()  # wait
                print_yellow("get data done")
                print(decoration + f" train N_LOAD={i} starts, len(traj_dir)={len(dir_name)} " + decoration)
                for _ in range(self.traj_reuse):
                    data = copy.copy(data)
                    print_dict(data)
                    try:
                        self.algo_foundation.train_on_data(data)
                    # except torch.OutOfMemoryError:
                    except RuntimeError as e:
                        if "out of memory" not in str(e).lower():
                            raise e
                        continue
                
                del data
        finally:
            if loader.is_alive():
                kill_process_and_its_children(loader)
            loader.join(timeout=5)


