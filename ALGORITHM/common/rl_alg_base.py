import time
from uhtk.UTIL.tensor_ops import __hash__, repeat_at
from uhtk.UTIL.colorful import *
from ALGORITHM.common.alg_base import AlgorithmBase

# model IO
class RLAlgorithmBase(AlgorithmBase):
    def __init__(self, n_agent, n_thread, mcv=None, team=None):
        super().__init__(n_agent, n_thread, mcv, team)
        
        # data integraty check
        self._unfi_frag_ = None

        # Skip currupt data integraty check after this patience is exhausted
        self.patience = 1000



    def interact_with_env(self, team_intel):
        raise NotImplementedError

    def save_model(self, update_cnt, info=None):
        raise NotImplementedError

    def process_framedata(self, framedata, new_intel):
        ''' 
            hook is called when reward and next moment observation is ready,
            now feed them into trajectory manager.
            Rollout Processor | 准备提交Rollout, 以下划线开头和结尾的键值需要对齐(self.n_thread, ...)
            note that keys starting with _ must have shape (self.n_thread, ...), details see fn:mask_paused_env()
        '''
        assert '_SKIP_' in framedata
        framedata['_DONE_'] = new_intel.pop('Env-Suffered-Reset')
        framedata['reward'] = new_intel.pop('Latest-Reward')
        framedata = self.mask_paused_env(framedata)
        self.traj_manager.feed_traj_framedata(framedata)



    def mask_paused_env(self, frag):
        running = ~frag['_SKIP_']
        if running.all():
            return frag
        for key in frag:
            if not key.startswith('_') and hasattr(frag[key], '__len__') and len(frag[key]) == self.n_thread:
                frag[key] = frag[key][running]
        return frag




    ''' 
        function to be called when reward is received
    '''
    def commit_traj_frag(self, unfi_frag, req_hook=True):
        assert self._unfi_frag_ is None
        self._unfi_frag_ = unfi_frag
        self._check_data_hash()  # check data integraty
        if req_hook:
            # leave a hook
            return self.traj_waiting_hook
        else:
            return None


    def traj_waiting_hook(self, new_intel):
        ''' 
            This function will be called from <multi_team.py::deal_with_hook()>
            hook is called when reward and next moment observation is ready
        '''
        # do data curruption check at beginning, this is important!
        self._check_data_curruption()
        # call upper level function to deal with frame data
        self.process_framedata(self._unfi_frag_ , new_intel)
        # delete data reference
        self._unfi_frag_ = None


    # protect data from overwriting
    def _check_data_hash(self):
        if self.patience > 0:
            self.patience -= 1
            self.hash_db = {}
            # for debugging, to detect write protection error
            for key in self._unfi_frag_:
                item = self._unfi_frag_[key]
                if isinstance(item, dict):
                    self.hash_db[key] = {}
                    for subkey in item:
                        subitem = item[subkey]
                        self.hash_db[key][subkey] = __hash__(subitem)
                else:
                    self.hash_db[key] = __hash__(item)

    # protect data from overwriting
    def _check_data_curruption(self):
        if self.patience > 0:
            self.patience -= 1
            assert self._unfi_frag_ is not None
            assert self.hash_db is not None
            for key in self._unfi_frag_:
                item = self._unfi_frag_[key]
                if isinstance(item, dict):
                    for subkey in item:
                        subitem = item[subkey]
                        assert self.hash_db[key][subkey] == __hash__(subitem), ('Currupted data!')
                else:
                    assert self.hash_db[key] == __hash__(item), ('Currupted data!')

