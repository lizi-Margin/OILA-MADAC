import os, time, torch, traceback, shutil, platform
import numpy as np
from pathlib import Path
from uhtk.UTIL.colorful import *
from config import GlobalConfig
from uhtk.UTIL.tensor_ops import repeat_at
from ALGORITHM.common.rl_alg_base import RLAlgorithmBase
from conf_system import get_conf
class AlgorithmConfig:
    '''
        AlgorithmConfig: This config class will be 'injected' with new settings from json.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
        (please see UTIL.config_args to find out how this advanced trick works out.)
    '''
    # configuration, open to jsonc modification
    gamma = 0.99
    tau = 0.95
    train_traj_needed = 512
    use_normalization = True
    add_prob_loss = False
    load_checkpoint = False
    load_specific_checkpoint = ''
    device = 'cuda'

    # PPO part
    clip_param = 0.15
    ppo_epoch = 16
    n_pieces_batch_division = 1
    value_loss_coef = 0.1
    entropy_coef = 0.05
    max_grad_norm = 0.5
    lr = 1e-4

    # sometimes the episode length gets longer,
    # resulting in more samples and causing GPU OOM,
    # prevent this by fixing the number of samples to initial
    # by randomly sampling and droping
    prevent_batchsize_oom = False
    gamma_in_reward_forwarding = False
    gamma_in_reward_forwarding_value = 0.99

    net_hdim = 256

    entity_emb_dim = 32

    n_entity_placeholder = 'auto load, do not change'
    rawob_shape = 'auto load, do not change'
    n_agent = 'auto load, do not change'

    # fp16
    use_fp16 = False

    # imitation
    save_traj = False

    # action space
    action_spec = 'auto load, do not change'

    
    
def str_array_to_num(str_arr):
    out_arr = []
    buffer = {}
    for str in str_arr:
        if str not in buffer:
            buffer[str] = len(buffer)
        out_arr.append(buffer[str])
    return out_arr

def itemgetter(*items):
    # same with operator.itemgetter
    def g(obj): return tuple(obj[item] if item in obj else None for item in items)
    return g

class ReinforceAlgorithmFoundation(RLAlgorithmBase):
    def __init__(self, n_agent, n_thread, mcv=None, team=None):
        from .shell_env import ShellEnvWrapper
        from .net import Net
        super().__init__(n_agent, n_thread, mcv, team)
        AlgorithmConfig.n_agent = n_agent
        self.device = AlgorithmConfig.device

        obs_shape = self.ScenarioConfig.obs_shape
        if self.ScenarioConfig.EntityOriented: 
            assert len(obs_shape) >= 2
            AlgorithmConfig.n_entity_placeholder = obs_shape[0]
            AlgorithmConfig.rawob_shape = obs_shape[1:]
        else: 
            AlgorithmConfig.n_entity_placeholder = 1
            AlgorithmConfig.rawob_shape = obs_shape

        self.AvailActProvided = get_conf(
            self.ScenarioConfig, 'AvailActProvided', default=False)

        self._init_action_spec()

        # change obs format, e.g., converting dead agent obs into NaN
        self.shell_env = ShellEnvWrapper(n_agent, n_thread, mcv, self, AlgorithmConfig, GlobalConfig.ScenarioConfig)
        # initialize policy
        self.policy = Net(
            rawob_shape=AlgorithmConfig.rawob_shape,
            action_spec=AlgorithmConfig.action_spec
        )
        self.policy = self.policy.to(self.device)

        # initialize optimizer and trajectory (batch) manager
        from .ppo import PPO
        from .trajectory import BatchTrajManager
        self.trainer = PPO(self.policy, ppo_config=AlgorithmConfig, mcv=mcv)
        self.traj_manager = BatchTrajManager(
            n_env=n_thread, traj_limit=int(GlobalConfig.ScenarioConfig.MaxEpisodeStep),
            trainer_hook=self.trainer.train_on_traj)

        # load checkpoints if needed
        self.load_model(AlgorithmConfig)

        # Initialize GRU hidden states for each thread
        self.n_thread = n_thread
        self.policy_hidden_states = np.zeros((n_thread, n_agent, self.policy.policy_hidden_dim), dtype=np.float32)
        self.critic_hidden_states = np.zeros((n_thread, n_agent, self.policy.critic_hidden_dim), dtype=np.float32)

    def _init_action_spec(self):
        n_actions = self.ScenarioConfig.n_actions
        if isinstance(n_actions, (list, tuple)):
            action_spec = [int(x) for x in n_actions]
        elif hasattr(n_actions, 'shape') and len(getattr(n_actions, 'shape')) > 0:
            action_spec = [int(x) for x in list(n_actions)]
        else:
            action_spec = [int(n_actions)]
        assert all(x > 0 for x in action_spec), f"Invalid action spec: {action_spec}"
        AlgorithmConfig.action_spec = action_spec
        self.action_branch_dim = len(action_spec)


    def action_making(self, StateRecall):
        # make sure hook is cleared
        assert ('_hook_' not in StateRecall)

        threads_active_flag = ~StateRecall['ENV-PAUSE']
        obs = StateRecall['Latest-Obs']
        avail_act = StateRecall.get('Avail-Act', None)


        # make sure obs is right
        assert obs is not None, ('Make sure obs is ok')
        assert len(obs) == sum(threads_active_flag), ('check batch size')
        # make sure avail_act is correct
        if self.AvailActProvided: assert avail_act is not None

        # Get hidden states for active threads
        active_indices = np.where(threads_active_flag)[0]
        done_indices = np.where(~threads_active_flag)[0]
        policy_hx = self.policy_hidden_states[active_indices]
        critic_hx = self.critic_hidden_states[active_indices]
        # Reset hidden states for done threads
        self.policy_hidden_states[done_indices] = np.zeros((AlgorithmConfig.n_agent, self.policy.policy_hidden_dim), dtype=np.float32)
        self.critic_hidden_states[done_indices] = np.zeros((AlgorithmConfig.n_agent, self.policy.critic_hidden_dim), dtype=np.float32)

        # make decision
        with torch.no_grad():
            action, value, action_log_prob, new_policy_hx, new_critic_hx = self.policy.act(
                obs=obs,
                test_mode=False,
                avail_act=avail_act,
                policy_hx=policy_hx,
                critic_hx=critic_hx
            )

        # Update hidden states for active threads
        self.policy_hidden_states[active_indices] = new_policy_hx
        self.critic_hidden_states[active_indices] = new_critic_hx
        # self.policy_hidden_states[~threads_active_flag] = np.zeros((AlgorithmConfig.n_agent, self.policy.policy_hidden_dim), dtype=np.float32)
        # self.critic_hidden_states[~threads_active_flag] = np.zeros((AlgorithmConfig.n_agent, self.policy.critic_hidden_dim), dtype=np.float32)

        # commit obs to buffer, vars named like _x_ are aligned, others are not!
        traj_framefrag = {
            "_SKIP_":        ~threads_active_flag,
            "value":         value,
            "actionLogProb": action_log_prob,
            "obs":           obs,
            "action":        action,
            "policy_hx":     policy_hx,  # Store hidden states at each step (for sequence training, first step should be zeros)
            "critic_hx":     critic_hx,
        }
        # from uhtk.print_pack import print_dict
        # print_dict(traj_framefrag)
        if avail_act is not None: traj_framefrag.update({'avail_act':  avail_act})
        # deal with rollout later when the reward is ready, leave a hook as a callback here
        StateRecall['_hook_'] = self.commit_traj_frag(traj_framefrag, req_hook = True)
        return action.copy(), StateRecall


    def interact_with_env(self, StateRecall):
        '''
            Interfacing with marl, standard method that you must implement
            (redirect to shell_env to help with history rolling)
        '''
        return self.shell_env.interact_with_env(StateRecall)


    def interact_with_env_genuine(self, StateRecall):
        '''
            When shell_env finish the preparation, interact_with_env_genuine is called
            (Determine whether or not to do a training routinue)
        '''
        self.train()  # when needed, train!
        return self.action_making(StateRecall)

    def train(self):
        '''
            Get event from hmp task runner, save model now!
        '''
        if self.traj_manager.can_exec_training():
            self.traj_manager.train_and_clear_traj_pool()
            self.save_model(self.trainer.ppo_update_cnt)
















    def save_model(self, update_cnt, info=None):
        '''
            save model now!
            save if triggered when:
            1. Update_cnt = 50, 100, ...
            2. Given info, indicating a hmp command
            3. A flag file is detected, indicating a save command from human
        '''
        if not os.path.exists('%s/history_cpt/' % GlobalConfig.logdir): 
            os.makedirs('%s/history_cpt/' % GlobalConfig.logdir)

        # dir 1
        pt_path = str(Path('%s/model.pt' % GlobalConfig.logdir))
        print绿('saving model to %s' % pt_path)
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.trainer.optimizer.state_dict(),
        }, pt_path)

        # dir 2
        if update_cnt % 10 == 0: 
            info = str(update_cnt) if info is None else ''.join([str(update_cnt), '_', info])
            pt_path2 = str(Path('%s/history_cpt/model_%s.pt' % (GlobalConfig.logdir, info)))
            if platform.system() == 'Windows':
                pt_path2 = pt_path2.replace(":", '-').replace("'", '')
            shutil.copyfile(pt_path, pt_path2)

        print绿('save_model fin')



    def load_model(self, AlgorithmConfig):
        '''
            load model now
        '''

        if AlgorithmConfig.load_checkpoint:
            manual_dir = AlgorithmConfig.load_specific_checkpoint
            ckpt_dir = '%s/model.pt' % GlobalConfig.logdir if manual_dir == '' else '%s/%s' % (GlobalConfig.logdir, manual_dir)
            device = 'cpu' if 'cpu' in self.device else self.device
            strict = True
            if not os.path.exists(ckpt_dir):
                print_bold_red('Error: [load_model] checkpoint not found:', ckpt_dir)
                return
            
            cpt = torch.load(ckpt_dir, map_location=device)
            self.policy.load_state_dict(cpt['policy'], strict=strict)
            # https://github.com/pytorch/pytorch/issues/3852
            try:
                self.trainer.optimizer.load_state_dict(cpt['optimizer'])
            except KeyError as e:
                print黄(f'optimizer not loaded {repr(e)}')

            print黄('loaded checkpoint:', ckpt_dir)

