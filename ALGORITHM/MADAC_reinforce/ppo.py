import torch, math, traceback
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from uhtk.UTIL.colorful import *
from uhtk.UTIL.tensor_ops import _2tensor, __hash__, __hashn__
from config import GlobalConfig as cfg
from .ppo_sampler import TrajPoolSampler
from uhtk.mcv_log_manager import LogManager

class PPO():
    def __init__(self, policy_and_critic, ppo_config, mcv=None):
        self.policy_and_critic = policy_and_critic
        self.clip_param = ppo_config.clip_param
        self.ppo_epoch = ppo_config.ppo_epoch
        self.n_pieces_batch_division = ppo_config.n_pieces_batch_division
        self.value_loss_coef = ppo_config.value_loss_coef
        self.entropy_coef = ppo_config.entropy_coef
        self.max_grad_norm = ppo_config.max_grad_norm
        self.add_prob_loss = ppo_config.add_prob_loss
        self.prevent_batchsize_oom = ppo_config.prevent_batchsize_oom
        # self.freeze_body = ppo_config.freeze_body
        self.lr = ppo_config.lr
        self.all_parameter = list(policy_and_critic.named_parameters())

        # if not self.freeze_body:
        self.parameter = [p for p_name, p in self.all_parameter]
        self.optimizer = optim.Adam(self.parameter, lr=self.lr)

        self.g_update_delayer = 0
        self.g_initial_value_loss = 0
        
        # 轮流训练式
        self.log_manager = LogManager(mcv=mcv, who='ppo.py')
        self.ppo_update_cnt = 0
        self.batch_size_reminder = True

        if ppo_config.use_fp16:
            tv = str(torch.__version__)
            if '+' in tv:
                tv = tv.split('+')[0]
            tv = tv.split('.')
            tv = tv[:3]
            print('torch.__version__', tv)
            if tv >= ['1', '6', '0']:
                self.scaler = torch.GradScaler('cuda', init_scale = 2.0**16)
            else:
                self.scaler = torch.cuda.amp.GradScaler(init_scale=2.0**16)
        else:
            self.scaler = None

        assert self.n_pieces_batch_division == 1


    def train_on_traj(self, traj_pool, task):
        while True:
            try:
                self.train_on_traj_(traj_pool, task) 
                break # 运行到这说明显存充足
            except RuntimeError as err:
                print(traceback.format_exc())
                if self.prevent_batchsize_oom:
                    # in some cases, reversing MaxSampleNum a single time is not enough
                    if TrajPoolSampler.MaxSampleNum[-1] < 0: TrajPoolSampler.MaxSampleNum.pop(-1)
                    assert TrajPoolSampler.MaxSampleNum[-1] > 0
                    TrajPoolSampler.MaxSampleNum[-1] = -1
                    print亮红('Insufficient gpu memory, using previous sample size !')
                else:
                    assert False
            torch.cuda.empty_cache()

    def train_on_traj_(self, traj_pool, task):

        ppo_valid_percent_list = []
        from .foundation import AlgorithmConfig
        sampler = TrajPoolSampler(n_div=1, traj_pool=traj_pool, flag=task, prevent_batchsize_oom=self.prevent_batchsize_oom, mcv=self.log_manager.mcv)
        # before_training_hash = [__hashn__(t.parameters()) for t in (self.policy_and_critic._nets_flat_placeholder_)]
        for e in range(self.ppo_epoch):
            sample_iter = sampler.reset_and_get_iter()
            self.optimizer.zero_grad()
            # ! get traj fragment
            sample = next(sample_iter)
            # ! build graph, then update network
            if self.scaler:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss_final, others = self.establish_pytorch_graph(task, sample, e)
            else:
                loss_final, others = self.establish_pytorch_graph(task, sample, e)
            loss_final = loss_final*0.5
            # if e==0: print('[PPO.py] Memory Allocated %.2f GB'%(torch.cuda.memory_allocated()/1073741824))
            print('[PPO.py] Memory Allocated %.2f GB'%(torch.cuda.memory_allocated()/1073741824))

            if self.scaler:
                self.scaler.scale(loss_final).backward()
            else:
                loss_final.backward()
            # log
            ppo_valid_percent_list.append(others.pop('PPO valid percent').item())
            self.log_manager.log_trivial(dictionary=others); others = None

            # Check for NaN in gradients before clipping
            has_nan_grad = False
            for name, param in self.policy_and_critic.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print亮红(f'[PPO.py] NaN gradient detected in {name}!')
                    has_nan_grad = True

            if has_nan_grad:
                print亮红('[PPO.py] NaN gradients detected! Skipping this update step.')
                self.optimizer.zero_grad()
                continue

            nn.utils.clip_grad_norm_(self.parameter, self.max_grad_norm)
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Check for NaN in weights after update
            for name, param in self.policy_and_critic.named_parameters():
                if torch.isnan(param).any():
                    print亮红(f'[PPO.py] NaN weight detected in {name} after update! Rolling back.')
                    # This is catastrophic - the model is corrupted
                    raise RuntimeError(f'NaN weights detected in {name}. Training cannot continue. Please reload from checkpoint.')

            
            if ppo_valid_percent_list[-1] < 0.45: 
                print亮黄('policy change too much, epoch terminate early'); break 
        pass # finish all epoch update

        print亮黄(np.array(ppo_valid_percent_list))
        self.log_manager.log_trivial_finalize()

        self.ppo_update_cnt += 1
                
        
        return self.ppo_update_cnt

    def freeze_body(self):
        assert False, "function forbidden"
        self.freeze_body = True
        self.parameter_pv = [p_name for p_name, p in self.all_parameter if not any(p_name.startswith(kw)  for kw in ('obs_encoder', 'attention_layer'))]
        self.parameter = [p for p_name, p in self.all_parameter if not any(p_name.startswith(kw)  for kw in ('obs_encoder', 'attention_layer'))]
        self.optimizer = optim.Adam(self.parameter, lr=self.lr)
        print('change train object')

    def establish_pytorch_graph(self, flag, sample, n):
        obs = _2tensor(sample['obs'])
        advantage = _2tensor(sample['advantage'])
        action = _2tensor(sample['action'])
        oldPi_actionLogProb = _2tensor(sample['actionLogProb'])
        real_value = _2tensor(sample['return'])
        avail_act = _2tensor(sample['avail_act']) if 'avail_act' in sample else None

        # Get hidden states from trajectory if available
        policy_hx = _2tensor(sample['policy_hx']) if 'policy_hx' in sample else None
        critic_hx = _2tensor(sample['critic_hx']) if 'critic_hx' in sample else None

        # Check if we're using sequence-based sampling (has traj_mask)
        traj_mask: torch.Tensor = _2tensor(sample['traj_mask']) if 'traj_mask' in sample else None
        is_sequence_mode = traj_mask is not None

        if is_sequence_mode:
            # Sequence mode: obs shape = (n_traj, seq_len, n_agent, ...)
            batch_agent_size = (traj_mask.sum() * advantage.shape[2]).int().item()  # valid_steps * n_agent
        else:
            raise ValueError('Flat mode already not supported!')
            # Flat mode: obs shape = (batch, n_agent, ...)
            batch_agent_size = advantage.shape[0] * advantage.shape[1]

        assert flag == 'train'
        newPi_value, newPi_actionLogProb, entropy, probs, others = \
            self.policy_and_critic.evaluate_actions(
                obs=obs,
                eval_actions=action,
                avail_act=avail_act,
                traj_mask=traj_mask)
        #######################################################################
        mask_expanded = traj_mask.unsqueeze(-1).unsqueeze(-1).expand_as(advantage).float()
        assert advantage.shape[2] == obs.shape[2], f"advantage shape {advantage.shape}, obs shape {obs.shape}"
        valid_steps = mask_expanded.sum().clamp_min(1.0)
        entropy_mask = traj_mask.unsqueeze(-1).expand_as(entropy).float()
        entropy_valid_steps = entropy_mask.sum().clamp_min(1.0)
        entropy_loss = (entropy * entropy_mask).sum() / entropy_valid_steps
        #######################################################################
        # entropy_loss = entropy.mean()
        #######################################################################


        n_actions = probs.shape[-1]
        if self.add_prob_loss: assert n_actions <= 15  #
        penalty_prob_line = (1/n_actions)*0.12
        probs_loss = (penalty_prob_line - torch.clamp(probs, min=0, max=penalty_prob_line)).mean()
        if not self.add_prob_loss:
            probs_loss = torch.zeros_like(probs_loss)

        # dual clip ppo core
        # RuntimeError: The size of tensor a (423) must match the size of tensor b (2) at non-singleton dimension 1
        # see _get_act_log_probs
        E = newPi_actionLogProb - oldPi_actionLogProb
        ratio = torch.exp(E)
        ratio_clipped = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        adv_masked = advantage * mask_expanded
        policy_loss_unclipped = ratio * adv_masked
        policy_loss_clipped = ratio_clipped * adv_masked
        policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped).sum() / valid_steps

        value_loss = 0.5 * ((real_value - newPi_value) ** 2 * mask_expanded).sum() / valid_steps

        AT_net_loss = policy_loss - entropy_loss*self.entropy_coef # + probs_loss*20
        CT_net_loss = value_loss*self.value_loss_coef


        loss_final = AT_net_loss + CT_net_loss

        #######################################################################
        clipped_mask = (torch.abs(ratio - ratio_clipped) > 1e-6).float() * mask_expanded
        ppo_valid_percent = 1.0 - clipped_mask.sum() / valid_steps
        #######################################################################
        # ppo_valid_percent = ((E_clip == E).int().sum()/batch_agent_size)
        #######################################################################

        #######################################################################
        nz_mask = (real_value != 0) & (mask_expanded > 0)
        #######################################################################
        # nz_mask = real_value != 0
        #######################################################################

        if nz_mask.any():
            value_loss_abs = (real_value[nz_mask] - newPi_value[nz_mask]).abs().mean()
        else:
            value_loss_abs = torch.tensor(0.0, device=real_value.device)

        others = {
            'Value loss Abs':           value_loss_abs,
            'PPO valid percent':        ppo_valid_percent,
            'AT_net_loss':              AT_net_loss,
            'Policy loss':              policy_loss,
            'Entropy loss':             entropy_loss,
            'CT_net_loss':              CT_net_loss,
            'Value loss':               value_loss,
        }

        return loss_final, others


