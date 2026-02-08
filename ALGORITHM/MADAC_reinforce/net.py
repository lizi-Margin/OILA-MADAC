import torch, math, copy
import numpy as np
import torch.nn as nn
from typing import Optional, Sequence, List
from uhtk.print_pack import *
from uhtk.siri.utils.iterable_tools import iterable_prod
from uhtk.encoder import ImgObsProcess, VectorObsProcess
from torch.distributions.categorical import Categorical
from uhtk.UTIL.tensor_ops import Args2tensor_Return2numpy, Args2tensor, __hashn__, my_view
from uhtk.UTIL.tensor_ops import pt_inf
from .foundation import AlgorithmConfig
from ALGORITHM.common.attention import SimpleAttention
from ALGORITHM.common.net_manifest import weights_init



"""
    network initialize
"""
class Net(nn.Module):
    def __init__(self, rawob_shape: tuple, action_spec: Optional[Sequence[int]], **kwargs):
        super().__init__()
        self.update_cnt = nn.Parameter(
            torch.zeros(1, requires_grad=False, dtype=torch.long), requires_grad=False)
        self.use_normalization = AlgorithmConfig.use_normalization
        self.action_spec: List[int] = list(action_spec)
        head_dim = int(sum(self.action_spec))

        h_dim = AlgorithmConfig.net_hdim
        if isinstance(rawob_shape, int): rawob_shape = (rawob_shape,)
        self.img_obs = False if len(rawob_shape) == 1 else True
        n_entity = AlgorithmConfig.n_entity_placeholder
        entity_emb_dim = AlgorithmConfig.entity_emb_dim
        # # # # # # # # # #  actor-critic share # # # # # # # # # # # #

        if entity_emb_dim == 0 or n_entity == 1 :
            self.entity_embedding = None
        else:
            self.entity_embedding = nn.Embedding(
                num_embeddings=n_entity,
                embedding_dim=entity_emb_dim
            )

        if self.img_obs:
            self.obs_encoder = ImgObsProcess(
                imgshape_chw=rawob_shape,
                # impala_chans=(8, 16, 16),
                impala_chans=(24, 48, 48),
                # cnn_outsize=h_dim//2,
                cnn_outsize=512,
                output_size=h_dim,
                pre_norm=self.use_normalization
            )
        else:
            entity_input_dim = rawob_shape[0] + entity_emb_dim
            self.obs_encoder = VectorObsProcess(entity_input_dim, h_dim, pre_dnorm=self.use_normalization)
        self.attention_layer = SimpleAttention(h_dim=h_dim)
        # # # # # # # # # #        actor        # # # # # # # # # # # #

        _size = n_entity * h_dim
        self.policy_feature = nn.Linear(_size, h_dim)
        self.policy_gru = nn.GRU(h_dim, h_dim, batch_first=True)
        self.policy_head = nn.Sequential(
            nn.Linear(h_dim, h_dim//2), nn.ReLU(inplace=True),
            nn.Linear(h_dim//2, head_dim))
        # # # # # # # # # # critic # # # # # # # # # # # #

        _size = n_entity * h_dim
        self.ct_encoder = nn.Sequential(nn.Linear(_size, h_dim), nn.ReLU(inplace=True), nn.Linear(h_dim, h_dim))
        self.ct_attention_layer = SimpleAttention(h_dim=h_dim)
        self.ct_gru = nn.GRU(h_dim, h_dim, batch_first=True)
        self.get_value = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),nn.Linear(h_dim, 1))

        # GRU hidden state dimension
        self.policy_hidden_dim = h_dim
        self.critic_hidden_dim = h_dim
        self.is_recurrent = True
        self.apply(weights_init)
        return
    # ==============================================================
    # ==========  ROLLOUT: one-step inference for environment ======
    # ==============================================================
    @Args2tensor_Return2numpy
    def act(self, obs=None, test_mode=None, avail_act=None,
            policy_hx=None, critic_hx=None):
        """
        Rollout mode: single environment step
        """
        # detect dead agents
        mask_dead = torch.isnan(obs)
        if self.img_obs:
            hwc = iterable_prod(mask_dead.shape[-3:])
            e_view = list(mask_dead.shape[:-3]) + [hwc]
        else:
            e_view = list(mask_dead.shape[:-1]) + [mask_dead.shape[-1]]
        mask_dead = mask_dead.reshape(e_view).any(-1)
        obs = torch.nan_to_num_(obs, 0)

        # -------- encode obs --------

        batch_size, n_agent = obs.shape[:2]
        n_entity = AlgorithmConfig.n_entity_placeholder

        if self.entity_embedding is not None:
            if not self.img_obs and len(obs.shape) == 4:
                    entity_indices = torch.arange(n_entity, device=obs.device)
                    entity_emb = self.entity_embedding(entity_indices)
                    entity_emb = entity_emb.unsqueeze(0).unsqueeze(0)
                    entity_emb = entity_emb.expand(batch_size, n_agent, -1, -1)
                    obs = torch.cat([obs, entity_emb], dim=-1)
            else: print("Warning: obs shape is not (n_traj, seq_len, n_agent, ...)")

        if self.img_obs:
            baec = self.obs_encoder(obs)
        else:
            baec = self.obs_encoder(obs, test_freeze_dnorm=test_mode)
        baec = self.attention_layer(k=baec, q=baec, v=baec, mask=mask_dead)

        batch_size, n_agent = obs.shape[:2]

        # -------- actor --------
        at_bac = my_view(baec, [0, 0, -1])  # (batch, n_agent, h_dim*n_entity)
        at_feat = torch.relu(self.policy_feature(at_bac))
        at_feat_seq = at_feat.view(batch_size * n_agent, 1, -1)

        if policy_hx is None:
            policy_hx = torch.zeros(1, batch_size * n_agent, self.policy_hidden_dim,
                                    device=at_feat.device, dtype=at_feat.dtype)
        else:
            policy_hx = policy_hx.view(1, batch_size * n_agent, self.policy_hidden_dim)

        gru_out, new_policy_hx = self.policy_gru(at_feat_seq, policy_hx)
        gru_out = gru_out.squeeze(1).view(batch_size, n_agent, -1)
        logits = self.policy_head(gru_out + at_feat)

        act, actLogProbs, distEntropy, probs = self._logit2act(
            logits_agent_cluster=logits,
            eval_mode=False,
            test_mode=test_mode,
            avail_act=avail_act
        )

        # -------- critic --------
        ct_bac = my_view(baec, [0, 0, -1])
        ct_bac = self.ct_encoder(ct_bac)
        ct_bac = self.ct_attention_layer(k=ct_bac, q=ct_bac, v=ct_bac)
        ct_bac_seq = ct_bac.view(batch_size * n_agent, 1, -1)

        if critic_hx is None:
            critic_hx = torch.zeros(1, batch_size * n_agent, self.critic_hidden_dim,
                                    device=ct_bac.device, dtype=ct_bac.dtype)
        else:
            critic_hx = critic_hx.view(1, batch_size * n_agent, self.critic_hidden_dim)

        ct_gru_out, new_critic_hx = self.ct_gru(ct_bac_seq, critic_hx)
        ct_gru_out = ct_gru_out.squeeze(1).view(batch_size, n_agent, -1)
        value = self.get_value(ct_gru_out + ct_bac)

        new_policy_hx = new_policy_hx.squeeze(0).view(batch_size, n_agent, self.policy_hidden_dim)
        new_critic_hx = new_critic_hx.squeeze(0).view(batch_size, n_agent, self.critic_hidden_dim)
        return act, value, actLogProbs, new_policy_hx, new_critic_hx

    # ==============================================================
    # ==========  EVALUATE (TRAIN): full trajectory pass ===========
    # ==============================================================
    @Args2tensor
    def evaluate_actions(self, obs=None, eval_actions=None, avail_act=None, traj_mask=None):
        """
        Eval/train mode: full trajectory forward
        """
        assert traj_mask is not None, "traj_mask must be provided during eval/train."

        mask_dead = torch.isnan(obs)
        if self.img_obs:
            hwc = iterable_prod(mask_dead.shape[-3:])
            e_view = list(mask_dead.shape[:-3]) + [hwc]
        else:
            e_view = list(mask_dead.shape[:-1]) + [mask_dead.shape[-1]]
        mask_dead = mask_dead.reshape(e_view).any(-1)
        obs = torch.nan_to_num_(obs, 0)

        n_traj, seq_len, n_agent = obs.shape[:3]
        n_entity = AlgorithmConfig.n_entity_placeholder

        if self.entity_embedding is not None:
            if not self.img_obs and len(obs.shape) == 5:
                    entity_indices = torch.arange(n_entity, device=obs.device)
                    entity_emb = self.entity_embedding(entity_indices)
                    entity_emb = entity_emb.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    entity_emb = entity_emb.expand(n_traj, seq_len, n_agent, -1, -1)
                    obs = torch.cat([obs, entity_emb], dim=-1)
            else: print("Warning: obs shape is not (n_traj, seq_len, n_agent, ...)")

        if self.img_obs:
            baec = self.obs_encoder(obs)
        else:
            baec = self.obs_encoder(obs, test_freeze_dnorm=True)
        baec = self.attention_layer(k=baec, q=baec, v=baec, mask=mask_dead)

        n_traj, seq_len, n_agent = obs.shape[:3]

        # -------- actor --------
        at_bac = my_view(baec, [0, 0, 0, -1])
        at_feat = torch.relu(self.policy_feature(at_bac))
        at_feat_seq = at_feat.permute(0, 2, 1, 3).reshape(n_traj * n_agent, seq_len, -1)

        policy_hx_init = torch.zeros(1, n_traj * n_agent, self.policy_hidden_dim,
                                     device=at_feat.device, dtype=at_feat.dtype)
        gru_out, _ = self.policy_gru(at_feat_seq, policy_hx_init)
        gru_out = gru_out.view(n_traj, n_agent, seq_len, -1).permute(0, 2, 1, 3)
        logits = self.policy_head(gru_out + at_feat)

        act, actLogProbs, distEntropy, probs = self._logit2act(
            logits_agent_cluster=logits,
            eval_mode=True,
            test_mode=False,
            eval_actions=eval_actions,
            avail_act=avail_act,
            traj_mask=traj_mask
        )

        # -------- critic --------
        ct_bac = my_view(baec, [0, 0, 0, -1])
        ct_bac = self.ct_encoder(ct_bac)
        ct_bac = self.ct_attention_layer(k=ct_bac, q=ct_bac, v=ct_bac)
        ct_bac_seq = ct_bac.permute(0, 2, 1, 3).reshape(n_traj * n_agent, seq_len, -1)

        critic_hx_init = torch.zeros(1, n_traj * n_agent, self.critic_hidden_dim,
                                     device=ct_bac.device, dtype=ct_bac.dtype)
        ct_gru_out, _ = self.ct_gru(ct_bac_seq, critic_hx_init)
        ct_gru_out = ct_gru_out.view(n_traj, n_agent, seq_len, -1).permute(0, 2, 1, 3)
        value = self.get_value(ct_gru_out + ct_bac)

        return value, actLogProbs, distEntropy, probs, {}


    def _logit2act(self, logits_agent_cluster, eval_mode, test_mode, eval_actions=None, avail_act=None, traj_mask: Optional[torch.Tensor]=None, **kwargs):
        # Mask padded timesteps to prevent NaN in distribution
        if eval_mode:
            assert traj_mask is not None, "traj_mask must be provided during eval."
            # traj_mask: (n_traj, seq_len) -> expand to (n_traj, seq_len, n_agent, n_action)
            mask_expanded = traj_mask.unsqueeze(-1).unsqueeze(-1)  # (n_traj, seq_len, 1, 1)
            # Replace NaN logits in padded positions with zeros (will be masked out in loss anyway)
            try:
                logits_agent_cluster = torch.where(mask_expanded > 0, logits_agent_cluster, torch.full_like(logits_agent_cluster, -1e8))
            except:
                logits_agent_cluster = torch.where(mask_expanded > 0, logits_agent_cluster, torch.full_like(logits_agent_cluster, -1e2))  # fp16
        
        if avail_act is not None:
            # logits_agent_cluster = torch.where(avail_act>0, logits_agent_cluster, -pt_inf())  # BUG overflow
            try:
                logits_agent_cluster = torch.where(
                    avail_act > 0,
                    logits_agent_cluster,
                    # torch.tensor(-1e8, dtype=logits_agent_cluster.dtype, device=logits_agent_cluster.device)
                    torch.tensor(-1e4, dtype=logits_agent_cluster.dtype, device=logits_agent_cluster.device)
                )
            except:
                logits_agent_cluster = torch.where(
                    avail_act > 0,
                    logits_agent_cluster,
                    torch.tensor(-1e2, dtype=logits_agent_cluster.dtype, device=logits_agent_cluster.device)  # fp16
                )

        logits_split = torch.split(logits_agent_cluster, self.action_spec, dim=-1)
        if avail_act is not None:
            avail_split = torch.split(avail_act, self.action_spec, dim=-1)
        else:
            avail_split = [None] * len(logits_split)

        if eval_mode and eval_actions is None:
            raise ValueError("eval_actions must be provided when eval_mode=True.")

        sampled_actions = []
        log_prob_sum = None
        entropy_terms = []
        probs_list = []

        for idx, (logits_head, avail_head) in enumerate(zip(logits_split, avail_split)):
            if avail_head is not None:
                try:
                    logits_head = torch.where(
                        avail_head > 0,
                        logits_head,
                        # torch.tensor(-1e8, dtype=logits_head.dtype, device=logits_head.device)
                        torch.tensor(-1e4, dtype=logits_head.dtype, device=logits_head.device)
                    )
                except:
                    logits_head = torch.where(
                        avail_head > 0,
                        logits_head,
                        torch.tensor(-1e2, dtype=logits_head.dtype, device=logits_head.device)  # fp16
                    )
            dist = Categorical(logits=logits_head)
            probs_list.append(dist.probs)

            if test_mode:
                action_head = torch.argmax(dist.probs, dim=-1)
            elif eval_mode:
                action_head = eval_actions[..., idx].long()
            else:
                action_head = dist.sample()

            sampled_actions.append(action_head)
            log_prob_head = dist.log_prob(action_head)
            log_prob_sum = log_prob_head if log_prob_sum is None else (log_prob_sum + log_prob_head)
            if eval_mode:
                entropy_terms.append(dist.entropy())

        act = torch.stack(sampled_actions, dim=-1)
        actLogProbs = log_prob_sum.unsqueeze(-1)
        distEntropy = torch.stack(entropy_terms, dim=-1).sum(dim=-1) if eval_mode else None
        probs = torch.cat(probs_list, dim=-1)
        return act, actLogProbs, distEntropy, probs
    
    
