import torch, math, copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, List
from uhtk.print_pack import *
from uhtk.siri.utils.iterable_tools import iterable_prod
from uhtk.encoder import ImgObsProcess, VectorObsProcess
from torch.distributions.categorical import Categorical
from uhtk.UTIL.tensor_ops import Args2tensor_Return2numpy, Args2tensor, __hashn__, my_view
from uhtk.UTIL.tensor_ops import pt_inf
from .foundation import AlgorithmConfig
from ALGORITHM.common.attention import SimpleAttention
from uhtk.encoder.norm import DynamicNormFix
from ALGORITHM.common.net_manifest import weights_init
from ALGORITHM.common.temporal_blocks import TemporalSequenceEncoder
from ALGORITHM.common.mlp_blocks import ResidualMLPBlock

class NetConfig:
    clamp_value = 10.0

    policy_temporal_layers = 2
    policy_temporal_heads = 4
    policy_temporal_dropout = 0.1
    policy_temporal_window = -1

    critic_temporal_layers = 4
    critic_temporal_heads = 4
    critic_temporal_dropout = 0.1
    critic_temporal_window = -1


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
        
        # # # # # # # # # #  actor-critic share # # # # # # # # # # # #
        if self.img_obs:
            self.obs_encoder = ImgObsProcess(
                imgshape_hwc=rawob_shape,
                impala_chans=(8, 16, 16),
                cnn_outsize=512,
                output_size=h_dim,
                pre_norm=self.use_normalization
            )
            self.vector_res_blocks = None
        else:
            self.obs_encoder = VectorObsProcess(rawob_shape[0], h_dim, pre_dnorm=self.use_normalization)
            self.vector_res_blocks = nn.Sequential(
                ResidualMLPBlock(h_dim, hidden_dim=h_dim * 2, dropout=0.05, clamp_value=NetConfig.clamp_value),
                ResidualMLPBlock(h_dim, hidden_dim=h_dim * 2, dropout=0.05, clamp_value=NetConfig.clamp_value),
                ResidualMLPBlock(h_dim, hidden_dim=h_dim * 2, dropout=0.05, clamp_value=NetConfig.clamp_value),
            )
        self.post_attn_norm = nn.LayerNorm(h_dim)
        self.pre_attn_norm = nn.LayerNorm(h_dim)
        self.attention_layer = SimpleAttention(h_dim=h_dim)
        # # # # # # # # # #        actor        # # # # # # # # # # # #

        _size = n_entity * h_dim
        self.policy_feature = nn.Linear(_size, h_dim)
        self.policy_feature_norm = nn.LayerNorm(h_dim)
        self.policy_temporal_encoder = TemporalSequenceEncoder(
            dim=h_dim,
            num_layers=NetConfig.policy_temporal_layers,
            num_heads=NetConfig.policy_temporal_heads,
            dropout=NetConfig.policy_temporal_dropout,
            window=NetConfig.policy_temporal_window,
        )
        self.policy_head = nn.Sequential(
            ResidualMLPBlock(h_dim, hidden_dim=h_dim * 2, dropout=0.1, clamp_value=NetConfig.clamp_value),
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, h_dim // 2),
            nn.SiLU(),
            nn.Linear(h_dim // 2, head_dim),
        )
        # # # # # # # # # # critic # # # # # # # # # # # #

        _size = n_entity * h_dim
        self.ct_encoder = nn.Sequential(
            nn.Linear(_size, h_dim),
            nn.SiLU(),
            ResidualMLPBlock(h_dim, hidden_dim=h_dim * 2, dropout=0.1, clamp_value=NetConfig.clamp_value),
        )
        self.ct_feature_norm = nn.LayerNorm(h_dim)
        self.ct_attention_layer = SimpleAttention(h_dim=h_dim)
        self.critic_temporal_encoder = TemporalSequenceEncoder(
            dim=h_dim,
            num_layers=NetConfig.critic_temporal_layers,
            num_heads=NetConfig.critic_temporal_heads,
            dropout=NetConfig.critic_temporal_dropout,
            window=NetConfig.critic_temporal_window,
        )
        self.get_value = nn.Sequential(
            ResidualMLPBlock(h_dim, hidden_dim=h_dim * 2, dropout=0.1, clamp_value=NetConfig.clamp_value),
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, h_dim // 2),
            nn.SiLU(),
            nn.Linear(h_dim // 2, 1)
        )

        self.policy_window = NetConfig.policy_temporal_window
        self.critic_window = NetConfig.critic_temporal_window
        self.policy_token_dim = h_dim
        self.critic_token_dim = h_dim
        self.policy_hidden_dim = (
            h_dim * self.policy_window + self.policy_window if self.policy_window > 0 else h_dim
        )
        self.critic_hidden_dim = (
            h_dim * self.critic_window + self.critic_window if self.critic_window > 0 else h_dim
        )
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
        if self.img_obs:
            baec = self.obs_encoder(obs)
        else:
            baec = self.obs_encoder(obs, test_freeze_dnorm=test_mode)
            if self.vector_res_blocks is not None:
                baec = self.vector_res_blocks(baec)

        baec = self.pre_attn_norm(baec)
        baec = self.attention_layer(k=baec, q=baec, v=baec, mask=mask_dead)
        baec = self.post_attn_norm(baec)

        batch_size, n_agent = obs.shape[:2]

        # -------- actor --------
        at_bac = my_view(baec, [0, 0, -1])  # (batch, n_agent, h_dim*n_entity)
        at_feat = self.policy_feature(at_bac)
        at_feat = self.policy_feature_norm(at_feat)
        at_feat = F.silu(at_feat)
        at_feat = at_feat.clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)
        at_feat_seq = at_feat.view(batch_size * n_agent, 1, -1)

        policy_state = self._unpack_temporal_state(
            policy_hx, batch_size, n_agent, self.policy_window, self.policy_token_dim, at_feat.device
        )
        attn_mask_step = mask_dead.bool().any(-1)  # squeeze the entity dim
        encoded_token, new_policy_state = self.policy_temporal_encoder.forward_step(
            at_feat, policy_state, token_mask=attn_mask_step
        )
        encoded_token = encoded_token.clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)
        head_input = encoded_token + at_feat
        logits = self.policy_head(head_input)

        act, actLogProbs, distEntropy, probs = self._logit2act(
            logits_agent_cluster=logits,
            eval_mode=False,
            test_mode=test_mode,
            avail_act=avail_act
        )

        # -------- critic --------
        ct_bac = my_view(baec, [0, 0, -1])
        ct_bac = self.ct_encoder(ct_bac)
        ct_bac = self.ct_feature_norm(ct_bac)
        ct_bac = F.silu(ct_bac)
        ct_bac = ct_bac.clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)
        ct_bac = self.ct_attention_layer(k=ct_bac, q=ct_bac, v=ct_bac)
        critic_state = self._unpack_temporal_state(
            critic_hx, batch_size, n_agent, self.critic_window, self.critic_token_dim, ct_bac.device
        )
        critic_encoded, new_critic_state = self.critic_temporal_encoder.forward_step(
            ct_bac, critic_state, token_mask=attn_mask_step
        )
        critic_encoded = critic_encoded.clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)
        value_input = critic_encoded + ct_bac
        value_input = value_input.clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)
        value = self.get_value(value_input)

        new_policy_hx = self._pack_temporal_state(new_policy_state, self.policy_window, batch_size, n_agent, self.policy_token_dim, at_feat.device)
        new_critic_hx = self._pack_temporal_state(new_critic_state, self.critic_window, batch_size, n_agent, self.critic_token_dim, ct_bac.device)
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

        if self.img_obs:
            baec = self.obs_encoder(obs)
        else:
            baec = self.obs_encoder(obs, test_freeze_dnorm=True)
            if self.vector_res_blocks is not None:
                baec = self.vector_res_blocks(baec)

        baec = self.pre_attn_norm(baec)
        baec = self.attention_layer(k=baec, q=baec, v=baec, mask=mask_dead)
        baec = self.post_attn_norm(baec)

        n_traj, seq_len, n_agent = obs.shape[:3]

        # -------- actor --------
        at_bac = my_view(baec, [0, 0, 0, -1])
        at_feat = self.policy_feature(at_bac)
        at_feat = self.policy_feature_norm(at_feat)
        at_feat = F.silu(at_feat)
        at_feat = at_feat.clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)
        dead_mask = mask_dead.bool()
        combined_mask = dead_mask
        if traj_mask is not None:
            temporal_mask = ~(traj_mask > 0).bool()
            temporal_mask = temporal_mask.unsqueeze(-1).unsqueeze(-1)  # agent, entity
            temporal_mask = temporal_mask.expand_as(dead_mask)
            combined_mask = combined_mask | temporal_mask
        combined_mask = combined_mask.any(-1)  # squeeze the entity dim
        combined_mask = combined_mask.permute(0, 2, 1)
        policy_sequence = at_feat.permute(0, 2, 1, 3)
        encoded_policy = self.policy_temporal_encoder.forward_sequence(policy_sequence, combined_mask)
        encoded_policy = encoded_policy.permute(0, 2, 1, 3)
        head_input = encoded_policy + at_feat
        logits = self.policy_head(head_input)

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
        ct_bac = self.ct_feature_norm(ct_bac)
        ct_bac = F.silu(ct_bac)
        ct_bac = ct_bac.clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)
        ct_bac = self.ct_attention_layer(k=ct_bac, q=ct_bac, v=ct_bac)
        critic_sequence = ct_bac.permute(0, 2, 1, 3)
        encoded_critic = self.critic_temporal_encoder.forward_sequence(critic_sequence, combined_mask)
        encoded_critic = encoded_critic.permute(0, 2, 1, 3)
        value_input = encoded_critic + ct_bac
        value_input = value_input.clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)
        value = self.get_value(value_input)

        return value, actLogProbs, distEntropy, probs, {}


    def _logit2act(self, logits_agent_cluster, eval_mode, test_mode, eval_actions=None, avail_act=None, traj_mask: Optional[torch.Tensor]=None, **kwargs):
        # Mask padded timesteps to prevent NaN in distribution
        if eval_mode:
            assert traj_mask is not None, "traj_mask must be provided during eval."
            # traj_mask: (n_traj, seq_len) -> expand to (n_traj, seq_len, n_agent, n_action)
            mask_expanded = traj_mask.unsqueeze(-1).unsqueeze(-1)  # (n_traj, seq_len, 1, 1)
            # Replace NaN logits in padded positions with zeros (will be masked out in loss anyway)
            logits_agent_cluster = torch.where(mask_expanded > 0, logits_agent_cluster, torch.full_like(logits_agent_cluster, -1e8))
        
        if avail_act is not None:
            # logits_agent_cluster = torch.where(avail_act>0, logits_agent_cluster, -pt_inf())  # BUG overflow
            logits_agent_cluster = torch.where(
                avail_act > 0,
                logits_agent_cluster,
                torch.tensor(-1e8, dtype=logits_agent_cluster.dtype, device=logits_agent_cluster.device)
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
                logits_head = torch.where(
                    avail_head > 0,
                    logits_head,
                    torch.tensor(-1e8, dtype=logits_head.dtype, device=logits_head.device)
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

    def _unpack_temporal_state(self, state, batch_size, n_agent, window, hidden_dim, device):
        if state is None or window <= 0:
            return None
        expected = window * hidden_dim + window
        state = state.view(batch_size, n_agent, expected)
        seq_flat = state[..., : window * hidden_dim]
        mask_flat = state[..., window * hidden_dim :]
        sequence = seq_flat.view(batch_size, n_agent, window, hidden_dim)
        mask = mask_flat.view(batch_size, n_agent, window) > 0.5
        return sequence.to(device), mask.to(device)

    def _pack_temporal_state(self, state_tuple, window, batch_size, n_agent, hidden_dim, device):
        expected = window * hidden_dim + window
        if window <= 0:
            return torch.zeros(batch_size, n_agent, hidden_dim, device=device)
        if state_tuple is None:
            return torch.zeros(batch_size, n_agent, expected, device=device)
        sequence, mask = state_tuple
        bsz, n_agent, seq_len, hidden_dim = sequence.shape
        if mask is None:
            mask = torch.zeros(bsz, n_agent, seq_len, dtype=torch.bool, device=sequence.device)
        if seq_len < window:
            pad_len = window - seq_len
            pad_seq = sequence.new_zeros(bsz, n_agent, pad_len, hidden_dim)
            pad_mask = torch.ones(bsz, n_agent, pad_len, dtype=torch.bool, device=sequence.device)
            sequence = torch.cat([pad_seq, sequence], dim=2)
            mask = torch.cat([pad_mask, mask], dim=2)
        elif seq_len > window:
            sequence = sequence[:, :, -window:, :]
            mask = mask[:, :, -window:]
        seq_flat = sequence.reshape(bsz, n_agent, window * hidden_dim)
        mask_flat = mask.reshape(bsz, n_agent, window).float()
        return torch.cat([seq_flat, mask_flat], dim=-1)
    
    
