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
from ALGORITHM.common.net_manifest import weights_init
from ALGORITHM.common.temporal_blocks import TemporalSequenceEncoder
from ALGORITHM.common.mlp_blocks import ResidualMLPBlock


class EntityEncoder(nn.Module):
    """Applies entity-wise self-attention with optional pre/post LayerNorm."""

    def __init__(self, dim: int, use_pre_norm: bool = True, use_post_norm: bool = True):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim) if use_pre_norm else None
        self.attention = SimpleAttention(h_dim=dim)
        self.post_norm = nn.LayerNorm(dim) if use_post_norm else None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        x = self.attention(k=x, q=x, v=x, mask=mask)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x

class NetConfig:
    clamp_value = 10.0

    shared_temporal_layers = 2
    shared_temporal_heads = 2
    shared_temporal_dropout = 0.1
    shared_temporal_window = 400


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
        self.entity_encoder = EntityEncoder(dim=h_dim, use_pre_norm=True, use_post_norm=True)
        flat_dim = n_entity * h_dim
        self.shared_temporal_encoder = TemporalSequenceEncoder(
            dim=flat_dim,
            num_layers=NetConfig.shared_temporal_layers,
            num_heads=NetConfig.shared_temporal_heads,
            dropout=NetConfig.shared_temporal_dropout,
            window=NetConfig.shared_temporal_window,
        )
        # # # # # # # # # #        actor        # # # # # # # # # # # #

        self.policy_feature = nn.Linear(flat_dim, h_dim)
        self.policy_feature_norm = nn.LayerNorm(h_dim)
        self.policy_gru = nn.GRU(h_dim, h_dim, batch_first=True)
        self.policy_head = nn.Sequential(
            ResidualMLPBlock(h_dim, hidden_dim=h_dim * 2, dropout=0.1, clamp_value=NetConfig.clamp_value),
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, h_dim // 2),
            nn.SiLU(),
            nn.Linear(h_dim // 2, head_dim),
        )
        # # # # # # # # # # critic # # # # # # # # # # # #

        self.ct_encoder = nn.Sequential(
            nn.Linear(flat_dim, h_dim),
            nn.SiLU(),
            ResidualMLPBlock(h_dim, hidden_dim=h_dim * 2, dropout=0.1, clamp_value=NetConfig.clamp_value),
        )
        self.ct_feature_norm = nn.LayerNorm(h_dim)
        self.ct_entity_encoder = EntityEncoder(dim=h_dim, use_pre_norm=False, use_post_norm=False)
        self.ct_gru = nn.GRU(h_dim, h_dim, batch_first=True)
        self.get_value = nn.Sequential(
            ResidualMLPBlock(h_dim, hidden_dim=h_dim * 2, dropout=0.1, clamp_value=NetConfig.clamp_value),
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, h_dim // 2),
            nn.SiLU(),
            nn.Linear(h_dim // 2, 1)
        )

        self.shared_window = NetConfig.shared_temporal_window
        self.shared_token_dim = flat_dim
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
        if self.img_obs:
            baec = self.obs_encoder(obs)
        else:
            baec = self.obs_encoder(obs, test_freeze_dnorm=test_mode)
            if self.vector_res_blocks is not None:
                baec = self.vector_res_blocks(baec)

        baec = self.entity_encoder(baec, mask=mask_dead)

        batch_size, n_agent = obs.shape[:2]

        # -------- shared temporal encoder --------
        entity_flat = my_view(baec, [0, 0, -1])  # (batch, n_agent, n_entity * h_dim)
        shared_state, policy_gru_hidden = self._unpack_policy_state(
            policy_hx, batch_size, n_agent, entity_flat.device
        )
        attn_mask_step = mask_dead.bool().any(-1)  # squeeze the entity dim
        shared_token, new_shared_state = self.shared_temporal_encoder.forward_step(
            entity_flat, shared_state, token_mask=attn_mask_step
        )
        shared_token = shared_token.clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)

        # -------- actor --------
        actor_feat = self.policy_feature(shared_token)
        actor_feat = self.policy_feature_norm(actor_feat)
        actor_feat = F.silu(actor_feat)
        actor_feat = actor_feat.clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)

        actor_token_seq = actor_feat.view(batch_size * n_agent, 1, -1)
        policy_gru_state = self._expand_gru_state(
            policy_gru_hidden, batch_size, n_agent, self.policy_hidden_dim, actor_feat.device
        )
        gru_out, new_policy_gru = self.policy_gru(actor_token_seq, policy_gru_state)
        gru_out = gru_out.squeeze(1).view(batch_size, n_agent, -1)
        head_input = (gru_out + actor_feat).clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)
        logits = self.policy_head(head_input)

        act, actLogProbs, distEntropy, probs = self._logit2act(
            logits_agent_cluster=logits,
            eval_mode=False,
            test_mode=test_mode,
            avail_act=avail_act
        )

        # -------- critic --------
        ct_bac = self.ct_encoder(shared_token)
        ct_bac = self.ct_feature_norm(ct_bac)
        ct_bac = F.silu(ct_bac)
        ct_bac = ct_bac.clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)
        ct_bac = self.ct_entity_encoder(ct_bac)
        critic_token_seq = ct_bac.view(batch_size * n_agent, 1, -1)
        critic_gru_state = self._expand_gru_state(
            critic_hx, batch_size, n_agent, self.critic_hidden_dim, ct_bac.device
        )
        ct_gru_out, new_critic_gru = self.ct_gru(critic_token_seq, critic_gru_state)
        ct_gru_out = ct_gru_out.squeeze(1).view(batch_size, n_agent, -1)
        value_input = (ct_gru_out + ct_bac).clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)
        value_input = value_input.clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)
        value = self.get_value(value_input)

        new_policy_gru_collapsed = self._collapse_gru_state(new_policy_gru, batch_size, n_agent)
        new_critic_gru_collapsed = self._collapse_gru_state(new_critic_gru, batch_size, n_agent)
        new_policy_hx = self._pack_policy_state(
            new_shared_state, new_policy_gru_collapsed, batch_size, n_agent, actor_feat.device
        )
        new_critic_hx = self._pack_critic_state(
            new_critic_gru_collapsed, batch_size, n_agent, ct_bac.device
        )
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

        baec = self.entity_encoder(baec, mask=mask_dead)

        n_traj, seq_len, n_agent = obs.shape[:3]

        # -------- actor --------
        entity_flat = my_view(baec, [0, 0, 0, -1])
        dead_mask = mask_dead.bool()
        combined_mask = dead_mask
        if traj_mask is not None:
            temporal_mask = ~(traj_mask > 0).bool()
            temporal_mask = temporal_mask.unsqueeze(-1).unsqueeze(-1)  # agent, entity
            temporal_mask = temporal_mask.expand_as(dead_mask)
            combined_mask = combined_mask | temporal_mask
        combined_mask = combined_mask.any(-1)  # squeeze the entity dim
        combined_mask = combined_mask.permute(0, 2, 1)
        shared_sequence = entity_flat.permute(0, 2, 1, 3)
        shared_encoded = self.shared_temporal_encoder.forward_sequence(shared_sequence, combined_mask)
        shared_encoded = shared_encoded.permute(0, 2, 1, 3)
        shared_encoded = shared_encoded.clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)

        actor_feat = self.policy_feature(shared_encoded)
        actor_feat = self.policy_feature_norm(actor_feat)
        actor_feat = F.silu(actor_feat)
        actor_feat = actor_feat.clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)

        actor_seq = actor_feat.permute(0, 2, 1, 3).reshape(n_traj * n_agent, seq_len, -1)
        policy_hx_init = torch.zeros(
            1, n_traj * n_agent, self.policy_hidden_dim,
            device=actor_seq.device, dtype=actor_seq.dtype
        )
        gru_out, _ = self.policy_gru(actor_seq, policy_hx_init)
        gru_out = gru_out.view(n_traj, n_agent, seq_len, -1).permute(0, 2, 1, 3)
        head_input = (gru_out + actor_feat).clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)
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
        ct_bac = self.ct_encoder(shared_encoded)
        ct_bac = self.ct_feature_norm(ct_bac)
        ct_bac = F.silu(ct_bac)
        ct_bac = ct_bac.clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)
        ct_bac = self.ct_entity_encoder(ct_bac)
        critic_seq = ct_bac.permute(0, 2, 1, 3).reshape(n_traj * n_agent, seq_len, -1)
        critic_hx_init = torch.zeros(
            1, n_traj * n_agent, self.critic_hidden_dim,
            device=critic_seq.device, dtype=critic_seq.dtype
        )
        ct_gru_out, _ = self.ct_gru(critic_seq, critic_hx_init)
        ct_gru_out = ct_gru_out.view(n_traj, n_agent, seq_len, -1).permute(0, 2, 1, 3)
        value_input = (ct_gru_out + ct_bac).clamp_(-NetConfig.clamp_value, NetConfig.clamp_value)
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

    def _unpack_policy_state(self, state, batch_size, n_agent, device):
        if state is None:
            return None, None
        shared_size = (
            self.shared_window * self.shared_token_dim + self.shared_window
            if self.shared_window > 0
            else 0
        )
        total_size = shared_size + self.policy_hidden_dim
        state = state.view(batch_size, n_agent, total_size)
        shared_flat = state[..., :shared_size] if shared_size > 0 else None
        if shared_flat is not None:
            shared_state = self._unpack_temporal_component(
                shared_flat, self.shared_window, self.shared_token_dim, device
            )
        else:
            shared_state = None
        gru_flat = state[..., shared_size:] if self.policy_hidden_dim > 0 else None
        gru_hidden = gru_flat.to(device) if gru_flat is not None else None
        return shared_state, gru_hidden

    def _pack_policy_state(self, shared_state, gru_hidden, batch_size, n_agent, device):
        shared_flat = self._pack_temporal_component(
            shared_state, self.shared_window, batch_size, n_agent, self.shared_token_dim, device
        )
        if gru_hidden is None:
            gru_hidden = torch.zeros(batch_size, n_agent, self.policy_hidden_dim, device=device)
        else:
            gru_hidden = gru_hidden.to(device).view(batch_size, n_agent, self.policy_hidden_dim)
        if shared_flat is None:
            return gru_hidden
        return torch.cat([shared_flat, gru_hidden], dim=-1)

    def _pack_critic_state(self, gru_hidden, batch_size, n_agent, device):
        if gru_hidden is None:
            return torch.zeros(batch_size, n_agent, self.critic_hidden_dim, device=device)
        return gru_hidden.to(device).view(batch_size, n_agent, self.critic_hidden_dim)

    def _unpack_temporal_component(self, flat, window, hidden_dim, device):
        if flat is None or window <= 0:
            return None
        bsz, n_agent, _ = flat.shape
        seq_flat = flat[..., : window * hidden_dim]
        mask_flat = flat[..., window * hidden_dim :]
        sequence = seq_flat.view(bsz, n_agent, window, hidden_dim)
        mask = mask_flat.view(bsz, n_agent, window) > 0.5
        return sequence.to(device), mask.to(device)

    def _pack_temporal_component(self, state_tuple, window, batch_size, n_agent, hidden_dim, device):
        if window <= 0:
            return None
        if state_tuple is None:
            return torch.zeros(batch_size, n_agent, window * hidden_dim + window, device=device)
        sequence, mask = state_tuple
        bsz, n_agent, seq_len, _ = sequence.shape
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

    def _expand_gru_state(self, gru_hidden, batch_size, n_agent, hidden_dim, device):
        if gru_hidden is None:
            return torch.zeros(1, batch_size * n_agent, hidden_dim, device=device)
        gru_hidden = gru_hidden.to(device).view(batch_size * n_agent, hidden_dim)
        return gru_hidden.unsqueeze(0)

    @staticmethod
    def _collapse_gru_state(gru_hidden, batch_size, n_agent):
        return gru_hidden.squeeze(0).view(batch_size, n_agent, -1)
    
    
