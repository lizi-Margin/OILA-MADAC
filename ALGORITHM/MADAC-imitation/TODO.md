# ppo_ma Enhancement Backlog

## Current Snapshot
- `ALGORITHM/ppo_ma` is still the GRU-based IPPO copy (`ALGORITHM/ppo_ma_gru`) with attention + recurrent encoders sized for multi-agent entity observations.
- Single-agent runs (e.g. `bvr_3d`) rely on the same stack, so per-thread GRU state (`foundation.py:104-150`) and attention (`net.py:45-105`) stay active even when teamwork structure is unnecessary.
- Trajectory handling (`trajectory.py:200-313`) already stores policy/critic hidden states, which opens the door for recurrent rollouts, imitation reward mixing, and truncated BPTT—none of which are customized for single-agent use yet.

## TODO (proposed)

### Fixes & Hygiene
- [ ] **Use config knobs consistently:** apply `AlgorithmConfig.value_loss_coef` and remove the duplicate `clip_param` assignment in `ALGORITHM/ppo_ma/ppo.py:25-117`; add an explicit `use_avail_act` flag instead of reusing `ppo_epoch` (`ppo.py:19`).
- [ ] **Permissive avail-act handling:** let `ShellEnvWrapper.interact_with_env` fall back gracefully when missions do not expose `Avail-Act` (typical in single-agent/vector tasks) by skipping the raise at `shell_env.py:38-44` and masking actions in-network when the info is missing.
- [ ] **Hidden-state hygiene:** reset GRU states when an agent finishes inside an episode (not only on `Env-Suffered-Reset`); detect done flags from `StateRecall['Latest-Info']` before `policy_hidden_states` is reused (`foundation.py:143-150`).
- [ ] **Truncation support:** add a configurable rollout window (e.g., `algo_config.recurrent_seq_len`) so PPO replays truncated sequences rather than full-episode GRU histories (`trajectory.py:200-235` and `ppo_sampler.py:18-73`).

### Single-Agent Focused Improvements
- [ ] **State normalization polish:** expose per-feature statistics / running mean var dumps from `VectorObsProcess` so single-agent continuous sensors stay calibrated; add warm-up and optional layer-norm after the encoder (`net.py:33-97`).
- [ ] **Bypass heavy attention when not needed:** introduce a simpler MLP path (or gating to skip `SimpleAttention`) when `n_agent == 1` or `ScenarioConfig.EntityOriented` is false to reduce compute and stabilize gradients (`net.py:45-133`).
- [ ] **Adjust batch cadence:** scale `AlgorithmConfig.train_traj_needed` with `n_agent` to keep update frequency reasonable for single-agent long episodes; wire this into `BatchTrajManager` initialization (`foundation.py:96-107`, `trajectory.py:237-305`).
- [ ] **Better rollout diagnostics:** log per-thread episode length, value targets, and gradient norms to `mcv` during PPO updates so we can tune single-agent hyper-parameters faster (`ppo.py:59-120`).

### Model Architecture Upgrades
- [ ] **Actor/critic decoupling:** allow different encoders or MLP depths per head (and optional residual/skip connections) to improve value accuracy without destabilizing the policy (`net.py:49-149`).
- [ ] **GRU regularisation:** add dropout / layer norm to `policy_gru` and `ct_gru`, plus a config toggle for initial hidden state scaling to prevent drift on very long single-agent episodes (`net.py:50-60`, `net.py:114-147`).
- [ ] **Action distribution upgrades:** expose temperature / epsilon-greedy schedules and categorical smoothing for sparse-reward single-agent scenarios (`net.py:163-171` and sampler usage).

### AIRL / Imitation Integration
- [ ] **Reward mixer interface:** extend trajectory aggregation so we can inject AIRL discriminator outputs alongside environment rewards (e.g., accept `info['airl_reward']` in `ShellEnvWrapper` and blend inside `trajectory.reward_push_forward`, `trajectory.finalize`).
- [ ] **Discriminator training hooks:** add placeholders in `BatchTrajManager.train_and_clear_traj_pool` to call an imitation trainer before PPO updates, ensuring on-policy data can update both the discriminator and policy.
- [ ] **Config plumbing:** define JSONC knobs (weighting factor, AIRL checkpoint paths) under `AlgorithmConfig` and make sure they propagate through `foundation.py` into the reward mixer.

### Tooling & Validation
- [ ] **Single-agent smoke tests:** script a lightweight regression run (1 env, short horizon) to validate PPO + GRU + AIRL blending; capture metrics in `RESULT/` or CI.
- [ ] **Profile pass:** benchmark vector vs. image obs throughput with and without the new single-agent shortcuts to confirm we did not regress multi-agent performance.

