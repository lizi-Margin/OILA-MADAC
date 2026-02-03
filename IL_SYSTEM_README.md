# Imitation Learning (IL) System - Quick Start Guide

## Overview

The IL system provides a parallel training pipeline to the RL system for learning from expert demonstrations using Behavior Cloning (BC).

## Architecture

```
main.py
  ↓ (checks cfg.runner == 'il_runner')
il_runner.py → Runner.run()
  ↓ (spawns data loader process)
data_loader.py → data_loader_process()
  ↓ (loads trajectories via multiprocessing queue)
ALGORITHM.imitation_bc.foundation → BehaviorCloningFoundation
  ↓ (trains policy network)
ALGORITHM.imitation_bc.bc → BCTrainer
```

## Files

- **config.py**: Contains `GlobalConfig` with `runner` field (set to 'il_runner' for IL)
- **main.py**: Entry point, routes to `il_runner` when `cfg.runner == 'il_runner'`
- **il_runner.py**: Main IL training loop, manages data loading and training
- **data_loader.py**: Asynchronous trajectory data loader using multiprocessing
- **ALGORITHM/imitation_bc/foundation.py**: BC algorithm configuration and model management
- **ALGORITHM/imitation_bc/bc.py**: BC trainer with loss computation and optimization
- **ALGORITHM/imitation_bc/net.py**: Policy network architecture (actor-critic with GRU)

## Quick Start

### 1. Prepare Your Data

Your trajectory data should be in a format compatible with `uhtk.imitation.utils.safe_load_traj_pool()`.
Each trajectory should contain:
- `obs`: Observations (numpy array)
- `action`: Actions (numpy array or action indices)

Place your trajectory data in a directory, e.g., `./data/expert_trajectories/`

### 2. Configure Your Training

Edit `MISSION/imitation_bc_example.jsonc`:

```jsonc
{
    "config.py->GlobalConfig": {
        "runner": "il_runner",  // MUST be "il_runner"
        "logdir": "./RESULT/my_bc_training/",
        "seed": 1111
    },

    "il_runner.py->ILRunnerConfig": {
        "traj_dir": "./data/expert_trajectories",  // YOUR DATA PATH
        "N_LOAD": 2000,      // Number of loading iterations
        "traj_per_LOAD": 2,  // Trajectories per load
        "traj_reuse": 1      // Data reuse multiplier
    },

    "ALGORITHM.imitation_bc.foundation.py->AlgorithmConfig": {
        "obs_shape": [84, 84, 3],  // YOUR OBS SHAPE (vector or image)
        "n_actions": 18,            // YOUR ACTION SPACE SIZE
        "EntityOriented": false,    // true if entity-based obs
        "lr": 0.01,
        "num_epoch_per_update": 4
    }
}
```

**IMPORTANT**: Set the correct values for:
- `traj_dir`: Path to your trajectory data
- `obs_shape`: Must match your environment's observation shape
- `n_actions`: Must match your environment's action space
- `EntityOriented`: `true` if observations are entity-based (shape: [n_entities, features])

### 3. Run Training

```bash
python main.py --cfg MISSION/imitation_bc_example.jsonc
```

### 4. Monitor Training

Logs and checkpoints are saved to the directory specified in `logdir`:
- `model.pt`: Latest model checkpoint
- `history_cpt/model_*.pt`: Historical checkpoints (saved every `save_checkpoint_interval` updates)
- Training metrics are logged via `mcv` (multiagent silent logging bridge)

## Key Configuration Parameters

### ILRunnerConfig (il_runner.py)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `alg_name` | Algorithm class path | `"ALGORITHM.imitation_bc.foundation->BehaviorCloningFoundation"` |
| `traj_dir` | Path(s) to trajectory data | `"./data/trajectories"` or `["path1", "path2"]` |
| `N_LOAD` | Number of data loading iterations | `2000` |
| `traj_per_LOAD` | Trajectories loaded per iteration | `2` |
| `traj_reuse` | Times to reuse each trajectory | `1` |

### AlgorithmConfig (ALGORITHM/imitation_bc/foundation.py)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `obs_shape` | Observation shape | `[12]` (vector) or `[84, 84, 3]` (image) |
| `n_actions` | Number of discrete actions | `12` |
| `EntityOriented` | Entity-based observations | `false` (vector) or `true` (entities) |
| `lr` | Learning rate | `0.01` |
| `lr_sheduler` | Use LR scheduler | `true` |
| `num_epoch_per_update` | Epochs per data batch | `4` |
| `sample_size_min/max` | Training batch size range | `45` |
| `save_checkpoint_interval` | Model save frequency (updates) | `10` |
| `net_hdim` | Network hidden dimension | `256` |
| `use_normalization` | Observation normalization | `true` |

## Network Architecture

The BC network (ALGORITHM/imitation_bc/net.py) uses:

- **Encoder**:
  - Vector observations: MLP with normalization
  - Image observations: IMPALA CNN
- **Attention Layer**: Self-attention over entities (if EntityOriented)
- **Policy Head**: GRU + MLP → action logits
- **Critic Head**: GRU + MLP → value estimate

## Data Loading

The data loader runs in a separate process and uses a multiprocessing queue:
1. Loads trajectories from disk asynchronously
2. Extracts `obs` and `action` fields
3. Feeds batches to the trainer via queue (maxsize=2 for buffering)

## Differences from RL Runner

| Aspect | RL Runner | IL Runner |
|--------|-----------|-----------|
| **Environments** | Parallel simulated environments | Offline trajectory data |
| **Multi-threading** | Required (`num_threads`, `fold`) | Not used |
| **Mission Config** | Required (`env_name`, `ScenarioConfig`) | Not used |
| **Data Source** | Live interaction | Pre-collected trajectories |
| **Algorithm** | RL algorithms (PPO, SAC, etc.) | BC (supervised learning) |

## Troubleshooting

### Issue: "has no such config item"
- **Cause**: JSONC file references a non-existent config attribute
- **Solution**: Check that all config keys match the class attributes in the Python files

### Issue: CUDA out of memory
- **Cause**: Batch size (`sample_size_min/max`) too large
- **Solution**: Reduce `sample_size_min` and `sample_size_max` in config
- **Note**: The trainer automatically reduces batch size by 10% on OOM and retries

### Issue: Data loading hangs
- **Cause**: Queue is full or data loader process crashed
- **Solution**: Check that `traj_dir` path is correct and trajectories are readable

### Issue: Checkpoint not saving
- **Cause**: Missing `self.optimizer` attribute in foundation.py
- **Solution**: Ensure BCTrainer exposes optimizer: `self.optimizer = self.trainer.optimizer`

## Advanced Usage

### Multiple Data Sources

```jsonc
"il_runner.py->ILRunnerConfig": {
    "traj_dir": ["./data/expert1", "./data/expert2", "./data/expert3"]
}
```

The data loader will randomly sample from all provided directories.

### Loading from Checkpoint

```jsonc
"ALGORITHM.imitation_bc.foundation.py->AlgorithmConfig": {
    "load_checkpoint": true,
    "load_specific_checkpoint": "./RESULT/prev_run/model.pt"
}
```

### Custom Network Architecture

Modify `ALGORITHM/imitation_bc/net.py` to change:
- Encoder type (CNN vs MLP)
- Hidden dimensions
- Number of layers
- Attention mechanisms

## Next Steps

1. Prepare your expert trajectory data
2. Configure observation and action spaces
3. Run training and monitor logs
4. Fine-tune hyperparameters (LR, batch size, epochs)
5. Evaluate trained policy in your environment

For questions, refer to:
- `CLAUDE.md`: Overall framework documentation
- `il_runner.py`: IL training loop implementation
- `ALGORITHM/imitation_bc/`: BC algorithm details
