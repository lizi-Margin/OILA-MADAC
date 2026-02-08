# OILA-MADAC

Online Imitation learning Augmented Multi-Actor-Double-Attention-Critic for cooperative BVR air combat.

## Overview

- **Algorithm**: OILA-MADAC combines online imitation learning with MADAC (Multi-Actor-Double-Attention-Critic)
- **Environment**: High-fidelity 2v2 BVR air combat simulator (6-DOF aircraft, AAM guidance, radar/Datalink)

## Installation

```bash
git clone --recurse-submodules https://github.com/lizi-Margin/OILA-MADAC.git
cd OILA-MADAC
pip install -r requirements.txt
```

## Environment Setup

**Prerequisites**
- **Windows**: Visual Studio 2019 or 2022 with "Desktop development with C++" workload
- **Linux**: GCC 15 is recommended

Build C++ environment first (required for training):

**Windows (MSVC)**
```bash
cd MISSION/bvr_sim
build_windows.bat
cd ../..
```

**Linux (GCC)**
```bash
cd MISSION/bvr_sim
chmod +x build_linux.sh
./build_linux.sh
cd ../..
```

## Training (Two-Phase)

**Phase 1: Online Imitation Learning**

```bash
python main.py --cfg MISSION/bvr_sim/conf_system/cpp/MADAC_imitation-2v2-entity-denseReward.jsonc
```

**Phase 2: Reinforcement Learning**

```bash
python main.py --cfg MISSION/bvr_sim/conf_system/cpp/MADAC_reinforce-2v2-entity-denseReward.jsonc
```

## Project Structure

```
MISSION/bvr_sim/
├── env_wrapper.py          # UHRL wrapper
├── bvr_env.py              # Pure Python environment
├── bvr_env_cpp.py          # C++ wrapper
├── build_windows.bat       # Windows build script
├── build_linux.sh          # Linux build script
└── conf_system/
    ├── cpp/                # C++ environment configs
    └── python/             # Python-only configs
ALGORITHM/
├── MADAC_imitation/        # Online imitation learning
├── MADAC_reinforce/        # MADAC PPO training
└── random/                 # Random baseline
```

## Citation

```bibtex
@misc{ouilamadac2024,
  title={OILA-MADAC: Online Imitation learning Augmented Multi-Actor-Double-Attention-Critic},
  author={},
  year={2026}
}
```
