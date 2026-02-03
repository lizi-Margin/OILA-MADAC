# UHRL - Universal Hulc Reinforcement Learning Framework

UHRL（Universal Hulc Reinforcement Learning）是一个多智能体强化学习框架，用于在复杂环境中训练和评估RL算法。该架构支持多团队场景和并行环境执行，使用模块化插件系统实现算法和任务。

## 项目概述

UHRL是基于UHTK（Universal Hulc ToolKit）组件库的多智能体强化学习框架，支持多团队场景和并行环境执行。它保留了原始HMP框架的解耦能力和配置文件注入系统，同时进行了简化：

1. 大幅缩减配置项数量，取消ChainVar，减小环境开发者的配置检查压力
2. 取消AsUnity系列配置（ObsAsUnity等），这些逻辑不必由平台实现
3. 观察/动作空间完全在ScenarioConfig中定义，而不通过参数传递给算法，提升了环境<->算法交互的灵活性
4. 只保留必须使用的代码

## 安装与运行系统

### 系统要求
- Python 3.7+ (目前主要适配3.7和3.12，其他版本应该也能用)
- CUDA 11+ (推荐，用于GPU训练)

### 安装步骤

1. 克隆仓库
```bash
git clone --recurse-submodules https://github.com/lizi-Margin/UHRL.git
# git submodule update --init --recursive
cd UHRL

2. 安装依赖
```bash
pip install -r requirements.txt
```

### 基本命令结构
```bash
python main.py --cfg <path_to_config.jsonc>
```

### 示例配置
配置文件采用JSONC格式（带注释的JSON），可以在以下位置找到：
- `RESULT/*/config_backup.jsonc`
- `MISSION/example.jsonc` (模板)

### 示例命令
```bash
python main.py --cfg MISSION/example.jsonc
```

或者使用PowerShell脚本:
```powershell
python ./main.py -c MISSION\bvr_3d_v2\conf_system\ppo_ma_gru-1v1-high_accuracy.jsonc
```

### 故障排除
- 已测试Python版本: 3.13, 3.12, 3.8, 3.7
- 如遇到CUDA相关问题，检查NVIDIA驱动, CUDA版本, pytorch版本(是否过旧)

## 架构组件

### 核心组件
1. **main.py** - 初始化配置、创建并行环境并启动训练runner的入口点
2. **conf_system.py** - 加载JSONC文件并覆盖Python类属性的配置系统
3. **task_runner.py** - 管理回合生命周期、奖励跟踪和指标日志记录的主训练循环协调器
4. **env_router.py** - 通过导入路径将环境名称映射到环境类的插件系统
5. **mt_mapper.py** (MTM) - 多团队映射器，路由环境和算法实例之间的观察和动作
6. **uhtk/** - Universal Hulc ToolKit：带有日志、可视化和实用模块的可重用工具库

### 任务系统
- **MISSION/** - 包含各种环境（lag, bvr_2d, bvr_3d, bvr_3d_v2, native_gym, narrow_gap, uezoo等）
- 每个任务提供ScenarioConfig、make_env工厂函数和实现标准接口的环境包装器

### 算法系统
- **ALGORITHM/** - 包含各种RL算法（ppo_ma_lag, ppo_ma_gru, random, stable_baselines3等）
- 每个算法提供AlgorithmConfig和实现interact_with_env方法的AlgorithmFoundation

## 现有强化学习算法

UHRL框架支持多种强化学习算法：

- **PPO Multi-Agent LAG (ppo_ma_lag)**: 针对LAG环境优化的近端策略优化算法
- **PPO Multi-Agent GRU (ppo_ma_gru)**: 使用门控循环单元的PPO算法
- **随机算法 (random)**: 用于基线比较的随机策略
- **Stable-Baselines3集成**: 支持集成Stable-Baselines3中的各种算法

## 现有环境

UHRL框架提供了多种强化学习环境：

- **LAG (Light Air Game)**: WVR/BVR空战仿真环境
- **BVR 2D/3D/3D_V2**: 超视距空战仿真环境
- **Native Gym**: 标准Gym环境的集成
- **Narrow Gap**: 窄缝导航环境
- **UEZoo**: Unreal Engine环境动物园

## 配置系统

该框架使用JSONC（带注释的JSON）文件覆盖Python类属性：

1. **GlobalConfig** (`config.py`) - 控制执行参数:
   - `env_name`: 选择任务/环境
   - `logdir`: 结果输出目录
   - `num_threads`: 并行环境数量
   - `fold`: 进程分组 (num_threads必须能被fold整除)
   - `max_n_episode`: 训练回合限制

2. **任务配置** (`MISSION.{name}.env_wrapper.py->ScenarioConfig`) - 环境特定设置:
   - `AGENT_ID_EACH_TEAM`: 智能体分组 (例如 `[[0, 1], [2, 3]]` 表示2v2)
   - `TEAM_NAMES`: 算法路径列表 (例如 `["ALGORITHM.ppo_ma_lag.foundation->ReinforceAlgorithmFoundation"]`)
   - `MaxEpisodeStep`: 回合长度限制

3. **算法配置** (`ALGORITHM.{name}.foundation.py->AlgorithmConfig`) - 算法超参数

## 执行流程

```
main.py
  ↓ (初始化配置系统)
conf_system.py (加载JSONC，覆盖Python类)
  ↓ (创建并行环境)
env_router.py → make_parallel_envs() → SmartPool (shm_pool.py)
  ↓ (生成带有环境的工作进程)
task_runner.py → Runner.run()
  ↓ (协调训练循环)
MTM (mt_mapper.py) → 路由观察/动作到团队 (MTM = Multi-Team Mapper)
  ↓
算法基础 (例如 ppo_ma_lag/foundation.py)
```

## 关键特性

- **多进程并行**: 使用共享内存（Linux）或管道（Windows）进行高效的进程间通信
- **线程对齐**: 所有并行环境暂停并一起重置以保持同步
- **团队隔离**: MTM确保每个算法只看到其自身智能体的观察
- **Hook**: 通过`_hook_`键进行延迟处理允许异步轨迹收集
- **模块化架构**: 用于轻松添加新环境和算法的插件系统

## 开发模式

### 添加新任务
1. 创建 `MISSION/{name}/env_wrapper.py` 并包含 `ScenarioConfig` 类和 `make_env()` 函数
2. 在 `MISSION/env_router.py` 中注册: 添加到 `import_path_ref` 和 `env_init_function_ref`
3. 实现 `{Name}Wrapper(BaseEnv)` 并包含 `step()` 和 `reset()` 方法
4. 在ScenarioConfig中定义 obs_shape, n_actions, AGENT_ID_EACH_TEAM
5. 创建一个JSONC配置文件将env_name链接到您的任务

### 添加新算法
1. 创建 `ALGORITHM/{name}/foundation.py` 并包含 `AlgorithmConfig` 类
2. 实现 `{Name}AlgorithmFoundation(RLAlgorithmBase)` 并包含 `interact_with_env()` 方法
3. 添加到JSONC配置中的 `TEAM_NAMES` 列表
4. 实现 `action_making()` 进行动作选择和 `train()` 进行学习
5. 使用 `mcv.rec()` 记录指标

## 特别说明

- **配置注入**: JSONC覆盖在导入时通过 `conf_system.py` 发生 - 如果直接修改Python配置类，它们将被覆盖
- **NaN信号**: `actions_list[ENV_PAUSE,:] = np.nan` 信号远程进程跳过步骤并回显先前的观察
- **种子管理**: 创建环境前设置numpy种子，之后设置PyTorch种子
- **日志路径冲突**: 如果logdir存在，系统暂停10秒（以防止覆盖）

---

# 原作者HMP2G简介

本项目基于HMP2G（Hybrid Multi-agent Playground）框架开发，但已大幅简化。原始框架是一个实验性框架，专为强化学习研究人员设计。

原始框架的作者是HMP2G团队，其原始项目地址为：https://github.com/binary-husky/hmp2g

但本项目已移除原框架的大部分功能，仅保留核心的强化学习功能，专注于多智能体强化学习任务。