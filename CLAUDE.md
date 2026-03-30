# MimicKit - Motion Imitation RL Framework

## Project Overview

Fork of [xbpeng/MimicKit](https://github.com/xbpeng/MimicKit) for studying and comparing motion imitation methods — primarily **DeepMimic** and **AMP**.

## Architecture

```
mimickit/
├── run.py              # Entry point: args → build env → build agent → train/test
├── engines/            # Simulator backends: IsaacGym, IsaacLab, Newton
├── envs/               # Environment hierarchy:
│   │                   #   BaseEnv → SimEnv → CharEnv → DeepMimicEnv → AMPEnv → ASEEnv
│   ├── deepmimic_env   # Pose tracking via reference motion (phase + target obs)
│   ├── amp_env         # Adversarial motion prior (discriminator obs, no pose termination)
│   ├── ase_env         # Adversarial skill embeddings (latent z conditioning)
│   ├── add_env         # Adversarial differential discriminator
│   └── task_*_env      # Downstream tasks (steering, location) with AMP prior
├── learning/           # RL algorithms:
│   ├── ppo_agent       # Proximal Policy Optimization (used by DeepMimic)
│   ├── amp_agent       # AMP: PPO + discriminator training
│   ├── ase_agent       # ASE: AMP + encoder + diversity loss
│   ├── awr_agent       # Advantage-Weighted Regression
│   ├── add_agent       # ADD: AMP with differential discriminator
│   └── lcp_agent       # Lipschitz-Constrained Policy (PPO wrapper)
├── anim/               # Motion data: Motion (.pkl), MotionLib, KinCharModel (URDF/MJCF/USD)
└── util/               # Arg parsing, logging (txt/tb/wandb), torch math, distributed utils
```

## Key Concepts

- **Engine**: Simulator backend abstraction. Configured via `data/engines/*.yaml`. Controls physics frequency, control mode, etc.
- **Environment**: Task definition. Configured via `data/envs/*.yaml`. Defines observation space, reward, termination.
- **Agent**: RL algorithm + model. Configured via `data/agents/*.yaml`. Defines network architecture, optimizer, hyperparameters.
- **Motion**: `.pkl` files with pose frames `[root_pos(3), root_rot(3 exp-map), joint_dofs...]`. Managed by MotionLib.

## Training Command Pattern

```bash
python mimickit/run.py \
  --mode train \
  --num_envs 4096 \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/<env>.yaml \
  --agent_config data/agents/<agent>.yaml \
  --devices cuda:0 --visualize false --logger tb \
  --out_dir output/<experiment_name>
```

Or use arg files: `python mimickit/run.py --arg_file args/<args>.txt`

## Experiment Plan

See `COMPARISION_EXPERIMENT_PLAN.md` for the full 8-experiment comparison:
- **Batch 1** (Exp1-4): DeepMimic vs AMP on walk/spinkick
- **Batch 2** (Exp5-8): Multi-skill (DM vs ASE), task extension (steering, location)
- Scripts: `scripts/run_batch1.sh`, `scripts/run_batch2.sh`, `scripts/run_tests.sh`

## Code Conventions

- All configs are YAML. Args can be CLI or `--arg_file`.
- Factory pattern everywhere: `env_builder.build_env()`, `agent_builder.build_agent()`, `engine_builder.build_engine()`.
- Environment class names map to `env_name` field in YAML configs.
- Agent class names map to `agent_name` field in YAML configs.
- Distributed training: `--devices cuda:0 cuda:1` spawns one process per device.
- `output/` is gitignored. Training artifacts go there.

## Dependencies

- PyTorch >= 1.9.1
- IsaacGym Preview 4 (primary simulator)
- See `requirements.txt` for full list
