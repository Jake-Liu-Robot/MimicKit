# MimicKit - Codebase Guide (for AI assistance)

> See `README.md` for project overview. See `COMPARISION_EXPERIMENT_PLAN.md` for experiment details.

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

- **Engine**: Simulator backend. Config: `data/engines/*.yaml`. Controls physics freq, control mode.
- **Environment**: Task definition. Config: `data/envs/*.yaml`. Defines obs space, reward, termination.
- **Agent**: RL algorithm + model. Config: `data/agents/*.yaml`. Defines network, optimizer, hyperparams.
- **Motion**: `.pkl` files with pose frames `[root_pos(3), root_rot(3 exp-map), joint_dofs...]`.

## Code Conventions

- All configs are YAML. Args can be CLI or `--arg_file`.
- Factory pattern: `env_builder.build_env()`, `agent_builder.build_agent()`, `engine_builder.build_engine()`.
- `env_name` in YAML maps to environment class. `agent_name` maps to agent class.
- Distributed training: `--devices cuda:0 cuda:1` spawns one process per device.
- `output/` is gitignored. Training artifacts go there.
- `data/motions/` and `data/assets/` are gitignored (large binary files).

## Dependencies

- PyTorch >= 1.9.1
- IsaacGym Preview 4 (primary simulator)
- See `requirements.txt` for full list
