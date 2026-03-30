# DeepMimic vs AMP: Motion Imitation Experiment

Based on [MimicKit](https://github.com/xbpeng/MimicKit) by Xue Bin Peng.

This project systematically compares two motion imitation paradigms through controlled experiments:

- **DeepMimic** — explicit pose tracking via reference motion reward
- **AMP (Adversarial Motion Priors)** — implicit style matching via learned discriminator

## Research Questions

| Question | Experiment |
|----------|-----------|
| DM vs AMP: precision vs robustness tradeoff? | Exp1-4 (walk & spinkick) |
| DM limitation on multi-skill learning? | Exp5a (DM diverse) |
| AMP prior reusability across tasks? | Exp6/8 (steering & location) |
| What role does early termination play in DM? | Ablation (planned) |
| How critical is phase observation for DM? | Ablation (planned) |

## Experiment Matrix

```
Batch 1 — Core DM vs AMP (4 GPUs, ~4h)
  Exp1   DeepMimic × walk         — tracking baseline
  Exp2   DeepMimic × spinkick     — tracking on dynamic motion
  Exp3   AMP × walk               — distribution matching baseline
  Exp4   AMP × spinkick           — AMP robustness test

Batch 2 — Multi-skill & Task Extension (4 GPUs, ~5h)
  Exp5a  DeepMimic × diverse      — DM multi-skill (expected: degrade)
  Exp6   AMP + steering           — gait switching emergence
  Exp8   AMP + location           — same prior, different task
  TBD    Ablation experiments     — isolate key DM/AMP design choices
```

## Quick Start

### Setup

```bash
# 1. Create conda environment
conda create -n mimickit python=3.8 -y && conda activate mimickit

# 2. Install Isaac Gym (from IsaacGym_Preview_4_Package/)
cd IsaacGym_Preview_4_Package/isaacgym/python && pip install -e . && cd -

# 3. Install dependencies
pip install -r requirements.txt
pip install tensorboard

# 4. Download motion data & assets from the link below, extract into data/
```

[Motion data & assets download](https://1sfu-my.sharepoint.com/:u:/g/personal/xbpeng_sfu_ca/EclKq9pwdOBAl-17SogfMW0Bved4sodZBQ_5eZCiz9O--w?e=bqXBaa)

### Training

```bash
# Run all Batch 1 experiments (4 GPUs parallel)
bash scripts/run_batch1.sh

# Run all Batch 2 experiments
bash scripts/run_batch2.sh

# Or run a single experiment
python mimickit/run.py --mode train --num_envs 4096 \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/exp1_dm_walk.yaml \
  --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
  --devices cuda:0 --visualize false --logger tb \
  --out_dir output/exp1_dm_walk
```

### Evaluation

```bash
# Test trained models
bash scripts/run_tests.sh

# Visualize training curves
tensorboard --logdir=output --port=6006

# Visualize agent behavior
python mimickit/run.py --mode test --num_envs 4 --visualize true \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/exp1_dm_walk.yaml \
  --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
  --model_file output/exp1_dm_walk/model.pt
```

## Key Metrics

- **Convergence**: return curves via TensorBoard
- **Tracking error** (7 metrics): root_pos_err, root_rot_err, body_pos_err, body_rot_err, dof_vel_err, root_vel_err, root_ang_vel_err
- **Qualitative**: visual comparison with reference motions

## Project Structure

```
├── mimickit/              # Core framework
│   ├── run.py             # Entry point
│   ├── envs/              # DeepMimic, AMP, ASE, ADD environments
│   ├── learning/          # PPO, AMP, ASE, AWR agents & models
│   ├── engines/           # IsaacGym / IsaacLab / Newton backends
│   └── anim/              # Motion data & character models
├── data/
│   ├── envs/              # Experiment environment configs (exp1-8)
│   ├── agents/            # Algorithm hyperparameter configs
│   └── motions/           # Motion capture data (.pkl)
├── scripts/               # Batch training & evaluation scripts
├── COMPARISION_EXPERIMENT_PLAN.md  # Detailed experiment plan
└── CLAUDE.md              # Codebase guide for AI assistance
```

## References

- [MimicKit Starter Guide](https://arxiv.org/abs/2510.13794)
- [DeepMimic paper](https://arxiv.org/abs/1804.02717) — Peng et al., 2018
- [AMP paper](https://arxiv.org/abs/2104.02180) — Peng et al., 2021

## License

Apache-2.0 (inherited from MimicKit)
