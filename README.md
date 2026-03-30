# DeepMimic vs AMP: Motion Imitation Experiment

Based on [MimicKit](https://github.com/xbpeng/MimicKit) by Xue Bin Peng.

This project systematically compares two motion imitation paradigms through controlled experiments:

- **DeepMimic** — explicit pose tracking via reference motion reward
- **AMP (Adversarial Motion Priors)** — implicit style matching via learned discriminator

## Research Questions

| Question | Experiment |
|----------|-----------|
| DM vs AMP: precision vs robustness tradeoff? | Exp1-4 (walk & spinkick) |
| How critical is pose termination for DM? | Exp-A (ablation) |
| DM vs AMP vs ASE on multi-skill? | Exp5a vs Exp5c vs Exp5b (same dataset) |
| Can AMP prior transfer to downstream tasks? | Exp6 (steering) |

## Experiment Matrix

```
Batch 1 — DM vs AMP 基础对比 (4 GPUs, ~4h)
  Exp1   DM × walk              — DM baseline
  Exp2   DM × spinkick          — DM dynamic motion
  Exp3   AMP × walk             — AMP baseline
  Exp4   AMP × spinkick         — AMP dynamic motion

Batch 2 — 消融 + 多技能 + 任务扩展 (4 GPUs, ~4h)
  Exp-A  DM spinkick (no pose term) — DM ablation: termination effect
  Exp5a  DM × diverse           — DM multi-skill
  Exp5c  AMP × diverse          — AMP multi-skill (same dataset)
  Exp6   AMP + steering         — AMP task extension

Batch 3 — ASE 多技能 (1 GPU, ~4h)
  Exp5b  ASE × diverse          — multi-modal policy (latent z)
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
