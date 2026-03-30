# MimicKit DeepMimic vs AMP vs ASE — Experiment Plan

## Overview

8 experiments, 2 batches, 4 GPUs parallel per batch.
RunPod: 4× RTX 4090. Total wall time ~9h. Total cost ~$30.
max_samples=500M per experiment (~3-4h on 4090 with 4096 envs).

---

## Experiment Matrix

```
Batch 1 (4 GPUs, ~4h): Core DM vs AMP comparison
  GPU 0: Exp1   DM × walk              — tracking baseline
  GPU 1: Exp2   DM × spinkick          — tracking on dynamic motion
  GPU 2: Exp3   AMP × walk             — distribution matching baseline
  GPU 3: Exp4   AMP × spinkick         — AMP robustness test

Batch 2 (4 GPUs, ~5h): Multi-skill & task extension
  GPU 0: Exp5a  DM × diverse motions   — DM multi-skill baseline (expected: degrade)
  GPU 1: Exp5b  ASE × diverse motions  — latent-conditioned multi-skill (same dataset)
  GPU 2: Exp6   AMP + steering         — gait switching emergence
  GPU 3: Exp8   AMP + location         — same prior, different task
```

### Narrative Structure

```
Layer 1 (Exp1-4):  DM vs AMP — precision vs robustness tradeoff
Layer 2 (Exp5a/b): Multi-skill — DM degrade vs ASE succeed → latent conditioning value
Layer 3 (Exp6/8):  Task extension — AMP prior reusability across tasks
```

---

## Env Configs

All DeepMimic configs based on official `deepmimic_humanoid_env.yaml`:
- `enable_phase_obs: False`, `tar_obs_steps: [1, 2, 3]`, `joint_err_w` included
- `log_tracking_error: True` enabled for analysis

All AMP configs based on official `amp_humanoid_env.yaml`:
- `global_obs: True`, `log_tracking_error: True`

### Exp1: data/envs/exp1_dm_walk.yaml
Official DM defaults + `motion_file: humanoid_walk.pkl` + `log_tracking_error: True`

### Exp2: data/envs/exp2_dm_spinkick.yaml
Official DM defaults + `motion_file: humanoid_spinkick.pkl` + `log_tracking_error: True`

### Exp3: data/envs/exp3_amp_walk.yaml
Official AMP defaults + `motion_file: humanoid_walk.pkl` + `log_tracking_error: True`

### Exp4: data/envs/exp4_amp_spinkick.yaml
Official AMP defaults + `motion_file: humanoid_spinkick.pkl` + `log_tracking_error: True`

### Exp5a: data/envs/exp5a_dm_diverse.yaml
Official DM defaults + `motion_file: exp5_diverse_motions.yaml` + `log_tracking_error: True`

### Exp5b: data/envs/exp5b_ase_diverse.yaml
Official ASE defaults + `motion_file: exp5_diverse_motions.yaml`

### data/datasets/exp5_diverse_motions.yaml (shared by Exp5a and Exp5b)
```yaml
motions:
  - file: "data/motions/humanoid/humanoid_walk.pkl"
    weight: 1.0
  - file: "data/motions/humanoid/humanoid_spinkick.pkl"
    weight: 1.0
  - file: "data/motions/humanoid/humanoid_dance_a.pkl"
    weight: 1.0
```

### Exp6: data/envs/amp_steering_humanoid_env.yaml (existing, no changes)

### Exp8: data/envs/amp_location_humanoid_env.yaml (existing, no changes)

---

## Agent Configs

```
Exp1, Exp2, Exp5a:  data/agents/deepmimic_humanoid_ppo_agent.yaml  (PPO, SGD 1e-4)
Exp3, Exp4:         data/agents/amp_humanoid_agent.yaml            (AMP, task_w=0.0, disc_w=1.0)
Exp5b:              data/agents/ase_humanoid_agent.yaml            (ASE, Adam 2e-5, latent_dim=64)
Exp6, Exp8:         data/agents/amp_task_humanoid_agent.yaml       (AMP, task_w=0.5, disc_w=0.5)
```

### Exp5a vs Exp5b variable differences (acknowledged)
| Variable | Exp5a (DM) | Exp5b (ASE) |
|----------|-----------|-------------|
| Reward | tracking | disc(0.5) + encoder(0.5) |
| Policy input | obs + tar_obs | obs + latent z |
| Network | 2-layer 1024 | 3-layer 1024 |
| Optimizer | SGD 1e-4 | Adam 2e-5 |
| Extra components | none | discriminator + encoder + diversity loss |

Multiple variables change. Cannot attribute success/failure to a single factor.
The comparison answers: "which METHOD works for multi-skill in MimicKit?"
Isolating individual factors is Future Work.

---

## Training Commands

Common args: `--num_envs 4096 --rand_seed 42 --max_samples 500000000 --logger tb --visualize false`
GPU isolation: `CUDA_VISIBLE_DEVICES=N` + `--devices cuda:0`

### Batch 1 — scripts/run_batch1.sh
```bash
# Exp1: DM walk
CUDA_VISIBLE_DEVICES=0 python mimickit/run.py --mode train --num_envs 4096 \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/exp1_dm_walk.yaml \
  --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
  --devices cuda:0 --visualize false --logger tb --rand_seed 42 \
  --max_samples 500000000 --out_dir output/exp1_dm_walk &

# Exp2: DM spinkick
CUDA_VISIBLE_DEVICES=1 python mimickit/run.py --mode train --num_envs 4096 \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/exp2_dm_spinkick.yaml \
  --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
  --devices cuda:0 --visualize false --logger tb --rand_seed 42 \
  --max_samples 500000000 --out_dir output/exp2_dm_spinkick &

# Exp3: AMP walk
CUDA_VISIBLE_DEVICES=2 python mimickit/run.py --mode train --num_envs 4096 \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/exp3_amp_walk.yaml \
  --agent_config data/agents/amp_humanoid_agent.yaml \
  --devices cuda:0 --visualize false --logger tb --rand_seed 42 \
  --max_samples 500000000 --out_dir output/exp3_amp_walk &

# Exp4: AMP spinkick
CUDA_VISIBLE_DEVICES=3 python mimickit/run.py --mode train --num_envs 4096 \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/exp4_amp_spinkick.yaml \
  --agent_config data/agents/amp_humanoid_agent.yaml \
  --devices cuda:0 --visualize false --logger tb --rand_seed 42 \
  --max_samples 500000000 --out_dir output/exp4_amp_spinkick &

wait
```

### Batch 2 — scripts/run_batch2.sh
```bash
# Exp5a: DM diverse (walk+spinkick+dance)
CUDA_VISIBLE_DEVICES=0 python mimickit/run.py --mode train --num_envs 4096 \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/exp5a_dm_diverse.yaml \
  --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
  --devices cuda:0 --visualize false --logger tb --rand_seed 42 \
  --max_samples 500000000 --out_dir output/exp5a_dm_diverse &

# Exp5b: ASE diverse (same dataset, latent conditioning)
CUDA_VISIBLE_DEVICES=1 python mimickit/run.py --mode train --num_envs 4096 \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/exp5b_ase_diverse.yaml \
  --agent_config data/agents/ase_humanoid_agent.yaml \
  --devices cuda:0 --visualize false --logger tb --rand_seed 42 \
  --max_samples 500000000 --out_dir output/exp5b_ase_diverse &

# Exp6: AMP + steering
CUDA_VISIBLE_DEVICES=2 python mimickit/run.py --mode train --num_envs 4096 \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/amp_steering_humanoid_env.yaml \
  --agent_config data/agents/amp_task_humanoid_agent.yaml \
  --devices cuda:0 --visualize false --logger tb --rand_seed 42 \
  --max_samples 500000000 --out_dir output/exp6_amp_steer &

# Exp8: AMP + location
CUDA_VISIBLE_DEVICES=3 python mimickit/run.py --mode train --num_envs 4096 \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/amp_location_humanoid_env.yaml \
  --agent_config data/agents/amp_task_humanoid_agent.yaml \
  --devices cuda:0 --visualize false --logger tb --rand_seed 42 \
  --max_samples 500000000 --out_dir output/exp8_amp_location &

wait
```

### Test Commands — scripts/run_tests.sh
```bash
# After training, run serial on single GPU:
python mimickit/run.py --mode test --num_envs 4096 --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/exp1_dm_walk.yaml --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
  --devices cuda:0 --visualize false --model_file output/exp1_dm_walk/model.pt --test_episodes 32

# Same pattern for Exp2 (exp2_dm_spinkick), Exp3 (exp3_amp_walk), Exp4 (exp4_amp_spinkick),
# Exp5a (exp5a_dm_diverse + deepmimic agent), Exp5b (exp5b_ase_diverse + ase agent),
# Exp6 (amp_steering_humanoid_env + amp_task agent), Exp8 (amp_location_humanoid_env + amp_task agent)
# Full commands in scripts/run_tests.sh
```

Note: ASE (Exp5b) has `normalizer_samples: 500000000`. With max_samples=500M,
normalizer may consume significant portion of training budget. If ASE doesn't
converge, consider rerunning with max_samples=1000000000 as a follow-up.

---

## Analysis Checklist

### After Batch 1: DM vs AMP

```
tensorboard --logdir=output --port=6006 --bind_all
```

A. Convergence (TensorBoard):
   - [ ] Exp1 vs Exp3 return curves (DM walk vs AMP walk)
   - [ ] Exp2 vs Exp4 return curves (DM spinkick vs AMP spinkick)

B. Tracking error (7 metrics from log_tracking_error):
   - [ ] Walk: DM root_pos_err=___ vs AMP root_pos_err=___ (paper: 0.009 vs 0.132)
   - [ ] Spinkick: DM root_pos_err=___ vs AMP root_pos_err=___ (paper: 0.078 vs 0.064)
   - [ ] AMP Test_Return=0.0 is normal (task_reward_weight=0.0)

C. Visual (--visualize true locally):
   - [ ] DM walk: overlaps reference?
   - [ ] AMP walk: similar style, own rhythm?
   - [ ] AMP spinkick: mode collapse?

### After Batch 2: Multi-skill & Task

D. DM vs ASE multi-skill (Exp5a vs Exp5b):
   - [ ] Exp5a: does DM learn all 3 motions or compromise?
   - [ ] Exp5b: does ASE show distinct modes for different latent z?
   - [ ] Compare tracking errors: Exp5a vs Exp1/2 (single clip)
   - [ ] Visual: sample different z in Exp5b → different motions?

E. AMP steering (Exp6):
   - [ ] Gait emergence: slow target → walk? fast target → run?
   - [ ] Natural direction changes?

F. AMP location (Exp8):
   - [ ] Same motion quality as Exp6? (same prior)
   - [ ] Walk to nearby targets, run to far ones?

---

## Known Limitations

1. No terrain/obstacle experiments (requires code extension).
2. Single seed per experiment. Rerun with --rand_seed 123 if extreme results.
3. Exp5a vs Exp5b has multiple variable changes (method-level comparison, not ablation).
4. ASE may need >500M samples for full convergence.
5. AMP Test_Return=0.0 for Exp3/4 is expected. Use tracking error metrics instead.

---

## Future Work

### Exp7 (deferred): AMP + steering + getup — robustness vs performance tradeoff

Uses `exp7_amp_steer_getup.yaml` (contact_bodies=[], getup weight 3%).
Config and dataset files already created. Deferred because:
- Exp6 vs Exp7 comparison is informative but Exp7 has weak success probability
- Freed GPU slot used for Exp5a (DM diverse) which provides stronger evidence

### Exp9 (planned): DM + task reward — tracking with goal conditioning
Requires code modification to DeepMimicEnv._update_reward(). Tests whether
DM tracking reward conflicts with task reward (expected: yes, because tracking
constrains trajectory while task requires different trajectory).

### Exp10 (planned): DM + multi-clip reward — best-match tracking
Requires code modification. Computes reward against all clips per frame, takes max.
Tests whether frame-level clip selection improves DM multi-skill (expected:
improves over Exp5a but produces unnatural transitions).

---

## Server Execution

```bash
# 1. SSH to RunPod 4×4090
# 2. Upload MimicKit directory
# 3. Setup
bash scripts/setup_server.sh
# 4. Run
conda activate mimickit
bash scripts/run_batch1.sh   # ~4h
bash scripts/run_batch2.sh   # ~5h
bash scripts/run_tests.sh    # ~30min
# 5. TensorBoard
tensorboard --logdir=output --port=6006 --bind_all
```
