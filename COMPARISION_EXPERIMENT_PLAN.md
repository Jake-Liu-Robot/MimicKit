# DeepMimic vs AMP — Experiment Plan

## Overview

8 experiments, 2 batches, 4 GPUs parallel per batch.
RunPod: 4x RTX 4090. Total wall time ~8h. Total cost ~$25.
max_samples=500M per experiment (~3-4h on 4090 with 4096 envs).

---

## Experiment Matrix

```
Batch 1 (4 GPUs, ~4h): DM vs AMP 基础对比
  GPU 0: Exp1    DM × walk              — DM baseline
  GPU 1: Exp2    DM × spinkick          — DM 动态动作
  GPU 2: Exp3    AMP × walk             — AMP baseline
  GPU 3: Exp4    AMP × spinkick         — AMP 动态动作

Batch 2 (4 GPUs, ~4h): 消融 + 多技能 + 任务扩展
  GPU 0: Exp-A   DM walk 无 pose term   — DM 消融: termination 的作用
  GPU 1: Exp5a   DM × diverse           — DM 多技能
  GPU 2: Exp5c   AMP × diverse          — AMP 多技能 (同数据集)
  GPU 3: Exp6    AMP + steering          — AMP 任务扩展
```

### Analysis Structure

```
Layer 1 (Exp1-4):   DM vs AMP — 相同动作上的表现差异
Layer 2 (Exp-A):    消融 — pose termination 对 DM 的影响
Layer 3 (Exp5a/5c): 多技能 — DM vs AMP 处理多动作数据集的能力
Layer 4 (Exp6):     任务扩展 — AMP 判别器先验 + 下游任务奖励
```

---

## Env Configs

### Exp1: data/envs/exp1_dm_walk.yaml
DeepMimic defaults + `motion_file: humanoid_walk.pkl` + `log_tracking_error: True`
- `pose_termination: True`, `enable_tar_obs: True`, `tar_obs_steps: [1, 2, 3]`

### Exp2: data/envs/exp2_dm_spinkick.yaml
Same as Exp1, `motion_file: humanoid_spinkick.pkl`

### Exp3: data/envs/exp3_amp_walk.yaml
AMP defaults + `motion_file: humanoid_walk.pkl` + `log_tracking_error: True`
- `pose_termination: False`, `enable_tar_obs: False`, `num_disc_obs_steps: 10`

### Exp4: data/envs/exp4_amp_spinkick.yaml
Same as Exp3, `motion_file: humanoid_spinkick.pkl`

### Exp-A: data/envs/expa_dm_walk_no_pose_term.yaml
**Ablation**: Exp1 with `pose_termination: False` (all other params identical)
- Tests whether pose termination is critical for DM convergence

### Exp5a: data/envs/exp5a_dm_diverse.yaml
DeepMimic + `motion_file: exp5_diverse_motions.yaml` (walk + spinkick + dance)

### Exp5c: data/envs/exp5c_amp_diverse.yaml
AMP + `motion_file: exp5_diverse_motions.yaml` (same dataset as Exp5a)
- Clean comparison: same data, different method

### Exp6: data/envs/amp_steering_humanoid_env.yaml (existing)
AMP + steering task + `dataset_humanoid_locomotion.yaml`

### data/datasets/exp5_diverse_motions.yaml (shared by Exp5a and Exp5c)
```yaml
motions:
  - file: "data/motions/humanoid/humanoid_walk.pkl"
    weight: 1.0
  - file: "data/motions/humanoid/humanoid_spinkick.pkl"
    weight: 1.0
  - file: "data/motions/humanoid/humanoid_dance_a.pkl"
    weight: 1.0
```

---

## Agent Configs

```
Exp1, Exp2, Exp-A, Exp5a:  data/agents/deepmimic_humanoid_ppo_agent.yaml  (PPO, SGD 1e-4)
Exp3, Exp4, Exp5c:         data/agents/amp_humanoid_agent.yaml            (AMP, task_w=0.0, disc_w=1.0)
Exp6:                      data/agents/amp_task_humanoid_agent.yaml       (AMP, task_w=0.5, disc_w=0.5)
```

### Key config differences: DM vs AMP

| Parameter | DeepMimic | AMP |
|-----------|-----------|-----|
| Reward source | Handcrafted tracking (pose/vel/root/key_pos) | Discriminator output |
| task_reward_weight | N/A (pure tracking) | 0.0 (motion only) or 0.5 (with task) |
| disc_reward_weight | N/A | 1.0 (motion only) or 0.5 (with task) |
| pose_termination | True (fail if >1m from ref) | False |
| enable_tar_obs | True (future ref frames) | False |
| Discriminator | None | fc_2layers_1024units, SGD 2.5e-4 |
| disc_obs_steps | N/A | 10 frames history |

---

## Training Commands

Common args: `--num_envs 4096 --rand_seed 42 --max_samples 500000000 --logger tb --visualize false`
GPU isolation: `CUDA_VISIBLE_DEVICES=N` + `--devices cuda:0`

See `scripts/run_batch1.sh` and `scripts/run_batch2.sh` for full commands.

---

## Analysis Checklist

### After Batch 1: DM vs AMP 基础对比

```
tensorboard --logdir=output --port=6006 --bind_all
```

A. Convergence (TensorBoard):
   - [ ] Exp1 vs Exp3 return curves (DM walk vs AMP walk)
   - [ ] Exp2 vs Exp4 return curves (DM spinkick vs AMP spinkick)
   - [ ] Note: AMP Test_Return=0.0 is normal (task_reward_weight=0.0)

B. Tracking error (7 metrics from log_tracking_error):
   - [ ] Walk: DM root_pos_err vs AMP root_pos_err
   - [ ] Spinkick: DM root_pos_err vs AMP root_pos_err

C. Visual (--visualize true):
   - [ ] DM walk: overlaps reference motion?
   - [ ] AMP walk: similar style but own rhythm?
   - [ ] DM spinkick vs AMP spinkick: precision vs naturalness?

### After Batch 2: 消融 + 多技能 + 任务扩展

D. Ablation — Exp1 vs Exp-A:
   - [ ] Does DM still converge without pose termination?
   - [ ] Convergence speed difference?
   - [ ] Final tracking error difference?
   - [ ] Compare Exp-A tracking error to Exp3 — does gap narrow?

E. Multi-skill — Exp5a vs Exp5c:
   - [ ] Exp5a: does DM learn all 3 motions or compromise?
   - [ ] Exp5c: does AMP discriminator cover all 3 motion styles?
   - [ ] Compare Exp5a tracking error vs Exp1 (single clip degradation)
   - [ ] Compare Exp5c quality vs Exp3 (single clip degradation)

F. Task extension — Exp6:
   - [ ] Does AMP produce natural locomotion while following steering commands?
   - [ ] Gait emergence: slow target → walk, fast target → run?
   - [ ] Natural direction changes?

---

## Known Limitations

1. Single seed (42) per experiment. Rerun with --rand_seed 123 if extreme results.
2. No terrain/obstacle experiments (requires code extension).
3. Exp5a vs Exp5c changes method + all associated design choices (not a single-variable ablation).
4. Only one DM ablation (pose termination). Target obs ablation deferred.
5. AMP discriminator itself has no ablation (e.g., varying disc_obs_steps).

---

## Future Work

- **DM target obs ablation**: Remove enable_tar_obs to test policy without future reference
- **AMP disc_obs_steps ablation**: Reduce from 10 to 2 to test discriminator history sensitivity
- **AMP + task reward on DM env**: Mix disc and tracking rewards
- **Multi-seed runs**: Validate results with different random seeds
- **Terrain experiments**: Extend environments with obstacles/stairs

---

## Server Execution

```bash
# 1. SSH to RunPod 4x4090
# 2. Upload MimicKit directory
# 3. Setup
bash scripts/setup_server.sh
# 4. Run
conda activate mimickit
bash scripts/run_batch1.sh   # ~4h
bash scripts/run_batch2.sh   # ~4h
bash scripts/run_tests.sh    # ~30min
# 5. TensorBoard
tensorboard --logdir=output --port=6006 --bind_all
```
