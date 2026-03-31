# DeepMimic vs AMP: Motion Imitation Experiment

Based on [MimicKit](https://github.com/xbpeng/MimicKit) by Xue Bin Peng.

Systematically compare two motion imitation paradigms:
- **DeepMimic** — explicit pose tracking via reference motion reward
- **AMP (Adversarial Motion Priors)** — implicit style matching via learned discriminator

## Research Questions

| Question | Experiment |
|----------|-----------|
| DM vs AMP: precision vs robustness tradeoff? | Exp1-4 (walk & spinkick) |
| How critical is pose termination for DM? | Exp-A (ablation) |
| DM vs AMP vs ASE on multi-skill? | Exp5a vs Exp5c vs Exp5b (same dataset) |
| Can AMP prior transfer to downstream tasks? | Exp6 (steering) |

---

## Codebase Architecture

```
mimickit/
├── run.py              # Entry point: args → build env → build agent → train/test
├── engines/            # Simulator backends: IsaacGym, IsaacLab, Newton
├── envs/               # BaseEnv → SimEnv → CharEnv → DeepMimicEnv → AMPEnv → ASEEnv
├── learning/           # PPO, AMP, ASE, AWR, ADD, LCP agents & models
├── anim/               # Motion (.pkl), MotionLib, KinCharModel (URDF/MJCF/USD)
└── util/               # Arg parsing, logging (txt/tb/wandb), torch math, distributed utils
```

**Key Concepts:**
- **Engine**: Simulator backend. Config: `data/engines/*.yaml`
- **Environment**: Task definition. Config: `data/envs/*.yaml`
- **Agent**: RL algorithm + model. Config: `data/agents/*.yaml`
- **Motion**: `.pkl` files with pose frames `[root_pos(3), root_rot(3 exp-map), joint_dofs...]`

**Code Conventions:**
- All configs are YAML. Factory pattern: `env_builder`, `agent_builder`, `engine_builder`
- `env_name` / `agent_name` in YAML maps to class. `output/` and `data/motions/` are gitignored

---

## Experiment Matrix

```
Batch 1 (4 GPUs, ~4h): DM vs AMP 基础对比
  GPU 0: Exp1    DM × walk              — DM baseline
  GPU 1: Exp2    DM × spinkick          — DM 动态动作
  GPU 2: Exp3    AMP × walk             — AMP baseline
  GPU 3: Exp4    AMP × spinkick         — AMP 动态动作

Batch 2 (4 GPUs, ~4h): 消融 + 多技能 + 任务扩展
  GPU 0: Exp-A   DM spinkick 无 pose term — DM 消融: termination 的作用
  GPU 1: Exp5a   DM × diverse           — DM 多技能
  GPU 2: Exp5c   AMP × diverse          — AMP 多技能 (同数据集)
  GPU 3: Exp6    AMP + steering          — AMP 任务扩展

Batch 3 (1 GPU, ~4h): ASE 多技能
  GPU 0: Exp5b   ASE × diverse          — 多模态策略 (latent z) 多技能
```

### Analysis Structure

```
Layer 1 (Exp1-4):      DM vs AMP — 相同动作上的表现差异
Layer 2 (Exp-A):       消融 — pose termination 对 DM 的影响
Layer 3 (Exp5a/5c/5b): 多技能三方对比 — 单高斯+跟踪 vs 单高斯+判别器 vs 多模态+判别器
Layer 4 (Exp6):        任务扩展 — AMP 判别器先验 + 下游任务奖励
```

---

## Experiment Configs

### Env Configs

| Experiment | Config | Key Differences |
|------------|--------|----------------|
| Exp1 | `exp1_dm_walk.yaml` | DM, walk, `pose_termination: True`, `enable_tar_obs: True` |
| Exp2 | `exp2_dm_spinkick.yaml` | Same as Exp1, spinkick |
| Exp3 | `exp3_amp_walk.yaml` | AMP, walk, `pose_termination: False`, `enable_tar_obs: False`, `num_disc_obs_steps: 10` |
| Exp4 | `exp4_amp_spinkick.yaml` | Same as Exp3, spinkick |
| Exp-A | `expa_dm_spinkick_no_pose_term.yaml` | **Ablation**: Exp2 with `pose_termination: False` only |
| Exp5a | `exp5a_dm_diverse.yaml` | DM, diverse dataset (walk+spinkick+dance) |
| Exp5c | `exp5c_amp_diverse.yaml` | AMP, same diverse dataset |
| Exp5b | `exp5b_ase_diverse.yaml` | ASE, same diverse dataset, `latent_dim: 64` |
| Exp6 | `amp_steering_humanoid_env.yaml` | AMP + steering task, locomotion dataset |

### Agent Configs

```
Exp1, Exp2, Exp-A, Exp5a:  deepmimic_humanoid_ppo_agent.yaml  (PPO, SGD 1e-4)
Exp3, Exp4, Exp5c:         amp_humanoid_agent.yaml            (AMP, task_w=0.0, disc_w=1.0)
Exp5b:                     ase_humanoid_agent.yaml            (ASE, Adam 2e-5, latent_dim=64)
Exp6:                      amp_task_humanoid_agent.yaml       (AMP, task_w=0.5, disc_w=0.5)
```

### Key DM vs AMP Differences

| Parameter | DeepMimic | AMP |
|-----------|-----------|-----|
| Reward source | Handcrafted tracking (pose/vel/root/key_pos) | Discriminator output |
| task_reward_weight | N/A (pure tracking) | 0.0 (motion only) or 0.5 (with task) |
| disc_reward_weight | N/A | 1.0 (motion only) or 0.5 (with task) |
| pose_termination | True (fail if >1m from ref) | False |
| enable_tar_obs | True (future ref frames) | False |
| Discriminator | None | fc_2layers_1024units, SGD 2.5e-4 |

---

## Execution

### Step 1: Setup

```bash
# Create conda environment
conda create -n mimickit python=3.8 -y && conda activate mimickit

# Install Isaac Gym
cd IsaacGym_Preview_4_Package/isaacgym/python && pip install -e . && cd -

# Install dependencies
pip install -r requirements.txt && pip install tensorboard

# Download motion data & assets, extract into data/
```

[Motion data & assets download](https://1sfu-my.sharepoint.com/:u:/g/personal/xbpeng_sfu_ca/EclKq9pwdOBAl-17SogfMW0Bved4sodZBQ_5eZCiz9O--w?e=bqXBaa)

Or use the setup script on server: `bash scripts/setup_server.sh`

### Step 2: Training

```bash
conda activate mimickit && mkdir -p output

bash scripts/run_batch1.sh   # Batch 1: 4 GPUs, ~4h
bash scripts/run_batch2.sh   # Batch 2: 4 GPUs, ~4h
bash scripts/run_batch3.sh   # Batch 3: 1 GPU, ~4h

# Or run a single experiment:
python mimickit/run.py --mode train --num_envs 4096 \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/exp1_dm_walk.yaml \
  --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
  --devices cuda:0 --visualize false --logger tb --out_dir output/exp1_dm_walk
```

### Step 3: Monitoring

```bash
tail -f output/exp1_dm_walk.log                        # Real-time log
tensorboard --logdir=output --port=6006 --bind_all     # TensorBoard
```

### Step 4: Testing

```bash
bash scripts/run_tests.sh   # All 9 experiments, serial, ~30min
```

### Step 5: Visualization

```bash
python mimickit/run.py --mode test --num_envs 4 --visualize true \
  --engine_config data/engines/isaac_gym_engine.yaml \
  --env_config data/envs/exp1_dm_walk.yaml \
  --agent_config data/agents/deepmimic_humanoid_ppo_agent.yaml \
  --model_file output/exp1_dm_walk/model.pt
```

---

## Output Data

### Per Experiment

```
output/{exp_name}/
├── model.pt                  # Final trained model
├── log.txt                   # All metrics (space-delimited text)
├── events.out.tfevents.*     # TensorBoard event files
├── engine_config.yaml        # Saved configs
├── env_config.yaml
└── agent_config.yaml
```

### Metrics by Algorithm

| Metric | PPO (DM) | AMP | ASE | TensorBoard Collection |
|--------|----------|-----|-----|----------------------|
| Test_Return | ✓ | ✓ | ✓ | 0_Main |
| Train_Return | ✓ | ✓ | ✓ | 0_Main |
| actor_loss, critic_loss, clip_frac | ✓ | ✓ | ✓ | 3_Loss |
| disc_loss, disc_agent/demo_acc | | ✓ | ✓ | 3_Loss |
| disc_reward_mean | | ✓ | ✓ | 3_Loss |
| enc_loss, diversity_loss | | | ✓ | 3_Loss |
| 7 tracking errors (root_pos/rot, body_pos/rot, dof_vel, root_vel/ang_vel) | ✓ | ✓ | ✓ | 2_Env |

**Notes:**
- AMP `Test_Return=0.0` for Exp3/4/5c is **normal** (task_reward_weight=0.0). Use `disc_reward_mean` instead.
- All experiments have `log_tracking_error: True`.

---

## Analysis Checklist

### Batch 1: DM vs AMP 基础对比

- [x] Exp1 vs Exp3: return curves + tracking error (walk) — DM 追踪精度优 3-284×
- [x] Exp2 vs Exp4: return curves + tracking error (spinkick) — DM 全面领先，AMP Disc_Reward 更高
- [ ] Visual: DM overlaps reference? AMP similar style but own rhythm?

### Batch 2: 消融 + 多技能 + 任务扩展

- [x] **Exp2 vs Exp-A**: DM convergence without pose termination? — 最终性能差 <1%，主要加速早期收敛
- [x] **Exp-A vs Exp4**: Does gap between DM(no term) and AMP narrow? — DM 仍优 14-52×，精度来自奖励设计而非 termination
- [x] **Exp5a (DM) vs Exp5c (AMP) vs Exp5b (ASE)**: Multi-skill 3-way comparison — DM 追踪最优，ASE 编码器收敛但多样性不足
- [x] ASE Diversity_Loss=0.96（未收敛），500M 样本不够 → 需 >1B 样本验证多模态优势
- [x] AMP diverse Disc_Reward=0.991 虚高，判别器决策边界被多动作模糊化
- [x] **Exp6**: AMP 先验成功迁移到 steering 任务，存在 task-style trade-off
- [ ] Exp6 gait emergence visualization (slow→walk, fast→run)

---

## Key Results

> Full analysis with formulas and data tables: [`docs/experiment_analysis.md`](docs/experiment_analysis.md)

| Research Question | Finding |
|-------------------|---------|
| DM vs AMP precision? | DM tracking error 3-284× lower; AMP global drift is by design |
| Pose termination critical? | <1% final performance impact; mainly accelerates early convergence |
| Multi-skill (DM/AMP/ASE)? | DM best tracking; ASE encoder converges but diversity insufficient at 500M samples |
| AMP prior transfer? | Successfully transfers to steering; task-style trade-off observed |

## Known Limitations

1. Single seed (42) per experiment
2. No terrain/obstacle experiments
3. Exp5a vs Exp5c is method-level comparison, not single-variable ablation
4. Only one DM ablation (pose termination)
5. ASE Diversity_Loss=0.96 at 500M samples — needs >1B for full convergence

## Future Work

- DM target obs ablation / AMP disc_obs_steps ablation
- Mixed disc + tracking rewards
- Multi-seed validation
- Terrain experiments

## References

- [MimicKit Starter Guide](https://arxiv.org/abs/2510.13794)
- [DeepMimic](https://arxiv.org/abs/1804.02717) — Peng et al., 2018
- [AMP](https://arxiv.org/abs/2104.02180) — Peng et al., 2021

## License

Apache-2.0 (inherited from MimicKit)
