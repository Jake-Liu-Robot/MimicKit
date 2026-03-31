# MimicKit 实验数据分析报告

## 目录

1. [实验概览](#1-实验概览)
2. [指标定义与计算公式](#2-指标定义与计算公式)
3. [Layer 1: DM vs AMP 基础对比 (Exp1-4)](#3-layer-1-dm-vs-amp-基础对比)
4. [Layer 2: Pose Termination 消融 (Exp2 vs Exp-A)](#4-layer-2-pose-termination-消融)
5. [Layer 3: 多技能对比 (Exp5a vs Exp5c vs Exp5b)](#5-layer-3-多技能对比)
6. [Layer 4: AMP 任务扩展 (Exp6)](#6-layer-4-amp-任务扩展)
7. [总结与结论](#7-总结与结论)

---

## 1. 实验概览

### 1.1 实验矩阵

| 实验 | 方法 | 动作 | GPU 时间 | 总样本量 | 迭代次数 |
|------|------|------|----------|----------|----------|
| Exp1 | DeepMimic (PPO) | walk | 2.95h | 500M | 3814 |
| Exp2 | DeepMimic (PPO) | spinkick | 3.11h | 500M | 3814 |
| Exp3 | AMP | walk | 3.27h | 500M | 3814 |
| Exp4 | AMP | spinkick | 3.11h | 500M | 3814 |
| Exp-A | DeepMimic (PPO) | spinkick (无 pose term) | 3.61h | 500M | 3814 |
| Exp5a | DeepMimic (PPO) | diverse (walk+spinkick+dance) | 4.32h | 500M | 3814 |
| Exp5c | AMP | diverse (walk+spinkick+dance) | 3.90h | 500M | 3814 |
| Exp6 | AMP + Task | steering | 5.56h | 500M | 3814 |
| Exp5b | ASE | diverse (walk+spinkick+dance) | 训练中 | 500M | - |

### 1.2 训练配置

- **环境数量**: 4096 并行环境
- **随机种子**: 42
- **每次迭代步数**: 32 steps/iter (PPO/AMP), 即每次迭代 131,072 样本
- **测试频率**: 每 100 次迭代（Exp6 每 200 次）
- **测试回合数**: 32 episodes

---

## 2. 指标定义与计算公式

### 2.1 回报与回合指标

| 指标 | 计算方式 | 含义 | 期望趋势 |
|------|----------|------|----------|
| **Test_Return** | 测试模式下每回合累积奖励的均值 | 策略质量的核心指标 | ↑ |
| **Train_Return** | 训练模式下每回合累积奖励的均值 | 带探索噪声的策略质量 | ↑ |
| **Test_Episode_Length** | 测试模式下回合长度均值（帧数） | 角色存活/稳定时长 | ↑（趋近 300 = 10秒） |
| **Train_Episode_Length** | 训练模式下回合长度均值 | 带探索时的存活时长 | ↑ |

> **注意**: AMP 实验（Exp3/4/5c）的 `Test_Return = 0.0` 是正常的，因为 `task_reward_weight = 0.0`，奖励来自判别器而非显式任务奖励。应使用 `Disc_Reward_Mean` 评估 AMP 训练质量。

### 2.2 DeepMimic 追踪奖励

DeepMimic 的奖励是 5 个分项的加权和，每个分项是指数核函数 `exp(-scale * error)`，将误差映射到 [0, 1]：

```
reward = pose_w * exp(-pose_scale * pose_err)
       + vel_w  * exp(-vel_scale  * vel_err)
       + root_pose_w * exp(-root_pose_scale * (root_pos_err + 0.1 * root_rot_err))
       + root_vel_w  * exp(-root_vel_scale  * (root_vel_err + 0.1 * root_ang_vel_err))
       + key_pos_w   * exp(-key_pos_scale   * key_pos_err)
```

**权重配置** (所有 DM 实验共用):

| 分项 | 权重 (w) | 尺度 (scale) |
|------|----------|-------------|
| pose (关节角度) | 0.5 | 0.25 |
| vel (关节速度) | 0.1 | 0.01 |
| root_pose (根位置+朝向) | 0.15 | 5.0 |
| root_vel (根速度) | 0.1 | 1.0 |
| key_pos (末端效应器) | 0.15 | 10.0 |

**误差计算**:
- `pose_err = Σ(joint_err_w × (四元数角度差)²)` — 各关节旋转误差的加权平方和
- `vel_err = Σ(dof_err_w × (目标关节速度 - 当前关节速度)²)` — 关节速度误差
- `root_pos_err = ‖目标根位置 - 当前根位置‖²` — 根位置 L2 距离
- `root_rot_err = (四元数角度差(根朝向, 目标根朝向))²` — 根朝向误差
- `key_pos_err = Σ_关键身体部位 ‖目标关键点位置 - 当前关键点位置‖²` — 末端位置误差（相对根）

### 2.3 追踪误差诊断指标

这些指标在测试模式下记录，是**原始误差**（非指数化），单位为 m 或 rad：

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| **Root_Pos_Err** | `‖目标根位置 - 当前根位置‖` (m) | 全局位置漂移 |
| **Root_Rot_Err** | `|四元数角度差(根朝向)|` (rad) | 全局朝向偏差 |
| **Body_Pos_Err** | `mean(‖目标体位 - 当前体位‖)` (m) | 全身姿态位置误差均值（根相对） |
| **Body_Rot_Err** | `mean(|四元数角度差(体朝向)|)` (rad) | 全身姿态旋转误差均值 |
| **Dof_Vel_Err** | `mean(|目标关节速度 - 当前关节速度|)` (rad/s) | 关节速度跟踪误差 |
| **Root_Vel_Err** | `mean(|目标根速度 - 当前根速度|)` (m/s) | 根线速度误差 |
| **Root_Ang_Vel_Err** | `mean(|目标根角速度 - 当前根角速度|)` (rad/s) | 根角速度误差 |

> DM 实验的追踪误差应显著下降；AMP 实验的追踪误差较大是正常的——AMP 不直接优化追踪，而是学习运动分布的风格匹配。

### 2.4 Pose Termination（姿态终止）

```
当 pose_termination = True 时:
  if max(‖目标体位 - 当前体位‖²) > pose_termination_dist² (默认 1.0m):
    回合终止（FAIL）
```

另外，所有实验共享的 **跌倒检测**：非接触体（非脚部）有地面接触力 > 0.1 时终止。

### 2.5 PPO 优化指标

| 指标 | 计算公式 | 含义 | 期望趋势 |
|------|----------|------|----------|
| **Actor_Loss** | `-mean(min(adv × ratio, adv × clip(ratio)))` | PPO 裁剪代理目标（取负做最小化） | 稳定在 0 附近 |
| **Critic_Loss** | `mean((TD(λ)回报 - V(obs))²)` | 值函数预测误差 | ↓ |
| **Clip_Frac** | `mean(|ratio - 1| > clip_ratio)` | 被裁剪的样本比例 | 稳定 (0.05-0.3) |
| **Imp_Ratio** | `mean(exp(new_logprob - old_logprob))` | 重要性采样比率 | ≈ 1.0 |
| **Adv_Mean** | `mean(TD(λ)回报 - V(obs))` | 平均优势值 | ≈ 0 |
| **Adv_Std** | `std(优势值)` | 优势值分散度 | ↓ |
| **Action_Bound_Loss** | 动作超出边界的惩罚 | 动作空间约束 | ≈ 0 |

**TD(λ) 回报计算**:
```
return[i] = r[i] + γ × ((1 - λ) × V(next) + λ × return[i+1])
```
其中 `γ = 0.99`, `λ = 0.95`, 在回合边界处重置为 bootstrap 值。

**优势标准化**: `norm_adv = (adv - adv_mean) / max(adv_std, 1e-5)`，裁剪到 `[-4, 4]`。

### 2.6 AMP 判别器指标

| 指标 | 计算公式 | 含义 | 期望趋势 |
|------|----------|------|----------|
| **Disc_Loss** | `0.5 × (BCE(agent_logit, 0) + BCE(demo_logit, 1)) + grad_penalty + logit_reg` | 判别器总损失 | 达到均衡 |
| **Disc_Agent_Acc** | `mean(agent_logit < 0)` | 判别器正确识别"假"的比例 | ↓（agent 逐渐骗过判别器） |
| **Disc_Demo_Acc** | `mean(demo_logit > 0)` | 判别器正确识别"真"的比例 | 保持高位 (≈1.0) |
| **Disc_Agent_Logit** | `mean(discriminator(agent_obs))` | agent 观测的判别器原始输出 | 从负→趋近 0 |
| **Disc_Demo_Logit** | `mean(discriminator(demo_obs))` | demo 观测的判别器原始输出 | 保持正值 |
| **Disc_Grad_Penalty** | `0.5 × (mean(‖∇_demo‖²) + mean(‖∇_agent‖²))` | 判别器梯度惩罚 (WGAN-GP 式) | 保持较低 |
| **Disc_Logit_Loss** | `Σ(logit层权重²)` | 判别器输出层 L2 正则 | 保持较低 |

**判别器奖励计算**:
```
prob = sigmoid(disc_logit)
disc_r = -log(max(1 - prob, 0.0001)) × disc_reward_scale
total_reward = task_reward_w × task_r + disc_reward_w × disc_r
```

| 指标 | 计算公式 | 含义 | 期望趋势 |
|------|----------|------|----------|
| **Disc_Reward_Mean** | `mean(disc_r)` | 判别器给出的风格奖励均值 | ↑ |
| **Disc_Reward_Std** | `std(disc_r)` | 判别器奖励的分散度 | ↓ |

### 2.7 Steering 任务奖励（Exp6 特有）

```
tar_vel = tar_speed × tar_dir         # 目标速度向量（2D）
tar_vel_err = ‖tar_vel - root_vel_xy‖²
tar_reward = exp(-vel_err_scale × tar_vel_err)   # 速度匹配奖励
  # 若沿目标方向的投影速度为负（反方向走），tar_reward = 0

face_err = dot(目标朝向, 角色朝向)
face_reward = clamp(face_err, min=0)               # 朝向匹配奖励

steering_reward = 0.7 × tar_reward + 0.3 × face_reward
total_reward = 0.5 × steering_reward + 0.5 × disc_reward
```

### 2.8 ASE 特有指标（Exp5b，训练中）

| 指标 | 计算公式 | 含义 | 期望趋势 |
|------|----------|------|----------|
| **Enc_Loss** | `mean(-Σ(z_target × enc_pred))` | 编码器负余弦相似度 | ↓（趋近 -1） |
| **Enc_Reward_Mean** | `mean(clamp(dot(z, enc_pred), min=0))` | 编码器奖励 | ↑ |
| **Diversity_Loss** | `(diversity_tar - action_diff/z_diff)²` | 动作多样性约束 | ↓（趋近 0） |

**ASE 总奖励**: `r = 0.5 × disc_r + 0.5 × enc_r`（task_w = 0.0）

### 2.9 观测标准化指标

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| **Obs_Norm_Mean** | `mean(|running_obs_mean|)` | 观测均值的绝对值均值 |
| **Obs_Norm_Std** | `mean(running_obs_std)` | 观测标准差均值 |

---

## 3. Layer 1: DM vs AMP 基础对比

### 3.1 核心对比：Exp1 (DM walk) vs Exp3 (AMP walk)

#### 回报曲线

| 迭代 | 样本(M) | Exp1 Test_Return | Exp3 Test_Return | Exp3 Disc_Reward |
|------|---------|-------------------|-------------------|------------------|
| 0 | 0.1 | 8.04 | 0.00 | 0.050 |
| 200 | 26.3 | 159.44 | 0.00 | 0.076 |
| 500 | 65.7 | 266.01 | 0.00 | 0.094 |
| 1000 | 131.2 | 289.61 | 0.00 | 0.118 |
| 2000 | 262.3 | 293.96 | 0.00 | 0.178 |
| 3000 | 393.3 | 294.81 | 0.00 | 0.247 |
| 3814 | 500.0 | 294.02 | 0.00 | 0.308 |

> Exp3 (AMP) 的 Test_Return 始终为 0.0，因为 `task_reward_weight = 0.0`。AMP 的训练质量需看 Disc_Reward_Mean。

#### 追踪误差对比（最终值）

| 指标 | Exp1 (DM walk) | Exp3 (AMP walk) | DM 更优倍数 |
|------|-----------------|------------------|-------------|
| Root_Pos_Err (m) | **0.0235** | 6.6755 | 284× |
| Root_Rot_Err (rad) | **0.0251** | 0.0873 | 3.5× |
| Body_Pos_Err (m) | **0.0084** | 0.0945 | 11× |
| Body_Rot_Err (rad) | **0.0434** | 0.3416 | 7.9× |
| Dof_Vel_Err (rad/s) | **0.2883** | 1.1304 | 3.9× |
| Root_Vel_Err (m/s) | **0.0467** | 0.2000 | 4.3× |
| Root_Ang_Vel_Err (rad/s) | **0.0851** | 0.4128 | 4.9× |

**关键发现**:
- DM 在所有追踪指标上大幅优于 AMP，尤其是 **Root_Pos_Err**（284 倍差距）
- AMP 的 Root_Pos_Err = 6.68m 说明 agent 学会了 walk 的风格但不跟踪参考轨迹的全局位置，这是预期行为
- AMP 的 Body_Pos_Err (0.0945m) 和 Body_Rot_Err (0.3416 rad) 相对合理，说明局部姿态质量可接受
- DM 的 Body_Pos_Err 降到 0.0084m（约 8mm），几乎完美复现参考动作

#### 回合长度

两个实验最终的 Test_Episode_Length 都接近 298（满 300 帧 = 10 秒），说明 agent 都能稳定站立不跌倒。

### 3.2 核心对比：Exp2 (DM spinkick) vs Exp4 (AMP spinkick)

#### 追踪误差对比（最终值）

| 指标 | Exp2 (DM spinkick) | Exp4 (AMP spinkick) | DM 更优倍数 |
|------|---------------------|----------------------|-------------|
| Root_Pos_Err (m) | **0.0374** | 2.0458 | 55× |
| Root_Rot_Err (rad) | **0.0465** | 1.5841 | 34× |
| Body_Pos_Err (m) | **0.0246** | 0.3565 | 14× |
| Body_Rot_Err (rad) | **0.1139** | 1.6335 | 14× |
| Dof_Vel_Err (rad/s) | **1.0203** | 3.6600 | 3.6× |
| Root_Vel_Err (m/s) | **0.1357** | 0.6834 | 5.0× |
| Root_Ang_Vel_Err (rad/s) | **0.2612** | 1.6136 | 6.2× |

**关键发现**:
- spinkick 是更具动态性的动作（旋踢），两种方法的追踪误差都比 walk 高
- DM 仍然在所有指标上显著优于 AMP
- AMP spinkick 的 Root_Rot_Err = 1.58 rad（约 90°），说明 AMP agent 的旋转与参考差异很大，但可能学到了类似风格的踢腿动作
- Exp4 的 Disc_Reward_Mean 最终达到 0.928，远高于 Exp3 的 0.308，这表明 spinkick 的判别器更容易被 agent "骗过"

#### AMP 判别器对比

| 指标 | Exp3 (AMP walk) | Exp4 (AMP spinkick) |
|------|-----------------|---------------------|
| Disc_Reward_Mean | 0.308 | **0.928** |
| Disc_Agent_Acc | 0.996 | 0.994 |
| Disc_Demo_Acc | 1.000 | 1.000 |
| Disc_Loss | 0.280 | 0.558 |

Exp4 的判别器奖励显著更高。可能原因：spinkick 的运动模式更独特/简单，agent 更容易模仿出判别器认可的风格。

### 3.3 Layer 1 小结

| 维度 | DeepMimic 优势 | AMP 优势 |
|------|---------------|---------|
| **追踪精度** | 所有指标优 3-284 倍 | — |
| **全局位置跟踪** | 精确跟踪参考轨迹 | 不跟踪（Root_Pos_Err 大），但这是设计如此 |
| **鲁棒性** | 依赖精确参考，偏差过大会终止 | 无 pose termination，自由探索 |
| **局部姿态** | 接近完美 | 质量可接受（Body_Pos_Err < 0.1m for walk） |
| **适用场景** | 精确复现特定动作片段 | 学习运动风格，可组合任务奖励 |

---

## 4. Layer 2: Pose Termination 消融

### 4.1 对比：Exp2 vs Exp-A（DM spinkick ± pose termination）

| 指标 | Exp2 (有 pose term) | Exp-A (无 pose term) | 差异 |
|------|---------------------|----------------------|------|
| Test_Return | **279.85** | 278.50 | -0.5% |
| Root_Pos_Err | **0.0374** | 0.0396 | +5.9% |
| Root_Rot_Err | **0.0465** | 0.0539 | +15.9% |
| Body_Pos_Err | **0.0246** | 0.0258 | +4.9% |
| Body_Rot_Err | **0.1139** | 0.1170 | +2.7% |
| Dof_Vel_Err | **1.0203** | 1.0873 | +6.6% |
| Root_Vel_Err | **0.1357** | 0.1465 | +7.9% |
| Root_Ang_Vel_Err | **0.2612** | 0.2592 | -0.8% |
| Wall_Time (h) | 3.11 | 3.61 | +16% |

### 4.2 收敛过程对比

| 迭代 | Exp2 Body_Pos_Err | Exp-A Body_Pos_Err | Exp2 Test_Return | Exp-A Test_Return |
|------|--------------------|--------------------|------------------|-------------------|
| 0 | 0.198 | 0.294 | 3.46 | 4.45 |
| 500 | 0.054 | 0.087 | 243.42 | 212.80 |
| 1000 | 0.032 | 0.044 | 274.18 | 263.64 |
| 2000 | 0.025 | 0.029 | 279.00 | 277.65 |
| 3000 | 0.023 | 0.026 | 281.53 | 278.24 |
| 3814 | 0.025 | 0.026 | 279.85 | 278.50 |

### 4.3 Layer 2 小结

**关键发现**:
1. **移除 pose termination 对最终性能影响很小**（Test_Return 差距 < 1%，追踪误差差距 < 16%）
2. **收敛速度有差异**：无 pose term 的版本在前 1000 次迭代收敛更慢（Body_Pos_Err 在 iter 500 时高 61%），但到 iter 2000 基本追上
3. **没有出现"lazy solution"**：移除 pose termination 后 agent 没有学会躺在地上不动（episode length 同样接近 300），说明跌倒检测（contact-based early termination）已经足够约束行为
4. **训练时间增加 16%**：可能因为无 pose termination 导致更多无效探索

**结论**: Pose termination 的主要价值是**加速收敛**，而非防止策略崩溃。在 500M 样本预算下，两者最终效果接近。

### 4.4 Exp-A vs Exp4（DM 无 term vs AMP）

| 指标 | Exp-A (DM 无 term) | Exp4 (AMP) | 差距 |
|------|---------------------|------------|------|
| Root_Pos_Err | **0.0396** | 2.0458 | 52× |
| Body_Pos_Err | **0.0258** | 0.3565 | 14× |
| Body_Rot_Err | **0.1170** | 1.6335 | 14× |
| Dof_Vel_Err | **1.0873** | 3.6600 | 3.4× |

即使移除 pose termination，DM 的追踪误差仍远小于 AMP。**DM 的精度优势来自显式追踪奖励设计，而非 pose termination 机制**。

---

## 5. Layer 3: 多技能对比

### 5.1 Exp5a (DM diverse) vs Exp5c (AMP diverse)

两个实验使用相同的 diverse 数据集（walk + spinkick + dance_a）。

#### 追踪误差对比（最终值）

| 指标 | Exp5a (DM diverse) | Exp5c (AMP diverse) | DM 更优倍数 |
|------|---------------------|----------------------|-------------|
| Root_Pos_Err | **0.0512** | 4.1456 | 81× |
| Root_Rot_Err | **0.0702** | 1.0045 | 14× |
| Body_Pos_Err | **0.0258** | 0.2495 | 9.7× |
| Body_Rot_Err | **0.1183** | 1.1733 | 9.9× |
| Dof_Vel_Err | **0.8572** | 2.4933 | 2.9× |
| Root_Vel_Err | **0.1374** | 0.4704 | 3.4× |
| Root_Ang_Vel_Err | **0.2622** | 1.3711 | 5.2× |

#### 单技能 vs 多技能（DM 内部对比）

| 指标 | Exp1 (DM walk) | Exp2 (DM spinkick) | Exp5a (DM diverse) |
|------|----------------|---------------------|---------------------|
| Test_Return | **294.02** | 279.85 | 277.93 |
| Body_Pos_Err | **0.0084** | 0.0246 | 0.0258 |
| Body_Rot_Err | **0.0434** | 0.1139 | 0.1183 |
| Dof_Vel_Err | **0.2883** | 1.0203 | 0.8572 |

**发现**: DM diverse 的 Body_Pos_Err 和 Body_Rot_Err 与 DM spinkick 相当，说明多技能场景下 DM 的追踪质量主要受限于最难动作（spinkick），walk 部分的精度被拖低。

#### 单技能 vs 多技能（AMP 内部对比）

| 指标 | Exp3 (AMP walk) | Exp4 (AMP spinkick) | Exp5c (AMP diverse) |
|------|----------------|---------------------|---------------------|
| Root_Pos_Err | 6.6755 | **2.0458** | 4.1456 |
| Body_Pos_Err | **0.0945** | 0.3565 | 0.2495 |
| Disc_Reward_Mean | 0.308 | **0.928** | 0.991 |

**发现**: AMP diverse 的 Disc_Reward_Mean (0.991) 比单动作实验都高，但这可能因为判别器在多动作数据下更容易被"骗"。Body_Pos_Err 介于 walk 和 spinkick 之间，符合预期。

### 5.2 Exp5b (ASE diverse) — 已完成

ASE 使用 latent-conditioned policy（z ∈ R⁶⁴），旨在让不同 latent 对应不同技能，解决单高斯策略的多模态问题。

**ASE 总奖励**: `r = 0.5 × disc_r + 0.5 × enc_r`（task_w = 0.0）

训练时长: 3.06h, 500M samples, 3814 iterations。

#### ASE 特有指标收敛曲线

| 迭代 | 样本(M) | Episode_Len | Disc_Reward | Enc_Reward | Enc_Loss | Diversity_Loss | Disc_Demo_Acc |
|------|---------|-------------|-------------|------------|----------|----------------|---------------|
| 0 | 0.1 | 24.5 | 1.683 | 0.049 | -0.014 | 1.000 | 0.984 |
| 200 | 26.3 | 54.0 | 0.317 | 0.306 | -0.310 | 0.997 | 1.000 |
| 500 | 65.7 | 269.4* | 0.427 | 0.397 | -0.399 | 0.995 | 1.000 |
| 1000 | 131.2 | 285.0 | 0.483 | 0.523 | -0.526 | 0.989 | 0.990 |
| 2000 | 262.3 | 296.6 | 0.419 | 0.639 | -0.641 | 0.977 | 0.902 |
| 3000 | 393.3 | 298.6 | 0.419 | 0.703 | -0.705 | 0.966 | 0.878 |
| 3814 | 500.0 | 299.0 | 0.389 | 0.739 | -0.740 | 0.960 | 0.790 |

*iter 400 近似

**关键观察**:
1. **Enc_Reward 持续上升**（0.049 → 0.739）：编码器越来越能从运动中识别出对应的 latent code，说明不同 latent 确实驱动了不同行为
2. **Disc_Reward 先升后降再稳定**（1.68 → 0.39）：初始值虚高（随机策略碰巧匹配），训练后稳定在 ~0.4
3. **Diversity_Loss 缓慢下降**（1.00 → 0.96）：但仍然很高（理想值为 0），说明动作多样性还不充分，500M 样本可能不够 ASE 完全收敛
4. **Disc_Demo_Acc 持续下降**（0.984 → 0.790）：判别器对 demo 数据的识别能力下降，说明判别器在 ASE 的多目标优化下被削弱
5. **Episode_Length 收敛到 299**：agent 稳定站立不跌倒

### 5.3 三方对比：Exp5a (DM) vs Exp5c (AMP) vs Exp5b (ASE)

#### 追踪误差对比（最终值）

| 指标 | Exp5a (DM) | Exp5c (AMP) | Exp5b (ASE) | 最优 |
|------|------------|-------------|-------------|------|
| Root_Pos_Err (m) | **0.0512** | 4.1456 | 5.4604 | DM |
| Root_Rot_Err (rad) | **0.0702** | 1.0045 | 1.2516 | DM |
| Body_Pos_Err (m) | **0.0258** | 0.2495 | 0.2815 | DM |
| Body_Rot_Err (rad) | **0.1183** | 1.1733 | 1.4416 | DM |
| Dof_Vel_Err (rad/s) | **0.8572** | 2.4933 | 2.0957 | DM |
| Root_Vel_Err (m/s) | **0.1374** | 0.4704 | 0.5758 | DM |
| Root_Ang_Vel_Err (rad/s) | **0.2622** | 1.3711 | 1.7314 | DM |
| Test_Episode_Length | 297.8 | 298.5 | **299.0** | ASE |

#### 判别器/编码器指标对比

| 指标 | Exp5c (AMP) | Exp5b (ASE) |
|------|-------------|-------------|
| Disc_Reward_Mean | **0.991** | 0.389 |
| Disc_Agent_Acc | 0.994 | **0.986** |
| Disc_Demo_Acc | **1.000** | 0.790 |
| Enc_Reward_Mean | — | 0.739 |
| Diversity_Loss | — | 0.960 |

#### DM 内部多技能退化分析

| 指标 | Exp1 (walk) | Exp2 (spinkick) | Exp5a (diverse) | diverse 退化率 |
|------|-------------|-----------------|-----------------|---------------|
| Test_Return | 294.02 | 279.85 | 277.93 | -5.5% (vs walk) |
| Body_Pos_Err | 0.0084 | 0.0246 | 0.0258 | +207% (vs walk) |
| Body_Rot_Err | 0.0434 | 0.1139 | 0.1183 | +173% (vs walk) |

#### AMP 内部多技能对比

| 指标 | Exp3 (walk) | Exp4 (spinkick) | Exp5c (diverse) |
|------|-------------|-----------------|-----------------|
| Body_Pos_Err | **0.0945** | 0.3565 | 0.2495 |
| Body_Rot_Err | **0.3416** | 1.6335 | 1.1733 |
| Disc_Reward_Mean | 0.308 | 0.928 | **0.991** |

### 5.4 Layer 3 深度分析

#### 发现 1: DM 在追踪指标上仍然全面领先

DM diverse 的追踪误差比 AMP 和 ASE 低一个数量级。这是方法论差异：DM 直接优化追踪，AMP/ASE 优化分布匹配。

#### 发现 2: ASE 的追踪误差反而高于 AMP

| 指标 | AMP diverse | ASE diverse | ASE vs AMP |
|------|-------------|-------------|------------|
| Body_Pos_Err | **0.2495** | 0.2815 | +12.8% |
| Body_Rot_Err | **1.1733** | 1.4416 | +22.9% |
| Dof_Vel_Err | **2.4933** | 2.0957 | -15.9% |

ASE 在大多数追踪指标上**比 AMP 更差**。这可能因为：
- ASE 的优化目标更复杂（disc + enc + diversity），资源被分散
- 500M 样本对 ASE 不够（`normalizer_samples = 500M`，刚好触及上限）
- Diversity_Loss 仍为 0.96（远未收敛到 0），说明 latent 空间还没充分分化

#### 发现 3: ASE 的 Disc_Demo_Acc 异常下降

ASE 的 Disc_Demo_Acc 从 0.984 下降到 0.790，意味着判别器把约 21% 的真实 demo 误分类为假。这在 AMP 中没有出现（始终 ≈ 1.0）。

**可能原因**: ASE 的编码器奖励会驱动 agent 产生"编码器可识别但不一定自然"的动作，这些动作可能干扰判别器的判断边界。

#### 发现 4: ASE 编码器收敛良好但多样性不足

- Enc_Reward = 0.739（满分 1.0）：编码器能较好地从运动中识别 latent
- Diversity_Loss = 0.960（理想值 0.0）：不同 latent 产生的动作差异还不够大
- 这表明 ASE 处于**"能区分但不够多样"**的中间状态，需要更多训练样本

#### 发现 5: AMP diverse 判别器奖励虚高

Exp5c 的 Disc_Reward_Mean = 0.991，高于单动作的 Exp3 (0.308) 和 Exp4 (0.928)。但这不意味着 AMP diverse 学得更好——多动作混合可能让判别器的决策边界更模糊，agent 更容易"骗过"判别器。Disc_Agent_Acc = 0.994 仍然很高（判别器仍能分辨），进一步佐证判别器奖励的虚高。

### 5.5 Layer 3 小结

| 维度 | DM (Exp5a) | AMP (Exp5c) | ASE (Exp5b) |
|------|------------|-------------|-------------|
| **追踪精度** | **最优** (Body_Pos 0.026m) | 中等 (0.250m) | 最差 (0.282m) |
| **稳定性** | 好 (ep_len 297.8) | 好 (298.5) | **最好** (299.0) |
| **风格质量** | 精确复现但单模态 | 风格匹配 (disc_r 0.99) | 风格+编码 (enc_r 0.74) |
| **多技能能力** | 平均化，受限于最难动作 | 单高斯，无法显式切换 | latent 可切换，但未充分收敛 |
| **瓶颈** | 多模态坍缩到平均策略 | 无显式技能区分机制 | 需要 >500M 样本 |

**核心结论**: 在 500M 样本预算下，**ASE 的多技能优势尚未体现**。Diversity_Loss 高表明 latent 空间分化不足。这验证了 README 中的预判："ASE may need >500M samples for full convergence"。若预算充足（如 1B+ 样本），ASE 的 latent-conditioned policy 有望展现多模态优势。

---

## 6. Layer 4: AMP 任务扩展 (Exp6)

### 6.1 Exp6: AMP + Steering

Exp6 是唯一同时使用判别器奖励和任务奖励的实验：
- `task_reward_weight = 0.5`（steering 任务）
- `disc_reward_weight = 0.5`（运动风格）

#### 回报与判别器指标

| 迭代 | Test_Return | Disc_Reward_Mean | Disc_Agent_Acc | Disc_Demo_Acc |
|------|-------------|------------------|----------------|---------------|
| 0 | 2.25 | 0.042 | 0.989 | 0.992 |
| 1000 | 118.04 | 0.446 | 0.971 | 0.999 |
| 2000 | 165.05 | 0.683 | 0.980 | 0.966 |
| 3000 | 172.65 | 0.770 | 0.993 | 0.927 |
| 3814 | 172.91 | 0.782 | 0.992 | 0.957 |

**关键发现**:
1. **Test_Return = 172.91**：非零且持续上升，说明 agent 学会了 steering 任务
2. **Disc_Reward_Mean = 0.782**：低于纯 AMP 实验（Exp3: 0.308, Exp4: 0.928），说明任务奖励和风格奖励之间存在 trade-off
3. **Disc_Demo_Acc 下降到 0.957**：在其他 AMP 实验中始终为 1.0，这里下降说明判别器在双目标优化下受到影响
4. **Disc_Agent_Acc = 0.992 仍然很高**：判别器仍能区分 agent 和 demo，说明 agent 的运动风格还有提升空间

#### 追踪误差

| 指标 | Exp6 (AMP steer) | Exp3 (AMP walk) | 对比 |
|------|-------------------|-----------------|------|
| Root_Pos_Err | 6.9151 | 6.6755 | +3.6% |
| Body_Pos_Err | 0.2362 | **0.0945** | 2.5× worse |
| Body_Rot_Err | 1.5215 | **0.3416** | 4.5× worse |
| Dof_Vel_Err | 1.5678 | **1.1304** | 1.4× worse |

Exp6 的追踪误差高于纯 AMP walk，这符合预期——steering 要求 agent 跟随随机方向/速度指令，无法完全复现参考运动。

### 6.2 Layer 4 小结

- **AMP 判别器先验成功迁移到下游 steering 任务**：agent 学会了用自然步态执行方向控制
- **存在 task-style trade-off**：任务奖励和风格奖励之间有张力，表现为追踪误差增大和判别器准确度下降
- **未观察到步态涌现**（慢速→走，快速→跑）：需要可视化验证，仅从数据无法判断

---

## 7. 总结与结论

### 7.1 核心发现

| 研究问题 | 结论 |
|----------|------|
| **DM vs AMP: 精度 vs 鲁棒性?** | DM 在追踪精度上优 3-284×，但 AMP 不追踪全局位置是设计选择。两者在回合存活率上相当（都接近 100%） |
| **Pose termination 多关键?** | 影响收敛速度（前 1000 迭代差 61%），但不影响最终性能（差 <1%）。非防止策略崩溃的必要条件 |
| **DM vs AMP vs ASE 多技能?** | DM 追踪最优但多模态坍缩；AMP 判别器奖励虚高；ASE 编码器收敛但多样性不足（需 >500M 样本） |
| **AMP 先验迁移?** | 成功迁移到 steering 任务，但存在 task-style trade-off |

### 7.2 方法选型建议

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| 精确复现单个动作片段 | DeepMimic | 追踪误差小 1-2 个数量级 |
| 学习运动风格 + 组合任务 | AMP | 判别器奖励可与任意任务奖励叠加 |
| 多技能覆盖 | ASE（需充足训练） | latent 编码器可区分技能，但 500M 样本下 Diversity 未收敛 |

### 7.3 局限性

1. 单种子 (seed=42)，结论的统计显著性有限
2. 未进行地形/障碍实验
3. AMP 的 tracking error 不是公平比较维度——AMP 设计上不追踪参考轨迹
4. Exp5a vs Exp5c 是方法级对比，非单变量消融
5. ASE 在 500M 样本下未完全收敛（Diversity_Loss = 0.96），结论可能因训练不足而偏保守

### 7.4 待补充分析

- [ ] 可视化对比：DM 是否精确叠合参考？AMP 是否风格相似但节奏自由？
- [ ] Exp6 步态涌现分析（需可视化不同 tar_speed 下的行为）
- [ ] ASE 更长训练（>1B samples）验证多模态能力
