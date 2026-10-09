# train.md — 训练过程与实测结论

配套：[`README.md`](README.md)（环境与代码实现）、[`experience.md`](experience.md)
（关键错误与修正）。

文中标 **[实测]** 的是本仓当场跑出来的数字，**[原项目]** 的是从完整仓
`D:\项目\RobortDog\upstair` 的历史记录（`summary.json` / tensorboard / `reports\`）里
读到的。

---

## 1. 训练是怎么组织的

`scripts\train.py` 驱动一条**课程**：每一档是一个 `StageSpec`（地形 + 毕业判据 +
预算），跑满预算或毕业就换下一档。

默认课程（`robotdog/curriculum.py` 的 `default_curriculum`）：

| 阶段 | 地形 | 目标成功率 | 最少回合 | 预算 |
| --- | --- | --- | --- | --- |
| `flat` | `flat_spec(n_up=0)` | 0.90 | 30 | 250,000 |
| `easy` | 5 cm × 8 级 | 0.80 | 40 | 300,000 |
| `medium` | 12 cm × 8 级 | 0.65 | 40 | 350,000 |
| `hard` | 19 cm × 8 级 | 0.50 | 40 | 400,000 |

`--stage-top-cm 11.0` 会把高于 11.0 cm 的阶段全部压到 11.0；`--mixed-heights 8.0,11.0`
让每个子环境回合随机取一个高度（8 个环境实测为
`8.0 / 8.4 / 8.9 / 9.3 / 9.7 / 10.1 / 10.6 / 11.0 cm`）。

毕业条件（`experience.md` §4.1 那条根因修正之后）三者同时成立：
`n_total >= min_episodes` **且** `success_rate >= target_success` **且**
`n_total >= min_timesteps`。`--min-stage-frac F`（默认 0.5）把每档
`min_timesteps` 设成 `int(max_timesteps × F)`；设 0 恢复旧行为。
预算耗尽但没达标时 `advance_on_budget=True` 仍然推进（会打印出来）。

PPO 超参（配方里显式给的）：`n_steps=256`、`n_envs=8` ⇒ 一次更新粒度 **2,048 步**；
`batch_size=512`、`n_epochs=5`、`target_kl=0.03`、`ent_coef=0.002`、
`net_arch=[512,256]`、`log_std_init=-2.0`、`gamma=0.99`、`learning_rate=1.5e-4`。

向量环境链路是 `DummyVecEnv → VecMonitor → VecNormalize`。
**`DummyVecEnv` 不是退路而是唯一选项**：受限沙箱禁止命名管道，`SubprocVecEnv` 起不来。
`norm_obs=True`、`norm_reward` 默认 `False`（见 `experience.md` §4.2）。

---

## 2. 一次典型训练长什么样

```bat
.venv\Scripts\python.exe scripts\train.py ^
  --run-name <name> --stage all --stage-top-cm 11.0 --sub-steps 1 ^
  --mixed-heights 8.0,11.0 --n-envs 8 --net-arch 512 256 ^
  --learning-rate 1.5e-4 --target-kl 0.03 --log-std-init -2.0 ^
  --gait-prior --gait-file artifacts\trot_stairs.json ^
  --obs-base-lin-vel-body --obs-base-height --obs-foot-height ^
  --obs-touch --obs-stair-probe --obs-drift --obs-gait-phase ^
  --terminate-contact trunk --log-interval 5 --save-every 50000 ^
  --min-stage-frac 0.5 --det-episodes 10 ^
  --cfg episode_length_s=20.0 --cfg reward.heading=-30.0 --cfg reward.heading_max=0.35 ^
  --cfg terminate_contact_steps=10 --cfg terminate_height_ref=support
```

启动时会打印几行**必须核对的**自检：

```
[步态先验] 已启用 residual RL: 5.48 Hz, ±0.265 thigh / ±0.986 calf (norm. units), ...
[观测] 每帧 66 维, history=10, 共 660 维
[课程] 最少训练步数下限 = 50% 预算 (最紧的一档 175,000 步 ≈ 85.4 次 PPO 更新, 每次 2,048 步)
[混合地形] 各子环境 8.0cm@50cm / 8.4cm / ... / 11.0cm@50cm
[模型] 观测维度=(660,) 动作维度=(12,) 网络=[512, 256] n_steps=256 batch=512 epochs=5 可训练参数量 = 942,873
```

`--mixed-heights 8.0,11.0` 只在 `medium` 段生效；`flat` / `easy` 阶段的地形由课程表决定。

### 产物

```
runs\<name>\
    model_final.zip                  最终权重（660 维约 11.4 MB，942,873 参数）
    model_final_vecnormalize.pkl     观测归一化（必须与 zip 成对使用）
    model_best.zip                   只在该 run 毕业过至少一个阶段时才写
    checkpoints\<stage>_<steps>_steps.zip
    checkpoints\<stage>_final.zip
    artifacts\best_<stage>.zip       阶段内最优
    summary.json                     全部超参 + 结果 + 逐阶段统计
    tensorboard\ppo_0\events.out.tfevents.*
```

`summary.json` 关键字段：`stages[]`（每档的 `timesteps` / `graduated` /
`success_rate` / `fall_rate` / `mean_return` / `wall_seconds`）、
`total_timesteps`、`steps_per_second`、`best_stage`、`env_config`、`obs_config`、
`reward_config`、`reproduce`。

> ⚠ `total_timesteps` 是**跨 resume 的全局累计**，不是单次运行的步数。
> `reproduce` 字段**分两代**：course31 之后原样回显真实命令行，可信；
> course22 只有 5 个参数，是有损的。

---

## 3. 实测复现结论（重要）

### 3.1 参考 checkpoint 的能力带能复现

用本仓的 `scripts\eval.py` 评估从完整仓取来的 course22 参考权重
（`runs\course22_ref\checkpoints\medium_final.zip`），结果与完整仓自己的官方报告
`upstair\runs\course22\reports\course22_eval_trained.json` **一致**：
`train_flat` 100%、`train_h050` 100%、`train_h120` **0%**、`train_h190` **0%**。
**[实测]**

用自己的确定性探针（每档 20 回合，`seed0=21000`，观测按同名
`*_vecnormalize.pkl` 归一化）测得的能力带：

| 高度 (cm) | 8.0 | 9.0 | 9.5 | 10.0 | 10.5 | 11.0 | 11.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 成功率 | 100% | 100% | 90% | 90% | 60% | 10% | 0% |
| 均前进 (m) | 4.56 | 4.56 | 4.13 | 4.15 | 3.48 | 1.56 | 0.80 |
| 主要结局 | success | success | 18/20 成 | 18/20 成 | 12 成 / 6 摔 / 2 超时 | 13 摔 / 5 超时 | 12 摔 / 8 超时 |

**11.0 cm 是墙**，且 11.0 cm 上失败以**摔倒**为主，11.5 cm 上是摔倒 + 超时各半。
**[实测]**

原项目交付的三条策略（6 档 = 8.0/9.0/9.5/10.0/10.5/11.0 cm，40 回合 × 6 档）：

| 标签 | 权重 | 8.0 | 9.0 | 9.5 | 10.0 | 10.5 | 11.0 | 均值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `c22 final` | `runs\course22\checkpoints\medium_final.zip` | 100 | 100 | 90 | 90 | 60 | 10 | 75.0% |
| `c31 best` | `runs\course31\artifacts\best_medium.zip` | 100 | 100 | 97.5 | 87.5 | 65 | 15 | 77.5% |
| `c49 final` | `runs\course49\model_final.zip` | 100 | 95 | 95 | 92.5 | 70 | 7.5 | 76.7% |

**[原项目]** 三者差异**统计上互不显著**；只承诺落进同一能力带，不承诺逐位复现
（同一条命令重训，档位有 ±5~10 pt 波动是常态）。

### 3.2 三条记录的配方

`course22`（从零）就是 §2 那条命令。`course31` / `course49` **都从 course22 续训，
不是前后相承**：

```bat
rem course31 = course22 的权重 + 两条 --cfg
... --resume runs\course22\checkpoints\medium_final.zip --stage medium --stage-timesteps 400000 ^
    --cfg reward.stair_progress_foot_ref=true --cfg reward.stair_progress_min_loaded_feet=2

rem course49 = course31 + 两条「高台阶躯干接触去抖收紧到 1 步」
... --cfg contact_debounce_adaptive=true --cfg contact_debounce_hi_steps=1
```

⚠ **course31 / course49 没有 `model_best.zip`**：续训后 `best_stage` 是 `null`
（没有重新毕业任何阶段），所以 `model_best` 从没被写过——尽管 `summary.json` 的
`artifacts.best_model` 里还写着这个路径。交付权重分别是
`artifacts\best_medium.zip` 与 `model_final.zip`。

⚠ `--resume` **静默忽略** `--net-arch` / `--learning-rate` / `--target-kl` /
`--n-epochs`，所以 §3.1 的两条续训只加 `--cfg`。

### 3.3 ❌ README §4.1「`--stage medium` 从零 22 分钟」不可复现

这是本次复现**最重要的一条负面结论**。

**证据链：**

1. 按那条命令（`--stage medium`，660 维，`--min-stage-frac 1.0`）从零训练，
   跑到 **360,736 步 / 440 回合仍是成功率 0.0%**，均回报 -3650，
   `train/explained_variance` 在 0.008~0.045 之间（critic 完全没学到），
   且 `--target-kl 0.03` 频繁触发早停（`Early stopping at step 2/3 due to reaching
   max kl`）。**[实测]**
2. 参考 run 的 tensorboard event 文件里，**全部 20 个 tag 的 step 区间都是
   `[3909256 .. 4441736]`**，53 个采样点，**完全没有 flat/easy 阶段的标量**。
   它的 `curriculum/stage_steps` 才是阶段内步数（10,240 → 542,720）。
   ⇒ tensorboard 的 step 轴是**全局累计步数**。**[原项目]**
3. 参考 `summary.json`：`stages` 只有 `medium` 一条（`timesteps=552,296`、
   `graduated=true`、`stop_reason` 含「成功率 66.0% >= 目标 65%」、
   `wall_seconds=1339.4`、`steps_per_second=412.3`），而 `total_timesteps=4451312`。
   ⇒ **medium 之前已消耗 3,909,256 步**，medium 阶段继承的是**已在 flat/easy 上
   学好的策略**。**[原项目]**
4. 佐证：course22 的 medium 阶段第一个 tensorboard 采样点（`stage_steps=10,240`）
   成功率就已经是 **0.4255**、`mean_return +302.75`；而从零训练的同一位置是
   **0.0000 / -5009.8**。**[实测]**
5. 血统：course21 的 `total_timesteps=3899016`，唯一 stage 是 medium（700k 步、
   未毕业）；course22 = course21 的续训。course31 = 4451312 + 400000 =
   4851312；course49 同为 4851312。**[原项目]**

**结论**：`README §4.1` 的「22 min / 4.45M 环境步」把**跨 resume 的全局累计步数**
当成了单次运行。`--min-stage-frac 1.0` 也与「552k 步毕业」不自洽
（1.0 要求先跑满 400k 才可能毕业，而 552,296 > 400,000 说明参考用的是更小的值，
大概率是默认的 0.5）。**「22 分钟」只对 medium 这一段成立**（552,296 步 ÷
412 步/s ≈ 22 min）；整条 flat→easy→medium 流水线约数小时。

### 3.4 从零训练能达到什么程度（本仓实测）

按 §2 的完整配方（`--stage all`）从零跑完：

```
runs\course22_improved   total_timesteps = 1,174,360   n_updates = 1521
train/explained_variance = 0.946      train/std = 0.135
flat   毕业（100%）
easy   预算耗尽 300,000 步，成功率 0.0% < 目标 80%  → advance_on_budget 推进
medium 预算耗尽 350,000 步，成功率 0.0% < 目标 65%
hard   预算耗尽 400,000 步，成功率 0.0%
```

**训练报告的摔倒率确实大幅改善**（这是训练循环自己的日志，与我的探针无关）：

| easy 阶段 | 60 回合 | 80 | 100 | 140 | 180 | 300 | 340 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 原始配方 | 58.0% | 54.0% | 46.0% | — | — | — | — |
| 改进配方 | — | — | 40.0% | 28.0% | 20.0% | 20.0% | 22.0% |

⇒ 摔倒率约为原始配方的**一半**。但 **medium（11 cm）成功率始终 0%**，
而且摔倒率一开始反而低（20 回合 20.0%、40 回合 12.5%），随后恶化到
360 回合 30.0% / 400 回合 46.0%。

**产出的策略与参考差距巨大**（6 回合/档，确定性）：5 cm **0% 成功 / 100% 摔倒**、
8 cm **0% / 100% 摔倒**、11 cm 0% / 83% 摔倒。

**但它确实在「努力」**：`mean|a| ≈ 0.25~0.29`，而参考是 **0.42**
（从零学不动的那些 run 只有 0.02~0.10）。逐高度的 `max base_x`：

| h (cm) | 2.0 | 3.0 | 5.0 | 6.0 | 8.0 | 11.0 |
| --- | --- | --- | --- | --- | --- | --- |
| `max base_x` (m) | 4.72 | 3.96 | 2.55 | 2.37 | 1.91 | 1.49 |
| `mean_vx` (m/s) | +1.197 | +0.895 | +0.457 | +0.588 | +0.269 | +0.044 |

⇒ 能走能爬 2~3 cm（到达 x≈4.7 / 4.0），5~6 cm 前进受限，11 cm 基本原地。
**训练有学习，但强度不足以复现参考。**

**尺度参考**：参考的 `mean|a| = 0.42` 是 flat→easy 共 3.9M 步攒出来的。
从零训练要进入参考的工作区间，需要 2~4M 步量级，150k~400k 步的 run 无法判定成败。

### 3.5 消融：什么杠杆有效、什么无效

**A. 奖励归一化（`--norm-reward`）— 温和的稳定器，不是解药。**
3 cm 台阶、8 环境、150k 步，双方各用**自己的**归一化器：

| arm | 成功率 | `mean|a|` | 11 cm |
| --- | --- | --- | --- |
| off | 0.474 | 0.070 → 0.063 | 无成功 |
| on | 0.539 | 0.080 → 0.083 | 无成功 |

**B. 侧向惩罚清零 + 前进奖励提到 8.0 — 无效。**
把所有侧向惩罚（`straight_line` / `heading` / `lateral_vel` /
`lateral_vel_track` / `yaw_rate`）设 0、`forward_vel=8.0`，从零训练 30,720 步时
残差仍停在 0.025，h200 摔倒率 0.88。⇒ **「惩罚压死残差」不是唯一机制。**

**C. `action_scale=0.60` — 只降摔倒，不增前进。**
11 cm 摔倒率 0.62 → 0.25 → 0.00，但 `adv` 反而从 0.54 降到 0.47。

**D. 奖励再平衡（`straight_line=-4.0`、`heading=-8.0`、`heading_max=0.6`）— 无效。**
只降低 return（-17853 → -14044），**不提高 advance**。

**E. 成功奖励放大（`reward.success=200`、`reward.termination=-20`）— 早期有效，但结论待复核。**
在 3 cm 上确实让 `|a|` 增长更快、11 cm 摔倒率下降。⚠ 这批数字来自一个后来被发现
**会报错回合**的探针（见 `experience.md` §5.1），**未用修正后的探针重跑**，
引用时请自行复核。

**F. 参考权重 warm-start 微调（8 cm，150k 步）：**

| 对照 | 8 cm 保持 | 11 cm |
| --- | --- | --- |
| 参考奖励 | 133k 步仍 100% | 0%（adv 1.44 → 0.88） |
| 新奖励项 | 106k 88% → **133k 62% → 160k 25%** | 0% |
| 新奖励 + `lr=3e-5` | 160k 75% | 12%（adv 2.56，最好） |

前 26,624 步三条对照**逐位相同**，起点干净。⇒ **参考奖励下 11 cm 是微调也翻不过去
的墙**；新奖励项在默认学习率下会**弄丢已经学会的 8 cm**（原因见
`experience.md` §6.5）。

### 3.6 11 cm 为什么是墙：不是奖励，也不是「抬不起来」

这一节是本仓的核心诊断，详细推导见 `experience.md` §6。

1. **参照面口径错了**，旧 `foot_clearance` 在 11 cm 上给出的绝对读数**比 8 cm 还
   大** —— 抬升信号完全反了。换成落脚面口径 `advance_clearance()` 后判别力
   与实测成功率**同形**（0.987 / 0.986 / 0.942 / 0.674 / 0.203 对
   100% / 100% / 90% / 60% / 10%）。
2. **不是「抬不起来」**：纯先验（零残差）在**平地**上的摆动峰值实测
   **0.1124 m**（默认频率）、**0.1658 m**（频率 ×0.6）、最高 **0.21 m**
   （gain 1.4 + 幅度 1.6 + 频率 ×0.6）。而参考策略在 11 cm 上只抬到约 0.09 m。
   ⇒ **先验有这个几何能力，策略没有用它。**
3. **但开环调参也翻不过去**：18 组（gain 1.0/1.4/1.8 × 幅度 1.0/1.3/1.6 ×
   频率 ×1.0/0.6）零残差下，9 / 11 / 13 cm 上**没有一次成功**。
   ⇒ 缺的不是峰值高度，而是**摆动相里的向前落脚**。

**下一步（唯一有证据支持的突破口）**：显式落脚相 IK —— 触地记抬脚点 →
用 `_step_surface_ahead()` 求目标踏面 → 摆动相把足端从抬脚点插值到该踏面上的
落点 → 解算髋/膝角替换 `gait.radians()`，残差策略仍叠在其上。这是对先验的
**结构性改动**，不是继续调奖励权重。

---

## 4. 训练时的核对清单

1. **看启动自检**：观测维数（默认 420 / 课程配方 660）、`最少训练步数下限`、
   `[混合地形]` 各子环境高度、`可训练参数量`（660 维应为 942,873）。
2. **看 `curriculum/success_rate` 与 `train/explained_variance` 一起**。
   `explained_variance ≈ 0` 意味着 critic 没学到，这个 run 不会有进展——
   不要继续等。
3. **看 `train/std`**。参考在 0.14 附近；从零学不动的 run 也停在 0.136 附近，
   所以它**不能**单独当判据。
4. **`Early stopping at step 1/2 due to reaching max kl` 频繁出现**说明
   `target_kl=0.03` 正在饿死梯度批次。参考 run 的 `approx_kl` 是 0.036~0.044。
5. **至少 40 回合评估**（`experience.md` §5.7）。
6. **并行 run ≤3**（`experience.md` §5.5）。
