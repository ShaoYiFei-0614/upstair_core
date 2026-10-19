# 消融配置（`ablations/`）

对应：`ablations/*.yaml`、`tests/test_ablation_levels.py`。本文只讲**怎么用、为什么这样分层、
证据与边界**；每个键的取值理由在 yaml 与 `config.yaml` 的注释里，不抄默认值。

## 0. 为什么是 `ablations/` 而不是 `configs/`

`.gitignore` 忽略 `configs/`（历史约定：训练配置快照是本地产物）。但**消融的每一级是
实验协议的一部分**，必须可追溯 ⇒ 放 tracked 目录 `ablations/`。

## 1. 四段消融（单变量递进）

| 文件 | Level | 与上一级的唯一差异 | 关键观测/奖励 |
| --- | --- | --- | --- |
| `crawl24.yaml` | —（还原） | crawl24 原始配方，无任何新特性 | 对称 AC、53 维、单级 15cm |
| `level0.yaml` | L0 | 与 `crawl24.yaml` 同内容（基线） | 同上 |
| `level1.yaml` | L1 | `obs.privileged_foot_contact: true` | 观测 57 维；actor 53 / critic 57，策略切 `AsymmetricActorCriticPolicy` |
| `level2.yaml` | L2 | `reward.contact_rewards: true` + `reward.foot_scuff: -8.0`（scuff 标定参数一并给） | 奖励**新增** `foot_scuff`/`foot_impact`/`load_share`/`com_margin` 项，旧项权重逐项不变 |
| `level3.yaml` | L3 | 课程采样器：两级 `[10.5, 15.0]cm`、逐级 timesteps/LR/三轴指令区间/reward_overrides | 同一训练脚本内跑两级，用于验证「采样器可配置」 |

**为什么 L0 与 `crawl24.yaml` 重复**：`crawl24.yaml` 是「历史还原」基准（回放对账用），
`level0.yaml` 是「消融起点」；两者内容相同但语义不同，分开可以在后续重训 L0 时不动还原基准。

## 2. 怎么跑（同一个训练脚本，靠配置切换）

```
.venv\Scripts\python.exe scripts/train.py --config ablations/level0.yaml --run-name ablation_l0
.venv\Scripts\python.exe scripts/train.py --config ablations/level1.yaml --run-name ablation_l1
.venv\Scripts\python.exe scripts/train.py --config ablations/level2.yaml --run-name ablation_l2
.venv\Scripts\python.exe scripts/train.py --config ablations/level3.yaml --run-name ablation_l3
```

训练脚本**不感知 Level**：所有差异都来自 yaml（策略类由 `obs.privileged_foot_contact` 推导、
奖励项由 `reward.*` 开关推导、级数与采样区间由 `curriculum.*` 推导）。回放时也不传配置：
`play.py` 从 checkpoint 同目录的 `<模型>_env.json` 还原环境。

## 3. 证据（2026-10-19，本机 `.venv`）

- **还原语义**：`ablations/crawl24.yaml` 重建的 `EnvConfig` 与
  `runs/crawl24/checkpoints/stair15_final_env.json` 的 `env_config` 旧键 **deep-diff 为空**；
  新字段全部等于旧行为默认（`privileged_foot_contact=False`、`contact_rewards=False`、
  `yaw_rate_mode="legacy"`、`stair_guidance_under_hold=True`、四个 contact 权重 0.0）。
- **回放与历史记录逐项相同**（100 回合、`--sampled-command --stage stair15`）：
  900k = 96.0% / 摔 4.0% / 2.90m / 1.14m / 0.29m/s / 507 步；
  final = 91.0% / 摔 9.0% / 2.83m / 1.12m / 0.32m/s / 453 步。
- **四个 Level 冒烟**（每级 512 步、2 envs，`scripts/train.py --config <副本>`）：全部 exit 0；
  日志确认 L0=`每帧 53 维`、L1/L2=`每帧 57 维 + 非对称 AC: actor 53 / critic 57`、
  L3=`两级课程`（10.5→15cm，逐级 LR/指令区间/奖励覆盖都按 yaml 生效）。
- **自描述与产物**：冒烟 checkpoint 的 `_env.json` 记录 `privileged_foot_contact` 与
  `contact_rewards`/`foot_scuff`；`load_policy` 重建后 L0=`ActorCriticPolicy`(53/53)，
  L1–L3=`AsymmetricActorCriticPolicy`(53/57)；zip 的 `policy_class`/`policy_kwargs` 里也留痕。
- **奖励项**（20 步随机动作）：L0/L1 只有 18 个基线项；L2/L3 额外出现四个 contact 项，
  其余项逐位不变（开关是「加项」不是「改项」）。
- **测试**：`tests/test_ablation_levels.py` 4 条；全套 `test_env_contract + test_asymmetric +
  test_ablation_levels` = 32 条 OK。

## 4. 边界（不要过度解读）

- 本轮只验证**接口/链路兼容**：冒烟每级 512 步，**不是能力对比**；四段消融的训练结论
  要等实际训练（L0 vs L1 判特权观测、L1 vs L2 判奖励项、L2 vs L3 判课程采样器）。
- L3 的两级课程只用来证明「采样器可配置」，不是 height-ladder 的正式配方。
- `ablations/` 的 yaml 是**配置快照**，不是结果快照；改配方请新建文件，不要原地改历史文件。
- 还原基准的可信来源是 `_env.json` 与历史 `train.md`/`experience.md`，不是「今天的默认值」。

## 5. 首轮实测（2026-10-19 夜，单 seed）与使用前自检

四臂同起点（crawl23 850k）、同 1.2M 步、100 回合 15cm 回放：**L0 97.0% / L1 90.0% /
L2 90.0% / L3 83.0%**（L3 的 15cm 阶段只有 600k 预算，不可直接比）。

- **L1 与 L2 训练逐位相同**：L2 的四个 contact term 在 stair15 上数值恒为 0 ⇒ 奖励没变。
  ⇒ 用 L2 之前必须先测 term 触发率（见下），否则 A/B 无意义。
- 同配方重训方差 ≥6 点（L0 97% vs 历史 crawl24 91%）⇒ 臂间 ~7 点内差异不可归因，需 3 seeds。
- 详细数据与产物路径见 `train.md` 的「ablation 训练对照」条目；教训见 `experience.md` §23.65。

**开跑前自检（新增奖励项时必做）**：

```
# 在目标地形上跑一个回合，统计每个新 term 的非零步数占比与均值；占比为 0 说明开关没生效
.venv\Scripts\python.exe .tmp/_probe_contact_fire.py      # 本轮的写法，可直接改路径复用
```

若占比为 0：先调阈值/触发条件（例如 `foot_scuff` 的滑擦判据），或换一个在该地形上会触发的
项（如 `foot_lift`/`air_time`）再做对照 —— 不要用「训练轨迹相同」的两臂去下奖励结论。