# 奖励项清单（当前实现，2026-10-20）

> **状态：只整理、未精简。** 本文是 `robotdog/reward.py::RewardConfig` 与 `config.yaml` 的逐项对照表，供后续精简/审计使用；**代码与默认配置没有改动**（唯一相关的近期改动是 `run.log_reward_terms` 训练期逐项记录，见 §5）。

唯一实现：`robotdog/reward.py`（取值理由见 `config.yaml` 的 reward 段注释；调参教训见 `experience.md` §12–§16/§21/§23；量级实测见 `doc/reward.md`）。

## 1. 怎么读这张表

- **代码默认** = `RewardConfig()` 的字段默认值；**config.yaml** = 当前生效值（config 是权威，缺省时才落默认）。
- **课程覆盖** = `curriculum.reward_overrides` 是否逐级改这个键（见 §3）。键在代码默认与 config 之间相同**不代表训练中恒定** —— 课程会在每级边界 `set_reward_overrides` 下发新值。
- 量级引用 `doc/reward.md` 的实测口径：平地零残差 / 3cm 被挡 / 已训策略 3cm 回合。
- 「保险丝」= 稳态≈0、只在姿态发散/动作抖动时起作用；「默认 0（已否决）」= 曾被消融否决，现在权重为 0（保留字段是为了记录与可复现，不是待开选项）。

当前 `compute_reward` 会写入 `terms` 的键共 **21** 个：
`action_rate`、`air_time`、`alive`、`base_height`、`body_collision`、`com_margin`、`command_follow`、`foot_hold`、`foot_impact`、`foot_lift`、`foot_scuff`、`forward_vel`、`joint_limit`、`joint_vel`、`leg_collision`、`load_share`、`orientation`、`pitch_posture`、`probe_lift`、`stair_progress`、`torque`。

## 2. 奖励项（按组）

| 组 | 项 | 代码默认 | config.yaml | 课程覆盖 | 状态 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 指令跟踪 | `forward_vel` | 10 | **10** | — | 保留 | \|v−cmd\| 超窗时饱和；超速侧再叠罚（见 forward_overspeed） |
| 指令跟踪 | `command_follow`（合成项） | — | — | — | 保留 | 横向/偏航两项按权重相加；分量权重是 `lateral_vel`/`yaw_rate`（见 §2/§4），不是单字段 |
| 爬升 | `stair_progress` | 3 | **1200** | — | 保留 | 一次性结算，只在四足支撑面刷新新高时付增量 |
| 步态 | `air_time` | 0.6 | **0.6** | — | 观察 | 2 Hz 贴地 crawl 步态≈0；换 trot/高抬腿才有量 |
| 步态 | `foot_hold` | 0 | **0** | — | 默认 0（已否决） | 2×2 消融否决；不要重新启用 |
| 步态 | `foot_lift` | 0.9 | **0.9** | 是 | 保留 | 足端速度门控 × 抬腿高度占比；目标随台阶高 |
| 步态/受阻 | `probe_lift` | 1 | **0** | 是 | 保留 | 势函数：只在受阻期间、只给低于台面的脚记新峰值 |
| 姿态 | `base_height` | -2 | **-2** | — | 保险丝 | 稳态≈0；姿态发散才咬 |
| 姿态 | `joint_limit` | -0.5 | **-0.5** | — | 保险丝 | 稳态≈0 |
| 姿态 | `orientation` | -1.2 | **-1.2** | — | 保留 | roll²+pitch²，量级小但方向正确 |
| 姿态 | `pitch_posture` | 0 | **0** | — | 默认 0（已否决） | 曾把 flat_4 76%→18%；重引入需重设死区 |
| 能耗 | `action_rate` | -0.02 | **-0.02** | — | 保险丝 | 稳态≈0；抖动才咬 |
| 能耗 | `joint_vel` | -0.00002 | **-0.00002** | — | 保险丝 | 稳态≈0 |
| 能耗 | `torque` | -0.00015 | **-0.00015** | — | 保留 | Στ²，量级 −0.1/步 |
| 碰撞 | `body_collision` | -1.2 | **-1.2** | — | 保留 | 躯干接触布尔罚 |
| 碰撞 | `leg_collision` | -0.25 | **-0.25** | — | 保留 | 腿部接触布尔罚 |
| 基础 | `alive` | 0.15 | **0.15** | — | 保留 | 与指令跟踪一起构成「活着且跟得上」的底分 |
| 终止 | `success` | 50 | **500** | — | 保留 | 登顶一次性 +500（config 覆盖，代码默认 50） |
| 终止 | `termination` | -15 | **-5000** | — | 保留 | 摔倒一次性 −5000（config 覆盖，代码默认 −15） |
| 接触组（默认关） | `com_margin` | 0 | **0** | — | 关（已否决） | R4；需 contact_rewards=true 才结算 |
| 接触组（默认关） | `foot_impact` | 0 | **0** | — | 关（已否决） | R2；需 contact_rewards=true 才结算 |
| 接触组（默认关） | `foot_scuff` | 0 | **0** | — | 关（已否决） | R1；需 contact_rewards=true 才结算 |
| 接触组（默认关） | `load_share` | 0 | **0** | — | 关（已否决） | R3；需 contact_rewards=true 才结算 |

> **代码默认不是当前配置**：config 覆盖了上表中 5 个字段的取值。引用权重时必须用 config.yaml（或 `runs/<run>/plan.json` 的逐级覆盖），不要用 `RewardConfig()` 的默认值。

## 3. 课程级奖励覆盖（`curriculum.reward_overrides`）

逐级下发的键与值（某级缺列时沿用该行最后一格；`None` 表示该级不动）：

| 键 | 0cm | 0cm | 0cm | 0cm | 3cm | 4.5cm | 6cm | 7.5cm | 9cm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `foot_lift` | 0.9 | 0.9 | 0.9 | 0.9 | 1 | 1.05 | 1.2 | 1.3 | 1.4 |
| `foot_lift_step_ratio` | 1.4 | 1.4 | 1.4 | 1.5 | 1.6 | 1.7 | 1.8 | 1.9 | 2 |
| `probe_lift` | 0 | 0 | 0 | 0.4 | 0.6 | 0.75 | 0.9 | 1.05 | 1.2 |
| `forward_overspeed` | 0 | 0 | 0 | 0.3 | 0.6 | 0.8 | 1 | 1 | 1 |
| `yaw_rate` | 8 | 6 | 6 | 4 | 4 | 4 | 4 | 4 | 4 |

同一节课还有指令范围覆盖（`command_vy_max` / `command_yaw_max`），它们改的是观测/奖励里的指令量程而不是奖励权重，见 `plan.json`。

## 4. 参数 / 阈值 / 开关（不是权重，但会被代码或课程读）

| 字段 | 代码默认 | config.yaml | 说明 |
| --- | --- | --- | --- |
| `air_time_max` | 0.6 | 0.6 | air_time 上限滞空（s） |
| `air_time_target` | 0.3 | 0.3 | air_time 目标滞空（s） |
| `com_margin_flat_only` | true | true | R4 只平地生效 |
| `com_margin_scale` | 0.15 | 0.15 | R4 裕度尺度 |
| `com_margin_tol` | 0.15 | 0.15 | R4 质心裕度死区 |
| `contact_rewards` | false | false | 接触组总开关（false=整组不结算、不进 terms） |
| `contact_telemetry` | false | false | 接触组遥测输出（默认关） |
| `foot_lift_gate_speed` | 0.25 | 0.25 | 足端速度门控阈值（m/s） |
| `foot_lift_step_ratio` | 1.4 | 1.4 | 抬腿目标 = ratio × 台阶高（课程逐级升） |
| `foot_lift_target` | 0.08 | 0.08 | 抬腿目标下限（m） |
| `forward_overspeed` | 1 | 1 | 超速罚斜率（课程逐级 0→1） |
| `forward_track_window` | 0.25 | 0.25 | 前向跟踪死区宽（|误差|/窗口） |
| `hold_elev_frac` | 0.3 | 0.3 | foot_hold 的「踩在台面」判定比例 |
| `impact_force_scale` | 450 | 450 | R2 冲击罚尺度（N） |
| `impact_force_tol` | 450 | 450 | R2 落地冲击容忍力（N） |
| `lateral_range_floor` | 0.15 | 0.15 | 横向跟踪窗口下限（零指令时防除零） |
| `lateral_vel` | 6 | 6 |  |
| `load_share_cap` | 0.7 | 0.7 | R3 负载共享上限 |
| `load_share_flat_only` | true | true | R3 只平地生效 |
| `load_share_min_feet` | 2 | 2 | R3 最少支撑脚数 |
| `overspeed_cap` | 2 | 2 | 超速罚上限（×权重） |
| `overspeed_range` | 0.45 | 0.45 | 超速罚起算窗口 |
| `pitch_cap` | 0.4 | 0.4 | pitch_posture 单项上限（rad） |
| `pitch_tol_flat` | 0.14 | 0.14 | pitch_posture 平地死区（rad） |
| `pitch_tol_per_m` | 1.2 | 1.2 | pitch_posture 死区随台阶高加的斜率 |
| `probe_blocked_frac` | 0.3 | 0.3 | 判「被阻挡」的速度占比 |
| `probe_lift_low_foot_only` | false | — | 只给低于台面的脚记势能（§23.26 修复开关） |
| `probe_lift_target` | 0.06 | 0.06 | 受阻抬腿起算线（m） |
| `scuff_mu_hard` | 0.8 | 0.8 | R1 硬阈值 |
| `scuff_mu_soft` | 0.35 | 0.35 | R1 软阈值 |
| `scuff_v_ref` | 0.15 | 0.15 | R1 滑擦参考速度 |
| `termination_progress_relief` | 0.5 | — | 摔倒罚随爬升进度减免的比例 |
| `track_tail` | 1 | 1.5 | 跟踪形状的线性尾项系数（禁止二次） |
| `yaw_range_floor` | 0.3 | 0.3 | 偏航跟踪窗口下限 |
| `yaw_rate` | 4 | 4 |  |
| `yaw_zero_gain` | 1.5 | — | 零偏航指令时的误差放大倍数 |

## 5. 训练期如何看到这些项

`run.log_reward_terms` 列出要记录的分量（空=关）。`ProgressCallback` 每行进度打印 `[奖励/步] 项=窗口均值`，并追加 `runs/<run>/reward_terms.jsonl`（每行 `{"step","stage","terms"}`）。只有 `compute_reward` 真的写进 `terms` 的键才会出现：接触组在 `contact_rewards=false` 时整组不入 terms，日志里也不会有。

`step` 是**该级内**的步数（每级重建 PPO，`num_timesteps` 从 0 起算），不是整条课程的累计；跨级比较请用 `stage` 分组。行距按进度打印节奏（`log_every` 相关），级末 drain 可能产生间隔异常的一行（用于把余量结清）。

当前 config 默认记录 13 项：`forward_vel`、`command_follow`、`foot_lift`、`air_time`、`probe_lift`、`stair_progress`、`alive`、`torque`、`joint_vel`、`action_rate`、`orientation`、`body_collision`、`leg_collision`。

## 6. 与「精简」有关的候选（**本轮不改**，仅登记）

- **可删候选（默认 0 且被否决）**：`pitch_posture`、`foot_hold`、接触组四项 —— 删它们只减配置面/维护面，训练收益无证据；删除前须按 `doc/reward.md`「改奖励的纪律」跑 eq_check + 250k stair3 对照。
- **不建议删的保险丝**：`base_height`、`joint_vel`、`action_rate`、`joint_limit` —— 稳态≈0，但姿态发散/抖动时是唯一兜底。
- **步态相关待定**：`air_time` 在 2 Hz 贴地 crawl 步态≈0；若以后只用 crawl 先验，可考虑降级为观察项；若还要 trot/抬腿步态则保留。
- **不可动**：`forward_vel`、`command_follow`、`stair_progress`、`success`/`termination`、`alive`、`foot_lift`、`probe_lift`、碰撞两项 —— 任务本体。

