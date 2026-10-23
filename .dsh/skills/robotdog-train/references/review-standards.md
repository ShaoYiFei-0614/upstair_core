# 审查标准：通用 / 任务族 / 任务实例

审查标准不是一张固定清单。它随任务变化，分三层；E、H、L5 风险审查都按这三层取材。

## 1. 三层结构

| 层 | 回答什么 | 来源 | 变化频率 |
| --- | --- | --- | --- |
| 通用层 | 任何具身 RL 都不该出的错 | 本文件 §2 | 低 |
| 任务族层 | 这类 MDP 特有的失败模式 | §3 + `doc/<module>.md` + `experience.md` | 中 |
| 任务实例层 | 这次实验的 claim/prediction/threshold 或 uncertainty/stop | 任务书 `task_book` | 每次实验 |

## 2. 通用层（所有具身 RL）

- **数值健康**：obs/action/reward 无 NaN/Inf；训练日志无异常 traceback；OOM 与编译错误分开。
- **维度与顺序**：obs/action 维度、拼接顺序、归一化、frame stack、warm-start 维度一致。
- **回合与终止**：回合长度、超时、终止条件、reset 后状态、seed 可复现。
- **奖励**：`reward_terms` 无退化（全 0/常数/爆炸）、尺度合理、门控不误杀。
- **动作**：边界、缩放、clip、执行器顺序；动作变化是否改变旧 checkpoint 语义。
- **配置与版本**：config/commit/工作区、默认值单源、checkpoint 与 config 兼容。
- **Trace 完整性**：`task_book`、`gate_result`、rollout、summary 能互相对齐；
  证据指针可追溯；没有“只凭记忆”的结论。
- **资源**：并发 ≤2 臂；长训练有 checkpoint；视频不会把内存打爆。

## 3. 任务族层（本仓库：运动控制）

- **基座/姿态**：高度、俯仰/滚转、接触、腾空/拖步；摔倒判据是否与任务定义一致。
- **步态**：相位、频率、步幅、左右对称；先验缩放是否放大 calf 等已知死路。
- **指令跟踪**：vx/vy/yaw 的采样口径、滤波、probe、deadband；训练分布与 eval 是否一致。
- **地形**：台阶高度/踏面/边缘、reset 解析高度、地形课程；最窄 tread/最高级是否覆盖。
- **终止**：躯干接触、越界、超时；终止是否误伤正常爬楼。
- **评测口径**：`--sampled-command` 与 pinned-command 不得混比；episodes ≥40；单 run n=1。
- **视觉/Trace 交叉**：画面中的高度/步态/结局是否与 rollout 的成功率、回合长度、终止原因一致。
- 其他任务族（操作/导航）在对应任务包中补，不把运动控制标准硬套。

## 4. 任务实例层

- 确认性：`claim` 是否可被否证；`prediction` 有方向/幅度/对照；
  `threshold` 是预先登记，不是事后拟合。
- 探索性：`uncertainty` 说清不知道什么；`prediction` 可证伪；
  `stop_condition` 明确何时停。
- 指标：stage、episodes、seeds、采样口径、基线是否与任务书一致。
- 结果：预登记的每个 `key_changes` 是否有量化影响与对照；无法隔离时写混淆项。

## 5. 状态与判定

| 状态 | 含义 | 谁能给 |
| --- | --- | --- |
| `PASS` | 证据满足判据，可继续 | E / 主代理按 C 设计执行 |
| `BLOCK` | L0–L4 失败或 blocker 未清，禁止放行 | E / gate_result |
| `SUSPECT` | 证据不足/方向不符，但不阻断；需独立审查 | gate_result / L5 审查 |
| `CONSISTENT` | 视觉与 Trace 指向同一结论 | H |
| `CONFLICT` | 视觉与 Trace 冲突，主代理必须定根因 | H |
| `INSUFFICIENT` | 帧/Trace 不足，不能判断 | H |

## 6. 视觉审查的特殊纪律

视觉子代理**不得单独给出“成功”结论**。它必须回答一个可操作问题：

> “画面中的行为是否与任务书声明的成功定义一致？如果不一致，Trace 中哪一段证据支持这一判断？”

要求：

- 先加载 skill `visual-monitor` 执行两段式 rubric；不另造视觉标准。
- 输出 `VISUAL_TRACE_VERDICT` + 观察 + Trace 证据指针 + 冲突/疑点 + 建议。
- 画面好但 Trace 成功率低、或 Trace 好但画面明显摔倒：都标 `CONFLICT`，不取平均。
- 证据不足写 `INSUFFICIENT`；不要用“看起来还行”代替证据。
- 两臂比较各派独立 H；不复用同一上下文。

## 7. E 的评审清单（按层取材）

1. 任务书类型/字段是否完整，claim/prediction/threshold 或 uncertainty/stop 是否可检验。
2. C 的 L0–L5 是否覆盖，命令/判据/证据指针/中止条件是否齐全。
3. D 的三层 Trace、gate_result、snapshot 字段是否够用，是否能重建结论。
4. 通用层：NaN、维度、回合、奖励、动作、config/commit、checkpoint、Trace 完整性。
5. 任务族层：基座/步态/指令/地形/终止/评测口径/视觉交叉。
6. 任务实例层：是否与任务书一致；有没有事后改阈值。
7. 真实 diff、默认值单源、文档同步、warm-start、经验死路。
8. 视觉纪律是否写清：不得单独下成功结论；必须给 Trace 证据。
9. L5 失败是否设计为“不阻断但独立审查”，而不是直接禁止训练。
10. 回滚：出问题能否只回滚本次改动；是否依赖未提交的用户改动。

## 8. L5 风险审查（独立子代理）

L5 `SUSPECT` 时另派 fresh 子代理，输入任务书 + L5 证据 + Trace 片段。输出：

```text
L5_RISK_VERDICT: RELEASE | RELEASE_WITH_MONITORING | DO_NOT_RELEASE
EVIDENCE:
  - ...
RISKS:
  - ...
MONITORING:
  - <放行后必须监控的事件/阈值>
STOP_CONDITION:
  - <什么情况下立即停>
CONFIDENCE: high | medium | low
```

主代理据此选择 `go_with_risk` 或 `fix_first`；不得把 L5 失败直接当 L0–L4 BLOCK。
