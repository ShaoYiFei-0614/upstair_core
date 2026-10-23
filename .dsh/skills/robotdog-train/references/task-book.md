# 任务书（Trace 字段，不是独立文档）

任务书是实验意图的压缩表示，写在 Trace 的 `task_book` 事件里。
它回答“这次实验想证明/探索什么、怎样算支持/否证、什么时候停”，不写完整方案。

## 1. 为什么不是独立文档

- 任务书必须和 `gate_result`、`rollout`、`snapshot_review` 在同一份 Trace 里，
  评审器和汇总才能按同一 experiment_id 对齐。
- 独立 Markdown 容易与代码/配置漂移，也容易变成第二套叙事。
- 人类检视时看 `human_summary` 一屏即可；细节在字段里，不在长文里。

## 2. 两种实验类型

| 类型 | 适用 | 必填核心 | 不通过的含义 |
| --- | --- | --- | --- |
| 确认性 `confirmatory` | 已有明确 claim，要验证或否证 | `claim` + `prediction`（方向/幅度）+ `threshold`（预先登记） | 预测未被数据支持；不是“再训久一点” |
| 探索性 `exploratory` | 机制/方向不确定，先探索 | `uncertainty` + 可证伪的 `prediction` + `stop_condition` | 没看到可证伪信号就按 stop 停，不无限加步数 |

类型不确定时按**探索性**写，明确不确定性；不要在看过结果后才补一个“刚好能过”的阈值。

## 3. 公共字段

| 字段 | 要求 |
| --- | --- |
| `experiment_id` | 唯一、稳定；建议 `<date>-<slug>` |
| `experiment_type` | `confirmatory` 或 `exploratory` |
| `change_scope` | 改动路径/符号；与 diff 限定路径一致 |
| `baseline` | commit、config、checkpoint、已有数字；没有基线就写“无”并说明 |
| `key_changes` | 每条：改了什么、对照是什么、看哪个指标；不许写“优化一下” |
| `metrics` | 指标名、stage、episodes、seeds、采样口径（`--sampled-command` 等） |
| `non_goals` | 本次不解决什么，防止范围膨胀 |
| `human_summary` | ≤5 行，人类检视用；不复制字段 |
| `stop_condition` | 探索性必填；确认性可写“达到 threshold 或预算耗尽” |

确认性还必须：

- `claim`：一句话，可被数据否证；
- `prediction`：方向 + 幅度/区间 + 对照；
- `threshold`：预先登记的通过线，不能事后改。

探索性还必须：

- `uncertainty`：到底不知道什么；
- `prediction`：可证伪的预判（可以是“哪个指标先动/不动”）；
- `stop_condition`：何时停止并报告，而不是无限训练。

## 4. 人类可读模板

```text
experiment_id: 2026-10-19-stair-hold
type: confirmatory
change_scope: robotdog/env.py::stair_guidance_under_hold
baseline: HEAD d1164e3; stair9cm success 0/3
claim: 在 9cm 楼梯启用 hold 后，成功率会上升且摔倒率不升
prediction: stair9cm success +5pp 到 +15pp，fall rate <= baseline
threshold: success >= baseline + 3pp 且 fall <= baseline；否则不支持
metrics: eval stair9cm, 100 episodes, seed 0, --sampled-command
non_goals: 不改课程高度、不改 PPO 超参
stop_condition: 达到 threshold，或 400k steps 后仍无方向
```

## 5. Trace JSON 例子

```json
{
  "ts": "2026-10-19T12:00:00+08:00",
  "layer": "agent",
  "event": "task_book",
  "experiment_id": "2026-10-19-stair-hold",
  "experiment_type": "confirmatory",
  "claim": "在 9cm 楼梯启用 hold 后，成功率会上升且摔倒率不升",
  "prediction": "stair9cm success +5pp 到 +15pp，fall rate <= baseline",
  "threshold": "success >= baseline + 3pp 且 fall <= baseline",
  "baseline": {"commit": "d1164e3", "stair9cm_success": "0/3"},
  "change_scope": ["robotdog/env.py::stair_guidance_under_hold"],
  "key_changes": [
    {"change": "hold 时保留 stair guidance", "control": "同 config 关闭 hold", "metric": "success/fall"}
  ],
  "metrics": {"stage": "stair9cm", "episodes": 100, "seed": 0, "sampling": "sampled-command"},
  "non_goals": ["不改课程高度", "不改 PPO 超参"],
  "human_summary": "验证 hold 是否让 9cm 成功率上升且不增加摔倒；100 回合固定种子。",
  "stop_condition": "达到 threshold 或 400k steps 无方向"
}
```

## 6. 质量检查

- 任务书能否在一屏内读完？不能就压缩 `key_changes`/`metrics`，不要加长文。
- claim 是否可被否证？prediction 是否有方向和对照？threshold 是否预先登记？
- 探索性是否写了 stop_condition？没有就退回重写。
- `change_scope` 是否与 diff 一致？不一致就说明是范围漂移。
- 是否引用了不存在的指标/阶段/命令？C/D/E 会核对。
- 修改任务书必须追加 `task_book_revision` 并说明原因，不覆盖旧事件。
