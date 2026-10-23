# 启动前展示与名词规范

本文件规定**面向用户**的展示：用什么名字、展示什么、什么时候展示、是否等待确认。
Trace JSON 内部仍用英文字段；人读摘要必须用中文全名，代号只放在括号里。

## 1. 四条硬规则

1. **中文全名优先**：阶段、测试级别、结果词、放行决定都用中文；代号只在括号里补充，
   例如“最小训练循环检查（L3）”“带风险放行（go_with_risk）”。
2. **配置写用途 + 路径**：不写裸 `level0`/`crawl24`/`stair9cm`；写成
   “消融 level1（加 privileged foot contact，`ablations/level1.yaml`）”。
3. **启动前先展示三块**：任务书、L0–L5 测试设计、Trace 设计；自动开始也先展示。
4. **不强制人工确认**：展示后可直接启动；只有用户明确说“先给我看/等我确认”才等待。

## 2. 名词表

### 2.1 阶段

| 代号 | 展示用中文 |
| --- | --- |
| A | 方案设计 |
| B | 任务书 |
| C | 冒烟与环境测试设计 |
| D | Trace 与审查标准设计 |
| E | 独立评审 |
| L0–L5 | 分级门控检查 |
| F | 仿真训练 |
| G | 门控驱动监控 |
| H | 视觉与 Trace 快照审查 |
| I | 结果汇总 |

### 2.2 测试级别

| 代号 | 展示用中文 |
| --- | --- |
| L0 | 导入与注册检查 |
| L1 | 接口契约检查 |
| L2 | 物理与 MDP 检查 |
| L3 | 最小训练循环检查 |
| L4 | 评估策略闭环检查 |
| L5 | 目标语义检查 |

### 2.3 状态与决定

| 代号 | 展示用中文 |
| --- | --- |
| PASS | 通过 |
| BLOCK | 阻断 |
| SUSPECT | 存疑 |
| CONSISTENT | 一致 |
| CONFLICT | 冲突 |
| INSUFFICIENT | 证据不足 |
| go | 放行 |
| go_with_risk | 带风险放行 |
| fix_first | 先修复 |

### 2.4 Trace 字段

| 代号 | 展示用中文 |
| --- | --- |
| task_book | 任务书字段 |
| task_book_revision | 任务书修订 |
| test_design | 测试设计字段 |
| trace_design | Trace 设计字段 |
| gate_result | 门控结果 |
| release_decision | 放行决定 |
| l5_review | 目标语义风险审查 |
| snapshot_review | 快照审查结果 |
| trace.jsonl | 实验记录（Trace 事件流） |

### 2.5 本仓训练配置别名

| 路径 | 展示时写什么 |
| --- | --- |
| `config.yaml` | 主训练配置（当前课程与奖励默认值） |
| `ablations/crawl24.yaml` | 历史还原基准（crawl24 原始配方，无新特性） |
| `ablations/level0.yaml` | 消融基线（与 crawl24 同内容） |
| `ablations/level1.yaml` | 消融 level1（加 privileged foot contact；观测 53→57 维，actor 53 / critic 57） |
| `ablations/level2.yaml` | 消融 level2（开启接触奖励组 contact_rewards，当前只启用 foot_scuff = -8.0；其余接触项默认关闭） |
| `ablations/level3.yaml` | 消融 level3（两级课程采样器 10.5–15.0 cm，逐级 timesteps/LR/指令区间/reward_overrides；接触奖励同 level2） |

阶段名（如 `stair9cm`、`flat`）展示时写“名称 + 地形/高度”，例如
“9cm 楼梯阶段（stair9cm）”；不要只写阶段代号。

## 3. 启动前摘要模板（≤ 25 行）

```text
启动前摘要
任务书
- 实验：<experiment_id> / <确认性 | 探索性>
- 目标：<claim 或 uncertainty>
- 预测：<prediction>
- 通过线/停止：<threshold 或 stop_condition>
- 基线/指标：<baseline；stage、episodes、seeds、--sampled-command>
- 改动：<key_changes 每条一行，带用途/对照>

L0–L5 测试设计
- 导入与注册检查（L0）：<检查> → <通过/阻断/存疑>（<证据指针>）
- 接口契约检查（L1）：<检查> → <状态>（<证据指针>）
- 物理与 MDP 检查（L2）：<检查> → <状态>（<证据指针>）
- 最小训练循环检查（L3）：<检查> → <状态>（<证据指针>）
- 评估策略闭环检查（L4）：<检查> → <状态>（<证据指针>）
- 目标语义检查（L5）：<方向/阈值> → <状态>（<证据指针>）

Trace 设计
- 事件层：<要记录的关键事件>
- 回合层：<stage_end/eval/checkpoint/video>
- 汇总层：<trace_summary.json / trace_report.md 的字段>
- 门控结果：<levels / l5_review_required / release_decision>
- 快照审查：<指定问题>；文件：<trace 路径>
放行决定：<放行 | 带风险放行 | 先修复>；<风险/阻断原因>
<已自动开始训练，run=... | 等待用户要求确认 | 不启动>
```

## 4. 自动开始与人工确认

- 用户任务已明确授权训练，且 `gate_result.release_decision` 为 `go` 或 `go_with_risk`：
  展示摘要后**直接启动**，不要问“是否开始”。
- 用户明确说“先给我看”“等我确认”“不要自动跑”：展示后停止，等回复。
- 用户任务只是改代码/改配置，没有授权训练：**不启动**，只展示 L0–L5 结果与建议。
- `fix_first`：展示阻断项和修复建议，不启动；修完重跑对应级别后重新展示。
- `go_with_risk`：必须同时展示 L5 风险、监控项、停止条件；用户已授权则自动开始，
  用户要求先看则等待。
- 自动开始后在摘要末尾写“已自动开始训练，run=<name>”，并给出 job/日志入口；
  不要只写“训练已启动”而不给 run 名。

## 5. 任务书摘要怎么写

- 确认性：`claim` 一句话；`prediction` 写方向 + 幅度/区间 + 对照；`threshold` 写预先登记的通过线；
  再补 baseline 与 metrics。
- 探索性：`uncertainty` 写不知道什么；`prediction` 写可证伪的预判；`stop_condition` 写何时停；
  再补 baseline 与 metrics。
- 一行一个要点；不要把 Trace JSON 整段贴出来。
- `key_changes` 每条写“改了什么 + 对照是什么 + 看哪个指标”，不要写“优化一下”。

## 6. L0–L5 摘要怎么写

- 每级一行：中文名（`Lx`）→ 检查内容 → 命令/判据 → 当前状态（通过/阻断/存疑）→ 证据指针。
- 状态用中文，代号放括号；例如“通过（PASS）”。
- 证据指针写文件::符号、命令、输出字段或产物路径；不要只写“已检查”。
- L5 为存疑时，多写一行“目标语义风险审查：<放行建议/风险/停止条件>”。
- 不把整张 L0–L5 JSON 表贴出来；完整字段留在 Trace。

## 7. Trace 设计摘要怎么写

- 三层各一行：事件层、回合层、汇总层。
- 门控结果一行：`levels`、`l5_review_required`、`release_decision`。
- 快照审查一行：指定问题 + Trace 文件路径；不要只写“看视频”。
- 不粘贴完整 schema；需要细节时链接 `references/trace-design.md`。

## 8. 正例与反例

| 反例 | 正例 |
| --- | --- |
| `L3 PASS` | 最小训练循环检查（L3）：2000 步/级，9 个课程标记齐全 → 通过（PASS） |
| `level1` | 消融 level1（加 privileged foot contact，`ablations/level1.yaml`） |
| `go_with_risk` | 带风险放行（`go_with_risk`） |
| `D 设计的 trace` | Trace 与审查标准设计（D）给出的三层实验记录 |
| `E 说不行` | 独立评审（E）返回 2 个阻断项：... |
| 直接问“是否开始训练？” | 已授权时直接写“已自动开始训练，run=...” |

## 9. 展示检查清单

- [ ] 三块齐全：任务书、L0–L5 测试设计、Trace 设计
- [ ] 总长 ≤ 25 行；一屏可读完
- [ ] 阶段/级别/结果/决定都有中文全名，代号只在括号里
- [ ] 训练配置写“用途 + 路径”，没有裸 `level0`/`crawl24`
- [ ] 每级 L0–L5 有状态与证据指针
- [ ] `fix_first` 不启动；`go_with_risk` 展示风险/监控/停止条件
- [ ] 已授权且放行时，展示后直接启动，不强制人工确认
- [ ] 自动开始给出 run 名与日志入口

