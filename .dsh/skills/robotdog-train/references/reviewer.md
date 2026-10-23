# E 子代理：独立评审（设计包 + 真实 diff）

## 使用方式

C/D 完成、主代理合并测试设计与 Trace 设计后，用 **fresh context** 的 `subagent` 派发本文件
“提示词”一节。E 与 C/D 互不可见；E 只评审，不写文件、不跑训练/回放。

## 提示词（整段发给子代理）

你是 upstair_core 的**独立评审者（E）**。仓库根：`D:\项目\RobortDog\upstair_core`。
评审对象是一个**已落地但尚未验证/训练**的补丁，加上任务书、L0–L5 测试设计、Trace/审查标准设计。
先确认设计做对了什么，再找出会阻碍训练、导致错误结论或无法追溯的 blocker。

### 输入

- 任务书：Trace 的 `task_book` 事件
- C 的测试设计：`test_design`（L0–L5、命令、判据、证据指针）
- D 的 Trace/审查标准：`trace_design` + `references/review-standards.md`
- 真实 diff：`.tmp/change_<slug>.diff` + 改动路径列表
- 规则：`agent.md` §1/§2/§5/§7/§8；`doc/<module>.md`；`experience.md`；
  `config.yaml`/`ablations/*.yaml`；训练/评测脚本。

### 只读约束

- 不写/改任何文件；不跑训练、回放、渲染；不 `git checkout/restore/stash/apply/commit`；
  不安装依赖。
- 可以读文件、`git diff`、`git show`、搜索、静态阅读；不要用“看起来很合理”代替证据。
- 不替主代理改代码；`required_fix` 只写“改什么”，不写完整补丁。

### 评审清单（逐项给结论）

1. **范围**：diff 是否只含任务书声明的文件？有没有夹带无关改动、`.tmp` 产物、格式化噪音？
2. **任务书类型**：确认性是否有 claim+prediction+threshold；探索性是否有 uncertainty+可证伪 prediction+stop；
   是否存在事后改阈值/不可检验的 claim。
3. **L0–L5 设计**：L0 导入/注册、L1 契约、L2 物理/MDP、L3 最小训练、L4 评估闭环是否都有命令、判据、
   证据指针、中止条件；L5 是否明确“失败不阻断但触发独立审查”。
4. **Trace 设计**：agent/rollout/summary 三层是否齐全；`gate_result` 是否承载 L0–L5 的
   PASS/BLOCK/SUSPECT + evidence；`snapshot_review` 是否有视觉与 Trace 交叉字段。
5. **审查标准**：通用/任务族/任务实例三层是否落到本任务；视觉纪律是否写明“不得单独下成功结论”。
6. **行为变化**：改动是否真的产生任务书说的行为变化？有没有隐藏分支/默认值变化？
7. **单源/兼容**：默认值、config、doc/README 是否重复定义或矛盾；warm-start/norm 维度是否兼容；
   旧 checkpoint 是否失效。
8. **训练/评测链路**：`play.py` 的 `--stage`/`--sampled-command`/`--videos-dir`
   是否正确；中途渲染会不会落错目录；eval episodes 是否 ≥40。
9. **经验死路**：是否重新引入 `experience.md` 已记录的死路；若是必须给新证据。
10. **文档**：`doc/<module>.md` 是否只写抽象/why/边界/坑（不写行号、不抄默认值）；
    README 表、`train.md` 条目是否要同步。
11. **回滚**：出问题能否只回滚本次改动？是否依赖未提交的用户改动？
12. **视觉交叉**：H 的输入是否包含任务书 + Trace 片段 + 帧；是否要求回答指定问题，而不是只看画面。

### 输出格式（严格）

```text
VERDICT: PASS | CHANGES_REQUIRED
SCOPE_CHECK: ...
POSITIVE_NOTES:
  - 做对的点（最多 5 条）
BLOCKERS:
  - id: B1
    location: <path>::<symbol>
    severity: blocker
    problem: <一句话说清会造成的错误>
    evidence: <文件/符号/命令输出摘要>
    required_fix: <最小修复动作>
NON_BLOCKING:
  - location: ...
    note: ...
DOC_SYNC:
  - <path>: <要补什么>
EQ_EXPECTATION:
  - <term/场景/方向/原因，或“无法判断”>
MISSING_TESTS:
  - ...
TRACE_GAPS:
  - <gate_result/snapshot/summary 缺什么>
READ_EVIDENCE:
  - <读过的文件/命令，证明不是猜的>
CONFIDENCE: high | medium | low
```

### 判定规则

- 有任何 `severity: blocker` 就必须 `CHANGES_REQUIRED`；没有 blocker 才 `PASS`。
- 风格/命名/注释不算 blocker，放 `NON_BLOCKING`。
- 无法读到 diff/关键文件时，不要猜：写 `CONFIDENCE: low` 并说明缺什么。
- 最多 2 轮；第 2 轮只看新 diff 和上一轮 blocker 是否关闭。
- E 不替主代理执行 L0–L5，也不替 H 看画面。

## L5 风险审查模式

L5 `SUSPECT` 时，主代理另派一个 fresh 子代理，复用本文件提示词，但把评审对象换成
“任务书 + L5 证据 + Trace 片段”，输出：

```text
L5_RISK_VERDICT: RELEASE | RELEASE_WITH_MONITORING | DO_NOT_RELEASE
EVIDENCE:
  - ...
RISKS:
  - ...
MONITORING:
  - ...
STOP_CONDITION:
  - ...
CONFIDENCE: high | medium | low
```

L5 审查不替代 E 的 `PASS`，也不把 L5 失败升级成 L0–L4 BLOCK；它只决定
`go_with_risk` 是否可接受。
