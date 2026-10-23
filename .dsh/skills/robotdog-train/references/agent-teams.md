# 与 dsh-agent-teams 的结合

结论：**可以结合，但默认关闭、按 campaign 启用**。单变更/单臂优先用 `subagent`；
AgentTeams 只在用户明确要求，或一次 campaign 有 ≥2 个改动/≥2 个训练臂时使用。

## 1. 为什么默认不用

- **fresh 隔离冲突**：E/H 必须 fresh context、不能复用上一轮判断；AgentTeams member 是 durable 的，
  容易把上一任务的上下文带进来。团队模式下 E/H 仍应优先由 captain 派 fresh `subagent`。
- **quality review 契约不匹配**：插件的 `kind=review` 需要 `reviewedTaskId` 指向
  implementation/repair 任务；本 skill 的实现者是 captain（主代理），不是 member。硬套会让 captain
  拥有实现任务，和“主代理改代码”的分工冲突。
- **staged approval 往返**：`agent_teams_create(approval="required")` 要等 Web Approve & Run；
  单次小改动用它不划算。用户已授权训练且不要求先看计划时，不要用 `required`，改用
  `automatic`，否则会变成强制人工确认。
- **长训练所有权**：训练 job 必须由主代理持有并做事件响应；member 不适合持有长任务。
- **插件并存**：本机同时有 @nanmicoder/dsh-agent-teams 与 experimental 团队插件；
  只用 `agent_teams_*`，不混用 `spawn_teammate`/`team_task_*`。

## 2. 模式 A：轻量协作（推荐首个团队用法）

适用：一次 campaign 有多个改动或多个臂，但代码仍由主代理改。

- captain = 主代理；member 只做只读/独立任务：
  - C 测试设计（work 任务，只读）
  - D Trace/审查标准设计（work 任务，只读）
  - E 评审（work 任务，只读；或仍用 fresh subagent）
  - 臂运行者（每个 run 一个 work 任务；write scope 只限自己的 `runs/<run>`）
- 任务图：`B -> C/D -> E -> L0–L5`；训练臂依赖 L0–L4 PASS + `gate_result`。
- 主代理在任务描述里贴任务书、diff 路径、输出 schema；不要只写“评审一下”。
- 团队创建：`agent_teams_create({name, description, approval})`。用户已授权训练且不要求先看团队
  计划时用 `approval="automatic"`；用户要求先看计划时用 `approval="required"`，等 Web Approve & Run。
  **绝不自己 approve**。
- E 的结论仍是 captain 写 Trace；member 不写 `.dsh`/Trace 文件。

## 3. 模式 B：完整质量（仅用户明确接受）

适用：用户要求完整质量门禁，且接受“implementer member 改代码”偏离主代理分工。

- 任务链：requirements → implementation → verification → review → integration；
- implementation 的 `inScope`/`verify` 必须来自任务书与 C 的设计；
- review 用插件的 quality review（`reviewedTaskId` 指向 implementation）；
- 失败自动 repair + 下一轮 review；不重造 DAG；
- captain 只在用户要求时 takeover；不得自审自过。

模式 B 不改变本 skill 的 fresh 视觉纪律：H 仍由 captain 派 fresh subagent。

## 4. 启用清单

- [ ] 用户明确要求 AgentTeams，或 campaign 有 ≥2 改动/≥2 臂
- [ ] 只用 `agent_teams_*`，不混用另一套团队工具
- [ ] 审批：用户要先看计划时 `approval="required"` 等 Web；已授权且不要求人工确认时
      `approval="automatic"`；任何情况都不自己 approve
- [ ] 任务有 writeScopes/`inScope` 且互不重叠；只读任务为空
- [ ] `.agent-teams/` 加入 `.gitignore`（如果会落盘）
- [ ] 训练臂 ≤2；每个臂独立 `runs/<run>`；主代理持有训练 job
- [ ] 训练启动前已展示任务书 / L0–L5 / Trace 摘要；不把“展示”变成强制人工审批
- [ ] E/H 仍 fresh；不复用 member 的旧上下文
- [ ] 任务结束删除/归档团队；**新任务开始先清理旧团队与多余 member**
- [ ] 汇报里说明用了哪种模式、哪些任务由 member 做、哪些仍由 captain 做

## 5. 新任务清理

- 开始前查看 `agent_teams_status` / `list_agents`；
- 归档/删除旧团队；移除不需要的 member；不复用旧 subagent/team ID；
- 旧 run、旧 Trace、旧 checkpoint 不删除（保留证据），但新任务用新 `runs/<run>` 与新
  `experiment_id`；
- 只保留本任务需要的 C/D/E/H；单变更时不要为了“用插件”而拉团队。

## 6. 不推荐的做法

- 用 AgentTeams member 代替 fresh E/H（上下文污染）；
- captain 自己 approve 自己创建的 staged plan；
- 把长训练 job 交给 member 持有；
- 为了并行而开 >2 臂；
- 让 member 写 `.dsh`/Trace/`train.md` 与主代理抢写；
- 在团队里复用上一任务的 reviewer/monitor。
