# 训练监控、事件响应与快照审查

本文件是 robotdog-train 阶段 G/H 的操作手册。视觉 rubric 在 skill `visual-monitor`，
这里只写启动、解析、事件响应、渲染、评估、Trace 与落账。

## 1. 启动前

- [ ] `gate_result.release_decision` 为 `go` 或 `go_with_risk`；L0–L4 无 BLOCK
- [ ] L5 若 SUSPECT，已有 `l5_review` 记录
- [ ] run 名唯一；`runs/<run>` 不存在（`train.py` 会拒绝已存在目录）
- [ ] 已记录 `git rev-parse HEAD`、`git status --short`、config、warm-start/norm
- [ ] 任务书、`test_design`、`trace_design` 已写入 Trace
- [ ] 已展示启动前摘要（任务书 / L0–L5 / Trace），放行决定为 `go` 或 `go_with_risk`
- [ ] `.tmp/review_play.yaml` 已就绪（从本次训练使用的 config 复制后只改
      `play.episodes_video: 1`、`play.video_max_frames: 500`；保留
      `play.stages: auto`、`play.episodes_eval: 100`、`play.batch: ""`）
- [ ] 磁盘/内存/并发 OK：训练 + 渲染合计 ≤2 臂

生成评审配置：

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
Copy-Item <cfg> .tmp\review_play.yaml   # <cfg> = config.yaml 或 ablations/*.yaml
# 只改 play.episodes_video: 1 与 play.video_max_frames: 500；
# 不要改 curriculum/train 段，渲染只读 play 与地形配置。
```

## 2. 启动

启动前先按 `references/presentation.md` 展示任务书 / L0–L5 / Trace 摘要；未展示不得启动。

```powershell
$env:PYTHONUNBUFFERED = "1"
.venv\Scripts\python.exe scripts\train.py --run-name <name> [--config <cfg>] [--warm-start <zip>] [--warm-norm auto|on|off]
```

用受管后台 job 启动，拿到 `job_id`；不要用前台超时命令跑长训练。
启动后立刻追加 `train_start` 到 `runs/<run>/trace.jsonl`（commit、config、
warm-start/norm、命令、job_id），并在 `train.md` 建条目骨架。

## 3. 解析与状态

每 10–15 分钟或收到完成通知时读一次 job 输出；不要忙轮询。

正则（按实际输出微调）：

```text
^\[课程\] >>> 第 (\d+)/(\d+) 级
^\[课程\] <<< '([^']+)' 成功率=([\d.]+) 摔倒率=([\d.]+) 均回报=([-\d.]+) 用时=([\d.]+)
```

状态文件 `.tmp/train_state/<run>.json`（scratch，不是最终记录）：

```json
{
  "run": "<run>",
  "experiment_id": "<id>",
  "config": "<cfg>",
  "commit": "<sha>",
  "started_at": "<iso>",
  "job_id": "<id>",
  "sps_baseline": null,
  "stages": [
    {"index": 1, "name": "...", "status": "done", "success_rate": 0.0,
     "fall_rate": 0.0, "mean_reward": 0.0, "duration_s": 0.0}
  ],
  "alerts": [],
  "renders": [],
  "snapshots": []
}
```

每级结束追加 `stage_end` 到 run Trace；不要只更新状态文件。

## 4. 事件规则（主代理门控驱动）

| 事件 | 触发 | 响应 |
| --- | --- | --- |
| stage_end | `[课程] <<<` | 解析四项指标；对照任务书红线；追加 stage_end |
| nan/traceback/OOM | 日志/异常 | 暂停判断、保留最后 checkpoint、渲染+H、记录 abort；不自动杀 |
| success_low | ≥50% 步数预算后成功率 <10% 且摔倒率 >60% | 告警/暂停；先查证据，不盲目加步数 |
| sps_drop | SPS 相对首级基线下降 >20% | 推迟中途渲染，结束后补 |
| checkpoint_ready | 阶段末 | 中途渲染 ≤2 阶段（语义变化 + 最后完成） |
| l5_suspect | 放行后早期目标未移动 | 记录、触发独立 L5/风险审查；不静默继续 |
| user_stop | 用户要求 | 保留 checkpoint、渲染+H、记 [中止] |

- 主代理持有 job 与响应权；H 只做快照审查，不控制训练。
- 所有事件追加到 `runs/<run>/trace.jsonl`；不要只靠记忆或聊天记录。
- 数值告警和视觉冲突都要能回到 Trace 证据。

## 5. 中途渲染与 H 快照

从 checkpoint zip 出发，必须显式 `--videos-dir`，否则视频落 `output/videos`：

```powershell
.venv\Scripts\python.exe scripts\play.py --run runs\<run>\checkpoints\<stage>_final.zip --mode video --stage <stage> --video-scale 1 --config .tmp\review_play.yaml --batch <batch> --videos-dir runs\<run>\videos
```

- stage 用当前课程可解析的阶段名；不在 config 时用高度 cm 试。
- 中途视觉默认 ≤2 阶段：语义变化阶段 + 最后完成阶段；更多先问用户。
- 渲染后粗切片：

```powershell
.venv\Scripts\python.exe scripts\frames.py runs\<run>\videos\<file>.mp4 .tmp\review\<run>\<stage>\coarse --coarse 12
```

- 把帧 + Trace 片段 + 任务书交给 **fresh** H，提示词取自 `references/snapshot-review.md`；
  每集 ≤6 张图；需要细看才用 `--window`。
- H 结果追加 `snapshot_review`；主代理定根因。

## 6. 训练结束

- 确认 `runs/<run>/checkpoints/<stage>_final.zip`、`_vecnormalize.pkl`、`_env.json`；
  全部跑完才有 `plan.json`/`summary.json`/`model_final.zip`。
- 数值评估：

```powershell
.venv\Scripts\python.exe scripts\play.py --run <run> --mode eval --stage <stage> --sampled-command
```

`episodes_eval` 默认 100、不得 <40；能力带结论要 3 seeds；单 run 标 n=1。
- 最终阶段渲染：run 已结束可用 `--run <run>`；中途只有 checkpoint 时用 zip。
- 最终阶段必做 H；语义变化阶段必做；两臂比较各派独立 H。
- 汇总 Trace：`runs/<run>/trace_summary.json` + `trace_report.md`。
- 落 `train.md`：改动 / 结果 / 解读 / 产物与 GIF / 去向；补 run、命令、commit、config、
  分级表、eval、视觉结论、中止原因；结果三件套（成本/成功率/key_changes 影响）。
- 确认 mp4/帧图在 `runs/<run>/videos/` 和 `.tmp/review/<run>/`，不在 `output/videos`。

## 7. 中止/恢复

中止后：读完整 job 输出；保留最后 checkpoint；渲染+H；`train.md` 记 `[中止]` 和原因；
`不要静默重跑`。要重跑先问用户，并复用/修订任务书与 Trace，不覆盖旧 run。

上下文压缩/断线后：先读 `runs/<run>/trace.jsonl`、`.tmp/train_state/<run>.json` 和
`runs/<run>/` 目录，以 checkpoint/plan/summary/train.md 为准；不要凭记忆重跑。
