# `scripts/train.py` — 训练入口

对应代码：`scripts/train.py`。训练结论与调参教训见 `experience.md` §4、§17、§21；
本文只讲抽象结构。

## 职责

按 `config.yaml` 的课程串行训练。CLI 只认 `--run-name` 与 `--config`，其余全部
来自配置文件。

## 核心抽象

### 逐级循环

```
读配置 → 生成阶段表 → 建 RunDir（已存在则拒绝启动）
→ for stage: make_env(stage) → train_stage(stage) → 记录结果 → 末级另存 model_final
```

每一级一个**全新的向量环境**（地形烘焙在编译后的 MJCF 里，换地形 = 重编译，
不能在 rollout 中间做）。

### 每级的环境下发（`Trainer.make_env`）

三条纪律，每条都对应一类静默出错的方式：

1. **每级深拷贝一份 `EnvConfig`**：下发给子环境的设置全是就地改写
   （`set_commands`、`set_reward_overrides`），直接传共享 cfg 会让设置跨级累积、
   顺序敏感。
2. **奖励覆盖「从基线叠」**：先显式抄回 `config.yaml` 的基线，再叠本级的键 ——
   否则上一级覆盖过、这一级没覆盖的键会残留上一级的值。
3. **指令量程无条件下发**：量程为 0 是「这一级不练这个方向」的有意义设置，
   `if stage.command_vy_max:` 之类的守卫会让该级继承上一级的范围。

### 调子环境方法一律走关键字（`kw_call`）

`venv.env_method` 的位置实参按形参顺序绑定 —— 「参数名对不上」要变成
`TypeError` 而不是静默传错值。

### 提前达标 / 起点选取（`scripts/train.py`）

两个**默认关闭**的可选机制（打开则日志与产物里都有记录）：

- **提前达标**（`--promote-rate/--promote-min-steps/--promote-min-episodes`）：滑窗成功率
  ≥ `promote_rate` 且样本量够时，这一级提前结束并跳到下一级。结果写进
  `summary.json` 的 `results[].extras`（trained_steps/promoted/budget_steps），
  并追加 `runs/<run>/promotions.json`。⚠ `trained_steps` 是 rollout 粒度，会**略超**
  预算（如 300→512、600→1024，因为 n_steps×n_envs），不是 bug；`summary.total_timesteps`
  记的是预算和。
- **起点选取**（`--select-warm-start --select-episodes N`）：每级开训前，把本 run 上一级
  的后段检查点（`<阶段>_*_steps.zip` 中步数 ≥ 0.6×预算的，外加 `_final`）逐个用
  `replay_success_rate()` 回放 N 回合，按（成功率、−平均步数）排序取最好的一份作为下一级
  warm-start。打印 `[warm-select] …` 候选行与「选用 … 作为下一级起点」。
  **历史链就是这么做的**（`train.md` 里 crawl 延伸链的「深扫最佳」），而课程配置的
  单 run 默认固定传 `_final` —— 这是「课程配方 ≠ 延伸链配方」差异来源之一
  （`experience.md` §23.76）。
  ⚠ 只对「有下一级」的级生效（1-based 序号 `1..n-1`；`robotdog.training.warm_select_stages`）：
  最后一级之后没有下一级、**不回放也不写 `warm_select`**；**单级配置恒不触发**
  （单级 LR A/B 必须显式 `--warm-start`，不要指望它自己选）。
  Trace 的口径：`episodes` 字段写的是回放回合数（历史链用 100，本轮的 chain_repl 用 40，
  对比时要注明）；控制台 sps 用 `budget/wall`，Trace 的 `sps` 用 `trained/wall`，两处不同口径。

### 回放评估助手（`replay_success_rate`）

`robotdog/training.py::replay_success_rate`：加载 checkpoint（含 `_env.json` 配置与观测统计）、
确定性回放、**不钉指令**（与训练/`--sampled-command` 同分布），返回
`{episodes, success_rate, fall_rate, mean_steps, mean_distance}`。它是 40/100 回合
「选点/门禁/对比」的统一入口 —— 不要在脚本里自己搭 env 循环。内部用 `load_policy`，
结束时会同时 `close()` 自建 env 与模型自带的 VecEnv（否则每候选泄漏一个向量环境，
长链选取会 OOM，并被包成「MJCF 编译失败」）。

### 产物三件套

每级结束存 `<阶段>_final.zip` + `<阶段>_final_vecnormalize.pkl` +
`<阶段>_final_env.json`；中间检查点（`CheckpointCallback`）**同样带 `_env.json`**
—— 它是唯一能把「这个权重当时跑在什么配置下」带出来的东西，且从子环境读回的是
**这一级实际生效**的配置（已叠好量程与覆盖），不是基线。训练另写
`plan.json`：每一级实际下发的量程与奖励覆盖的实施记录。

### 结构化 Trace（`runs/<run>/trace.jsonl`）

每行一个 JSON 事件，`layer` 字段区分来源：`rollout`（`scripts/train.py` 写进
`runs/<run>/trace.jsonl` 的 `stage_start` / `stage_end` / `warm_select`）与 `agent`
（方案 `design`、任务书 `task_book`、监控告警 `alert`，落在 `.tmp/trace/*.jsonl`）。
`stage_end` 字段包括
`step_height/tread/learning_rate/success_rate/fall_rate/mean_return/mean_length/`、
`trained_steps/budget_steps/promoted/sps/wall_s/checkpoint`。
⚠ Trace **不含逐级 `reward_overrides` 全文** ⇒ 复盘要「Trace + `plan.json`」一起看：
Trace 回答「哪一级、训了多久、为什么提前达标、起点选了谁」，配方全文在 `plan.json`。
Trace 写失败静默（不能拖垮训练）。

## 并发与内存（`SubprocVecEnv`）

训练（`n_envs > 1`，默认 8）用 **`SubprocVecEnv`**：每个 run 会派生 `n_envs` 个
python 子进程（Windows spawn，每个约 0.4–0.6 GB 常驻，含 MuJoCo 模型）。
**并发上限按内存算，与 CPU 核数无关**：15.67 GB 的机器上实测**只能串行 1 个 8-env
run**；两个 8-env run 并发 = 16 个 worker ≈ 8–10 GB，叠加系统/编辑器后直接
`MemoryError` 或 `OSError: [WinError 1114] ... torch/lib/shm.dll`（后者是内存耗尽
的伪装，不要当成 torch 安装问题）。评估/回放（`n_envs == 1`）走 `DummyVecEnv`，
内存小、可并发。

发车脚本应有**资源闸门**（示例 `.tmp/run_stance_lr_chain.py::wait_resources`）：
发下一个 run 前等「可用物理内存 ≥ 6 GB 且没有其它 `scripts/train.py` 进程」，
否则会重演 2026-10-21 的 OOM 事故（4 个 8-env run 同时跑，全部被系统杀死，
`runs/lr2e4`、`runs/lr9e5` 只剩 `trace.jsonl`）。

## 踩坑指引

用另一套配置回放权重会得到相反的评测结论 —— 见 `experience.md` §20.7。
