# 代码审计报告 —— `upstair_core`（Go2 盲走楼梯 · MuJoCo + 残差 PPO）

* 审计对象：`D:\项目\RobortDog\upstair_core`（工作树快照，见下方「快照状态」）
* 审计方式：**只读**。未修改任何被审计文件；仅新建本目录与 `.tmp/zz_audit_probe*.py`（临时探针，`.tmp/` 已被 `.gitignore` 忽略）。
* 审计范围：`robotdog/`（顶层 9 个 .py + `tools/` 4 个 .py）、`scripts/`（6 个 .py）、`config.yaml`、四份根文档、`docs/`（14 份）、`.gitignore`、`requirements.txt`、`artifacts/`、`runs/` 产物形状。
* 判据来源：仓库自带契约文档 `agent.md`（§2 红线 / §3 仓库地图 / §5 硬约束 / §6 约定 / §7 验证流程 / §8 已确认事实 / §9 已知坑），以及 `.dsh/skills/legged-sim-rl-build` 的十二条硬契约。

## 快照状态（重要）

审计开始时 `git status` 已经有 3 个未提交修改；**审计结束时是 15 项**（说明这个仓在本次审计期间仍被继续改动，其中 `agent.md` / `README.md` 的规模表就是在这期间被改对的）：

```
 M README.md                        M docs/robotdog/tools/__init__.md
 M agent.md                        M docs/robotdog/tools/console.md
 M config.yaml                     M docs/robotdog/tools/paths.md
 M docs/robotdog/__init__.md       M experience.md
 M docs/robotdog/config.md         M robotdog/config.py
 M docs/robotdog/env.md            MM robotdog/policy.py      （staged + 再改）
 M docs/robotdog/policy.md         M scripts/play.py
 M train.md                        ?? analysis/               （本次新增，未被忽略）
```

`.tmp/` 里的探针（`zz_probe_*.py`、`zz_audit_*.py`、`audit_*.py`）与 `analysis/` 是本次审计新增的产物；`.tmp/` 在 `.gitignore` 内，`analysis/` **尚未**被忽略（只读约束要求不改 `.gitignore`，所以它会以未跟踪状态出现）。所有结论对应**当时的当前工作树**，不对应 `HEAD`。若要复现本文档里任何「实测」数字，请先确认工作树没有被继续改动（尤其 `scripts/play.py` 的 `:654` 行号锚点）。

## 审计期间的实测规模（写报告时重测；本会话内该文件被继续改动过）

| 文件 | 实测行数 | 备注 |
| --- | --- | --- |
| `robotdog/env.py` | **1066** | `agent.md:68-69` 曾写 895，**审计期间已被同仓改动修成 1066** |
| `scripts/play.py` | **667** | 审计过程中从 658 变到 667；`agent.md` 曾写 462、`README.md` 曾写 482，**已被改成 665**（与 667 仍差 2） |
| `robotdog/config.py` | 443 | 一致 |
| `robotdog/reward.py` | 375 | 一致 |
| `robotdog/policy.py` | 341 | 一致 |
| `robotdog/curriculum.py` | 332 | |
| `robotdog/training.py` | 323 | |
| `scripts/train.py` | 396 | |
| `scripts/teleop.py` | 288 | |
| `scripts/cmd_demo.py` | 192 | |
| `robotdog/gait.py` | 191 | |
| `robotdog/go2_model.py` | 460 | `README.md:455` 也写 460，一致 |
| `robotdog/tools/terrain.py` | 242 | |
| `config.yaml` | 184 | |

> 因此本文档与 `04-training-scripts.md` 里的 `play.py` 行号锚点对应 **667 行**这一版；若该文件再被改动，S-1 的 `:654` 可能漂移（`agent.md §6` 自己也警告过「行号最容易腐坏」）。
>
> 仍未修好的规模类文案：`README.md:512` 的小节标题写着「`env.py` 为什么还有 **895** 行」，与同文档表格里的 1066 自相矛盾。

## 结论概览

整体判断：**这个仓的代码质量高于其文档质量**。奖励数学、观测契约、动作流水线、终止判据、分层依赖这五条核心设计都与其自述契约一致，且都能在源码里逐条对上；主要问题集中在**配置入口的校验不对称**、**一处让默认回放模式必崩的调用错误**、**评估产物不落盘**、**行为回归完全缺位**，以及**一批文档行号/计数已经腐烂**。

> ⚠ **最优先的一条是 S-1（blocker）**：`python scripts/play.py --run <run>`（默认 `--mode gif`）因为 `play.py:654` 把 `device` **函数**当设备名传参，对任何权重都会以 `TypeError` 终止；只有 `--mode eval` 这一条非默认分支能跑通。

严重度口径：`blocker` = 会静默改变训练/评估语义（S-1 是会**直接中断**默认工作流，同级）；`high` = 会误导使用者做出错误决策；`medium` = 明确缺陷但影响面有限；`low` = 卫生问题。

| 编号 | 严重度 | 一句话 | 详见 |
| --- | --- | --- | --- |
| C-1 | **blocker** | `ppo:` / `run:` / `play:` 三段**零键校验**，拼错的超参被静默丢弃（`ent_coeff: 999` 实测无报错、也不生效） | `01-config.md` |
| C-2 | high | `ppo:` 段静默丢键而 `env:`/`reward:` 段硬报错；且 `_apply()` 只 `setattr`、**不做类型校验**（`action_scale: "0.30"` 实测一路活到 `env.step()` 才炸） | `01-config.md` |
| R-1 | high | `reward.py` 的 `compute_reward` docstring 声称「只读状态、不改任何字段」，实际改了 3 个 `env._*` 字段（它是**有状态**的） | `02-reward.md` |
| R-2 | medium | `act_low`/`act_high` 取自 `joint_ranges`（弧度）却被命名得像 `ctrl_ranges`（N·m）；两处用法**都对**，坏在名字会引诱人「修好它」从而破坏动作管线 | `02-reward.md` |
| R-3 | medium | `RewardConfig` 有 8 个「配置里写、代码不读」的字段，其中 `forward_vel_sigma`/`lateral_vel_sigma`/`yaw_rate_sigma`/`probe_lift_ratio`/`probe_lift_max` 会让调参者白改 | `02-reward.md` |
| R-4~R-6 | low | 三条奖励细节：`probe_lift` 的全局峰值 vs 逐足求和、`foot_lift` 目标值的除零路径、`air_time == 0` 时术语键仍存在 | `02-reward.md` |
| E-1 | high | 裸 `Go2StairsEnv(cfg=...)`（**正式路径之外**）会报 `StairSpec.step_height = 0.05`，即「没有楼梯却告诉策略有 5 cm 台阶」，完全静默 | `03-env-obs.md` |
| E-2 | medium | `set_command_now()` 只刷新 `_hist[-1]`，**改不到调用方已持有的 `obs` 数组**（值语义）；`play.py` 补了、`teleop.py`/`cmd_demo.py` 没补 | `03-env-obs.md` |
| E-3 | low | `_obs_frame()` 在一次调用里被算两遍（实测幂等）；`reset()` 把同一帧复制 `history_len` 次 | `03-env-obs.md` |
| E-4 | medium | 「观测维数恒定」这条契约**没有任何机器守卫**（`ObsConfig.base_frame_size` 可写而 `_obs_frame()` 无条件拼前 42 维） | `03-env-obs.md` |
| E-5 | medium | `EnvConfig.reward_overrides` 是**无读者的影子字段**，docstring 却说 `config_to_env` 会填它；真实链路只在 `train.py` | `03-env-obs.md` |
| E-6/E-7 | low | `_start_terrain_height` 死字段；`_fall_reference_height()` 对未知 `terminate_height_ref` 静默兜底（与 `render()` 对未知相机名报错风格不一致） | `03-env-obs.md` |
| **S-1** | **blocker** | **`scripts/play.py:654` 把函数 `device`（`robotdog.config.device`）当设备字符串传进 `open_runner` ⇒ GIF 模式（默认模式）对任何权重都在第一阶段抛 `TypeError`，一个 GIF 都写不出** | `04-training-scripts.md` |
| S-2 | high | `--mode eval` 把统计量算好了却不写盘（payload 只 print），评估**没有机器可读产物** | `04-training-scripts.md` |
| S-3 | high | `scripts/teleop.py` 与 `scripts/cmd_demo.py` 都有 `--config`，但 `main()` 里调用的是 `load_config()`（不传参）—— 该参数**完全无效** | `04-training-scripts.md` |
| S-4 | high | 每级 `VecNormalize` 统计**不随 warm-start 迁移**：`set_parameters` 只搬权重，`restore_vecnormalize` 只在 `load_policy` 里被调用 | `04-training-scripts.md` |
| S-5 | medium | 产物里的「逐级配置」是错的：每级 `_env.json`/`summary.json` 写的是**基线**（或最后一级）⇒ teleop 读到恒为 0 的量程并**静默降级限幅** | `04-training-scripts.md` |
| S-6 | medium | `teleop.py` / `cmd_demo.py` 在 `set_command_now()` 后**不重取观测**，污染 `cmd_demo` 的斜率/相关系数统计 | `04-training-scripts.md` |
| S-7 | medium | `mode_eval` 不传 `command`，于是评估用的是**基线 vx=0.55** 与随机 `vy/wz`，不是该阶段训练时的指令 | `04-training-scripts.md` |
| S-8 | low | 9 条卫生项：`make_env(warm_start=)` 死形参、日志只在回合结束打、`_stage_plan` 的 `"auto"` 用 A 级 checkpoint 的列表配最后一级权重、`cmd_demo` 自谦过头/旧字段名/固定平地、缺统计只告警、GIF 帧全留内存（~1.9 GB 量级） | `04-training-scripts.md` |
| T-1 | high | **没有任何行为断言**：无 `tests/`、无 `test_*.py`；`check_lines.py` 实跑 **退出码 1（4 处空行引用）**，`ref_truth.py` 17 条可疑却仍退出 0 | `05-tooling-and-docs.md` |
| T-2 | medium | 行号锚点仍在腐烂（`env.md:318` 引 `env.py:210` 期望 `_build_model`，实际在 `:303`）；**规模表已被本轮改动修正**（`agent.md` 现写 1066/665） | `05-tooling-and-docs.md` |
| T-3 | medium | 漂移的固定模式是「加 docstring ⇒ 行号整体下移」（`README.md:566`、`experience.md:566` 都记录过同一事故） | `05-tooling-and-docs.md` |
| T-4 | medium | `.tmp/` 下约 500 个一次性脚本/日志事实上是本仓的实验台账，与 `agent.md §9`「重要结论不要留在 `.tmp/`」冲突 | `05-tooling-and-docs.md` |
| T-5 | medium | GIF 落点有**三套互不相同的说法**（`config.py:430` 注释 / `play.py:485` help / `play.py:523+636` 实现） | `05-tooling-and-docs.md` |
| T-6~T-7 | low | `runs/` 里 2 个 run 缺 `model_final.zip` 会被兜底链静默掩盖；`pillow` 是可选依赖但缺它报的是 `ModuleNotFoundError` | `05-tooling-and-docs.md` |
| T-9a | low | 步态先验的复现链路已断（`search_trot.py` 从未入库，`gait.md:375-388` 已如实记载）；风险在 `load_gait()` 只用 `"top" in data` 判定 schema —— 缺该键时先验**静默退化**为 `amp_th=0.0` / 2.00 Hz / 0.70 rad | `05-tooling-and-docs.md` |
| T-9b | low | `GaitParams.normalize()` 静默裁剪越界步态（`freq 99→10`、`amp 5→0.5`），而 `calf_shape=5` 抛的是 `IndexError` 而非 `ValueError` | `05-tooling-and-docs.md` |
| T-9c | medium | `GaitParams.from_dict()` 静默丢弃未知键 ⇒ 步态 JSON 里把 `amp_th` 拼成 `amp_thigh` 时 `amp_th` 静默变 0、`action_scale` 悄悄从 1.0 掉回 0.7，训练照跑 | `05-tooling-and-docs.md` |
| T-9d | low | `StairSpec.height_at()` 的顶层平台分支缺 `n_up > 0` 守卫（`step_blocks()` 有），今天靠「`crest_height == 0`」侥幸一致 | `05-tooling-and-docs.md` |
| T-9e | low | 仓根 `MUJOCO_LOG.TXT` **被 git 跟踪**，内容是 `Nan, Inf or huge value in QACC at DOF 0 … simulation is unstable`（物理发散告警） | `05-tooling-and-docs.md` |
| T-9f | medium | `flat_spec()` **不会**把 `step_height` 归零（只 `setdefault` `n_up`/`n_down`）⇒ `teleop.py:77` 与 `cmd_demo.py:152` 的平地上，观测里恒为「5 cm 台阶」（奖励侧无影响：`max(0.08, 1.4×0.05)=0.08`） | `05-tooling-and-docs.md` |
| T-9g | medium | `StairSpec` 无任何几何校验：`n_down>0` 分支的 crest 块缺 `n_up>0`/`crest_height>0` 守卫、`step_height=0` 会产出零高度 box、`tread=0 & n_down>0` 抛裸 `ZeroDivisionError` —— 全部只在 MuJoCo 编译期以 `RuntimeError` 暴露 | `05-tooling-and-docs.md` |
| T-9h | low | 注释链指向**不存在的** `robotdog.tools.touch` / `touch.riser_graze` / `mj_ray` 前向探针 / `.tmp/stall_probe.py`（结论仍成立，理由虚构） | `05-tooling-and-docs.md` |
| T-9i | low | `extract_meta()` 用 `name.startswith("stair_")` 识别地形 ⇒ 非 `StairSpec` 地形不会被排除出 `trunk_contact`；`floor_geom_id` 找不到时是 `-1` 且不报错 | `05-tooling-and-docs.md` |
| T-9j | low | `console.py:27-29` 静默吞掉 `reconfigure()` 失败（正好废掉它存在的理由）；docstring 漏列两个调用方 | `05-tooling-and-docs.md` |
| — | — | 其余工具层/文档项见 `05-tooling-and-docs.md` 的 T-8 与 T-9 的「已验证无问题」小节 | `05-tooling-and-docs.md` |

> **编号提示**：上表按各分册自己的编号给出，未做全局重编。`C-1` 与 `S-1` 是两条独立的 blocker；`E-5` 与更早汇总里的 `C-4` 是**同一条**（`reward_overrides` 影子字段），以 `E-5` 为准；`C-3` 与 `R-3` 同理，以 `R-3` 为准。

**没有发现问题的地方（同样是结论）**：奖励纯函数与 `env.py` 的转发契约、`reward == sum(info["reward_terms"].values())` 恒等式、48 维观测的逐项真机出处、`base_height` 的 `ValueError` 硬门、动作流水线唯一性、终止判据只用高度/倾角/超时、命令跟踪项的门控、`probe_lift` 的势函数塑形、工具层不 import `robotdog.env`。逐条证据见各分册的「已验证无问题」小节。

## 分册

| 文件 | 覆盖 |
| --- | --- |
| `00-index.md` | 本文件：口径、快照、结论概览 |
| `01-config.md` | `robotdog/config.py`、`config.yaml`、课程阶段表生成 |
| `02-reward.md` | `robotdog/reward.py`、奖励项与 `RewardConfig` 字段面 |
| `03-env-obs.md` | `robotdog/env.py`、`gait.py`、`go2_model.py`、观测/动作/终止/指令 |
| `04-training-scripts.md` | `robotdog/training.py`、`policy.py`、`curriculum.py`、`scripts/*.py` |
| `05-tooling-and-docs.md` | `robotdog/tools/*`、`agent.md`/`README.md`/`docs/`、`.tmp/`、`.gitignore`、`requirements.txt`、回归门 |

## 复现方式

本次审计用到的探针都在 `.tmp/`（已被 `.gitignore` 忽略）：

```powershell
$env:PYTHONIOENCODING = "utf-8"          # 否则 ref_truth.py 会因 ⭐ 抛 UnicodeEncodeError
.\.venv\Scripts\python.exe .tmp\zz_audit_probe.py     # config_to_* 的键集合/类型实验
.\.venv\Scripts\python.exe .tmp\zz_audit_probe2.py    # env/reward/obs 契约核对
.\.venv\Scripts\python.exe .tmp\zz_audit_stage_cfg.py # 逐级 EnvConfig 真值表（S-5）
.\.venv\Scripts\python.exe .tmp\zz_probe_playdev.py   # 复现 S-1 的 TypeError
.\.venv\Scripts\python.exe .tmp\zz_probe_gait_t9.py   # 先验加载三分支的实际值（T-9a/c）
.\.venv\Scripts\python.exe .tmp\zz_probe_go2_t9.py    # 场景编译 + group 分布 + StairSpec 剖面
```

这些探针只做 `import` + 读常量 + 内存实验，不改动任何被审计文件。回归门（`check_lines.py` / `ref_truth.py` / `check_doc_code.py` / `check_xrefs.py` / `audit_docs.py`）在 `.tmp/` 与 `.dsh/skills/legged-sim-rl-build/assets/gates/` 各有一份，实跑结果见 `05-tooling-and-docs.md` 的 T-1。
