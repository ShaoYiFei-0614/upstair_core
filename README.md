# Go2 上下楼梯强化学习 · 最小核心仓

在一段 8 级、踏面 50 cm 的楼梯上（`start_x = 1.50`，顶层平台 0.88 m）训练一只
Unitree Go2 上下楼梯。**动作空间是「trot 步态先验 + PPO 残差」，不是从零学步态**：
策略只需要学「怎么改」先验，不需要学怎么抬腿。

本文件只讲**环境配置**与**具体代码实现**。配套两份：

* [`experience.md`](experience.md) —— 踩过的坑、错过的判据、以及为什么默认值是现在这个值。
* [`train.md`](train.md) —— 训练过程、实测复现结论、能力边界。

---

## 1. 环境

* **Python 3.10+**（实测 3.13.0）
* **CPU 就够**。8 个环境串行（`DummyVecEnv`，单进程）实测约 **250~420 向量步/秒**，
  视并行负载而定
* Windows 10/11 实测；代码里没有平台相关分支

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt ^
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple ^
    --extra-index-url https://download.pytorch.org/whl/cpu
```

`torch==2.14.1+cpu` 的 `+cpu` 后缀来自 PyTorch 官方 CPU 源，所以命令里**必须**带
`--extra-index-url https://download.pytorch.org/whl/cpu`（清华/清华镜像是主源）。

依赖（`requirements.txt`，版本就是训练时用的版本）：

| 包 | 版本 | 必需 |
| --- | --- | --- |
| mujoco | 3.12.0 | ✔ |
| gymnasium | 1.3.0 | ✔ |
| stable-baselines3 | 2.9.0 | ✔ |
| torch | 2.14.1+cpu | ✔ |
| numpy | 2.2.4 | ✔ |
| tensorboard | 2.21.0 | ✔ |
| onnx / onnxruntime | 1.23.1 / 1.30.0 | 导出用 |
| pillow | 10.4.0 | 渲染用 |
| matplotlib | 3.9.2 | 画图用 |

`models\unitree_go2\`（MJCF + obj 网格）和 `artifacts\trot_stairs.json`（trot 步态
先验，11 KB，**唯一不可重建的 artifact**）已经在仓里；想重新拉一份 Go2 模型跑
`scripts\download_go2.py`。

> 沙箱注意：`python -m venv .venv` 可能因 `ensurepip` 失败而报
> `No module named pip`，退路是 `--without-pip` 再喂 `get-pip.py`；且 pip 会往
> 临时目录写，受限沙箱下需要放宽权限。详见 `experience.md` §5.8。

---

## 2. 目录

```
robotdog\
    paths.py       所有路径常量（工作区 = 本仓根目录）
    terrain.py     楼梯地形（解析式高度场 height_at(x) + MuJoCo 几何体）
    go2_model.py   组装 Go2 MJCF（字符串手术 + 场景编译）
    gait.py        trot 步态先验（GaitParams）
    touch.py       触觉/前视探测（足底法向力、滑移、立面擦碰、前方射线扇）
    env.py         Gymnasium 环境（观测/奖励/终止/域随机化）★核心
    curriculum.py  课程（平地 → 3~8 → 8~11 cm，含混合高度）
    training.py    PPO 组装 / 运行目录 / VecNormalize / 断点续训
    evaluate.py    评估矩阵 + ONNX / TorchScript 导出
    render.py      GIF 写出（纯 Python + Pillow，不需要 ffmpeg）
    png.py         PNG 写出（zlib）
    console.py     UTF-8 控制台
scripts\
    train.py               训练入口（所有开关都在这里）
    eval.py                评估 / 导出入口
    render_checkpoint.py   渲染 GIF
    download_go2.py        （可选）重新下载 Go2 MJCF
tools\
    ab_pair.py   6 档高度 × 40 回合的配对普查（验收用）
    ab_merge.py  合并多份普查缓存并出表
models\unitree_go2\          Go2 MJCF + 网格
artifacts\trot_stairs.json   trot 先验
```

训练/评估产物都落在本仓根目录下（`runs\`、`logs\`、`output\gifs\`），目录由
`robotdog\paths.py` 自己建。`robotdog\__init__.py` **只**导出
`setup_utf8_console`，脚本要显式 `from robotdog.env import ...`。

---

## 3. 代码实现

### 3.1 时间与动作管线

| 量 | 值 |
| --- | --- |
| 物理步长 `timestep` | 0.002 s（500 Hz） |
| `decimation` | 10 |
| 控制频率 | 50 Hz（`control_dt = timestep × decimation`） |
| PD 增益 | `kp = (30, 30, 45)`、`kd = (0.6, 0.6, 0.9)` |

动作是 12 个关节目标角，`action_mode="residual"` 时：

```
target = home_angles + gait_gain × gait.radians(phase) + action_scale × action
```

`action_scale=0.30` 是**策略**的权限，`gait.action_scale` 是**先验**的幅度，两者
刻意分开：共用一个 scale 会让策略的探索噪声随先验放大（实测 2 s 内就摔）。  
先验与目标都先 `clip` 到 `[act_low, act_high]`（来自 MJCF 的关节限位）。
`_apply_latency` 每控制步只抽一次延迟，缓冲用第一条命令 priming，**不能用 zeros**。

### 3.2 观测 `ObsConfig`

`obs.size = frame_size × history_len`，默认 `history_len = 10`。一帧的固定 42 维：

```
关节角(12) + 关节角速度(12) + 机身 rpy(3) + 机身角速度(3) + 上一帧动作(12)
```

可选块按开关追加，**全部默认关闭**：

| 开关 | 维度 | 含义 |
| --- | --- | --- |
| `base_lin_vel` | 3 | 世界系线速度（不推荐，yaw 一转含义就变） |
| `base_lin_vel_body` | 3 | **机身系**线速度 |
| `base_height` | 1 | 机身高度（参照面见 `base_height_support`） |
| `base_height_support` | +0 | 只换参照面：改用四足足下最低地形，不改变维度 |
| `foot_height` | 4 | 每足离地间隙 |
| `gait_phase` | 2 | sin/cos 步态时钟（有先验时应有） |
| `touch` | 9 | 4 足底法向力 + 4 滑移比 + 1 蹭立面标志 |
| `stair_probe` | 3 | 前方障碍高度 / 距离 / 是否命中 |
| `stair_probe_full` | +2 | 追加 `probe_slope`、`probe_far_step` |
| `drift` | 2 | 横向偏移（对楼梯中线）+ 航向误差（对楼梯轴） |

常用维度：默认 **420**（42×10）；课程配方（`base_lin_vel_body + base_height +
foot_height + gait_phase + touch + stair_probe + drift`）⇒ 每帧 66、共 **660**。

帧约定：`qpos[0:3]` 世界位置、`qpos[3:7]` 四元数 `(w,x,y,z)`、`qvel[0:3]` 世界
线速度、**`qvel[3:6]` 本地（机身）角速度** —— IMU 陀螺直接就是后者。

### 3.3 奖励 `RewardConfig`

用户原始要求的五项标 `[SPEC]`：

| 项 | 默认 | 口径 |
| --- | --- | --- |
| `forward_vel` | 2.5 | `exp(-(vx - command_vx)² / σ²)`，`σ = 0.22` |
| `orientation` | -1.2 | `roll² + pitch²` |
| `torque` | -2.0e-4 | `Σ τ²`（`data.actuator_force`） |
| `foot_lift` | 0.6 | 见下 |
| `body_collision` | -4.0 | 躯干碰地形 |

补齐与 shaping：

| 项 | 默认 | 口径 |
| --- | --- | --- |
| `forward_progress` | 1.0 | 线性 `clip(vx, -0.5, 1.5)`，`vx=0` 时恰为 0 |
| `air_time` | 0.6 | 落地那一步按滞空时长结算（`target=0.30`、`max=0.60`） |
| `lateral_vel` / `yaw_rate` | -0.8 / -0.15 | `vy²` / `(wz - cmd)²` |
| `base_height` | -3.0 | **仍用单点参照面**（故意不跟着观测改） |
| `joint_vel` / `action_rate` / `joint_limit` | -2e-5 / -0.02 / -0.5 | |
| `alive` / `termination` / `success` | 0.12 / -2.0 / 20.0 | 终局项记在 `terms["terminal"]` |
| `stair_progress` | 2.5 | 爬升进度；`foot_ref` / `min_loaded_feet` 见下 |
| `foothold` | 0.0 | 四足足下最低地形相对上一步的**增量** / 台阶高 |
| `riser_graze` | -0.60 | 蹭立面；`mode` / `probe_gate` 见下 |
| `touch_impact` | -0.50 | 落地法向力超 `target=95 N` 的平方超量 |
| `touch_slip` | -0.35 | 只对承重脚（`force > 10 N`） |
| `stair_probe_lift` | 1.20 | 测距看到台阶 **且**前两足已抬起 |
| `leg_collision` | -0.25 | 腿 geom 碰地形（**不是**终局） |
| `straight_line` | -20.0 | `clip(横向偏移, ±1.0)²` |
| `heading` | -8.0 | `clip(航向误差, ±0.60)²` |
| `lateral_vel_track` | 0.5 | `exp(-(vy - cmd_vy)² / 0.15²)` |

`_compute_reward` 返回 `(sum(terms), terms)`，**账目恒等**
（`sum(reward_terms) == reward`，终局步也成立）。

抬升目标：`target = max(foot_lift_target, foot_lift_step_ratio × h)`，连续项按
`clip(foot_clearance / target, 0, 1)` 给分。`foot_lift_step_ratio = 1.4` 是针对
**正下方参照面**标定的（鞋底必须抬过头高才能跨过去）；5 cm 时
`max(0.08, 0.07) = 0.08`，与最初的常数目标逐位相同，所以已经学会的 easy 档不会退化。
这个参照面在台阶处是阶跃的，由此引出的问题与「为什么没换成落脚面口径」见
`experience.md` §6.1 与 `train.md` §3.6。

### 3.4 终止 `EnvConfig`

| 开关 | 默认 | 说明 |
| --- | --- | --- |
| `terminate_contact` | `"trunk"` | `any` / `trunk` / `none` |
| `terminate_contact_steps` | 1 | 需**连续** N 个控制步才判死 |
| `contact_debounce_adaptive` | False | 高台阶上收紧到 `contact_debounce_hi_steps` |
| `terminate_height_ref` | `"base"` | `base` / `support_mean` / `support_load` / `support` |
| `terminate_min_height` | 0.13 | 低于此判摔 |
| `terminate_max_tilt` | 1.0 rad | 倾角超过此判摔 |
| `terminate_body_contact` | True | 关掉则任何接触都不致命 |
| `trunk_contact_adaptive` | False | 高台阶上躯干接触**不致命**（仍照常扣分） |

摔倒检测（`_terminated`）：`height < terminate_min_height` 或
`max(|roll|, |pitch|) > terminate_max_tilt`。参照面由
`_fall_reference_height()` 决定，**默认是机身 x 处的单点高度**（阶跃，会幽灵摔倒），
配方里统一用 `terminate_height_ref=support`。

### 3.5 其它环境侧要点

* `_success_reached()`：楼梯上要求 `base_x > up_end_x` **且**
  `terrain_height_under_base >= crest_height`；平地上用 `flat_success_distance`。
  成功是 **sticky** 的（`_reached_top`）。
* `reset()`：x 从 `start_x - spawn_back(0.55)` 起、带 `reset_xy_noise=0.12`；
  z = 地形高 + `reset_z(0.32)`；yaw 带 `reset_yaw_noise=0.10`；关节加
  `reset_joint_noise=0.06`；`qvel = U(-0.05, 0.05)`。
* `_apply_domain_randomization()`：摩擦 / 质量 / 惯量 / 观测延迟 / IMU 偏置，
  全部由 `randomize_scale` 统一收缩，默认**全关**。
* `render()` 内部**强制打开 6 个 geomgroup**，否则 group 3/4/5 默认隐藏、楼梯在
  GIF 里会消失。
* 课程切换地形靠 `set_stair()`；先验靠 `set_gait_prior(enable, gait)` —— 注意它只设
  `cfg.gait_prior`，构造时若漏设同一个字段，先验会完全不加（见 `experience.md` §5.1
  的同类陷阱）。

---

## 4. 常用命令

```bat
rem 从零训练（完整配方见 train.md）
.venv\Scripts\python.exe scripts\train.py --help

rem 冒烟：4 个阶段各压到 512 步，约 1 分钟
.venv\Scripts\python.exe scripts\train.py --smoke --run-name smoke --n-envs 2

rem 评估（trained = 训练时见过的 4 个地形；unseen = 9 个没见过的）
.venv\Scripts\python.exe scripts\eval.py --run <run> --model runs\<run>\model_final.zip --only trained --episodes 40

rem 导出 ONNX + TorchScript（归一化已烘焙进计算图）
.venv\Scripts\python.exe scripts\eval.py --run <run> --model runs\<run>\model_final.zip --only trained --episodes 6 --export --export-dir runs\<run>\export --export-name policy

rem 渲染 GIF
.venv\Scripts\python.exe scripts\render_checkpoint.py --run <run> --checkpoint runs\<run>\model_final.zip --step-height-cm 10.5 --n-up 8 --episodes 4 --name-tag demo --no-strip --no-report

rem 验收：6 档 × 40 回合配对普查
.venv\Scripts\python.exe tools\ab_pair.py "label=runs/<run>/model_final.zip=runs/<run>" --episodes 40
```

`--model` 指到哪个 zip，就自动配同目录的 `<同名>_vecnormalize.pkl`。
**观测归一化必须跟着权重走**，手工加载时别只加载 zip。
`scripts\eval.py` 会从 `--run` 目录的 `summary.json` 恢复训练时的 `EnvConfig`，
所以评估参考 checkpoint 时观测维度自动正确。
