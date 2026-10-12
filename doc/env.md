# `robotdog/env.py` — Gymnasium 环境

对应代码：`robotdog/env.py`（`Go2StairsEnv` + 三个配置 dataclass）。
调试记录与设计依据见 `experience.md`；本文只讲抽象结构。

## 职责

一只走在参数化箱式楼梯上的 Unitree Go2。**唯一**负责物理推进（`mj_step`）的地方；
观测组装、指令管理、成功/终止判定、渲染也在这里，奖励数学全部在 `robotdog/reward.py`。

## 核心抽象

### 一步的固定时序（`step()`）

每一步按固定顺序做七件事，顺序本身是设计：

```
指令到期重抽 → 组装 PD 目标（home + 先验 + 残差）→ 物理推进（decimation 次 mj_step）
→ 接触扫描（一次遍历得 躯干/腿/逐足 三类接触）→ 势函数状态机推进（reward.py）
→ 低通滤波 → 跟踪统计与成功判定 → 奖励结算 → 动作历史与观测历史推进
```

马尔可夫性的约定是：**奖励读到什么量，观测里就必须报同一个量**（典型：横向/偏航
指令跟踪读低通滤波值，观测里就有 `vel_filter` 那两维）。任何一侧单独改动都会
破坏「策略看到的和它被评判的是同一个数」。

### 观测组装（`_obs_frame()`）

固定 42 维本体感知块 + 按开关追加的块（线速度、步态相位、台阶高、三轴指令、
滤波量），再堆叠 `history_len` 帧。**盲走契约**：观测里不得出现任何地形真值；
`base_height` 开关在 `ObsConfig.__post_init__` 里被硬性禁止（构造即报错，分支已删），
因为「默认关掉」不是保证。信息来源逐项清单见 `agent.md` §8.6。

### 指令的两态模型（`CommandConfig`）

三轴速度指令 `[vx, vy, wz]` 有两层状态：**采样范围**（课程在阶段边界用
`set_commands()` 改写）与**当前值**（外部控制器每步用 `set_command_now()` 改写，
`hold=True` 可钉住整回合）。奖励、观测、成功判据读的都是当前值 `env._command`，
它与 `cfg.command._now` 是同一个列表，不存在两份状态。

### 成功判据的双轨（`_success_reached()`）

- **楼梯段**：位置判据 —— 机身越过 `stair.up_end_x` 且站在 crest 高度上。
- **平地段**：指令跟随判据 —— 撑满回合 + 脱靶步占比低于阈值 + 实测量与指令的
  相关系数达标（只对指令本身在变的轴判定）。相关系数这一条是必要的：
  平均绝对误差在「输出常数」时取最小，单独使用会把常数输出判成成功。

### 终止三判据（`_terminated()`）

只有**高度**（相对支撑面，默认四足足下最低地形）、**倾角**、**超时**三条。
接触类信号不终止回合 —— 躯干接触的代价由逐步的 `body_collision` 承担，膝/胫接触
只推进诊断计数。完整论证见 `experience.md` §12。

### 与 reward 的分工

势函数的**状态字段**（`_peak_terrain_height`、`_probe_best`、`_blocked` 等）声明
并清零在 env（reset 是 env 的职责、info 要上报），而读写这些状态的**机制**全部在
`robotdog/reward.py`。env 的 `step()` 在奖励结算之前调用
`track_swing_peak(self, ...)` 与 `update_probe_state(self)`。

### 支撑面与帧约定

`_support_height()` / `_support_height_mean()`（四足足下地形）服务摔倒判据与
爬升结算；`terrain_height_under_base`（躯干 x 的阶梯函数）只用于观测外的场合。
帧约定与 MuJoCo 一致：`qpos[0:3]` 世界位置、四元数 `(w,x,y,z)`、
`qvel[3:6]` 是机身系角速度（IMU 陀螺口径）。模型在 `__init__` 编译一次，
生命周期内不变（课程的每一级都是新建环境，不存在换地形）。

## 渲染

离屏渲染 5 个相机预设（`_CAMERA_VIEW` 表 + 注视点查表），地形 geom group
必须落在 MuJoCo 默认可见的组里（见 `experience.md` §1.2）。

## 踩坑指引

零高度台阶、共面接缝夹脚、幽灵摔倒、步态振荡淹没偏航信号 —— 全部见
`experience.md` §1–§3、§13、§16。
