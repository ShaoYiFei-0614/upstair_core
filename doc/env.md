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

### 观测：策略每一步能看到什么（逐块讲解）

拼接规则（`_obs_frame()`）：**恒在 48 维**（42 维本体 + 两个必选块），再按
`ObsConfig` 的开关依声明顺序追加可选块，最后沿时间堆叠 `history_len` 帧。
当前 `config.yaml`（`history_len: 1`，gait_phase/step_height/vel_filter 开、
gait_cmd 关）每帧 **53 维**；「全部可选开关开」的 actor 帧是 **55 维**
（48 + gait_phase 2 + step_height 1 + vel_filter 2 + gait_cmd 2）。打开特权块
（`privileged_foot_contact`，仅 critic）时 `frame_size`/`size` 再 **+4**，而
`actor_size` 不变。

**恒在 48 维（全部真机口径；后两块为必选项，构造期强制，改 false 直接抛
ValueError）：**

| 块 | 维 | 物理量 | 真机出处 | 为什么必须给策略 |
| --- | --- | --- | --- | --- |
| `joint_pos` | 12 | 关节角 | 关节编码器 | 本体感觉的根：腿的几何姿态决定一切力学行为 |
| `joint_vel` | 12 | 关节角速度 | 编码器微分 | 同上的速度侧；接触冲击最先体现在这里 |
| `base_rpy` | 3 | 机身 roll/pitch/yaw（世界系欧拉） | IMU 姿态估计 | 重力方向 =「哪边是上」—— 盲走爬楼最重要的一个量 |
| `base_ang_vel_body` | 3 | 机身角速度（机体系） | IMU 陀螺 | 与真机陀螺口径一致（`qvel[3:6]` 本就是机体系） |
| `_last_action` | 12 | 上一步下发的动作 | 自己 | **马尔可夫性**：PD 目标含上一步先验+残差，不给它则同一观测对应不同下一步 |
| `base_lin_vel_body` | 3 | 机身系线速度 | 腿式里程计 + IMU 融合（本体感觉估计，**非特权**） | 指令跟踪的「实测」侧：不知道自己在往哪走，就无法修正 |
| `command` | 3 | 三轴速度指令（**机体系投影**，世界系指令经 `command_body()` 每步转换） | 上层控制器/规划器 | 指令跟踪的「指令」侧；课程只能改采样范围，不能改维数 |

**可选开关块（维数一旦定下，整条课程不可变 —— PPO 网络与 RolloutBuffer 不重建）：**

| 块 | 维 | 物理量 | 真机出处 | 为什么（或不）给 |
| --- | --- | --- | --- | --- |
| `vel_filter` | 2 | 横向速度 / 偏航角速度的**低通值**（τ=0.2 s） | 对同一信号滤波，真机可算 | 瞬时量被步态摆动主导（wz 摆幅可达 ±0.37 rad/s）；**奖励读同一批量，必须成对出现**（马尔可夫性） |
| `gait_phase` | 2 | 步态时钟 sin/cos | 内部时钟 | 残差策略需要知道先验处于周期哪一点（当前 config 打开） |
| `step_height` | 1 | 本回合台阶高（m） | **任务参数**，随任务下发 | 唯一非传感器项：告诉策略「在哪条楼梯上」，但**不含下一道立面在哪** |
| `gait_cmd` | 2 | 步态指令（freq [Hz] + swing 缩放） | 上层指令（Walk-These-Ways 式条件化） | 策略**伺服**的步态参数：楼梯档/平地档是同一策略的两个条件点，部署时 `set_gait_cmd_now()` 实时切换；P2 的教训是参数放 action、WtW 放 obs |

**特权块（`privileged_foot_contact`，**只给 critic**，默认关）：**

| 块 | 维 | 物理量 | 真值来源 | 为什么只给 critic |
| --- | --- | --- | --- | --- |
| `privileged_foot_contact` | 4 | 逐足是否接触地形（足端 geom × floor/stair geom） | MuJoCo 接触对（仿真器真值） | actor 部署到真机必须盲走；critic 只是训练期的价值函数，吃特权信息降低价值方差（非对称 actor-critic，`robotdog/asymmetric.py`） |

打开后每帧末尾 `+4` 维，`ObsConfig.size` 变大而 `actor_size` **不变**：`step()` 返回的
观测是**全量**（critic 用），actor 的切片由 `ObsConfig.actor_index` 给出（帧优先条带，
支持 `history_len > 1`），策略侧在 `AsymmetricActorCriticPolicy` 里 `index_select`。
关闭（默认）时与对称 AC 的观测**逐位相同**（前向兼容的判据，见 `train.md` 的冒烟条目）。

注意：无论开关开不开，`info["foot_contact"]`（4 个 float）**每步都在**，它是只读诊断量，
不进观测、不进奖励/终止 —— 写对比 `info` 键集合的探针时要把它排除在外。

**盲走契约（硬边界）**：**actor 的观测**里不得出现任何**地形几何真值**。`base_height`
（机身相对正下方地形的高度）曾被实现过 —— 它等于把地形读数塞进观测；现在
`ObsConfig.__post_init__` 构造期直接拒绝，实现分支也整体删除（「默认关掉」
不是保证，experience.md §11）。

特权块是这条硬边界的**唯一例外形态**：它只进 critic 的 value 网络（训练期存在、
部署时不存在），由非对称 AC 在结构上切开（`ObsConfig.actor_index`），因此 actor 的
输入里依然没有任何真值。**不要把特权维直接追加给 actor** —— 那就是把 `base_height`
的事故重演一遍。

**特权信息白名单**（真值只进奖励/终止，绝不进观测 —— RL 里这是允许的「上帝
视角」，但一进观测就破坏盲走）：`foot_clearance`（足端离地）、
`_support_height`（四足支撑面：摔倒判据与 stair_progress）、`_contact_scan`
（接触掩码：碰撞罚与 air_time）。逐项真机等价物与审计结论见
`experience.md` §11.4。

以上约束的是 **actor 的观测**；critic 的**特权观测**是唯一例外形态（非对称
actor-critic，见上文特权块与 `experience.md` §23.60）：它只进 value 网络、
部署时不存在，不进 actor 的输入。

### 动作：从策略输出到电机

动作空间是 `Box(-1, 1)`，维度与顺序**随 `action_mode` 变**：

**`residual`（12 维，旧口径）** —— 每条腿 `hip / thigh / calf` × 腿序
`FL, FR, RL, RR` 的关节残差，叠加在固定先验上：

    target = clip(home + gait_gain · gait.radians(t) + action_scale · action, lo, hi)

**`gait_params`（8 维，P2）** —— `[前对(amp_th, amp_ca, off_th, off_ca), 后对(同四项)]`：

    target = clip(home + traj(参数, θ), lo, hi)
    traj: thigh_l = off_th[pair_l] + amp_th[pair_l]·sin(θ + φ_l + ph_th)
          calf_l  = off_ca[pair_l] + amp_ca[pair_l]·wave(θ + φ_l + ph_ca)

其中 θ = 2π·步态时钟，φ_l 用固定的 **trot 对角反相**（`(0, π, π, 0)`），
波形与相位偏置沿用先验结构；幅值参数映射到 `[0, param_amp_max]`（楼梯上再乘
`stair_stride_boost`），偏置映射到 `±param_off_max`。步频由地形决定
（`gait_freq_flat` / `gait_freq_stair`）。

**为什么这样切**（§23.17 P2）：结构（sin + trot 相位）保证随机探索也落在
**协调步态**上 —— 这解决了「无先验 → 奖励平台期 → 学不出行走」的老问题；
而**前/后对分开参数化**让「前脚驻留（前对幅值→0）+ 后腿抬升（后对幅值拉大）」
这类受阻态行为**可表达** —— 固定先验的对角耦合做不到（§23.20 的 P1 实验
证明：受阻时把先验整个关掉会毁掉爬楼，因为节奏同时是登踏的动力）。

**残差模式的两个幅值刻意分开**：`gait.action_scale` 表示先验动作有多大，
`cfg.action_scale` 表示策略可以偏离它多少。共用一个尺度会让策略的探索噪声
随先验一起缩放，瞬间失稳。gait_params 模式只有先验结构、没有残差项。

### 按地形选先验（`gait_stair`）

`residual` 模式下，楼梯级与平地级可以跑**不同的先验**（`gait_file_stair` →
`EnvConfig.gait_stair`；`_active_gait()` 按 `stair.n_up > 0` 切换）：

* 平地 → `gait_file`（`artifacts/trot_natural167.json`，1.67 Hz 自然小跑，
  平地自然高抬腿就是在这上面学出来的）；
* 楼梯 → `gait_file_stair`（`artifacts/crawl_125.json`，1.25 Hz **爬行**：
  四分之一错相 + `swing_frac 0.25` ⇒ 一次一足、三足支撑）。

**为什么楼梯要爬行**：实测前后足 x 间距 0.367 m > 踏面 0.30 m ⇒ 四足**无法
同踩一级**、必须跨级站立；跨级时小跑的 2 足支撑余量不足，静态支撑是稳定性来源
（`experience.md` §23.32）。

频率的取值优先级（`_gait_freq_eff`）：条件化采样的指令 → **楼梯专用先验自带的
频率** → `gait_freq_stair_fixed`（>0 时钉死）→ 按台阶高在「先验 json 频率 → 1.0」
之间线性插值。奖励侧（`foot_lift` 速度门控、`air_time` target）读的是**同一个**
函数 —— 用错基准会让等价性悄悄破掉。

**执行**：目标角交给关节 PD（`tau = kp(target−q) − kd·qd`，kp/kd 按腿为
`[30, 30, 45] / [0.6, 0.6, 0.9]`），每个控制步内做 `decimation=10` 次
500 Hz 物理步，力矩按 `ctrl_ranges` 截断。选**位置控制**而非力矩控制：
四足位置伺服是 Unitree 出厂形态，sim-to-real 差距最小。

**与奖励的耦合**：`action_rate` 罚相邻步动作差（平滑，按 `num_action` 维计），
`torque` 罚力矩平方（节能），两者共同把动作压向「必要的变化量」；探索幅度由
PPO 的 `log_std_init` 决定（`doc/training.md`）。

### 指令的三层模型（`CommandConfig`）

三轴速度指令 `[vx, vy, wz]` —— **vx/vy 在世界系给定**（vx 沿课程方向 +x），
wz 为绕竖直轴的角速度。三层状态：

1. **采样范围**：课程在阶段边界用 `set_commands()` 改写；
2. **当前值**：`reset()` 时采样、回合内周期重抽，外部控制器可用
   `set_command_now()` 随时改写（`hold=True` 钉住整回合）；
3. **缓变扰动**（`command_wander_std/tau`）：训练时叠加 OU 过程模拟真实
   操作手的轻微无规律输入（稳态 ≈0.026 m/s、限幅 ±0.08）；`hold=True` 不生效。

**世界系 → 机体系**（`command_body()`，每步现算）：执行器只产生机体系速度，
所以奖励的比较、观测的呈现、脱靶判据用的都是投影后的机体系指令
（`[vx_b, vy_b] = R(yaw)ᵀ·[vx, vy]`，wz 两系同值）。世界系指令在机身转向后
含义不变 —— 掉头背对指令方向，机体系指令立刻变成「后退」，误差自然惩罚它
（§21.14 的「掉头走开」漏洞由此在结构上关闭，不再需要按地形切换参照系）。

**第四层：楼梯段闭环引导**（`_guidance_command()`，`stair_guidance` 开关）。
楼梯级（n_up > 0）的 vy/wz 不来自课程采样，而由**外部控制器每步按状态生成**：
vy 按偏离台阶中线的距离反号拉回（增益 1.2/m、限幅 ±0.15），wz 按朝向误差
对准 +x（增益 1.5/rad、限幅 ±0.6）；缓变扰动叠加在引导值上。这是「人在遥感
操控」的机制化 —— 跟随引导指令 = 保持中线与航向，狗走偏时指令会主动纠正它
（开环直线指令对漂移无能为力，§23.14）。vx 仍由课程采样；平地级不受影响。

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
