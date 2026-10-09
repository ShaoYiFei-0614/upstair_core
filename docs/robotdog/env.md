# `robotdog/env.py` —— 核心 Gymnasium 环境

## 1. 一句话定位

属于**环境层**：把 `robotdog/go2_model.py` 编译出的「Go2 + 参数化楼梯」MJCF 包成一个
Gymnasium 环境（`Go2StairsEnv`），提供 50 Hz 控制 / 500 Hz 物理的观测、奖励、终止与成功判据，
并把步态先验、域随机化、触觉/测距读数、离屏渲染统一挂在一个接口上。

## 2. 职责边界

**做：**

* 观测拼装：固定 42 维单帧（`ObsConfig.base_frame_size`）+ 10 帧历史栈（`history_len`），
  可变块由布尔开关追加。
* 奖励计算：`RewardConfig` 的每一条塑形项、终止奖励与成功奖励（`_compute_reward`）。
* 终止/成功判据：摔倒（高度 / 倾角）、接触致命性（躯干 / 腿 / 任意 / 无）、超时截断、
  终点粘性标记（`_reached_top`）。
* 控制回路：`action_mode="residual"` 的关节目标合成、PD 律、
  `decimation` 次物理子步推进（`mujoco.mj_step`）。
* 步态先验注入：`gait_prior` / `gait_gain` / `gait_phase_randomize` 与 `_gait_t` 时钟。
* 域随机化（摩擦 / 质量 / 延迟 / IMU 偏置）与 `randomize_scale` 严重度插值。
* reset 采样（出生位置、朝向、关节噪声、初速度）与触觉读数 priming。
* 离屏渲染（`render`）与资源释放（`close`）。
* 调度触觉/测距读取（`self.touch_sensor.read(...)`），但**不实现**读取算法。

**不做：**

* 不做 MJCF 字符串拼接、不编译模型、不解析几何 id —— 全部委托 `go2_model.compile_scene`
  （`extract_meta` 返回的 `Go2Meta` 是本模块唯一的数据来源）。
* 不定义楼梯几何、不算高度剖面 —— `StairSpec` / `build_terrain_xml` 属于 `robotdog/tools/terrain.py`；
  本模块只用 `step_height` / `crest_height` / `up_end_x` / `center_y` / `start_x` / `height_at()`。
* 不算步态轨迹形状 —— `GaitParams` 属于 `robotdog/gait.py`；本模块只调 `gait.radians(t)` / `gait.period`。
* 不读接触力、不投射射线 —— `TouchSensor` / `TouchReading` 属于 `robotdog/tools/touch.py`。
* 不训练、不建向量环境、不做回调、不做课程推进、不做评估矩阵、不导 ONNX/TorchScript、
  不写文件、不打印日志（除了 `render` 的异常）。
* 不改任何 vendored 模型文件；`models/` 只读。

## 3. 关键设计决策与理由

### 3.1 残差 RL 的两个尺度必须分开

`target = home_angles + gait_gain * gait.radians(phase) + action_scale * action`。

`EnvConfig.action_scale`（默认 `0.30`）是**策略权限**，`GaitParams.action_scale`（默认
`default_trot()` = `0.90`）是**先验幅度**。两者刻意不共用：

* **否决方案**：共用一个 `action_scale`。先验幅度与策略探索噪声会被同一个系数缩放，
  实测在 2 s 内摔倒（见 `gait.py` 的 `radians()` docstring 与 `experience.md` §2）。
* **否决方案**：直接让策略学绝对关节角（`action_mode="absolute"` 仍保留为可选分支）。
  本沙箱禁用 Windows 命名管道，只能 `DummyVecEnv`（8 环境单进程），PPO 无从发现步态；
  两次标定 120k / 150k 步平地训练均 0% 成功、位移约 0.02 m，学到的策略比零动作站桩更差
  （`experience.md` §2.2、`train.md` §1）。
* `gait_prior` 默认 `False`，所以「无先验」这条路径仍然可用；`_gait_action()` 在关闭或
  `gait_gain == 0.0` 时返回全零，使 `_target_from_action()` 对两种模式是**同一条代码路径**。
* `set_gait_prior()` 可以在线切换：它只改 PD 目标的构造方式，不改观测维数，所以不需要重建
  `observation_space`（改观测块才需要）。

### 3.2 观测：42 维基础帧 + 可选块**全默认关闭**

基础 42 维是「用户规格」的最小集：`joint_pos(12) + joint_vel(12) + base_rpy(3) +
base_ang_vel_body(3) + last_action(12)`。所有后来加的可选块默认 `False`，理由是
**旧 checkpoint 必须位对位可复现**（`EnvConfig` 的 docstring 明说「Defaults are the historical
behaviour」）。

* **否决方案**：把有用的块（`base_lin_vel_body` / `gait_phase` / `stair_probe`）默认打开。
  一是破坏旧 checkpoint；二是观测与奖励耦合时后果严重 —— 把 `base_height` 的参照面同时接进
  观测**和奖励**，均回报从 +554 掉到 -674 / 回合（`experience.md` §3.3）。
* `base_height_support` **不改变维数**（+0），它只切换 `_height_reference()` 用哪个面；
  因此它是纯粹的「单变量开关」，见 3.6。
* `history_len=10` 与 `base_frame_size=42` 一起给出默认 `obs.size = 420`；
  课程配方常用 66 维/帧 ⇒ 660 维（`README.md` §3.2）。
* `observation_space` 用 `spaces.Box(-np.inf, np.inf, ...)`：没有任何一项被证明有硬边界，
  写死上下界会在归一化前就引入裁剪。

### 3.3 帧约定：`qvel[3:6]` 直接当陀螺用

`qpos[0:3]` 世界位置、`qpos[3:7]` 四元数 `(w,x,y,z)`、`qvel[0:3]` 世界线速度、
**`qvel[3:6]` 是机身（local）角速度** —— 这正是真实 IMU 陀螺的输出，所以
`base_ang_vel_body` 直接返回它，不做任何旋转变换。

* **否决方案**：把角速度转到世界系再用，或从四元数数值微分。前者随 yaw 改变含义
  （和 `base_lin_vel_world` 一样是「yaw 变了含义就变了」的坏特征），后者引入差分噪声。
* `base_lin_vel_body` 用 `R(q)^T @ qvel[0:3]`，理由同样：这样「前进」永远是 `[0]`。
  代码里两个都留着，因为世界系版本是历史行为、且对「绝对方向」类诊断有用。

### 3.4 `forward_progress`：高斯项在 vx=0 不为零

`forward_vel` 是指令跟踪的**高斯**，`exp(-(vx-0.55)²/0.22²)`；它在 `vx=0` 处
仍然等于 `exp(-(0.55/0.22)²) ≈ 0.0018`，配合 `alive` 就让「站着不动」成为一个正回报的局部最优。

* 修法：加**线性**伴项 `forward_progress * clip(vx, -0.5, 1.5)`，它在 `vx=0` 处**恰好为 0**，
  梯度永不消失；再配合把 `alive` 压到 `0.12`，站桩不再划算。实测 12 万步平地上策略
  只学出 0.00 m/s、0.02 m（`experience.md` §2.2）。
* **否决方案**：只调小 `forward_vel_sigma`。σ→0 让高斯变成尖峰，远离指令处梯度全为 0；
  而且 `vx=0` 的正基座只被削小、不会消失。

### 3.5 `alive=0.12`：从 0.4 压下来的

`alive=0.4` 时，一个 600 控制步（12 s）的「站着不动」回合回报里有 **49% 来自 `alive`**
（`experience.md` §2.2）。压到 `0.12` 后它仍是「活着的小费」，但不足以对抗前进项。
（12 s 一集的 `alive` 上限 = `600 × 0.12 = 72`。）

### 3.6 `base_height` 奖励**故意**仍用单点参照面

`terms["base_height"]` 用的是 `self.terrain_height_under_base`（= `stair.height_at(base_x)`，
x 的**阶梯函数**），**不是** `_height_reference()`：

* 实测：把 `obs.base_height_support` 同时接进奖励，均回报从 **+554 崩到 -674 / 回合**
  （`experience.md` §3.3）。噪声来源是在立面附近参照面本身跳变，`(z - ref - 0.27)²` 出现
  人为跳变。
* 因此 `obs.base_height_support` **必须保持「只影响观测」的单变量开关**语义。
* 需要「基于脚的参照面」做奖励时，用另一条通路：`reward.stair_progress_foot_ref`
  （它切的是 `stair_progress`，见 `experience.md` §2.7）。这条「故意的不一致」是设计，
  不是遗漏 —— 想统一之前请先复现 §3.3 的对照实验。

### 3.7 `touch_excess_clip=5.0`：无上限的平方力比会毁掉整段 value function

`touch_impact` 与 `touch_slip` 都是**物理比值的平方**，而 MuJoCo 的接触力在脚被卡在立面边缘
或硬着陆时会无界尖峰。不加上限时**单个病态控制步**可达 O(10^5)，实测把整段 value function
打炸（`course11` 的 `easy2` 段：回合均回报 **-586285**，`experience.md` §2.1）。

* 修法：裁剪**归一化超额**（`(f - ref)/ref` 与 `slip - safe`），每步每脚最坏贡献被限制在
  `touch_excess_clip² = 25` 以内。
* **否决方案**：直接删掉触觉项。触觉是 11 cm 台阶上唯一能区分「卡住」与「正在跨越」的信号
  （见 `riser_graze_mode`）。
* 同样的原则施加在 `straight_line` / `heading`：**先 clip 再平方**（`straight_line_max=1.00`、
  `heading_max=0.60`）。不裁剪的平方项曾让一个触觉步达到 -5.5e7。

### 3.8 `termination=-2.0` 与 `success=20.0` 的量级

* `termination=-2.0`：摔倒的即时惩罚约等于 **0.2 s 的 `alive`**，不是「巨额罚款」。
  它只负责让「活得更久」比「马上死」好一点点，真正的信号来自 `body_collision` 等逐步项。
* `success=20.0`：终点奖励必须显著大于一集的 `alive` 累积（72）才有牵引力，但也不能大到
  让 value 目标爆掉。**否决方案**：`success=200` / `termination=-20` —— 该组消融结论待复核
  （`train.md` §3.5 的 E 组），因此没有采用。
* 终止奖励与成功奖励都记进 `terms["terminal"]`，所以
  `sum(info["reward_terms"].values()) == 返回的 reward` 在**终止步也成立**
  （见 §9 关于 `_compute_reward` 本身返回值不含 `terminal` 的说明）。

### 3.9 `straight_line=-20.0`：-2.0 时它等于不存在

`straight_line=-2.0` 时该项只占 200 分回合回报的 **约 0.65%**，策略完全感知不到；改成 `-20.0`
并加 `straight_line_max=1.00` 后才真正约束横向漂移（`experience.md` §2.8）。

* 参照量也换过：`lateral_offset` 从「相对出生 y」改成「相对 `stair.center_y`」。
  旧口径下 reset 的 `±reset_xy_noise`（0.12 m）抖动被算成策略的错，「保持出生线」本身就是
  随机的；新口径是「对齐楼梯中线」，在 2.0 m 宽楼梯里要求收敛到中线是公平的
  （`experience.md` §3.2）。`spawn_lateral_offset` / `spawn_heading_error` 保留下来做诊断，
  用于区分「reset 把我扔偏了」和「我自己飘了」。

### 3.10 `foot_lift_step_ratio=1.4`：目标随台阶高度长，但 5 cm 上位对位不变

`lift_target = max(foot_lift_target, foot_lift_step_ratio * stair.step_height)`。

* 理由：`foot_lift_target=0.08` 是**目标下限**而不是目标本身。脚下参照面口径下，脚掌必须
  **越过**台阶高度才能跨过去（`experience.md` §6.1），写死 0.08 让 8 cm 台阶上目标完全饱和
  （`experience.md` §2.3）。
* 5 cm 时 `max(0.08, 1.4×0.05=0.07) = 0.08`，与原来的常数目标**逐位相同**，
  所以已经学会的 easy 段不会回退。
* **否决方案**：直接把 `foot_lift_target` 调大。19 cm 会要求 0.266 m 的抬脚高度，超出腿的几何
  极限（`experience.md` §2.3）。
* 3-D 足速门（`foot_lift_gate_speed=0.25` m/s）也是刻意选 3-D 而不是水平分量：只看水平速度会
  奖励「快速贴地平扫」，同时把高台阶需要的「慢抬」错误地掐断。
* `foot_clearance = foot_pos[:,2] - foot_radius - terrain_height(foot_x)`：`geom_xpos` 是**球心**，
  必须先减半径才是脚面与地形的真实间隙。

### 3.11 终止：`terminate_contact="trunk"` 默认 + 腿接触只罚不死

* `"trunk"`：只有焊在 `base` 上的躯干碰撞体碰地才致命（真摔）。
* `"any"`（历史行为）：任何非足机器人几何碰地即判死。**被否决**：`course12` 在 10 cm 上
  平均只活 **62.6 个控制步**、从未越过 `x = 1.42 m`，而第一级立面在 `x = 1.50 m` ——
  机器人还没碰到台阶就被「腿刮一下」判死了，试错被彻底杀掉（`experience.md` §2.4）。
* `"none"`：接触永不终止，留给消融用。
* 折中物是 `leg_collision=-0.25`：小腿/膝盖蹭立面**每步扣分但不终止**，保留试错。
* `terminate_contact_steps` 是**连续**控制步阈值（`1` 复现历史行为）；
  `contact_debounce_adaptive` 在高台阶上切到 `contact_debounce_hi_steps`。
  streak 计数器在 `_contact_fatal()` 内推进而不是在 `step()` 里，这样「一次 `_terminated()`
  == 一个控制步」对直接驱动 `_terminated` 的回归测试也成立。
* `trunk_contact_adaptive`：在 `step_height >= trunk_contact_hi_step_h`（0.105 m）上把躯干蹭
  立面从「致命」降级为「贵」，由 `_trunk_contact_relaxed()` 判定。

### 3.12 `terminate_height_ref`：默认 `"base"`，但配方必须用 `"support"`

四个取值：`"base"` / `"support"` / `"support_mean"` / `"support_load"`，分别对应
`terrain_height_under_base` / `_support_height()`（四足足下**最低**地形）/
`_support_height_mean()` / `_support_height_load()`（法向力加权）。

* 为什么默认仍是 `"base"`：历史行为的位对位复现（改默认会让旧 checkpoint 的评测口径变化）。
* 为什么配方必须改 `"support"`：单点参照面是 x 的**阶梯函数**，在立面附近会产生**幽灵摔倒**。
  实测样本：`base_x=1.503` 时 `Δref = +0.1100`，`base_x=1.491` 时 `Δref = -0.1100`
  （`experience.md` §1.4）。
* `_support_height_mean()`：`min` 在最后一只脚落地的那一帧会跳一整个 riser；均值把它摊成
  四次 `h/4` 增量。它**不是**更宽松 —— 跨越过程中均值高于最小值，会更早触发；它去掉的是
  单帧不连续。
* `_support_height_load()`：回答「哪一级**真的在承重**」。与 `support` 在四足同踏面时等价，
  承重转移时连续，并且补上了 `support` 的一个**漏判**：前足已经上去、后足从下一级塌下去时，
  `min` 会把躯干离地高度高估整整一个 riser。无脚承重时回退到 `min`。

### 3.13 触觉两罚的生效范围

* `touch_impact` 只在 `landed = foot_contact & ~_prev_foot_contact` 的那一步付钱 —— 也就是
  「累积腾空时间后着地」的那一步；`_foot_air_time` 存的是**本步之前**累积的飞行时间，
  `_prev_foot_contact` 是本步之前的接触状态，所以账正好落在触地那一步。
* `touch_slip` 使用 `touch.foot_normal_force > cfg.touch.slip_loaded_force`（10 N）门控：
  摆动脚法向力≈0，切向/法向比值纯属噪声。
* `touch_impact_step_scale`（默认 `0.0`）：目标力随台阶高度放大，
  `ref = 95 × (1 + scale × (h/0.05 - 1))`；默认 0 时整段与历史行为逐位相同
  （`experience.md` §2.6）。
* `riser_graze_mode`：`"legacy"`（默认）只要任一足有竖直面接触就罚；
  `"lifting"` 只罚**真卡住**（蹭立面的脚全都没有向上速度 `vz <= riser_graze_vz_eps`）的步，
  正在抬脚跨越的脚免罚。理由：10 / 10.5 / 11 cm 上 `riser_graze` 分别占
  5.7% / 13.1% / 31.9% 的控制步，而 11 cm 上每步罚 -0.192 已经超过 `foot_lift` 实得的 +0.129
  —— 惩罚与任务方向直接打架（`experience.md` §2.5）。

### 3.14 `_apply_latency`：每控制步抽一次，且用第一条命令 priming

* **每控制步**（20 ms）抽一次延迟，**不是**每物理子步（2 ms）。在 decimation 循环里做缓冲会让
  `latency_steps <= 3` 实际只延迟 ≤ 6 ms，远低于任何真实执行器延迟，作为域随机化毫无意义。
* 缓冲区填满前返回 `self._pending_ctrl[0]`（**第一条**命令），**绝不用 zeros**：
  `ctrl` 是 PD 关节**角度目标**，全零目标等于「把 12 个关节全部伸直」，那其实是摔。
  这个 bug 的实测代价见 `experience.md` §5。
* 未启用延迟或 `_latency_steps == 0` 时直接返回原命令，不建缓冲区。

### 3.15 域随机化：先恢复 pristine，再按 `randomize_scale` 向标称插值

* `_apply_domain_randomization()` 每次都先把 `body_mass` / `body_inertia` / `geom_friction`
  从 `_nominal_*` 快照整体还原，再施加缩放。**否决方案**：在上一回合的模型上继续乘 —— 那会跨回合
  复利，20 回合后质量能漂到荒谬值。
* `randomize_scale`（默认 `1.0`）把每个**乘性**范围朝 1.0 收缩：`1 + w*(lo-1)` / `1 + w*(hi-1)`，
  `w = clip(randomize_scale, 0, 1)`。课程把 `w` 从 0 拉到 1，因为一上来就全范围在本控制器下
  不可学。摩擦额外 `clip(..., 0.05, 3.0)`，质量与惯量用同一个 `scale`。
* 延迟的实现是 `max_steps = int(round(w * randomize_latency_max))`，再取 `integers(0, max_steps+1)`；
  IMU 偏置是 `normal(0, w * randomize_imu_sigma, 3)`。
* `set_domain_randomization(scale=None, **flags)` 只接受四个合法 flag 名，未知名字抛
  `AttributeError`（见 §9）。

### 3.16 渲染：强制打开全部 6 个 geomgroup

`mujoco.mjv_defaultOption` 出厂设置是 `geomgroup = [1, 1, 1, 0, 0, 0]` —— **group 3/4/5 默认隐藏**。
地形在 `TERRAIN_GEOM_GROUP = 1`（在默认可见区内），但机器人自己的碰撞体在 group 3。
`render()` 里把六个组全部置 1，就是为了让渲染器**永远不可能**静默丢掉世界：
曾经把楼梯盖进隐藏组，结果物理照常爬、GIF 里机器人在爬空气（`experience.md` §1.2）。

* `mujoco.Renderer` 绑定到**一个具体的 `MjModel` 实例**，所以 `_build_model()` 里必须先
  `close()` 旧的 renderer 再重建，否则 `set_stair()` 之后画面还是上一块地形。
* `render()` 的相机参数是写死的五套 preset（见 §6），`camera` 不在 `CAMERAS` 里抛 `ValueError`。

### 3.17 其它被否决 / 被固定下来的选择

* **reset 时把同一帧复制 `history_len` 次**（而不是补零）：补零会让首帧观测里出现
  「关节角=0、上一动作=0」这种分布外样本。
* **reset 里 priming 触觉读数**：`self._touch = self.touch_sensor.read(...)` 让第一帧观测就带
  **真实接触力**，而不是 `empty_reading()` 的中性占位。
* **`_prev_support_height` 必须在 reset 归位**：`foothold` 奖励是「本步四足足下最低地形相对
  **上一步**的增量」，不归位的话跨回合第一个控制步会看到上一回合结尾的高度，凭空吃到一次
  「爬升」。注意 `_prev_support_height` 在 `_compute_reward()` 内部**无条件**推进（即使
  `foothold == 0.0`），所以它是奖励函数的一个副作用，不是纯函数。
* **`step()` 里 `_gait_phase = self._gait_t` 在推进 `_gait_t` 之前**：观测里报告的相位必须是
  本步**真正使用**的相位，否则 transition 不再是 Markov 的。
* **`touch_sensor` 只在被读时才开射线扇**：`probe_enabled = obs.stair_probe or reward.stair_probe_lift != 0.0`。
  射线扇约 60 µs / 控制步 ≈ 半个物理步，是整套触觉里最贵的一半。
* **`terminate_on_success=True`**：到达顶部就**成功终止**，而不是让机器人继续走出楼梯远端摔下去。
  `_reached_top` 在回合内**粘性**，所以更早一步摔过的话仍然算失败。

## 4. 公开 API

### 4.1 `ObsConfig`

```python
@dataclass
class ObsConfig:
    history_len: int = 10
    base_lin_vel: bool = False          # world-frame velocity -- poor feature
    base_lin_vel_body: bool = False     # body-frame velocity (preferred)
    base_height: bool = False
    base_height_support: bool = False   # use foot support plane as the reference
    foot_height: bool = False
    gait_phase: bool = False            # sin/cos gait clock
    touch: bool = False                 # 9: 4 forces + 4 slips + 1 riser flag
    stair_probe: bool = False           # 3: obstacle height / distance / hit
    stair_probe_full: bool = False      # +2: probe_slope, probe_far_step
    drift: bool = False                 # 2: lateral offset, heading error
    base_frame_size: int = 42

    @property
    def frame_size(self) -> int: ...

    @property
    def size(self) -> int: ...
```

* `frame_size`：`base_frame_size` + 已启用块的维数。各块维数：
  `base_lin_vel` 3、`base_lin_vel_body` 3、`base_height` 1、`base_height_support` **+0**
  （只改参照面）、`foot_height` 4、`gait_phase` 2、`touch` 9、`stair_probe` 3、
  `stair_probe_full` **+2 但仅在 `stair_probe=True` 时生效**（条件是 `self.stair_probe and self.stair_probe_full`）、
  `drift` 2。默认全关 ⇒ `frame_size == 42`。
* `size`：`frame_size * history_len`，默认 `42 × 10 = 420`。
* 追加顺序 = **字段声明顺序**（`_obs_frame()` 里的 `if` 序列）。

### 4.2 `RewardConfig`

47 个字段，全部形如 `字段名: 类型 = 默认值`。`[SPEC]` 标记用户明确要求的五项。

```python
@dataclass
class RewardConfig:
    # --- the five requested terms ---
    forward_vel: float = 2.5            # [SPEC] Gaussian tracking of command_vx
    forward_vel_sigma: float = 0.22
    orientation: float = -1.2           # [SPEC] roll^2 + pitch^2
    torque: float = -2.0e-4             # [SPEC] sum(tau^2)
    foot_lift: float = 0.6              # [SPEC] swing-clearance shaping
    body_collision: float = -4.0        # [SPEC] torso touching the terrain
    # --- forward motion ---
    forward_progress: float = 1.0
    forward_progress_max: float = 1.5
    forward_progress_min: float = -0.5
    # --- swing-foot lift ---
    foot_lift_target: float = 0.08      # floor of the target, not the whole target
    foot_lift_step_ratio: float = 1.4   # target = max(target, ratio * step_height)
    foot_lift_gate_speed: float = 0.25  # m/s 3-D foot speed gate
    # --- gait discovery ---
    air_time: float = 0.6
    air_time_target: float = 0.30
    air_time_max: float = 0.60
    # --- supporting shaping ---
    lateral_vel: float = -0.8
    yaw_rate: float = -0.15
    base_height: float = -3.0
    joint_vel: float = -2.0e-5
    action_rate: float = -0.02
    joint_limit: float = -0.5
    alive: float = 0.12
    termination: float = -2.0
    success: float = 20.0
    # --- staircase progress ---
    stair_progress: float = 2.5
    stair_progress_foot_ref: bool = False
    stair_progress_min_loaded_feet: int = 0
    stair_progress_force_eps: float = 5.0
    foothold: float = 0.0
    # --- torso collision ---
    body_collision_hi_scale: float = 1.0
    # --- tactile / collision ---
    riser_graze: float = -0.60
    riser_graze_mode: str = "legacy"    # "legacy" | "lifting"
    riser_graze_vz_eps: float = 0.10
    touch_impact: float = -0.50
    touch_impact_target: float = 95.0
    touch_impact_step_scale: float = 0.0
    touch_slip: float = -0.35
    touch_slip_safe: float = 0.55
    touch_excess_clip: float = 5.0
    stair_probe_lift: float = 1.20
    leg_collision: float = -0.25
    # --- walking straight ---
    straight_line: float = -20.0
    straight_line_max: float = 1.00
    heading: float = -8.0
    heading_max: float = 0.60
    lateral_vel_track: float = 0.5
    lateral_vel_sigma: float = 0.15
```

各项公式（`lin = base_lin_vel_world`、`rpy = base_rpy`、
`tau = data.actuator_force`、`h = stair.step_height`、`lift_target = max(foot_lift_target,
foot_lift_step_ratio * h)`）：

| key | 公式 |
| --- | --- |
| `forward_vel` | `forward_vel * exp(-(lin[0] - command_vx)² / forward_vel_sigma)` |
| `forward_progress` | `forward_progress * clip(lin[0], forward_progress_min, forward_progress_max)` |
| `orientation` | `orientation * (rpy[0]² + rpy[1]²)` |
| `torque` | `torque * sum(tau²)` |
| `foot_lift` | `foot_lift * mean(gate * lift)`，`gate = clip(‖Δfoot/foot_dt‖ / max(foot_lift_gate_speed,1e-6), 0, 1)`，`lift = clip(clearance / lift_target, 0, 1)` |
| `body_collision` | `body_collision * collision_scale * float(trunk_contact)`，`collision_scale = body_collision_hi_scale`（仅当 `trunk_contact_adaptive` 且 `h >= trunk_contact_hi_step_h`，否则 1.0） |
| `leg_collision` | `leg_collision * float(leg_contact)` |
| `air_time` | `air_time * sum((clip(air - air_time_target, 0, None) - clip(air - air_time_max, 0, None)) * landed)`；`air_time == 0.0` 时恒 `0.0` |
| `lateral_vel` | `lateral_vel * lin[1]²` |
| `yaw_rate` | `yaw_rate * (base_ang_vel_body[2] - command_yaw_rate)²` |
| `base_height` | `base_height * (base_pos[2] - terrain_height_under_base - nominal_pose)²` |
| `joint_vel` | `joint_vel * sum(joint_vel²)` |
| `action_rate` | `action_rate * sum((action - _prev_action)²)` |
| `joint_limit` | `joint_limit * sum(clip(lo - q, 0, None) + clip(q - hi, 0, None))` |
| `alive` | `alive`（常数） |
| `stair_progress` | `stair_progress * max(0, progress_ref - _start_terrain_height)`；`progress_ref = _support_height()`（若 `stair_progress_foot_ref`，且当 `stair_progress_min_loaded_feet > 0` 而承重脚数不足时回退到 `_start_terrain_height`）否则 `terrain_height_under_base` |
| `foothold` | `foothold * clip((_support_height() - _prev_support_height) / max(h, 1e-6), 0, 1)`；`foothold == 0.0` 时恒 `0.0`（但 `_prev_support_height` 仍更新） |
| `riser_graze` | `"lifting"` 模式：`riser_graze * float(stuck)`；否则 `riser_graze * float(touch.riser_graze.any())` |
| `touch_impact` | `touch_impact * sum(over²)`，`over = clip((foot_normal_force[landed] - ref)/ref, 0, touch_excess_clip)`，`ref = max(touch_impact_target, 1e-6)`（`touch_impact_step_scale != 0` 时按 `h/0.05 - 1` 放大） |
| `touch_slip` | `touch_slip * sum(clip(foot_slip - touch_slip_safe, 0, touch_excess_clip)² * loaded)`，`loaded = foot_normal_force > cfg.touch.slip_loaded_force` |
| `stair_probe_lift` | `stair_probe_lift * touch.trigger(cfg.touch) * mean(lift[:2])`（只用前两只脚） |
| `straight_line` | `straight_line * y_err²`，`y_err = clip(lateral_offset, ±straight_line_max)` |
| `heading` | `heading * h_err²`，`h_err = clip(heading_error, ±heading_max)` |
| `lateral_vel_track` | `lateral_vel_track * exp(-(lin[1] - command_vy)² / lateral_vel_sigma²)` |

### 4.3 `EnvConfig`

44 个字段。

```python
@dataclass
class EnvConfig:
    timestep: float = 0.002
    decimation: int = 10                    # -> 50 Hz control
    episode_length_s: float = 12.0
    action_mode: str = "residual"           # "residual" | "absolute"
    action_scale: float = 0.30              # rad, policy authority (residual only)
    command_vx: float = 0.55                # m/s forward command
    command_vy: float = 0.0
    command_yaw_rate: float = 0.0
    gait_prior: bool = False
    gait: GaitParams = field(default_factory=GaitParams.default_trot)
    gait_gain: float = 1.0
    gait_phase_randomize: bool = False
    kp: tuple[float, float, float] = (30.0, 30.0, 45.0)
    kd: tuple[float, float, float] = (0.6, 0.6, 0.9)
    terminate_contact: str = "trunk"
    terminate_body_contact: bool = True
    terminate_contact_steps: int = 1
    contact_debounce_adaptive: bool = False
    contact_debounce_hi_step_h: float = 0.105
    contact_debounce_hi_steps: int = 1
    trunk_contact_adaptive: bool = False
    trunk_contact_hi_step_h: float = 0.105
    terminate_height_ref: str = "base"
    terminate_min_height: float = 0.13
    terminate_max_tilt: float = 1.0         # rad, roll or pitch
    terminate_on_success: bool = True
    reset_z: float = 0.32
    reset_joint_noise: float = 0.06
    reset_xy_noise: float = 0.12
    reset_yaw_noise: float = 0.10
    flat_success_distance: float = 3.0      # flat stages only
    spawn_back: float = 0.55                # m in front of the first riser
    randomize_friction: bool = False
    randomize_mass: bool = False
    randomize_latency: bool = False
    randomize_imu: bool = False
    randomize_friction_range: tuple[float, float] = (0.7, 1.3)
    randomize_mass_range: tuple[float, float] = (0.92, 1.08)
    randomize_latency_max: int = 1          # control steps
    randomize_imu_sigma: float = 0.02       # rad
    randomize_scale: float = 1.0
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    touch: TouchConfig = field(default_factory=TouchConfig)

    @property
    def control_dt(self) -> float: ...
```

* `control_dt = timestep * decimation`（默认 `0.002 × 10 = 0.02` s，即 50 Hz 控制 / 500 Hz 物理）。
* `obs` / `reward` / `touch` 是嵌套 dataclass，就地改字段即可（如
  `cfg.obs.touch = True`、`cfg.reward.stair_progress_foot_ref = True`）。
* `kp` / `kd` 是**每条腿三个**（hip, thigh, calf），在 `__init__` 里
  `np.tile(cfg.kp, 4)` 展开成 12 维。

### 4.4 `Go2StairsEnv`

```python
class Go2StairsEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    CAMERAS = ("side", "track", "chase", "front", "top")

    def __init__(self, stair: StairSpec | None = None, cfg: EnvConfig | None = None,
                 *, seed: int | None = None) -> None
```

* `stair=None` ⇒ `StairSpec()`；`cfg=None` ⇒ `EnvConfig()`；`seed` 只用于建 `self._rng`
  （`np.random.default_rng(seed)`）。
* 副作用：编译模型、建 `MjData`、建 `TouchSensor`、定义 `action_space`（`Box(-1, 1, (12,),
  float32)`）与 `observation_space`（`Box(-inf, inf, (obs.size,), float32)`）。
* 异常：模型编译失败时 `compile_scene` 抛 `RuntimeError("MJCF compile failed; dumped to ...")`
  或 `FileNotFoundError`（缺 mesh）。

```python
    def _build_model(self) -> None
    def set_stair(self, stair: StairSpec) -> None
    def set_gait_prior(self, enable: bool, gait: GaitParams | None = None) -> None
    def set_domain_randomization(self, scale: float | None = None, **flags: bool) -> None
    def _success_target_x(self) -> float
    def _success_reached(self) -> bool
    def reset(self, *, seed: int | None = None, options=None)
    def step(self, action: np.ndarray)
    def render(self, camera: str = "side", *, width: int = 640, height: int = 480)
    def close(self) -> None
```

* `set_stair(stair)`：换地形并**重新编译模型**；调用方之后**必须** `reset()`（旧的
  `MjModel`/`MjData` 已被替换，旧状态无意义）。同时刷新 `_success_x` 并把 `_reached_top` 置 `False`。
* `set_gait_prior(enable, gait=None)`：在线开关先验；`gait` 不为 `None` 时同时替换 `cfg.gait`；
  无论哪条分支都把 `_gait_t` / `_gait_phase` 清零。不改观测量，所以不重建 `observation_space`。
* `set_domain_randomization(scale=None, **flags)`：合法 flag 名只有
  `randomize_friction` / `randomize_mass` / `randomize_latency` / `randomize_imu`；
  未知名字抛 `AttributeError(f"unknown domain-randomization flag(s) {sorted(unknown)}; valid: {sorted(valid)}")`；
  `scale` 被 `np.clip(scale, 0.0, 1.0)` 后写入 `cfg.randomize_scale`。
* `_success_target_x()`：有上行台阶时返回 `stair.up_end_x`；平地阶段返回
  `stair.start_x - spawn_back + flat_success_distance`。
* `_success_reached()`：阶梯条件 = `base_pos[0] > stair.up_end_x` **且**
  `terrain_height_under_base >= stair.crest_height - 1e-9`；平地条件 = `base_pos[0] > _success_x`。
* `reset(seed=None, options=None)`：返回 `(observation, {})`（info 是空 dict）。
* `step(action)`：返回 `(observation, reward: float, terminated: bool, truncated: bool, info: dict)`。
  `action` 会被 `np.asarray(..., float64).clip(-1.0, 1.0)`。
* `render(camera="side", *, width=640, height=480)`：返回 `(height, width, 3)` 的 `uint8` RGB 数组；
  `camera` 不在 `CAMERAS` 时抛 `ValueError(f"未知相机 {camera!r}, 可选 {self.CAMERAS}")`。
  相机尺寸变了会重建 `mujoco.Renderer`。
* `close()`：释放 renderer，把 `_render_size` / `_scene_option` 置 `None`。

只读属性（全部返回 `np.ndarray` 或 `float`，无 setter）：

```python
    @property
    def base_pos(self) -> np.ndarray              # data.qpos[0:3]，世界系
    @property
    def base_rpy(self) -> np.ndarray              # quat_to_rpy(qpos[3:7])
    @property
    def base_quat(self) -> np.ndarray             # data.qpos[3:7] (w,x,y,z)
    @property
    def base_lin_vel_world(self) -> np.ndarray    # data.qvel[0:3]
    @property
    def base_lin_vel_body(self) -> np.ndarray     # R(q)^T @ qvel[0:3]
    @property
    def base_ang_vel_body(self) -> np.ndarray     # data.qvel[3:6]（陀螺读数）
    @property
    def joint_pos(self) -> np.ndarray             # qpos[qpos_joint_adr]
    @property
    def joint_vel(self) -> np.ndarray             # qvel[qvel_joint_adr]
    @property
    def foot_pos(self) -> np.ndarray              # (4,3) geom_xpos[foot_geom_ids]，球心
    @property
    def terrain_height_under_base(self) -> float  # stair.height_at(base_pos[0])
    @property
    def yaw(self) -> float                        # base_rpy[2]
    @property
    def lateral_offset(self) -> float             # base_pos[1] - stair.center_y
    @property
    def spawn_lateral_offset(self) -> float       # base_pos[1] - self._y0
    @property
    def heading_error(self) -> float              # yaw 相对世界 +x，wrap 到 (-pi, pi]
    @property
    def spawn_heading_error(self) -> float        # yaw - self._yaw0，wrap 到 (-pi, pi]
    @property
    def foot_clearance(self) -> np.ndarray        # (4,) 脚面到足下地形的间隙
```

内部辅助（下划线开头，但**被文档化以便 grep**，也是测试直接驱动的入口）：

```python
    def _terrain_height(self, x: np.ndarray | float) -> np.ndarray
    def _projected_gravity(self) -> np.ndarray           # R^T @ (0,0,-1)，机身系重力方向
    def _obs_frame(self) -> np.ndarray                   # float64，长度 obs.frame_size
    def _observation(self) -> np.ndarray                 # float32，长度 obs.size
    def _compute_reward(self, action: np.ndarray, trunk_contact: bool,
                        foot_contact: np.ndarray | None = None,
                        leg_contact: bool = False) -> tuple[float, dict[str, float]]
    def _foot_contact_mask(self) -> np.ndarray           # (4,) bool
    def _advance_foot_air_time(self, foot_contact: np.ndarray) -> None
    def _contact_flags(self) -> tuple[bool, bool]        # (trunk_contact, leg_contact)
    def _effective_contact_steps(self) -> int
    def _trunk_contact_relaxed(self) -> bool
    def _contact_fatal(self, trunk_contact: bool, leg_contact: bool) -> bool
    def _support_height(self) -> float
    def _support_height_mean(self) -> float
    def _support_height_load(self) -> float
    def _height_reference(self) -> float                 # 仅 obs.base_height 使用
    def _loaded_foot_count(self, eps: float) -> int
    def _fall_reference_height(self) -> float
    def _terminated(self, trunk_contact: bool, leg_contact: bool) -> tuple[bool, bool]
    def _gait_action(self) -> np.ndarray
    def _target_from_action(self, action: np.ndarray) -> np.ndarray
    def _apply_domain_randomization(self) -> None
    def _apply_latency(self, ctrl: np.ndarray) -> np.ndarray
```

* `_terrain_height(x)`：接受标量或数组；**平地阶段（`n_up == 0 and n_down == 0`）直接返回
  `np.zeros_like`**，不走 `stair.height_at()`。
* `_compute_reward(...)` 返回 `(float, dict[str, float])`，且
  `reward == sum(terms.values())` 恒成立（返回值里**没有** `terminal`，见 §9）。
  `foot_contact=None` 时它内部自己调 `_foot_contact_mask()`；`leg_contact=False` 时
  `terms["leg_collision"] == 0.0`。
  **副作用**：每次都推进 `self._prev_support_height`。
* `_advance_foot_air_time(foot_contact)`：一次控制步调用一次（`step` 在奖励之后调），
  `_foot_air_time = where(foot_contact, 0.0, _foot_air_time + control_dt)`，并更新 `_prev_foot_contact`。
* `_contact_fatal`：**会写** `_trunk_contact_streak` / `_leg_contact_streak`。
* `_target_from_action(action)`：`residual` 分支
  `home_angles + _gait_action() + action_scale * action`；
  `absolute` 分支 `lo + (action + 1) * 0.5 * (hi - lo)`（先验在 absolute 下**被故意忽略**，
  残差先验在那里没有意义）。两支都用 `np.clip(target, act_low, act_high)` 收尾。

`step()` 返回的 `info` 字段（键名逐字）：`reward_terms`、`body_contact`、`trunk_contact`、
`leg_contact`、`terminate_contact`、`terminate_contact_steps`、
`terminate_contact_steps_effective`、`trunk_contact_relaxed`、`trunk_contact_streak`、
`leg_contact_streak`、`base_x`、`base_y`、`base_z`、`terrain_height`、`forward_vel`、`roll`、
`pitch`、`crest_height`、`reached_top`、`is_success`、`fallen`、`gait_prior`、
`lateral_offset`、`spawn_lateral_offset`、`heading_error`、`spawn_heading_error`、`probe_hit`、
`probe_step`、`probe_range`、`probe_trigger`、`riser_graze`、`total_normal_force`、`max_slip`。
其中 `fallen = bool(terminated) and not self._reached_top`。

## 5. 内部结构与数据流

### 5.1 构造

```
Go2StairsEnv.__init__()
  → _build_model()
      → build_terrain_xml(self.stair)          [tools/terrain]
      → compile_scene(..., opts=SceneOptions(timestep=cfg.timestep), floor_geom=default_floor_geom())
          → build_scene_xml() → _go2_inner_xml() → go2_assets()
          → mujoco.MjModel.from_xml_string()
          → extract_meta(model) → Go2Meta
      → mujoco.MjData(model)
      → TouchSensor(model, data, meta, cfg.touch, probe_enabled=obs.stair_probe or reward.stair_probe_lift != 0.0)
      → （若已有 renderer）renderer.close()
  → act_low/act_high/home_angles ← meta.joint_ranges / meta.home_joint_angles
  → action_space / observation_space
  → _kp = tile(cfg.kp, 4) ; _kd = tile(cfg.kd, 4)
  → _max_steps = int(cfg.episode_length_s / cfg.control_dt)     # 600
  → 其余状态字段初始化（_hist、_prev_action、_reached_top、streaks、_prev_foot_pos …）
```

`set_stair()` → `_build_model()` → `_success_target_x()` → `_reached_top = False`。

### 5.2 `reset()` 管线（严格按源码顺序）

1. `super().reset(seed=seed)`；`seed is not None` 时 `self._rng = np.random.default_rng(seed)`。
2. `_apply_domain_randomization()` —— **在 `mj_resetData` 之前**。
3. `mujoco.mj_resetData(model, data)`。
4. 采样 `x = stair.start_x - spawn_back + U(-reset_xy_noise, +reset_xy_noise)`、
   `y = stair.center_y + U(...)`、`yaw = U(-reset_yaw_noise, +reset_yaw_noise)`。
5. `data.qpos[:] = home_qpos(meta, x=x, y=y, z=terrain_height_under_base + reset_z, yaw=yaw)`。
6. `data.qpos[qpos_joint_adr] += U(-reset_joint_noise, +reset_joint_noise, 12)`。
7. `data.qvel[:] = U(-0.05, 0.05, model.nv)`（**整机**，含浮动基座）。
8. `mujoco.mj_forward(model, data)`。
9. `nominal_pose = float(meta.home_qpos[2])`（0.27）。
10. `_start_terrain_height = terrain_height_under_base`。
11. `_y0 = base_pos[1]`、`_yaw0 = base_rpy[2]`。
12. `_prev_action = zeros(12)`、`_last_action = zeros(12)`。
13. `_step_count = 0`、`_reached_top = False`、两条 streak = 0、`_pending_ctrl.clear()`。
14. `_prev_foot_pos = foot_pos.copy()`。
15. `_prev_support_height = _support_height()`。
16. `_foot_air_time = zeros(4)`、`_prev_foot_contact = _foot_contact_mask()`。
17. `self._touch = touch_sensor.read(base_pos=..., yaw=..., terrain_height=terrain_height_under_base)`
    （priming）。
18. 步态时钟：`gait_prior and gait_phase_randomize and gait.freq > 0` 时
    `_gait_t = U(0, gait.period)`，否则 `0.0`；`_gait_phase = _gait_t`。
19. `_hist.clear()`；`frame = _obs_frame()`；把 `frame.copy()` 追加 `history_len` 次。
20. `return self._observation(), {}`。

### 5.3 `step()` 管线（严格按源码顺序）

1. `action = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)`。
2. `_gait_phase = _gait_t`（本步真正使用的相位）。
3. `target = _target_from_action(action)` → `home + _gait_action() + action_scale*action`
   → `clip(act_low, act_high)`。
4. `_gait_t += control_dt`。
5. `target = _apply_latency(target)`（每控制步一次，priming 用第一条命令）。
6. `for _ in range(decimation)`：`q = qpos[qpos_joint_adr]`、`qd = qvel[qvel_joint_adr]`、
   `tau = _kp * (target - q) - _kd * qd`、`data.ctrl[:] = clip(tau, ctrl_ranges[:,0], ctrl_ranges[:,1])`、
   `mujoco.mj_step(model, data)`。
7. `_step_count += 1`。
8. `trunk_contact, leg_contact = _contact_flags()`；`body_contact = bool(trunk or leg)`。
9. `foot_contact = _foot_contact_mask()`。
10. `self._touch = touch_sensor.read(...)` —— **必须**在物理步之后、奖励之前取。
11. `terminated, truncated = _terminated(trunk_contact, leg_contact)`。
12. 成功判定：`if _success_reached(): _reached_top = True; if cfg.terminate_on_success and not
    terminated: terminated = True; truncated = False`。
13. `reward, terms = _compute_reward(action, trunk_contact, foot_contact, leg_contact=leg_contact)`。
14. `_advance_foot_air_time(foot_contact)`。
15. `terminal = reward.success if _reached_top else reward.termination`（仅 `terminated` 时非零）；
    `reward += terminal`；`terms["terminal"] = float(terminal)`。
16. `_prev_action = _last_action.copy()`；`_last_action = action.copy()`。
17. `_prev_foot_pos = foot_pos.copy()`（脚的 xy 位移必须恰好跨一个控制步）。
18. `_hist.append(_obs_frame())`。
19. 组 `info`，返回 `(_observation(), float(reward), terminated, truncated, info)`。

### 5.4 `_obs_frame()` 装配顺序

```
_obs_frame()
  → base_ang_vel_body（若 randomize_imu，加 _imu_bias）
  → parts = [joint_pos(12), joint_vel(12), base_rpy(3), ang_vel(3), _last_action(12)]
  → if obs.base_lin_vel:      append base_lin_vel_world(3)
  → if obs.base_lin_vel_body: append base_lin_vel_body(3)
  → if obs.base_height:       append [base_pos[2] - _height_reference()](1)
  → if obs.foot_height:       append foot_clearance(4)
  → if obs.gait_phase:        append [sin(theta), cos(theta)]，theta = 2*pi*gait.freq*_gait_phase
  → if obs.touch:             append _touch.foot_obs(cfg.touch)(9)
  → if obs.stair_probe:       append _touch.probe_obs_full(...)(5) 或 _touch.probe_obs(...)(3)
  → if obs.drift:             append [lateral_offset, heading_error](2)
  → np.concatenate(parts).astype(float64)
```

### 5.5 奖励调用链

```
_compute_reward(action, trunk_contact, foot_contact, leg_contact)
  → base_rpy / base_lin_vel_world / nominal_pose
  → foot_pos, foot_clearance, stair.step_height, stair_progress_foot_ref
      → _support_height() / _loaded_foot_count(stair_progress_force_eps)
  → （foothold）_support_height() → 写 self._prev_support_height
  → self._touch.{riser_graze, foot_normal_force, foot_slip, trigger}
  → _foot_contact_mask()（若 foot_contact is None）
  → sum(terms.values()), terms
```

### 5.6 终止调用链

```
_terminated(trunk_contact, leg_contact)
  → base_rpy
  → _fall_reference_height()
      → "support_mean" → _support_height_mean()
      → "support_load" → _support_height_load()
      → "support"      → _support_height()
      → 其它           → terrain_height_under_base
  → (_support_height* → stair.height_at(foot_pos[:,0])) / (_support_height_load → _touch.foot_normal_force)
  → _contact_fatal(trunk, leg)
      → _trunk_contact_relaxed()  （需要 trunk_contact_adaptive + terminate_contact=="trunk" + h >= hi_step_h）
      → _effective_contact_steps()（contact_debounce_adaptive + h >= hi_step_h → hi_steps）
      → 写 _trunk_contact_streak / _leg_contact_streak
  → truncated = _step_count >= _max_steps
  → (fell or contact_fatal), truncated
  fell = height < terminate_min_height or |roll| > terminate_max_tilt or |pitch| > terminate_max_tilt
```

### 5.7 `render()`

```
render(camera, width, height)
  → 校验 camera ∈ CAMERAS（否则 ValueError）
  → 尺寸变了或没有 renderer：Renderer(model, h, w)；MjvCamera()；MjvOption() + mjv_defaultOption()
      → for g in range(6): geomgroup[g] = 1
  → mjv_defaultFreeCamera(model, cam)
  → 按 preset 设 azimuth / elevation / distance / lookat
  → renderer.update_scene(data, camera=cam, scene_option=_scene_option)
  → renderer.render()
```

## 6. 关键常量与默认值

**类级常量**

| 名称 | 值 |
| --- | --- |
| `Go2StairsEnv.metadata` | `{"render_modes": ["rgb_array"], "render_fps": 50}` |
| `Go2StairsEnv.CAMERAS` | `("side", "track", "chase", "front", "top")` |
| `self.num_act` | `12` |
| `self.nominal_pose` | `float(meta.home_qpos[2])` = `0.27` m |

**时间 / 步数**

| 项 | 值 |
| --- | --- |
| `timestep` | `0.002` s（500 Hz 物理） |
| `decimation` | `10` |
| `control_dt` | `0.02` s（50 Hz 控制） |
| `episode_length_s` | `12.0` |
| `_max_steps` | `int(12.0 / 0.02) = 600` |
| `metadata["render_fps"]` | `50` |

**控制增益与动作**

| 项 | 值 |
| --- | --- |
| `kp` | `(30.0, 30.0, 45.0)` → `_kp = [30,30,45] × 4` |
| `kd` | `(0.6, 0.6, 0.9)` → `_kd = [0.6,0.6,0.9] × 4` |
| `action_mode` | `"residual"` |
| `action_scale` | `0.30` rad |
| `command_vx` | `0.55` m/s |
| `command_vy` | `0.0` m/s |
| `command_yaw_rate` | `0.0` rad/s |
| `gait_prior` | `False` |
| `gait` | `GaitParams.default_trot()`（`freq=6.0`，`amp_th=0.24914502679724`，`amp_ca=0.5518579715236906`，`ph_th=1.6754541487333945`，`ph_ca=-2.146505768979483`，`off_th=0.25539290883226595`，`off_ca=0.09025271647195035`，`calf_shape="sin"`，`action_scale=0.90`，`leg_phase=(0.0, pi, pi, 0.0)`） |
| `gait_gain` | `1.0` |
| `gait_phase_randomize` | `False` |

**reset 噪声与出生**：`reset_z=0.32`、`reset_joint_noise=0.06`、`reset_xy_noise=0.12`、
`reset_yaw_noise=0.10`、`spawn_back=0.55`、`flat_success_distance=3.0`、
初速度 `U(-0.05, 0.05)`（施加到全部 `nv`）。

**终止**：`terminate_contact="trunk"`、`terminate_body_contact=True`、
`terminate_contact_steps=1`、`contact_debounce_adaptive=False`、
`contact_debounce_hi_step_h=0.105`、`contact_debounce_hi_steps=1`、
`trunk_contact_adaptive=False`、`trunk_contact_hi_step_h=0.105`、
`terminate_height_ref="base"`、`terminate_min_height=0.13`、`terminate_max_tilt=1.0`、
`terminate_on_success=True`。

**域随机化**：四个 flag 全 `False`；`randomize_friction_range=(0.7, 1.3)`、
`randomize_mass_range=(0.92, 1.08)`、`randomize_latency_max=1`、`randomize_imu_sigma=0.02`、
`randomize_scale=1.0`；摩擦裁剪到 `[0.05, 3.0]`。

**观测**：`history_len=10`、`base_frame_size=42`、`size=420`；各块维数见 §4.1。

**奖励**：47 个字段的默认值见 §4.2；其中 `[SPEC]` 五项
`forward_vel=2.5`（σ`0.22`）/ `orientation=-1.2` / `torque=-2.0e-4` / `foot_lift=0.6` /
`body_collision=-4.0`；`alive=0.12`、`termination=-2.0`、`success=20.0`、
`touch_excess_clip=5.0`、`straight_line=-20.0`、`heading=-8.0`、`foot_lift_target=0.08`、
`foot_lift_step_ratio=1.4`、`foot_lift_gate_speed=0.25`、`riser_graze=-0.60`、
`touch_impact=-0.50`、`touch_impact_target=95.0`、`touch_slip=-0.35`、`touch_slip_safe=0.55`、
`stair_probe_lift=1.20`、`leg_collision=-0.25`、`stair_progress=2.5`、`foothold=0.0`、
`air_time=0.6` / `air_time_target=0.30` / `air_time_max=0.60`、
`lateral_vel=-0.8` / `yaw_rate=-0.15` / `base_height=-3.0` / `joint_vel=-2.0e-5` /
`action_rate=-0.02` / `joint_limit=-0.5` / `body_collision_hi_scale=1.0`。

**渲染 preset**（`render()` 内写死）

| camera | azimuth | elevation | distance | lookat |
| --- | --- | --- | --- | --- |
| `side` | 90.0 | -12.0 | 4.0 | `(stair.start_x + 1.0, 0.0, 0.3)` |
| `track` | 90.0 | -9.0 | 2.5 | `(base_x + 0.35, base_y, base_z + 0.02)` |
| `chase` | 135.0 | -22.0 | 3.0 | `(base_x + 0.6, base_y, base_z + 0.10)` |
| `front` | 180.0 | -10.0 | 3.0 | `(base_x + 1.2, base_y, base_z + 0.10)` |
| `top` | 90.0 | -85.0 | 4.5 | `(base_x + 0.8, base_y, base_z)` |

## 7. 与其它模块的关系

```
gymnasium / mujoco / numpy
        ↑
robotdog.env
   ↑ 依赖（上游，本模块 import 它们）
   ├── robotdog.gait            : GaitParams（cfg.gait 的默认值来源）
   ├── robotdog.go2_model       : SceneOptions, compile_scene, default_floor_geom, home_qpos,
   │                              quat_to_mat, quat_to_rpy
   └── robotdog.tools.terrain   : StairSpec, build_terrain_xml
       robotdog.tools.touch     : TouchConfig, TouchReading, TouchSensor, empty_reading
   ↓ 被依赖（下游）
   robotdog.training / curriculum / evaluate / render
   scripts/*（train.py、eval.py、render_checkpoint.py、verifications/*）
```

方向为什么是这样（`docs/README.md` 的分层规则）：

* 依赖必须**单向** `训练层 → 环境层 → 工具层`。`tools/` 里**不 import `robotdog.env`**，
  所以 `terrain` / `touch` 可以是纯几何 / 纯 MuJoCo 读取，不知道奖励、课程、PPO 的存在。
* `render.py` 因为要驱动环境而依赖 `env`，所以它留在 `robotdog/` 顶层而**没有**进 `tools/`
  —— 否则 `tools ↔ env` 成环。
* `env` 只从 `gait` 拿**轨迹**、从 `go2_model` 拿**模型与 id**、从 `terrain` 拿**几何剖面**、
  从 `touch` 拿**传感器读数**。它自己不做这四件事中的任何一件。
* 反过来的耦合被刻意避免：`env` 不 import `training`/`curriculum`/`evaluate`/`render`。
  课程通过 `set_stair()` + `set_domain_randomization()` 从外部驱动环境，
  而不是环境去问课程「我现在在第几段」。

## 8. 实测与经验教训

本节**只列引用**，结论请直接读原文，不要在这里复述：

* 奖励设计侧：`experience.md` §2.1（惩罚项无上限毁掉 value function）、§2.2（高斯前进奖励在
  vx=0 非零 ⇒ 学会站桩）、§2.3（抬升目标写死 0.08 m ⇒ 在 8 cm 完全饱和）、§2.4（腿碰地立刻判死
  杀掉试错）、§2.5（蹭立面惩罚与任务方向打架）、§2.6（落地力阈值是常数）、§2.7（`stair_progress`
  奖励「躯干冲线」）、§2.8（走直线权重太小）。
* 观测设计侧：`experience.md` §3.1（策略被打分却看不到被打分的量）、§3.2（漂移量按出生位姿量）、
  §3.3（`base_height` 参照面只能动观测）、§3.4（残差策略不知步态相位）。
* 物理 / 几何侧：`experience.md` §1.4（摔倒判据用阶跃函数 ⇒ 幽灵摔倒）、§1.2（地形 geomgroup
  被 MuJoCo 默认隐藏）。
* 训练与课程侧：`experience.md` §4.1（课程毕业判据把梯度饿死）、§4.2（`VecNormalize` 奖励归一化
  只是温和稳定器）；`train.md` §1（训练组织与默认课程表）、§3.1（参考 checkpoint 能力带）、
  §3.2（三条记录的配方）、§3.4（从零训练能达到什么程度）、§3.5（消融 A~F）。
* 工具与测量侧：`experience.md` §5.5（并行 run 数上限≈3）、§5.7（10 回合评估给出 ±15 pt 错觉）、
  §5.8（`--resume` 静默忽略超参、`--tread` 保持 0.50 m、沙箱禁命名管道 ⇒ 只能 `DummyVecEnv`）。

## 9. 已知限制与坑

1. **11 cm 是墙。** 策略的摆动相脚掌峰值抬升在任何台阶高度上都饱和在约 `0.09 m`，
   而纯开环先验在平地上自己就能抬 `0.1124 m` —— 所以缺的**不是峰值高度，而是摆动相里的
   向前落脚**。详见 `experience.md` §6 与 `train.md` §3.6（正确下一步是显式落脚相 IK 替换
   `gait.radians()`，不是继续调奖励权重）。
2. **从零训练 `--stage medium` 单独跑学不出来**（约 360k 步 / 440 回合仍 0.0% 成功率），
   `README.md` §4.1 的那条配方**不可复现** —— 见 `train.md` §3.3。
3. **奖励里的 `foot_clearance` 参照面在立面处是 x 的阶梯函数**（`stair.height_at(x)` 按
   `(x - start_x) // tread` 取整）。在立面附近同一步内参照面会跳一个 riser，抬升读数与真实间隙
   不一致；`experience.md` §6.1 给了「旧口径在 11 cm 上读数比 8 cm 还大」的实测。
4. **`_compute_reward()` 的返回值里没有 `terminal`。** 它返回 `sum(terms.values())`，而
   `terminal` 是 `step()` 在 `terms` 上补写的。因此「`sum(reward_terms) == reward`」这条恒等式
   对 `info["reward_terms"]` 成立（25 个键），对 `_compute_reward()` 的 `terms` 也成立
   （24 个键，无 `terminal`），但**不是同一个 dict**。要按 key 逐项对账时注意这一点。
5. **`_compute_reward()` 有副作用**：每次都写 `self._prev_support_height`（即使 `foothold == 0.0`）。
   单独调用它两次会破坏 `foothold` 的差分基准。
6. **`_contact_fatal()` 也会写状态**（两条 streak）。它是 `_terminated()` 的一部分，
   「一次 `_terminated()` == 一个控制步」是它的契约；不要在同一个控制步里调两次。
7. **`terminate_height_ref="base"` 的幽灵摔倒**：样本 `base_x=1.503 → Δref=+0.1100`、
   `base_x=1.491 → Δref=-0.1100`（`experience.md` §1.4）。课程配方统一用 `"support"`；
   改默认值会破坏旧 checkpoint 的复现，所以默认保持 `"base"`。
8. **`obs.base_height_support` 不能顺手接进奖励**：同时接观测与奖励使均回报从 +554 掉到 -674
   （`experience.md` §3.3）。要基于脚的奖励参照面请用 `reward.stair_progress_foot_ref`。
9. **`stair_progress` 默认口径是「躯干冲线」**：只有 `stair_progress_foot_ref=True`
   才改用四足足下最低地形，且 `stair_progress_min_loaded_feet > 0` 时会用
   `_loaded_foot_count(force_eps)` 门控，承重脚不足时把 `progress_ref` 回退到
   `_start_terrain_height`（即该项为 0）。见 `experience.md` §2.7。
10. **`riser_graze_mode="legacy"` 会罚正在跨越的脚**；高台阶上该罚项可以超过 `foot_lift` 的实得
    （`experience.md` §2.5），需要 `"lifting"` 模式。
11. **`_support_height()` 在楼梯几何之外返回 0**：`StairSpec.height_at` 在
    `x < start_x` 或超过 `up_end_x + top_landing`（`n_down == 0` 时）返回 `0.0`。
    机器人走过顶部平台远端之后 `support` 会掉回 0，`terminate_height_ref="support"` 的
    高度判据在那里会失效（这也是 `terminate_on_success=True` 默认开启的原因之一）。
12. **`_foot_contact_mask()` / `_contact_flags()` 是 O(ncon) 的 Python 循环**，每控制步各跑一次
    （`_foot_contact_mask` 在 `step` + `_compute_reward` 里可能各跑一次）。接触数大的场景是热点。
13. **`_apply_latency` 的缓冲区只增不清**（除了 `reset()` / `_apply_domain_randomization()` 里的
    `_pending_ctrl.clear()`）；`self._pending_ctrl[0]` 在缓冲区填满前作为 priming 返回，
    语义是「重复第一条命令」，不是「零延迟」。
14. **`render()` 的报错文本是中文**：`ValueError(f"未知相机 {camera!r}, 可选 {self.CAMERAS}")`；
    域随机化的报错是
    `AttributeError(f"unknown domain-randomization flag(s) {sorted(unknown)}; valid: {sorted(valid)}")`。
15. **编译失败会写盘**：`compile_scene` 在 `write_to is None` 时把失败的 XML 落到
    `.generated/failed_scene.xml` 并抛 `RuntimeError("MJCF compile failed; dumped to ...")`。
16. **`subprocess` / `SubprocVecEnv` 在本沙箱不可用**（禁用命名管道），所以并行度只能靠单进程
    多环境；这也决定了 8 环境的天花板（`experience.md` §5.8、`gait.py` 模块说明）。
17. **`stair_probe_lift` 只奖赏前两只脚**（`np.mean(lift[:2])`，顺序 FL, FR, RL, RR），
    与 `TouchReading.trigger()` 的触发条件共同决定；改动脚序会静默改变该奖励的含义。

## 10. 示例用法

```python
"""最小可运行：建环境 → 用默认步态先验走一集 → 打印终止原因。"""
import numpy as np

from robotdog.env import EnvConfig, Go2StairsEnv
from robotdog.gait import GaitParams
from robotdog.tools.terrain import StairSpec

cfg = EnvConfig()
cfg.obs.base_lin_vel_body = True     # 42 -> 45 维/帧
cfg.obs.touch = True                 # +9
cfg.obs.drift = True                 # +2  => 53 维/帧，530 维观测
cfg.reward.stair_progress_foot_ref = True
cfg.reward.stair_progress_min_loaded_feet = 2
cfg.terminate_height_ref = "support"            # 避免幽灵摔倒
cfg.gait_prior = True
cfg.gait = GaitParams.default_trot()
cfg.action_scale = 0.30                         # 策略权限（与先验幅度无关）

stair = StairSpec(step_height=0.05, tread=0.50, n_up=8, n_down=0)
env = Go2StairsEnv(stair=stair, cfg=cfg, seed=0)

obs, info = env.reset(seed=0)
assert obs.shape == (env.cfg.obs.size,)

total = 0.0
terminated = truncated = False
for _ in range(env._max_steps):
    action = np.zeros(env.action_space.shape, dtype=np.float32)   # 纯先验，零残差
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    if terminated or truncated:
        break

print(f"return={total:.2f} steps={env._step_count} success={info['is_success']} "
      f"fallen={info['fallen']} x={info['base_x']:.3f} crest={info['crest_height']:.3f}")
print({k: round(v, 4) for k, v in info["reward_terms"].items()})

frame = env.render("track", width=640, height=480)   # (480, 640, 3) uint8
env.close()
```
