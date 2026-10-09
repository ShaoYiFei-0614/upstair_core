# `robotdog/env.py` —— 核心 Gymnasium 环境

## 1. 一句话定位

属于**环境层**：把 `robotdog/go2_model.py` 编译出的「Go2 + 参数化楼梯」MJCF 包成一个
Gymnasium 环境（`Go2StairsEnv`），提供 50 Hz 控制 / 500 Hz 物理的观测、奖励、终止与成功判据，
并把**残差步态先验**与**离屏渲染**挂在这一个接口上。

它是**盲走（blind locomotion）**环境：`robotdog/env.py:3` 的模块 docstring 逐字写着
「There is no touch sensor, no ray casting and no contact force reading anywhere in this module.」
环境只能通过 `mjData` 里的关节状态、机身位姿与**逐 geom 的接触布尔量**感知世界，
唯一的几何先验是 `ObsConfig.step_height`（直接告诉策略这一集的台阶高度）。

## 2. 职责边界

**做：**

* 观测拼装：固定 42 维单帧（`ObsConfig.base_frame_size`）+ `history_len` 帧历史栈，
  四个可选块由布尔开关按声明顺序追加（`_obs_frame` / `_observation`）。
* 奖励计算：`RewardConfig` 的 14 个逐项塑形项 + 终止/成功奖励（`_compute_reward`）。
* 终止/成功判据：摔倒（高度 / 倾角）、**躯干连续接触**致命性、超时截断、
  终点粘性标记（`_reached_top`）。
* 控制回路：`action_mode="residual"` 的关节目标合成、PD 律、
  `decimation` 次物理子步推进（`mujoco.mj_step`）。
* 步态先验注入：`gait_prior` / `gait_gain` 与 `_gait_t` 时钟（`_gait_action`）。
* reset 采样（出生位置、朝向、关节噪声、初速度）。
* 离屏渲染（`render`）与资源释放（`close`）。

**不做：**

* 不做 MJCF 字符串拼接、不编译模型、不解析几何 id —— 全部委托 `go2_model.compile_scene`
  （`extract_meta` 返回的 `Go2Meta` 是本模块唯一的数据来源）。
* 不定义楼梯几何、不算高度剖面 —— `StairSpec` / `build_terrain_xml` 属于 `robotdog/tools/terrain.py`；
  本模块只用 `step_height` / `crest_height` / `up_end_x` / `center_y` / `start_x` / `height_at()`。
* 不算步态轨迹形状 —— `GaitParams` 属于 `robotdog/gait.py`；本模块只调 `gait.radians(t)`。
* **不读接触力、不投射射线、没有触觉传感器** —— 已经没有 `robotdog/tools/touch.py` 这个模块了。
  本模块能拿到的接触信息只有「某个 geom 这一帧有没有和地形接触」这一个布尔量
  （`_foot_contact_mask` / `_contact_flags`）。
* **不做域随机化**：摩擦 / 质量 / 延迟 / IMU 偏置的 `randomize_*` 字段与
  `set_domain_randomization()` 已从公开 API 删除（历史动机见 `experience.md` §2）。
* 不训练、不建向量环境、不做回调、不做课程推进、不写文件、不打印日志（除了 `render` 的异常）。
* 不改任何 vendored 模型文件；`models/` 只读。

## 3. 关键设计决策与理由

### 3.1 残差 RL 的两个尺度必须分开

`target = home_angles + gait_gain * gait.radians(phase) + action_scale * action`
（`_target_from_action`，`robotdog/env.py:707-728`）。

`EnvConfig.action_scale`（默认 `0.30`）是**策略权限**，`GaitParams.action_scale`（默认
`default_trot()` = `0.90`）是**先验幅度**。两者刻意不共用：

* **否决方案**：共用一个 `action_scale`。先验幅度与策略探索噪声会被同一个系数缩放，
  实测在 2 s 内摔倒（`robotdog/gait.py:133-142` 的 `radians()` docstring；`experience.md` §2）。
* **否决方案**：直接让策略学绝对关节角。本沙箱禁用 Windows 命名管道，只能 `DummyVecEnv`
  （8 环境单进程），PPO 无从发现步态；两次标定 120k / 150k 步平地训练均 0% 成功、
  位移约 0.02 m，学到的策略比零动作站桩更差（466 vs 496 return；`experience.md` §2）。
  `action_mode="absolute"` 仍作为可选分支保留（映射为 `lo + (action+1)*0.5*(hi-lo)`），
  它**故意忽略先验**。
* `gait_prior` 默认 `False`，所以「无先验」这条路径仍然可用；`_gait_action()`
  （`robotdog/env.py:697-705`）在关闭或 `gait_gain == 0.0` 时返回 `zeros(12)`，
  使 `_target_from_action()` 对两种模式是**同一条代码路径**。
* `set_gait_prior()` 可以在线切换：它只改 PD 目标的构造方式，不改观测维数，
  所以不需要重建 `observation_space`（只有改观测块才需要）。

### 3.2 观测默认**不是**全关：`base_lin_vel_body=True`、`step_height=True`

```python
history_len: int = 10
base_lin_vel_body: bool = True      # 3
base_height: bool = False           # 1
gait_phase: bool = False            # 2
step_height: bool = True            # 1
base_frame_size: int = 42
```

⇒ **默认 `frame_size = 42+3+1 = 46`，`size = 460`**。

* `step_height=True` 是刻意的：`robotdog/env.py:411-415` 的注释写着盲走策略「cannot infer
  step height cheaply」，所以把这一集的台阶高度作为一个**标量**直接喂进去，而不是让策略去估。
* `base_lin_vel_body=True` 也是默认开的；机身系速度让「前进」永远是分量 `[0]`，
  而世界系版本是「yaw 一变含义就变」的坏特征 —— 所以 `base_lin_vel_world` 现在**根本不再是
  可选项**，只是内部属性。
* **否决方案**：把 `base_height` / `gait_phase` 也默认打开。观测与奖励耦合时后果严重 ——
  把 `base_height` 的参照面同时接进观测**和奖励**，均回报从 +554 掉到 -674 / 回合
  （`experience.md` §3.3）。所以 `_height_reference()` 只服务观测。
* `history_len` 是**固定 10**：没有 CLI 开关，`scripts/train.py` 也不覆盖它，
  所以 `size` 只能取 420 / 430 / … / 490 这八个值（见 §6 的块组合表）。
* `observation_space` 用 `spaces.Box(-np.inf, np.inf, ...)`：没有任何一项被证明有硬边界，
  写死上下界会在归一化前就引入裁剪。

### 3.3 帧约定：`qvel[3:6]` 直接当陀螺用

`qpos[0:3]` 世界位置、`qpos[3:7]` 四元数 `(w,x,y,z)`、`qvel[0:3]` 世界线速度、
**`qvel[3:6]` 是机身（local）角速度** —— 这正是真实 IMU 陀螺的输出，所以
`base_ang_vel_body`（`robotdog/env.py:339-341`）直接返回它，不做任何旋转变换
（docstring 逐字：「Gyro reading. qvel[3:6] is already in the body frame.」）。

* **否决方案**：把角速度转到世界系再用，或从四元数数值微分。前者随 yaw 改变含义，
  后者引入差分噪声。
* `base_lin_vel_body` 用 `R(q)^T @ qvel[0:3]`，理由同样：这样「前进」永远是 `[0]`。
  世界系版本 `base_lin_vel_world` 仍留着，因为**奖励**（`forward_vel` / `forward_progress`）
  用的就是它，而且它对「绝对方向」类诊断有用。

### 3.4 `forward_progress`：高斯项在 vx=0 不为零

`forward_vel` 是指令跟踪的**高斯**，`exp(-(vx-0.55)²/0.25)`；它在 `vx=0` 处
仍然等于 `exp(-(0.55/0.25)²) ≈ 0.0079`，配合 `alive` 就让「站着不动」成为一个正回报的局部最优。

* 修法：加**线性**伴项 `forward_progress * clip(vx, -0.5, 1.5)`，它在 `vx=0` 处**恰好为 0**，
  梯度永不消失；再配合把 `alive` 压在 `0.15`，站桩不再划算（`experience.md` §2.2）。
* 注意 `forward_vel_sigma` 在源码里**没有被平方**：分母就是 `rc.forward_vel_sigma` 本身
  （`robotdog/env.py:436-438`），所以这个参数不是标准差的单位。
* **否决方案**：只调小 `forward_vel_sigma`。σ→0 让高斯变成尖峰，远离指令处梯度全为 0；
  而且 `vx=0` 的正基座只被削小、不会消失。

### 3.5 `base_height` 奖励**故意**仍用单点参照面

`terms["base_height"]` 用的是 `self.terrain_height_under_base`（= `stair.height_at(base_x)`，
x 的**阶梯函数**），**不是** `_height_reference()`：

* 实测：把观测用的脚底参照面同时接进奖励，均回报从 **+554 崩到 -674 / 回合**
  （`experience.md` §3.3）。噪声来源是在立面附近参照面本身跳变，`(z - ref - 0.27)²` 出现
  人为跳变。
* 因此 `_height_reference()` **必须保持「只影响观测」的单变量开关**语义。
* 这条「故意的不一致」是设计，不是遗漏 —— 想统一之前请先复现 §3.3 的对照实验。
  奖励侧的 `stair_progress` 现在也回到单点口径（没有足底参照面的分支了），
  但它的参照量是**本回合起点**而不是绝对地形高度，见 §3.7。

### 3.6 `foot_lift_step_ratio=1.4`：目标随台阶高度长，但 5 cm 上位对位不变

`lift_target = max(foot_lift_target, foot_lift_step_ratio * stair.step_height)`。

* 理由：`foot_lift_target=0.08` 是**目标下限**而不是目标本身。脚下参照面口径下，脚掌必须
  **越过**台阶高度才能跨过去（`experience.md` §6.1），写死 0.08 让 8 cm 台阶上目标完全饱和
  （`experience.md` §2.3）。
* 5 cm 时 `max(0.08, 1.4×0.05=0.07) = 0.08`，与原来的常数目标**逐位相同**，
  所以已经学会的 easy 段不会回退（源码注释也在说这件事）。
* **否决方案**：直接把 `foot_lift_target` 调大。19 cm 会要求 0.266 m 的抬脚高度，超出腿的
  几何极限（`experience.md` §2.3）。
* 3-D 足速门（`foot_lift_gate_speed=0.25` m/s）也是刻意选 3-D 而不是水平分量：只看水平速度会
  奖励「快速贴地平扫」，同时把高台阶需要的「慢抬」错误地掐断。
* `foot_clearance = foot_pos[:,2] - foot_radius - terrain_height(foot_x)`：`geom_xpos` 是**球心**，
  必须先减半径才是脚面与地形的真实间隙。

### 3.7 `stair_progress`：相对**本回合起点**的净爬升，且不可回退

`terms["stair_progress"] = stair_progress * max(0.0, terrain_height_under_base - _start_terrain_height)`。

* 参照量是 `reset()` 时记录的 `_start_terrain_height`，所以「站在已经到达的台阶上不动」
  永远不再得分 —— 它奖励的是**本回合新增**的高度，不是「当前高度」。
* `max(0.0, ...)` 让它对下楼梯给 0 而不是负分。
* 这一项用的是**躯干**下的地形（单点），历史对照实验见 `experience.md` §2.7。

### 3.8 终止：`terminate_contact_steps=10` 的躯干接触消抖 + 腿接触只罚不死

`_contact_fatal()`（`robotdog/env.py:582-595`）**只**对躯干接触计数：

```python
self._trunk_contact_streak = self._trunk_contact_streak + 1 if trunk_contact else 0
if not trunk_contact:
    return False
return self._trunk_contact_streak >= self._effective_contact_steps()
```

* `_effective_contact_steps() = max(1, int(cfg.terminate_contact_steps))`，默认 `10` ⇒
  躯干必须**连续** 10 个控制步（200 ms）压在地形上才判死，单帧撞击被消抖。
* **腿接触永不致命**：`leg_collision=-0.25` 每步扣分但不终止。
  **否决方案**：`"any"`（任何非足机器人几何碰地即判死）是历史行为，`course12` 在 10 cm 上
  平均只活 **62.6 个控制步**、从未越过 `x = 1.42 m`，而第一级立面在 `x = 1.50 m` ——
  机器人还没碰到台阶就被「腿刮一下」判死了，试错被彻底杀掉（`experience.md` §2.4）。
* streak 计数器在 `_contact_fatal()` 内推进而不是在 `step()` 里，这样「一次 `_terminated()`
  == 一个控制步」对直接驱动 `_terminated` 的回归测试也成立。
* `_contact_flags()`（`robotdog/env.py:542-576`）对空集合显式提前返回 `(False, False)`。

### 3.9 `terminate_height_ref`：默认 `"support"`，不是历史上的 `"base"`

`_fall_reference_height()`（`robotdog/env.py:622-635`）的三个分支：

| 取值 | 参照面 |
| --- | --- |
| `"support_mean"` | `_support_height_mean()`：四足足下地形的**均值** |
| `"support"`（默认） | `_support_height()`：四足足下地形的**最低值**，无脚时回退 `terrain_height_under_base` |
| 其它任意值（含 `"base"`） | `terrain_height_under_base`，即 x 的**阶梯函数** |

* 为什么默认已经改成 `"support"`：单点参照面在立面附近会产生**幽灵摔倒**。
  实测样本：`base_x=1.503` 时 `Δref = +0.1100`，`base_x=1.491` 时 `Δref = -0.1100`
  （`experience.md` §1.4）。`EnvConfig.terminate_height_ref` 的注释里也逐字留着这段
  「`"base"` -> terrain under the base (step function, gives ghost falls)」。
* `_support_height_mean()`：`min` 在最后一只脚落地的那一帧会跳一整个 riser；均值把它摊成
  四次 `h/4` 增量。它**不是**更宽松 —— 跨越过程中均值高于最小值，会更早触发；它去掉的是
  单帧不连续。
* 历史上有第四个取值 `"support_load"`（法向力加权），随触觉一起删除。

### 3.10 渲染：强制打开全部 6 个 geomgroup

`mujoco.mjv_defaultOption` 出厂设置是 `geomgroup = [1, 1, 1, 0, 0, 0]` —— **group 3/4/5 默认隐藏**。
地形在 `TERRAIN_GEOM_GROUP = 1`（在默认可见区内），但机器人自己的碰撞体在 group 3。
`render()` 里 `for g in range(6): self._scene_option.geomgroup[g] = 1` 就是为了让渲染器
**永远不可能**静默丢掉世界：曾经把楼梯盖进隐藏组，结果物理照常爬、GIF 里机器人在爬空气
（`experience.md` §1.2）。

* `mujoco.Renderer` 绑定到**一个具体的 `MjModel` 实例**，所以 `_build_model()` 里必须先
  `close()` 旧的 renderer 再重建，否则 `set_stair()` 之后画面还是上一块地形。
* `render()` 的五套相机 preset 是写死的（见 §6），`camera` 不在 `CAMERAS` 里抛 `ValueError`。

### 3.11 其它被否决 / 被固定下来的选择

* **reset 时把同一帧复制 `history_len` 次**（而不是补零）：补零会让首帧观测里出现
  「关节角=0、上一动作=0」这种分布外样本。
* **`step()` 里 `_gait_phase = self._gait_t` 在推进 `_gait_t` 之前**：观测里报告的相位必须是
  本步**真正使用**的相位，否则 transition 不再是 Markov 的。
* **`terminate_on_success=True`**：到达顶部就**成功终止**，而不是让机器人继续走出楼梯远端摔下去。
  `_reached_top` 在回合内**粘性**，所以更早一步摔过的话仍然算失败。
* **`alive` 是常数项**：`terms["alive"] = rc.alive` 与任何状态无关，作用是把「活满一集」和
  「提前终止」的回报差拉开，见 §3.12。
* **单文件即全部**：环境、奖励、终止、渲染都在 `robotdog/env.py` 一个文件里，
  历史上有 `robotdog/render.py` / `robotdog/evaluate.py`，都已删除。

### 3.12 `termination=-15.0` 与 `success=50.0` 的量级

* `termination=-15.0`：摔倒的即时惩罚在数值上约等于 **1 s 的 `alive`**（`0.15 × 50` 控制步），
  不是「巨额罚款」。它负责让「活得更久」比「马上死」好，真正的信号仍来自 `body_collision`
  等逐步项。
* `success=50.0`：终点奖励必须显著大于一集的 `alive` 累积（`20 s` × 50 = 1000 控制步 ×
  0.15 = 150）的**一部分**才有牵引力，但也不能大到让 value 目标爆掉。
* 终止奖励与成功奖励都记进 `terms["terminal"]`，所以
  `sum(info["reward_terms"].values()) == 返回的 reward` 在**终止步也成立**
  （见 §9 关于 `_compute_reward` 本身返回值不含 `terminal` 的说明）。

## 4. 公开 API

### 4.1 `ObsConfig`（6 个字段）

```python
@dataclass
class ObsConfig:
    history_len: int = 10
    base_lin_vel_body: bool = True      # 3: body-frame velocity
    base_height: bool = False           # 1: height above the terrain under the base
    gait_phase: bool = False            # 2: sin/cos gait clock
    step_height: bool = True            # 1: the staircase height of this episode
    base_frame_size: int = 42

    @property
    def frame_size(self) -> int: ...     # 42 + 3*lin_vel_body + 1*base_height + 2*gait_phase + 1*step_height
    @property
    def size(self) -> int: ...           # frame_size * history_len
```

* 四个开关的贡献维数固定为 `base_lin_vel_body=3`、`base_height=1`、`gait_phase=2`、
  `step_height=1`；**全部 16 种组合分别是 42/43/44/45/46 … 直到 49**，见 §6。
* `base_frame_size` 本身是可覆盖字段，但没有任何脚本改它，默认就是 42。
* 追加顺序 = **字段声明顺序**（`_obs_frame()` 里的 `if` 序列）。

### 4.2 `RewardConfig`（24 个字段）

全部形如 `字段名: 类型 = 默认值`。`[SPEC]` 标记用户明确要求的五项。

```python
@dataclass
class RewardConfig:
    # --- the five requested terms ----------------------------------------- #
    forward_vel: float = 2.5            # [SPEC] Gaussian tracking of command_vx
    forward_vel_sigma: float = 0.25
    orientation: float = -1.2           # [SPEC] roll^2 + pitch^2
    torque: float = -1.5e-4             # [SPEC] sum(tau^2)
    foot_lift: float = 0.9              # [SPEC] swing-clearance shaping
    body_collision: float = -4.0        # [SPEC] torso touching the terrain
    # --- forward motion ---------------------------------------------------- #
    forward_progress: float = 1.5       # linear in vx, so vx=0 earns nothing
    forward_progress_max: float = 1.5
    forward_progress_min: float = -0.5
    # --- swing-foot lift --------------------------------------------------- #
    foot_lift_target: float = 0.08      # floor of the target, not the whole target
    foot_lift_step_ratio: float = 1.4   # target = max(target, ratio * step_height)
    foot_lift_gate_speed: float = 0.25  # m/s 3-D foot speed gate
    # --- gait discovery ---------------------------------------------------- #
    air_time: float = 0.6
    air_time_target: float = 0.30
    air_time_max: float = 0.60
    # --- supporting shaping ------------------------------------------------ #
    base_height: float = -2.0
    joint_vel: float = -2.0e-5
    action_rate: float = -0.02
    joint_limit: float = -0.5
    alive: float = 0.15
    termination: float = -15.0
    success: float = 50.0
    # --- staircase progress ------------------------------------------------ #
    stair_progress: float = 3.0
    leg_collision: float = -0.25
```

各项公式（`lin = base_lin_vel_world`、`rpy = base_rpy`、`tau = data.actuator_force`、
`h = stair.step_height`、`lift_target = max(foot_lift_target, foot_lift_step_ratio * h)`）：

| key | 公式 |
| --- | --- |
| `forward_vel` | `forward_vel * exp(-(lin[0] - command_vx)² / forward_vel_sigma)`（σ 不平方） |
| `forward_progress` | `forward_progress * clip(lin[0], forward_progress_min, forward_progress_max)` |
| `orientation` | `orientation * (rpy[0]² + rpy[1]²)` |
| `torque` | `torque * sum(tau²)` |
| `foot_lift` | `foot_lift * mean(gate * lift)`，`gate = clip(foot_speed / max(foot_lift_gate_speed,1e-6), 0, 1)`，`foot_speed = ‖(feet - _prev_foot_pos)/control_dt‖`（3-D），`lift = clip(clearance / lift_target, 0, 1)` |
| `body_collision` | `body_collision * float(trunk_contact)` |
| `leg_collision` | `leg_collision * float(leg_contact)` |
| `air_time` | `air_time * sum((clip(air - air_time_target, 0, None) - clip(air - air_time_max, 0, None)) * landed)`；`landed = foot_contact & ~_prev_foot_contact`，`air = _foot_air_time`（**本步之前**累积的腾空时间）；`air_time == 0.0` 时恒 `0.0` |
| `base_height` | `base_height * (base_pos[2] - terrain_height_under_base - nominal_pose)²` |
| `joint_vel` | `joint_vel * sum(joint_vel²)` |
| `action_rate` | `action_rate * sum((action - _prev_action)²)` |
| `joint_limit` | `joint_limit * sum(clip(lo - q, 0, None) + clip(q - hi, 0, None))` |
| `alive` | `alive`（常数） |
| `stair_progress` | `stair_progress * max(0.0, terrain_height_under_base - _start_terrain_height)` |

`_compute_reward()` 返回 `(float(sum(terms.values())), terms)`，`terms` 恰好 **14 个键**
（上表全部键名，实测 `len(terms) == 14`），**不含 `terminal`**。
`RewardConfig` 的 24 个字段里，`forward_vel_sigma` / `forward_progress_max` /
`forward_progress_min` / `foot_lift_target` / `foot_lift_step_ratio` / `foot_lift_gate_speed` /
`air_time_target` / `air_time_max` / `termination` / `success` 这 10 个是**参数**，不是键。

### 4.3 `EnvConfig`（26 个字段）

```python
@dataclass
class EnvConfig:
    timestep: float = 0.002
    decimation: int = 10                    # -> 50 Hz control
    episode_length_s: float = 20.0
    action_mode: str = "residual"           # "residual" | "absolute"
    action_scale: float = 0.30              # rad, policy authority (residual only)
    command_vx: float = 0.55                # m/s forward command
    command_vy: float = 0.0
    command_yaw_rate: float = 0.0           # 只写进 cfg，当前环境不读
    gait_prior: bool = False
    gait: GaitParams = field(default_factory=GaitParams.default_trot)
    gait_gain: float = 1.0
    kp: tuple[float, float, float] = (30.0, 30.0, 45.0)
    kd: tuple[float, float, float] = (0.6, 0.6, 0.9)
    terminate_contact_steps: int = 10
    terminate_height_ref: str = "support"
    terminate_min_height: float = 0.13
    terminate_max_tilt: float = 1.0         # rad, roll or pitch
    terminate_on_success: bool = True
    reset_z: float = 0.32
    reset_joint_noise: float = 0.06
    reset_xy_noise: float = 0.12
    reset_yaw_noise: float = 0.10
    flat_success_distance: float = 3.0      # flat stage only
    spawn_back: float = 0.55                # m in front of the first riser
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @property
    def control_dt(self) -> float: ...      # timestep * decimation
```

* `control_dt = timestep * decimation`（默认 `0.002 × 10 = 0.02` s，即 50 Hz 控制 / 500 Hz 物理）。
* `obs` / `reward` 是嵌套 dataclass，就地改字段即可（如 `cfg.obs.gait_phase = True`）。
* `kp` / `kd` 是**每条腿三个**（hip, thigh, calf），在 `__init__` 里
  `np.tile(cfg.kp, 4)` / `np.tile(cfg.kd, 4)` 展开成 12 维。
* `command_vy` 与 `command_yaw_rate` 目前**只被存下来**：奖励里没有横向速度项、没有 yaw rate 项，
  两个字段保留是为了配置兼容。

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
* 副作用：编译模型、建 `MjData`、定义 `action_space`（`Box(-1, 1, (12,), float32)`）与
  `observation_space`（`Box(-inf, inf, (obs.size,), float32)`）、
  建 `_hist`（`deque(maxlen=cfg.obs.history_len)`）。
* 其它实例属性：`num_act = 12`、`act_low` / `act_high`（来自 `meta.joint_ranges[:,0/1]`）、
  `home_angles`（= `meta.home_joint_angles`）、`_kp` / `_kd`、`_max_steps`、
  `nominal_pose = float(meta.home_qpos[2])`（0.27）、`_gait_t` / `_gait_phase`。
* 异常：模型编译失败时 `compile_scene` 抛 `RuntimeError("MJCF compile failed; dumped to ...")`
  或 `FileNotFoundError`（缺 mesh）。

```python
    def _build_model(self) -> None
    def set_stair(self, stair: StairSpec) -> None
    def set_gait_prior(self, enable: bool, gait: GaitParams | None = None) -> None
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
    def foot_clearance(self) -> np.ndarray        # (4,) 脚面到足下地形的间隙
```

内部辅助（下划线开头，但**被文档化以便 grep**，也是测试直接驱动的入口）：

```python
    def _terrain_height(self, x: np.ndarray | float) -> np.ndarray
    def _projected_gravity(self) -> np.ndarray           # R^T @ (0,0,-1)；已定义但当前无人调用
    def _obs_frame(self) -> np.ndarray                   # float64，长度 obs.frame_size
    def _observation(self) -> np.ndarray                 # float32，长度 obs.size
    def _compute_reward(self, action: np.ndarray, trunk_contact: bool,
                        foot_contact: np.ndarray | None = None,
                        leg_contact: bool = False) -> tuple[float, dict[str, float]]
    def _foot_contact_mask(self) -> np.ndarray           # (4,) bool
    def _advance_foot_air_time(self, foot_contact: np.ndarray) -> None
    def _contact_flags(self) -> tuple[bool, bool]        # (trunk_contact, leg_contact)
    def _effective_contact_steps(self) -> int
    def _contact_fatal(self, trunk_contact: bool, leg_contact: bool) -> bool
    def _support_height(self) -> float
    def _support_height_mean(self) -> float
    def _height_reference(self) -> float                 # 仅 obs.base_height 使用
    def _fall_reference_height(self) -> float
    def _terminated(self, trunk_contact: bool, leg_contact: bool) -> tuple[bool, bool]
    def _gait_action(self) -> np.ndarray
    def _target_from_action(self, action: np.ndarray) -> np.ndarray
```

* `_terrain_height(x)`：接受标量或数组；**平地阶段（`n_up == 0 and n_down == 0`）直接返回
  `np.zeros_like`**，不走 `stair.height_at()`。
* `_compute_reward(...)` 返回 `(float, dict[str, float])`，且
  `reward == sum(terms.values())` 恒成立（返回值里**没有** `terminal`，见 §9）。
  `foot_contact=None` 时它内部自己调 `_foot_contact_mask()`；`leg_contact=False` 时
  `terms["leg_collision"] == 0.0`。
* `_advance_foot_air_time(foot_contact)`：一次控制步调用一次（`step` 在奖励之后调），
  `_foot_air_time = where(foot_contact, 0.0, _foot_air_time + control_dt)`，
  并更新 `_prev_foot_contact`。
* `_contact_fatal`：**会写** `_trunk_contact_streak`（`_leg_contact_streak` 只被清零/记录，
  不参与致命判定）。
* `_target_from_action(action)`：`residual` 分支
  `home_angles + _gait_action() + action_scale * action`；
  `absolute` 分支 `lo + (action + 1) * 0.5 * (hi - lo)`（先验在 absolute 下**被故意忽略**）。
  两支都用 `np.clip(target, act_low, act_high)` 收尾。
* `_projected_gravity()` 当前**是死代码**：定义了 `R^T @ (0,0,-1)` 的展开式，但 `env.py` 里
  没有任何调用点，观测里也没有重力块。

`step()` 返回的 `info` 字段（**恰好 18 个键，键名逐字**）：
`reward_terms`、`trunk_contact`、`leg_contact`、`terminate_contact_steps`、
`trunk_contact_streak`、`base_x`、`base_y`、`base_z`、`terrain_height`、`forward_vel`、`roll`、
`pitch`、`crest_height`、`reach_top_x`、`reached_top`、`is_success`、`fallen`、`gait_prior`。
其中：

* `reward_terms` = `_compute_reward` 的 14 键 dict **再补一个 `"terminal"`** ⇒ 15 键；
* `terminate_contact_steps` = `_effective_contact_steps()`（已消抖后的阈值，不是原始字段）；
* `reach_top_x` = `_success_target_x()`；
* `fall` 判据的实际高度 = `base_pos[2] - _fall_reference_height()`，但 `base_z` 报的是**原始**
  `base_pos[2]`，`terrain_height` 报的是**躯干**下的 `terrain_height_under_base`
  （不是终止用的参照面）；
* `is_success = bool(_reached_top)`；`fallen = bool(terminated) and not self._reached_top`。

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
      → （若已有 renderer）renderer.close()
  → act_low/act_high/home_angles ← meta.joint_ranges / meta.home_joint_angles
  → action_space / observation_space
  → _kp = tile(cfg.kp, 4) ; _kd = tile(cfg.kd, 4)
  → _max_steps = int(cfg.episode_length_s / cfg.control_dt)     # 1000
  → 其余状态字段初始化（_hist、_prev_action、_reached_top、streak、_prev_foot_pos …）
```

`set_stair()` → `_build_model()` → `_success_target_x()` → `_reached_top = False`。

### 5.2 `reset()` 管线（严格按源码顺序）

1. `super().reset(seed=seed)`；`seed is not None` 时 `self._rng = np.random.default_rng(seed)`。
2. `mujoco.mj_resetData(model, data)`。
3. 采样 `x = stair.start_x - spawn_back + U(-reset_xy_noise, +reset_xy_noise)`、
   `y = stair.center_y + U(...)`、`yaw = U(-reset_yaw_noise, +reset_yaw_noise)`。
4. `data.qpos[:] = home_qpos(meta, x=x, y=y, z=terrain_height_under_base + reset_z, yaw=yaw)`。
5. `data.qpos[qpos_joint_adr] += U(-reset_joint_noise, +reset_joint_noise, 12)`。
6. `data.qvel[:] = U(-0.05, 0.05, model.nv)`（**整机**，含浮动基座）。
7. `mujoco.mj_forward(model, data)`。
8. `nominal_pose = float(meta.home_qpos[2])`（0.27）。
9. `_start_terrain_height = terrain_height_under_base`。
10. `_prev_action = zeros(12)`、`_last_action = zeros(12)`。
11. `_step_count = 0`、`_reached_top = False`、`_trunk_contact_streak = 0`。
12. `_prev_foot_pos = foot_pos.copy()`。
13. `_foot_air_time = zeros(4)`、`_prev_foot_contact = _foot_contact_mask()`。
14. `_gait_t = 0.0`；`_gait_phase = 0.0`（**没有相位随机化**了）。
15. `_hist.clear()`；`frame = _obs_frame()`；把 `frame.copy()` 追加 `history_len` 次。
16. `return self._observation(), {}`。

### 5.3 `step()` 管线（严格按源码顺序）

1. `action = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)`。
2. `_gait_phase = _gait_t`（本步真正使用的相位）。
3. `target = _target_from_action(action)` → `home + _gait_action() + action_scale*action`
   → `clip(act_low, act_high)`。
4. `_gait_t += control_dt`。
5. `for _ in range(decimation)`：`q = qpos[qpos_joint_adr]`、`qd = qvel[qvel_joint_adr]`、
   `tau = _kp * (target - q) - _kd * qd`、`data.ctrl[:] = clip(tau, ctrl_ranges[:,0], ctrl_ranges[:,1])`、
   `mujoco.mj_step(model, data)`。
6. `_step_count += 1`。
7. `trunk_contact, leg_contact = _contact_flags()`。
8. `foot_contact = _foot_contact_mask()`。
9. `terminated, truncated = _terminated(trunk_contact, leg_contact)`。
10. 成功判定：`if _success_reached(): _reached_top = True; if cfg.terminate_on_success and not
    terminated: terminated = True; truncated = False`。
11. `reward, terms = _compute_reward(action, trunk_contact, foot_contact, leg_contact=leg_contact)`。
12. `_advance_foot_air_time(foot_contact)`。
13. `terminal = reward.success if _reached_top else reward.termination`（仅 `terminated` 时非零）；
    `reward += terminal`；`terms["terminal"] = float(terminal)`。
14. `_prev_action = _last_action.copy()`；`_last_action = action.copy()`。
15. `_prev_foot_pos = foot_pos.copy()`（脚的位移必须恰好跨一个控制步）。
16. `_hist.append(_obs_frame())`。
17. 组 `info`，返回 `(_observation(), float(reward), terminated, truncated, info)`。

### 5.4 `_obs_frame()` 装配顺序

```
_obs_frame()
  → parts = [joint_pos(12), joint_vel(12), base_rpy(3), base_ang_vel_body(3), _last_action(12)]   # 42
  → if obs.base_lin_vel_body: append base_lin_vel_body(3)
  → if obs.base_height:       append [base_pos[2] - _height_reference()](1)
  → if obs.gait_phase:        append [sin(theta), cos(theta)]，theta = 2*pi*gait.freq*_gait_phase
  → if obs.step_height:       append [float(stair.step_height)](1)
  → np.concatenate(parts).astype(float64)
```

`_observation()` = `np.concatenate(self._hist).astype(np.float32)` —— 历史栈**先到先出**，
所以第 0..9 帧的排布是「最旧 → 最新」，最新一帧在**尾部**。
`_height_reference()` 直接返回 `terrain_height_under_base`（`robotdog/env.py:618-620`）。

### 5.5 奖励调用链

```
_compute_reward(action, trunk_contact, foot_contact, leg_contact)
  → base_rpy / base_lin_vel_world / nominal_pose
  → foot_pos, foot_clearance, _prev_foot_pos, stair.step_height
  → （air_time != 0）_foot_contact_mask()（若 foot_contact is None）
  → terrain_height_under_base, _start_terrain_height
  → sum(terms.values()), terms          # 14 键
```

### 5.6 终止调用链

```
_terminated(trunk_contact, leg_contact)
  → base_rpy
  → _fall_reference_height()
      → "support_mean" → _support_height_mean()
      → "support"      → _support_height()      （无脚时回退 terrain_height_under_base）
      → 其它           → terrain_height_under_base
  → _support_height* → stair.height_at(foot_pos[i,0]) 逐脚取 min / mean
  → _contact_fatal(trunk, leg)
      → _effective_contact_steps() = max(1, int(cfg.terminate_contact_steps))
      → 写 _trunk_contact_streak
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
| `episode_length_s` | `20.0` |
| `_max_steps` | `int(20.0 / 0.02) = 1000` |
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

**reset 噪声与出生**：`reset_z=0.32`、`reset_joint_noise=0.06`、`reset_xy_noise=0.12`、
`reset_yaw_noise=0.10`、`spawn_back=0.55`、`flat_success_distance=3.0`、
初速度 `U(-0.05, 0.05)`（施加到全部 `nv`）。

**终止**：`terminate_contact_steps=10`（⇒ `_effective_contact_steps()==10`）、
`terminate_height_ref="support"`、`terminate_min_height=0.13`、`terminate_max_tilt=1.0`、
`terminate_on_success=True`。

**观测块组合**（`history_len=10`，`base_frame_size=42`；`size = frame_size × 10`）

| `base_lin_vel_body` | `base_height` | `gait_phase` | `step_height` | `frame_size` | `size` |
| --- | --- | --- | --- | --- | --- |
| F | F | F | F | 42 | 420 |
| T | F | F | F | 45 | 450 |
| F | T | F | F | 43 | 430 |
| T | T | F | F | 46 | 460 |
| F | F | T | F | 44 | 440 |
| T | F | T | F | 47 | 470 |
| F | T | T | F | 45 | 450 |
| T | T | T | F | 48 | 480 |
| F | F | F | T | 43 | 430 |
| **T** | **F** | **F** | **T** | **46** | **460 ← 默认** |
| F | T | F | T | 44 | 440 |
| T | T | F | T | 47 | 470 |
| F | F | T | T | 45 | 450 |
| T | F | T | T | 48 | 480 |
| F | T | T | T | 46 | 460 |
| T | T | T | T | 49 | 490 |

**奖励**：24 个字段的默认值见 §4.2；其中 `[SPEC]` 五项
`forward_vel=2.5`（σ`0.25`）/ `orientation=-1.2` / `torque=-1.5e-4` / `foot_lift=0.9` /
`body_collision=-4.0`；`alive=0.15`、`termination=-15.0`、`success=50.0`、
`foot_lift_target=0.08`、`foot_lift_step_ratio=1.4`、`foot_lift_gate_speed=0.25`、
`leg_collision=-0.25`、`stair_progress=3.0`、
`air_time=0.6` / `air_time_target=0.30` / `air_time_max=0.60`、
`base_height=-2.0` / `joint_vel=-2.0e-5` / `action_rate=-0.02` / `joint_limit=-0.5`。

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
   ├── robotdog.gait            : GaitParams（cfg.gait 的字段类型与默认值来源）
   ├── robotdog.go2_model       : SceneOptions, compile_scene, default_floor_geom, home_qpos,
   │                              quat_to_mat, quat_to_rpy
   └── robotdog.tools.terrain   : StairSpec, build_terrain_xml, TERRAIN_GEOM_GROUP
   ↓ 被依赖（下游）
   robotdog.policy / robotdog.training / robotdog.curriculum / robotdog.__init__
   scripts/train.py、scripts/play.py
```

方向为什么是这样（`docs/README.md` 的分层规则）：

* 依赖必须**单向** `训练层 → 环境层 → 工具层`。`tools/` 里**不 import `robotdog.env`**，
  所以 `terrain` 可以是纯几何，不知道奖励、课程、PPO 的存在。
* `env` 只从 `gait` 拿**轨迹**、从 `go2_model` 拿**模型与 id**、从 `terrain` 拿**几何剖面**。
  它自己不做这三件事中的任何一件。
* **触觉那条边已经不存在**：历史上 `robotdog/tools/touch.py` 是 `env` 的上游，
  现在整条链路（射线、接触力、`TouchConfig`）被删除，`env` 的接触信息只剩逐 geom 布尔量。
* **能驱动环境的渲染模块也没有了**：`robotdog/render.py` 曾因为要驱动环境而留在顶层
  （进 `tools/` 会形成 `tools ↔ env` 环），该模块已删除，渲染能力现在只在 `env.render()` 里。
* 反过来的耦合被刻意避免：`env` 不 import `training` / `curriculum`。
  课程通过 `set_stair()` 从外部驱动环境，而不是环境去问课程「我现在在第几段」。

## 8. 实测与经验教训

本节**只列引用**，结论请直接读原文，不要在这里复述：

* 奖励设计侧：`experience.md` §2.1（惩罚项无上限毁掉 value function）、§2.2（高斯前进奖励在
  vx=0 非零 ⇒ 学会站桩）、§2.3（抬升目标写死 0.08 m ⇒ 在 8 cm 完全饱和）、§2.4（腿碰地立刻判死
  杀掉试错）、§2.5（蹭立面惩罚与任务方向打架）、§2.6（落地力阈值是常数）、§2.7（`stair_progress`
  奖励「躯干冲线」）、§2.8（走直线权重太小）。
* 观测设计侧：`experience.md` §3.1（策略被打分却看不到被打分的量）、§3.2（漂移量按出生位姿量）、
  §3.3（`base_height` 参照面只能动观测）、§3.4（残差策略不知步态相位）。
* 物理 / 几何侧：`experience.md` §1.4（摔倒判据用阶跃函数 ⇒ 幽灵摔倒）、§1.2（地形 geomgroup
  被 MuJoCo 默认隐藏）、§1.1（踏步立面与顶面共面接缝把脚夹死）、§1.3（零高度台阶无法编译）。
* 能力边界：`experience.md` §6（11 cm 的墙；摆动相峰值抬升饱和与「缺的不是峰值高度而是
  摆动相里的向前落脚」）。
* 训练与课程侧：`experience.md` §4.1（课程毕业判据把梯度饿死）、§4.2（`VecNormalize` 奖励归一化
  只是温和稳定器）；`train.md` §1（训练组织与默认课程表）、§2（训练配方）、§3（参考 checkpoint
  与消融）。
* 工具与测量侧：`experience.md` §5.5（并行 run 数上限≈3）、§5.7（10 回合评估给出 ±15 pt 错觉）、
  §5.8（沙箱禁命名管道 ⇒ 只能 `DummyVecEnv`；非 ASCII 路径）。

## 9. 已知限制与坑

1. **11 cm 是墙。** 详见 `experience.md` §6；正确下一步是显式落脚相 IK 替换 `gait.radians()`，
   而不是继续调奖励权重。
2. **`_compute_reward()` 的返回值里没有 `terminal`。** 它返回 `sum(terms.values())`（14 键），
   而 `terminal` 是 `step()` 在 `terms` 上补写的。因此「`sum(reward_terms) == reward`」这条恒等式
   对 `info["reward_terms"]` 成立（15 键），对 `_compute_reward()` 的 `terms` 也成立
   （14 键，无 `terminal`），但**不是同一个 dict**。要按 key 逐项对账时注意这一点。
3. **`_contact_fatal()` 会写状态**（`_trunk_contact_streak`）。它是 `_terminated()` 的一部分，
   「一次 `_terminated()` == 一个控制步」是它的契约；不要在同一个控制步里调两次。
4. **`terminate_height_ref` 的默认值已经换成 `"support"`**：任何按老文档写
   `cfg.terminate_height_ref = "base"` 的脚本仍然能跑，但会重新打开幽灵摔倒
   （`base_x=1.503 → Δref=+0.1100`、`base_x=1.491 → Δref=-0.1100`，`experience.md` §1.4）。
   非法取值的处理是**静默回退到 `terrain_height_under_base`**，不会报错。
5. **`command_vy` / `command_yaw_rate` 是死字段**：奖励里没有横向速度项、没有 yaw rate 项，
   改它们对训练**没有任何影响**。`grep "command_vy"` 只会命中 dataclass 定义处。
6. **观测维度不可能取到 460/10 以外的「历史值」**：`history_len` 固定 10、`base_frame_size` 固定 42，
   所以合法 `obs.size` 只有 420/430/…/490 这八个（§6 表）。`robotdog/policy.py` 的 docstring 拿
   「474 维」举例，那个数**不是** `10 × frame_size` 能凑出来的（474/10 = 47.4）；
   `_obs_config_for_dim(total, history_len=10)` 因此也永远无法反推出它。
7. **`step()` 的 `info` 只有 18 个键**：老代码里的 `body_contact` / `leg_contact_streak` /
   `lateral_offset` / `spawn_*` / `heading_error` / `probe_*` / `riser_graze` /
   `total_normal_force` / `max_slip` **全部已删除**，`grep` 这些键只会命中零处。
8. **`obs.base_height` 的参照面不能顺手接进奖励**：同时接观测与奖励使均回报从 +554 掉到 -674
   （`experience.md` §3.3）。`_height_reference()` 目前就是 `terrain_height_under_base`，
   奖励里的 `base_height` 项也用它 —— 也就是说这个「只影响观测」的语义现在是靠**两边同源**实现的，
   改动任一侧前请先读 §3.5。
9. **`_support_height()` 在楼梯几何之外返回 0**：`StairSpec.height_at` 在
   `x < start_x` 或超过 `up_end_x + top_landing`（`n_down == 0` 时）返回 `0.0`。
   机器人走过顶部平台远端之后 `support` 会掉回 0，`terminate_height_ref="support"` 的
   高度判据在那里会失效（这也是 `terminate_on_success=True` 默认开启的原因之一）。
10. **`_foot_contact_mask()` / `_contact_flags()` 是 O(ncon) 的 Python 循环**，每控制步各跑一次
    （`foot_contact` 在 `step` 里算一次后作为参数传进 `_compute_reward`，所以正常路径只跑一次
    `_foot_contact_mask`）。接触数大的场景是热点。
11. **`render()` 的报错文本是中文**：`ValueError(f"未知相机 {camera!r}, 可选 {self.CAMERAS}")`；
    这是本模块唯一会打印/抛出的用户可见文案。域随机化的
    `AttributeError(f"unknown domain-randomization flag(s) ...")` 已随
    `set_domain_randomization()` 一起删除。
12. **编译失败会写盘**：`compile_scene` 在 `write_to is None` 时把失败的 XML 落到
    `.generated/failed_scene.xml` 并抛 `RuntimeError("MJCF compile failed; dumped to ...")`。
13. **`subprocess` / `SubprocVecEnv` 在本沙箱不可用**（禁用命名管道），所以并行度只能靠单进程
    多环境；这也决定了 8 环境的天花板（`experience.md` §5.8、`robotdog/training.py` 模块说明）。
14. **`_projected_gravity()` 是死代码**：定义了却不被调用，观测里没有重力方向块。
    想验证「躯干倾角」是否能做特征时，注意 `base_rpy` 已经在基础 42 维里。
15. **`set_stair()` 之后不 `reset()` 会读到跨地形的脏状态**：新的 `MjModel`/`MjData` 已换掉，
    但 `_hist`、`_prev_foot_pos`、`_start_terrain_height` 都还指向旧地形。
16. **`_fall_reference_height()` 的取值判定是 `.strip().lower()` 之后精确比较**：
    `"Support"` / `"support "` 会被规范化，但 `"support_mean"` 必须完全一致，
    写错一个字符就静默回退到 `terrain_height_under_base`。

## 10. 示例用法

```python
"""最小可运行：建环境 → 用默认步态先验走一集 → 打印终止原因。"""
import numpy as np

from robotdog.env import EnvConfig, Go2StairsEnv
from robotdog.gait import GaitParams
from robotdog.tools.terrain import StairSpec

cfg = EnvConfig()                    # 默认 obs.frame_size=46（base_lin_vel_body + step_height）
cfg.obs.gait_phase = True            # +2 => 48 维/帧，480 维观测
cfg.terminate_height_ref = "support"             # 默认已是 support，显式写出便于对账
cfg.gait_prior = True
cfg.gait = GaitParams.default_trot()
cfg.action_scale = 0.30                         # 策略权限（与先验幅度无关）

stair = StairSpec(step_height=0.05, tread=0.50, n_up=8, n_down=0)
env = Go2StairsEnv(stair=stair, cfg=cfg, seed=0)

obs, info = env.reset(seed=0)
assert obs.shape == (env.cfg.obs.size,) == (480,)

total = 0.0
terminated = truncated = False
for _ in range(env._max_steps):
    action = np.zeros(env.action_space.shape, dtype=np.float32)   # 纯先验，零残差
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    if terminated or truncated:
        break

print(f"return={total:.2f} steps={env._step_count} success={info['is_success']} "
      f"fallen={info['fallen']} x={info['base_x']:.3f} crest={info['crest_height']:.3f} "
      f"trunk_streak={info['trunk_contact_streak']}/{info['terminate_contact_steps']}")
print({k: round(v, 4) for k, v in info["reward_terms"].items()})   # 15 键（含 terminal）

frame = env.render("track", width=640, height=480)   # (480, 640, 3) uint8
env.close()
```
