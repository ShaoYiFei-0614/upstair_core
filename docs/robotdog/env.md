# `robotdog/env.py` —— 核心 Gymnasium 环境

## 1. 一句话定位

属于**环境层**：把 `robotdog/go2_model.py` 编译出的「Go2 + 参数化楼梯」MJCF 包成一个
Gymnasium 环境（`Go2StairsEnv`），提供 50 Hz 控制 / 500 Hz 物理的观测、奖励、终止与成功判据，
并把**残差步态先验**与**离屏渲染**挂在这一个接口上。

它是**盲走（blind locomotion）**环境：`robotdog/env.py:3-6` 的模块 docstring 逐字写着
原文是「There is no contact sensor, no ray casting, and no contact force reading anywhere in
this module.」，现译文为「本模块中没有任何触觉传感器、没有射线投射，也没有任何接触力读取」。
环境只能通过 `mjData` 里的关节状态、机身位姿与**逐 geom 的接触布尔量**感知世界，
唯一的几何先验是 `ObsConfig.step_height`（直接告诉策略这一集的台阶高度）。

## 2. 职责边界

**做：**

* 观测拼装：固定 42 维单帧（`ObsConfig.base_frame_size`）+ `history_len` 帧历史栈，
  四个可选块由布尔开关按声明顺序追加（`_obs_frame` / `_observation`）。
* 奖励计算：**公式不在这里**，整块搬到了 `robotdog/reward.py`；本模块只留一个三行转发方法 `_compute_reward`（`robotdog/env.py:531`）。
* 终止/成功判据：摔倒（高度 / 倾角）、**躯干连续接触**致命性、超时截断、
  终点粘性标记（`_reached_top`）。
* 控制回路：残差关节目标合成、PD 律、
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
（`_target_from_action`，`robotdog/env.py:827`）。

`EnvConfig.action_scale`（默认 `0.30`）是**策略权限**，`GaitParams.action_scale`（默认
`default_trot()` = `0.90`）是**先验幅度**。两者刻意不共用：

* **否决方案**：共用一个 `action_scale`。先验幅度与策略探索噪声会被同一个系数缩放，
  实测在 2 s 内摔倒（`robotdog/gait.py:94-103` 的 `radians()` docstring；`experience.md` §2）。
* **否决方案**：直接让策略学绝对关节角。本沙箱禁用 Windows 命名管道，只能 `DummyVecEnv`
  （8 环境单进程），PPO 无从发现步态；两次标定 120k / 150k 步平地训练均 0% 成功、
  位移约 0.02 m，学到的策略比零动作站桩更差（466 vs 496 return；`experience.md` §2）。
  **否决方案**：保留一个 `action_mode="absolute"` 分支（映射为 `lo + (action+1)*0.5*(hi-lo)`，
  故意忽略先验）。它从未被任何一次训练或被参数搜索用过，已随本轮清理连同
  `EnvConfig.action_mode` 字段一起删除；现在只有残差一条路径。
* `gait_prior` 默认 `False`，所以「无先验」这条路径仍然可用；`_gait_action()`
  （`robotdog/env.py:817-825`）在关闭或 `gait_gain == 0.0` 时返回 `zeros(12)`，
  使 `_target_from_action()` 对「无先验」和「`GaitParams.static()`」是**同一条代码路径**。
* 先验**只能在构造时定**：`set_gait_prior()`（在线切换先验的旧 API，docstring 曾自称
  "used by the CLI"，但那个 CLI 开关早已删除）已在死代码清理中删除。要换先验就换
  `cfg.gait` / `cfg.gait_gain` 后重建环境 —— 反正它本来也只改 PD 目标的构造方式，
  不改观测维数。

### 3.2 观测：`base_lin_vel_body=True`、`step_height=True`，`base_height` **禁止打开**

```python
history_len: int = 10
base_lin_vel_body: bool = True      # 3
base_height: bool = False           # 1，禁止打开：__post_init__ 抛 ValueError
gait_phase: bool = False            # 2
step_height: bool = True            # 1
command: bool = True                # 2：横向速度指令 + 偏航角速度指令
base_frame_size: int = 42
```

⇒ **`frame_size = 42+3+1+2 = 48`**。`ObsConfig` 的 dataclass 默认是 `history_len = 10`
（`size = 480`），但**仓根 `config.yaml` 把它设成了 `1`**，所以本仓实际训练出来的是
单帧 **48 维** —— 对照表见 §6。

* `step_height=True` 是刻意的：`robotdog/env.py:58` 的注释写着盲走策略「无法低成本
  推断台阶高度」，所以把这一集的台阶高度作为一个**标量**直接喂进去，而不是让策略去估。
  它是整个观测里**唯一一个非传感器项**，但也只是标量任务参数（真机可以随任务下发），
  且不含「下一级立面在哪」。
* `base_lin_vel_body=True` 也是默认开的；机身系速度让「前进」永远是分量 `[0]`，
  而世界系版本是「yaw 一变含义就变」的坏特征 —— 所以 `base_lin_vel_world` 现在**根本不再是
  可选项**，只是内部属性。
* ⭐ **`base_height` 现在是硬错误，不再是「默认关掉的开关」**：`ObsConfig.__post_init__`
  在它为 `True` 时直接抛 `ValueError`。原因是这一项的实现是
  `base_pos[2] - _height_reference()`，而 `_height_reference()` 就是
  `terrain_height_under_base` —— **地形几何真值**。盲走策略拿到它就不再是盲走。
  连带删除的还有 `_obs_frame()` 里那条分支与 `_height_reference()` 本身。
  完整的信息来源清单见 `agent.md` §8.6。
* **否决方案**：把 `base_height` / `gait_phase` 也默认打开。观测与奖励耦合时后果严重 ——
  把 `base_height` 的参照面同时接进观测**和奖励**，均回报从 +554 掉到 -674 / 回合
  （`experience.md` §3.3）。这条历史教训与上面那条新禁令是同一件事的两面：
  当时靠「只服务观测」的约定，现在直接把这个开关封死。
* `command=True` 是**恒真**的，不是可选优化：PPO 的网络与 `RolloutBuffer` 维数不可变，
  课程只能改指令的**范围**（`CommandConfig.vy_max` / `yaw_max`）而不能开关这 2 维。
  内容是本回合当前指令的 `[vy, wz]`（`vx` 是定值 0.55，不进观测）。见 §4.5 的指令分节。
* `history_len` 默认 **10**，但**不是固定值**：`config.yaml` 的 `env.obs.history_len`
  直接写进 `ObsConfig`（`robotdog/config.py` 的 `config_to_env`），所以 `size` 可以取
  `frame_size × 任意帧数`。默认 10 时是 430 / 440 / … / 500 这几个值（见 §6 的块组合表）；
  实测改成 **1**（48 维）在课程内每一级都不差、3 cm/6 cm 还更好（`train.md` §3.6）。
* `observation_space` 用 `spaces.Box(-np.inf, np.inf, ...)`：没有任何一项被证明有硬边界，
  写死上下界会在归一化前就引入裁剪。

### 3.3 帧约定：`qvel[3:6]` 直接当陀螺用

`qpos[0:3]` 世界位置、`qpos[3:7]` 四元数 `(w,x,y,z)`、`qvel[0:3]` 世界线速度、
**`qvel[3:6]` 是机身（local）角速度** —— 这正是真实 IMU 陀螺的输出，所以
`base_ang_vel_body`（`robotdog/env.py:434-417`）直接返回它，不做任何旋转变换
（docstring：「陀螺读数。qvel[3:6] 已经在机体系下。」）。

* **否决方案**：把角速度转到世界系再用，或从四元数数值微分。前者随 yaw 改变含义，
  后者引入差分噪声。
* `base_lin_vel_body` 用 `R(q)^T @ qvel[0:3]`，理由同样：这样「前进」永远是 `[0]`。
  世界系版本 `base_lin_vel_world` 仍留着，因为**奖励**（`forward_vel`）与
  `info["forward_vel"]` 用的就是它，而且它对「绝对方向」类诊断有用。

### 3.4 `forward_vel_sigma` 是**标准差**，分母里必须平方

`forward_vel` 是指令跟踪的**高斯**：`forward_vel * exp(-(vx-0.55)² / σ²)`，`σ = 0.25 m/s`。

* 历史写法漏了平方，分母直接就是 `rc.forward_vel_sigma`（旧 `robotdog/reward.py:113`），
  等于把 0.55 m/s 的偏差只按 `sqrt(0.25) = 0.5 m/s` 折算：`exp(-(0.55)²/0.25) = exp(-1.21)
  ≈ 0.30`，于是**站着不动白拿 `2.5 × 0.30 ≈ +0.72/步`**（实测 +0.86/步），这是「原地拖延」
  局部最优的第一根支柱（`experience.md` §12）。
* 现在分母是 `rc.forward_vel_sigma ** 2`（`robotdog/reward.py:117`）：`vx=0` 时
  `exp(-(0.55/0.25)²) = exp(-4.84) ≈ 0.0079`，该项只有 **+0.02/步**，站桩不再有利可图。
* 原来的**线性**伴项 `forward_progress * clip(vx, -0.5, 1.5)` 已随之**删除**（连同
  `forward_progress` / `forward_progress_max` / `forward_progress_min` 三个字段）：
  σ 修正之后高斯项本身在 `vx=0` 处已经足够小，不需要再加一个线性项来压它，而线性项
  在 0.55 m/s 附近会给出 +1.1/步的「超线性」加成，反而扭曲速度跟踪。
* **否决方案**：只调小 `forward_vel_sigma`。σ→0 让高斯变成尖峰，远离指令处梯度全为 0。

### 3.5 `base_height` 奖励项用的是**单点**参照面，且与观测完全解耦

`terms["base_height"]` 用的是 `self.terrain_height_under_base`（= `stair.height_at(base_x)`，
x 的**阶梯函数**），**不是**足底参照面：

* 实测：把观测用的脚底参照面同时接进奖励，均回报从 **+554 崩到 -674 / 回合**
  （`experience.md` §3.3）。噪声来源是在立面附近参照面本身跳变，`(z - ref - 0.27)²` 出现
  人为跳变。
* 这条历史教训现在的落实方式是**更彻底的硬隔离**：`obs.base_height` 被
  `ObsConfig.__post_init__` 直接禁止（打开就抛 `ValueError`），`_height_reference()`
  与 `_obs_frame()` 里那条分支都已删除。所以「奖励用的参照面」与「观测用的参照面」
  再也不可能被不小心接成同一路 —— 观测侧根本没有参照面了。
* 奖励侧的 `stair_progress` 用的是单点口径，且参照量是**本回合曾经达到的最高地形高**，
  见 §3.7。

### 3.6 `foot_lift_step_ratio=1.4`：目标随台阶高度长，但 5 cm 上位对位不变

`lift_target = max(foot_lift_target, foot_lift_step_ratio * stair.step_height)`。

* 理由：`foot_lift_target=0.08` 是**目标下限**而不是目标本身。脚下参照面口径下，脚掌必须
  **越过**台阶高度才能跨过去，写死 0.08 让 8 cm 台阶上目标完全饱和（`experience.md` §2.3）。
* 5 cm 时 `max(0.08, 1.4×0.05=0.07) = 0.08`，与原来的常数目标**逐位相同**，
  所以已经学会的 easy 段不会回退（源码注释也在说这件事）。
* **否决方案**：直接把 `foot_lift_target` 调大。19 cm 会要求 0.266 m 的抬脚高度，超出腿的
  几何极限（`experience.md` §2.3）。
* 3-D 足速门（`foot_lift_gate_speed=0.25` m/s）也是刻意选 3-D 而不是水平分量：只看水平速度会
  奖励「快速贴地平扫」，同时把高台阶需要的「慢抬」错误地掐断。
* `foot_clearance = foot_pos[:,2] - foot_radius - terrain_height(foot_x)`：`geom_xpos` 是**球心**，
  必须先减半径才是脚面与地形的真实间隙。

### 3.7 `stair_progress`：**一次性**结算「刷新最高台阶」，站桩拿不到

```python
climbed = max(0.0, env.terrain_height_under_base - env._peak_terrain_height)
if climbed > 0.0:
    env._peak_terrain_height = env.terrain_height_under_base
terms["stair_progress"] = rc.stair_progress * float(climbed)
```

* 历史写法是 `stair_progress * max(0.0, 下方地形高 - _start_terrain_height)`，它**每步都付**：
  只要机身正下方是高台阶就一直拿钱 —— 实测「爬到某一级然后站着不动」每一步白拿
  **+0.083/步**（15 cm 台阶），这是「原地拖延」局部最优的第二根支柱（`experience.md` §12）。
* 改成按 `_peak_terrain_height`（本回合历史最高）结算后，**累计总量不变**（仍是
  `stair_progress × 爬升高度`，与走多少步无关），但站着不动再也拿不到新的钱。
* 同一级往返（下去再上来）不会再付一次，因为它只认「刷新历史最高」。
* `max(0.0, ...)` 让它对下楼梯给 0 而不是负分。
* 这一项用的是**躯干**下的地形（单点），历史对照实验见 `experience.md` §2.7。

### 3.8 终止：只有「高度 + 倾角」两条判据；接触地形**永不**终止回合

```python
fell = (height < terminate_min_height
        or abs(rpy[0]) > terminate_max_tilt
        or abs(rpy[1]) > terminate_max_tilt)
contact_fatal = self._contact_fatal(trunk_contact, leg_contact)   # 恒为 False
return bool(fell or contact_fatal), truncated
```

* 历史上的 `terminate_contact_steps=10`（躯干连续接触 10 个控制步才算摔倒）与
  `_effective_contact_steps()` **都已删除**，`_contact_fatal()` 现在只维护
  `_leg_contact_streak` 供诊断用，恒返回 `False`。删除依据（`experience.md` §12）：
  - 躯干碰撞体最低点在 **0.118 m**，而高度阈值比它还高 ⇒ 躯干一碰地，高度判据**早就**触发了，
    这条规则在直立姿态下是冗余的；
  - 48 个回合的驱动实验里，躯干连续接触**最长只有 8 步**，从未到过 10 步 ⇒ 它从未生效；
  - 反之，躯干在 `h ≈ 0.28 m`、倾角 0.067 rad 的**完全安全**姿态下贴立面蹭可以连续到 8 步，
    离阈值只差 2 步 ⇒ 高台阶上随时可能误杀，正是「机身上台架天然要蹭上去」的情形。
* 代价改由 `reward.body_collision` 承担（**逐步、有界**，-1.2/步）。原值 -4.0 在 GIF
  复盘里暴露了副作用：6/12/15 cm 上机器人「贴着立面蹭」被罚得比站桩还狠，于是它学会
  在两下尝试后就放弃（`experience.md` §13）。
* **腿接触永不致命**：`leg_collision=-0.25` 每步扣分但不终止。
  **否决方案**：`"any"`（任何非足机器人几何碰地即判死）是历史行为，`course12` 在 10 cm 上
  平均只活 **62.6 个控制步**、从未越过 `x = 1.42 m`，而第一级立面在 `x = 1.50 m` ——
  机器人还没碰到台阶就被「腿刮一下」判死了，试错被彻底杀掉（`experience.md` §2.4）。
* `terminate_min_height` 本轮从 **0.13 提到 0.15**（躯干最低点 0.118 ⇒ 留 3.2 cm 余量），
  更早抓到「趴下去」，同时仍然允许躯干贴立面蹭。
* `_contact_flags()`（`robotdog/env.py:566-602`）对空集合显式提前返回 `(False, False)`。

### 3.9 `terminate_height_ref`：默认 `"support"`，不是历史上的 `"base"`

`_fall_reference_height()`（`robotdog/env.py:647-663`）的三个分支：

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
    base_lin_vel_body: bool = True      # 3：机身系线速度
    base_height: bool = False           # 1：禁止打开（__post_init__ 抛 ValueError）
    gait_phase: bool = False            # 2：步态时钟 sin/cos
    step_height: bool = True            # 1：本回合的楼梯高度
    base_frame_size: int = 42

    def __post_init__(self) -> None: ...  # base_height=True 时抛 ValueError
    @property
    def frame_size(self) -> int: ...     # 42 + 3*lin_vel_body + 2*gait_phase + 1*step_height
    @property
    def size(self) -> int: ...           # frame_size * history_len
```

* 四个可组合开关的贡献维数固定为 `base_lin_vel_body=3`、`gait_phase=2`、
  `step_height=1`、`command=2`；因此 `frame_size` 的可能取值是
  **42/44/45/47/48/49/50**（`command` 恒真 ⇒ 实际只会出现 47~50，`config.yaml` 用 48），
  `history_len=10` 时 `size` 是 420/440/450/470/480/490/500，见 §6。
* **`base_height` 是第四个字段但不可组合**：它为 `True` 时 `__post_init__` 直接抛
  `ValueError`（含地形真值，破坏盲走），`_obs_frame()` 里对应的分支已删除。见 §3.2 与
  `agent.md` §8.6。
* `base_frame_size` 本身是可覆盖字段，但没有任何脚本改它，默认就是 42。
* 追加顺序 = **字段声明顺序**（`_obs_frame()` 里的 `if` 序列）。

### 4.2 `RewardConfig`（30 个字段）—— 定义在 `robotdog/reward.py`

**整块奖励数学已经搬到 `robotdog/reward.py`**，本模块只剩一个三行转发方法
`_compute_reward`（`robotdog/env.py:531`）。完整的字段表、逐项公式、数据流与踩坑
见 [`reward.md`](reward.md)；这里只记与本模块的接缝。

`EnvConfig.reward` 的默认工厂在 `robotdog/env.py:200`，而 `RewardConfig` 本身由
`robotdog/env.py:37` 的导入重导出，所以 `from robotdog.env import RewardConfig`
仍然可用（`robotdog/policy.py:17` 就是这么用的）。

`_compute_reward(action, trunk_contact, foot_contact=None, leg_contact=False,
terminated=False)` 返回 `(逐步奖励 + 终止奖励, terms)`，其中 `terms` 恰好 **15 个键**
= 14 个逐步成形项 + `"terminal"`。转发方法在 `robotdog/env.py:531`：

```python
base, terms, terminal = compute_reward(self, action, trunk_contact, foot_contact,
                                       leg_contact=leg_contact, terminated=terminated)
terms["terminal"] = terminal
return base + terminal, terms
```

因此 `step()` 返回的 `reward == sum(info["reward_terms"].values())` 恒成立。

### 4.3 `EnvConfig`（22 个字段）

```python
@dataclass
class EnvConfig:
    timestep: float = 0.002
    decimation: int = 10                    # -> 50 Hz control
    episode_length_s: float = 20.0
    action_scale: float = 0.30              # rad, policy authority (residual)
    command: CommandConfig = field(default_factory=CommandConfig)
    gait_prior: bool = False
    gait: GaitParams = field(default_factory=GaitParams.default_trot)
    gait_gain: float = 1.0
    kp: tuple[float, float, float] = (30.0, 30.0, 45.0)
    kd: tuple[float, float, float] = (0.6, 0.6, 0.9)
    terminate_height_ref: str = "support"
    terminate_min_height: float = 0.15
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

* `control_dt = timestep * decimation`（默认 `0.002 x 10 = 0.02` s，即 50 Hz 控制 / 500 Hz 物理）。
* `obs` / `reward` 是嵌套 dataclass，就地改字段即可（如 `cfg.obs.gait_phase = True`）。
* `kp` / `kd` 是**每条腿三个**（hip, thigh, calf），在 `__init__` 里
  `np.tile(cfg.kp, 4)` / `np.tile(cfg.kd, 4)` 展开成 12 维。选型依据（实测关节惯量、
  自然频率与阻尼比）见 `README.md` §3.4。
* **只有一个动作映射**：`target = home + gait_gain * gait.radians(t) + action_scale *
  action`（`robotdog/env.py:827`）。历史上有一个 `action_mode: "absolute"` 分支
  （`target = lo + (a+1)/2*(hi-lo)`），它从未被训练或被搜索用过，已随本次清理删除。
* `command_vy` / `command_yaw_rate` **已删除**（它们是死字段，奖励里从来没有读者）。
  现在**指令是一个嵌套 dataclass `CommandConfig`**：`vx` 是定值基线（课程可用
  `curriculum.command_vx` 逐级覆盖），`vy_max` / `yaw_max` 是**采样范围**（课程按级放宽），
  当前值 `[vx, vy, wz]` 存在
  `_now` 列表里、与 `env._command` 是同一个对象。改 `config.yaml` 写这些老键现在会
  报错并列出字段名。见 §4.5。

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
    def _sample_command(self) -> list[float]
    def set_commands(self, vy_max: float | None = None, yaw_max: float | None = None) -> None
    def set_command_now(self, vy: float | None = None, yaw_rate: float | None = None,
                        vx: float | None = None) -> list[float]
    def _success_target_x(self) -> float
    def _success_reached(self) -> bool
    def reset(self, *, seed: int | None = None, options=None)
    def step(self, action: np.ndarray)
    def render(self, camera: str = "side", *, width: int = 640, height: int = 480)
    def close(self) -> None
```

* `set_stair(stair)`：换地形并**重新编译模型**；调用方之后**必须** `reset()`（旧的
  `MjModel`/`MjData` 已被替换，旧状态无意义）。同时刷新 `_success_x` 并把 `_reached_top` 置 `False`。
* `set_gait_prior(enable, gait=None)`：**已在死代码清理中删除**（无调用方；最后的使用者是
  一个早已移除的 CLI 开关）。先验现在只在构造时由 `cfg.gait_prior` / `cfg.gait` 决定。
* **指令三件套**（`set_commands` / `set_command_now` 是给外部控制器的公开接口）：
  * `_sample_command()`：在 `[−vy_max, +vy_max] × [−yaw_max, +yaw_max]` 内重抽 `[vx, vy, wz]`；
    `reset()` 每回合调一次。`vx` 取的是**当前值**而不是基线字段，所以控制器压低的
    前进指令不会被下一次 reset 抹掉。
  * `set_commands(vy_max, yaw_max)`：改**采样范围**，课程在阶段边界调（`scripts/train.py`
    的 `Trainer.make_env` 走 `venv.env_method("set_commands", ...)`）。
  * `set_command_now(vy, yaw_rate, vx)`：改**当前值**，控制器可以每个控制步调一次。
    传了 `vx` 时会**同时改写基线** `cfg.command.vx` —— 否则 `reset()` 重抽会把课程下发的
    定值前进指令丢掉（这是 `curriculum.command_vx` 生效的路径）。
    `env._command` 与 `cfg.command._now` 是**同一个 list 对象**，所以没有两份状态不同步的窗口。
  * ⚠ `vy_max = yaw_max = 0` 时 `_sample_command` 恒返回 `[vx, 0, 0]`，等价于加指令之前的行为。
* `set_flat_success_distance(distance)`：改写平地阶段的前进成功距离并**立刻**刷新
  `_success_x`。指令放宽后机器人会为了跟指令而绕路，原来 3 m 的判据会把成功拖成超时，
  所以平地预热级要能单独调短它（课程键 `curriculum.flat_success_distance`）。
* `yaw_rate` 属性（`robotdog/env.py:460`）：返回**机体系 z 角速度**，即
  `(quat_to_mat(base_quat).T @ data.qvel[3:6])[2]`。**不要**把它换成世界系角速度
  `wz + tan(pitch)·(cos(yaw)·wx + sin(yaw)·wy)` —— 那个量被与指令同阶的步态振荡主导
  （三种独立测法给出 0.165~0.283 rad/s 的偏差，机体系只有 0.13），拿来当奖励参考等于
  让策略去跟一个它控制不了的量，见 `experience.md` §16。
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
    def _obs_frame(self) -> np.ndarray                   # float64，长度 obs.frame_size
    def _observation(self) -> np.ndarray                 # float32，长度 obs.size
    def _compute_reward(self, action: np.ndarray, trunk_contact: bool,
                        foot_contact: np.ndarray | None = None,
                        leg_contact: bool = False, terminated: bool = False
                        ) -> tuple[float, dict[str, float]]
    def _foot_contact_mask(self) -> np.ndarray           # (4,) bool
    def _advance_foot_air_time(self, foot_contact: np.ndarray) -> None
    def _contact_flags(self) -> tuple[bool, bool]        # (trunk_contact, leg_contact)
    def _contact_fatal(self, trunk_contact: bool, leg_contact: bool) -> bool  # 恒 False
    def _support_height(self) -> float    def _support_height_mean(self) -> float
    def _fall_reference_height(self) -> float             # 读 stair.height_at()，只服务终止
    def _terminated(self, trunk_contact: bool, leg_contact: bool) -> tuple[bool, bool]
    def _gait_action(self) -> np.ndarray
    def _target_from_action(self, action: np.ndarray) -> np.ndarray
```

* `_terrain_height(x)`：接受标量或数组；**平地阶段（`n_up == 0 and n_down == 0`）直接返回
  `np.zeros_like`**，不走 `stair.height_at()`。
* `_compute_reward(...)` 是 `robotdog.reward.compute_reward` 的转发（`robotdog/env.py:531`）：
  它调 `compute_reward(self, ...)`，把返回的 `terminal` 写进 `terms["terminal"]`，再返回
  `(base + terminal, terms)`，因此 `reward == sum(terms.values())` 恒成立。
  `foot_contact=None` 时下游会自己调 `_foot_contact_mask()`；`leg_contact=False` 时
  `terms["leg_collision"] == 0.0`。公式与 21 个权重见 [`reward.md`](reward.md)。
* `_advance_foot_air_time(foot_contact)`：一次控制步调用一次（`step` 在奖励之后调），
  `_foot_air_time = where(foot_contact, 0.0, _foot_air_time + control_dt)`，
  并更新 `_prev_foot_contact`。
* `_contact_fatal`：现在**恒返回 `False`**，只维护 `_leg_contact_streak` 供诊断。
  「躯干接触地形就终止回合」那条规则已删（见 §3.8 与 `experience.md` §12）。
* `_target_from_action(action)`：只有一条路径
  `home_angles + _gait_action() + action_scale * action`，用
  `np.clip(target, act_low, act_high)` 收尾（`robotdog/env.py:827`）。
* `_projected_gravity()` 已**删除**（死代码清理）：它定义了 `R^T @ (0,0,-1)` 的展开式，
  却没有任何调用点，观测里也没有重力块。**如果你需要一个「机体系下的重力方向」观测块，
  这段实现只在 HEAD 里能找到**，当前工作区没有。

`step()` 返回的 `info` 字段（**恰好 17 个键，键名逐字**）：
`reward_terms`、`trunk_contact`、`leg_contact`、`leg_contact_streak`、
`base_x`、`base_y`、`base_z`、`terrain_height`、`forward_vel`、`roll`、
`pitch`、`crest_height`、`reach_top_x`、`reached_top`、`is_success`、`fallen`、`gait_prior`。
其中：

* `reward_terms` = 13 个逐步成形项 **加上 `"terminal"`** ⇒ 14 键；`"terminal"` 由
  `_compute_reward` 自己写入（它内部调 `compute_reward` 拿到 `terminal`）；
* `leg_contact_streak` = `_leg_contact_streak`（胫部连续蹭立面的步数，**只诊断、不致命**）；
* `reach_top_x` = `_success_target_x()`；
* `fall` 判据的实际高度 = `base_pos[2] - _fall_reference_height()`，但 `base_z` 报的是**原始**
  `base_pos[2]`，`terrain_height` 报的是**躯干**下的 `terrain_height_under_base`
  （不是终止用的参照面）；
* `is_success = bool(_reached_top)`；`fallen = bool(terminated) and not self._reached_top`。

### 4.5 每个成员是干什么的、谁在调它

`Go2StairsEnv` 共 **37 个成员**（不含 dunder）。按「谁读它」分成三类，调用点行号都是实测值
（`.tmp/env_member_use.py` / `.tmp/env_internal_calls.py`）。

**① 只读属性（13 个，各 2~3 行）—— 把 `data` 翻成人话**

| 成员 | 定义 | 返回什么 | 谁在用 |
| --- | --- | --- | --- |
| `base_pos` | `:408` | `qpos[0:3]`，机身世界系位置 | `reward.py:176`（高度项）、`play.py:280/295-298`（前进距离） |
| `base_rpy` | `:412` | `quat_to_rpy(qpos[3:7])`，roll/pitch/yaw | `reward.py:104`（姿态项）、`env.py:512`（终止判据） |
| `base_quat` | `:416` | `qpos[3:7]`，四元数 `(w,x,y,z)` | 只被 `base_lin_vel_body` 用（`env.py:402`） |
| `base_lin_vel_world` | `:420` | `qvel[0:3]`，**世界**系线速度 | `reward.py:105`（前进速度项）、`env.py:618`（`info["forward_vel"]`） |
| `base_lin_vel_body` | `:424` | `R(q)^T @ qvel[0:3]`，机体系线速度 | **观测块**（`env.py:336`，`config.yaml` 已开） |
| `base_ang_vel_body` | `:434` | `qvel[3:6]`，陀螺读数 | **观测块**（`env.py:332`，恒开） |
| `joint_pos` | `:439` | `qpos[qpos_joint_adr]`，12 个关节角 | **观测块**（`env.py:329`）、`reward.py:183`（关节限位） |
| `joint_vel` | `:443` | `qvel[qvel_joint_adr]`，12 个关节速度 | **观测块**（`env.py:330`）、`reward.py:178`（关节速度惩罚） |
| `foot_pos` | `:447` | `(4,3)` 四个足端**球心**的世界位置 | `reward.py:129`（足端速度门控）、`env.py:302/549/578/623/637/647` |
| `terrain_height_under_base` | `:452` | `stair.height_at(base_pos[0])`，**躯干**下方地形高 | `reward.py:176/192`、`env.py:250/518/528/617` |
| `yaw` | `:456` | `base_rpy[2]` | **目前没有任何调用点**（保留的理由：平地/未来朝向条件任务需要，且它是文档化的公开属性） |
| `yaw_rate` | `:460` | **机体系** z 角速度 `(R^T @ qvel[3:6])[2]` | `reward.py` 的 `command_follow` 偏航项、`step` 的 `info["yaw_rate"]`。**不要换成世界系公式**，见 §4.4 与 `experience.md` §16 |
| `foot_clearance` | `:479` | `(4,)` 足端**表面**到其正下方地形的间隙（已减足端半径） | `reward.py:140`（足端抬升项） |
| `probe_target` | `:620` | `cfg.reward.probe_lift_target`，`probe_lift` 的起算线 [m] | `reward.py` 的 `probe_lift` 势函数塑形 |

**② 内部辅助（18 个）—— 每个只回答一个问题**

| 成员 | 定义 | 作用 | 内部调用点 |
| --- | --- | --- | --- |
| `_build_model()` | `:284` | 编译 MJCF、建 `MjModel`/`MjData`、抽 `meta` | `__init__` `:170`、`set_stair` `:238` |
| `_success_target_x()` | `:385` | 成功要越过的 x（楼梯 = 顶端远边缘；平地 = 固定前进距离） | `:194`、`:240` |
| `_success_reached()` | `:397` | 本步是否已登顶（x 越过顶端 **且** 高度到位） | `step` `:626` |
| `_terrain_height(x)` | `:334` | 任意 x 处的地形高；**平地直接返回 0**，不走 `height_at` | `foot_clearance` `:332` |
| `_obs_frame()` | `:498` | 装配**单帧**观测（按 `ObsConfig` 开关拼接） | `reset` `:564`、`step` `:643` |
| `_observation()` | `:526` | `np.concatenate(_hist)` → float32，长度 `obs.size` | `reset` `:568`、`step` `:664` |
| `_compute_reward(...)` | `:370` | 转发 `robotdog.reward.compute_reward`，把 `terminal` 塞进 `terms` | `step` `:632` |
| `_foot_contact_mask()` | `:544` | `(4,)` bool：每只足端 geom 是否接触地形 | `reset` `:558`、`step` `:620`；**也被 `reward.py:163` 跨模块调用**（`air_time` 项 `foot_contact=None` 时） |
| `_advance_foot_air_time(fc)` | `:404` | 累加腾空时间、刷新 `_prev_foot_contact` | `step` `:636`（**必须在奖励之后**） |
| `_contact_flags()` | `:652` | `(trunk_contact, leg_contact)`：躯干 / 腿部是否接触地形 | `step` `:619` |
| `_contact_fatal(tc, lc)` | `:449` | **恒返回 `False`**，只维护 `_leg_contact_streak`（诊断用，永不致命）。躯干接触的致命判据与 `_effective_contact_steps()` 都已删除 | `_terminated` `:518` |
| `_support_height()` | `:709` | 四足下方**最低**地形高（真正撑住机器人的那个面） | `_fall_reference_height` `:508` |
| `_support_height_mean()` | `:723` | 四足下方地形高的**均值**（可选终止参照，见下） | `_fall_reference_height` `:505` |
| `_fall_reference_height()` | `:733` | 按 `terminate_height_ref` 三选一：`support_mean` / `support` / 其它→躯干下 | `_terminated` `:512` |
| `_terminated(tc, lc)` | `:510` | `(terminated, truncated)`：摔倒（高度 / 倾角）/ 超时 | `step` `:621` |
| `_gait_action()` | `:817` | 先验在当前相位的关节角偏移；先验关时恒零 | `_target_from_action` `:596` |
| `_target_from_action(a)` | `:580` | 唯一的动作映射：`home + gait + action_scale*a`，然后 clip | `step` `:606` |

> `_support_height()` 与 `_support_height_mean()` 只差一个 `min`/`mean`，但**不是「更宽松」**：
> 跨越立面时均值高于最小值，所以会**更早**触发终止；它消除的是 **单帧不连续**
> （`min` 会在最后一只脚落地时一帧跳一整个立面 ⇒ 幽灵摔倒）。
> 当前 `config.yaml` 用默认的 `"support"`。

**③ Gymnasium 接口（`__init__` / `reset` / `step` / `render` / `close`）**

| 成员 | 定义 | 谁在调它 |
| --- | --- | --- |
| `__init__(stair=None, cfg=None, *, seed=None)` | `:157` | `scripts/train.py:282` 的 `make_env_fn`、`scripts/play.py` |
| `reset(*, seed=None, options=None)` | `:524` | SB3 自动调；`scripts/play.py:279` 手动调（每回合带新 seed）；`env.py:239` 的 `set_stair` 里也调 |
| `step(action)` | `:600` | SB3 自动调；`scripts/play.py:288` 手动调（**注意**：`env.py` 里那条 `def step` 上方的 docstring 是说明，不是调用） |
| `render(camera="side", *, width, height)` | `:671` | 只有 `scripts/play.py:291`（GIF 与静帧） |
| `close()` | `:988` | `scripts/play.py:458`、`scripts/train.py:290` |
| `set_stair(stair)` | `:227` | **当前没有任何调用点**。唯一调用者 `training.set_terrain()` 已在本轮清理中删除；课程每一级走 `make_vec_env` 新建环境，而不是换地形。保留是因为「换地形」是真需求且文档把它当公开 API |

**当前实际用到哪些？**

* **`config.yaml` 已开的观测块**只用到 4 个属性：`joint_pos`、`joint_vel`、`base_ang_vel_body`、
  `base_lin_vel_body`（外加 `base_rpy` 恒在基础帧里）。`base_height` **已被硬性禁止**
  （`ObsConfig.__post_init__` 抛 `ValueError`），`gait_phase` 关着；`_height_reference()`
  这个方法本身也已随该分支删除。
* **两个零调用点成员**：`yaw`、`set_stair()`。两者都是**文档化的公开属性/方法**，
  按本仓的判据保留（见 `experience.md` §9 的「三类假死」）。
* 其余成员全部有真实读者：奖励侧 9 处（`reward.py` 的 `:104/105/129/134/140/158-165/176-183/192-200`）、
  播放侧 `play.py`、以及 `step`/`reset` 自身的内部流水线。
* **`_foot_contact_mask()` 是唯一被跨模块调用的下划线方法**（`reward.py:163`）—— 改它的签名
  要同时改 `robotdog/reward.py`。

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
11. `_step_count = 0`、`_reached_top = False`、`_leg_contact_streak = 0`、`_peak_terrain_height = terrain_height_under_base`。
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
11. `reward, terms = _compute_reward(action, trunk_contact, foot_contact,
    leg_contact=leg_contact, terminated=terminated)` —— 终止奖励在这一步内部结算
    （`robotdog/env.py:531`）。
12. `_advance_foot_air_time(foot_contact)`（**必须在奖励之后**：`air_time` 读的是落地那一步的
    腾空时长，而奖励已经用掉了这一步的 `_foot_air_time`）。
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
`_height_reference()` 直接返回 `terrain_height_under_base`，不加任何偏置
（实现见 `robotdog/env.py:461-464`）。

### 5.5 奖励调用链

```
_compute_reward(action, trunk_contact, foot_contact, leg_contact, terminated)   [env]
  → compute_reward(self, action, ...)                                          [reward]
      → base_rpy / base_lin_vel_world / nominal_pose
      → foot_pos, foot_clearance, _prev_foot_pos, stair.step_height
      → _prev_action, _foot_air_time, _prev_foot_contact
      → terrain_height_under_base, _start_terrain_height, _reached_top
      → sum(terms.values()), terms        # 13 键
      → terminal = success if _reached_top else termination   # 仅 terminated
  → terms["terminal"] = terminal；返回 (base + terminal, terms)                 # 14 键
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
      → 恒返回 False（躯干接触致命判据已删除）
      → 只维护 _leg_contact_streak（诊断用，永不致命）
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
| `action_scale` | `0.30` rad |
| `command.vx` | `0.55` m/s（定值；`vy` / `wz` 由控制器给） |
| `gait_prior` | `False` |
| `gait` | `GaitParams.default_trot()`（`freq=6.0`，`amp_th=0.24914502679724`，`amp_ca=0.5518579715236906`，`ph_th=1.6754541487333945`，`ph_ca=-2.146505768979483`，`off_th=0.25539290883226595`，`off_ca=0.09025271647195035`，`calf_shape="sin"`，`action_scale=0.90`，`leg_phase=(0.0, pi, pi, 0.0)`） |
| `gait_gain` | `1.0` |

**reset 噪声与出生**：`reset_z=0.32`、`reset_joint_noise=0.06`、`reset_xy_noise=0.12`、
`reset_yaw_noise=0.10`、`spawn_back=0.55`、`flat_success_distance=3.0`、
初速度 `U(-0.05, 0.05)`（施加到全部 `nv`）。

**终止**：`terminate_height_ref="support"`、`terminate_min_height=0.15`、`terminate_max_tilt=1.0`、
`terminate_on_success=True`。

**观测块组合**：`base_height` 已被 `__post_init__` 硬性禁止，所以合法组合只剩
四个开关（`base_lin_vel_body` / `gait_phase` / `step_height` / `command`）的 **16 种**：

```
frame_size = 42
           + 3 * base_lin_vel_body
           + 2 * gait_phase
           + 1 * step_height
           + 2 * command
```

| `base_lin_vel_body` | `gait_phase` | `step_height` | `command` | `frame_size` | `size`（`history_len=10`） |
| --- | --- | --- | --- | --- | --- |
| F | F | F | F | 42 | 420 |
| T | F | F | F | 45 | 450 |
| F | T | F | F | 44 | 440 |
| T | T | F | F | 47 | 470 |
| F | F | T | F | 43 | 430 |
| T | F | T | F | 46 | 460 |
| F | T | T | F | 45 | 450 |
| T | T | T | F | 48 | 480 |
| F | F | F | T | 44 | 440 |
| T | F | F | T | 47 | 470 |
| F | T | F | T | 46 | 460 |
| T | T | F | T | 49 | 490 |
| F | F | T | T | 45 | 450 |
| **T** | **F** | **T** | **T** | **48** | **480** ← dataclass 默认；`config.yaml` 把 `history_len` 设成 1 ⇒ **48 维** |
| F | T | T | T | 47 | 470 |
| T | T | T | T | 50 | 500 |

`command` 在 `config.yaml` 里**恒为 true**：PPO 的网络与 RolloutBuffer 维数不可变，
课程只能改指令范围、不能开关这 2 维。

**奖励**：30 个字段的默认值见 §4.2；其中 `[SPEC]` 九项
`forward_vel=2.5`（σ`0.25`）/ `orientation=-1.2` / `torque=-1.5e-4` / `foot_lift=0.9` /
`body_collision=-1.2` / `lateral_vel=1.5`（σ`0.25`）/ `yaw_rate=1.0`（σ`0.25`）；
`alive=0.15`、`termination=-15.0`、`success=50.0`、
`foot_lift_target=0.08`、`foot_lift_step_ratio=1.4`、`foot_lift_gate_speed=0.25`、
`probe_lift=1.0` / `probe_lift_target=0.06` / `probe_blocked_frac=0.30`、
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
2. **`robotdog/reward.py` 的 `compute_reward()` 返回值里没有 `terminal`。** 它返回
   `(sum(terms.values()), terms, terminal)`（13 键）；补写 `terms["terminal"]` 的是
   `Go2StairsEnv._compute_reward()`。因此「`sum(reward_terms) == reward`」这条恒等式
   对 `info["reward_terms"]` 成立（14 键），对 `robotdog/reward.py` 的 `terms` 也成立
   （13 键，无 `terminal`），但**不是同一个 dict**。要按 key 逐项对账时注意这一点。
3. **`_contact_fatal()` 现在恒返回 `False`**，只维护 `_leg_contact_streak`（诊断用）。
   躯干接触的致命判据与 `_trunk_contact_streak` 都已删除（理由见 `experience.md` §12：
   躯干在 `h ≈ 0.28 m` 的安全姿态下贴立面蹭可以连续到 8 步，阈值 10 只差 2 步，
   纯风险零收益，而且那条规则从未生效过）。躯干接触的代价由
   `reward.body_collision`（逐步、有界）承担。
4. **`terminate_height_ref` 的默认值已经换成 `"support"`**：任何按老文档写
   `cfg.terminate_height_ref = "base"` 的脚本仍然能跑，但会重新打开幽灵摔倒
   （`base_x=1.503 → Δref=+0.1100`、`base_x=1.491 → Δref=-0.1100`，`experience.md` §1.4）。
   非法取值的处理是**静默回退到 `terrain_height_under_base`**，不会报错。
5. **`command_vy` / `command_yaw_rate` / `action_mode` 都已删除**：奖励里从来没有横向速度项、
   没有 yaw rate 项，`action_mode` 的 `"absolute"` 分支也从未被用过。把它们写进
   `config.yaml` 现在会走 `_apply` 的未知键报错并列出可用字段，不会静默忽略。
6. **`yaw_rate` 只能用机体系**。见 §4.4 的说明与 `experience.md` §16：世界系 z 角速度
   在 trot 步态下的振荡幅度与指令量程相当，会让「跟指令」这一项变成噪声。
6. **`obs.size` 一定是 `history_len × frame_size`**：`base_frame_size` 由开关固定为
   42~49，`history_len` 是 `config.yaml` 的 `env.obs.history_len`（`ObsConfig` dataclass
   默认 10，**本仓现值 1**，见 §3.2）。所以 `history_len=1` 时合法 `obs.size` 是 42~49，
   `history_len=10` 时是 420/430/…/490（§6 表）。`robotdog/policy.py` 的
   docstring 拿「474 维」举例，那个数**不是** `10 × frame_size` 能凑出来的（474/10 = 47.4）；
   `_obs_config_for_dim(total, history_len=10)`（该函数的默认值仍是 10）因此也永远无法反推出它；
   反推本仓权重时要显式传 `history_len=1`，或者更稳妥地用权重自带的 `_env.json`。
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
14. **`_projected_gravity()` 已删除**（死代码清理，无调用点）。想验证「躯干倾角」是否能做特征时，
    注意 `base_rpy` 已经在基础 42 维里。
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

cfg = EnvConfig()                    # 默认 obs.frame_size=48（base_lin_vel_body + step_height + command）
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
      f"leg_streak={info['leg_contact_streak']}")
print({k: round(v, 4) for k, v in info["reward_terms"].items()})   # 14 键（含 terminal）

frame = env.render("track", width=640, height=480)   # (480, 640, 3) uint8
env.close()
```
