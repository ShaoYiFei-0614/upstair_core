# 陷阱库（训练不收敛 / 指标反常时先查这里）

来源：`experience.md`（1151 行，§N 为原文章节号，可回溯）。
读法：每条四段式 **症状 → 根因 → 修法 → 判据/实测数字**，末尾 `→ 新项目` 是复用建议。
适用：从零搭建另一个腿式/其他机器人仿真 + RL 训练仓。

---

## 1. 地形几何与物理建模（§1）

### 1.1 踏步立面与顶面共面接缝把脚夹死
- **症状**：8 cm × 8 级楼梯只有 1/10 回合能上，9.0 cm 是 0%。
- **根因**：脚球心落在两级踏面的共享平面上时，两个 box 沿最小穿透轴反向推它，z 向无支撑 ⇒ 卡死。实测两只前脚在 `x = 5.500 == up_end_x` 冻结 700 个控制步，离终点只差 0.114 m。
- **修法**：`robotdog/tools/terrain.py` 的 `CREST_SEAM_OVERLAP = 1.0` —— 顶层平台向最后一级**回伸一个踏面**，让接缝不再落在立面棱上。
- **判据/数字**：修前 8 cm 1/10 → 修后 8/10；9.0 cm 0% → 65%。
- → 新项目：台阶几何一律让相邻 box **重叠 ≥ 半个踏面**，绝不让两个碰撞体的棱共面；建完先跑「脚心坐标 == 接缝坐标」的静态探测。

### 1.2 地形 geomgroup 被 MuJoCo 默认隐藏
- **症状**：狗在爬空气，HUD 却报高度。
- **根因**：`mjv_defaultOption` 默认隐藏 geomgroup 3/4/5，而 Go2 的 MJCF 占用了 group 2/3。
- **修法**：地形一律用 `TERRAIN_GEOM_GROUP = 1`；`render()` 里强制打开全部 6 个组。
- → 新项目：地形/场景 geom 的 group 号必须先查**机器人 MJCF 占了哪几个组**，再定常量，不要照抄别处的默认值。

### 1.3 零高度台阶无法编译
- **症状**：`demo_stairs_spec(step_height=0.0, n_up=8)` 编译失败。
- **根因**：生成的 box 是 `size="0.250000 1.000000 0.000000"`，MuJoCo 抛 `ValueError: Error: size 2 must be positive in geom`（`robotdog/go2_model.py` 包装成 `RuntimeError: MJCF compile failed` 并 dump 到 `.generated/failed_scene.xml`）。
- **修法**：平地走 `flat_spec(n_up=0, n_down=0, step_height=0.0)`（`n_up=0` 时不生成任何台阶 box），**不要**用「8 级零高度台阶」表示平地。
- → 新项目：地形生成器要显式区分「空地」与「零参数地形」两条路径；每条断言都该带一个可复现的最小 CLI 参数。

### 1.4 摔倒判据用了阶跃函数 ⇒ 幽灵摔倒
- **症状**：机器人站得稳稳的却突然判摔。
- **根因**：`terrain_height_under_base` 是**阶跃函数** —— 实测 `base_x=1.503` 时 `Δref=+0.1100`，紧接 `base_x=1.491` 时 `Δref=-0.1100`，逐帧正负翻转。
- **修法**：`terminate_height_ref = "support"`，参照面改成**四足足下地形的最低点**；另有 `"support_mean"`（四足足下均值）与 `"base"`（旧口径）。
- → 新项目：任何「机器人下方地形高度」查询在台阶地形上都是阶跃的；参考高度一律取**多接触点聚合**（min/mean over feet），并在设计文档里列出可选口径。

### 1.5 `base_height` 观测参照面同源阶跃（§3.3）
- **症状**：观测里的机身高度在立面处逐帧乱跳。
- **根因**：与 A4 同源，`terrain_height_under_base` 在立面处逐帧翻转。
- **修法**：观测侧加 `base_height_support` 开关，改用足下参照面。**注意：不要把同一开关接到奖励上** —— 实测均回报会从 +554 崩到 -674/回合。
- **判据**：观测与奖励必须能用**独立的单变量开关**分别验证。
- → 新项目：同一物理量的「观测口径」与「奖励口径」是两个独立旋钮，永远不要共用一个布尔量。

---

## 2. 奖励塑形（§2、§12、§13、§15）

### 2.1 惩罚项没有上限 ⇒ 单步毁掉整个 value function
- **症状**：某一步的回报是 O(10^5) 量级，训练直接崩。
- **根因**：`touch_excess_clip` 之类的裁剪缺失时，单个控制步的惩罚可以任意大。
- **修法**：所有惩罚项都要有上限；`clip` 的常数要按「物理量纲」定，不要按「希望它多痛」定。
- → 新项目：`RewardConfig` 每个 field 强制配一个 clip；上线前跑一遍「单步最坏值」打印。

### 2.2 ⭐ 高斯分母漏平方：`vx = 0` 仍白拿 +0.86/步（§2.2 → §12）
- **症状**：平地训 120k 步，前进速度 0.00 m/s，策略学会站着不动。
- **根因**：`RewardConfig.forward_vel_sigma = 0.25` 是**标准差**，代码写 `exp(-(vx-cmd)**2 / rc.forward_vel_sigma)`，等于按 `sqrt(0.25)=0.5 m/s` 折算 ⇒ `vx=0` 时 `exp(-(0.55)²/0.25)=exp(-1.21)≈0.30` ⇒ **+0.72/步**（实测 +0.86/步）。
- **修法**：分母改 `rc.forward_vel_sigma ** 2`：`vx=0` 时 +0.72/步 → **+0.02/步**。**同时删除** `forward_progress` / `forward_progress_max` / `forward_progress_min` 三个字段及线性项（σ 修正后高斯项已足够小；而线性项在 0.55 m/s 附近给 **+1.1/步**的超线性加成，反而扭曲速度跟踪）。`RewardConfig` 字段 **24 → 21**。
  - ⚠ §2.2 里那版「加线性项 + alive 0.4→0.12」是**错修**，已加引用块更正，别再按它调。
- **判据**：`exp(-4.84)` vs `exp(-1.21)` 是静默的 **3.8 倍**放大，无任何报错。
- → 新项目：所有字段名含 `sigma`/`std` 的分母一律写成 `** 2`，并在命名上区分 `sigma`（标准差）与 `var`（方差）；单测断言 `exp` 项在误差 0 处 == 1。

### 2.3 抬升目标写死 0.08 m ⇒ 在 8 cm 处完全饱和
- **症状**：course13 卡在 7.3 cm（4%）、course14 卡在 7.8 cm（8%），而 6.4 cm 能到 80%。
- **根因**：`foot_lift_target = 0.08` 是常数，8 cm 以上所有台阶得到的抬升分数完全一样（都饱和），策略收不到「还要更高」的信号。
- **修法**：`foot_lift_step_ratio = 1.4` —— 有效目标 `= max(0.08, 1.4 × 台阶高)`。5 cm 时等于 0.08，与旧行为**逐位相同**。
- → 新项目：任何「目标高度/力/速度」都要按**当前地形/任务难度**缩放，并保证低难度区与旧版本逐位一致（便于 A/B）。

### 2.4 ⭐ 与前进无关却每步都付的正项（§12.1-12.3、§13.3）
- **症状**：`stair12`/`stair15` 上「摔倒率 0%、超时率 100%」，而**平均回报 688~822**，比「平地成功走完 3 m」（回报 358）高一倍以上 ⇒ 站着不动比走完更赚。
  - 零动作分档结算（400 步上限）：`0 cm` 113 步 **+334.5**（+2.960/步，`forward_progress +204.6` / `forward_vel +61.5` / `terminal +50`）；`8 cm` 400 步 +90.6（+0.226/步）；`15 cm` 400 步 **−245.2**（−0.613/步）。
- **根因**：两个机制叠加 —— (1) `forward_vel` 分母漏平方（见 B2）；(2) `stair_progress` **每步都付**，读 `terrain_height_under_base - _start_terrain_height`，只要机身正下方是高台阶就一直拿钱。
- **修法**：`stair_progress` 改**一次性结算**（只对新高付钱）：`climbed = max(0.0, env.terrain_height_under_base - env._peak_terrain_height)`，`if climbed > 0.0: env._peak_terrain_height = env.terrain_height_under_base`。累计总量不变（仍 `stair_progress × 爬升高度`），但站桩拿不到新钱、同一级往返不付两次；`__init__` 与 `reset()` 各加一个 `_peak_terrain_height`。
- **判据**：15 cm 站桩 **+0.226/步 → −1.435/步**（−1.66/步），前进 −0.446/步 ⇒ **站桩不再优于前进**。诚实注记：`0 cm 静止` 仍 +0.849/步，那是 trot 先验在走路（触发 `terminal=+50`），不是站桩。
- → 新项目：新加任何**逐步**正项后，第 0 件事是打 `reward_terms` 并算「机器人什么都不做」时的每步值；「每步都付」+「与目标无关」= 局部最优陷阱。

### 2.5 可选奖励项不加门控 ⇒ 又造一个白拿正项（§13.3）
- **症状**：加了横向/偏航指令项后，机器人**只是直走**，却拿到 `lateral_vel = +139.73`、`yaw_rate = +28.03`（5 cm，125 步）。
- **根因**：`exp(-(vy_err**2)/sigma**2)` 在误差恰好落在 σ 附近时凭空给约 **+1.0/步**；期望值本该是 `1.5×125=187.5` 与 `1.0×125=125`。与 B2 是**完全同一类 bug**（同一作者犯了两次）。
- **修法**：门控 + 合并键。只在指令范围真放开时才计分：`if cmd.vy_max:` / `if cmd.yaw_max:`，`follow = max(follow, ...)` 合并成单键 `command_follow`（使「只往前走」阶段键集合与加指令前**逐位一致**）。
- **判据**：修后 `command_follow = +0.000e+00`（逐位精确 0）。
- **后续（§18）**：`max` 合并被改成**求和** —— 取 max 时哪个方向误差小就只保留那一个，横向跟得还行时偏航梯度被整个丢掉（正是 §17「横向 r=+0.661 有微弱响应、偏航 r=−0.085 完全无响应」的直接原因）。求和让两方向各自有梯度，同时保留 `if` 门控。
- → 新项目：可选奖励项必须有 `if 能力已开启:` 门控 —— 这不是优化，是**正确性**：它让「没开这个能力」与「加这个能力之前」在数值上完全等价。

### 2.6 走直线项权重太小 ⇒ 形同不存在，调大又反噬（§2.8）
- **根因**：`straight_line = -2.0` 时只占 200 分回报的 0.65%，策略根本不理它。
- **修法**：提到 -20.0。**但要注意反噬**：它和 `heading_max` 的裁剪合起来会把横向漂移变成**截断后饱和的巨大惩罚**，实测能占**全部惩罚量的 99.9%**，把策略压回「几乎不动」。
- **最终决策**：整组侧向奖励（`lateral_vel`、`yaw_rate`、`straight_line`、`heading`、`lateral_vel_track`）**删除**，改由 `forward_vel` 与 `base_height` 承担方向约束。逐项审计：`straight_line` 一项 −12.67/步（72.7%），而唯一前进奖励 `forward_vel` 只有 +0.36/步；机器人明明能爬 2/3 段楼梯，却因横向漂移 0.8~1.3 m 被罚到 −17.4/步、**−6622/回合**。
- → 新项目：**删掉一整个奖励族比调权重更安全**；若某项惩罚占比 > 50%，先怀疑裁剪饱和而不是「权重不够」。

### 2.7 落地力/蹭立面阈值是常数 ⇒ 高难度上从提醒变乱罚（§2.5-2.6）
- **根因**：`touch_impact_target = 95 N` 是常数，高台阶上正常落地的冲击也超阈值；`riser_graze` 只看脚的法向力方向，分不清「顶着立面推进」和「贴着立面抬高」。
- **修法**：`touch_impact_step_scale` 让参考力随台阶高放大；`riser_graze_mode = "lifting"` —— 只有**所有蹭立面的脚都没有向上速度**（`vz <= riser_graze_vz_eps = 0.10 m/s`）时才罚。
- **注意**：触觉相关项（`touch`、`stair_probe`、接触力）已随「盲走」重构**全部删除**（原 `robotdog/tools/touch.py` 与 `env.py` 里 91 处引用）。
- → 新项目：力/接触阈值必须随地形难度缩放；若最终目标是盲走，就别在早期把触觉奖励写进主线（会形成依赖）。

### 2.8 `stair_progress` 奖励「躯干冲线」而不是「脚上了一级」（§2.7）
- **根因**：参考面取躯干正下方地形，躯干一到立面上方就拿到整级进度分 ⇒ 策略学会「把躯干往前扑」而不是「把脚踩上去」。
- **修法**：`stair_progress_foot_ref = True` 改用足端参照面，并加 `stair_progress_min_loaded_feet` 要求至少 N 只脚受力。
- **注意**：这两项已随重构删除，现在 `stair_progress` 回到躯干口径（`README.md` §3.3）。
- → 新项目：进度类奖励的**参考点必须与被激励的行为同源**（激励抬脚就用足端，激励推进才用躯干）。

### 2.9 终止型判据换成逐步惩罚：腿碰撞（§2.4）
- **症状**：course12 在 10 cm 上平均 62.6 个控制步就结束，从未越过 `x = 1.42`（第一级立面在 1.50）。
- **根因**：小腿擦到台阶就终止回合，策略连「撞一次、学一次」的机会都没有。
- **修法**：腿接触改成**逐帧小惩罚** `leg_collision = -0.25`，不终止；只有躯干接触才是致命的（且要连续 `terminate_contact_steps` 步）。
- **后续更正（§12.4）**：那半句「躯干接触致命 + 连续 10 步去抖」**也已删除**，见 E3。
- → 新项目：**碰撞/擦碰默认用有界逐步惩罚，不用终止**；终止权只留给不可恢复状态（高度、倾角、超时）。

### 2.10 「被阻挡时抬高抬腿」势函数塑形（§15）
- **症状（尝试一）**：检测链条根本没通 —— 500 步里 `probe_count = 0`、`probe_lift = +0.000`，即使门槛降到 0.02 m 也不触发。根因：摆动峰值写成 `np.where(foot_contact, 0.0, np.maximum(...))`，**一接触地形就清零**，而结算发生在落地帧（`landed = foot_contact & ~_prev_foot_contact`），读到的正是刚被清零的旧值，比真实峰值低 0.01~0.03 m。
  - **修法**：清零必须发生在结算**之后** —— 落地那帧保留旧峰值：
    ```python
    self._peak_swing_clearance = np.where(landed, self._peak_swing_clearance,
        np.where(foot_contact, 0.0, np.maximum(self._peak_swing_clearance, clearance)))
    ```
  - **定位手段**：临时逐帧打印（`PROBE_DEBUG` 环境变量门控），打完就删。教训：**「有梯度但没人走」与「根本没梯度」从回报上看不出区别**。
- **症状（尝试二 · 门槛标定）**：原设计门槛 `1.6 × 台阶高` —— 9 cm 门槛 0.144（被阻挡帧峰值 均值 0.0444 / p99 0.1698）、12 cm 0.192（0.0271 / 0.1054）、15 cm 0.240（0.0254~0.0261 / p99 0.0959~0.1079）。
  - **根因**：`action_scale=0.30` 下 7 种解析动作最好值只有 **0.1479 m**（`thigh +1 calf +1`）；放大到 `action_scale=1.00` 才有 `hip+1 thigh+1 calf+1` 的 0.2576 m 与随机动作的 0.5845 m ⇒ **15 cm 的 0.24 m 门槛在 0.30 权限下不可达，梯度恒为零**。
  - **修法**：门槛改成跟「当前策略自己能抬多高」比 —— `probe_lift_target = 0.06`（普通步态峰值带 0.06~0.12 的下沿）。
- **修法（尝试三 · 势函数）**：不再「落地帧越线给固定分」，而是「被阻挡期间最高摆动峰值又提高了多少就付多少」：
  ```python
  floor = max(env._probe_best, env.probe_target)      # probe_target = 0.06
  gain = float(np.sum(np.clip(peaks - floor, 0.0, None)))
  terms["probe_lift"] = rc.probe_lift * gain
  ```
  性质三条：**总量有界**、**不可刷分**、**有梯度**；`_update_probe_state()` 在解除阻挡时把 `_probe_best` 清零。
- **判据/量级**（`probe_lift = 1.0`，400 步回合）：9 cm 零动作 0.392 / `thigh+1` 0.943 / 随机最好 1.189（峰值 0.2793）；15 cm 0.141 / 0.073 / 0.909（峰值 0.2078）。**15 cm 上 `probe_lift ≈0.9/回合` 远不足以补 `body_collision ≈−548/回合`** —— 它只是把「抬更高」变成可优化方向，**不是解药**。
- → 新项目：奖励项先做**可达性标定**再写权重（永远拿不到的奖励与权重 0 对训练等价，但对调试者是完全不同的错觉）；门槛跟**当前策略能力**走，不跟任务参数走；优先用**势函数**而非「事件越线给分」。

---

## 3. 观测设计（§3、§11）

### 3.1 策略被打分却看不到被打分的量（§3.1）
- **症状**：奖励里有四个纯 `vy`/`wz` 函数项（横向速度、航向误差等），策略却上不去。
- **根因**：观测里**没有** `vy`/`wz`，策略无法知道自己正被扣分；更糟的是老判据把「横向漂移」按**出生位姿**量，等于让策略奖励自己保持随机朝向。
- **修法**：先把被扣分的量加进观测；漂移改成按**楼梯中线**量（§3.2）。
- **后续**：整组侧向奖励已随重构删除，观测里也相应不再有漂移量。
- → 新项目：奖励里出现的每一个物理量，先问「它在观测里吗」；不在就补上或不罚。

### 3.2 漂移量按「出生位姿」量 ⇒ 奖励保持随机朝向（§3.2）
- **根因**：按 `y0/yaw0`（reset 时的随机位姿）量漂移。
- **修法**：按楼梯中线 `stair.center_y` 与楼梯轴量。
- **判据**：平地上这个量能被验证成 0：末 20% − 首 20% 的航向差**只有 -0.53°**。
- → 新项目：任何「偏移/对齐」必须有**世界坐标绝对参照系**（场地中线/轴），绝不能用出生位姿。

### 3.3 残差策略不知道自己在步态周期哪一点（§3.4）
- **根因**：`gait.radians(t)` 是周期函数，策略要在它之上做残差，却看不到相位。
- **修法**：`gait_phase` 观测块给 `sin/cos` 两个值。
- → 新项目：凡是有**开环先验/周期性基元**叠加残差，观测必须包含相位（sin/cos 成对，避免 2π 跳变）。

### 3.4 ⭐ `base_height` 从「默认关掉」升级为「禁止打开」（§11）
- **症状**：`obs.base_height` 一直是 `false`，但只靠**默认值和一行注释**守住：
  ```python
  if obs_cfg.base_height:
      parts.append(np.array([self.base_pos[2] - self._height_reference()]))   # _height_reference -> terrain_height_under_base
  ```
  只要 `config.yaml` 里改成 `true`，**机身正下方地形几何真值就进了观测**，盲走任务变成「直接读地形高度」，训出的策略真机上没有对应传感器。该开关历史上真被打开过一次（§3.3 事故）。
- **根因**：**约定不是保证** —— 「默认关掉 + 注释」只对读注释的人有效。
- **修法（四步）**：(1) `ObsConfig.__post_init__` 在 `base_height=True` 时**直接抛 `ValueError`**，错误信息说明为何含真值及替代做法（保持 `step_height`）；(2) 删掉 `_obs_frame()` 那条分支并删 `_height_reference()` 本身 —— 让「往观测塞真值」**在结构上不可能**；(3) `robotdog/policy.py` 的 `_obs_config_for_dim()`（按观测维度反推开关）把 `base_height` 移出枚举，副作用是旧维数 474 反推返回 `None` 而非崩（刻意，那些归档 checkpoint 与现维数 46/48 本就不兼容）；(4) `config.yaml` 注释升级为「**必须 false，且不能改成 true**」。
- **审计结论（§11.4）**：「读了什么」与「它去了哪里」要分开判 —— 本仓**允许读真值做奖励与终止，只禁止进观测**。
  | 量 | 真值来源 | 进观测 | 结论 |
  |---|---|---|---|
  | `_contact_flags()` | 接触对（躯干/腿 × 地形） | 否，只进奖励与终止 | 可接受 |
  | `_foot_contact_mask()` | 接触对（足端 × 地形） | 否，只进 `air_time` | 可接受（真机有力矩残差/电流尖峰/IMU 冲击等价估计） |
  | `_support_height()` / `_support_height_mean()` | `stair.height_at()` 地形几何 | 否，只进摔倒判据 | **特权信息**，但终止本就是上帝视角，允许 |
  | `foot_clearance` | 足端位置 − 地形高 | 否，只进 `foot_lift` 奖励 | **特权信息**，策略只能学「按 `step_height` 抬够高」的相关性 |
  | `base_height`（观测） | `terrain_height_under_base` | **是** | ⭐ 破坏盲走，已删除并禁止 |
  结论：**只有 `base_height` 一项**真正破坏盲走前提；观测逐项都能落到真机传感器（编码器 / IMU / 上一步动作 / 本体感觉 / 上层指令），唯一非传感器项是标量任务参数 `step_height`。
- → 新项目：破坏前提的开关要在 `__post_init__` 报错封死**并且删掉实现路径**（留代码就是留被重新接上的可能）；「是否引入特权信息」审计是**一次性判断题，没有任何自动检查器能抓**，结论必须写进文档（同 §10 行号漂移）。

---

## 4. 课程与训练（§4、§14、§17、§18）

### 4.1 ⭐ 课程「毕业」判据把 PPO 的梯度更新饿死（§4.1）
- **症状**：策略成功率和「零动作纯先验」一模一样。`scripts/diagnose.py policy` 同地形同 seed 三条基线（course7，3.5 cm × 5 级，20 回合）：
  | 基线 | 成功率 | `mean|a|` | `|a| < 0.05` 占比 |
  | --- | --- | --- | --- |
  | 策略 | 75.0% | **0.0109** | **99.9%** |
  | 零动作（纯先验） | **75.0%** | 0 | — |
  | 随机 `U(-1,1)` | 25.0% | 0.4998 | — |
- **根因**：毕业条件只有「回合数够 + 成功率够」，**没有最少步数**。有开环先验时未训练策略第一轮就达标，`learn()` 立刻返回，每档只拿到 0~1 次梯度更新（PPO 更新粒度 = `n_steps 256 × n_envs 8 = 2048` 步；course7 各档实际 1824 / 2272 / 2752 / 3456 步 = **0 / 1 / 1 / 1** 次更新）。而 `log_std_init = -2.5` ⇒ 确定性均值 ≈ 0 ⇒ 策略输出 ≈ 0。
- **修法**：毕业条件加第三项 `min_timesteps`，由 `--min-stage-frac`（默认 0.5）算成 `预算 × F`。
- **判据**：第一条正面证据是 course8 的 easy5（4.6 cm × 7）：策略 45.0% vs 纯先验 20.0% vs 随机 5.0%，`mean|a| = 0.0893`。
- **注意**：本仓重构后**没有** `min_timesteps` 门（六级全跑满固定预算），教训保留。
- → 新项目：**任何早停/毕业条件都必须有最少步数下限**；`mean|a|`（残差幅度）是判断「策略是否真的在输出」的第一诊断量，训练循环应常规统计它。

### 4.2 VecNormalize 奖励归一化：温和稳定器，不是解药（§4.2）
- **判据**（同配置 3.0 cm 台阶、8 环境、150k 步的诚实 A/B）：
  | arm | 回合 | 平均回报 | 成功率 | `explained_variance` | 5 cm `mean\|a\|` |
  | --- | --- | --- | --- | --- | --- |
  | `norm_reward=False` | 822 | -567.8 | 0.474 | 0.2163 | 0.0698 |
  | `norm_reward=True` | 790 | -546.9 | 0.539 | -0.2363 | 0.0803 |
  ⇒ 3 cm 成功率 0.474 → 0.539，残差幅度 0.070 → 0.080。**两侧在 11 cm 上都没有成功。**
- **决策**：`norm_reward` 默认设 **True**（有帮助），但不要指望它解决高台阶。
- **撤回记录**：早期草稿引用过一组「0.025→0.033 / 0.88→0.12」，来自一个后来被发现**报错回合**的探针（见 F1），已撤回。
- → 新项目：归一化能稳 3 cm 级别，不能造能力；凡是从报错探针得到的数字必须整批作废。

### 4.3 课程切换不能在 rollout 中间做（§4.3）
- **根因**：换地形要重编译 MJCF（约 **250 ms/子环境**），在 rollout 中间换会让当次观测与状态不一致。
- **修法**：`Curriculum._on_step` 返回 `False` 只结束 `learn()`；真正切阶段由 `scripts/train.py` 在返回之后做，并且逐级 warm-start（`model.set_parameters(上一级权重)`）。
- **架构变更**：`Curriculum` 类与 `StageTrainer` 协议已在死代码清理中**删除** —— `scripts/train.py` 从来就是自带 `for` 循环直接调 `Trainer.train_stage`，那个驱动器是孤儿。现在的「不中途切地形」靠 `Trainer.train_stage` 每级只 `learn()` 一次固定预算保证（见 `docs/robotdog/curriculum.md` §3.1）。
- → 新项目：地形/场景**由向量环境的重建实现切换**，不要做「热切换」；每级 warm-start 传权重。

### 4.4 ⭐ 课程级数是能力的一部分，不是中性热身（§17）
- **症状**：`runs/course3`（1,350,000 步、51 分钟）逐级末 50 回合滚动成功率 `flat 100% / flat_2 100% / flat_3 100% / stair3 0%（摔 86%）/ stair6 8%（摔 92%）/ stair9 0%（摔 58%）`；而同机同评测器下旧课程 `course1`/`course2`（1 级平地 + 无指令楼梯）是 `3 cm 40% / 6 cm 0% / 9 cm 0%` ⇒ **3 cm 从 40% 掉到 0%**。
- **先排除三个假嫌疑人**：(1) 不是评估口径（训练日志里 stair3 自己也是 0%）；(2) 不是 `VecNormalize` 没装回（同源 8 环境 + `VecMonitor` + 装回 `stair3_final_vecnormalize.pkl` 重测仍有 11.1%）；(3) 不是地形取错 —— 但这里踩到**真陷阱**：`load_policy(model_path)` 不给 `stair=` 时建 `demo_stairs_spec()` 默认 **5 cm** 楼梯，而不是该 checkpoint 记录的 3 cm ⇒ 任何独立评估都必须显式 `env.set_stair(demo_stairs_spec(step_height=h/100, tread=0.5, n_up=8))`。
- **三臂消融**（把「平地级数」与「楼梯段是否带指令」解耦）：
  | 臂 | 平地 | 楼梯段指令 | 3 cm 成功率 | 3 cm 摔倒率 | 3 cm 前进 |
  |---|---|---|---|---|---|
  | A `runs/courseA_nocmd` | 1 级 100k | 无 | 0.0% | 30% | 0.098 m/s |
  | B `runs/courseB_staircmd` | 3 级 300k | vy±0.30 wz±0.35 | 0.0% | 0% | 0.175 m/s |
  | C `runs/courseC_nocmd` | 3 级 300k | 无 | 0.0% | 0% | 0.137 m/s |
  | 对照 course1/course2 | 1 级 100k | 无 | **40%** | 60% | **0.914 m/s** |
  B 与 C 只差楼梯段指令、差异为 0 ⇒ **回归由「三级平地预热」造成，与楼梯段指令无关**。形态差异是关键：A 是「摔 30% + 超时 70%」，B/C 是「100% 超时、几乎不摔、慢挪 0.14~0.18 m/s」，course1 是「0.914 m/s 冲上去、60% 摔但 40% 成功」。**平地预热学出的是更慢更稳的步态，它上不了台。**
- **修法**：新建 `.tmp/courses/course4.yaml`（**一级平地 + 3/6/9/12/15 cm**，平地 `vx=0.55`、`flat_success_distance=3.0`，楼梯段 `vx` 基线 0.55 加 vy/wz 指令，奖励覆盖 `foot_lift [0.9,1.0,1.2,1.4,1.6,1.8]`、`probe_lift [0,0.6,0.9,1.2,1.5,1.8]`）。
- **判据（诚实）**：course4（1,350,000 步、2,512 s）确定性评估 `0 cm 100% / 3 cm 10% / 6 cm 0% / 9 cm 0% / 12 cm 0% / 15 cm 0%`，**仍低于 course1 的 3 cm 40%**；把楼梯段指令置零后 3 cm 反而降到 0%（100% 超时）⇒ **指令跟踪与爬楼必须分开解决，先解决前者**。
- → 新项目：课程形状（级数、级距）是**超参的一部分**，改形状的 A/B 必须与控制变量（指令、观测维数、奖励）解耦；「平地热身」在小步态任务上会把策略锁进慢稳局部最优。

### 4.5 ⭐ 统计强度：单级跑间方差 36%~72%（§14）
- **症状**：`body_collision` 降到 −1.2 并加指令（观测 46 → 48 维）后，`runs/course1` 的 stair3 只拿到 **44.0%**，而上一轮 `runs/rewardfix`（`body_collision=-1.2`、无指令、46 维）在 3 cm 上是 **80.0%** ⇒ 第一反应是「加指令把 3 cm 弄丢了」。
- **测法**：同一份 `config.yaml`、只保留 3 cm 一级、250,000 步、8 环境，只换 `run.seed`：
  | 运行 | seed | 3 cm 成功率 | 摔倒率 | 均步数 | 回合数 |
  | --- | --- | --- | --- | --- | --- |
  | `runs/course1` 的 stair3 级 | 0 | 44.0% | 56.0% | 379.7 | 1,188 |
  | `runs/seed3cm_11` | 11 | **72.0%** | 28.0% | 376.3 | 1,096 |
  | `runs/seed3cm_12` | 12 | **36.0%** | 64.0% | 243.3 | 1,202 |
- **结论**：**36%~72% 的摆动全部来自 `run.seed`**。三个运行都跑满 250,000 步、约 1,100 回合 ⇒ 不是采样噪声（回合数的标准误只有几个百分点），而是「不同种子收敛到不同局部最优」。
- **判据/纪律**：本仓**单次训练成功率差异在 ±18 个百分点以内不做结论**；说「A 比 B 好」前必须**同一 seed 集合上各跑 ≥3 次**。历史笔记里的单次对照（§13.1 的 `rewardfix` vs `ab_base` 表、`train.md` §3.6 三臂表）**趋势可信、具体百分点不可信**。
- **另外两条**：视觉复核是不可省的裁判（GIF 姿态比单个成功率更能说明问题）；压低方差只有两条路 —— 加长每级预算（`curriculum.timesteps` 现实下限约 **250,000**）或增加环境多样性（`config.yaml` 里 `run.seed` 是唯一随机源）。
- **成本**：同一个 3 cm 单级只花约 **7.5 分钟** ⇒ 多 seed 复现在本仓是廉价的，没有理由跳过。
- → 新项目：从第一天起就把「多 seed 复现」写进流程和脚本；单级预算不低于数十万步；结论必须附 seed 集合。

### 4.6 指令课程设计：先量动作权限，再改奖励（§17-18）
- **症状（§17）**：指令课程训出的策略**基本不转向**。`runs/course2` 的 `flat_final.zip` 在平地上被要求 `wz = 0.60 rad/s` 转 3 秒（≈1.8 rad），实际只转了 **0.31 rad**，偏航通道平均绝对误差却高达 **0.735 rad/s**。
  - 实测（`runs/course3/checkpoints/flat_3_final.zip`，平地，40 回合）：横向 vy 与实测**相关系数 +0.661，但回归斜率只有 +0.028**（指令 ±0.36，实测只有 −0.03~−0.07 的基线漂移）；偏航 wz **相关系数 −0.085、斜率 −0.006**（完全没跟）；前进 vx 恒定指令 0.45，实测均值 0.285（偏低 37%），而 `teleop --hold 0.45` 下实测 **1.26 m/s（超调 2.8 倍）** ⇒ 训练时指令每回合随机抽，策略学到「不管指令、按自己节奏走」。
- **第二层根因（零策略基线，`.tmp/prior_cmd_response.py`）**：残差动作固定为 0，只让 trot 先验走，实测 `vx=+1.547 vy=+0.004 wz=-0.058`，**对任何 vy/wz 指令完全不变**；手工只拧髋侧摆（`a[0::3]=±1.0`）也只换来 `vy=-0.03~+0.04`、`wz=-0.19~+0.03` ⇒ trot 先验是**纯前进步态**，横移/偏航必须由策略主动推翻它，而奖励里跟指令有关的项只有 `lateral_vel=1.5` / `yaw_rate=1.0`（高斯），前进项是 2.5 ⇒ 「不跟指令、拿满前进分」是更稳的局部最优。
- **第一步：先排除动作权限，而不是先调权重**（`.tmp/action_authority.py`，平地、随机常量动作 200 组 × 120 步）：
  | action_scale | 最好 \|平均 vy\| | 最好 \|平均 wz\| | 指令量程 |
  |---|---|---|---|
  | 0.30 | **0.435 m/s** | **1.824 rad/s** | vy ±0.40 / wz ±0.70 |
  | 0.60 | 0.493 m/s | 3.036 rad/s | 同上 |
  | 0.80 | 0.665 m/s | 2.945 rad/s | 同上 |
  | 1.00 | 0.512 m/s | 2.558 rad/s | 同上 |
  解析动作对照：`hip 前+1 后−1`（对角髋反向）在 `scale=0.30` 下就给出 `vy=+0.333`、`wz=-0.808` ⇒ **`action_scale=0.30` 权限已超过指令量程，「学不会跟踪」不是权限不够**。这条否证把问题从「环境改不了」推回「奖励与课程设计」，也避免盲目放大 `action_scale`（会让先验零点偏移，先验是按 0.30 标定的）。
- **第二步：改奖励结构（`robotdog/reward.py`）**：(1) `lateral_vel` 1.5 → **4.0**、`yaw_rate` 1.0 → **3.0**；(2) `command_follow` 从 `max(横向, 偏航)` 改成**求和**；(3) 两项仍被 `if cmd.vy_max:` / `if cmd.yaw_max:` 门控，范围为零的方向不参与。
- **第三步：建「指令跟踪专项」课程 `.tmp/courses/cmdtrack.yaml`（900,000 步、4 级全平地）**：
  | 级 | 指令 | vx | 奖励覆盖 | 预算 |
  |---|---|---|---|---|
  | `flat` | vy±0.40 | 0.0 | `forward_vel=0` | 200k |
  | `flat_2` | wz±0.70 | 0.0 | `forward_vel=0` | 200k |
  | `flat_3` | vy±0.40 | 0.45 | 正常 | 250k |
  | `flat_4` | vy±0.40 wz±0.70 | 0.45 | 正常 | 250k |
  **关键设计**：前两级把 `forward_vel` 覆盖成 0 且 `vx=0` —— 原地学横移/扭转，此时「只顾往前走」拿不到前进分，策略只能靠 `command_follow` 得分；横移与偏航**分开教**，避免一上来就双通道；`action_scale` 保持 0.30 不动。
  **第 0 级为什么用 `flat_success_distance=1.0`**：原地阶段没有前进指令，若沿用 3.0 m 判据，`is_success` 永远不触发、进度行成功率恒为 0，看不出学习信号。
- → 新项目：**零策略基线（先验单独走）是判断「这个能力是策略学来的还是本来就有的」的最便宜实验**；指令跟踪类任务先量 action 权限（解析动作 + 随机常量动作扫描），确认权限足够后再动奖励；多通道指令**分通道、分阶段**教。

---

## 5. 终止判据（§1.4、§2.4、§12.4）

### 5.1 摔倒判据的阶跃参照面
见 **A4**（`terminate_height_ref = "support"`）。

### 5.2 终止 vs 逐步惩罚的取舍
见 **B9**（腿接触 `leg_collision = -0.25` 不终止）。

### 5.3 ⭐ 删掉「躯干接触致命性」：从未生效过，且随时可能误杀（§12.4）
- **缘起（用户问题）**：「躯干持续接触 10 个控制步（0.2 s）才算摔倒 —— 去抖，避免单帧擦碰就判死是否合理？机身上台架天然要蹭上去。」
- **测一（几何）**：躯干碰撞体 3 个，最低点在 `0.118 m`（机身下探 `0.2020 m`），而 `terminate_min_height` 原来是 `0.13` —— 比它还高 `0.012 m` ⇒ 躯干一碰地，**高度判据早就触发了**（差 0.012 m ≈ **6 个物理步**），这条接触规则在直立姿态下纯属冗余。
- **测二（驱动实验，5/8/11/15 cm × forward/forward_fast/tilt/random × seed 0/1/2，共 48 回合）**：躯干接触的**最长连续步数全场只有 8**，从未触发 10 步判据；躯干接触恰恰发生在「高度完全安全」时 —— 「首次安全躯干」与「首次躯干」逐条相同（130/130、51/51、39/39 …），状态是 `h ≈ 0.24~0.28 m`、倾角 0.05 rad 贴着立面蹭、腿也在蹭；15 cm 上连续接触最长 4~5 步（random 档 8 步），**距离阈值只剩 2 步**。
- **测三（反事实）**：把「安全」高度阈值从 0.13 放宽到 0.10 与 0.08，最长「安全躯干接触」连续步数**仍然是 8**（same case：15 cm / random / seed 3 / 第 59 步，`h = 0.277`、倾角 `0.067`）⇒ 抬高安全余量也救不了：**「躯干接触」这件事本身与「摔倒」无关**。
- **修法**：删 `EnvConfig.terminate_contact_steps` 与 `_effective_contact_steps()`、删 `_trunk_contact_streak`；`_contact_fatal()` 重写为**恒返回 `False`**，只维护 `_leg_contact_streak`（诊断用，永不致命）；保留方法名是为了让 `_terminated()` 调用形状不变。`info` 里 `terminate_contact_steps` / `trunk_contact_streak` 换成 `leg_contact_streak`（键数 **18 → 17**）。`terminate_min_height` **0.13 → 0.15**，换来 **3.2 cm** 余量。躯干接触代价继续由 `body_collision = -4.0/步`（逐步、有界）承担，**不由终止承担**。
- **判据（修复后实测，400 步上限）**：
  ```
      0cm 静止:  113步 累计    +95.9 每步 +0.849
      0cm 前进:  400步 累计   +272.9 每步 +0.682
     15cm 静止:  400步 累计   -573.9 每步 -1.435   ← 从 +0.226/步 翻转
     15cm 前进:  400步 累计   -178.6 每步 -0.446
  ```
  15 cm 站桩 **+0.226/步 → −1.435/步**（进步 −1.66/步），前进 −0.446/步 ⇒ **站桩不再优于前进**。诚实记录：`0 cm 静止` 仍 +0.849/步，是 trot 先验在走路推 3 m 并触发 `terminal=+50`，不是站桩；`body_collision` 在 15 cm 上仍达 **−1.37/步**（躯干蹭立面不可避免），高台阶可能仍是局部最优（下一轮 §13 把它从 −4.0 降到 −1.2）。
- **教训（原文）**：阈值型终止判据要先量「真实分布的最长连续步数」再定阈值（8 与 10 只差 2 步，规则从未生效，既不提供保护又随时可能误杀）；**用户的直觉值得直接测量** —— 「机身上台架天然要蹭上去」一句话就指向了正确的诊断方向：先量接触时的姿态，再决定规则是否合理。
- → 新项目：**阈值型终止规则上线前必须先测真实分布**（最长连续步数），若与阈值同量级或从未触发，直接删；终止只留不可恢复状态；碰撞类一律逐步有界惩罚。

### 5.4 `body_collision = -4.0` 让「蹭立面」比站桩还亏（§13.1-13.2）
- **症状**：GIF 里「试两下就放弃」。改完 §12 后重训全课程（`runs/rewardfix`，1,350,000 步）：
  | 阶段 | 改后 `rewardfix` | 改前 `ab_base` |
  |---|---|---|
  | flat | 100% / 摔 0% | 100% / 0% |
  | stair3 | **80% / 摔 20%** | 62% / 38% |
  | stair6 | **10% / 摔 90%** | 0% / 90% |
  | stair9 | 0% / 摔 86% / 超时 14% | 0% / 摔 44% / 超时 56% |
  | stair12 | 0% / 摔 38% / 超时 62% | 0% / 摔 16% / 超时 84% |
  | stair15 | 0% / **摔 100%** | 0% / **超时 100%** |
  站桩被杀掉了，但只有 3 cm 真正变好。逐档 GIF（`output/gifs/rewardfix_{3,6,9,12,15}cm.gif`）：3 cm 真在爬（前腿上一级、后腿下一级、重心前压、四腿交替踩踏面）；6 cm 跨上第一级（前进 +1.86 m）后躯干越贴越低，进入「趴着用肚子蹭」然后翻倒；12 cm 前腿搭立面、后腿撑地，**速度只有 0.03 m/s**，三个回俩超时；15 cm 三回合形态一致（前进 +0.70/+0.77/+0.80 m → 前腿搭上、后腿蹬 → 上不去 → 侧前方翻倒）。
- **根因**：`body_collision` 是逐步量，权重 −4.0，15 cm 上一项就吃掉 **−548**（≈ −1.37/步），占全部负值一半以上。策略学到的最优解是**试两下就放弃**。这与 §12「站桩被奖励」是同一个病的两面：**只要某个每步都付的项与「继续尝试」负相关，策略就会停止尝试**。
- **修法**：`-4.0 → -1.2`。躯干接触本来就不再终止回合（E3），所以它只是「不舒服」的提醒，不该重到压过前进动力。
- **互不影响的证明（用户明确要求）**：(1) **动力学完全没动** —— 固定动作序列在 12 个用例（平地/5/8/11 cm × seed 0/1/7）上**步数逐位一致**（127/138/122/125/140/210/180/300/300/124/128/300）；(2) **奖励差完全由 `body_collision` 解释** —— `expected(-4.0) = sum(r) + n_contact × 2.8`，5 cm `125 步 body=-1.20 sum=+99.60 expected=+96.80`、8 cm `180 步 body=-14.40 sum=+95.55 expected=+61.95`、11 cm `124 步 body=-3.60 sum=-7.26 expected=-15.66`。
- → 新项目：改单个奖励权重时必须给出「动力学不变」的证据（固定动作序列逐步比对）；每一个「每步都付」的惩罚都要问「它是否惩罚了继续尝试」。

### 5.5 奖励账逐项打印（§12.6）
- **教训**：`forward_vel` 在 `vx = 0` 处 +0.86/步这个数，**只有真的把 `reward_terms` 打出来才看得见**。
- → 新项目：`reward_terms` 必须是可机读的结构化输出（每回合按项聚合 + 每步 trace），并把「零动作基线」设为常备诊断项。

---

## 6. 评测与判据（§5、§9.3、§10、§14、§16）

### 6.1 ⭐ 探针报错回合：`probe()` 与独立评测不一致（§5.1）
- **症状**：同一模型、同一 seed，`probe()` 报 `succ=0.00 adv=0.23`，独立脚本报 `succ=True adv=4.636`。
- **排查**：配置逐字段 diff 为 0、归一化打印正确、`inspect.getsourcefile` 指向正确文件、同进程内两条路径结果不同 —— 根因**始终没有定位**。
- **修法（比找根因更重要）**：把探针的内层循环替换成**唯一的共享 rollout 函数**，`probe()` 只做聚合。设计原则：**探针与所有独立评测必须调用同一个 rollout**，这样 harness 的 bug 就不可能让两边不一致。
- **代价**：这次事故让此前一大批 `probe()` 数字全部作废。
- → 新项目：**评测只有一条代码路径**（train/eval/probe/GIF 全走同一个 `rollout()`）；任何第二份手搓评测循环都是作废数字的来源。

### 6.2 ⭐ 同一个权重两套结论：`VecNormalize` 统计量没装回模型（§5.9）
- **症状**：对同一个 `runs/ab_hist1/model_final.zip`，两条评测路径在 3 cm 上给出**相反**答案：
  | 路径 | 3 cm 结果 |
  |---|---|
  | `scripts/play.py --mode eval`（自己驱动裸 `Go2StairsEnv` + `ObsNorm`） | 成功率 100%、摔倒 0%、前进 4.54 m |
  | 自写脚本走 `model.get_env()` → `VecEnv.step()` | `succ=0.00 fall=1.00 adv=3.17m steps=347.8` |
- **真因（`robotdog/training.py: load_policy()`）**：它读 `*_vecnormalize.pkl` 只为了**返回**一个 `ObsNorm` 给调用方，却**没有把统计量装回模型自带的那个 `VecNormalize`**。`model.set_parameters()` 只还原网络权重，`build_ppo` 建出来的 `VecNormalize` 仍是初始的 `mean=0 / var=1`。于是：裸 env 路径手动 `ObsNorm` 归一化 ⇒ 策略看到真统计；`model.get_env()` 路径包装器还是 `mean=0/var=1` ⇒ **策略看到未归一化观测**。
- **定位手法（可复用）**：并排打印两条路径**重置后的前 6 维原始观测与归一化观测**，分叉一眼可见：
  ```
  raw obs0 A  [-0.0006  0.0068 -0.0087 ...]   ← 空
  raw obs0 B  [-0.058   0.9376 -1.7505 ...]   ← 真实关节角
  norm obs0 A [-0.0006  0.0068 -0.0087 ...]   ← 归一化等于没做（mean=0 / var=1）
  norm obs0 B [ 0.2344 -3.8073  0.4135 ...]   ← 真统计
  ```
- **修法**：`load_policy` 在 `set_parameters` 之后调一次 `restore_vecnormalize(venv, cand)`（`robotdog/training.py` 里这个函数早就写好、也在 `__all__` 里，**但此前零调用者**）。修完两条路径一致。
- **顺带第二个坑**：`.tmp/path_cmp.py` 里 `make_env_fn(stair, None, seed=seed)` 传了 `cfg=None` ⇒ 建出默认 `history_len=10`（460 维）的环境，而权重是 `history_len=1`（46 维），报 `ValueError: operands could not be broadcast together with shapes (1,460) (46,)`。**跑权重必须用它自带的 `EnvConfig`**（`env_config_for_model(model_path)`）；另外 `make_env_fn` 返回的是**零参工厂**，必须再调一次 `make_env_fn(...)()`。
- **教训**：评测器与训练器必须共用同一个「建环境 + 归一化」实现；任何**手搓**评测路径都要先和仓库自带脚本对一次数，否则 `|a|`、成功率、摔倒率全都不可信。
- → 新项目：checkpoint 必须自带 `env_config`（含维数、history_len、归一化统计）并在加载时**自动装回包装器**；`restore_*` 这类函数若无调用者，视为未完成而非「备用」。

### 6.3 `pickle.loads(vecnormalize.pkl)` 的三个坑（§5.4）
- **症状/根因**：(1) 得到的是 **`VecNormalize` 对象**不是 dict ⇒ `data["obs_rms"]` 报 `TypeError: 'VecNormalize' object is not subscriptable`；(2) 该对象**多访问一个不存在的属性就 `RecursionError`**（`BaseVecEnv.__getattr__` → `getattr_depth_check` → `_get_all_attributes` 递归）；(3) 它**不可调用** ⇒ `model.predict(vec_normalize_obj, ...)` 报 `TypeError: 'VecNormalize' object is not callable`。
- **修法**：立刻把 `vec.obs_rms.mean/var`、`vec.epsilon`、`vec.clip_obs` 拷进一个普通命名空间对象，然后 `del vec`，此后只用那份拷贝（`robotdog/policy.py` 的 `ObsNorm` 就是这么做的）。
- → 新项目：**不要长期持有 SB3 包装器对象**；加载后立即抽成纯数据结构（dataclass/namedtuple）。

### 6.4 估值脚本单位 bug：cm 当 m（§5.2）
- **症状**：所有高度的前进距离都「卡在 0.5 m」。
- **根因**：CLI 传 cm（5/8/11），脚本里直接 `demo_stairs_spec(step_height=height)` ⇒ 实际在测 **500 / 800 / 1100 cm** 的台阶（顶端 40~88 m）。
- **修法**：`step_height=height / 100.0`。**单位换算只做一次，并且写在边界上。**
- → 新项目：内部统一 SI（m）；所有 CLI/配置边界处一次性换算并注释单位。

### 6.5 打开观测归一化却没跟上权重 ⇒ 策略退化成随机（§5.3）
- **症状**：5 cm 上成功率从 100% 掉到 **0%**。
- **根因**：策略是在**归一化观测**上训练的，直接喂原始观测等于喂垃圾。
- **修法**：`*_vecnormalize.pkl` 必须与 `*.zip` **成对使用**；`play.py` 自动找同名 pkl，找不到会打印警告。
- → 新项目：权重与归一化统计**绑定为一个 artifact**（同一 checksum / 同一命名前缀），缺失时**报错而非静默降级**。

### 6.6 10 回合的评估能给出 ±15 pt 的错觉（§5.7）
- **修法**：评估至少 **40 个回合**；`play.py --mode eval` 默认 **100**。
- → 新项目：eval 默认回合数 ≥ 40，并在输出里附标准误。

### 6.7 行号漂移：`check_lines.py` 的两个盲点（§9.3、§10）
- **症状一（§9.3）**：删掉 `mean_abs_action` 后 `robotdog/training.py` 从 350 行变成 **323 行**，文档里 6 处指向该文件第 39 行（`RUNS_DIR` 与 `StairSpec` 的 import）与一处 `:295-297`（`RunDir.summary`）的引用全部指向空行或错误位置。`.tmp/check_lines.py` 抓到**全部 7 处**，但抓到的是**症状**（「这一行是空行」），不是病因。
- **症状二（§10.1）**：给 `robotdog/env.py` 的 `set_stair()` 补 5 行 docstring，文件从 696 → **701 行**，后面所有定义整体下移 5 行：`_compute_reward` 344→**349**、`_target_from_action` 539→**544**、`_gait_action` 529→**534**、`_contact_flags` 379→**384**、`_contact_fatal` 418→**423**、`_fall_reference_height` 455→**460**、`base_ang_vel_body` 274→**279**、`step` 559→**564**。`check_lines.py` 只报两条（它检查的是**范围的两个端点**，而旧写法那两个范围的末尾恰好还落在下一个函数的 docstring 里）；其余十几处引用（`README.md`、`docs/robotdog/reward.md`、`docs/robotdog/gait.md` …）**行号完全合法、指向非空行**，只是指向了**别的函数**（旧数字 344 现在落到 `_observation` 的 `def` 行上）。
- **病因**：> **行号是「相对坐标」，只要上游插/删一行，下游全部失效。** `check_lines.py` 能抓的是「越界」和「空行」——那是**症状**；「漂移」没有任何症状，因为漂移后的行号依然指向一个合法、非空的 Python 行。
- **修法（两个层次）**：(1) **引用写法**：文档引用代码改成 `` `robotdog/env.py` 的 `compute_reward()` `` 这种**文件 + 符号名**形式，行号只在确实要特指某一行实现时才写（符号名不会漂移）；(2) **新增检查器 `.tmp/ref_truth.py`**：把全仓 `.md` 里每条 `文件.py:行号` 引用解析出来，打印它**实际指向的那一行内容**。判据三条 —— 越界 → `OOR`；空行 → `BAD`；**指向 `def`/`class` 行但未预先登记 → `BAD`**（裸数字指到定义行几乎总是漂移后随手写下的数字）。同时维护 `EXPECT` 白名单（「第 N 行应该含 `def xxx`」）。本轮跑完：`refs checked: 210   suspicious: 0`。
- **纪律**：**每次删函数、加函数，都必须重跑 `check_lines.py`。** 它报的每一条「是空行」都在说「上游源码行数变了」；它**永远**不会报「这个函数不存在」——`ensure_dir` 被删后文档里 20 多处提及它依然「行号合法」（文档写的是散文，没有行号）。
- → 新项目：文档引用**一律用符号名**；行号检查器要报「这一行是什么」而不只是「是否合法」；新增检查器时先问「它**不能**抓什么」。

### 6.8 偏航指令读错量：奖励信号本身在抖（§16）
- **症状**：指令课程训出的策略**基本不转向** —— `runs/course2` 的 `flat_final.zip` 在平地上被要求 `wz = 0.60 rad/s` 转 3 秒（≈1.8 rad），实际只转了 **0.31 rad**，而偏航通道的「平均绝对误差」却高达 **0.735 rad/s**。
- **诊断**（`.tmp/check_yaw_rate.py` / `check_yaw_rate2.py` / `check_yaw_rate3.py`，平地、默认先验、零动作、seed 0、`control_dt = 0.02`，三种独立方法求逐帧真实偏航角速度）：
  | 方法 | 与 `env.yaw_rate` 的偏差均值 | 与 `qvel[5]` 的偏差均值 |
  |---|---|---|
  | 逐帧有限差分 | 0.165 | 0.083 |
  | ±2 步中心差分 | 0.256 | 0.157 |
  | 四元数相对旋转反解（精确） | **0.283** | 0.156 |
  真值 −0.0654、`env.yaw_rate` −0.0289、`qvel[5]` −0.0473 rad/s ⇒ **偏差幅度 0.28 rad/s 与要跟踪的指令量程（±0.70）同一量级**；偏差随 |roll|/|pitch| 单调增大（|倾角| ≥ 0.15 的帧上 0.615 vs qvel5 的 0.114）。
- **根因（不是 bug，是「读错了量」）**：`env.yaw_rate` 实现的是**世界系 z 轴角速度**（`wz + tan(pitch)·(cos(yaw)·wx + sin(yaw)·wy)`），却被拿去和逐帧的 `yaw` 欧拉角比较；两者时间导数只在姿态为零时相等。四足步态下躯干以几十赫兹小幅摆动，roll/pitch 的乘积项就能贡献 **0.3 rad/s**，正好把「平稳的偏航指令」淹没。
- **为什么致命**：策略无法控制那个振荡分量 ⇒ `command_follow` 偏航通道有一大半方差是它拿不到的；相对地「干脆不转、把前进项拿满」反而是更稳的局部最优（实测两回合前进速度 1.30 / 1.34 m/s，指令 0.55，**超调 2.4 倍**）。
- **建议修法（尚未实施，需用户确认，因为会改变奖励语义）**：偏航读数换成**机体系 z 角速度** `wz_body = float((quat_to_mat(self.base_quat).T @ self.data.qvel[3:6])[2])` —— 振荡更小、计算更便宜，且**更贴近真机**（机载陀螺直接给机体系角速度）。
- **教训**：「物理上正确的公式」不等于「适合当奖励信号的公式」；**奖励项上线前要量它的噪声底**（噪声底 0.28 vs 信号量程 0.70 同阶 ⇒ 等价于把一半权重扔进策略控制不到的通道）。这与 B10 的「可达性标定」是同一类错误的两面：一个是奖励**拿不到**，一个是奖励**读不准**。
- → 新项目：任何进入奖励的**导数类量**都要先与「精确反解真值」对比噪声底；四足躯干的高频摆动会把欧拉角导数污染到与指令同阶 —— 优先用机体系角速度。

### 6.9 `yaw_rate` 不是 `qvel[5]`（§13.4）
- **根因**：`data.qvel[3:6]` 是**机体系**角速度，不是世界系欧拉角速率；`base_rpy` 是 `mj_objectState` 给出的世界系 rpy。
- **检验**：5000 组随机姿态 + 从旋转矩阵 `dR = [ω]× R` 反解真实偏航角速度，三条候选公式**全部失配**：`-sin(p)·wx + cos(p)cos(r)·wy + cos(p)sin(r)·wz`（500/500 失配）、`wz + (sin p / cos p)·wy`（500/500 失配）、`sin(r)/cos(p)·wy + cos(r)/cos(p)·wz`（只在个别点碰对）。
- **正确写法（5000/5000 零失配）**：
  ```python
  wx, wy, wz = quat_to_mat(self.base_quat).T @ self.data.qvel[3:6]   # 先转世界系
  roll, pitch, yaw = base_rpy
  yawdot = wz + tan(pitch) * (cos(yaw) * wx + sin(yaw) * wy)
  ```
- → 新项目：坐标系约定要**在命名里写死**（`yaw_rate_world` vs `yaw_rate_body`）；凡涉及四元数/qvel 的公式都用**随机姿态批量数值验证**（对照旋转矩阵反解），不要靠推导。

### 6.10 独立评估必须显式设地形（§17）
- **症状**：`load_policy(model_path)` 不给 `stair=` 时建 `demo_stairs_spec()` 的默认 **5 cm** 楼梯，而不是该 checkpoint 记录的 3 cm ⇒ 评估在测一个它没学过的高度。
- **修法**：任何独立评估都必须显式 `env.set_stair(demo_stairs_spec(step_height=h/100, tread=0.5, n_up=8))`（`.tmp/eval_run.py` 已默认这样做）。
- **相关决策（§7.4）**：训练时每个阶段写一份 `<模型>_env.json`（`policy.save_env_json`），`report.step_height` 记下该权重的地形；`play.py` 用 `step_height_from_model()` 读它并打印「地形来自权重自带记录」，找不到时（老权重）才退回默认值（写进 `README.md` §5 已知限制）。
- → 新项目：**地形是 checkpoint 的一部分**，随权重一起序列化；评估端默认从权重读，CLI 显式值才覆盖。

### 6.11 旧结论要用同一把尺子重测（§17）
- **判据**：「course1 3 cm 44%」本来是**训练期滚动统计**，换成**确定性评估**后是 40%，两者接近才让回归变成可比的结论。
- → 新项目：训练日志里的滚动成功率 ≠ 确定性评估成功率；跨结论比较必须换到同一评测器上重测。

---

## 7. 配置化（§7）

### 7.1 ⭐ 重建模型必须用训练时那一套网络结构（§7.1）
- **症状**：`load_policy()` 按默认 `DEFAULT_NET_ARCH = (512, 256)` 建网再 `set_parameters(checkpoint)`，只要训练时不是默认结构（如 `ppo.net_arch: [64, 64]`）就必炸：
  ```
  RuntimeError: Error(s) in loading state_dict for ActorCriticPolicy:
      size mismatch for mlp_extractor.policy_net.0.weight: copying a param with shape
      torch.Size([64, 460]) from checkpoint, the shape in current model is
      torch.Size([512, 460]).
      size mismatch for action_net.weight: copying a param with shape torch.Size([12, 64])
      from checkpoint, the shape in current model is torch.Size([12, 256]).
  ```
- **根因**：「配置化」放大出来的问题 —— 结构以前写死在两个地方，恰好一致；现在配置能改它，两处就必须真的对齐。
- **修法**：新增 `robotdog/policy.py:policy_arch(model_path)`，从 SB3 `.zip` 的 `data` 条目里读 `policy_kwargs.net_arch`（**是普通 JSON** `{"pi": [...], "vf": [...]}`，不需要解 `":serialized:"` 的 base64 pickle），`load_policy` 用它建网：`arch = policy_arch(model_path) or DEFAULT_NET_ARCH`。
- → 新项目：**网络结构是 checkpoint 的一部分**（同 F10 的地形原则）；「默认值恰好一致」是隐性耦合，配置化会把它炸出来。

### 7.2 `RunDir.__init__` 里 eager `mkdir` 让「目录已存在」的保护永远为真（§7.2）
- **症状**：`train.py` 想拦住「重复用同一个 `--run-name` 覆盖上次结果」：
  ```python
  rundir = RunDir(name)
  if rundir.root.exists():
      raise SystemExit(f"输出目录已存在：{rundir.root}")
  ```
  但 `RunDir.__init__` 自己就把 `root/checkpoints/tensorboard/artifacts` 建好了 ⇒ **第一次运行就报「输出目录已存在」**，且退出时留下四个空目录（第二次运行照样报）。
- **修法**：`__init__` 只算路径不建目录，新增 `ensure()` 供调用方在检查之后显式创建；`save_model` / `write_summary` / `CheckpointCallback` 自己 `mkdir(parents=True)`。
- → 新项目：**构造器不做副作用 IO**（建目录/写文件）；「检查 → 创建」必须由调用方按序执行。

### 7.3 推理时只建 1 个环境 ⇒ SB3 mini-batch 警告（§7.3）
- **症状**：`batch_size=512` 与 `n_steps × n_envs` 不整除时 SB3 在 `PPO.__init__` 里 warn：
  ```
  UserWarning: You have specified a mini-batch size of 512, but because the
  RolloutBuffer is of size `n_steps * n_envs = 256`, after every 0 untruncated
  mini-batches, there will be a truncated mini-batch of size 256
  ```
- **根因**：训练时 8 环境（2048，整除）；**推理时 `load_policy` 只建 1 个环境**（256，不整除）。
- **修法**：对只做 `predict` 的场景这条警告没意义，在 `build_ppo` 里用 `warnings.catch_warnings()` + `filterwarnings("ignore", message=".*mini-batch size.*")` 压掉。
- → 新项目：训练与推理的环境数不同会触发 SB3 这类形状警告；按「是否真的会训练」分流抑制，不要全局 ignore。

### 7.4 地形默认值必须每个 checkpoint 各记一份（§7.4）
见 **F10**。

### 7.5 `env_config_from_summary` 必须重建嵌套 dataclass（§13.5）
- **症状**：`_env.json` / `summary.json` 往返之后 `env_config["command"]` 变成普通 `dict`，直接塞给 `EnvConfig` 会让 `env._command` 变成 dict，任何 `set_command_now` 调用当场炸：`AttributeError: 'dict' object has no attribute '_now'`。
- **修法**：`robotdog/policy.py` 里 `if isinstance(env_kw.get("command"), dict): env_kw["command"] = _rebuild(CommandConfig, env_kw["command"])`。**任何嵌套 dataclass 字段过一遍 JSON 都需要这一步。**
- → 新项目：配置（反）序列化用**类型驱动的递归重建**，或直接用 pydantic/`dacite` 之类的库，不要手写 `dict → dataclass`。

---

## 8. 文档/代码腐坏与重构纪律（§6、§8、§9）

### 8.1 三个必须记住的重构原则（§6.3）
1. **能删就删**。用户明确允许破坏现有功能时，「保留实验性旋钮」的代价远大于收益：这些旋钮默认关闭、实测**没有收益**（新奖励项 `foot_lift_clear` 在默认学习率下把已经学会的 8 cm 从 **100% 掉到 25%**），却要每个读者都去理解它们。
2. **删之前先备份**。删掉的代码留在 `.tmp/knob_backup/`（含一份 `git diff` 补丁）。
3. **删之后要证明「行为不变」**：用 `importlib` 以**私有模块名**加载改前的文件，在同一串固定动作下逐项比较 reward / obs / 终止标志 / `reward_terms`。实测 **12 组全部 `max|Δreward| = 0.000e+00`、`max|Δobs| = 0.000e+00`**。
- → 新项目：把「删除 = 备份 + 等价性证明」做成标准流程；用 `importlib` 私有模块名同时加载新旧两版做对照。

### 8.2 这次重构删掉了什么（§6.1）
- **触觉 / 测距 / 接触力整套逻辑**（原 `robotdog/tools/touch.py` 与 `env.py` 里 **91 处**引用）：目标是**盲走楼梯**，机器人只靠本体感觉 + 一个台阶高度标量。
- **全部侧向奖励**（见 B6）。
- **域随机化**、**触觉相关观测块**（`touch`、`stair_probe`、`drift`、`foot_height`）、**逐级晋级/回退的自动课程**（改成固定六级）、`render.py` 与 `evaluate.py`（GIF 内联进 `scripts/play.py`，评估就是 `play.py --mode eval`）。
- → 新项目：确认最终任务形态（如「盲走」）后，把与前提冲突的整条能力线**彻底删除**，不要留成「默认关闭的旋钮」。

### 8.3 ⭐ 死代码清理：三类假死（§9.1）
判据是**「这个符号有没有任何读者」**，用 `.tmp/dead3.py` 按 token 流做全仓引用计数（排除定义行）。它一次报出 **13 个「零引用」候选，真正该删的只有 6 个** —— 剩下 7 个每次落进三类**假死**：
| 类型 | 例子 | 为什么不能删 |
| --- | --- | --- |
| **字符串派发** | *（本仓目前没有实例）* | `venv.env_method("某个方法", ...)` 把名字藏在字符串里，AST 看不见。最容易误删，改 `DummyVecEnv` 相关代码时留意 |
| **框架回调** | `ProgressCallback._on_step`（`scripts/train.py:84`）、`CheckpointCallback._on_step`（`scripts/train.py:107`） | SB3 的 `BaseCallback` 按固定名字回调，从不被本仓代码显式调用 |
| **文档化的公开 API** | `GaitParams.static`（`robotdog/gait.py:170`）、`GaitParams.targets`（`robotdog/gait.py:90`）、`policy.Policy`（`robotdog/policy.py:21`）、`StairSpec.down_end_x`（`robotdog/tools/terrain.py:108`）、`Go2StairsEnv.yaw`（`robotdog/env.py:448`）、`Go2StairsEnv.set_stair`（`robotdog/env.py:291`） | 设计文档把它们的签名与理由逐条写下来了，是给人用的接口面。删掉等于同时删两处文档，而它们各只有 1–5 行 |
**留下的判据**：`GaitParams.static()` 是文档里「怎么关掉先验」的**唯一正确写法**（`docs/robotdog/gait.md:434` 的示例直接 `off = GaitParams.static()`），所以哪怕零调用者也要留 —— 删了文档就在教一个不存在的函数。
**`set_stair()` 的处境**：它**曾经**有一处真实调用者 —— `robotdog/training.py` 的 `set_terrain(venv, stair)`（函数体就是 `venv.env_method("set_stair", stair)` + `venv.reset()`）。而 `set_terrain` 自己是零读者，于是被删；删掉它之后 `set_stair` 才变成零调用点。**当前课程根本不用换地形**（`scripts/train.py` 每级都调 `trainer.make_env(stage)` 新建整个向量环境）。真相是「**一个零读者的包装器是另一个符号的唯一调用者**」⇒ 删之前必须**看两跳**。
> **修正一条曾经的错误结论**：早期 `experience.md` 与 `README.md` 都把 `set_stair` 归类为「字符串派发，调用点在 `robotdog/training.py:99` 的 `venv.env_method("set_stair", stair)`」。**那是错的** —— 全仓 grep `env_method` 与 `set_stair` 只剩定义行与注释，`training.py:99` 是 `build_ppo` 的 `seed: int = 0,` 形参。两个符号当时**都已经删了**。教训：**「字符串派发」这个类别要靠 grep 证明，不能靠印象归类。**
- → 新项目：死代码扫描必须**人工核对**，并显式排除框架回调 / 字符串派发 / 文档化 API；删除前 grep 两跳；「零调用者」不等于「该删」。

### 8.4 真正删掉的 6 个（§9.2）
| 符号 | 位置 | 为什么它真的死了 |
| --- | --- | --- |
| `load_ppo(path, venv, *, device)` | `robotdog/training.py` | `load_policy` 是唯一实际入口；它是 `PPO.load()` 的薄包装，无调用者 |
| `mean_abs_action(actions)` | `robotdog/training.py` | 三行诊断量（`float(np.mean(np.abs(a)))`），训练循环从不统计，只有临时脚本会算 |
| `set_terrain(venv, stair)` | `robotdog/training.py` | 两行包装，无调用者。删前看过两跳（见 H3） |
| `GO2_SCENE_XML` | `robotdog/tools/paths.py:15` | 本仓从不用官方 `scene.xml`（场景由 `compile_scene()` 现场拼 XML）；它只在 `scripts/download_go2.py:43` 的 `WANT_FILES` 里以**字符串字面量**出现 |
| `ensure_dir(path)` | `robotdog/tools/paths.py:29` | 零调用者，且自带两个坑（传 `str` 抛 `AttributeError`、对文件路径会建出同名目录） |
| `ensure_output_dirs()` | `robotdog/tools/paths.py:35` | 零调用者；`scripts/play.py:248` 的 `write_gif` 自己 `path.parent.mkdir(...)` |
顺带删掉三个**失去全部使用者**的 import：`robotdog/training.py` 的 `numpy`（`mean_abs_action` 是唯一使用者）、`robotdog/policy.py` 的 `Sequence`、`scripts/download_go2.py` 的 `sys`（末尾写的是 `raise SystemExit(main())` 而不是 `sys.exit`）。
- → 新项目：删除清单要留档（符号 / 位置 / 死因），并同步删减 import；`raise SystemExit(main())` 比 `sys.exit` 好，能让 `sys` import 自动消失。

### 8.5 LF/CRLF 在两种工具之间来回翻转（§9.4）
- **症状**：本仓约定 **CRLF**（`core.autocrlf=true`），但用 `write` 工具整篇重写文件时输出是 **LF**：(1) `git status` 警告 `LF will be replaced by CRLF the next time Git touches it`；(2) 更隐蔽的是**行尾不一致本身** —— 同一文件既有 LF 又有 CRLF 时，某些编辑器显示成「整篇都改了」。
- **实测**：**11 个文件**处于裸 LF 状态（`robotdog/env.py`、`robotdog/reward.py`、`scripts/train.py`，以及 8 份文档）。
- **修法**：`.tmp/fix_eol.py` —— 把 `\r\n` 先归一成 `\n` 再统一展开成 `\r\n`，**只处理含裸 LF 的文本文件**。跑完复检 `still bare-LF: none`，且 `git diff --shortstat HEAD` 没有因行尾产生额外行数变化。
- → 新项目：**整篇 `write` 重写文件之后，要么用 `edit` 做增量修改（保留原行尾），要么立刻跑一次行尾归一化**；CI 加一条裸 LF 检查。

### 8.6 已知但**故意不改**的注释（§8）
- `robotdog/go2_model.py` 的注释说 `HOME_JOINT_ANGLES` 来自 `go2.xml` 的 `<keyframe name="home">`。vendored XML 实际是 `<keyframe><key name="home" qpos="..."/></keyframe>`（`<keyframe>` 没有 name 属性），且 `extract_meta()` **从不读 keyframe** —— `home_qpos` 纯靠硬编码常量，**无运行期校验**。**改成读 keyframe 会改变 `home_qpos` 取值 ⇒ 改变 reset 与所有已训权重的前提**，所以只记录不动。
- `extract_meta()` 的 `floor_geom_id` 走 `mj_name2id` 且**不检查 `< 0`**（其余 id 都走 `_resolve_id` 抛 `KeyError`）；`_OPTION` 正则只认自闭合 `<option .../>`，改成非自闭合会**静默失效**。
- → 新项目：凡是「改了它就会让所有既有权重失效」的常量，**记录下来并加注释说明为什么不改**，比偷偷修好更有价值。

### 8.7 派 subagent 写核心模块的教训（§6.2）
- **症状**：派后台 subagent 从零重写 `robotdog/training.py`，交出的 **784 行**版本**运行期直接炸**：
  - `make_env_fn` 里写 `replace(cfg, stair=stair)` —— `EnvConfig` 没有 `stair` 字段；
  - `Monitor(Go2StairsEnv(env_cfg, seed=seed))` —— 位置参数错（第一个位置参数是 `stair`）；
  - `@dataclass class RunDir` 让 `RunDir('smoke')` 报 `TypeError: RunDir.__init__() missing 4 required positional arguments`；
  - 还有一大批没人调用的死代码。
- **教训**：派 subagent 写核心模块时，**不能只看它自报的验证输出**，必须**逐条运行你自己脚本里的调用路径**。它自己的四条验证里有一条根本没过。最后 `training.py` 由作者重写。
- **并发写作教训**：subagent 在作者编辑同一批文件时报告「有别的 writer 在抢」——那是**作者本人**。多 writer 场景下要么串行、要么明确停掉一个，不要让它自己去「修复」。
- → 新项目：**subagent 的产物必须由主控跑真实调用路径验收**；同一批文件的写者必须唯一（串行化，或显式声明 ownership）。

---

## 9. 工具链与沙箱（§4.1、§5.5-5.6、§5.8）

### 9.1 并行 run 数上限 ≈ 3（§5.5）
- **症状**：6 个 run（各 8 环境）并行时集体瞬间退出（exit code 1，**无 traceback**），前台重跑同一个命令却完全正常。
- **根因**：内存耗尽。traceback 是 `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.29 MiB for an array with shape (512, 660) and data type float32`，发生在 `stable_baselines3/common/buffers.py:514`。
- **修法**：并行 run 压到 **≤ 3**；并且**用 `*> file` 而不是 `| Tee-Object`** 启动后台 run。
- → 新项目：并行度按**内存**算（`n_envs × buffer 形状`），不按 CPU 核数；上限写进脚本默认值。

### 9.2 后台 run 的 stderr 会丢失（§5.6）
- `Tee-Object` 只捕获 stdout；且 stderr 的行号会错乱指向**正在被编辑**的文件，极具误导性。一律 `*> .tmp/<name>.log`。
- → 新项目：后台任务日志**一律合并 stdout+stderr 落文件**（`*>`），并记录启动时间与 commit。

### 9.3 PyTorch/SB3 杂坑（§5.8）
- `model.policy(tensor)` 的返回值**不是二元组**（`ValueError: too many values to unpack`），要取动作就用 `model.predict(x, deterministic=True)`。
- `EnvConfig` 里**没有** `stair` 字段，`dataclasses.replace(cfg, stair=...)` 必炸。
- 编辑代码块时 `edit` 的 `old_string` 只覆盖一部分 ⇒ 收尾的 `"""` 残留成语法错误或脏文本。**改完必须 `py_compile` 并读回确认。**
- → 新项目：任何编辑后强制 `py_compile` + 读回；`dataclasses.replace` 用错字段要能立刻报错（`__post_init__` 校验）。

### 9.4 Windows/PowerShell 环境坑（§5.8）
- `python -m venv` 可能因 `ensurepip` 失败而建不出 pip；用 `--without-pip` + `get-pip.py`。
- 受限环境里 curl / `Invoke-WebRequest` 的 HTTPS 可能被 TLS 挡住，但 Python 的 `urllib` 正常 —— **下载用 Python**。
- `pwsh` 的 `>` 重定向会写成 **UTF-16**（首字节 `0xff`），`ast.parse` 报 `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0`。要 UTF-8 就用 `cmd /c "git show ... > file"` 或 Python 的 `Path.write_text`。
- `pwsh` **不支持 heredoc**：`python - <<'PY'` 报 `Missing file specification after redirection operator`。要把 Python 落成 `.py` 文件再跑。
- → 新项目：Windows 上**所有文本写出走 Python**（`Path.write_text(encoding="utf-8")`），不要用 shell 重定向；临时脚本一律落盘再执行。

---

## 10. 其它（无法归类的遗留）

- `foot_lift_clear` 新奖励项在默认学习率下把已经学会的 8 cm 从 **100% 掉到 25%** ⇒ 作为「实验性旋钮」被整体删除（§6.3-1）。
- 曲线上的「同一错误可以犯两次」：§12 刚修掉「与前进无关却每步都付的正项」，§13 又手写了一个（B5）。加任何新逐步奖励项前**先算它「机器人什么都不做、只直走」时的值**（§13.7）。
- 「指标名不等于物理量」：`qvel[5]` 不是偏航角速度；`sigma` 没平方就是另一个标准差。这类错误不会报错，只会让训练悄悄变差（§13.7）。
- 用户直觉的价值：§12.4 的「机身上台架天然要蹭上去」直接指出了正确诊断方向；§12.6 结论是「**用户的直觉值得直接测量**」。
- 文档/代码腐坏的**共同盲点**：行号漂移（F7）与特权信息审计（C4）都是**一次性判断题，没有任何自动检查器能抓到** ⇒ 结论必须写进文档，而不是只留在当次对话里。
