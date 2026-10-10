# 02 · 奖励实现：`robotdog/reward.py`

审计对象：`robotdog/reward.py`（**375 行**）。核心是 `@dataclass RewardConfig`（30 个字段）与 `compute_reward(env, action, trunk_contact, foot_contact=None, leg_contact=False, terminated=False) -> tuple[float, dict[str,float], float]`。

## 先说结论：数学是干净的，脏在「契约与文档」

这个模块的实现质量明显高于本仓其余部分：

* **恒等式成立**。实测 `abs(reward - sum(info["reward_terms"].values())) == 0`。`compute_reward` 返回的 `terms` 里每一项**已经乘过权重**（`reward.py:15-17` 的 docstring 明说），调用点只有 `env.py:957` 一处。
* **奖励与观测读同一个量**。`vel_filter = bool(env.cfg.obs.vel_filter)`（`reward.py:147`），打开滤波时跟踪项读 `env.filtered_lateral_vel` / `env.filtered_yaw_rate`（`:237`、`:242`），与观测里报的是同一个数。低通在奖励之前推进（`env.py:941-946`）。
* **`forward_vel` 的死区形状是对的**。`:172-176`：`err = lin[0] - cmd.vx`、`fwd_range = max(abs(cmd.vx), FORWARD_RANGE_FLOOR=0.30)`、`max(0, 1 - |err|/range)`。`vx=0` 时原地不动拿满分、往前冲掉分，注释里还留了「为什么必须这样」的实测（cmdtrack4 第 1 级：0.24 m/s 白拿 93 分盖过 45 分的横向跟踪）。
* **`stair_progress` 是正确的一次性结算**。`:333-336`：`climbed = max(0, terrain - _peak_terrain_height)`，只在刷新最高高度时付款、付的是增量 ⇒ 累计总量 = `stair_progress × 爬升高度`，与步数无关。「爬到某一级站着不动」再也拿不到新的钱（这是 experience.md §12 那根支柱的拆解）。
* **`probe_lift` 是真正的势函数塑形**。`:357-364`：`floor = max(_probe_best, env.probe_target)`、`gain = sum(clip(peaks - floor, 0, None))`、只在 `gain > 0` 时更新 `_probe_best` ⇒ 总量有界、重复同样高度不可刷分。起算线取 `env.probe_target`（= `probe_lift_target`，实测恒 0.06，与台阶高无关）而不是 `ratio × 台阶高`，理由在 `:84-89` 写得很清楚：`ratio × 15cm = 0.24` 会高于 `action_scale=0.30` 下能搜到的最好固定动作（0.148 m），梯度恒为零。
* **门控是对的**。`lateral_vel`/`yaw_rate` 只在 `cmd.vy_max` / `cmd.yaw_max` 非零时参与（`:236`、`:241`），所以「只往前走」的阶段与旧行为逐位一致。
* **`yaw` 的零指令基线**（`:249-250`）解决了「yaw_max=0 的阶段没有任何一项读 wz、策略能以 −0.2 rad/s 稳定自转」的问题。求和而不是取 max（`:188-192`）也留了实测依据（取 max 时偏航梯度被丢光，course3 flat_3 检查点横向 r=+0.661 而偏航 r=−0.085）。

**结论：不要动数学。** 下面 R-1 … R-6 全部是「代码与自己的 docstring / 与 `config.yaml` / 与 `docs/` 不一致」这一类，不改数值。

---

## R-1（high）`compute_reward` 的 docstring 说「不推进物理、也不改任何字段」，但它改了环境状态

`reward.py:127-128` 原文：

> `:param env:` `Go2StairsEnv`。只**读**状态（位置/速度/接触/腾空时间/地形），不推进物理，也不改任何字段。

实际有 **3 处写入 `env`**：

| 行 | 写入 | 用途 |
| --- | --- | --- |
| `reward.py:335` | `env._peak_terrain_height = env.terrain_height_under_base` | `stair_progress` 的「历史最高」游标 |
| `reward.py:362` | `env._probe_best = float(np.max(peaks))` | `probe_lift` 的势函数基线 |
| `reward.py:363` | `env._probe_count += 1` | 只进 `info`，供人看 |

为什么这不是吹毛求疵：

1. **势函数塑形的正确性完全依赖「每次状态转移只结算一次」**。任何「为了诊断把一个 transition 重算一遍奖励」的代码（日志、调试、`evaluate` 复算、未来的 off-policy 回放）都会**永久污染** `_peak_terrain_height` / `_probe_best`，而且是**静默**的 —— 分数只会偏高或偏低，不会报错。
2. 本仓已经把奖励数学抽成单独模块、并宣称是纯函数，**读者会照着这句话去信它**。这是最容易在未来被人踩中的一条。
3. 三个字段目前只在 `reward.py` 里被写、在 `env.py:264/271-272/853/863-864` 里被 reset/maintain ⇒ 把写入搬回 `env.py` 的 `step()`（紧接 `_compute_reward()` 之后）不改变任何数值。

### 建议

两条路，选一条：

* **A（最小改动）**：把 `_peak_terrain_height` / `_probe_best` / `_probe_count` 的更新移到 `env.py:957-961` 之间（`_compute_reward()` 调用之后、`_advance_foot_air_time()` 之前），`compute_reward` 改成返回这三个「待写回值」（例如并入 `terms` 之外的第四个返回值，或让 env 侧算）。然后把 docstring 改成「本函数不修改 `env`」的事实陈述，并加一条断言式注释说明为什么必须一次性。
* **B（更彻底）**：让 `compute_reward` 只接受一个不可变的 `RewardInput`（dataclass），把「游标」也放进去，由 env 负责写回。

我倾向 A —— B 的接口改动会波及 `docs/robotdog/reward.md` 的公式章节。

---

## R-2（medium）`act_low` / `act_high` 是**力矩**上下限，却被当成**关节角度**上下限用了两次

`env.py:244-245`：

```python
self.act_low  = self.meta.joint_ranges[:, 0].astype(np.float32)
self.act_high = self.meta.joint_ranges[:, 1].astype(np.float32)
```

而 `go2_model.py:322` 的 `ctrl_ranges` 才是致动器量程：

```python
ctrl_ranges = np.array([model.actuator_ctrlrange[a] for a in actuator_ids], dtype=float)
```

实测（`.tmp/zz_probe3.py`，`compile_scene()` + `extract_meta()`）：

```text
 0 FL_hip_joint    jnt=[-1.0472  1.0472]  ctrl=[-23.7  23.7]
 1 FL_thigh_joint  jnt=[-1.5708  3.4907]  ctrl=[-23.7  23.7]
 2 FL_calf_joint   jnt=[-2.7227 -0.83776] ctrl=[-45.43 45.43]
 7 RL_thigh_joint  jnt=[-0.5236  4.5379]  ctrl=[-23.7  23.7]
```

`go2.xml` 里致动器是 `<motor ctrlrange="-23.7 23.7"/>` / `<motor ctrlrange="-45.43 45.43"/>`（`models/unitree_go2/go2.xml:10`、`:24`），所以 `ctrlrange` 的单位是 **N·m（力矩）**，而关节量程单位是 **rad**。

名字叫 `act_low/act_high`（"actuator"）会让人以为它取自 `ctrl_ranges`，实际上取自 `joint_ranges`。两处用法因此语义错位：

1. **`env.py:911` 把它当成位置限位**：

   ```python
   target = (self.home_angles + self._gait_action() + self.cfg.action_scale * action)
   return np.clip(target, self.act_low, self.act_high)
   ```

   这是**动作管线里的第三段**，对应 skill 契约 5 的 `clip(target, act_low, act_high)`。用关节量程（如膝关节 `[-2.7227, -0.83776]` rad）截位置目标是**合理且必要**的（否则 `kp*(target-q)` 会把关节顶死在 `jnt_range` 上）；用 ±23.7/±45.43 截就会完全失效。**所以这里的实际行为是对的，坏在名字与数据来源对不上**：任何人「修好」它去读 `ctrl_ranges`，动作管线立刻退化成不截断。

2. **`reward.py:317-320` 把它当成 `joint_limit` 越界判据**：

   ```python
   lo, hi = env.act_low, env.act_high
   over = np.clip(lo - q, 0, None) + np.clip(q - hi, 0, None)
   terms["joint_limit"] = rc.joint_limit * float(np.sum(over))
   ```

   这里用 `joint_ranges` 是**唯一正确**的选择（惩罚关节超出自己的行程）。

### 建议

把 `env.py:244-245` 改名为 `self.joint_low / self.joint_high`（或 `q_low/q_high`），保留 `ctrl_ranges` 给 `env.py:926-928` 的力矩截断用；如果希望 `reward.py` 不依赖 env 的属性名，就在 `Go2Meta` 上直接暴露 `joint_low/joint_high`。顺带把 `reward.py:317` 的 `env.act_low, env.act_high` 改成显式的 `env.joint_low/...`，`docs/robotdog/reward.md` 的 `joint_limit` 小节也一并指名来源。

---

## R-3（medium）`RewardConfig` 有 8 个「配置里写、代码不读」的字段，其中 2 个是活动的误导项

实测逐项核对了 `RewardConfig` 的 30 个字段在 `reward.py` 正文里的读取情况：

| 字段 | 声明处 | 被 `compute_reward` 读？ | 评注 |
| --- | --- | --- | --- |
| `forward_vel_sigma` | `:53` | **否** | 注释自承「历史字段，不再参与计算」 |
| `lateral_vel_sigma` | `:69` | **否** | 同上 |
| `yaw_rate_sigma` | `:71` | **否** | 同上 |
| `probe_lift_ratio` | `:92` | **否** | 注释自承「记录用，已弃用」 |
| `probe_lift_max` | `:93` | **否** | 注释自承「记录用，不再按次数截断」 |
| **`base_height`** | `:102` | **是**（`:310`） | 但见下 |
| **`foot_lift_gate_speed`** | `:76` | **是**（`:271`） | 正常 |
| **`probe_blocked_frac`** | `:94` | 不在 reward，在 `env.py:668` | 正常（跨文件读取） |

### R-3a：`base_height` 是整个奖励表里最贵的一项，而观测层已经把它永久禁用

`reward.py:310-312`：

```python
terms["base_height"] = rc.base_height * float(
    (env.base_pos[2] - env.terrain_height_under_base - target_height) ** 2
)
```

`rc.base_height = -2.0`（`config.yaml` 的 `reward.base_height: -2.0`），`target_height = env.nominal_pose`（实测 0.27）。这是**唯一一项同时读「机身绝对高度」和「正下方地形真值」的奖励**。它本身**不是观测泄漏** —— 奖励允许读真值，只要它不进观测。但：

* `ObsConfig.__post_init__()` **硬拒** `base_height=True`（`env.py` 的观测层，config.yaml 里也写「必须保持 false，改成 true 会直接抛 ValueError」）；
* `agent.md §8.1` 声称 `base_height` 分支与 `_height_reference()` **已整体删除**，而实际上 `_support_height()` / `_support_height_mean()` / `_fall_reference_height()` 仍在 `env.py`，`reward.py:310` 也仍在读 `terrain_height_under_base`。

于是当前状态是：**观测里看不到高度，奖励里却有一项 −2.0 × (高度误差)² 在牵制它**。这是资源分配上的真实选择（README/experience.md 里有对应的调参史），但文档说「已删除」会让人以为这项不再起作用，进而在诊断「为什么策略贴地走」时漏掉它。实测单步该值 `-0.00454`（`base_z` 接近 nominal 时很小），一旦姿态下沉就会快速变大。

**建议**：`agent.md §8.1` 改成「观测侧的 `base_height` 分支已删除并加了 `ValueError` 硬门；奖励侧的 `base_height`（`reward.py:310`）**保留**，它读 `terrain_height_under_base` 与 `nominal_pose`」，并在 `docs/robotdog/reward.md` 的目标高度小节注明 `target = env.nominal_pose`。

### R-3b：`probe_lift_max` 的类型是 `int`，但它描述的是「米」

`:93` 写的是 `probe_lift_max: int = 12  # 记录用：势函数本身有界，不再按次数截断`。注释说的是「次数」，类型是 `int`，而同一段注释里 `probe_lift_target: float = 0.06` 是米。这两个字段挤在一行会让人读错单位。既然它已不再参与计算，建议要么删除（连同 `config.yaml` 的同名键），要么明确改成 `# 已弃用` 并把默认值换成 `None`。

### R-3c：`set_reward_overrides` 给了「静默改死字段」的入口

`env.py:687-710` 的实现是 `setattr(self.cfg.reward, key, value)`，只检查 `hasattr`。课程覆盖表 `curriculum.reward_overrides` 目前只动 `foot_lift` / `foot_lift_step_ratio` / `probe_lift` 三个键（`config.yaml`），是安全的。但如果有人在 `curriculum.reward_overrides` 里写 `forward_vel_sigma: 0.9`，会**成功写进去且毫无效果** —— 而这条路径正是「唯一的逐级调参入口」。建议在 `set_reward_overrides` 里对「已弃用字段」发一次 `warnings.warn`（或维护一个 `DEPRECATED_REWARD_FIELDS` 集合直接拒绝）。

---

## R-4（low）`probe_lift` 的 `_probe_best` 用的是**四足全局最大值**，而 `gain` 是**逐足求和**

`:358-363`：

```python
peaks = env._peak_swing_clearance          # shape (4,)，逐足本回合峰值
floor = max(env._probe_best, env.probe_target)
gain  = float(np.sum(np.clip(peaks - floor, 0.0, None)))   # 四足不足量之和
if gain > 0.0:
    env._probe_best = float(np.max(peaks))                 # ← 单一标量
```

`floor` 是一个**标量**，代表「本轮已经达到过的最好峰值」。这意味着：

* 如果 A 足抬到 0.15 m（记为 best），B 足随后抬到 0.10 m，`gain = sum(clip([0.10]-0.15, 0, None)) = 0`，B 足的抬升**完全拿不到分**；
* 如果 A、B 两足**同一步**都到 0.10 m（best 还是 0.06），`gain = 0.04 + 0.04 = 0.08`，而 `_probe_best` 只更新到 0.10 —— **两侧同时抬起的收益被计了两次**。

这是一个有界的、无法刷到无穷的偏差（`_probe_best` 单调上升到物理可达上限后 `gain` 恒 0），但它的**梯度形状不是「抬得更高」而是「同时抬得更高」**。`probe_lift` 的权重很小（`config.yaml` 基线 0.0，最高一级 1.2），所以实际影响有限。

**建议**：如果想让这一项精确对应「被阻挡时把脚抬得比平时更高」的意图，把 `gain` 改成 `max(0, max(peaks) - floor)`（单足口径），或者把 `_probe_best` 也做成 `(4,)` 的逐足基线（口径一致）。**改之前必须先跑等价性证明**（固定动作序列、容差 `0.000e+00`→ 允许不为零但要有记录），因为它会改变 `probe_lift` 的数值。

---

## R-5（low）`foot_lift` 的目标值有除零路径，而相邻的 `foot_lift_gate_speed` 有守卫

`:271-272`：

```python
gate = np.clip(foot_speed / max(float(rc.foot_lift_gate_speed), 1e-6), 0.0, 1.0)   # ← 有守卫
...
lift_target = max(rc.foot_lift_target, rc.foot_lift_step_ratio * float(env.stair.step_height))
lift = np.clip(clearance / lift_target, 0.0, 1.0)                                  # ← 无守卫
```

同一个函数里，`foot_lift_gate_speed` 有 `max(..., 1e-6)` 防零除，`lift_target` 没有。当 `foot_lift_target == 0` 且 `env.stair.step_height == 0`（平地）时，`lift_target = 0` ⇒ `clearance / 0` 产生 `inf`/`nan`，经 `np.clip` 后 `nan` 会**穿过** clip（`np.clip(nan, 0, 1) == nan`），`np.mean` 得到 `nan`，整步奖励变 `nan`。numpy 会同时打一条 RuntimeWarning，但训练脚本通常不把 warning 当错误。

当前 `config.yaml` 的 `foot_lift_target: 0.08`，所以**不会触发**。这是给「调参调到 0」的人留的坑，和 R-2 是同一类（`max(..., 1e-6)` 值得照抄）。

---

## R-6（low）`air_time` 的术语键在 `air_time == 0` 时仍然存在

`:296-307`：`rc.air_time != 0.0` 才计算，否则 `terms["air_time"] = 0.0`。这样做是刻意的（保证 `reward_terms` 的键集合恒定，便于日志对齐），但也意味着**当 `reward.air_time: 0` 时，`terms` 里那 15 个键有一个永远是 0**，而 `info` 里 16 个键（含 `terminal`）也没有任何东西指示「这一项被关掉了」。诊断「哪一项把分扣光」的人会白看一行。

**建议**（可选）：在 `info` 里补一个 `"reward_terms_active": [...]`，或干脆保持现状但在 `docs/robotdog/reward.md` 里写明「键集合恒定，权重为 0 的项恒为 0」。

---

## 已核对无问题的项（避免后人重复审）

* **键集合**：`compute_reward` 写入 `terms` 的 15 个键（AST 实测）= `action_rate, air_time, alive, base_height, body_collision, command_follow, foot_lift, forward_vel, joint_limit, joint_vel, leg_collision, orientation, probe_lift, stair_progress, torque`；`env.py:971` 把 `terms` 放进 `info["reward_terms"]`，`:373` 的 `terminal` **不**并入 `terms` 而是作为 `compute_reward` 的第三个返回值、由 `env.py:957` 单独加进 `reward`。所以 `info["reward_terms"]` 是 **15 键**（`docs/robotdog/env.md:917`/`:1008` 说 16 键 = 把 `terminal` 算进去了，README.md:274 与 reward.md:54 说 15 键）。**以 `info["reward_terms"]` 为准是 15**；`16` 只在「把 terminal 也算作一项」的口径下成立 —— 这是**口径不一致**而不是某个数字错，建议文档统一写「15 项 + terminal = 16」。
* **`stair_progress` 与 `probe_lift` 的游标是 reset 时归零的**：`env.py:853`（`_peak_terrain_height = terrain_height_under_base`，即**回合起点高**而不是 0 —— 这样 spawn 在楼梯前的斜坡上不会白拿分）、`:863-864`。
* **`_blocked` 的进入/解除条件**（`env.py:662-671`）：进入要求 `cmd_vx > 1e-6` **且** `vx < probe_blocked_frac * cmd_vx`；解除要求机身下方地形刷新了 `_blocked_peak`。**推论**：`command_vx == 0`（课程第 4–6 级，`config.yaml` 的 `curriculum.command_vx = [0.45, 0.45, 0.45, 0.0, 0.0, 0.0]`）的台阶阶段**永远无法进入被阻挡状态**，`probe_lift` 在这些阶段**恒为 0**。三级的 `reward_overrides.probe_lift = [.., 0.6, 0.9, 1.2]` 因此在那几级是死权重 —— 除非策略自己采样出横向/偏航指令前进（但 `command_vx=0` 时它也不该前进）。**这是设计上的矛盾，值得确认是否有意为之。**
* **`orientation` 用 `rpy[0]**2 + rpy[1]**2`（平方和）**而不是 `1 - up·z`，与 docstring 一致。
* **`torque` 读 `env.data.actuator_force`**，是「实际施加的力矩」而不是 `data.ctrl` ⇒ 与 `env.py:926` 的截断后一致。
* **足端离地间隙 `env.foot_clearance` 已减掉球半径**（`:273-275` 注释），`foot_radius` 实测 0.022 m。
* **`joint_vel` / `action_rate` / `joint_limit` / `alive` 都是无门控的逐步项**，与 `config.yaml` 的权重符号一致（负项为罚）。
