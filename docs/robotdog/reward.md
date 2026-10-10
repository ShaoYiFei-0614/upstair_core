# `robotdog/reward.py`

## 1. 一句话定位

本仓库**所有奖励数学的唯一来源**：`RewardConfig` 的 28 个权重/阈值加一个纯函数
`compute_reward(env, action, ...)`（`robotdog/reward.py:114`），输出逐步奖励、逐项明细与终止奖励。

## 2. 职责边界

**做：**

* 定义 `RewardConfig`（`config.yaml` 的 `reward:` 段，`robotdog/config.py` 的
  `config_to_env` 用 `_apply` 逐字段写进来）。
* 计算 15 个逐步成形项 + 1 个终止项，并给出每一项的加权值（已经乘过权重）。
* 把「成功登顶」与「摔倒」翻译成 `success` / `termination` 两个终局奖励。

**不做：**

* 不推进物理、不改环境状态、不写任何字段（**只读** `env`）。
* 不决定「本步是否终止」——那是 `robotdog/env.py` 的 `_terminated()` /
  `_success_reached()` 的职责，`compute_reward` 只是通过 `terminated` 参数接收结论。
* 不管观测、不管课程、不管 PPO。

## 3. 关键设计决策与理由

### 3.1 从 `env.py` 整块搬出来（D1）

奖励是这套系统里唯一「写错了不会报错、只会几小时后训不出来」的部分。原先它作为
`Go2StairsEnv._compute_reward` 埋在环境文件中间（该文件现在 1101 行），改一个权重要在
物理推进、观测拼接、终止判定之间翻找。现在 `env.py:611-622` 只剩一个 `_compute_reward` 转发方法：

```python
def _compute_reward(self, action, trunk_contact, foot_contact=None,
                    leg_contact=False, terminated=False):
    base, terms, terminal = compute_reward(self, action, trunk_contact,
                                           foot_contact, leg_contact=leg_contact,
                                           terminated=terminated)
    terms["terminal"] = terminal
    return base + terminal, terms
```

**被否决的方案**：把 20 多个状态量逐个作为参数传进 `compute_reward`（显式但调用点
要写 20 行，且新增一项就得改签名）、或把整个状态先打包成一个 dataclass（多一层
拷贝，且 `env` 的每个属性本来就是只读 property）。

### 3.2 逐项明细里已经乘过权重（D2）

`terms["torque"]` 是 `-1.5e-4 * sum(tau²)`，不是 `sum(tau²)`。于是
`sum(terms.values())` **就是**本步奖励，不需要另一张「权重 × 原始值」的表。
代价是明细里看不出「原始量有多大」，但那个量在 `info` 里另有一份
（`base_x` / `forward_vel` / `roll` / `pitch` / `base_z` …）。

**实测**：`robotdog/env.py:611-622` 的 `_compute_reward` 转发方法返回的 `terms` 与
`robotdog/reward.py:114` 的 `compute_reward()` 的 `terms` 有 **15 个共同键**；`info["reward_terms"]` 再补一个
`"terminal"`，共 **16 键**。

### 3.3 终止奖励单列而不是塞进 `terms`（D3）

`compute_reward` 返回三元组 `(逐步奖励, terms, 终止奖励)`。`terms` 里**不含**
`terminal`，而 `step()` 把它补进 `info["reward_terms"]`。这样两条恒等式同时成立：
`step` 返回的 `reward == sum(info["reward_terms"].values())`，以及
`reward == 逐步奖励 + 终止奖励`。

### 3.4 `foot_lift` 用目标下限 + 台阶比例（D4）

`lift_target = max(foot_lift_target, foot_lift_step_ratio * step_height)`。5 cm 时
两项恰好相等（`0.08 = 1.4 × 0.05`，取 `max` 得 `0.08`），所以把目标改成随地形缩放
**不会**让已经学会的 3~5 cm 阶段回退；而 9 cm 以上时目标才真正抬高（`1.4 × 0.09 =
0.126`）。

### 3.5 抬脚奖励用三维足端速度门控（D5）

`gate = clip(‖(feet - _prev_foot_pos)/control_dt‖ / foot_lift_gate_speed, 0, 1)`。
**被否决的方案**：只用水平速度——那会奖励「贴地平扫」（快但抬不起来），同时砍掉高
台阶真正需要的慢速抬腿。

### 3.6 躯干碰撞逐步行、两种碰撞都不致命（D6）

`body_collision` 与 `leg_collision` 都是 `权重 × float(bool)` 的**逐步**量，而不是
终止步的一次性惩罚，所以它与擦拭地形的时长成正比。**两者都不终止回合**：
`_contact_fatal()` 现在恒返回 `False`（历史上躯干连续接触 10 步会结束回合，那条规则
从未真正触发过，却随时可能误杀「上台架时蹭立面」的正当姿态，实测证明见
`experience.md` §12）。因此 `body_collision` 是唯一的兜底手段，权重从 `-4.0` 降到
`-1.2` —— 原值让「贴着立面蹭」比站桩还亏，策略学会试两下就放弃（GIF 复盘，见
`experience.md` §13）。

### 3.7 `probe_lift` 用势函数塑形，起算线不跟台阶高走（D7）

**要解决的问题**：机器人撞上够不着的立面后，唯一能改变局面的行为是「把脚抬得比
平时更高再试」。原稿想奖励「被阻挡时完成一次抬高→重新落脚」，但三条路都试过：

1. **落地帧越线给固定分** —— 检测链先写错了：`_track_swing_peak` 一接触地形就把峰值
   清零，而结算发生在落地那一帧，读到的是被清零的旧值，比真实峰值低 0.01~0.03 m，
   500 步里一次都不触发。
2. **把门槛定成 `1.6 × 台阶高`** —— 实测（`.tmp/calib_probe3.py`）在 15 cm 立面之前
   被阻挡时，默认步态的摆动峰值只到 0.09~0.12 m；而 `action_scale=0.30` 下搜遍固定
   动作，最好的一种也只到 **0.148 m**（`.tmp/calib_probe4.py`）。门槛取 0.24 m 时梯度
   恒为零。
3. **门控不做**（不加「只在被阻挡时给分」）—— 那就变成奖励普通步态。

**最终形态**是势函数塑形：`env._probe_best` 记住本回合「被阻挡期间达到过的最高摆动
峰值」，每一步只结算「比历史最好还高出多少」，起算线只取 `probe_lift_target = 0.06`
（普通步态峰值带 0.06~0.12 的下沿）。性质三条：总量有界（≤ `probe_lift ×（可达最高
峰值 − 起算线）`）、不可刷分（重复同样高度拿不到第二次）、有梯度（抬得更高就被记录，
即使那一脚最终没踩上去）。实测梯度（`.tmp/probe_gain.py`，`probe_lift = 1.0`）：
9 cm 上零动作 0.392 / `thigh +1` 0.943 / 40 组随机里最好 1.189；15 cm 上零动作 0.141 /
最好 0.909。**在 15 cm 上这一项远不足以补上 `body_collision` 的量级**，它只是把
「抬得更高」变成一个可优化方向，不是解药。

## 4. 公开 API（逐字）

```python
@dataclass
class RewardConfig:
    forward_vel: float = 10.0           # [SPEC] 对 vx 指令的线性死区跟踪
    forward_overspeed: float = 1.0      # [SPEC] 超速侧惩罚斜率
    overspeed_range: float = 0.45       # 超速惩罚的斜率量程 [m/s]
    forward_range_floor: float = 0.30   # vx 指令为 0 时归一化误差的量程下限 [m/s]
    orientation: float = -1.2           # [SPEC] roll^2 + pitch^2（平方和）
    torque: float = -1.5e-4             # [SPEC] sum(tau^2)（关节力矩平方和）
    foot_lift: float = 0.9              # [SPEC] 摆动期离地间隙成形
    body_collision: float = -1.2        # [SPEC] 躯干接触地形
    lateral_vel: float = 6.0            # [SPEC] 横向速度指令跟踪（仅在 vy_max > 0 时计入）
    yaw_rate: float = 4.0               # [SPEC] 偏航角速度指令跟踪（仅在 yaw_max > 0 时计入）
    foot_lift_target: float = 0.08      # 目标值的下限，而不是完整的目标值
    foot_lift_step_ratio: float = 1.4   # 取较大者：target = max(target, ratio * step_height)
    foot_lift_gate_speed: float = 0.25  # m/s，三维足端速度门限
    probe_lift: float = 1.0             # 被阻挡时的抬高进展（势函数塑形）
    probe_lift_target: float = 0.06     # m，进展的起算线
    probe_blocked_frac: float = 0.30    # 实测速度 < 该比例 × 指令速度 ⇒ 被阻挡
    air_time: float = 0.6
    air_time_target: float = 0.30
    air_time_max: float = 0.60
    base_height: float = -2.0
    joint_vel: float = -2.0e-5
    action_rate: float = -0.02
    joint_limit: float = -0.5
    alive: float = 0.15
    termination: float = -15.0
    success: float = 50.0
    stair_progress: float = 3.0
    leg_collision: float = -0.25


def compute_reward(env, action: np.ndarray, trunk_contact: bool,
                   foot_contact: np.ndarray | None = None,
                   leg_contact: bool = False,
                   terminated: bool = False
                   ) -> tuple[float, dict[str, float], float]:
```

`__all__` 未定义（模块内两个公开名即可用）。

### 4.1 逐项公式

记 `lin = env.base_lin_vel_world`、`rpy = env.base_rpy`、
`tau = env.data.actuator_force`、`h = env.stair.step_height`、
`lift_target = max(foot_lift_target, foot_lift_step_ratio * h)`。

| key | 公式 |
| --- | --- |
| `forward_vel` | `forward_vel * track`，其中 `err = lin[0] - command.vx`、`fwd_range = max(\|command.vx\|, rc.forward_range_floor)`、`track = max(0, 1 - \|err\| / fwd_range)`；`err > 0` 时再减 `rc.forward_overspeed * err / rc.overspeed_range`（**线性死区 + 超速惩罚**：死区在 `\|err\| > fwd_range` 之后恒 0、梯度消失，超速是主动选择所以单独有价钱）。`forward_range_floor = 0.30` 在 `robotdog/reward.py:58` |
| `orientation` | `orientation * (rpy[0]² + rpy[1]²)` |
| `torque` | `torque * sum(tau²)` |
| `foot_lift` | `foot_lift * mean(gate * lift)`；`gate = clip(foot_speed / max(foot_lift_gate_speed,1e-6), 0, 1)`，`foot_speed = ‖(feet - _prev_foot_pos)/control_dt‖`（三维），`lift = clip(clearance / lift_target, 0, 1)` |
| `body_collision` | `body_collision * float(trunk_contact)` |
| `probe_lift` | 仅当 `rc.probe_lift != 0 and env._blocked` 时非零：`peaks = env._peak_swing_clearance`、`floor = np.maximum(env._probe_best, env.probe_target)`、`gain = np.sum(np.clip(peaks - floor, 0, None))`、`terms["probe_lift"] = probe_lift * gain`，并在 `gain > 0` 时把**逐足地板** `env._probe_best = np.maximum(env._probe_best, peaks)`、`_probe_count += 1`（**势函数塑形**，总量有界、不可刷分）。`env._probe_best` 是**逐足 ndarray**（先前是标量：四足同一步同时抬高时，标量地板会把只抬了一只脚的足也算成进展）；`env.probe_target` 是只读属性（`robotdog/env.py:700`），直接取 `reward.probe_lift_target`（0.06 m），**不乘台阶高** |
| `command_follow` | **求和**：`vy_max > 0` 时 `follow += lateral_vel * max(0, 1 - \|vy − command.vy\| / vy_max)`；`yaw_max > 0` 时 `follow += yaw_rate * max(0, 1 - yaw_err / yaw_max)`，其中 `yaw_err = \|env.yaw_rate − command.wz\|`，且 `command.wz == 0.0` 时先做 `yaw_err = 0.5 * (yaw_err + \|wz\|)`（零指令基线惩罚）；两者都不开时**恒 0.0**。`vy` 取滤波量还是瞬时量由 `obs.vel_filter` 决定 |
| `leg_collision` | `leg_collision * float(leg_contact)` |
| `air_time` | `air_time * sum((clip(air - air_time_target, 0, None) - clip(air - air_time_max, 0, None)) * landed)`；`landed = foot_contact & ~_prev_foot_contact`，`air = _foot_air_time`（**本步之前**累积的腾空时间）；`air_time == 0.0` 时恒 `0.0` |
| `base_height` | `base_height * (base_pos[2] - terrain_height_under_base - nominal_pose)²` |
| `joint_vel` | `joint_vel * sum(joint_vel²)` |
| `action_rate` | `action_rate * sum((action - _prev_action)²)` |
| `joint_limit` | `joint_limit * sum(clip(lo - q, 0, None) + clip(q - hi, 0, None))` |
| `alive` | `alive`（常数） |
| `stair_progress` | `stair_progress * max(0.0, env.terrain_height_under_base - env._peak_terrain_height)`，并在 `> 0` 时把 `_peak_terrain_height` 前移（**一次性结算**，站桩拿不到第二次） |
| `terminal` | 若 `terminated` 为真：`success`（当 `env._reached_top`）否则 `termination`；否则 0.0。**这一项由 `compute_reward` 单独返回、不在 `terms` 里** |

`RewardConfig` 的 **28 个字段**里有 14 个是**参数**而不是键：`forward_overspeed`、
`overspeed_range`、`forward_range_floor`、`lateral_vel`、`yaw_rate`、
`foot_lift_target`、`foot_lift_step_ratio`、`foot_lift_gate_speed`、
`probe_lift_target`、`probe_blocked_frac`、
`air_time_target`、`air_time_max`、`termination`、`success`
（后四个字段是门槛/终局值：它们只作为其它项的门槛或终局结算参与计算，本身不产生 `terms` 键）。

> **`command_follow` 必须门控**（一次真实的自伤）：第一版没有 `if cmd.vy_max:` 判断，
> 直接按 `lateral_vel` / `yaw_rate` 两项计分。实测 5 cm 上 125 步白拿
> `lateral_vel = +139.73`、`yaw_rate = +28.03` —— 机器人只是在直走，横向/偏航误差恰好落在
> σ 附近，于是它凭空得到约 +1.0/步，**等于又造了一个「原地拖延」正项**。加上门控后
> `command_follow` 恒为 `+0.000e+00`（逐位精确 0），不含这一项的阶段与加指令之前完全一致。
> 另外 `yaw_rate` 读的是 `env.yaw_rate`，它是**机体系**偏航角速度 `(Rᵀ qvel[3:6])[2]`，
> **不是** `qvel[5]`，也不是曾经用过的世界系公式；详见 `docs/robotdog/env.md` 与
> `experience.md` §16（世界系读数被与指令量程同阶的步态振荡主导，策略只要凑对直流
> 分量就能拿分，实测平地权重把预算全压在前进了）。

> 本轮删掉了 `forward_progress` / `forward_progress_max` / `forward_progress_min`
> 三个字段以及对应的线性项（见 `experience.md` §12）。
>
> 更后一轮发现高斯的根问题：`forward_vel` 以前**只拿 `command.vx` 当目标**，而 `vx` 是
> 定值基线、不随回合重抽，于是课程里 `command_vx = 0.0` 的级别「指令说别前进，却没有
> 任何一项惩罚前进」，小跑步态先验自带的前进偏置白拿分（实测该级均回报 9052.5/1000 步
> ≈ 9.05/步，同时横向指令 ±0.10 只跟出 ±0.02 m/s）。现已改成**按量程归一化的线性死区**：
> 量程 `max(|cmd.vx|, rc.forward_range_floor = 0.30)`，权重 2.5 → **10.0**，`vx = 0` 时该项
> 恰好为 0。原先的 5 个字段（`forward_vel_sigma` / `lateral_vel_sigma` / `yaw_rate_sigma` /
> `probe_lift_ratio` / `probe_lift_max`）已在本轮随线性死区改造一并从 `RewardConfig` 删除，
> 原因见 `experience.md` §2.2 与 §12。

## 5. 内部结构与数据流

```
robotdog/env.py:942  step()
    │  物理推进 decimation 次（env.py:951-958）
    │  trunk_contact, leg_contact = _contact_flags()          env.py:740
    │  foot_contact = _foot_contact_mask()                    env.py:624
    │  _track_swing_peak(foot_contact)                        env.py:652
    │  _update_probe_state(foot_contact)                      env.py:672
    │  terminated, truncated = _terminated(...)               env.py:842
    │  低通滤波推进 filtered_lateral_vel / filtered_yaw_rate  env.py:973-975
    │  if _success_reached(): _reached_top = True             env.py:462
    ▼
robotdog/env.py:989  _compute_reward(..., terminated=terminated)   ← 调用点在 env.py:989
    ▼
robotdog/reward.py:114  compute_reward(env, action, trunk_contact,
                                       foot_contact, leg_contact, terminated)
    │  读 env.cfg.reward / env.cfg.command / env.cfg.obs.vel_filter
    │     env.base_rpy / env.base_lin_vel_world / env.base_lin_vel_body / env.data.actuator_force
    │     env.foot_pos / env._prev_foot_pos / env.foot_clearance / env.stair.step_height
    │     env._foot_air_time / env._foot_contact_mask / env._prev_foot_contact / env.base_pos
    │     env.terrain_height_under_base / env._peak_terrain_height / env.nominal_pose
    │     env.joint_vel / env.joint_pos / env.act_low / env.act_high / env._prev_action
    │     env._blocked / env._peak_swing_clearance / env._probe_best / env._probe_count
    │     env.probe_target / env.filtered_lateral_vel / env.filtered_yaw_rate
    │     env.yaw_rate / env._reached_top
    │  （全部只读；唯一「写」的是 probe_lift 那一项把 env._probe_best/_probe_count 前移）
    ▼
return (sum(terms.values()), terms, terminal)
    ▼
robotdog/env.py:621  terms["terminal"] = terminal
                     return base + terminal, terms
    ▼
robotdog/env.py:989  reward, terms = ...；_advance_foot_air_time(foot_contact)   env.py:993
                     info["reward_terms"] = terms
```

**顺序上的两个关键点**（改动 `step` 时必须保持）：

1. `_success_reached()`（`env.py:462`）必须在 `_compute_reward` **之前**跑，因为
   `_reached_top` 决定 `terminal` 取 `success` 还是 `termination`。
2. `_advance_foot_air_time()`（`env.py:645`）必须在 `_compute_reward` **之后**跑 ——
   `air_time` 项读的是「本步之前」累积的腾空时间，先累加就等于把当步白送出去。

## 6. 关键常量与默认值

| 名字 | 值 | 含义 |
| --- | --- | --- |
| `RewardConfig.forward_vel` | `10.0` | 唯一的大额正项（线性死区，见 §4.1） |
| `RewardConfig.forward_overspeed` | `1.0` | 超速侧惩罚斜率（`err > 0` 时按 `err / overspeed_range` 扣） |
| `RewardConfig.overspeed_range` | `0.45` m/s | 超速惩罚的斜率量程（超速 0.45 m/s 扣光这一项） |
| `RewardConfig.forward_range_floor` | `0.30` m/s | `vx = 0` 时归一化误差的量程下限 |
| `RewardConfig.body_collision` | `-1.2` | 躯干接触地形（**不致命**，唯一的兜底） |
| `RewardConfig.lateral_vel` | `6.0` | 横向指令跟踪的权重（`vy_max > 0` 才计入 `command_follow`） |
| `RewardConfig.yaw_rate` | `4.0` | 偏航指令跟踪的权重（`yaw_max > 0` 才计入 `command_follow`） |
| `RewardConfig.foot_lift` | `0.9` | 抬脚成形项上限 |
| `RewardConfig.foot_lift_target` | `0.08` m | 目标下限；`1.4 × 0.05 = 0.07`，取 `max` 后是 `0.08` |
| `RewardConfig.foot_lift_step_ratio` | `1.4` | 台阶越高目标越高 |
| `RewardConfig.foot_lift_gate_speed` | `0.25` m/s | 三维足端速度门限 |
| `RewardConfig.air_time` | `0.6` | 落地那一步结算的腾空成形项 |
| `RewardConfig.alive` | `0.15` | 每步常数 |
| `RewardConfig.success` | `50.0` | 成功登顶一次性 |
| `RewardConfig.termination` | `-15.0` | 摔倒一次性 |
| `RewardConfig.stair_progress` | `3.0` | 爬升进展（**一次性结算**：只在刷新本回合最高台阶时付） |

## 7. 与其它模块的关系

* `robotdog/env.py:38` `from robotdog.reward import RewardConfig, compute_reward` ——
  `EnvConfig.reward` 的默认工厂（`env.py:228`）与 `_compute_reward` 的转发都靠它。
  历史上 `RewardConfig` 定义在 `env.py` 里，现在 `env.py` **重导出**它，所以
  `from robotdog.env import RewardConfig` 仍然可用。
* `robotdog/config.py:296` `_apply(cfg.reward, _section(raw, "reward"), where="reward")`
  —— 权重来自 `config.yaml` 的 `reward:` 段，未知键会报错并列出字段名。
* `robotdog/policy.py:17` `from robotdog.env import CommandConfig, EnvConfig, ObsConfig, RewardConfig`
  —— `reward_config` 会随每个 checkpoint 的 `_env.json` 一起存盘（`policy.py:95`
  用 `_rebuild(RewardConfig, ...)` 还原），所以历史权重的奖励口径可回溯。
* `scripts/train.py:334` 打印 `[奖励] forward_vel=... foot_lift=... success=...
  termination=...` 作为启动横幅。

## 8. 实测与经验教训

* **加权明细是定位「谁把分扣光了」的唯一手段。** 早期一次诊断就是靠
  `info["reward_terms"]` 排序发现 `straight_line` + `heading` + `lateral_vel` 三项吃掉
  了 99.9% 的惩罚质量（-12.67 + -2.83 + -1.91 每步），而唯一的正向前进项只有
  +0.36 —— 那三项随后被删除。见 `experience.md` §2、`train.md` §3。
* **`|a|`（平均动作绝对值）是「有没有在学」的判据。** 参考策略 0.42（p95 顶到
  1.00），原地站死只有 0.09~0.14。奖励改错时这个数会先掉下去。
* **抽离本身做了等价性验证**：`.tmp/verify_reward_refactor.py` 用
  `git show HEAD:robotdog/env.py` 建私有模块 `old_env`，对 flat / 5 / 8 / 11 cm ×
  seed 0/1/7 共 4,503 个控制步逐字段比对，`max|d reward| = 0.000e+00`、
  `max|d obs| = 0.000e+00`、`term/trunc diffs = 0`。

## 9. 已知限制与坑

* **`compute_reward` 依赖 `env` 的私有属性与命名约定**（`_prev_foot_pos`、
  `_foot_air_time`、`_prev_foot_contact`、`_prev_action`、`_peak_terrain_height`、
  `_peak_swing_clearance`、`_probe_best`、`_probe_count`、`probe_target`、`_reached_top`）。
  改 `env.py` 里这些名字而不改 `reward.py` 会在运行期
  `AttributeError`，`compileall` 抓不到。
* **`air_time` 项的时序极易踩错**：`air` 读的是**本步之前**的累积腾空时间。把
  `_advance_foot_air_time()` 挪到奖励计算之前，这一项就永远读到 0。
* **5 个历史字段已从 `RewardConfig` 删除。** `forward_vel_sigma`、`lateral_vel_sigma`、
  `yaw_rate_sigma`、`probe_lift_ratio`、`probe_lift_max` 现在都不存在（原因见
  `experience.md` §2.2 与 §12）；写进 `config.yaml` 的 `reward:` 段会被配置校验直接拒绝。
  前进/横向/偏航三项的容差是「指令量程」（`max(|cmd.vx|, forward_range_floor)` 或
  `vy_max` / `yaw_max`），不是 σ。
* **`terminal` 不在 `compute_reward` 返回的 `terms` 里**，只有
  `info["reward_terms"]` 有它。直接对 `terms` 求和得到的是逐步奖励，差一个终局项。
* `limit` 项用的 `act_low/act_high` 是**关节软限位**而不是 `ctrl_ranges`（力矩限幅），
  两者不可混用。

## 10. 示例用法

```python
import numpy as np
from robotdog.env import EnvConfig, Go2StairsEnv
from robotdog.tools.terrain import demo_stairs_spec

env = Go2StairsEnv(demo_stairs_spec(step_height=0.08, tread=0.50, n_up=8),
                   EnvConfig(), seed=0)
obs, _ = env.reset(seed=0)
for _ in range(50):
    obs, reward, terminated, truncated, info = env.step(np.zeros(12))
    terms = info["reward_terms"]
    if terminated or truncated:
        break

# 逐项按值排序，看清是谁在扣分
for name, value in sorted(terms.items(), key=lambda kv: kv[1]):
    print(f"{name:18s} {value:+9.4f}")
print("sum(terms) =", sum(terms.values()), " reward =", reward)
assert abs(sum(terms.values()) - reward) < 1e-9
print("term count:", len(terms))          # 16（15 项 + terminal）
```

单独研究某一项时可以直接调纯函数，只要给一个字段齐全的 `env`：

```python
from robotdog.reward import compute_reward
base, terms, terminal = compute_reward(env, np.zeros(12), trunk_contact=False,
                                       leg_contact=False, terminated=False)
print(base, sorted(terms), terminal)      # 15 个键，终局项恒 0.0
```
