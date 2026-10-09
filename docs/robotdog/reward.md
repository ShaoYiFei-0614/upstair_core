# `robotdog/reward.py`

## 1. 一句话定位

本仓库**所有奖励数学的唯一来源**：24 个权重/阈值（`RewardConfig`）加一个纯函数
`compute_reward(env, action, ...)`，输出逐步奖励、逐项明细与终止奖励。

## 2. 职责边界

**做：**

* 定义 `RewardConfig`（`config.yaml` 的 `reward:` 段，`robotdog/config.py` 的
  `config_to_env` 用 `_apply` 逐字段写进来）。
* 计算 14 个逐步成形项 + 1 个终止项，并给出每一项的加权值（已经乘过权重）。
* 把「成功登顶」与「摔倒」翻译成 `success` / `termination` 两个终局奖励。

**不做：**

* 不推进物理、不改环境状态、不写任何字段（**只读** `env`）。
* 不决定「本步是否终止」——那是 `robotdog/env.py` 的 `_terminated()` /
  `_success_reached()` 的职责，`compute_reward` 只是通过 `terminated` 参数接收结论。
* 不管观测、不管课程、不管 PPO。

## 3. 关键设计决策与理由

### 3.1 从 `env.py` 整块搬出来（D1）

奖励是这套系统里唯一「写错了不会报错、只会几小时后训不出来」的部分。原先它作为
`Go2StairsEnv._compute_reward` 埋在 800 行的环境文件中间，改一个权重要在物理推进、
观测拼接、终止判定之间翻找。现在 `env.py` 只剩一个三行的转发方法：

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

**实测**：`robotdog/env.py:349` 的转发方法返回的 `terms` 与
`robotdog/reward.py` 的 `compute_reward()` 的 `terms` 有 14 个共同键；`info["reward_terms"]` 再补一个
`"terminal"`，共 **15 键**。

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

### 3.6 躯干碰撞逐步行、腿部碰撞不致命（D6）

`body_collision` 与 `leg_collision` 都是 `权重 × float(bool)` 的**逐步**量，而不是
终止步的一次性惩罚，所以它与擦拭地形的时长成正比。区别在于 `trunk_contact` 是致命
的（`_contact_fatal` 连续 10 步就结束回合），而 `leg_contact` **永不**终止回合 —— 高
楼梯上胫部蹭到立面几乎不可避免，因为它而终止会抹掉整条试错轨迹。

## 4. 公开 API（逐字）

```python
@dataclass
class RewardConfig:
    forward_vel: float = 2.5            # [SPEC] 对 command_vx 的高斯跟踪
    forward_vel_sigma: float = 0.25
    orientation: float = -1.2           # [SPEC] roll^2 + pitch^2（平方和）
    torque: float = -1.5e-4             # [SPEC] sum(tau^2)（关节力矩平方和）
    foot_lift: float = 0.9              # [SPEC] 摆动期离地间隙成形
    body_collision: float = -4.0        # [SPEC] 躯干接触地形
    forward_progress: float = 1.5       # 对 vx 线性，所以 vx=0 时一分也拿不到
    forward_progress_max: float = 1.5
    forward_progress_min: float = -0.5
    foot_lift_target: float = 0.08      # 目标值的下限，而不是完整的目标值
    foot_lift_step_ratio: float = 1.4   # 取较大者：target = max(target, ratio * step_height)
    foot_lift_gate_speed: float = 0.25  # m/s，三维足端速度门限
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
| `forward_vel` | `forward_vel * exp(-(lin[0] - command_vx)² / forward_vel_sigma)`（σ **不**平方） |
| `forward_progress` | `forward_progress * clip(lin[0], forward_progress_min, forward_progress_max)` |
| `orientation` | `orientation * (rpy[0]² + rpy[1]²)` |
| `torque` | `torque * sum(tau²)` |
| `foot_lift` | `foot_lift * mean(gate * lift)`；`gate = clip(foot_speed / max(foot_lift_gate_speed,1e-6), 0, 1)`，`foot_speed = ‖(feet - _prev_foot_pos)/control_dt‖`（三维），`lift = clip(clearance / lift_target, 0, 1)` |
| `body_collision` | `body_collision * float(trunk_contact)` |
| `leg_collision` | `leg_collision * float(leg_contact)` |
| `air_time` | `air_time * sum((clip(air - air_time_target, 0, None) - clip(air - air_time_max, 0, None)) * landed)`；`landed = foot_contact & ~_prev_foot_contact`，`air = _foot_air_time`（**本步之前**累积的腾空时间）；`air_time == 0.0` 时恒 `0.0` |
| `base_height` | `base_height * (base_pos[2] - terrain_height_under_base - nominal_pose)²` |
| `joint_vel` | `joint_vel * sum(joint_vel²)` |
| `action_rate` | `action_rate * sum((action - _prev_action)²)` |
| `joint_limit` | `joint_limit * sum(clip(lo - q, 0, None) + clip(q - hi, 0, None))` |
| `alive` | `alive`（常数） |
| `stair_progress` | `stair_progress * max(0.0, terrain_height_under_base - _start_terrain_height)` |
| `terminal` | `success` 若 `env._reached_top` 否则 `termination`；仅当 `terminated=True` |

`RewardConfig` 的 **24 个字段**里有 10 个是**参数**而不是键：`forward_vel_sigma`、
`forward_progress_max`、`forward_progress_min`、`foot_lift_target`、
`foot_lift_step_ratio`、`foot_lift_gate_speed`、`air_time_target`、`air_time_max`、
`termination`、`success`。

## 5. 内部结构与数据流

```
robotdog/env.py:564  step()
    │  物理推进 decimation 次（env.py:566-574）
    │  trunk_contact, leg_contact = _contact_flags()          env.py:384
    │  foot_contact = _foot_contact_mask()                    env.py:362
    │  terminated, truncated = _terminated(...)               env.py:474
    │  if _success_reached(): _reached_top = True             env.py:242
    ▼
robotdog/env.py:349  _compute_reward(..., terminated=terminated)
    ▼
robotdog/reward.py:83  compute_reward(env, action, trunk_contact,
                                      foot_contact, leg_contact, terminated)
    │  读 env.cfg.reward / env.base_rpy / env.base_lin_vel_world / env.data.actuator_force
    │     env.foot_pos / env._prev_foot_pos / env.foot_clearance / env.stair.step_height
    │     env._foot_air_time / env._prev_foot_contact / env.base_pos
    │     env.terrain_height_under_base / env.nominal_pose / env.joint_vel
    │     env.act_low / env.act_high / env.joint_pos / env._prev_action
    │     env._start_terrain_height / env._reached_top
    │  （全部只读；一个字段都不写）
    ▼
return (sum(terms.values()), terms, terminal)
    ▼
robotdog/env.py:349  terms["terminal"] = terminal
                     return base + terminal, terms
    ▼
robotdog/env.py:564  reward, terms = ...；_advance_foot_air_time(foot_contact)   env.py:377
                     info["reward_terms"] = terms
```

**顺序上的两个关键点**（改动 `step` 时必须保持）：

1. `_success_reached()`（`env.py:237`）必须在 `_compute_reward` **之前**跑，因为
   `_reached_top` 决定 `terminal` 取 `success` 还是 `termination`。
2. `_advance_foot_air_time()`（`env.py:372`）必须在 `_compute_reward` **之后**跑 ——
   `air_time` 项读的是「本步之前」累积的腾空时间，先累加就等于把当步白送出去。

## 6. 关键常量与默认值

| 名字 | 值 | 含义 |
| --- | --- | --- |
| `RewardConfig.forward_vel` | `2.5` | 唯一的大额正项 |
| `RewardConfig.forward_vel_sigma` | `0.25` | 高斯分母（不平方），对应 ~0.25 m/s 的速度容差 |
| `RewardConfig.forward_progress` | `1.5` | 线性项，`vx=0` 时恰为 0 |
| `RewardConfig.foot_lift` | `0.9` | 抬脚成形项上限 |
| `RewardConfig.foot_lift_target` | `0.08` m | 目标下限；`1.4 × 0.05 = 0.07 < 0.08`，所以 5 cm 处取它 |
| `RewardConfig.foot_lift_step_ratio` | `1.4` | 台阶越高目标越高 |
| `RewardConfig.foot_lift_gate_speed` | `0.25` m/s | 三维足端速度门限 |
| `RewardConfig.air_time` | `0.6` | 落地那一步结算的腾空成形项 |
| `RewardConfig.alive` | `0.15` | 每步常数 |
| `RewardConfig.success` | `50.0` | 成功登顶一次性 |
| `RewardConfig.termination` | `-15.0` | 摔倒一次性 |
| `RewardConfig.stair_progress` | `3.0` | 爬升进展（相对回合起点） |

## 7. 与其它模块的关系

* `robotdog/env.py:37` `from robotdog.reward import RewardConfig, compute_reward` ——
  `EnvConfig.reward` 的默认工厂（`env.py:126`）与 `_compute_reward` 的转发都靠它。
  历史上 `RewardConfig` 定义在 `env.py` 里，现在 `env.py` **重导出**它，所以
  `from robotdog.env import RewardConfig` 仍然可用。
* `robotdog/config.py:163` `_apply(cfg.reward, _section(raw, "reward"), where="reward")`
  —— 权重来自 `config.yaml` 的 `reward:` 段，未知键会报错并列出字段名。
* `robotdog/policy.py:17` `from robotdog.env import EnvConfig, ObsConfig, RewardConfig`
  —— `reward_config` 会随每个 checkpoint 的 `_env.json` 一起存盘（`policy.py:95`
  用 `_rebuild(RewardConfig, ...)` 还原），所以历史权重的奖励口径可回溯。
* `scripts/train.py:263` 打印 `[奖励] forward_vel=... foot_lift=... success=...
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
  `_foot_air_time`、`_prev_foot_contact`、`_prev_action`、`_start_terrain_height`、
  `_reached_top`）。改 `env.py` 里这些名字而不改 `reward.py` 会在运行期
  `AttributeError`，`compileall` 抓不到。
* **`air_time` 项的时序极易踩错**：`air` 读的是**本步之前**的累积腾空时间。把
  `_advance_foot_air_time()` 挪到奖励计算之前，这一项就永远读到 0。
* **`forward_vel` 的高斯分母 `forward_vel_sigma` 没有平方**（写成 `exp(-err²/σ)` 而
  非 `exp(-err²/σ²)`），这是历史行为，已由上述等价性验证钉住；改它会改变所有已存档
  权重的奖励尺度。
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
print("term count:", len(terms))          # 15（14 项 + terminal）
```

单独研究某一项时可以直接调纯函数，只要给一个字段齐全的 `env`：

```python
from robotdog.reward import compute_reward
base, terms, terminal = compute_reward(env, np.zeros(12), trunk_contact=False,
                                       leg_contact=False, terminated=False)
print(base, sorted(terms), terminal)      # 14 个键，终局项恒 0.0
```
