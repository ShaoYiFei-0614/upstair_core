# 03 · 环境与观测契约：`robotdog/env.py`（1066 行）

审计对象：`robotdog/env.py`（**1066 行**）。重点核对了 skill 契约 3（观测项必须有真机来源、禁用真值）、契约 4（观测维数全程恒定）、契约 5（唯一动作管线）、契约 7（终止只来自高度/倾角/超时）。

## 先说结论：四条硬契约目前**都成立**，但守卫方式不均衡

| 契约 | 状态 | 守卫强度 |
| --- | --- | --- |
| 3 · 观测项的真机来源 | **成立** | 只有 `base_height` 一条硬门（`__post_init__` 抛 `ValueError`）；其余靠代码里不留分支 |
| 4 · 观测维数恒定 | **成立** | **无任何断言** —— 靠 `ObsConfig` 是冻结语义 + 课程只改范围不改开关 |
| 5 · 唯一动作管线 | **成立** | 无第二路径（`"absolute"` 模式已删），但没有测试锁定 |
| 7 · 终止判据 | **成立** | `_contact_fatal()` 恒 `False`；无「连续 N 次接触」去抖 |

逐条证据见下。**E-1 / E-2 / E-3 是真正值得修的三处**，其余是文档与命名问题。

---

## E-1（high）裸构造环境时 `step_height` 报 0.05 m —— 没有楼梯却告诉策略「楼梯有 5 cm」

### 现状

```python
# robotdog/tools/terrain.py:78 与 :221
step_height: float = 0.05     # StairSpec 与 demo_stairs_spec 的默认值
```

```python
# robotdog/env.py:231-240
def __init__(self, stair: StairSpec | None = None, cfg: EnvConfig | None = None, ...):
    ...
    self.stair = stair or StairSpec()          # ← 不传 stair 就是「5 cm 台阶」的规格
```

```python
# robotdog/env.py:559-562
if obs_cfg.step_height:
    parts.append(np.array([float(self.stair.step_height)]))
```

实测（`.tmp/zz_audit_probe*.py`）：

```text
>>> from robotdog.env import Go2StairsEnv, EnvConfig
>>> e = Go2StairsEnv(cfg=EnvConfig(), seed=0)      # 不传 stair
>>> e._obs_frame()[45]
0.05
```

也就是**平地环境里观测报「台阶高 5 cm」**。

### 为什么这会真的咬人

`StairSpec()` 的默认 `n_up=0, n_down=0` 意味着**地形是平的**，而 `step_height=0.05` 纯粹是 dataclass 的字段默认值。两者被拼在了一起：

* `env.stair.step_height` 同时喂给 **观测第 45 维**（策略的唯一地形线索）与 **`foot_lift` 的目标高**（`reward.py:278-280` 的 `max(foot_lift_target, ratio * step_height)`）⇒ 影响一个 0.05 的恒定观测值，以及 `lift_target` 从 0.08 变成 0.07（无害）。
* 真正的问题是**语义**：本仓的全部设计前提是「`step_height` 是策略唯一被允许知道的地形信息」（`env.py:5-6`、`agent.md §8.6`、`ObsConfig` docstring）。一个**凭空出现的非零值**直接违反这个前提，而且它是**静默**的 —— 没有任何警告、没有断言，观测里的 0.05 看起来完全合法。

正式路径确实都显式传参（`training.make_vec_env(..., stair=...)`、`play.stair_from_height()`、`config_to_stages()` 的 `flat_spec(step_height=0.0)`），**所以训练不受影响**。受影响的是**任何直接 `Go2StairsEnv(cfg=...)` 的研究代码**：写个 10 行的探针脚本、跑个平地 Ablation，就会得到「观测里有 5 cm 台阶、地形却是平的」这种自相矛盾的组合，而它不会报错。

**已确认的真实受害者（仓库内）**：`scripts/teleop.py:77` 与 `scripts/cmd_demo.py:152` 都用 `flat_spec_for(...)` / `flat_spec(tread=0.50, width=2.0, start_x=1.50)` 造平地且**不传 `step_height`**，而 `terrain.py:213-216` 的 `flat_spec()` 只 `setdefault` 了 `n_up`/`n_down`（`step_height` 保留 `StairSpec` 的类默认 `0.05`）⇒ 这两个脚本的**每一次平地演示**都在给策略喂「这里有 5 cm 台阶」，与它们自己在 HUD 里打印的「地形 平地（无台阶）」直接矛盾。`env.py:559-562` 是无条件的 `if obs_cfg.step_height: parts.append(np.array([float(self.stair.step_height)]))`，所以只要 `obs.step_height` 开着（`ObsConfig` 默认 True、`config.yaml:81` 也是 true），这个 5 cm 就一定会进观测。

### 建议（三选一，按侵入性排序）

1. **最小、最稳**：把 `StairSpec.step_height` 的默认值改成 `0.0`，并让 `demo_stairs_spec()` 保留 `0.05` 的显式默认（它本来就是「造楼梯」的函数，给个非零默认是合理的）。这样「裸 `StairSpec()`」= 平地 = `step_height=0`，语义自洽。
2. `Go2StairsEnv.__init__` 里加一致性断言：`if stair is None and cfg.obs.step_height: warn/raise`（或直接按 `flat_spec()` 造地形）。
3. 在 `_obs_frame()` 里把 `step_height` 的取值来源改成「本回合实际要爬的高度」= `stair.crest_height if stair.n_up else 0.0`。**不推荐** —— 它会改变观测数值，需要等价性证明，而且 `crest_height` 与 `step_height` 在 `n_down>0` 时不等。

我推荐 **1**：改一个默认值，行为在正式路径上逐位不变。

---

## E-2（medium）`set_command_now()` 刷新了 `_hist[-1]`，但调用方手里那份 `obs` 仍是旧的 —— 三个脚本里有两个没补

`env.py:392-419` 的 `set_command_now()` 里有一段**刻意**的刷新：

```python
# ⭐ 刷新「已经算好、还没被取走」的那一帧观测。
if self._hist and self._hist[-1].shape == self._obs_frame().shape:
    self._hist[-1] = self._obs_frame()
```

动机在注释里写得很清楚（`:409-416`）：外部控制器是在**两次 `step()` 之间**下指令的，`step()` 结束时已经把上一帧压进 `_hist`，不刷新就会有一个控制步（20 ms）的指令滞后。指令是外生输入，改它不破坏马尔可夫性。

**但这段刷新只改了 `_hist` 的尾部，改不了调用方手里那个已经返回的 `numpy` 数组**。于是：

| 脚本 | 写法 | 后果 |
| --- | --- | --- |
| `scripts/play.py` `run_episode` | `set_command_now(...)` 之后 **重新取** `obs = env._observation()`（注释也写了原因） | 正确：动作按新指令算 |
| `scripts/teleop.py` `run_teleop` | `set_command_now(...)` 之后**直接** `env.step(runner.predict(obs))` | 每回合第 0 步的动作仍按 reset 时的**随机**指令算 |
| `scripts/cmd_demo.py` `run_demo` | 同样「先设指令、不重取 obs」 | 每段第一帧按上一段的指令算 |

实测确认刷新本身生效（`.tmp/zz_audit_probe2.py`）：`set_command_now(vy=0.3, yaw_rate=0.5)` 之后 `_hist[-1][46:48]` 从 `[0, 0]` 变成 `[0.3, 0.5]`。

### 为什么要单独列出来

这是**同一条 API 有两种自洽但不同的用法**导致的静默偏差：

* 「我改完指令，下一帧观测自然是新的」—— 直觉（也是 `_hist[-1]` 被刷新的用意）；
* 实际：`obs` 是 `numpy` 数组，是**值语义**，刷新 `_hist` 改不到它。

`play.py` 的作者显然踩过一次（那里留了注释和重取），`teleop.py` / `cmd_demo.py` 没跟上。偏差只有**一步**，对遥操作的观感几乎无影响，但对 `cmd_demo.py` 的**斜率/相关系数统计**（它的整个卖点就是量跟踪相关性）是污染：每段第一帧用的是上一段的指令，而段边界正是它想量清楚的地方。

### 建议

* 短期：给 `set_command_now()` 的 docstring 加一句 **「本方法只刷新 `_hist` 尾部；调用方若已持有 `obs` 副本，必须重新 `env._observation()`」**，并在 `teleop.py` / `cmd_demo.py` 补上与 `play.py` 相同的重取。
* 更稳：让 `set_command_now()` **返回新观测**（或新增 `set_command_now_and_obs()`），把「重取」变成类型系统强制的事，而不是靠注释。

---

## E-3（low）`_obs_frame()` 在一次调用里被算两遍

`:417-418`：

```python
if self._hist and self._hist[-1].shape == self._obs_frame().shape:
    self._hist[-1] = self._obs_frame()
```

`_obs_frame()` 做的是 `np.concatenate([joint_pos, joint_vel, base_rpy, base_ang_vel_body, _last_action, base_lin_vel_body, [step_height], command])`，其中 `base_rpy`（四元数→欧拉角）、`base_ang_vel_body`（`R^T @ ω`）、`base_lin_vel_body`（`R^T @ v`）都各构造一次旋转矩阵。一个控制步一次调用而已，**性能无关紧要**，真正的成本是**正确性**：

* 这个 `if` 的判据是「形状相等」。把同一个表达式算两遍，如果两者之间有任何状态变化（今天的 `_obs_frame()` 是幂等的 —— 实测连续两次相等），条件就失去意义；
* 更实际的是**可读性**：这段代码要表达的是「把尾部那帧换成按新指令重算的帧」，写成一次求值更直接。

### 建议

```python
if self._hist:
    frame = self._obs_frame()
    if self._hist[-1].shape == frame.shape:
        self._hist[-1] = frame
```

---

## E-4（medium）观测维数恒定这条契约**没有任何机器守卫**

契约 4 要求「整个课程里观测维数恒定（PPO 网络与 `RolloutBuffer` 不可变）」。当前实现依赖：

* `ObsConfig` 的 6 个布尔开关在 `EnvConfig` 里是普通字段，**可以被 `setattr` 改**；
* `observation_space = Box(-inf, inf, (self.cfg.obs.size,))`（`env.py:249-251`）在 `__init__` 里按当时的 `cfg.obs` 定死，**之后不再校验**；
* `build_ppo()` 从 `venv.observation_space` 取维度建网。

所以「维数恒定」是**行为约定**，不是**被检查的不变量**。三条具体的裂缝：

1. `ObsConfig.base_frame_size: int = 42` 是**可写字段**，而 `_obs_frame()`（`:543-549`）无条件拼前 42 维（`joint_pos` 12 + `joint_vel` 12 + `base_rpy` 3 + `base_ang_vel_body` 3 + `_last_action` 12）。把它改成 40 或 50，`cfg.obs.size` 与 `env._observation().shape` 会**当场不一致**，而报错会迟到 PPO 的 `ValueError: Observation spaces do not match`（那次报错也未必指向真正的原因）。
2. `train.py` 用 `copy.deepcopy(self.cfg)` 给每一级建环境，**课程只改 `command.vy_max/yaw_max` 与 `reward.*`** ⇒ 今天没有路径会改观测开关。但 `set_commands()` 的 docstring（`:362-364`）自己承认「训练中途改动维数会让 PPO 的网络与 `RolloutBuffer` 对不上」，却没有代码阻止它。
3. 权重加载侧有**启发式反推**（`policy.py:289` 的 `_obs_config_for_dim`），它枚举 `base_lin_vel_body/gait_phase/step_height/command/vel_filter` 五个开关在 `history_len=10` 下的 32 种组合去凑总维度。多组组合能给出同一 `size` 时它**取枚举顺序里的第一个**（注释说这是刻意的：`base_lin_vel_body` 与 `command` 默认开，排前面）。对 48 维（本仓真正在用的维度）它在 `history_len=10` 下**一个都凑不出**，于是走 `SystemExit` 分支 —— 行为是对的，但代价是「`summary.json` 丢了就彻底加载不了 48 维权重」。

### 建议

在 `Go2StairsEnv.__init__` 末尾加一条**自检**（开发期几乎零成本、发布期可选关闭）：

```python
frame = self._obs_frame()           # reset() 之外也能算：dependencies 都是 data 派生
expected = self.cfg.obs.frame_size
if frame.shape[0] != expected:
    raise ValueError(
        f"ObsConfig.frame_size={expected} 与 _obs_frame() 实际输出 {frame.shape[0]} 不一致；"
        f"base_frame_size 应为 42（joint_pos 12 + joint_vel 12 + base_rpy 3 + "
        f"base_ang_vel_body 3 + last_action 12）"
    )
```

并在 `set_commands()` / 任何可能被误用的入口上写明「不得改观测开关」。`_obs_frame()` 在 `__init__` 阶段是可算的（`self._last_action` 已在 `:259` 初始化，其余都来自 `data`），所以这条断言不会引入初始化顺序问题。

顺带：`policy.py:289` 的 `_obs_config_for_dim(total, history_len=10)` 建议把 `history_len` 也纳入枚举（`history_len ∈ {1, 10}` 足够覆盖本仓），并把默认值 `10` 写进 docstring —— 现在它是隐式的。

---

## E-5（medium）`EnvConfig.reward_overrides` 是个没有任何读者的字段，而它的 docstring 说「`config_to_env` 会填这里」

`env.py:211-214`：

```python
# 逐级奖励覆盖：``config.yaml`` 的 ``curriculum.reward_overrides`` 会按阶段
# 填这里，``config_to_env`` 与 ``set_reward_overrides()`` 把它叠到 ``reward`` 上。
# 默认空 ⇒ 与没有这个机制时逐位不变。
reward_overrides: dict[str, float] = field(default_factory=dict)
```

实测：`config_to_env()` **从不写这个字段**（`config.py` 里 `reward_overrides` 的 6 处引用全部在 `config_to_stages()` / `_reward_override_table()` 里，写的是 `StageSpec.reward_overrides`；`config_to_env` 里没有这一行）。`env.py` 里对该名的其余 5 处命中**全是注释与 docstring**。

真正生效的链路是：

```
config.yaml: curriculum.reward_overrides
  → config_to_stages() → StageSpec.reward_overrides           (config.py:318)
  → scripts/train.py:200      venv.env_method("set_reward_overrides", dict(asdict(基线)))
  → scripts/train.py:203      venv.env_method("set_reward_overrides", dict(stage.reward_overrides))
  → robotdog/env.py:687       set_reward_overrides()  → setattr(cfg.reward, ...)
```

也就是说 `EnvConfig.reward_overrides` 是一个**与生效机制同名、同语义、但从未被读写的影子字段**。

### 风险

* **误读**：任何人读 `EnvConfig` 会以为「给这个字段赋值就能改奖励」，而它毫无作用；更糟的是 `_rebuild()`（`policy.py:78`）会把它从 `summary.json` 原样恢复进来 —— 于是 `runs/*/summary.json` 里能看到 `"reward_overrides": {...}`，看起来像生效过的证据。**实际写进 `summary.json` 的是 `asdict(基线 cfg)`**，所以它必然是 `{}`，但「看到键就以为有值」这个坑仍然在。
* **文档漂移**：`docs/robotdog/env.md` 按本仓传统是逐字段对齐的，这个字段的说明会跟着一起错。

### 建议

二选一：

* **A（推荐）**：删掉 `EnvConfig.reward_overrides`，把 `env.py:211-213` 的注释改成「逐级奖励覆盖不经过 `EnvConfig`；入口是 `StageSpec.reward_overrides` → `scripts/train.py` → `set_reward_overrides()`」。同时 `policy.py` 的 `_TUPLE_FIELDS` / `_rebuild()` 不受影响（它按字段名过滤）。
* **B**：让 `config_to_env()` 真的填它，并让 `__init__` 在 `reward_overrides` 非空时自动叠加（等于把 `train.py:200-203` 的两步合成一步）。**这要小心**：`set_reward_overrides` 的 docstring（`:695-701`）明确论证了「必须从基线叠、每级要有自己的 cfg 副本」，把叠加藏进 `__init__` 会让「同一进程里多级共享 cfg」的隐患重新变得不可见。

我推荐 A —— 保留现在这条**显式两步**的叠加链路，它是对的。

---

## E-6（low）`_start_terrain_height` 是死字段

`env.py:261` 初始化、`:852` 在 `reset()` 里赋值，**没有任何读者**（全仓 grep 只有这两处）。它曾是 `stair_progress` 的参照（「下方地形高 − 回合起点高」），改成按 `_peak_terrain_height` 结算（`reward.py:333-336`）之后就废弃了，但 field 与赋值都留着。

顺带说明为什么现在这样是对的：`reset()` 把 `_peak_terrain_height` 初始化成 **`terrain_height_under_base` 而不是 0**（`:853`）。`spawn_back=0.55` 意味着机器人出生在第一个立面之前、地形高为 0，所以两者今天数值相同；但如果将来 `spawn_back` 变小到出生就在台阶上，这个初始化仍然保证 **不会白拿出生点的爬升分** —— 这是刻意的，值得在 `docs/robotdog/env.md` 里写成「初始化为什么不是 0」。

**建议**：删掉 `_start_terrain_height`（两行）。

---

## E-7（low）其他命名 / 契约细节

* **`act_low` / `act_high` 的命名误导**（详见 `02-reward.md` 的 R-2）：`env.py:244-245` 取自 `joint_ranges`（弧度），名字却像 `ctrl_ranges`（N·m）。`:911` 用它截位置目标（正确），`reward.py:317` 用它算关节越界（正确）——**两处都对，坏在名字**。
* **`terminate_height_ref` 是「静默 fallback」而不是「硬校验」**：`_fall_reference_height()`（`:806-811`）用 `mode = str(...).strip().lower()`，只有 `support_mean` 与 `support` 被识别，**其他任何字符串都退回 `terrain_height_under_base`**。对比 `render()`（`:1008-1009`）对未知相机名直接 `ValueError`，两种风格不一致。`config.yaml` 用 `"support"`，没问题；但拼错成 `"suport"` 会静默启用「机身下方地形」这一条 —— 而代码自己的注释（`:185`、`:802-804` 与 `EnvConfig.terminate_height_ref` 的说明）明确说这条会产生**幽灵摔倒**。建议改成白名单校验 + `ValueError`。
* **`_contact_flags()` 与 `_foot_contact_mask()` 各自扫一遍 `data.ncon`**（`:714-750`、`:598-617`），每步两次 O(ncon) 循环 + 两次集合构造。50 Hz × 8 env 完全无所谓，但它是那种「以后加了接触传感器会变成热点」的位置，且两个函数里的 `terrain` 集合构造完全重复。可合并为一次扫描（注意 `_foot_contact_mask` 在 `reset()` 里也被调用，合并时要保留两个入口）。
* **`ObsConfig.__post_init__` 只挡 `base_height`**：`gait_phase=True`（观测里放步态时钟的 sin/cos，`env.py:556-558`）在真机上需要步态时钟本身就是外部驱动量 —— 本仓有 `gait_prior`，所以这一项**有真机来源**，可以放行。但 `skill` 契约 3 里「步态相位」被列为允许项，代价是要么策略本来就跑开环时钟、要么真机也能同步同一个时钟。当前 `config.yaml` 关着它，**建议在 `docs/robotdog/env.md` 的信息来源表里显式写明它的真机来源**（现在的 docstring 只解释了 `step_height` 与 `command`）。
* **`reset()` 里把同一帧复制 `history_len` 次**（`:877-879`）：`frame = self._obs_frame()` 然后 `for _ in range(history_len): self._hist.append(frame.copy())`。这是**正确的**（回合起点没有任何历史，重复当前帧是标准做法），但意味着 `history_len>1` 时策略在第一个 `history_len-1` 步看到的「历史」是伪造的平稳段。`config.yaml` 用 `history_len: 1` 所以完全不受影响；如果将来恢复多帧堆叠，这里值得写一次注释说明「前 N 帧是复制而非真实历史」。
* **`_peak_swing_clearance` 的清零时机**（`:626-644`）：`landed` 帧**保留**旧峰值、其余接触帧清零，注释（`:633-636`）说明这是踩过的坑（一接触就清零会让地面帧读到比真实峰值低 0.01~0.03 m 的值）。实现是
  ```python
  self._peak_swing_clearance = np.where(
      landed, self._peak_swing_clearance,
      np.where(foot_contact, 0.0,
               np.maximum(self._peak_swing_clearance, clearance)),
  )
  ```
  逐分支核对：`landed` → 保留；`not landed and contact` → 0；`not contact` → 取 max。**与注释一致，无问题。**

---

## 已核对无问题的项

* **契约 3（观测真值）**：`_obs_frame()`（`:541-578`）的全部 8 个块 —— `joint_pos`/`joint_vel`（编码器）、`base_rpy`/`base_ang_vel_body`（IMU）、`_last_action`（上一步动作）、`base_lin_vel_body`（本体感知估计，注释 `:467-474` 说明为什么用机体系）、`step_height`（标量任务参数）、`command[1:]`（外部指令）、`filtered_*`（同一批量的低通）—— **没有一项含地形/接触真值**。`foot_clearance`（`:522-532`）与 `_support_height()`（`:771-783`）都明确标注「特权信息，只进奖励/终止」，且 `_obs_frame()` 里确实没有它们的读取。
* **契约 5（唯一动作管线）**：`_target_from_action()`（`:893-911`）只有一条路径 `target = home + gait_gain*gait.radians(t) + action_scale*action` 再 `np.clip(target, act_low, act_high)`；`step()` 开头 `action = np.asarray(action, float64).clip(-1, 1)`（`:914`）把策略输出限幅到 ±1；`"absolute"` 模式已删（`:904-906` 的注释）。**没有第二条管线。**
* **契约 6（幅值解耦）**：`gait.action_scale`（先验自带，实测 artifacts 加载出 1.0）与 `cfg.action_scale`（残差权限，0.30）是两个独立字段（`:900-902` 的注释论证了「共用一个尺度会让探索噪声随先验缩放」）。**解耦成立。**
* **契约 7（终止）**：`_terminated()`（`:813-823`）只由三条触发 —— `base_pos[2] - _fall_reference_height() < terminate_min_height`、`|roll|` 或 `|pitch| > terminate_max_tilt`、`_step_count >= _max_steps`（截断）。`_contact_fatal()`（`:752-769`）**恒返回 `False`**，且里面只维护 `_leg_contact_streak` 供诊断。**没有「连续 N 次接触判死」去抖，躯干接触不终止回合**（代价由 `body_collision` 逐步承担）。与契约一致。
* **`truncated` 与 `terminated` 的组合**：`step()` 里 `_success_reached()` 命中时（`:951-955`）若尚未摔倒则 `terminated=True, truncated=False` —— 明确避免「成功与超时同时为真」。**这一处是刻意的，无问题。**
* **`_success_reached()`**（`:435-446`）：楼梯 = `base_pos[0] > up_end_x` **且** `terrain_height_under_base >= crest_height - 1e-9`（两个条件缺一不可，防止「从旁边绕过立面」）；平地分两支，`flat_success_needs_progress=False` 时是「撑满整个回合」并且**带 `_step_count >= _max_steps - 1` 条件**（`:443-445` 的注释说明不带这个条件会第 1 步就白拿 success 奖励）。**逻辑严谨。**
* **`base_lin_vel_body` / `yaw_rate` 的帧约定**：`docstring`（`:15-21`）说明 `qvel[0:3]` 是世界系、`qvel[3:6]` 是机体系；`base_lin_vel_body` 用 `R^T @ qvel[0:3]`、`yaw_rate` 用 `(R^T @ qvel[3:6])[2]`。**换算正确**，且 `yaw_rate` 的 docstring（`:504-517`）给出了「为什么不用世界系 z 角速度」的定量理由（`tan(pitch)` 乘积项贡献 ≈0.28 rad/s，与指令量程 ±0.70 同阶）。
* **`_track_swing_peak` 的触发顺序**：`step()` 里 `_track_swing_peak` → `_update_probe_state` → `_terminated` → 低通推进 → `_compute_reward`（`:937-960`）。`_advance_foot_air_time` 在奖励**之后**（`:961`），这保证 `air_time` 项读到的是「本步之前累积的腾空时间」、恰好在落地步结算（`reward.py:293-295` 的注释）。**顺序正确。**
* **观测的 dtype**：`_obs_frame()` 返回 `float64`，`_observation()` 返回 `float32`（`:578`、`:581`），与 `observation_space` 的 `dtype=np.float32` 一致。`_hist` 里存的是 float64，`np.concatenate` 后再降位。**一致。**
