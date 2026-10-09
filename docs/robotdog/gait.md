# `robotdog/gait.py` —— trot 步态先验

## 1. 一句话定位

环境层的**离线搜索出来的固定开环 trot 轨迹**（一个 `@dataclass GaitParams` + 一个
`load_gait(path)` 加载器），给残差 RL 提供一个「已经在走」的起点。它只产生关节角偏移
（弧度），不含任何控制器、不闭环、不知道地形。

## 2. 职责边界

**做**：

* 用 `freq / amp_th / amp_ca / ph_th / ph_ca / off_th / off_ca / calf_shape /
  leg_phase / action_scale` 参数化一条周期性的**归一化动作单位**轨迹；
* `actions(t)` 给出 `(12,)` 归一化偏移（顺序 = 执行器顺序），`radians(t)` 乘上
  `action_scale` 得到弧度偏移，`targets(t, home)` 再叠加 home 角；
* JSON 读写（`to_dict/save/from_dict/from_json`）、搜索结果的解析
  （`from_search_json`）、三个内置先验（`default_trot/fast_trot/static`）；
* `load_gait()` 的统一入口（`None` → 默认先验；`.json` → 普通 gait 或搜索结果）。

**不做**：

* **不含闭环**：不读 qpos/qvel、不做平衡、不知道台阶，相位纯由外部累加的时间 `t` 决定；
* 不做 PD 控制、不写 `mjData`、不碰 MuJoCo；
* 不决定「先验开不开」——`gait_prior` / `gait` / `gait_gain` 都是 `EnvConfig` 的字段，
  `_gait_action()`（`robotdog/env.py:697-705`）才决定要不要用它；
* 不决定观测里有没有步态时钟——那是 `ObsConfig.gait_phase`（`robotdog/env.py:69`）；
* **不做相位随机化**：`gait_phase_randomize` 字段已从 `EnvConfig` 删除，
  `reset()` 里 `_gait_t = _gait_phase = 0.0` 是固定值（`robotdog/env.py:651-695`）。

## 3. 关键设计决策与理由

### D1 —— 为什么要有一条先验，而不是从零学步态

因为**从零学不会**，原因是平台期而不是权重 bug：随机关节抖动不产生净推进，所以
前向速度奖励在出现**协调的周期步态**之前一直是平的；350k 参数的 MLP 以 50 Hz 采样
不可能撞上那条轨迹。两次标定（平地 120k 与 150k 步）都停在 0% 成功率、约 0.02 m 位移，
学到的策略比零动作站桩**更差**（466 vs 496 每回合 return）。此外本仓只能用
`DummyVecEnv`（沙箱禁止 Windows 命名管道，`SubprocVecEnv` 起不来，`train.md` §1），
8 个环境串行，样本效率也不支持从零探索。

因此改成残差 RL（Johannink et al., *Residual Reinforcement Learning*, 2019）：
先离线搜一条固定开环 trot，策略只学**残差**。好处是探索空间从「怎么产生周期步态」
缩到「怎么改这条步态」，用户要求的观测/动作接口完全不变（动作仍是 12 维关节目标偏移，
输出仍是关节空间位置指令）。

**被否决的方案**：把先验做成一个隐藏的闭环控制器（如相位驱动的足端 IK）。否决原因——
那条路会让「策略学到什么」不可解释，而且先验变成不可检查的黑盒；本模块的先验是一条
可打印、可存 JSON、可离线重现的**轨迹**。`EnvConfig.gait_prior` 默认 `False`
（`robotdog/env.py:148`），所以「纯残差、无先验」的朴素形式依然可用；
而 `scripts/train.py` 的 `--gait-prior` 默认 `True`，训练侧默认是**开**先验的。

### D2 —— 幅度单位用「归一化动作单位」，不用弧度

`amp_*` / `off_*` 都是**与 `action` 同一个单位**，乘 `action_scale` 才变弧度。
理由：这样 `action_scale` 变了先验仍然成立，且归一化域里有一条可检查的不变量
`|off| + amp <= 1`（`normalize()` 强制执行，见 D5）。若直接用弧度写死，
`action_scale` 一改先验就跟着变，JSON 里的数字会失去意义。

### D3 —— 先验幅度 `gait.action_scale` 与策略权限 `EnvConfig.action_scale` 刻意分开

`radians()` 的注释（`robotdog/gait.py:133-142`）写得很直白：环境用 `radians()` 而**不是**
`actions()`，就是为了让先验的幅度独立于 `EnvConfig.action_scale`（后者只有 `0.30`，
只控制**策略残差**的权限）。混用一个 scale 会让策略的探索噪声随先验一起放大——
实测**2 s 内就摔**。所以动作管线是

```
target = home + gait_gain * gait.radians(t) + action_scale * action
```

`gait.action_scale` 说「先验动作多大」（它是搜索出来的一部分），
`cfg.action_scale` 说「策略能偏离多少」（`robotdog/env.py:707-728`）。
`gait_gain` 默认 `1.0` 且**没有 CLI 开关**，只能改 `cfg`。

**被否决的方案**：共用一个 `action_scale`（先验与策略同权）。否决原因就是上面那条实测：
探索噪声被先验幅度乘了一遍，机器人立刻倒。

### D4 —— `calf_shape="sin"` + `leg_phase=(0, π, π, 0)` 就是对角小跑

`LEG_ORDER = ("FL", "FR", "RL", "RR")`，`TROT_PHASE = (0.0, π, π, 0.0)`：FL 与 RR
同相、FR 与 RL 同相、两对相差 180°，即**对角步态**。`leg_phase` 是**加在采样相位上的
弧度偏移**（`leg_phases(t) = 2π·freq·t + leg_phase`），与 `ph_th/ph_ca` 同量纲，
不是「周期的小数比例」——写 `0.25` 得到的是 0.25 rad，不是 1/4 个周期。

`calf_shape` 保留了三个可选波形（`CALF_SHAPES = ("sin", "dip", "half")`），因为搜索器
会扫它们；出货先验用的是 `"sin"`（`artifacts/trot_stairs.json` 里写成整数 `0`，由
`from_dict()` 映射回 `CALF_SHAPES[0]`）。另外 `actions()` 的髋关节列**恒为 0**
（只有 `a[:, 1] = thigh; a[:, 2] = calf`），也就是先验从不做外展——髋一直停在 home 角。

**被否决的方案**：只用 `sin`、去掉 `"dip"` / `"half"`。否决原因——搜索结果里
`calf_shape` 是搜索维度之一，删掉就没有可比的历史结果；保留三个分支的成本只是一个
`if`，且非法值会明确抛 `ValueError`。

### D5 —— `normalize()` 兜底，而不是在加载时报错

`normalize()` 把 `freq` 夹到 `[0, 10]`、四个幅度/偏移夹到 `[-1, 1]`，然后对
`(amp_th, off_th)` 与 `(amp_ca, off_ca)` 各查一次 `|off| + amp > 1`：超了就整体乘
`s = 1 / max(|off| + amp, 1e-9)`，**同时缩 offset 与 amplitude**（保持波形比例，只是变小）。
相位用 `atan2(sin, cos)` 折回 `(-π, π]`。理由：搜索出来的 JSON 可能落在物理无意义的
角落，与其拒绝加载（用户拿到一个打不开的 artifact）不如就地收缩到合法区域——先验只要
「大致能走」就能当起点。硬约束（`calf_shape` 非法、`leg_phase` 长度不为 4）仍然抛异常。

`action_scale` **不在夹取范围内**：`normalize()` 完全不碰它，所以 JSON 里的 `1.0` 会原样留下。

### D6 —— 出货先验是搜索结果，不是 `default_trot()`

出货命令用 `--gait-file artifacts\trot_stairs.json`（`scripts/train.py:237` 的默认值），
`load_gait()` 看到文件里有 `"top"` 键就走 `from_search_json(path, rank=0)`，
并把条目上的 `action_scale` 抬到顶层：所以**先验幅度是 1.00 rad**，不是 dataclass
默认的 `0.70`，也不是 `_FALLBACK_TROT` 的 `0.90`。

`GaitParams.default_trot()` 是另一条（`_FALLBACK_TROT`，6.0 Hz / scale 0.90）：
它只在 `load_gait(None)` 时兜底，而 `scripts/train.py` **总会**传一个路径
（`--gait-file` 默认非 None），所以训练路径上 `default_trot()` 其实用不到。
两者刻意不是同一条：`default_trot()` 是「速度目标 0.55 m/s 那一档搜索的胜者」，
出货 JSON 是「**stairs** 目标重新搜出来的胜者」。

## 4. 公开 API

### 模块级常量

```python
LEG_ORDER = ("FL", "FR", "RL", "RR")
TROT_PHASE = (0.0, math.pi, math.pi, 0.0)
CALF_SHAPES = ("sin", "dip", "half")
_FALLBACK_TROT: dict   # default_trot() 的来源，不是公开名
FAST_TROT: dict        # fast_trot() 的来源
```

### `@dataclass class GaitParams`

字段（名字 / 类型 / 默认值 / 含义）：

| 字段 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `freq` | `float` | `2.0` | 步态周期频率 [Hz] |
| `amp_th` | `float` | `0.0` | 大腿摆动幅度（归一化单位） |
| `amp_ca` | `float` | `0.0` | 小腿摆动幅度（归一化单位） |
| `ph_th` | `float` | `0.0` | 大腿相位偏移 [rad] |
| `ph_ca` | `float` | `0.0` | 小腿相位偏移 [rad] |
| `off_th` | `float` | `0.0` | 大腿常值偏移 |
| `off_ca` | `float` | `0.0` | 小腿常值偏移 |
| `calf_shape` | `str` | `"sin"` | `sin` / `dip` / `half` |
| `action_scale` | `float` | `0.70` | 归一化单位 → 弧度（rad per unit） |
| `leg_phase` | `tuple[float, float, float, float]` | `field(default_factory=lambda: tuple(TROT_PHASE))` | 四条腿的相位偏移 [rad] |

方法：

* `_calf_wave(x: np.ndarray) -> np.ndarray` —— 私有。`"sin"` → `np.sin(x)`；
  `"dip"` → `-0.5 * (1 - cos x)`（恒 ≤ 0）；`"half"` → `-np.maximum(0, np.sin(x))`
  （只取负半周）。未知形状抛 `ValueError(f"unknown calf_shape {self.calf_shape!r}")`。
* `leg_phases(t: float) -> np.ndarray` —— `2π·freq·t + np.asarray(leg_phase)`，
  形状 `(4,)`，`float64`。
* `actions(t: float) -> np.ndarray` —— `(12,)`，执行器顺序，**归一化单位**。
  髋列恒 0，大腿 `off_th + amp_th·sin(p + ph_th)`，小腿
  `off_ca + amp_ca·_calf_wave(p + ph_ca)`，最后 `np.clip(a, -1, 1).ravel()`。
* `radians(t: float) -> np.ndarray` —— **环境真正调用的那个**：`action_scale * actions(t)`，
  返回弧度偏移 `(12,)`。
* `targets(t: float, home: np.ndarray) -> np.ndarray` —— `home + self.radians(t)`。
* `@property period -> float` —— `1.0 / freq`；`freq <= 0` 时返回 `float("inf")`。
* `normalize() -> GaitParams` —— **就地**夹取（见 §3 D5），返回 `self`。
  抛 `ValueError(f"calf_shape must be one of {CALF_SHAPES}")`、
  `ValueError("leg_phase must have 4 entries (FL, FR, RL, RR)")`。
  注意先夹 `calf_shape` 合法性再夹 `leg_phase` 长度，且 `leg_phase` 会被转成 `tuple(float, ...)`。
* `@property is_static -> bool` —— `amp_th == 0.0 and amp_ca == 0.0`（默认构造即为 True）。
* `to_dict() -> dict` —— `asdict()` 且 `leg_phase` 转 `list`（便于 json 序列化）。
* `save(path: str | Path) -> Path` —— 建父目录，写 `json.dumps(..., indent=2)`，
  `encoding="utf-8"`，返回 `Path`。
* `@classmethod from_dict(d: dict) -> GaitParams` —— **按已知字段名过滤**后构造，
  再 `.normalize()`；`leg_phase` 转 tuple；`calf_shape` 是 `int` 时映射
  `CALF_SHAPES[int(...)]`。
* `@classmethod from_json(path) -> GaitParams` —— 读 JSON 后走 `from_dict`。
* `@classmethod from_search_json(path, rank: int = 0) -> GaitParams` —— 取
  `data["top"][rank]`，用 `entry["gait"]` 加上顶层的 `entry["action_scale"]`（`float(...)`）。
  `top` 为空抛 `ValueError(f"{path} has no 'top' entries")`；`rank` 越界是裸
  `IndexError`。
* `@classmethod default_trot() -> GaitParams` —— `_FALLBACK_TROT`（平地 0.550 m/s 那档）。
* `@classmethod fast_trot() -> GaitParams` —— `FAST_TROT`（1.702 m/s，距离目标胜者）。
* `@classmethod static() -> GaitParams` —— `cls(freq=0.0, amp_th=0.0, amp_ca=0.0).normalize()`，
  等价于「没有先验」；注意它的 `action_scale` 仍是默认 `0.70`，`period` 是 `inf`。
* `summary() -> str` —— 一行人类可读摘要，格式见 §6。

### `def load_gait(path: str | Path | None) -> GaitParams`

* `path is None` → `GaitParams.default_trot()`；
* 文件不存在 → `FileNotFoundError(f"gait file not found: {path}")`；
* JSON 里有 `"top"` 键 → `GaitParams.from_search_json(path)`（`rank=0`）；
* 否则 → `GaitParams.from_dict(data)`。

## 5. 内部结构与数据流

**构造默认值（`EnvConfig` 实例化时就发生）**：

```
EnvConfig()
  → field(default_factory=GaitParams.default_trot)          # robotdog/env.py:149
    → GaitParams.from_dict(dict(_FALLBACK_TROT))            # robotdog/gait.py:212-214
      → cls(**kw).normalize()
```

**控制步（每个 `step()` 一次）**：

```
Go2StairsEnv.step(action)                                    # robotdog/env.py:730
  → self._gait_phase = self._gait_t                          # env.py:735（本步实际用的相位）
  → self._target_from_action(action)                         # env.py:707
    → self._gait_action()                                    # env.py:697
       若 not cfg.gait_prior 或 cfg.gait_gain == 0.0 → zeros(12)
       否则 → cfg.gait_gain * cfg.gait.radians(self._gait_phase)
              → GaitParams.radians(t)                        # gait.py:133
                → self.action_scale * self.actions(t)
                  → GaitParams.actions(t)                    # gait.py:119
                    → GaitParams.leg_phases(t)               # gait.py:116
                    → GaitParams._calf_wave(p + ph_ca)       # gait.py:107
  → target = home + _gait_action() + cfg.action_scale * action
  → np.clip(target, act_low, act_high)                       # env.py:725
  → self._gait_t += cfg.control_dt                           # env.py:739（20 ms）
```

**观测里的步态时钟**（`ObsConfig.gait_phase=True`，每帧 2 维）：

```
_obs_frame()                                                 # robotdog/env.py:408-410
  → theta = 2π * cfg.gait.freq * self._gait_phase
  → np.array([np.sin(theta), np.cos(theta)])
```

**CLI 加载**：

```
scripts/train.py --gait-prior [--gait-file PATH]             # scripts/train.py:235-237, 194-196
  → load_gait(PATH)                                          # robotdog/gait.py:233
      → GaitParams.from_search_json(PATH)                    # 有 "top" 键时
        → from_dict(entry["gait"] | action_scale=entry["action_scale"])
  → cfg.gait_prior = True; cfg.gait = gait
  → print(f"[步态] {cfg.gait.summary()}")                     # scripts/train.py:278
```

## 6. 关键常量与默认值

**dataclass 默认值**：`freq=2.0`、`amp_th=0.0`、`amp_ca=0.0`、`ph_th=0.0`、`ph_ca=0.0`、
`off_th=0.0`、`off_ca=0.0`、`calf_shape="sin"`、`action_scale=0.70`、
`leg_phase=(0.0, 3.141592653589793, 3.141592653589793, 0.0)`。

**`normalize()` 的边界**：`freq ∈ [0.0, 10.0]`；`amp_th/amp_ca/off_th/off_ca ∈ [-1.0, 1.0]`；
`|off| + amp <= 1.0`（超出时按 `s = 1 / max(|off| + amp, 1e-9)` 同缩）；
相位折回 `(-π, π]`；`leg_phase` 必须 4 项；`action_scale` **不夹**。

**`_FALLBACK_TROT`（`default_trot()`，平地 0.550 m/s 档）**：

| 键 | 值 |
| --- | --- |
| `freq` | `6.0` |
| `amp_th` | `0.24914502679724` |
| `amp_ca` | `0.5518579715236906` |
| `ph_th` | `1.6754541487333945` |
| `ph_ca` | `-2.146505768979483` |
| `off_th` | `0.25539290883226595` |
| `off_ca` | `0.09025271647195035` |
| `calf_shape` | `"sin"` |
| `action_scale` | `0.90` |

`summary()` → `6.00 Hz, ±0.249 thigh / ±0.552 calf (norm. units), shape=sin, scale=0.90 rad, period=167 ms`

**`FAST_TROT`（`fast_trot()`，1.702 m/s 距离目标胜者）**：

| 键 | 值 |
| --- | --- |
| `freq` | `4.0` |
| `amp_th` | `0.6093323116494958` |
| `amp_ca` | `0.9305071165813157` |
| `ph_th` | `1.533544334735976` |
| `ph_ca` | `2.1658126114417198` |
| `off_th` | `0.3906676883505043` |
| `off_ca` | `0.0694928834186843` |
| `calf_shape` | `"sin"` |
| `action_scale` | `0.70` |

`FAST_TROT` 与 `fast_trot()` 在仓内**没有任何调用点**（`grep FAST_TROT` 只命中定义与这个
classmethod 自身），是保留下来的基线/对照记录，不是运行路径的一部分。

**出货先验 `artifacts/trot_stairs.json` 经 `load_gait()` 之后的真实值**（`top[0]`，
`calf_shape: 0` → `"sin"`，`leg_phase` 缺省 → trot，`action_scale: 1.0`）：

| 字段 | 值 |
| --- | --- |
| `freq` | `5.477741882247174` |
| `amp_th` | `0.2650288316349047` |
| `amp_ca` | `0.9863652312670887` |
| `ph_th` | `-1.819353252233276` |
| `ph_ca` | `0.08531503233940636` |
| `off_th` | `0.2933585372533412` |
| `off_ca` | `0.013634768732911319` |
| `calf_shape` | `"sin"` |
| `action_scale` | `1.0` |
| `leg_phase` | `(0.0, 3.141592653589793, 3.141592653589793, 0.0)` |
| `period` | `0.18255697721736439 s` → `183 ms` |
| `\|off_ca\| + amp_ca` | `0.013634768732911319 + 0.9863652312670887` 在 float64 下**恰好 == 1.0**，`1.0 > 1.0` 为假 ⇒ `normalize()` **不缩放**（位对位保留 JSON 数字） |
| `summary()` | `5.48 Hz, ±0.265 thigh / ±0.986 calf (norm. units), shape=sin, scale=1.00 rad, period=183 ms` |

JSON 文件自身的其它键：`objective="stairs"`、`target_speed=null`、`step_height=0.05`、
`tread=0.5`、`clearance_weight=6.0`、`seconds=6.0`、`trials=500`、`refine=900`、
`top[0].score=22.235335505909532`、`top[0].eval.distance=6.308551063546455`、
`top[0].eval.mean_speed=1.0572354656849792`、`top[0].eval.fell=true`。
`top` 共 10 条（`rank=0..9` 都可用）。

**关节侧的关键值**：`EnvConfig.action_scale = 0.30 rad`（策略权限）、`gait_gain = 1.0`、
`control_dt = 0.002 × 10 = 0.02 s`（50 Hz）。

## 7. 与其它模块的关系

**上游（本模块依赖谁）**：只有标准库 `json` / `math` / `dataclasses` / `pathlib`
与 `numpy`。没有 MuJoCo，也没有 `robotdog` 内任何模块。

**下游（谁依赖本模块）**：

* `robotdog/env.py` `from robotdog.gait import GaitParams` —— 作为 `EnvConfig.gait`
  的字段类型与默认值，并在 `_gait_action()` 里驱动 PD 目标；
* `robotdog/policy.py` `env_config_from_summary()` 里**延迟 import**
  `from robotdog.gait import GaitParams`，把 `summary.json` 里的 `env_config.gait`
  字典还原成 `GaitParams.from_dict(...)`；
* `scripts/train.py:53` `from robotdog.gait import load_gait` —— CLI 开关；
* 文档与评估侧的「纯先验（动作=0）」基线本质上也是「只跑这条轨迹」，
  现在由 `scripts/play.py` 用**零动作策略**实现（历史上那个 `PriorPolicy`
  在已删除的 `scripts/render_checkpoint.py` 里）。

**方向为什么是这样**：`gait.py` 属于环境层（`docs/README.md` 分层规则表），只允许依赖
工具层与标准库。它**不依赖 `env.py`**，所以 `env → gait` 单向；反过来的话
`EnvConfig` 的类型注解会成环。`policy.py` 那条边是延迟 import，正是为了不把
`policy → gait → …` 变成 import 期的环。它也不进 `tools/`，因为它虽然无副作用，但语义上是
「一个策略的先验」，与「路径 / PNG / 地形」这类纯工具不同。

## 8. 实测与经验教训

* `experience.md` §4.1 —— 课程「毕业」判据把 PPO 的梯度更新饿死（策略输出 ≈ 0）：
  **有开环先验时未训练策略第一轮就达标**，`learn()` 立刻返回 False 进入下一档，
  于是 `mean|a| = 0.0109`、策略成功率与纯先验一模一样（75.0% vs 75.0%）。这条结论
  直接决定了 `StageSpec.min_timesteps` / `--min-stage-frac` 的存在——先验越强，
  毕业判据越必须有「最少步数」下限。
* `experience.md` §3.4 —— 残差策略不知道自己处在步态周期的哪一点：叠加在周期先验上
  的残差策略若无法判断相位就不可能是最优的，修法是 `ObsConfig.gait_phase`
  （sin/cos 两维）。实现就在 `robotdog/env.py:408-410`。
* 先验自身的标定数字（在平地、确定性 reset、零动作噪声下用真实 `Go2StairsEnv` 测）：
  `default_trot()` 速度目标 0.550 m/s、8 s 走 4.39 m、100% 存活、机身高度 0.268 m；
  `fast_trot()` 1.702 m/s、6 s 走 10.19 m。见 `robotdog/gait.py:28-34` 的模块 docstring。
* 先验的几何能力（摆动峰值 0.1124 m / 频率 ×0.6 → 0.1658 m）与「开环调参也翻不过去」
  见 `experience.md` §6（能力边界：11 cm 的墙）、§6.3、§6.4；`train.md` §3 的消融说明
  **改先验/改策略权限都不等于能上更高的台阶**。
* `experience.md` §2 —— 残差 RL 两个尺度分开的动机（共用一个 scale ⇒ 2 s 内摔倒），
  以及「从零学步态落在平台期」的两次标定。
* `train.md` §1 的启动自检要求核对 `[步态] ...` 那一行——本文 §6 给出了它的逐字段来源。

## 9. 已知限制与坑

* **开环楼梯小跑对台阶高不单调**：实测 2 cm → 40%、3 cm → 40%、**4 cm → 100%**、
  5 cm → 20%、6 cm → 0%（`robotdog/curriculum.py` 的课程注释里记录了这个数）。原因：
  固定 **183 ms** 的步态周期在约 4 cm 处与 50 cm 踏面**共振**。所以「从 2 cm 起步」
  的阶梯会从先验自身能力**之下**开始，白费第一级（`expand_stage(start_step_height=...)`
  就是为绕开它而存在的）。
* **纯先验（零残差）能上 4.4 m 但会摔**：1.5 cm → 88%、2.0 cm → 50%、2.5 cm → 12%、
  3.0 cm → 12%、4.0 cm → 25%。它能迈上去，但不稳——先验是**起点不是解**。
* **`_FALLBACK_TROT` 注释里的 `artifacts/trot_gait.json` 在仓里不存在**：`artifacts/`
  下只有 `trot_stairs.json`（`robotdog/gait.py:59` 的引用已过期）。
* **产生出货 JSON 的脚本不在仓里**：`robotdog/gait.py:28` 与 `from_search_json`
  的 docstring（`robotdog/gait.py:201`）都写 `scripts/search_trot.py`，
  但 `scripts/` 下现在只有 `download_go2.py` / `play.py` / `train.py` / `__init__.py`。
  所以 `artifacts/trot_stairs.json` 是**不可重建的 artifact**——改了它无法重搜。
* **`from_dict()` 静默忽略未知键**（`kw = {k: v for k, v in d.items() if k in known}`）：
  JSON 里把 `amp_th` 拼错成 `amp_thigh` 不会报错，只会掉到默认值 `0.0`，表现为
  「先验完全不抬腿」。写 JSON 后务必 `print(gait.summary())` 核对。
* **`normalize()` 是就地修改并返回 `self`**，不是返回副本：`g = p; g.normalize()` 会把
  `p` 一起改掉。要注意 `EnvConfig.gait` 与调用方手里的是**同一对象**。
* **`load_gait()` 对 `"top"` 键的判定优先于普通 gait 格式**：一份普通 gait JSON 只要
  多了一个无关的 `"top"` 字段，就会被当成搜索结果解析，然后在
  `ValueError: {path} has no 'top' entries` 或 `IndexError` 上炸掉。
* **两个 `action_scale` 同名但不同义**：`GaitParams.action_scale`（出货值 `1.0`）是
  「先验幅度」，`EnvConfig.action_scale`（`0.30`）是「策略权限」。源码里没有任何一处把
  两者相加或相乘后再 clip；真实实现（`robotdog/env.py:707-728`）是
  `target = clip(home + gait_gain*gait.radians(t) + cfg.action_scale*action, act_low, act_high)`
  ——clip 的是最终关节目标（用 MJCF 关节限位），没有「先求和再 clip 到 ±1」这一步。
  `grep "action_scale"` 会同时命中两处，对账时注意区分。以代码为准。
* **`static()` 的 `period` 是 `inf`**：`freq=0.0`。`reset()` 现在把相位硬置 0
  （不再有 `cfg.gait.freq > 0` 的随机化分支），但任何自己算 `1/freq` 的代码仍要小心除零。
* **`leg_phase` 是弧度不是周期比例**：写 `1.0` 得到 1 rad，不是「一整个周期」。
* **`FAST_TROT` 是死常量**：没有调用点，改它不会影响任何训练或推理路径。

## 10. 示例用法

```python
import numpy as np

from robotdog.gait import GaitParams, load_gait

# 1) 出货先验：搜索结果 JSON（top[0]），action_scale 来自条目本身 = 1.0
g = load_gait("artifacts/trot_stairs.json")
print(g.summary())
# 5.48 Hz, ±0.265 thigh / ±0.986 calf (norm. units), shape=sin, scale=1.00 rad, period=183 ms

# 2) 某一步的关节目标（与 env._target_from_action 同构）
home = np.zeros(12)
t = 0.10                                   # 控制步内累计的步态时间 [s]
target = home + 1.0 * g.radians(t) + 0.30 * np.zeros(12)   # gait_gain=1.0, cfg.action_scale=0.30
assert target.shape == (12,)

# 3) 手写一条先验并落盘（会被 normalize() 夹到合法区域）
custom = GaitParams(freq=4.0, amp_th=0.4, amp_ca=0.6,
                    off_th=0.3, off_ca=0.1, calf_shape="sin").normalize()
custom.save("artifacts/tmp_gait.json")
again = GaitParams.from_json("artifacts/tmp_gait.json")     # 读回来
assert again.summary() == custom.summary()

# 4) 关闭先验（等价于 GaitParams.static()）
off = GaitParams.static()
assert off.is_static and np.allclose(off.radians(t), 0.0)

# 5) 点名要 fallback 那一档（default_trot 只在这条路径上出现）
print(GaitParams.default_trot().summary())
# 6.00 Hz, ±0.249 thigh / ±0.552 calf (norm. units), shape=sin, scale=0.90 rad, period=167 ms
```
