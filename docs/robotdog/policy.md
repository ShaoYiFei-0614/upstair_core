# `robotdog/policy.py` 设计文档

## 1. 一句话定位

**训练层**的「读回」模块：把一个已经训练完的产物（SB3 `.zip` 权重、配套的
`*_vecnormalize.pkl`、`summary.json` 配置）还原成**能直接推理的策略 + 能直接建环境的
配置**，并集中处理「配置怎么恢复 / 观测维度是多少 / 归一化统计怎么用」这三个陷阱。

## 2. 职责边界

**做**：

* 定义推理侧的最小接口 `Policy`（只要 `predict(obs, deterministic=True)`）。
* 把 `VecNormalize` 的观测统计拷成只读的 `ObsNorm`，并提供 `normalize()`。
* 从 `summary.json` / `<模型>_env.json` / `.zip` 里保存的观测维度，三种途径恢复
  `EnvConfig`。
* 由台阶高度（cm）造一条 `StairSpec`。

**不做**：

* **不加载 PPO**。加载 PPO 需要先有一个环境，那是 `robotdog/training.py` 的
  `load_policy()` 的活；本模块只负责「环境怎么配」和「观测怎么归一化」。
* **不训练、不评估、不渲染**。评估与 GIF 在 `scripts/play.py`。
* **不碰奖励归一化**。`VecNormalize.ret_rms` 只在导出时有用，推理不需要。
* **不写文件**。

## 3. 关键设计决策与理由

### 3.1 `ObsNorm` 是 `VecNormalize` 的一份**拷贝**，不是引用

`pickle.loads("..._vecnormalize.pkl")` 得到的是一个 **`VecNormalize` 对象**，它有三个
致命问题，全都实测踩过：

| 问题 | 报错原文 |
| --- | --- |
| 不能当 dict 用 | `TypeError: 'VecNormalize' object is not subscriptable` |
| **多访问一个不存在的属性就递归爆栈** | `RecursionError: maximum recursion depth exceeded`（`BaseVecEnv.__getattr__` → `getattr_depth_check` → `_get_all_attributes`） |
| 不可调用 | `TypeError: 'VecNormalize' object is not callable` |

所以 `from_path()` 把 `obs_rms.mean`、`obs_rms.var`、`epsilon`、`clip_obs` 四个值拷出来，
然后 **`del raw`**（在 `finally` 里），此后整个仓只使用这份普通 dataclass。

**被否决的方案**：直接持有 `VecNormalize` 引用并写一个受限的访问包装器。
否决原因：递归问题来自 SB3 的 `__getattr__` 本身，任何 `getattr` 都可能触发它，
包装器无法保证不踩到；拷贝是唯一没有隐藏路径的做法。

### 3.2 观测维度的恢复有**三条**路径，优先级从精确到猜测

```
1. 同目录 summary.json            —— 精确（train.py 写下的完整配置）
2. <模型名>_env.json              —— 精确（权重被单独搬走时用）
3. 从 .zip 里保存的 obs 维度反推   —— 猜测（会打印一行提示）
```

**为什么必须有第 3 条**：`EnvConfig()` 默认 480 维（`history_len=10`、`command` 恒开），
而本仓 `config.yaml` 训出来是 48 维。拿默认配置去加载会抛
`ValueError: Observation spaces do not match`，而用户往往只从别处拷来一个 `.zip`。

**为什么反推是安全的**：四个开关的维度 3 / 2 / 1 / 2 **互不互质**，组合数只有 16 种，
枚举一遍就能唯一确定；`base_lin_vel_body` 默认开，所以把它排在枚举顺序最前面，
使最简配置优先命中。`base_height` 已从枚举中删除（它在 `ObsConfig.__post_init__` 里
就是硬错误），所以反推**不可能**给出一个开着 `base_height` 的配置。
反推不出来时**主动报错**而不是静默用默认值：

```
模型观测维度 {dim} 无法用 EnvConfig 的观测块还原。请把模型放回它的 runs/<name>/ 目录
（读 summary.json），或显式指定观测开关。
```

### 3.3 为什么 `observation_dim()` 不解包整个模型

`zipfile` 只读 `data` 这一个条目：

* SB3 ≥ 2.0：`data` 是 JSON，观测空间在 `observation_space[":serialized:"]` 里是一段
  base64 pickle；
* SB3 < 2.0：`data` 整个是 pickle。

只读一个条目意味着**不需要 torch、不需要构造环境、几十毫秒返回**。这是一个探测函数，
所以所有异常都吞掉并返回 `None`，由调用方决定怎么报错。

### 3.4 `_TUPLE_FIELDS` 是必须的

JSON 往返会把元组变成 list。`EnvConfig.kp` / `kd` 和 `GaitParams` 里的
`gravity` / `friction` 在代码里是元组，不转回来的话下游 `for kp, kd in zip(...)` 之类
的行为会变（更糟的是不报错、只是悄悄用错）。

### 3.5 `stair_from_height()` 用 `flat_spec` 表示平地

**不能用** `demo_stairs_spec(step_height=0.0, n_up=8)`：那会生成
`size="0.250000 1.000000 0.000000"` 的 box，MuJoCo 直接抛
`ValueError: Error: size 2 must be positive in geom`。平地必须走
`flat_spec(n_up=0, n_down=0, ...)`（见 `experience.md` §1.3）。

### 3.6 重建策略必须读**权重自带**的网络结构，而不是默认值

`load_policy()` 要先把 PPO 建出来再 `set_parameters()`。如果按仓里默认的
`DEFAULT_NET_ARCH = (512, 256)` 建网，去加载一个用 `ppo.net_arch: [64, 64]` 训出来的
权重，会在装载 state_dict 时炸：

```
RuntimeError: Error(s) in loading state_dict for ActorCriticPolicy:
    size mismatch for mlp_extractor.policy_net.0.weight: copying a param with shape
    torch.Size([64, 460]) from checkpoint, the shape in current model is
    torch.Size([512, 460]).
    size mismatch for action_net.weight: copying a param with shape torch.Size([12, 64])
    from checkpoint, the shape in current model is torch.Size([12, 256]).
```

所以 `policy_arch()` 从 `.zip` 的 `data` 条目里把 `policy_kwargs.net_arch` 读出来，
`load_policy()` 用 `arch = policy_arch(model_path) or DEFAULT_NET_ARCH`。
**这条是「超参搬进 `config.yaml`」之后才暴露的**：结构以前写死在两处、恰好一致，
现在配置能改它，两处就必须真的对齐（见 `experience.md` §7.1）。

### 3.7 每个 checkpoint 各记一份地形，而不是只记整条课程

`summary.json` 记的是**整条课程**的配置（最后一级 15 cm），而每个
`checkpoints/<阶段>_final.zip` 是在**各自的台阶**上训出来的。用最后一级的地形去评估
中间 checkpoint，等于在测一个它没学过的高度。

所以 `train.py` 每级额外写一份 `<模型>_env.json`（`save_env_json()`），report 段里带
`step_height` / `stage`；`play.py` 用 `step_height_from_model()` 优先读它。
**被否决的方案**：把 `step_height` 塞进 `summary.json` —— 那是一份**每次运行只有一份**
的文件，无法表达「每级不同」；也否决了「按 `--stage` 参数猜」——中间 checkpoint
搬出 `runs/` 之后就没有阶段信息了。老权重没有 `_env.json` 时返回 `None`，
`play.py` 退回默认 3 cm（见 `README.md` §5）。

## 4. 公开 API

```python
class Policy(Protocol):
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray: ...

@dataclass
class ObsNorm:
    mean: np.ndarray
    var: np.ndarray
    epsilon: float = 1e-8
    clip: float = 10.0

    def normalize(self, obs: np.ndarray) -> np.ndarray: ...      # float32
    @classmethod
    def from_path(cls, path: str | Path) -> ObsNorm: ...

def env_config_from_summary(summary_path: str | Path) -> EnvConfig: ...
def observation_dim(model_path: str | Path) -> int | None: ...
def policy_arch(model_path: str | Path) -> tuple[int, ...] | None: ...
def env_config_for_model(model_path: str | Path) -> EnvConfig: ...
def env_json_path(model_path: str | Path) -> Path: ...
def save_env_json(model_path: str | Path, cfg: EnvConfig, *,
                  step_height: float | None = None, stage: str = "") -> Path: ...
def step_height_from_model(model_path: str | Path) -> float | None: ...
def stair_from_height(height_cm: float | None, *, tread: float = 0.50,
                      n_up: int = 8) -> StairSpec | None: ...

__all__ = ["ObsNorm", "Policy", "env_config_for_model", "env_config_from_summary",
           "env_json_path", "observation_dim", "policy_arch", "save_env_json",
           "stair_from_height", "step_height_from_model"]
```

语义与异常：

* `ObsNorm.normalize(obs)` — `clip((x - mean) / sqrt(var + epsilon), ±clip)`，返回
  `float32`。与 SB3 的 `VecNormalize` 逐位一致（同一公式、同一 `epsilon`、同一 `clip`）。
* `ObsNorm.from_path(path)` — 读**裸 pickle**。文件不存在时由 `open()` 抛
  `FileNotFoundError`；文件不是 `VecNormalize` 时 `raw.obs_rms` 抛 `AttributeError`。
* `env_config_from_summary(path)` — 读 `env_config` / `obs_config` / `reward_config`
  三段；未知键被忽略（`_rebuild` 只保留 dataclass 有的字段），所以旧 `summary.json`
  里的废弃字段不会让它崩。`gait` 段用 `GaitParams.from_dict` 重建。
* `observation_dim(model_path)` — **从不抛异常**，任何问题都返回 `None`。
* `policy_arch(model_path)` — **从不抛异常**，返回 `(隐藏层宽度…)` 或 `None`。
  读的是 `data` 里的 `policy_kwargs.net_arch`（普通 JSON `{"pi": [...], "vf": [...]}`，
  `tuple` 或 `list` 都接受）；`net_arch` 只有一个 int 时返回 `(w,)`，非法值返回 `None`。
* `env_config_for_model(model_path)` — 三条路径都失败时抛 `SystemExit`（带中文提示）。
* `env_json_path(model_path)` — `Path("<stem>_env.json")`，与
  `_vecnormalize.pkl` 同一命名规矩；**不检查文件是否存在**。
* `save_env_json(model_path, cfg, *, step_height=None, stage="")` — 写
  `{"env_config":…, "obs_config":…, "reward_config":…, "report": {"step_height":…,
  "stage":…}}`（`json.dumps(indent=2, ensure_ascii=False)`），返回写入路径；
  父目录不存在时自己 `mkdir(parents=True, exist_ok=True)`。
* `step_height_from_model(model_path)` — 读 `report.step_height`（**单位 m**）；
  文件不存在、JSON 坏、字段缺失或不可转 `float` 时一律 `None`（**从不抛异常**）。
* `stair_from_height(height_cm, *, tread, n_up)` — `None` → `None`（沿用配置里的地形）；
  `<= 0` → `flat_spec(n_up=0, n_down=0, step_height=0.0, tread=tread)`；否则
  `demo_stairs_spec(step_height=height_cm/100, tread=tread, n_up=n_up, top_landing=1.20)`。

## 5. 内部结构与数据流

```
play.py
  resolve_target(run)
      _resolve_model(run)            → 找到 .zip
      _resolve_norm(.zip)            → 找到同名 *_vecnormalize.pkl
      env_config_for_model(.zip) ──┐
      step_height_from_model(.zip) → play 的默认地形
                                   │
env_config_for_model ──┬── summary.json 存在 → env_config_from_summary()
                       ├── <stem>_env.json 存在 → env_config_from_summary()
                       └── observation_dim() → _obs_config_for_dim() → EnvConfig(obs=…)

train.py（每级存完权重之后）
  policy.save_env_json(<阶段>_final.zip, cfg, step_height=…, stage=…)
       → <阶段>_final_env.json（"report": {"step_height", "stage"}）
  末级三件套 shutil.copy2 → model_final{.zip,_vecnormalize.pkl,_env.json}

training.load_policy(model_path, cfg, stair)
      → policy_arch(.zip) or DEFAULT_NET_ARCH   # 网络结构必须与训练时一致
      → 建 1 环境 DummyVecEnv（cfg 来自上面那条链）
      → PPO.load / set_parameters
      → ObsNorm.from_path(norm_path)      # 读裸 pickle 后立刻 del
      → (PPO, ObsNorm | None)

play.Runner.predict(obs)
      → ObsNorm.normalize(obs) → PPO.predict(x, deterministic=True)
```

`_zip_data()` 是 `observation_dim()` 与 `policy_arch()` 共用的底座：从 `.zip` 里取出
`data` 条目（`.npz`），失败返回 `None`；`_deserialize()` 则负责把 SB3 存进去的值还原
（含 `":serialized:"` 前缀的 base64 pickle，但 `net_arch` **不需要**走这条路 —— 它是
普通 JSON）。

## 6. 关键常量与默认值

| 名称 | 值 | 说明 |
| --- | --- | --- |
| `ObsNorm.epsilon` | `1e-8` | 与 SB3 `VecNormalize` 的默认一致 |
| `ObsNorm.clip` | `10.0` | 来自 `VecNormalize.clip_obs`，训练时就是 10.0 |
| `_TUPLE_FIELDS` | `frozenset({"kp", "kd", "gravity", "friction"})` | JSON 往返后要转回元组的字段 |
| `_obs_config_for_dim` 的 `history_len` | `10` | 观测块枚举时的固定历史长度 |
| `_obs_config_for_dim` 的枚举顺序 | `("base_lin_vel_body", "gait_phase", "step_height", "command")` | **四个开关**（`base_height` 已删除）；`product((False, True), repeat=4)`；`base_lin_vel_body` 默认开 ⇒ 最简配置先命中 |
| `save_env_json` 的 report 键 | `{"step_height", "stage"}` | `step_height` 单位 **m**，`stage` 是阶段名（如 `stair15`） |
| `stair_from_height` 的 `tread` / `n_up` | `0.50` / `8` | 与课程一致 |
| `stair_from_height` 的 `top_landing` | 硬编码 `1.20` | 与 `demo_stairs_spec` 的课程默认一致 |
| `EnvConfig()` 默认观测 | `frame_size=48`、`size=480` | 反推失败时的兜底值（`history_len=10`） |
| `_rebuild` 覆盖的嵌套字段 | `command: CommandConfig` | JSON 往返后 `command` 是 `dict`，**必须重建**，否则 `set_command_now` 抛 `AttributeError: 'dict' object has no attribute '_now'` |

## 7. 与其它模块的关系

```
                 robotdog/tools/terrain.py  (StairSpec / flat_spec / demo_stairs_spec)
                              ▲
robotdog/env.py (EnvConfig/ObsConfig) + robotdog/reward.py (RewardConfig) ──►  robotdog/policy.py
                              ▲                              ▲
                     robotdog/training.py ──────────────────┘   (load_policy)
                              ▲
                        scripts/play.py
```

* **向下**依赖 `env`（三个 dataclass）与 `tools/terrain`（几何）。它是训练层里唯一
  只读 `env` 的**配置类**、而完全不需要实例化环境的模块 —— 这是刻意的：`.zip` 的
  观测维度探测必须在**没有环境**的情况下也能跑。
* **向上**被 `training.load_policy()` 与 `scripts/play.py` 使用。
* **与 `robotdog/config.py` 的分工**：`config.py` 管**「怎么训」**（`config.yaml` →
  `EnvConfig` / 阶段表，见 `docs/robotdog/config.md`），本模块管**「训完之后怎么读回去」**
  （`.zip` → 网络结构 / 观测维度 / 地形记录）。两者都产出 `EnvConfig`，但方向相反，
  且**不互相 import**。
* **为什么 `env_config_from_model` 住在这里而不是 `training.py`**：`training.py` 要
  import `stable_baselines3`（重），而「读 `summary.json`」不需要 SB3。分开之后，
  只想读配置的调用方不必拖进 torch。

## 8. 实测与经验教训

* `experience.md` §5.4 —— `pickle.loads(vecnormalize.pkl)` 的三个坑（subscriptable /
  递归 / 不可调用），`ObsNorm` 就是这三条的直接产物。
* `experience.md` §5.3 —— 打开观测归一化却没跟上权重 ⇒ 5 cm 上成功率从 100% 掉到 0%。
* `experience.md` §1.3 —— 零高度台阶无法编译，所以平地走 `flat_spec`。
* `experience.md` §7.1 —— 按默认结构重建模型会 `size mismatch`，`policy_arch()` 的由来。
* `experience.md` §7.4 —— 为什么地形记录必须**每个 checkpoint 各一份**。
* `README.md` §3.2 —— 观测块的维度表（42 基础块 + 3/1/2/1 四个开关）。
* `README.md` §4.1 —— `*_vecnormalize.pkl` 必须与 `.zip` 成对。
* `README.md` §4.4 —— 超参全在 `config.yaml`（`ppo.net_arch` 就是 `policy_arch()` 要读的那个键）。

## 9. 已知限制与坑

* **`env_config_for_model()` 的第 3 条路径是启发式的**。若将来观测块出现维度相同的
  不同组合（例如再加一个 1 维开关），反推结果可能不是训练时用的那个；此时应当把
  权重放回它的 `runs/<名字>/` 目录（有 `summary.json`）。
* `observation_dim()` 只读 `data` 条目。SB3 改掉这个内部布局就会返回 `None`（静默）。
* **`policy_arch()` 也是静默降级**：读不到 `net_arch` 就返回 `None`，`load_policy()`
  会退回 `DEFAULT_NET_ARCH = (512, 256)`。一个非默认结构、又没有 `policy_kwargs` 的
  老权重会重新撞上 `size mismatch` —— 这时唯一的线索是 traceback 里的两个形状。
* **`<模型>_env.json` 只在训练时写**（`train.py` 每级一次 + 末级复制一份）。手工搬动
  过的老权重没有它，`step_height_from_model()` 返回 `None`，`play.py` 会用默认 3 cm；
  **评估结果会偏低，别把这种偏低当成策略退化**。
* `step_height_from_model()` 返回的是**训练时**的地形，不是「这个权重能爬的最高台阶」。
* `stair_from_height()` 的 `top_landing=1.20` 是硬编码的，不接受参数；要别的平台长度
  就直接用 `tools.terrain` 的函数。
* `Policy` 是 `Protocol`，**不提供任何实现**；实现分别在 `training.load_policy()`
  （PPO）与 `scripts/play.py` 的 `Runner`（PPO + 归一化）里。

## 10. 示例用法

```python
from robotdog.policy import (ObsNorm, env_config_for_model, policy_arch,
                            step_height_from_model, stair_from_height)

# 恢复训练时的配置（优先 summary.json，其次 <stem>_env.json，再按观测维度反推）
cfg = env_config_for_model("runs/walk1/model_final.zip")
print(cfg.obs.frame_size, cfg.obs.size)          # 48 480（本仓 config.yaml 是 48）

# 重建模型要用的结构，以及这份权重当时是在多高的台阶上训的
print(policy_arch("runs/walk1/model_final.zip"))         # (512, 256)
print(step_height_from_model("runs/walk1/model_final.zip"))  # 0.15（m）

# 归一化统计
norm = ObsNorm.from_path("runs/walk1/model_final_vecnormalize.pkl")
z = norm.normalize(obs)                          # float32，已 clip

# 造一条 9 cm 的楼梯 / 一条平地
stair = stair_from_height(9.0)
flat = stair_from_height(0.0)
```
