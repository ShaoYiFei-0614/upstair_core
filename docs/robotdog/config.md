# `robotdog/config.py` 设计文档

## 1. 一句话定位

**配置层**：把仓根的 `config.yaml` 读成两个已经可直接使用的东西 ——
:class:`~robotdog.env.EnvConfig`（一个环境实例的全部参数）和
:class:`~robotdog.curriculum.StageSpec` 列表（六级课程），让 `scripts/train.py` 与
`scripts/play.py` 不必再暴露一堆命令行开关。

## 2. 职责边界

**做**：

* 读 YAML（`load_config`），给出中文的错误信息（文件不存在 / 语法错 / 顶层不是映射）。
* 把 `env` + `env.obs` + `reward` 三段写进 `EnvConfig`（`config_to_env`）。
* 把 `curriculum` 段展开成阶段表，按高度生成 `flat_spec` 或 `demo_stairs_spec`
  （`config_to_stages`）。
* 补齐缺省值（`config_to_run` / `config_to_ppo` / `config_to_play` / `n_envs` / `device`）。
* 校验：**未知键报错**、等高线升序、步数非负、`timesteps` 长度自适应。

**不做**：

* **不训练、不建环境、不加载权重**。它只产出 dataclass，`Go2StairsEnv` 由
  `robotdog/training.py:make_vec_env` 建。
* **不解析 PPO 的键**。`config_to_ppo` 原样返回 `ppo` 段，合法键由
  `training.build_ppo()` 决定；这里只把它整段透传，避免同一份参数在两处各写一遍默认值。
* **不写文件**。产物与 `summary.json` 都在 `scripts/train.py` 与 `robotdog/policy.py`。
* **不碰地理/物理**。地形几何在 `robotdog/tools/terrain.py`，这里只是挑参数。
* 不负责 play 的渲染实现（Pillow 代码在 `scripts/play.py`），只提供 `play` 段的默认值。

## 3. 关键设计决策与理由

### 3.1 未知键**报错**，而不是静默忽略

`_apply()` 先取 `target.__dataclass_fields__`，把不认识的键收集起来一次性抛出：

```
config.yaml 的 env: 无法识别的键 ['command_vx_']; 可用键 = ['action_mode', ...]
```

**理由**：配置文件是手写的，`forward_vel_sigma` 写成 `forward_vel_sigme`、`n_envs`
写成 `num_envs` 这类错误，YAML 本身完全合法。静默忽略意味着**你以为调了，其实没调**，
而且会浪费一次几十分钟的训练。报错 + 列出可用键是成本最低的防呆。

**被否决的方案**：用一个宽松的 `setattr` 循环（旧 `build_config(args)` 就是这种形状，
只不过它的「未知键」来自 argparse，天生不存在打错的可能）。搬到 YAML 之后这个前提没了。

### 3.2 缺省值只写一遍：**没写的键保持 dataclass 默认值**

`config_to_env` 先 `cfg = EnvConfig()`，再只对 YAML 里出现的键 `setattr`。所以：

* 代码里的 dataclass 是**唯一**的默认值来源（`config.yaml` 只列出想显式固定的项）；
* `config.yaml` 里可以只写五行就跑起来（`curriculum.heights_cm` 一项都能省）；
* 三份根文档里的默认值表与 dataclass 不可能漂移。

### 3.3 `timesteps` / `learning_rate` 允许标量或列表，长度自适应

* `timesteps: 2000` ⇒ 每级都 2000（冒烟用）。
* `timesteps: [100000, 250000, ...]` 且长度等于级数 ⇒ 原样使用。
* 长度**不等** ⇒ 按比例重采样到级数（保持总量，**余数补给最难的一级**），
  这样把 `heights_cm` 从六级改成七级时不必手改预算。

`learning_rate` 同理，但缺省时给统一值 `1.5e-4`，而不是 `curriculum.py` 里那条
逐级递减的表 —— **哪套学习率是「配置」而不是「常量」**：`config.yaml` 显式给了就用它，
没给就用统一值，避免配置文件与代码各持一份互相矛盾的表。

### 3.4 阶段名与标签从高度推导

`heights_cm: [0.0, 3.0, ...]` ⇒ 名字 `flat` / `stair3` / `stair6` / …（`3.5` 会写成
`stair3.5`），标签 `平地` / `3cm 台阶` / …。

**理由**：名字是**产物路径的一部分**（`checkpoints/<名字>_final.zip`），也是 `play.py`
与日志里引用的标识。从高度推导就保证了「名字与地形永远一致」，不需要人去维护一张
`name → height` 的映射表。**被否决的方案**：让配置文件自己写 `name`（多一个能写错的
字段，且可能与 `heights_cm` 矛盾）。

### 3.5 平地走 `flat_spec` 而不是「高度为 0 的台阶」

`demo_stairs_spec(step_height=0.0)` 会让 MJCF 编译失败：

```
ValueError: Error: size 2 must be positive in geom
（MjModel 编译失败，场景 dump 到 .generated/failed_scene.xml）
```

所以 `h_cm <= 0.0` 走 `flat_spec(n_up=0, n_down=0, step_height=0.0, tread=tread)`。
这与 `robotdog/policy.py:stair_from_height` 的处理必须一致，否则「平地」在训练侧与
播放侧会是两种地形。

## 4. 公开 API

```python
DEFAULT_CONFIG_PATH: Path = paths.WORKSPACE / "config.yaml"

def load_config(path: str | Path | None = None) -> dict[str, Any]
def config_to_env(raw: Mapping[str, Any]) -> EnvConfig
def config_to_stages(raw: Mapping[str, Any]) -> list[StageSpec]
def config_to_ppo(raw: Mapping[str, Any]) -> dict[str, Any]
def config_to_run(raw: Mapping[str, Any]) -> dict[str, Any]
def config_to_play(raw: Mapping[str, Any]) -> dict[str, Any]
def n_envs(raw: Mapping[str, Any]) -> int          # env.n_envs, 缺省 8
def device(raw: Mapping[str, Any]) -> str          # ppo.device, 缺省 "cpu"
```

内部辅助：

```python
def _section(raw, name) -> dict[str, Any]           # 取子段，非映射即报错
def _timesteps(value, n_stages, *, where) -> list[int]
def _floats(value, n_stages, *, where, default) -> list[float]
def _spread(weights, n) -> list[float]              # 线性重采样
def _apply(target, values, *, where) -> None        # 未知键报错
def _load_gait(gait_file) -> GaitParams
```

`config_to_env` 会**弹出**三个键再交给 `_apply`：`obs`（嵌套子段，单独处理）、
`n_envs`（属于运行参数，不属于 `EnvConfig`）、`gait_file`（只有开了 `gait_prior`
才真的读文件）。

## 5. 内部结构与数据流

```
config.yaml
   │  yaml.safe_load
   ▼
raw: dict
   ├── run:         config_to_run    → {name, seed, save_every, log_every}
   ├── curriculum:  config_to_stages → [StageSpec(flat), StageSpec(stair3), ...]
   │                 └── flat_spec / demo_stairs_spec（tools/terrain.py）
   ├── env:         config_to_env    → EnvConfig
   │     ├── env.*      → EnvConfig 字段（_apply）
   │     ├── env.obs.*  → ObsConfig 字段（_apply）
   │     └── gait_file  → load_gait（仅 gait_prior=true 时）
   ├── reward:      ├── → RewardConfig 字段（_apply）
   ├── ppo:         config_to_ppo    → dict（原样透传）
   └── play:        config_to_play   → dict（补渲染默认值）
```

## 6. 关键常量与默认值

| 常量 | 值 | 说明 |
| --- | --- | --- |
| `DEFAULT_CONFIG_PATH` | `<仓根>/config.yaml` | `paths.WORKSPACE / "config.yaml"` |
| `DEFAULT_STAGE_TIMESTEPS` | `[100_000] + [250_000] × 5` | `curriculum.timesteps` 省略时用 |
| `DEFAULT_LR` | `1.5e-4` | `curriculum.learning_rate` 省略时用 |
| `env.n_envs` | `8` | |
| `run.name` / `seed` / `save_every` / `log_every` | `course1` / `0` / `50000` / `20` | |
| `play.episodes_gif` / `episodes_eval` | `10` / `100` | |
| `play.camera` / `width` / `height` / `fps` / `scale` | `track` / `960` / `540` / `25` / `2` | |
| `play.seed` | `0` | |
| `ppo.device` | `cpu` | |

默认课程（不写 `curriculum` 段时）来自 `robotdog/curriculum.py` 的
`STAGE_HEIGHTS_CM`（`(0.0, 3.0, 6.0, 9.0, 12.0, 15.0)`）与
`DEFAULT_TREAD`/`DEFAULT_N_UP`（`0.50` m / `8` 级），总预算 1,350,000 步。

## 7. 与其它模块的关系

* **依赖** `robotdog.env`（`EnvConfig`）、`robotdog.curriculum`（`StageSpec` 与默认
  常量）、`robotdog.gait`（`load_gait`）、`robotdog.tools.terrain`（造地形）、
  `robotdog.tools.paths`（定位仓根与先验文件）、第三方 `yaml`。
* **被依赖** `scripts/train.py`（`config_to_run` / `config_to_env` / `config_to_stages` /
  `config_to_ppo` / `n_envs`）与 `scripts/play.py`（`config_to_play`）。
* **为什么放在 `robotdog/` 而不是 `scripts/`**：它要对 `EnvConfig` 的字段做白名单
  校验（`__dataclass_fields__`），因此必须 import `env`；而 `scripts/` 只是入口，
  不该给库代码反向依赖。同理由也适用于 `policy.py`（见 `docs/robotdog/policy.md` §7）。
* 与 `robotdog/policy.py` 的分工：**config.py 管「怎么训」**（读进来），
  **policy.py 管「训完之后怎么读回去」**（读出去，含网络结构与地形记录）。

## 8. 实测与经验教训

* 首次跑通：`load_config()` → `frame 46 / obs 460`、`gait_prior=True`、
  六级打印正确、总预算 1,350,000 步。见 `experience.md` §7 的开头。
* `policy_arch()` 的必要性（不读配置的结构就会 `size mismatch`）见
  `experience.md` §7.1。
* 每个 checkpoint 各存一份地形记录（`<模型>_env.json`）见 `experience.md` §7.4。
* `yaml` 是**新增依赖**：`ModuleNotFoundError: No module named 'yaml'` 是这条路径上
  第一个会撞到的错，已经写进 `README.md` §1 的依赖表。

## 9. 已知限制与坑

* `_spread()` 的重采样是**线性插值**，只在「列表长度不等于级数」时触发。它保持总量，
  但会抹平原始列表的锯齿形状 —— 想精确控制每一级就直接写等长列表。
* `_timesteps()` 里 `out[-1] += ...` 会把重采样余数全部补给**最难的一级**；
  若最后一级本来就最小，取整误差可能让它反超前一级（差 1~2 步，不影响训练）。
* **不支持 YAML 的锚点/多文档/自定义标签**（用的是 `yaml.safe_load`，遇到会报错）。
* `config_to_stages` 只认「高度升序」；想跑非单调的课程（例如交错）需要改代码。
* 配置文件里 `env.gait_file` 的路径**相对仓根**解析（不是相对 `config.yaml`）；
  配置文件如果被搬到别处，先验路径仍然指向仓根。
* `curriculum.timesteps` 给标量时**不会**按平地/台阶 1:2.5 加权，是每级等量 ——
  这是刻意的（标量的语义就是「每级这么多」），要加权就写列表。

## 10. 示例用法

```python
from robotdog.config import load_config, config_to_env, config_to_stages

raw = load_config()                       # 默认 <仓根>/config.yaml
cfg = config_to_env(raw)
stages = config_to_stages(raw)

print(cfg.obs.size)                       # 460
print([(s.name, s.timesteps) for s in stages])
print(sum(s.timesteps for s in stages))   # 1350000
```

命令行侧（`scripts/train.py` 与 `scripts/play.py` 只暴露两个开关）：

```bat
.venv\Scripts\python.exe scripts\train.py --run-name walk1
.venv\Scripts\python.exe scripts\train.py --config .tmp\smoke_config.yaml
.venv\Scripts\python.exe scripts\play.py --run walk1 --mode eval
```
