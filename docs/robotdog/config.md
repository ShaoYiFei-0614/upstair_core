# `robotdog/config.py` 设计文档

## 1. 一句话定位

**配置层**：把仓根的 `config.yaml` 读成两个已经可直接使用的东西 ——
:class:`~robotdog.env.EnvConfig`（一个环境实例的全部参数）和
:class:`~robotdog.curriculum.StageSpec` 列表（当前九级课程），让 `scripts/train.py` 与
`scripts/play.py` 不必再暴露一堆命令行开关。

## 2. 职责边界

**做**：

* 读 YAML（`load_config`），给出中文的错误信息（文件不存在 / 语法错 / 顶层不是映射）。
* 把 `env` + `env.obs` + `env.command` + `reward` 四段写进 `EnvConfig`（`config_to_env`）。
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
config.yaml 的 env: 无法识别的键 ['command_vx_']; 可用键 = ['action_scale', 'command', 'decimation', ...]
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
  这样把 `heights_cm` 从八级改成九级时不必手改预算。

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

### 3.6 指令范围也走课程，而且与观测维数解耦

`curriculum.command_vy_max` / `command_yaw_max` 是两条**与 `heights_cm` 等长**的数组，
写进 `StageSpec.command_vy_max` / `command_yaw_max`，由 `scripts/train.py` 在每级建完
环境后调 `venv.env_method("set_commands", vy, yaw)` 下发。

**为什么范围可以逐级变、但观测那 2 维必须恒在**：PPO 的 `ActorCritic` 网络与
`RolloutBuffer` 的维度在 `learn()` 开始后就固定了，中途改维数会直接崩。所以
`ObsConfig.command` 是**恒真**的开关，课程只改**采样范围**（`vy_max` / `yaw_max`），
不改维数。**被否决的方案**：让每一级自带一份 `EnvConfig`（维数会随级变化，网络得重建，
warm-start 也就断了）。

`env.command` 段本身是**嵌套 dataclass**（`CommandConfig`）：`vx` 是定值，
`vy_max` / `yaw_max` 是采样范围，当前值 `[vx, vy, wz]` 存在 `_now` 里并与
`env._command` 是**同一个 list 对象**。`config_to_env` 会先 `env.pop("command")` 再
`_apply(cfg.command, ...)`，最后补一次 `cfg.command.__post_init__()` 把 `vx` 同步进
当前值 —— 少了这一步，`_sample_command()` 会抽到过期的 `vx`。

## 4. 公开 API

```python
DEFAULT_CONFIG_PATH: Path = paths.WORKSPACE / "config.yaml"
PLAY_COMMAND_FRAC: float = 0.8

def load_config(path: str | Path | None = None) -> dict[str, Any]
def config_to_env(raw: Mapping[str, Any]) -> EnvConfig
def config_to_stages(raw: Mapping[str, Any]) -> list[StageSpec]
def config_to_ppo(raw: Mapping[str, Any]) -> dict[str, Any]
def config_to_run(raw: Mapping[str, Any]) -> dict[str, Any]
def config_to_play(raw: Mapping[str, Any]) -> dict[str, Any]
def command_for_stage(stage: StageSpec, *, frac: float = PLAY_COMMAND_FRAC,
                      vx: float | None = None) -> tuple[float, float, float]
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
def _bools(value, n_stages, *, where, default) -> list[bool]
def _unique_name(base, seen) -> str               # 阶段重名时加 _2 / _3
def _load_gait(gait_file) -> GaitParams
```

`config_to_env` 会**弹出**四个键再交给 `_apply`：`obs` 与 `command`（嵌套子段，
各自单独 `_apply` 到 `cfg.obs` / `cfg.command`）、`n_envs`（属于运行参数，不属于
`EnvConfig`）、`gait_file`（只有开了 `gait_prior` 才真的读文件）。读进先验之后，
若 `gait_amp_scale != 1.0`，就把 `amp_th` / `amp_ca` 各乘一次该系数再 `normalize()`
—— 只缩振幅、保留高步频（先验自身 2.05 m/s，而指令区间是 0.2~0.6 m/s）。

`command_for_stage(stage)` 是**播放侧**用的：给一个课程阶段，返回一条「落在该级
指令量程内」的 `(vx, vy, yaw_rate)` —— 量程为 0 的方向给 0，有量程的给
`frac × 量程`，`vx` 显式传入则覆盖阶段的前进指令。为什么要它：逐阶段渲染 GIF 时
必须钉一条**该级真能执行**的指令，否则（例如）在 `flat` 那一级（`vy_max = 0`）
钉一条 ±0.3 的横移指令，看画面的人会以为策略不会横移，其实是量程本来就没放开。

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
   │     ├── env.command.* → CommandConfig 字段（_apply + __post_init__）
   │     └── gait_file  → load_gait（仅 gait_prior=true 时）
   ├── reward:      cfg.reward ← _apply（在 config_to_env 内，robotdog/config.py:246；字段定义在 robotdog/reward.py）
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
| `run.name` / `seed` / `save_every` / `log_every` | `course1` / `0` / `50000` / `20` | 代码缺省是 `course1`（`robotdog/config.py:380-387`）；`config.yaml` 现值是 `course3` |
| `play.episodes_gif` / `episodes_eval` | `10` / `100` | |
| `play.gif_max_frames` | `300`（`PLAY_GIF_MAX_FRAMES`） | 单回合最多写入 GIF 的帧数；防渲染中途 `MemoryError` |
| `play.seed` / `stages` / `batch` | `0` / `auto` / `""` | `stages`：`auto`/`curriculum`/阶段名或高度列表 |
| `ppo.device` | `cpu` | |

> `play.camera` / `width` / `height` / `fps` / `scale` **已经不在 `config_to_play` 里**
> （2026-06 移出）：它们改成 `scripts/play.py` 的模块常量 `CAMERA_GIF = "track"`、
> `FRAME_WIDTH = 960`、`FRAME_HEIGHT = 540`、`GIF_FPS = 25`、`GIF_SCALE = 2`。
> 理由：相机与编码参数是「看画面」的事，不是训练行为，留成 yaml 旋钮只会让人以为
> 调它有意义（用户明确要求「相机和 GIF 渲染参数完全可以固定」）。
> `play.py` 的 `write_gif()` 也随之不再收 `fps` / `scale` 参数。

默认课程（不写 `curriculum` 段时）来自 `robotdog/curriculum.py` 的
`STAGE_HEIGHTS_CM`（`(0.0, 0.0, 0.0, 3.0, 6.0, 9.0, 12.0, 15.0)`）与
`DEFAULT_TREAD`/`DEFAULT_N_UP`（`0.50` m / `8` 级），**默认**总预算 1,850,000 步。
⚠ 2026-10-12 起 `config.yaml` **不再与这套默认值一致**：它显式给出**九级**、
总预算 2,408,192 步，并给 3 / 6 cm 各只留 4,096 步 —— 真值以 `config.yaml` 为准
（重新分配的理由记在 `experience.md` §21）。

## 7. 与其它模块的关系

* **依赖** `robotdog.env`（`EnvConfig`）、`robotdog.curriculum`（`StageSpec` 与默认
  常量）、`robotdog.gait`（`load_gait`）、`robotdog.tools.terrain`（造地形）、
  `robotdog.tools.paths`（定位仓根与先验文件）、第三方 `yaml`。
  奖励字段（`reward:` 段）定义在 `robotdog/reward.py`，但本模块**不直接 import 它**：
  靠 `robotdog/env.py:37` 的重导出，`_apply(cfg.reward, ...)`（`robotdog/config.py:296`）
  就能工作 —— 依赖方向仍是 `config.py → env.py → reward.py`，没有新增边。
* **被依赖** `scripts/train.py`（`config_to_run` / `config_to_env` / `config_to_stages` /
  `config_to_ppo` / `n_envs`）与 `scripts/play.py`（`config_to_play`）。
* **为什么放在 `robotdog/` 而不是 `scripts/`**：它要对 `EnvConfig` 的字段做白名单
  校验（`__dataclass_fields__`），因此必须 import `env`；而 `scripts/` 只是入口，
  不该给库代码反向依赖。同理由也适用于 `policy.py`（见 `docs/robotdog/policy.md` §7）。
* 与 `robotdog/policy.py` 的分工：**config.py 管「怎么训」**（读进来），
  **policy.py 管「训完之后怎么读回去」**（读出去，含网络结构与地形记录）。

## 8. 实测与经验教训

* 首次跑通：`load_config()` → `frame 48 / obs 48`（`history_len=1`，当时指令占 2 维）、`gait_prior=True`、
  课程打印正确。见 `experience.md` §7 的开头。
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

print(cfg.obs.size)                       # 51（history_len=1，含三轴指令与两条滤波量）
print([(s.name, s.timesteps) for s in stages])
print(sum(s.timesteps for s in stages))   # 1850000
```

命令行侧（`scripts/train.py` 与 `scripts/play.py` 只暴露两个开关）：

```bat
.venv\Scripts\python.exe scripts\train.py --run-name walk1
.venv\Scripts\python.exe scripts\train.py --config .tmp\smoke_config.yaml
.venv\Scripts\python.exe scripts\play.py --run walk1 --mode eval
```
