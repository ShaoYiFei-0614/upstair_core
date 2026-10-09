# `robotdog/curriculum.py` —— 固定六级课程与报表

## 1. 一句话定位

**训练层**里的**纯数据 + 报表模块**：定义课程的一级长什么样（`StageSpec`）、
训练过程中收集回合统计（`EpisodeStats`）、阶段结束后留下实测结果（`StageResult`），
并把它们排成可打印的定宽表格与 `summary.json`。

它**不是**驱动器 —— 真正逐级跑循环的是 `scripts/train.py`；本模块只回答
「训哪几级、按什么顺序、训多少步、成绩怎么记」。

## 2. 职责边界

**做**：

* 固定台阶高度梯度 `STAGE_HEIGHTS_CM = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)`。
* 用 `StageSpec` 描述一级：名字、台阶高、`StairSpec` 地形、预算步数、学习率、中文标签。
* 用 `EpisodeStats` 在滑动窗口里统计成功率 / 摔倒率 / 均回报 / 均步数。
* 用 `StageResult` 记录一级训练完的实测结果（含墙钟用时、模型路径、观测维度）。
* 用 `format_curriculum` / `write_summary` / `format_seconds` 出报表。

**不做**：

* **不训练、不建环境、不建 PPO**。本模块不 import `stable_baselines3`，不 import
  `robotdog.env` / `robotdog.training`。
* **不做阶段调度**：没有自动晋级 / 回退 / 断点续训 / 高度带混合。循环在
  `scripts/train.py` 里，是几行普通的 `for`。
* **不读 `config.yaml`**。`config_to_stages()` 在 `robotdog/config.py` 里，
  它构造本模块的 `StageSpec`（方向是「配置层 → 本模块」，反过来不存在）。
* **不写 tensorboard、不存模型**（那是 `robotdog/training.py` 的 `RunDir`）。

## 3. 关键设计决策与理由

### 3.1 课程是**数据**，不是类

`StageSpec` 是 `@dataclass`，一门课就是一个 `list[StageSpec]`。
`scripts/train.py` 直接 `for stage in stages:` 逐级调用 `Trainer.train_stage`。

**被否决的方案**：保留一个 `Curriculum` 类，带 `run()` / `table()` / `total_timesteps()`
与一个 `StageTrainer` Protocol。**否决理由**：那个类从 `scripts/train.py` 拿走控制权后，
要做 warm-start、每级学习率、每级产物命名，都必须在「类 → 协议 → 训练器」之间来回传参；
实际使用中 `train.py` 从来**没有构造过 `Curriculum`**，它自带 for 循环直接调
`train_stage` —— 于是整个驱动器成了孤儿（只有 `default_curriculum()` 在造它）。
删掉后本模块从 558 行降到 308 行，且没有任何行为变化。

教训：**当调用方绕过你的抽象层自己写循环时，那不是调用方的错，是抽象层的错**。

### 3.2 阶段表由「高度梯度 + 预算表」两张平行表决定，且要求高度升序

`STAGE_HEIGHTS_CM` 是固定六档；步数与学习率来自 `config.yaml` 的
`curriculum.timesteps` / `curriculum.learning_rate`（长度不足时由
`robotdog/config.py` 的 `_spread()` 线性重采样补齐）。

`config_to_stages()` 显式要求高度升序，否则报错。理由：课程的核心假设是
「下一级从上一级 warm-start」；高度乱序会让后一级比前一级更容易，
warm-start 的方向就反了，而这种错误从训练曲线上看不出来（成绩照样在涨）。

### 3.3 `EpisodeStats` 用滑动窗口，不累计

窗口（`window=50`）内的成功率才是「当前水平」；累计成功率会被早期的失败拖住，
在阶段切换后给出严重滞后的读数。**被否决的方案**：只留累计值 —— 否决，因为
训练日志里真正要看的是「最近 50 个回合」而不是「历史全部」。

`add_info` 只认 `info["episode"]`（`VecMonitor` 打的键）；没有这个键的 `info`
直接忽略并返回 `False`，所以调用方可以无脑遍历一批 `infos`。

### 3.4 `_spec_as_dict` 把地形摊平成 JSON 可序列化的字典

`vars(spec)` 拿到 dataclass 字段后，额外补 `friction`（tuple → list）、
`crest_height`、`up_end_x` 两个**派生属性**（它们不是字段）。理由：读
`summary.json` 的人想知道「这条楼梯多高、到哪结束」，而这两个量在
`StairSpec` 上是 property，`vars()` 看不到。

### 3.5 `write_summary` 只有一个参数顺序

签名是 `write_summary(path, stages, results=None, meta=None)`。

历史上它带过一个 `if isinstance(path, dict): path, meta = stages, path` 的重载，
用来兼容「先攒 payload、再给路径」的老调用习惯（`write_summary(payload, path)`）。
那个调用方是已被删除的 `Curriculum.run()`，**重载分支已随死代码清理删除** ——
现在传 dict 会被当成路径并在 `Path(dict)` 处抛 `TypeError`。
完整签名见 §4。

## 4. 公开 API

```python
STAGE_HEIGHTS_CM: tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)
DEFAULT_TREAD: float = 0.50
DEFAULT_N_UP: int = 8


@dataclass
class StageSpec:
    name: str
    step_height: float          # 台阶高 [m]，平地 = 0.0
    spec: StairSpec
    timesteps: int
    learning_rate: float
    label: str = ""
    description: str = ""

    @property
    def is_flat(self) -> bool: ...        # spec.n_up == 0 and spec.n_down == 0
    def summary(self) -> str: ...


@dataclass
class EpisodeStats:
    window: int = 50
    n_total: int = 0

    def __post_init__(self) -> None: ...
    def add(self, *, success: bool, fallen: bool, ret: float, length: float) -> None: ...
    def add_info(self, info: dict[str, Any]) -> bool: ...
    def add_infos(self, infos: Iterable[dict[str, Any]]) -> int: ...

    @property
    def n_window(self) -> int: ...
    @property
    def success_rate(self) -> float: ...
    @property
    def fall_rate(self) -> float: ...
    @property
    def mean_return(self) -> float: ...
    @property
    def mean_length(self) -> float: ...

    def as_dict(self) -> dict[str, float]: ...
    def format_line(self) -> str: ...


def _nanmean(values: Iterable[float]) -> float: ...      # 内部


@dataclass
class StageResult:
    name: str
    label: str = ""
    step_height: float = 0.0
    timesteps: int = 0
    mean_return: float = 0.0
    success_rate: float = 0.0
    fall_rate: float = 0.0
    wall_seconds: float = 0.0
    model_path: str = ""
    mean_length: float = 0.0
    episodes: int = 0
    obs_size: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]: ...


def format_curriculum(
    stages: Sequence[StageSpec],
    results: Sequence[StageResult] | None = None,
) -> str: ...


def write_summary(
    path: str | Path,
    stages: Sequence[StageSpec],
    results: Sequence[StageResult] | None = None,
    meta: dict[str, Any] | None = None,
) -> Path: ...


def _spec_as_dict(spec: StairSpec) -> dict[str, Any]: ...   # 内部


def format_seconds(seconds: float) -> str: ...   # 12.3s / 4.5min / 1.20h
```

`__all__` = `["STAGE_HEIGHTS_CM", "EpisodeStats", "StageResult", "StageSpec",
"format_curriculum", "format_seconds", "write_summary"]`
（`robotdog/curriculum.py:300-308`）

语义要点：

* `EpisodeStats.window` 不是构造后的只读量：`__post_init__` 用它给四个
  `deque(maxlen=window)` 定容量。构造时传入的 `_success` 等既有关卡会被保留。
* `_nanmean` 先 `np.isfinite` 过滤再平均，空数组返回 `0.0` 而不是 `nan`
  —— 报表里出现 `nan` 比出现 `0.0` 更难解释。
* `StageSpec.is_flat` 判的是 `spec` 的两级数，**不是** `step_height`
  （虽然两者在本仓总是一致的）。
* `format_curriculum` 的表格宽度按最长表头算，`results=None` 时只输出定义部分。
* `write_summary` 里 `out.parent.mkdir(parents=True, exist_ok=True)` 已包含建目录；
  不需要先调 `RunDir.ensure()`。
* `as_dict()` 的浮点都做了 `round`：`success_rate`/`fall_rate` 四位、`mean_return` 三位。

## 5. 内部结构与数据流

阶段定义 → 训练 → 报表的完整链路（跨模块）：

```
config.yaml (curriculum: heights_cm / tread / n_up / timesteps / learning_rate)
  → robotdog/config.py  config_to_stages()
      → flat_spec(tread=...)                    # 第 1 级
      → demo_stairs_spec(step_height=h/100, tread=..., n_up=...)   # 其余 5 级
      → [StageSpec, StageSpec, ...]
  → scripts/train.py    for stage in stages:
      → Trainer.train_stage(stage, venv, warm_start=...)
          → EpisodeStats.add_infos(infos)        # 每个 rollout 迭代
          → StageResult(...)                     # 阶段结束
      → write_summary(run.summary, stages, results, payload)
          — meta 里的 env_config / obs_config / reward_config / final_model /
            wall_seconds 被平铺到顶层，再补 stages / results / total_timesteps
```

本模块内部：

```
format_curriculum(stages, results)
  → by_name = {r.name: r for r in results}
  → 每个 stage 一行；results 非空时追加 成功率/摔倒率/均回报/用时/模型名
  → 末行 "合计 N 级, 总预算 M 步"

write_summary(path, stages, results, meta)
  → _spec_as_dict(s.spec) 逐级摊平地形
  → payload["results"] = [r.as_dict() ...]
  → payload["total_timesteps"] = sum(s.timesteps)
  → json.dumps(indent=2, ensure_ascii=False)
```

## 6. 关键常量与默认值

| 项 | 真实值 |
| --- | --- |
| `STAGE_HEIGHTS_CM` | `(0.0, 3.0, 6.0, 9.0, 12.0, 15.0)` —— 固定六档，平地 + 3/6/9/12/15 cm |
| `DEFAULT_TREAD` | `0.50` m |
| `DEFAULT_N_UP` | `8` |
| `EpisodeStats.window` 默认 | `50` |
| `format_seconds` 分界 | `< 60` → `12.3s`；`< 3600` → `4.5min`；否则 `1.20h` |
| `as_dict()` 舍入 | `success_rate`/`fall_rate` 4 位，`mean_return` 3 位，`mean_length` 1 位 |
| 文件行数 | 308 行 |

## 7. 与其它模块的关系

**上游**：stdlib（`json` / `time` / `collections.deque` / `dataclasses` / `pathlib` /
`typing`）+ `numpy` + `robotdog.tools.terrain.StairSpec`（`robotdog/curriculum.py:23`）。
**不 import `robotdog.env` / `robotdog.training` / `stable_baselines3`**。

**下游**：

| 位置 | 用到的符号 |
| --- | --- |
| `robotdog/config.py` | `StageSpec`、`STAGE_HEIGHTS_CM`、`DEFAULT_TREAD`、`DEFAULT_N_UP`（构造阶段表） |
| `scripts/train.py` | `EpisodeStats`、`StageResult`、`format_curriculum`、`write_summary`、`format_seconds` |

依赖方向是 `scripts/train.py → robotdog/curriculum.py → robotdog/tools/terrain.py`，
永远向下，不成环。

## 8. 实测与经验教训

* 六级课程的实测数字、每级产物清单，见 `train.md` §1、§2。
* 「抽象层被调用方绕过」这条教训的完整背景（旧的 `Curriculum` + `StageTrainer`
  被删除），见 `experience.md` 的死代码清理小节。
* 配置层如何把 `config.yaml` 的六档高度变成 `StageSpec` 列表，见
  `docs/robotdog/config.md` §4。

## 9. 已知限制与坑

* **没有断点续训**。`scripts/train.py` 的循环被杀了就只能从该级之前重来
  （`runs/<名字>/` 已存在时训练入口直接拒绝启动，避免覆盖）。
* **没有自动晋级**。每一级都训满 `timesteps`，哪怕上一级还没学会。
* **`EpisodeStats` 只在窗口内统计**：`n_total` 是累计回合数，但
  `success_rate` 只看最近 `window` 个。读日志时不要把两者混起来算。
* **`StageResult` 默认 `step_height=0.0`**，直接构造而不传该字段会得到
  「看起来像平地」的报表行。
* **`write_summary` 只有一个参数顺序**：`(path, stages, results, meta)`。老文档描述的
  `isinstance(path, dict)` 重载已随死代码清理删除，传 `dict` 给 `path` 会在
  `Path(dict)` 处抛 `TypeError`，不会静默走错分支。
* **本模块不再有 `Curriculum` / `StageTrainer` / `default_curriculum`**。
  在旧文档或旧脚本里看到这些名字，一律是过期引用。

## 10. 示例用法

```python
from robotdog.curriculum import (
    EpisodeStats,
    StageResult,
    StageSpec,
    format_curriculum,
    format_seconds,
    write_summary,
)
from robotdog.tools.terrain import demo_stairs_spec, flat_spec

stages = [
    StageSpec(name="flat", step_height=0.0, spec=flat_spec(n_up=0, n_down=0),
              timesteps=100_000, learning_rate=1.5e-4, label="平地"),
    StageSpec(name="stair3", step_height=0.03,
              spec=demo_stairs_spec(step_height=0.03, tread=0.50, n_up=8),
              timesteps=250_000, learning_rate=1.5e-4, label="3cm"),
]

print(format_curriculum(stages))          # 只有定义部分

stats = EpisodeStats(window=50)
stats.add(success=True, fallen=False, ret=412.5, length=168.0)
stats.add_info({"episode": {"r": -30.0, "l": 90.0}, "is_success": False, "fallen": True})
print(stats.format_line())
print(format_seconds(272.4))              # 4.5min

results = [StageResult(name="flat", label="平地", success_rate=1.0,
                       mean_return=330.87, wall_seconds=272.4,
                       model_path="runs/course1/checkpoints/flat_final.zip")]
print(format_curriculum(stages, results))
write_summary("runs/course1/summary.json", stages, results, meta={"seed": 0})
```
