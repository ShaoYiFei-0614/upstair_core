# `robotdog/curriculum.py` 设计文档

## 1. 一句话定位

**训练层**的「课程表 + 进度账本」：说明要练哪几级台阶、按什么顺序、每级训多少步，
并在每级结束后把实测结果记成一张表、汇总成 `summary.json`。它**不认识 PPO、也不碰奖励**。

## 2. 职责边界

**做**：

* 定义固定的六级梯度 `STAGE_HEIGHTS_CM = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)`（平地 + 五级台阶）。
* 把每一级包成 `StageSpec`（地形 `StairSpec` + 步数预算 + 学习率 + 中文标签）。
* 累计训练期的回合统计 `EpisodeStats`（滑动窗口）。
* 按 `StageTrainer` 协议**驱动**训练者跑完所有阶段，阶段之间做 warm-start。
* 把阶段定义与结果写成 `summary.json`，并渲染一行式表格。

**不做**：

* **不 import SB3**，不构造环境，不建网络。真正的 PPO 在 `robotdog/training.py`，
  装配在 `scripts/train.py`（它实现了 `StageTrainer`）。
* **不做地形随机化**。每一级只有**一种**几何，阶段内没有高度带、没有踏面网格。
* **不做毕业判定**。预算到了就换下一级，没有「成功率达标提前毕业」这回事
  （历史上那套判据会饿死 PPO，见 `experience.md` §4.1）。
* 不做评估、不渲染、不写 GIF。

## 3. 关键设计决策与理由

### 3.1 课程是**串行**的，一级一个几何

早期版本把一整条高度带同时铺到 8 个子环境上（8.0/8.4/…/11.0 cm）。那样做的动机是
防止灾难性遗忘，但代价是**每个子环境各有一份编译好的 MJCF**，而换地形要重编译
（`set_stair` 约 250 ms × `n_envs`），并且策略永远不知道自己站在哪一级台阶上。

现在是「一级 → 训练 → 存权重 → 下一级从上一级 `PPO.set_parameters` 续训」。好处：

* 每级只有一种几何，`set_stair` 只在**两个 `train_stage` 调用之间**发生，不打断 rollout
  （换地形会同时改变观测参照面与已累积的 rollout buffer 语义）。
* 权重迁移承担了「不遗忘」的责任，而不是让观测去承担。
* `obs.step_height` 这个 1 维标量就能告诉策略「现在在哪一级」，不需要高度带。

### 3.2 步数预算：平地给得少，台阶给得多

```python
_FLAT_BASE_STEPS  = 100_000     # 平地
_STAIR_BASE_STEPS = 250_000     # 每一级台阶
```

开环 trot 先验（`artifacts/trot_stairs.json`，5.48 Hz）本身就能在平地上走完全程，
所以平地段几乎不产生学习信号——实测成功率很快顶住，继续训只是空转。真正发生学习的
是台阶阶段。默认总预算 **1,350,000 步**（100k + 5 × 250k）。

`default_curriculum(total_timesteps=...)` 给出时按同样的 `100k : 250k` 权重比例缩放，
保证每级至少 1 步，**余数补给最后一级（最难的）**：

```python
budgets = [max(1, int(total * w / wsum)) for w in weights]
budgets[-1] += total - sum(budgets)
```

### 3.3 学习率随台阶升高而**下降**

```python
lrs = [1.5e-4, 1.5e-4, 1.2e-4, 1.0e-4, 8e-5, 6e-5]
```

高立面要求更细的更新。用同一个学习率往下走，一次过大的更新就可能把已经学会的
踏步节律踢掉，而策略再也回不到那个解（这一级没有更简单的起点）。

### 3.4 平地必须用 `flat_spec`，不能用 0 高度的楼梯

`demo_stairs_spec(step_height=0.0, n_up=8)` 会生成 `size="0.250000 1.000000 0.000000"`
的 box，MuJoCo 直接抛 `ValueError: Error: size 2 must be positive in geom`。所以
平地走 `flat_spec(n_up=0, n_down=0, step_height=0.0, ...)`，并且 `is_flat` 的判据是
`spec.n_up == 0 and spec.n_down == 0` 而不是 `step_height == 0.0`。

### 3.5 训练者用**结构化子类型**（`Protocol`）而不是基类

`StageTrainer` 只声明四个方法（`make_env` / `train_stage` / `save_model` / `close`），
`scripts/train.py` 里的一个普通类就能满足，不需要 import 本模块再继承。这样
`curriculum.py` 可以保持**零 SB3 依赖**——它是纯账本，测试时不需要装 torch。

### 3.6 `save_model` 允许抛 `NotImplementedError`

`Curriculum.run()` 用 `try/except NotImplementedError` 包住它，拿不到路径就退回去读
模型对象自己的 `model_path` / `path` 属性（`_model_path_of`）。这让一个只做内存内
训练的最小训练者也能跑课程。

## 4. 公开 API

```python
STAGE_HEIGHTS_CM: tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)
DEFAULT_TREAD = 0.50
DEFAULT_N_UP  = 8
DEFAULT_TOP_LANDING = 1.20
DEFAULT_WIDTH = 2.0
DEFAULT_START_X = 1.50

@dataclass
class StageSpec:
    name: str
    step_height: float          # [m]，平地 = 0.0
    spec: StairSpec
    timesteps: int
    learning_rate: float
    label: str = ""
    description: str = ""

    @property
    def is_flat(self) -> bool: ...        # spec.n_up == 0 and spec.n_down == 0
    def summary(self) -> str: ...

def default_curriculum(*, tread: float = 0.50, n_up: int = 8,
                       total_timesteps: int | None = None) -> list[StageSpec]: ...

@dataclass
class EpisodeStats:
    window: int = 50
    n_total: int = 0

    def add(self, *, success: bool, fallen: bool, ret: float, length: float) -> None: ...
    def add_info(self, info: dict[str, Any]) -> bool: ...      # 只有带 "episode" 的才算
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

class StageTrainer(Protocol):
    def make_env(self, stage: StageSpec, *, warm_start: str | None = None) -> Any: ...
    def train_stage(self, stage: StageSpec, venv: Any, *,
                    warm_start: str | None = None,
                    tensorboard_log: str | Path | None = None,
                    **kwargs: Any) -> tuple[Any, EpisodeStats]: ...
    def save_model(self, model: Any, stage: StageSpec, venv: Any) -> Path: ...
    def close(self) -> None: ...

@dataclass
class Curriculum:
    stages: list[StageSpec]
    results: list[StageResult] = field(default_factory=list)
    model_path: str | None = None
    summary_path: Path | None = None

    @classmethod
    def default(cls, *, total_timesteps: int | None = None) -> Curriculum: ...
    def names(self) -> list[str]: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def run(self, trainer: StageTrainer, *,
            resume_from: str | None = None,
            tensorboard_log: str | Path | None = None,
            summary_path: str | Path | None = None,
            meta: dict[str, Any] | None = None,
            verbose: int = 1,
            **train_kwargs: Any) -> list[StageResult]: ...
    def table(self) -> str: ...
    def total_timesteps(self) -> int: ...

def format_curriculum(stages: Sequence[StageSpec],
                      results: Sequence[StageResult] | None = None) -> str: ...
def write_summary(path, stages, results, meta) -> Path: ...
def format_seconds(seconds: float) -> str: ...
```

`__all__ = ["STAGE_HEIGHTS_CM", "Curriculum", "EpisodeStats", "StageResult",
"StageSpec", "StageTrainer", "default_curriculum", "format_curriculum",
"format_seconds", "write_summary"]`

语义要点：

* `EpisodeStats.add_info()` **只在 `info` 含 `dict` 类型的 `"episode"` 键时才记一笔**，
  其余 info 被静默忽略 —— 所以调用方可以无脑遍历整个 `infos` 列表。返回 `True`
  表示真的记了一笔。
* `success_rate` / `fall_rate` 是**窗口内**的；`n_total` 是**全程**的。前者用于进度显示，
  后者用于判断样本量够不够。
* `mean_return` / `mean_length` 走 `_nanmean`，会丢弃 `nan`（例如某些子环境没有回报时）。
* `Curriculum.run()` 每级结束后**务必**调用 `venv.close()`（在 `finally` 里），因为
  下一级会新建一个向量环境；不关会泄漏 8 个 MuJoCo 模型。
* `meta` 会被合进 `summary.json`，并且 `run()` 会补三个默认键
  （`stage_heights_cm`、`n_stages`、`total_timesteps`）与两个总是覆盖的键
  （`resume_from`、`final_model`）。

## 5. 内部结构与数据流

```
scripts/train.py
  config_to_stages(raw)                      # config.yaml 的 curriculum 段
      └─ flat_spec / demo_stairs_spec → [StageSpec × 6]
         （Curriculum.default() / default_curriculum() 是库内等价的旧入口，
            当前 CLI 不再调用它，见 §9）
  Curriculum.run(Trainer(), resume_from=…, summary_path=…, meta={env_config, obs_config, reward_config})
      for stage in stages:
          venv   = trainer.make_env(stage, warm_start=…)      # 重编译 MJCF
          model, stats = trainer.train_stage(stage, venv, warm_start=…)
          path   = trainer.save_model(model, stage, venv)      # checkpoints/<stage>_final.zip
          venv.close()
          warm   = path                                         # 下一级的 warm start
          results.append(StageResult(…stats…))
      write_summary(summary_path, stages, results, payload)
```

`EpisodeStats` 的填充路径：

```
VecMonitor 把 info["episode"] = {"r": …, "l": …} 挂在回合结束那一步的 info 上
    → scripts/train.py 的 ProgressCallback._on_step 收集 infos
    → EpisodeStats.add_infos(infos)      → 返回新增回合数
    → EpisodeStats.format_line()         → 一行进度
```

## 6. 关键常量与默认值

| 名称 | 值 | 说明 |
| --- | --- | --- |
| `STAGE_HEIGHTS_CM` | `(0.0, 3.0, 6.0, 9.0, 12.0, 15.0)` | 六级，平地 + 五级台阶 |
| `DEFAULT_TREAD` | `0.50` m | 踏面。**不要改**：实测 0.40 / 0.60 会让 34% 的成功率掉到 10%，0.65 直接 0% |
| `DEFAULT_N_UP` | `8` | 上行级数 |
| `DEFAULT_TOP_LANDING` | `1.20` m | 顶层平台长度 |
| `DEFAULT_WIDTH` | `2.0` m | 楼梯宽度 |
| `DEFAULT_START_X` | `1.50` m | 第一级立面的 x |
| `_FLAT_BASE_STEPS` | `100_000` | 平地预算 |
| `_STAIR_BASE_STEPS` | `250_000` | 每级台阶预算 |
| 默认总预算 | `1,350,000` 步 | `100k + 5 × 250k` |
| 学习率表 | `[1.5e-4, 1.5e-4, 1.2e-4, 1.0e-4, 8e-5, 6e-5]` | 索引与 `STAGE_HEIGHTS_CM` 对齐 |
| `EpisodeStats.window` | `50` | 滑动窗口长度 |
| 阶段名 | `flat`、`stair3`、`stair6`、`stair9`、`stair12`、`stair15` | 也是 `checkpoints/<名字>_final.zip` 的文件名；由 `config_to_stages()` 按高度推导 |

实测 `default_curriculum()` 展开：

```
flat     h=0.00cm  100,000 步  lr=1.5e-4
stair3   h=3.00cm  250,000 步  lr=1.5e-4
stair6   h=6.00cm  250,000 步  lr=1.2e-4
stair9   h=9.00cm  250,000 步  lr=1.0e-4
stair12  h=12.00cm 250,000 步  lr=8e-5
stair15  h=15.00cm 250,000 步  lr=6e-5
（全部 n_up=8, tread=0.50）
```

## 7. 与其它模块的关系

```
robotdog/tools/terrain.py   (StairSpec / flat_spec / demo_stairs_spec)
            ▲
robotdog/curriculum.py  ────►  scripts/train.py  (实现 StageTrainer，持有 PPO)
                                        │
                                robotdog/training.py  (make_vec_env / build_ppo / RunDir)
```

* **唯一的下游依赖是 `tools.terrain`**。没有 `env`、没有 `training`、没有 SB3 ——
  这是刻意的，它保证「课程表」可以在没有 torch 的环境里被读出来（文档、报表、
  `--help` 打印都不需要装深度学习栈）。
* **向上**只有 `scripts/train.py` 消费它。仓库里没有其它调用方。
* **与 `training.py` 的分工**：`training.py` 知道「怎么建环境、怎么建 PPO、存哪里」；
  `curriculum.py` 只知道「跑几级、每级多少步」。两者通过 `StageTrainer` 协议接触，
  由 `scripts/train.py` 这一个具体类把两边接起来。

## 8. 实测与经验教训

* `experience.md` §4.1 —— 旧的「成功率达标即毕业」判据会把 PPO 的梯度更新饿死
  （每级只跑 0~1 次更新，`mean|a| = 0.011`，策略退化成纯先验）。现在的课程**没有**
  这个门，只有步数预算，就是这条教训的直接结果。
* `experience.md` §4.2 —— 奖励归一化是温和的稳定器而非解药（3 cm 成功率
  0.474 → 0.539），所以 `norm_reward` 默认开但不要指望它单独解决问题。
* `train.md` §1.1 —— 六级课程表与学习率。
* `train.md` §3.1 —— 12,000 步冒烟六级全部跑完的实测输出。
* `README.md` §3.5 —— 踏面不可改、`geomgroup=1`、`CREST_SEAM_OVERLAP`。

## 9. 已知限制与坑

* **`Curriculum.run()` 没有断点续训**。进程在第三级中途被杀，就只能从**该级之前的
  checkpoint 手工重建**（当前 CLI 没有 `--resume` / `--from-stage` 一类入口，
  课程表与预算是 `config.yaml` 的 `curriculum` 段一次性给死的）。
  要把「从某个权重接着练」加回来，得在 `scripts/train.py` 里新接一条路径。
* **`default_curriculum()` / `Curriculum.default()` 现在没有调用方**。课程的构造权已经
  交给 `robotdog/config.py:config_to_stages()`（读 `config.yaml` 的 `curriculum` 段，
  见 `docs/robotdog/config.md` §3.4）。这两个入口保留在库里，是因为它们自带
  「1:2.5 权重分配 + 学习率表」这套**参考实现**，配置侧可以选择调用它们或自建
  `StageSpec`。改课程默认值时，两处都要看。
* **阶段切换期间没有验证**。每级结束后只报训练期的滑动窗口统计（采样动作口径），
  不是确定性评估；要确定性成功率请用 `scripts/play.py --mode eval`。
* `default_curriculum` 的 `tread` / `n_up` 会**同时**作用于平地与所有台阶阶段，
  没有「只改某一级」的入口。要非均匀课程就直接构造 `StageSpec` 列表。
* `_spec_as_dict()` 写进 `summary.json` 的字段是 `vars(spec)` 加 `crest_height` /
  `up_end_x`，`friction` 被转成 list（JSON 没有元组）。读回时不要假设它是 tuple。
* `StageResult.episodes` 是**窗口之外**的总回合数（`stats.n_total`），
  而 `success_rate` 是窗口内的——两者口径不同，别相除。
* `format_curriculum()` 与 `StageSpec.summary()` 的行宽是对齐好的，改标签长度会错位。

## 10. 示例用法

```python
from robotdog.curriculum import Curriculum, default_curriculum, format_curriculum

# 看课程表（不需要 SB3、不需要环境）
stages = default_curriculum()
print(format_curriculum(stages))

# 缩到 30 万步跑一遍（只在测试里这么干）
cur = Curriculum.default(total_timesteps=300_000)
for s in cur:
    print(s.name, s.timesteps, f"{s.step_height*100:.1f}cm", s.is_flat)

# 真正驱动训练（Trainer 由 scripts/train.py 提供）
# cur.run(trainer,
#         resume_from=None,
#         summary_path="runs/walk1/summary.json",
#         meta={"env_config": ..., "obs_config": ..., "reward_config": ...})

# 单独用统计累加器
from robotdog.curriculum import EpisodeStats
stats = EpisodeStats(window=50)
stats.add(success=True, fallen=False, ret=123.4, length=210.0)
print(stats.format_line())
```
