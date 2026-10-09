# `robotdog/curriculum.py` — 课程学习与训练回调

## 1. 一句话定位

训练层（第 6 步）：把「平地 → 3~8 cm → 8~15 cm → 15~22 cm，每阶段成功率高就升难度」这条规格编码成 `StageSpec` 列表，并提供一组 SB3 `BaseCallback` 在训练过程中统计成功率、决定何时毕业、什么时候存检查点。它不训练、不组装环境，也不评估最终策略。

## 2. 职责边界

**做：**

- 默认四阶段课程表（`default_curriculum`）与把一个阶段拆成上升阶梯（`expand_stage` / `expand_curriculum`）。
- 一个阶段内让不同子环境待在不同台阶高/踏面（`mixed_stage_specs` / `mixed_grid_specs`），把整条能力带塞进每个梯度批次。
- 毕业判定：滚动成功率的统计（`EpisodeStats`）与三条件判定（回合数 + 成功率 + 步数下限）。
- 训练过程回调：阶段性日志与 tensorboard 标量、周期检查点、阶段内最优模型、墙钟限时。
- 阶段末的**确定性**成功率测量（`deterministic_success`）。
- 按名字挑阶段（`select_stages`）与写课程报告（`write_curriculum_report`）。

**不做：**

- 不做 `model.learn()` 的循环、不切地形、不写 `summary.json`——这些都在 `scripts/train.py`。
- **不自己换地形**：`_on_step` 返回 `False` 只是提前结束 `learn()`，切换由驱动脚本在返回后做。原因见 §3.6。
- 不实现成功判据：`is_success` 来自 `Go2StairsEnv._success_reached`，本模块只读 `info`。
- 不组装 PPO、不建向量化环境（`robotdog/training.py`）。
- 不做多种子/多 run 的调度与并发控制。

## 3. 关键设计决策与理由

### 3.1 默认阶段表用区间中点，不用每回合随机高度

```python
def default_curriculum(step_heights: Sequence[float] = (0.05, 0.12, 0.19),
                       tread: float = 0.50, n_up: int = 8) -> list[StageSpec]:
```

用户规格给的是区间（3~8 / 8~15 / 15~22 cm），实现取**区间中点**。理由写在 docstring 里：地形每阶段重编译（`set_stair` 重建 MJCF，约 250 ms），所以每回合随机化高度不可行。区间随机化的替代方案由 `mixed_stage_specs` 部分实现——不是每回合变，而是每个**子环境**固定一个高度。

四阶段预算（真实值）：

| 阶段 | 地形 | `target_success` | `min_episodes` | `max_timesteps` | `label` / `description` |
| --- | --- | --- | --- | --- | --- |
| `flat` | `flat_spec(n_up=0, n_down=0, step_height=0.0, tread=0.50, width=2.0, start_x=1.50)` | 0.90 | 30 | 250,000 | `平地` / `平地行走, 建立稳定的步态与前进速度` |
| `easy` | `demo_stairs_spec(step_height=0.05, tread=0.50, n_up=8, top_landing=1.20)` | 0.80 | 40 | 300,000 | `3~8cm 台阶` / `5 cm 台阶 (3~8 cm 区间中点), 学会抬腿跨越` |
| `medium` | `demo_stairs_spec(step_height=0.12, ...)` | 0.65 | 40 | 350,000 | `8~15cm 台阶` / `12 cm 台阶 (8~15 cm 区间中点)` |
| `hard` | `demo_stairs_spec(step_height=0.19, ...)` | 0.50 | 40 | 400,000 | `15~22cm 台阶` / `19 cm 台阶 (15~22 cm 区间中点)` |

`flat_spec` 显式传 `step_height=0.0`：`StairSpec` 的默认值是 `0.05`（为了让 `demo_stairs_spec` 的默认值统一），但 `n_up=0` 的平地留 0.05 会让人误以为平地段有 5 cm 台阶，也让 level 报表出现假高度。

### 3.2 毕业条件必须是三件事同时成立

```python
ready = (self.stats.n_total >= self.stage.min_episodes
         and self.stats.success_rate >= self.stage.target_success
         and steps_ok)          # steps_ok = steps >= self.stage.min_timesteps
```

这是 `StageSpec.min_timesteps` 存在的**唯一**理由，也是课程学习最容易踩的坑：有强开环步态先验时，**未训练**的策略第一轮就达标，`learn()` 立刻返回，每阶段只拿到 0~1 次 PPO 更新。PPO 一次更新的粒度是 `n_steps 256 × n_envs 8 = 2048` 步。

证据（course7，`easy4_final`，3.5 cm × 5 级）：

- 策略 75.0% vs 纯先验 75.0% vs 随机 25.0%——**完全一样**。
- `mean|a| = 0.0109`，`|a| < 0.05` 占 99.9% 的步——策略输出约等于 0，即「什么都别做，让先验走」。
- 各档实际步数 1824 / 2272 / 2752 / 3456 → **0 / 1 / 1 / 1 次**梯度更新。

配套的驱动侧旋钮是 `scripts/train.py --min-stage-frac F`（默认 `0.5`），把每档 `min_timesteps` 设成 `int(max_timesteps × F)`；**`--min-stage-frac 0` 恢复旧行为**。详见 `experience.md` §4.1 与 `train.md` §3.3。

**被否决的替代方案：** 只调高 `min_episodes`。否决理由是粒度不对——`min_episodes` 数的是回合，而 PPO 饿死是**步数**问题；20 s 的回合长度下 40 回合可能只有几千步，仍然不足一次更新。

### 3.3 `deterministic_success` 与 `EpisodeStats.success_rate` 是两个不同的量，两个都要报

`EpisodeStats.success_rate` 统计的是 `learn()` 的 rollout，那些用**采样**动作——PPO 每步给 12 个动作维度加 `log_std ≈ 0.137` 的探索噪声。所以它衡量的是「探索噪声能不能活过楼梯」，不是「策略能不能爬上去」，会**系统性低估**交付策略。

实测（`course15`）：课程以 **10%** 关掉了 `medium3`（7.8 cm × 8），而同一个 checkpoint 确定性跑 15 回合是 **33%**——差距全来自动作噪声。

所以阶段结束（毕业或预算耗尽）时额外跑一次 `deterministic_success`，两个数字都进 `result()`。没有第二个数字，课程会把一个几乎解决的阶段当成翻不过去的墙，把整个预算耗在上面。

### 3.4 `mixed_stage_specs` 不做每回合随机化，而是每子环境固定一个高度

把 8.0 → 8.5 → 9.0 → 9.5 → 10.0 cm 一级一级爬，每一级会成为 PPO 梯度唯一见到的地形，策略**灾难性遗忘**已经越过的台阶（每级只在它训练的那个高度最好，且都比参考策略在 7.5~8.0 cm 上更差）。让一个阶段内各子环境处于不同高度，整条带就都出现在每个梯度批次里。

**刻意不用每回合随机化**：`set_stair` 每个子环境都重编译 MJCF（约 250 ms），每个 `reset()` 都重编译会比训练本身更耗时。高度线性铺开且**包含两端**，避免批次被两端主导。这是「一次全爬上去」的对立面——见 `expand_stage` 的 `n_up_start`（§3.5）。

### 3.5 `mixed_grid_specs`：更长踏面不给余量，只让起跨失步

`course21` final，每格 20 个确定性回合（通过缩放 `n_up` 把上升段固定在 4.0 m，免得 20 s 上限混淆结果），10.0 cm 台阶：

| 踏面 | 35 cm | 50 cm | 65 cm | 80 cm |
| --- | --- | --- | --- | --- |
| 成功率 | 0% | 55% | 0% | 0% |

⇒ **更长的踏面不会多给调整时间**：65/80 cm 崩到 0%，35 cm 也崩。策略是围绕训练时的 50 cm 调好的，残差步态与它相位锁定，所以**改变**踏面是让起跨失步。结论不是「用更长踏面」，而是「别让策略依赖某一种踏面」：让各子环境用不同踏面，逼迫起跨对任意到达相位都成立。

网格分配：高度轴拿大头（`n_h = n_envs // n_t`），踏面轴占 2 个子环境（`n_envs >= 4 and n_envs % 2 == 0` 时；否则 1 个），8 个子环境 = 4 高度 × 2 踏面。`n_envs == 1` 取右上角（最高最长），与 `mixed_stage_specs` 一致。

### 3.6 阶段切换不在 rollout 中间发生

`CurriculumCallback._on_step` 返回 `False` 提前结束 `learn()`；换地形由驱动脚本在 `learn()` 返回之后做。理由：换地形要重编译 MJCF，中途换会让当次 rollout 的观测/状态不一致（同一批 GAE 里出现两种地形的高度场）。

### 3.7 `expand_stage`：开环先验对台阶高**不单调**，所以首级高度要能覆盖

平地调好的先验只把最低足抬约 20 mm，所以直接上 5 cm 立面就是墙。

**实测（纯开环 trot，`start_step_height` 这个覆盖参数的存在理由）：**

| 台阶高 | 2 cm | 3 cm | 4 cm | 5 cm | 6 cm |
| --- | --- | --- | --- | --- | --- |
| 成功率 | 40% | 40% | **100%** | 20% | 0% |

固定 183 ms 的步态周期在 ~4 cm 处与 50 cm 踏面**共振**。所以从 2 cm 起步的阶梯会从**先验自身能力之下**开始，白费第一级。`start_step_height`（绝对米数）覆盖第一级的 `start_fraction`。

**`n_up_start`：第二个独立难度轴 = 立面的数量。** 终止成功奖励对整段楼梯是全有全无的：8 级 × 每级存活 0.85 = **0.27** 成功率，正是前四次训练撞到的平台（先验 25%，PPO 75k 步后 25~32%）。从 **2 级**起步把同样的每级能力变成约 **72%** 成功率，策略才能拿到密集的 +20 终止奖励而不是稀有的那一个。

`expand_curriculum` 只把 `start_step_height` / `n_up_start` 用在**第一个**台阶阶段上：之后的台阶阶段从上一阶段满高起步。否则 `--stage easy,medium --sub-start-cm 4` 会把 `medium` 展开成 4/6.7/9.3/12 cm，白费第一块预算重学 `easy` 已解决的级。

### 3.8 早级用更高的目标成功率

`expand_stage` 里每个子级都设 `target_success=max(stage.target_success, 0.80)`（源码注释：`Early rungs must be cleared comfortably before the riser grows.`），并把 `min_episodes` 减半（下限 8）。理由：早期级是「先给策略能清掉的地形」，如果按 0.50 就放行，立面还没长起来策略就先退化了。

### 3.9 `BestModelCallback` 需要 `floor`，且 `model_best.zip` 不保证存在

`floor` 让驱动跨阶段沿用最佳速率。新实例从 `best = -1.0` 开始，没有 floor 的话**后面**更难阶段的第一次成功就会覆盖文件——19 cm 上 3% 会替换掉平地 100% 的策略。

另一个必须记住的事实：`model_best.zip` 只在 run **毕业过至少一个阶段**时才写。`course31` / `course49` 都是从 `course22` 续训的，续训后没有重新毕业任何阶段，`summary.json` 的 `best_stage` 是 `null`——尽管 `artifacts.best_model` 里还写着那个路径。交付权重分别是 `artifacts\best_medium.zip` 与 `model_final.zip`。见 `train.md` §3.2。

### 3.10 域随机化的 severity ramp

`randomization_on` 只表示驱动已经把开关打开；ramp 再把 `randomize_scale` 从 0 走到 1，让策略先在**标称**模型上解决本阶段，再遇扰动。**不 ramp 的实测**：全量随机化 368k 步只到 26% 采样成功率（目标 65%）。ramp 是单调不减的（只在 `scale > self._dr_scale` 时下发），中途绝不降低，否则策略会在两个难度间来回震荡。`dr_ramp_steps < 0` 表示**故意禁用** ramp（直接 1.0）。

### 3.11 `select_stages` 过去会把字符串逐字符迭代

`"flat,easy"` 裸字符串曾被逐字符迭代，于是报 `未知阶段 [',', 'a', 'e', ...]`——一个单字符笔误被报成这种错很难读。现在 `isinstance(names, str)` 时先按 `,` 切分并 strip。

## 4. 公开 API

### `deterministic_success`

```python
def deterministic_success(model, spec: StairSpec, cfg: Any = None, *,
                          n_episodes: int = 10, seed: int = 12345) -> float | None:
```

- 语义：在 `spec` 上跑**确定性**策略，返回成功率。手工复现 `VecNormalize` 的观测归一化：`x = (x - obs_rms.mean) / np.sqrt(obs_rms.var + eps)` 再 `np.clip(x, -clip_obs, clip_obs)`，然后 `model.predict(x.astype(np.float32), deterministic=True)`。不建向量化环境——直接 `Go2StairsEnv(stair=spec, cfg=cfg, seed=seed)`。
- 参数：`model` 任意有 `get_env()` / `predict()` 的对象；`cfg=None` 时用 `EnvConfig()` 默认；`n_episodes` 被 `max(1, int(n_episodes))` 夹住；`seed` 逐回合递增（`seed + ep`）；归一化统计来自 `getattr(venv, "obs_rms", None)` / `clip_obs`（默认 10.0）/ `epsilon`（默认 1e-8）。
- 返回：`float | None`——`wins / max(1, int(n_episodes))`；**测不出来返回 `None`**，使测量失败永远不会打断训练。
- 异常：**不抛**。`model.get_env()` 失败、环境构造失败、rollout 中断全部 `return None`。`env.close()` 在 `finally` 里。
- 关键防御顺序：先问 `get_env()` 再编译场景——单元测试用手写 stub 驱动 `_on_step()`，没有 `get_env`，否则测量本身会崩掉它要保护的那个 callback。

### `class StageSpec`

```python
@dataclass
class StageSpec:
    name: str
    label: str
    spec: StairSpec
    target_success: float
    min_episodes: int
    max_timesteps: int
    min_timesteps: int = 0
    description: str = ""
```

- `name`：唯一标识，`select_stages` 按它匹配；`expand_stage` 用 `f"{stage.name}{i + 1}"` 生成子级名。
- `label`：人读的中文标签。
- `spec`：本阶段地形。
- `target_success`：滚动成功率超过它就毕业。
- `min_episodes`：回合数下限。
- `max_timesteps`：硬预算；预算耗尽但没达标时是否推进由 `advance_on_budget` 决定。
- `min_timesteps`：**步数下限**，见 §3.2。
- `description`：进日志。
- 方法 `def summary(self) -> str:` 返回单行阶段摘要：`f"{self.name:<7} {self.label:<12} 台阶高={s.step_height * 100:5.1f}cm 踏面={s.tread * 100:5.1f}cm 级数={s.n_up:<2d} 顶端={s.crest_height:5.2f}m | 目标成功率={self.target_success:.0%} 最少回合={self.min_episodes} 预算={self.max_timesteps:,}{extra}"`，其中 `extra = f" 最少步数={self.min_timesteps:,}" if self.min_timesteps > 0 else ""`。

### `default_curriculum`

```python
def default_curriculum(step_heights: Sequence[float] = (0.05, 0.12, 0.19),
                       tread: float = 0.50, n_up: int = 8) -> list[StageSpec]:
```

- 语义：返回 §3.1 表格里的四阶段，顺序 `flat` → `easy` → `medium` → `hard`。
- 参数：`step_heights` 被解包成 `h_lo, h_mid, h_hi`（**长度必须恰好为 3，否则 `ValueError: not enough values to unpack`**）；`tread` 同时用于平地与三级台阶；`n_up` 只影响台阶阶段。
- 返回：`list[StageSpec]`。
- 异常：`step_heights` 长度不是 3 时解包报错。

### `mixed_stage_specs`

```python
def mixed_stage_specs(spec: StairSpec, n_envs: int, lo_cm: float,
                      hi_cm: float) -> list[StairSpec]:
```

- 语义：返回 `n_envs` 个 spec，台阶高在 `[min(lo_cm,hi_cm)/100, max(...)/100]` 上**线性铺开且含两端**。
- 参数：单位是 **cm**；`n_envs = max(1, int(n_envs))`。
- 返回：`n_envs == 1` 或 `hi <= lo` 时返回 `[replace(spec, step_height=hi)]`；否则 `[replace(spec, step_height=round(lo + span * i / (n_envs - 1), 6)) for i in range(n_envs)]`。
- 异常：无。

### `mixed_grid_specs`

```python
def mixed_grid_specs(spec: StairSpec, n_envs: int, *,
                     heights_cm: tuple[float, float],
                     treads: tuple[float, float]) -> list[StairSpec]:
```

- 语义：台阶高 × 踏面网格，见 §3.5。高度轴铺满全范围，踏面轴占 `n_t` 个子环境；子环境数不足时用最后一个 spec 补齐再切片。
- 参数：两个区间都以 **cm** 给出，函数内部 `sorted` 后 `/100`；`n_t` 规则：`t_hi <= t_lo` → 1；`n_envs >= 4 and n_envs % 2 == 0` → 2；否则 1。`spread(lo, hi, n)` 在 `n == 1 or hi <= lo` 时返回 `[hi]`，否则 6 位小数的等分。
- 返回：`n_envs == 1` 时 `[replace(spec, step_height=h_hi, tread=t_hi)]`；否则长度恰好 `n_envs` 的列表。
- 异常：无。

### `expand_stage`

```python
def expand_stage(stage: StageSpec, n_levels: int = 4, *,
                 start_fraction: float = 0.4,
                 min_step_height: float = 0.02,
                 start_step_height: float | None = None,
                 n_up_start: int | None = None) -> list[StageSpec]:
```

- 语义：把一个台阶阶段拆成 `n_levels` 个**上升**的子级。高度 `heights[i] = lo + (full - lo) * i / (n_levels - 1)`，且 `heights[-1]` 被强制成 `full`（最后一级**恰好**是原 `stage.spec` 的高度，保持原合同）。预算均分 `per = max(1, stage.max_timesteps // n_levels)`，余数给最后一级。
- 参数：`lo = max(min_step_height, full * start_fraction)`，被 `start_step_height` 覆盖时 `lo = max(min_step_height, min(start_step_height, full))`；`n_up_start is None` 时每级都用 `full_n_up`，否则从 `n_lo = max(1, min(n_up_start, full_n_up))` 线性升到 `full_n_up` 且 `n_ups[-1] = full_n_up`。
- 返回：`n_levels == 1` 或 `stage.spec.n_up == 0` 时返回 `[stage]`（**同一个对象**，不复制）；否则 `n_levels` 个新 `StageSpec`，每个：`name=f"{stage.name}{i + 1}"`、`label=f"{stage.label.rsplit(' ', 1)[0]} {h * 100:.1f}cm"`、`spec=replace(stage.spec, step_height=h, n_up=n_ups[i])`、`target_success=max(stage.target_success, 0.80)`、`min_episodes=max(8, stage.min_episodes // 2)`、`description=f"{stage.description} | 子级 {i + 1}/{n_levels}: 台阶高 {h * 100:.1f}cm x {n_ups[i]} 级"`。
- 异常：无。

### `expand_curriculum`

```python
def expand_curriculum(stages: Sequence[StageSpec], n_levels: int = 4,
                      **kw) -> list[StageSpec]:
```

- 语义：对每个台阶阶段套 `expand_stage`（平地原样通过）。`start_step_height` 只钉住**第一个**台阶阶段的最低一级，之后的台阶阶段从上一阶段满高起步；`n_up_start` 同理只对第一个台阶阶段生效（`stage_kw["n_up_start"] = None`）。
- 参数：`**kw` 原样透传给 `expand_stage`。
- 返回：展开后的完整阶段列表。
- 异常：`kw` 里有 `expand_stage` 不接受的键时 `TypeError`。

### `class EpisodeStats`

```python
class EpisodeStats:
    def __init__(self, window: int = 50) -> None:
```

- 语义：已完成回合的滚动窗口。四条 `deque(maxlen=window)`：`success`、`fall`、`ret`、`length`；`n_total` 从 0 起累加。
- `def add(self, *, success: bool, fallen: bool, ret: float, length: float) -> None:` 全部按关键字调用，`success`/`fallen` 存 1.0/0.0。
- 属性：`n_window` → `len(self.success)`；`success_rate` → `float(np.mean(self.success)) if self.success else 0.0`；`fall_rate` 同形；`mean_return` / `mean_length` 空时为 0.0，否则只对有限值取均值。
- `def as_dict(self) -> dict[str, float]:` → 键 `episodes_total`、`episodes_window`、`success_rate`（round 4）、`fall_rate`（round 4）、`mean_return`（round 3）、`mean_length`（round 1）。

### `class CurriculumCallback`

```python
class CurriculumCallback(BaseCallback):
    def __init__(
        self,
        stage: StageSpec,
        *,
        window: int = 50,
        advance_on_budget: bool = True,
        print_every_episodes: int = 20,
        verbose: int = 1,
        cfg: Any = None,
        det_episodes: int = 10,
        stage_index: int = 0,
        randomization_on: bool = False,
        dr_ramp_steps: int = 0,
    ) -> None:
```

- 语义：跟踪当前阶段成功率，达标即让 `learn()` 停下。`_on_training_start` 记 `start_timesteps = model.num_timesteps`、重置 `_floor_warned` 与墙钟；打开 DR 时下发 `set_domain_randomization(scale=...)`。
- `_on_step() -> bool`：读 `self.locals["dones"]` / `self.locals["infos"]`，对每个结束的子环境取 `info["is_success"]`、`info["fallen"]`、`info["episode"]["r"]` / `["l"]` 累进 `stats`。每 `print_every_episodes` 回合打印一行并记 tensorboard：`curriculum/success_rate`、`curriculum/fall_rate`、`curriculum/mean_return`、`curriculum/mean_ep_length`、`curriculum/episodes`、`curriculum/stage_steps`（+ DR 时 `curriculum/dr_scale`）。
- `_dr_ramp_span() -> int`：`self.dr_ramp_steps if > 0 else max(1, int(self.stage.min_timesteps))`——默认取阶段最小预算，ramp 恰好在阶段**允许毕业**时满量。
- 三条件判定见 §3.2。步数未满但成功率已达标时打印一次：`[课程] {name}: 成功率 {..:.1%} 已达标, 但最少训练步数 {min_timesteps:,} 未满 (当前 {steps:,}) -- 继续训练以让 PPO 真正学到东西`。
- 毕业时 `stop_reason = f"毕业: 成功率 {rate:.1%} >= 目标 {target:.0%} (共 {n_total} 回合)"`；预算耗尽时 `advance_on_budget=True` → `f"预算耗尽({max:,} 步): 成功率 {rate:.1%} < 目标 {target:.0%} -- 仍推进 (advance_on_budget=True)"`，否则 `f"预算耗尽({max:,} 步): 成功率 {rate:.1%} 未达标, 停留在本阶段"`。
- `stop_reason != "running"` 时调一次 `deterministic_success(self.model, self.stage.spec, self.cfg, n_episodes=self.det_episodes, seed=12345 + self.stage_index)`，打印 `[课程] {name}: 确定性成功率 {d:.1%} (随机动作下 {s:.1%}, {det_episodes} 回合)` 与 `[课程] <<< 阶段 '{name}' 结束: {stop_reason}`，返回 `False`。
- `def result(self) -> dict[str, Any]:` → 键 `stage`、`label`、`step_height`、`n_up`、`tread`、`crest_height`、`target_success`、`success_rate_deterministic`（`None` 或 round 4）、`graduated`、`stop_reason`、`timesteps`（`num_timesteps - start_timesteps`），再并入 `stats.as_dict()`。**地形形状必须跟高度一起记录**：阶段由 `(step_height, n_up)` 定义，没有 `n_up` 的话 `scripts/render_checkpoint.py` / `scripts/eval.py` 无法只凭 `summary.json` 重建同一地形。

### `class PeriodicCheckpointCallback`

```python
class PeriodicCheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_dir: str | Path,
        save_every: int,
        *,
        name_prefix: str = "ckpt",
        keep: int = 5,
        verbose: int = 1,
    ) -> None:
```

- 语义：按**总**环境步保存（相对 `save_freq` 的语义无歧义），同时镜像 `VecNormalize` 统计使 checkpoint 可独立加载，只保留最新 `keep` 个编号检查点。`_on_training_start` 把 `_last_save` 设为当前 `model.num_timesteps`（续训时不会立刻重存）。文件名 `f"{name_prefix}_{num_timesteps}_steps"`。
- `_prune()`：`keep <= 0` 直接返回；按 `int(p.stem.split("_")[-2])` 排序 `glob(f"{name_prefix}_*_steps.zip")`，删掉除最新 `keep` 个之外的全部，每个连同 `old.stem + "_vecnormalize.pkl"`，`FileNotFoundError` 忽略。
- `self.saved: list[str]` 记录写出的 `.zip` 路径（字符串，带 `.zip` 后缀）。

### `class BestModelCallback`

```python
class BestModelCallback(BaseCallback):
    def __init__(self, save_path: str | Path, monitor: CurriculumCallback,
                 *, floor: float = -1.0, verbose: int = 1) -> None:
```

- 语义：滚动成功率提升时保存；`_on_step` 先要求 `s.n_total >= self.monitor.stage.min_episodes`，然后 `threshold = max(self.best, self.floor)`，`s.success_rate > threshold + 1e-9` 才写。同时写 `<stem>_vecnormalize.pkl`。
- 参数：`floor` 见 §3.9；`self.best` 初值 `-1.0`。
- 返回：`_on_step` 永远返回 `True`（不停训练）。

### `class WallClockLimitCallback`

```python
class WallClockLimitCallback(BaseCallback):
    def __init__(self, max_seconds: float, verbose: int = 1) -> None:
```

- 语义：墙钟预算后中止 `learn()`。`_on_training_start` 记 `self._t0 = time.time()`；`_on_step` 超时则打印 `[限时] 超过 {max_seconds:.0f}s, 停止本阶段训练` 并返回 `False`。

### `select_stages`

```python
def select_stages(all_stages: Sequence[StageSpec], names: Sequence[str] | str) -> list[StageSpec]:
```

- 语义：按名挑阶段并保持课程顺序。`names` 可以是名字序列或逗号分隔字符串（`"flat,easy1"`）。
- 返回：`all_stages` 的浅拷贝列表（`not names or "all" in names` 时）。
- 异常：有名字匹配不上时 `raise ValueError(f"未知阶段 {sorted(missing)}; 可选 {[s.name for s in all_stages]}")`。

### `write_curriculum_report`

```python
def write_curriculum_report(path: str | Path, results: list[dict[str, Any]],
                            extra: dict[str, Any] | None = None) -> Path:
```

- 语义：写 `{"stages": results}`，`extra` 非空时 `payload.update(extra)`；`path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")`，父目录先建。
- 返回：`Path`（`path` 本身）。

## 5. 内部结构与数据流

课程构造：

```
default_curriculum()
  → stage "flat" / "easy" / "medium" / "hard"
      → expand_curriculum(stages, n_levels, start_step_height=..., n_up_start=...)
          → expand_stage(stage, n_levels, ...)              # 台阶阶段
              → replace(stage.spec, step_height=..., n_up=...)
          → stage（平地原样通过）
      → select_stages(all_stages, names)                     # 按名挑 + 保顺序
```

一个阶段内的训练循环：

```
scripts/train.py
  → set_terrain / set_terrain_per_env                     # 见 training.md
  → build_ppo / load_ppo
  → model.learn(total_timesteps=stage.max_timesteps,
                callback=CallbackList([curriculum_cb, checkpoint_cb, best_cb, wall_cb]))
      → CurriculumCallback._on_training_start()            # 记 start_timesteps, 起 DR ramp
      → 每个 env step: CurriculumCallback._on_step()
          → self.locals["dones"] / ["infos"]
          → EpisodeStats.add(success=…, fallen=…, ret=…, length=…)
          → (每 20 回合) print + logger.record(...)
          → (DR) training_env.env_method("set_domain_randomization", scale=…)
          → 三条件判定:
               毕业 → graduated=True, stop_reason="毕业: …", 调 deterministic_success(), return False
               预算耗尽 → stop_reason="预算耗尽…", advance_on_budget 决定 return False 与否
      → PeriodicCheckpointCallback._on_step() → model.save + _save_vecnormalize + _prune
      → BestModelCallback._on_step()          → model.save + <stem>_vecnormalize.pkl
  → CurriculumCallback.result()                            # 进 summary.json 的 stages[]
  → write_curriculum_report(LOGS_DIR / f"{run_name}_curriculum.json", results, extra)
```

## 6. 关键常量与默认值

| 名称 | 值 | 说明 |
| --- | --- | --- |
| `deterministic_success.n_episodes` | `10` | 阶段末确定性测量 |
| `deterministic_success.seed` | `12345` | 实际逐回合用 `seed + ep`；callback 传 `12345 + stage_index` |
| `deterministic_success` 归一化默认 | `clip_obs=10.0`、`epsilon=1e-8` | 缺属性时的兜底 |
| `default_curriculum.step_heights` | `(0.05, 0.12, 0.19)` | 单位米 |
| `default_curriculum.tread` / `n_up` | `0.50` / `8` | |
| `flat` 预算 | `target_success=0.90`、`min_episodes=30`、`max_timesteps=250_000` | |
| `easy` 预算 | `0.80` / `40` / `300_000` | |
| `medium` 预算 | `0.65` / `40` / `350_000` | |
| `hard` 预算 | `0.50` / `40` / `400_000` | |
| `StageSpec.min_timesteps` 默认 | `0` | 设 0 即旧行为 |
| `expand_stage.n_levels` | `4` | |
| `expand_stage.start_fraction` | `0.4` | |
| `expand_stage.min_step_height` | `0.02` | 20 mm |
| `expand_stage` 子级 `target_success` | `max(stage.target_success, 0.80)` | |
| `expand_stage` 子级 `min_episodes` | `max(8, stage.min_episodes // 2)` | |
| `mixed_grid_specs` 踏面轴宽度 | `n_t ∈ {1, 2}` | 见 §3.5 |
| `EpisodeStats.window` | `50` | |
| `CurriculumCallback.print_every_episodes` | `20` | |
| `CurriculumCallback.det_episodes` | `10` | |
| `CurriculumCallback.advance_on_budget` | `True` | `scripts/train.py --no-advance-on-budget` 关掉 |
| `CurriculumCallback.dr_ramp_steps` | `0` | `0` → 用 `max(1, stage.min_timesteps)`；`< 0` → 立即全量 |
| `PeriodicCheckpointCallback.keep` | `5` | `keep <= 0` 不删 |
| `PeriodicCheckpointCallback.name_prefix` | `"ckpt"` | `scripts/train.py` 传阶段名 |
| `BestModelCallback.floor` | `-1.0` | |
| `BestModelCallback.best` 初值 | `-1.0` | |
| 成功奖励 | `reward.success = 20.0` | 全有全无，`n_up_start` 的理由 |
| 单级存活率 | 约 `0.85` | 8 级 → 0.27 的由来 |
| 开环先验对台阶高的成功率 | 2/3/4/5/6 cm → 40/40/100/20/0% | 183 ms 周期与 50 cm 踏面在 ~4 cm 共振 |

## 7. 与其它模块的关系

```
robotdog/tools/terrain.py (StairSpec, demo_stairs_spec, flat_spec)
robotdog/env.py          (Go2StairsEnv — 只在 deterministic_success 里**局部** import)
        ↓
robotdog/curriculum.py    ← 本模块
        ↑
scripts/train.py          (唯一的驱动方：select_stages / expand_curriculum / 回调)
        ↓
robotdog/training.py      (同层：本模块不 import 它，靠鸭子类型调 model.get_env()/predict())
```

- **依赖 `robotdog/tools/terrain.py`**：阶段表就是地形表。方向不能反——`terrain.py` 不知道阶段、不知道毕业条件。
- **`Go2StairsEnv` 是函数内局部 import**：模块顶层不 import `robotdog.env`，这样单元测试可以在没有 MuJoCo 场景的情况下 import `curriculum`，也避免与 `training.py` 形成同层耦合。
- **被 `scripts/train.py` 消费**：驱动负责 `expand_curriculum` / `select_stages` / `--min-stage-frac` / `--sub-steps` / `--sub-start-cm` / `--sub-n-up`，并把回调装进 `CallbackList`；阶段结束时读 `result()` 写 `summary.json` 与 `logs/<run>_curriculum.json`。
- **与 `robotdog/training.py` 同层但单向**：`training.py` 不 import `curriculum.py`；`curriculum.py` 也不 import `training.py`。两者之间的唯一契约是「`model` 有 `get_env()` 与 `predict()`」「`venv` 有 `env_method`」。
- **`render_checkpoint.py` 依赖 `result()` 字段**：`step_height` / `n_up` / `tread` 是它从 `summary.json` 重建地形所必需的信息，所以这三个字段不能从 `result()` 里删。

## 8. 实测与经验教训

- 毕业判据饿死 PPO 的完整证据链、修法与正面证据（`easy5` 拿到约 17 次更新后 40~52% vs 先验 25~37.5%）：`experience.md` §4.1。
- `--min-stage-frac` 的打印口径与「最紧的一档 ≈ 多少次更新」：`train.md` §1、§3.3。
- 参考 run 的 `medium` 继承的是已在 flat/easy 上训练 3,909,256 步的策略（因此「22 分钟从零练出 medium」不可复现）：`train.md` §3.3。
- `course31` / `course49` 没有 `model_best.zip` 的原因：`train.md` §3.2。
- 从零训练时课程各档的真实毕业/耗尽情况（flat 100% 毕业，easy/medium/hard 全部预算耗尽 0%）：`train.md` §3.4。
- 什么杠杆有效/无效（奖励归一化、清零侧向惩罚、`action_scale`、奖励再平衡、成功奖励放大、warm-start 微调）：`train.md` §3.5。
- 域随机化不 ramp 的实测（368k 步 26%）：`CurriculumCallback` 的源码注释。

## 9. 已知限制与坑

1. **`min_timesteps` 默认是 0**，也就是**旧行为**。忘了传 `--min-stage-frac` 就会重新踩 §3.2 的坑：有开环先验时每阶段只有 0~1 次 PPO 更新。

2. **`--min-stage-frac > 0` 会把步数下限**直接改写**成预算的一个比例**：`scripts/train.py` 里的赋值是 `s.min_timesteps = max(1, int(s.max_timesteps * args.min_stage_frac))`（外面套 `if args.min_stage_frac > 0.0:`），不是「与原值取大」。所以：

   - `F=0` → `min_timesteps` 保持 0，恢复旧行为；
   - `F=0.5`（默认）→ 每档至少跑一半预算；
   - `F=1.0` → 步数下限等于整块预算，即「先跑满预算才可能毕业」——这与参考 run 的 `552,296` 步毕业不自洽（1.0 要求先跑满 400k 才可能毕业，而 552,296 > 400,000，说明参考用的是更小的值，大概率是默认的 0.5）。见 `train.md` §3.3。

3. **`expand_curriculum` 只在第一个台阶阶段用 `start_step_height` / `n_up_start`。** 想让 `medium` 也从 4 cm 起步是做不到的；它从 `easy` 的满高（约 5 cm）起步。

4. **`select_stages` 的 `"all"` 是魔法值**：任何位置出现 `"all"` 都会返回全部阶段，所以不能有一个真名叫 `all` 的阶段。

5. **`deterministic_success` 在 stderr 上完全静默**。它返回 `None` 时课程照常推进，日志里只会看到 `确定性成功率 None` 之类的格式化结果，不会被当成错误。

6. **`EpisodeStats` 的窗口是 `maxlen=window`，`n_total` 却无限增长。** `success_rate` 是窗口内的比例，`result()["episodes_total"]` 是全阶段回合数——两者口径不同，别混用。

7. **毕业或预算耗尽才做确定性测量**，中途没有。所以一个阶段如果 `max_timesteps` 设得很大而策略早就解决，也要等到预算或毕业才看到那个第二数字。

8. **`mixed_*` 的 `n_envs == 1` 取值是右上角（最高/最长踏面）**，不是中点。单环境调试时的难度与多环境配置不同。

9. **`_prune` 只删本 run 按 `name_prefix` 匹配的文件**，用别的前缀存的检查点不会被回收。

10. **`BestModelCallback._on_step` 在 `min_episodes` 之前完全不看成功率**，所以一个刚 `--resume` 的 run 在头几个回合不会回写 `best_<stage>.zip`。

## 10. 示例用法

打印默认课程并把台阶阶段展开成阶梯：

```python
from robotdog.curriculum import (expand_curriculum, default_curriculum,
                                 select_stages)

stages = select_stages(default_curriculum(), "easy,medium")     # 或 "all"
levels = expand_curriculum(stages, n_levels=4, start_step_height=0.04, n_up_start=2)
for s in levels:
    print(s.name, s.summary())
```

把它接进训练（真实驱动见 `scripts\train.py`）：

```python
from robotdog.curriculum import (BestModelCallback, CurriculumCallback,
                                 PeriodicCheckpointCallback, select_stages,
                                 default_curriculum, write_curriculum_report)
from stable_baselines3.common.callbacks import CallbackList

results = []
for i, stage in enumerate(select_stages(default_curriculum(), "all")):
    stage.min_timesteps = int(stage.max_timesteps * 0.5)   # 等价于 --min-stage-frac 0.5
    monitor = CurriculumCallback(stage, cfg=None, stage_index=i, det_episodes=10)
    cbs = CallbackList([
        monitor,
        PeriodicCheckpointCallback(f"runs/demo/checkpoints/{stage.name}",
                                   stage.min_timesteps, name_prefix=stage.name, keep=5),
        BestModelCallback(f"runs/demo/artifacts/best_{stage.name}.zip", monitor, floor=-1.0),
    ])
    model.learn(total_timesteps=stage.max_timesteps, callback=cbs, reset_num_timesteps=False)
    results.append(monitor.result())

write_curriculum_report("logs/demo_curriculum.json", results, {"run_name": "demo"})
```

单独量一个 checkpoint 的确定性成功率：

```python
from robotdog.curriculum import deterministic_success
from robotdog.tools.terrain import demo_stairs_spec

spec = demo_stairs_spec(step_height=0.078, tread=0.50, n_up=8, top_landing=1.20)
print(deterministic_success(model, spec, n_episodes=15))     # 可能是 None（测量失败）
```

命令行：

```bat
.venv\Scripts\python.exe scripts\train.py --run-name demo --stage all --sub-steps 4 ^
  --sub-start-cm 4.0 --sub-n-up 2 --min-stage-frac 0.5
```
