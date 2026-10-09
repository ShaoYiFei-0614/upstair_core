# `robotdog/training.py` — 训练基础设施 (PPO 组装 + 向量化环境 + 运行目录 + 断点续训)

## 1. 一句话定位

训练层（第 7 步）：把 `robotdog.env.Go2StairsEnv` 组装成 Stable-Baselines3 能吃的向量化环境，按固定超参建 PPO，并负责「一个 run 的目录结构、检查点、观测归一化统计、从磁盘上的 checkpoint 反推当初的 `EnvConfig`」。它不定义环境动力学，也不定义课程表。

## 2. 职责边界

**做：**

- `DummyVecEnv(n) → VecMonitor → VecNormalize` 这条链的组装与参数固化。
- `PPO` 的构造（策略网络结构、`log_std_init`、KL 早停阈值等训练超参）。
- `runs/<name>/` 的目录约定与四类产物（最终权重 / 最优权重 / `VecNormalize` 统计 / `summary.json`）。
- 从磁盘上的 `.zip` 反推它需要什么样的 `EnvConfig`——观测维度反推（`_obs_config_for_dim`）与 `summary.json` 精确恢复（`env_config_from_run`）两条路径。
- 把 `RunDir.save_model` 写出的 `<模型>_vecnormalize.pkl` 还原进现有 venv（`restore_vecnormalize`）。

**不做：**

- 不做课程推进/毕业判定（`robotdog/curriculum.py`），不训练、不调 `learn()`（`scripts/train.py`）。
- 不评估、不出报表（`robotdog/evaluate.py`）。
- 不定义地形几何（`robotdog/tools/terrain.py` 的 `StairSpec`），不实现 `set_stair`（`Go2StairsEnv`），本模块只是通过 `env_method` 转发。
- 不决定 `EnvConfig` 的默认值（那是 `robotdog/env.py` 的事）；反推路径刻意**不能**还原 `--cfg` 覆盖过的超参。
- 不做日志/控制台编码设置（`robotdog/tools/console.py`）。

## 3. 关键设计决策与理由

### 3.1 向量化链路是 `DummyVecEnv → VecMonitor → VecNormalize`，`SubprocVecEnv` 不可用

沙箱禁止 Windows 命名管道：`SubprocVecEnv` / `multiprocessing` 直接抛 `PermissionError: [WinError 5]`。所以「并行环境数 8」实际是 `DummyVecEnv(8)`——**同一进程里顺序推进 8 个环境**。在 `DummyVecEnv` 与 `VecNormalize` 之间必须夹 `VecMonitor`（源码注释：`VecMonitor sits *inside* so episode returns stay unnormalised.`），否则 episode return 会被奖励归一化污染，`curriculum/mean_return` 与 `info["episode"]["r"]` 都失去意义。

**被否决的替代方案：** 线程池。`mj_step` 不释放 GIL，多线程只会增加上下文切换开销，吞吐不会涨。真正的多进程被沙箱否掉之后没有再找替代（没有别的进程模型可用）。

### 3.2 `VecNormalize(..., training=True, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0)`

- **观测归一化默认开**：关节速度量程 ±20 rad/s、关节角 ±3 rad，量纲差一个数量级，不归一化网络第一层就被速度项主导。`clip_obs=10.0` 兜住异常值。
- **奖励归一化默认关**（`norm_reward: bool = False`）：课程会在阶段间切换地形，在 5 cm 上学到的奖励尺度到 11 cm 就漂移了。保留这个旋钮的理由是侧向惩罚主导每步奖励（`straight_line` 一项占 72.7%，而 `forward_vel` 只贡献 +0.356），挣扎中的策略回报接近 -6000，critic 拟合不动，归一化能恢复一个可用的 critic 目标。

**它到底值多少（写进 docstring 的 A/B 表，3.0 cm 台阶、150k 步、8 环境，两个 arm 各用自己的归一化器，探针在 `.tmp/norm_ab.py`）：**

| arm | 成功率 | `mean|a|` | 11 cm |
| --- | --- | --- | --- |
| `off`（默认） | 0.474 | 0.070 → 0.063 | 无成功 |
| `on` | 0.539 | 0.080 → 0.083 | 无成功 |

`explained_variance`：`off` 为 0.2163，`on` 为 -0.2363。**结论：温和的稳定器，不是解药**——两侧在 11 cm 上都没有成功，而 explained_variance 甚至更差。

**⚠ 已撤回的数字：** 本 docstring 早期草稿声称奖励归一化让残差增长三倍、把 11 cm 摔倒率从 0.88 降到 0.12（残差从 0.025 涨到 0.033）。那些数字来自一个后来被判定**会报错回合**的探针，已全部撤回，见 `experience.md` §5.1 与 §4.2。

**被否决的替代方案：** 把奖励归一化设成默认开。否决理由就是上表——增益落在噪声里，却永久改变了 `summary.json` 里回报数值的口径（`info["episode"]["r"]` 与 tensorboard 的 `rollout/ep_rew_mean` 从此不可与非归一化 run 直接比较）。

### 3.3 `_obs_config_for_dim` 存在是因为观测块维度不互质，会撞出同一个总宽度

`ObsConfig` 的可选块里 `base_lin_vel` / `base_lin_vel_body` / `stair_probe` 都是 +3 维，`gait_phase` / `drift` 都是 +2 维。于是**不同的开关组合可以给出完全相同的总维度**。枚举顺序（`_obs_config_candidates` 的 `flags` 元组）被刻意排成「新标志排在旧标志前面」：`itertools.product` 让**靠后**的标志变化最快，而函数返回的是枚举顺序里的**第一个**匹配，所以把新的 `base_lin_vel_body` 放在旧的 `base_lin_vel` 之前，就能让「反推 660 维」这类老维度先命中老配置 `base_lin_vel=True`，而不是新的 `base_lin_vel_body=True`。

这里有两个层次，要分清楚：

- **歧义来自维度不互质**：同一个总宽度可能对应多个开关组合，所以 `_obs_config_for_dim` 只能给出「枚举顺序里的第一个」，要判断歧义得用 `_obs_config_candidates`。
- **必须存在的直接动机是一个真实事故**（写在 `scripts/eval.py` 的注释里）：带 `--obs-foot-height --obs-base-height` 训练的模型是 **490 维**，而 `EnvConfig()` 默认是 **420 维**；直接用默认配置建环境再 `PPO.load` 会抛 `ValueError: Observation spaces do not match`——**注意这不是同宽碰撞，是「默认配置的宽度根本不对」**。所以 `env_config_for_model` 的首要路径永远是 run 目录里的 `summary.json`（精确记录，包括 `--cfg` 覆盖过的超参），维度反推只是没有 `summary.json` 时的降级路径。

### 3.4 `log_std_init=-0.7` 而不是默认的 0.0

默认 `0.0` 意味着归一化动作空间里 std = 1.0，即残差动作每一步都把 ±0.30 rad 的关节偏置打满——对一只站着的狗来说是纯噪声。`-0.7` 对应 std ≈ 0.5。

### 3.5 `find_run_dir` 只往上找 4 层

`max_up: int = 4` 是刻意的小值：`runs/<name>/checkpoints/<stage>_final.zip` 到 `runs/<name>/` 正好 2 层。限制层数可以避免从工作区某个深层路径一路摸到仓库根目录下的无关 `summary.json`。

### 3.6 `_ENV_CONFIG_TUPLE_FIELDS` 的必要性

`EnvConfig` 里 `kp` / `kd` / `gravity` / `friction` 是元组，但 `summary.json` 经 JSON 往返后全部变成 list。直接 `EnvConfig(**env_kw)` 会把这些字段变成可变序列，后面的行为（尤其是需要不可变哈希/比较的地方）与训练时不再一致。恢复时必须转回 tuple。

## 4. 公开 API

### `make_env_fn`

```python
def make_env_fn(
    stair: StairSpec,
    cfg: EnvConfig | None = None,
    seed: int = 0,
) -> Callable[[], Go2StairsEnv]:
```

docstring：`A zero-arg factory, as required by SB3's VecEnv constructors.`

- 语义：返回一个零参数闭包 `_init()`，每次调用构造一个新 `Go2StairsEnv(stair=stair, cfg=cfg, seed=seed)`。
- 参数：`stair` 地形；`cfg=None` 时由 `Go2StairsEnv` 内部取 `EnvConfig()`；`seed` 直接透传。**同一个 seed 会被所有子环境共享**——去重发生在调用方（`make_vec_env` 传 `seed + i`）。
- 返回：`Callable[[], Go2StairsEnv]`。
- 异常：无（构造错误推迟到调用闭包时由 `Go2StairsEnv` 抛）。

### `make_vec_env`

```python
def make_vec_env(
    stair: StairSpec,
    n_envs: int = 8,
    *,
    cfg: EnvConfig | None = None,
    seed: int = 0,
    normalize: bool = True,
    monitor: bool = True,
    norm_reward: bool = False,
) -> VecEnv:
```

- 语义：建 `n_envs` 个子环境，包成 `DummyVecEnv`；`monitor=True` 再套 `VecMonitor`；`normalize=True` 再套 `VecNormalize(training=True, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0)`。返回**最外层**包装器。
- 参数：`n_envs=8`；`normalize` / `monitor` 可关掉以拿到裸 `DummyVecEnv`；`norm_reward` 见 §3.2。
- 返回：`VecEnv`（最外层）。注意返回类型标注是 `VecEnv`，实际最外层可能是 `VecNormalize`——调用方用 `_find_vecnormalize` 而不是硬编码类型。
- 异常：`n_envs < 1` → `raise ValueError("n_envs must be >= 1")`。

### `set_terrain`

```python
def set_terrain(venv: VecEnv, stair: StairSpec) -> None:
```

- 语义：`venv.env_method("set_stair", stair)` 之后 `venv.reset()`。
- **刻意复用同一个 `VecNormalize` 包装器**：上一阶段积累的观测统计仍然有效，这正是课程迁移干净的原因。
- 开销：地形烘进编译后的 MJCF，每个子环境重编译模型（约 250 ms × n_envs）。
- 返回：`None`。异常：无显式抛出。

### `set_terrain_per_env`

```python
def set_terrain_per_env(venv: VecEnv, specs: Sequence[StairSpec]) -> None:
```

- 语义：给每个子环境单独设地形，逐个 `venv.env_method("set_stair", spec, indices=[i])`，最后 `venv.reset()`。`--mixed-heights` / `--mixed-treads` 用它让一个阶段内各子环境处于不同台阶高度/踏面，从而让**整条能力带**都出现在每个 PPO 梯度批次里，避免灾难性遗忘已经爬过的台阶。
- 耗时与 `set_terrain` 相同（每个子环境都要重编译）。
- 异常：`len(specs) != venv.num_envs` → `raise ValueError(f"需要 {n} 个地形, 收到 {len(specs)} 个")`，其中 `n = int(venv.num_envs)`。

### `DEFAULT_NET_ARCH`

```python
DEFAULT_NET_ARCH = (256, 256)
```

模块级常量，`build_ppo` 的 `net_arch` 默认值。

### `build_ppo`

```python
def build_ppo(
    venv: VecEnv,
    *,
    seed: int = 0,
    n_steps: int = 256,
    batch_size: int = 512,
    n_epochs: int = 5,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.002,
    vf_coef: float = 0.5,
    max_grad_norm: float = 1.0,
    target_kl: float | None = 0.025,
    log_std_init: float = -0.7,
    net_arch: Sequence[int] = DEFAULT_NET_ARCH,
    tensorboard_log: str | Path | None = None,
    device: str = "cpu",
    verbose: int = 1,
) -> PPO:
```

docstring：`PPO with settings tuned for 420-dim locomotion observations on CPU.`

- 语义：构造 `PPO("MlpPolicy", venv, ...)`，`policy_kwargs` 固定为 `dict(net_arch=dict(pi=list(net_arch), vf=list(net_arch)), activation_fn=nn.Tanh, log_std_init=log_std_init)`。`tensorboard_log=str(tensorboard_log) if tensorboard_log else None`。
- 参数：`net_arch` 同时用于 actor 与 critic；`target_kl=0.025` 是 KL 早停阈值；`log_std_init=-0.7` 见 §3.4。
- 返回：未训练的 `PPO`。
- 异常：无显式抛出（`venv` 观测空间与策略不匹配时会由 SB3 抛）。
- 注意：`batch_size` 由调用方保证不超过一次 rollout 的样本数（`scripts/train.py` 传 `batch_size=min(args.batch_size, args.n_steps * args.n_envs)`）。

### `load_ppo`

```python
def load_ppo(path: str | Path, venv: VecEnv, *, device: str = "cpu") -> PPO:
```

- 语义：`PPO.load(str(path), env=venv, device=device)`。**只还原网络权重**，不还原 `VecNormalize` 的 running mean/var（见 `restore_vecnormalize`）。
- 异常：观测空间不匹配时 SB3 抛 `ValueError: Observation spaces do not match`。

### `_ENV_CONFIG_TUPLE_FIELDS`

```python
_ENV_CONFIG_TUPLE_FIELDS = frozenset({"kp", "kd", "gravity", "friction"})
```

### `find_run_dir`

```python
def find_run_dir(start: str | Path, *, max_up: int = 4) -> Path | None:
```

- 语义：从 `start` 往上找最近的含 `summary.json` 的目录。`start` 是文件时先取 `p.parent`；循环 `max_up + 1` 次；到根（`p.parent == p`）就停。
- 返回：`Path | None`（找不到返回 `None`）。

### `_model_observation_dim`

```python
def _model_observation_dim(model_path: str | Path) -> int | None:
```

- 语义：**不加载模型**，直接从 SB3 的 `.zip` 里读出保存的观测维度。SB3 2.9 把 `data` 存成 JSON，每个对象是 `{":type": ..., ":serialized": "<base64 pickle>"}`，所以走 JSON → base64 → pickle 三步；SB3 < 2.0 直接 pickle `data`（保留为回退路径）。内层 `_shape_of(obj)` 取 `getattr(obj, "shape", None)`，取不到再看 `obj.get("shape")`，最后 `int(shape[0])`。
- 返回：`int | None`。**任何异常都返回 `None`**（包括 `TypeError` / `IndexError` / `ValueError` 与整段的 `except Exception`）。

### `_obs_config_candidates`

```python
def _obs_config_candidates(total: int, history_len: int = 10) -> list[ObsConfig]:
```

- 语义：枚举所有能拼出 `total` 维的可选观测块组合。`flags` 顺序（**逐字符即优先级**）：`("stair_probe_full", "touch", "stair_probe", "drift", "base_lin_vel_body", "base_lin_vel", "base_height", "foot_height", "gait_phase")`；对 `product((False, True), repeat=len(flags))` 的每个组合构造 `ObsConfig(history_len=history_len, **kw)`，`cand.size == total` 的收进结果。
- 约束：`stair_probe_full` 只在 `stair_probe` 也打开时才考虑（`if kw["stair_probe_full"] and not kw["stair_probe"]: continue`），它多给 2 维。
- 返回：`list[ObsConfig]`，顺序即优先级顺序。

### `_obs_config_for_dim`

```python
def _obs_config_for_dim(total: int, history_len: int = 10) -> ObsConfig | None:
```

- 语义：反推是哪几个可选观测块拼出了 `total` 维，返回 `_obs_config_candidates` 的第一个。docstring 明确：各块维度不互质（`base_lin_vel`/`base_lin_vel_body`/`stair_probe` 都是 3，`gait_phase`/`drift` 都是 2），所以存在维度相同的不同组合；枚举顺序被刻意排成「新标志最后开」，使老维度总是反推回老配置。判断歧义用 `_obs_config_candidates`，**首选路径始终是 run 目录里的 `summary.json`**（精确记录）。
- 返回：`ObsConfig | None`。

### `env_config_for_model`

```python
def env_config_for_model(model_path: str | Path) -> EnvConfig:
```

- 语义：给一个 `.zip` 模型恢复出它能加载的 `EnvConfig`。先 `find_run_dir(model_path)` → 命中则 `env_config_from_run(run_dir)`；否则从 `.zip` 里保存的观测维度反推。
- 反推失败：读不到维度时打印 `f"[配置] {Path(model_path).name} 里读不到观测维度, 退回 EnvConfig() 默认值"` 并返回 `EnvConfig()`；维度无法用可选块还原时 `raise SystemExit(f"模型观测维度 {dim} 无法用 EnvConfig 的可选观测块还原（期望 42/44/47/49 x history_len 的整数倍）。请用 --cfg 显式指定, 或提供 run 目录的 summary.json。")`。
- 歧义：多于一个组合时打印 `[配置] ⚠ 观测维度 {dim} 对应 {n} 种可选块组合 ({names}), 已选第一个; 想要精确配置请提供 run 目录的 summary.json。`，`_flag_names = ("base_lin_vel", "base_height", "foot_height", "gait_phase", "touch", "stair_probe", "drift")`。
- 副作用：`if obs.gait_phase: cfg.gait_prior = True`——观测里有步态时钟说明几乎必然开了先验；不开的话机器人只会站在原地，成功率恒 0，会被误读成「策略什么都没学」。
- 打印：`[配置] 模型旁没有 summary.json, 已从观测维度 {dim} 反推: frame={obs.frame_size} (base_height=..., foot_height=..., gait_phase=...), history={obs.history_len}{prior_note}。**注意**: --cfg 覆盖过的超参(如 episode_length_s) 无法反推, 已用默认值。`
- 异常：`SystemExit`（见上）。

### `env_config_from_run`

```python
def env_config_from_run(run_dir: str | Path) -> EnvConfig:
```

- 语义：从 `runs/<name>/summary.json` 恢复训练时的 `EnvConfig`。`summary.json` 存了 `env_config` / `obs_config` / `reward_config` 三份 `asdict`。
- 关键细节：`_ENV_CONFIG_TUPLE_FIELDS` 里的 list 转回 tuple；`reward_config` 过滤成 `RewardConfig` 的字段名后重建；`obs_config` 同理重建 `ObsConfig`；`gait` 是嵌套 dataclass（asdict 摊平成 dict），必须重建 `GaitParams`，其中 `calf_shape` 若是 int 要从 `robotdog.gait.CALF_SHAPES[int(...)]` 取回——否则 `cfg.gait.radians()` 会炸；`touch` 是 dict 时重建 `TouchConfig`。
- 找不到文件：打印 `[配置] 没有 {summary_path.name}, 退回 EnvConfig() 默认值 (若模型观测不是 420 维, 加载会失败)` 并返回 `EnvConfig()`。
- 未知字段：`[配置] summary.json 里有当前 EnvConfig 不认识的字段, 已忽略: {unknown}`。
- 成功：打印 `[配置] 从 {summary_path.name} 恢复 EnvConfig: 观测 {cfg.obs.size} 维 (frame={cfg.obs.frame_size}, history={cfg.obs.history_len}), episode={cfg.episode_length_s}s, action_scale={cfg.action_scale}`。

### `class RunDir`

```python
class RunDir:
    def __init__(self, name: str, root: str | Path | None = None) -> None:
```

- 语义：`runs/<name>/` 的目录对象。`self.root = Path(root) if root else (RUNS_DIR / name)`；`checkpoints = root/"checkpoints"`、`tensorboard = root/"tensorboard"`、`artifacts = root/"artifacts"`，四个目录（含 root）全部 `mkdir(parents=True, exist_ok=True)`。
- 属性（均为 `Path`）：`final_model` → `root/"model_final"`；`best_model` → `root/"model_best"`；`vecnormalize` → `root/"vecnormalize.pkl"`；`summary` → `root/"summary.json"`。
- `def save_model(self, model: PPO, path: Path, venv: VecEnv | None = None) -> None:` → `model.save(str(path))`；再解析归一化器：`target = venv if venv is not None else getattr(model, "env", None)`，`vec = getattr(model, "get_vec_normalize_env", lambda: None)() or target`，`vec_norm = _find_vecnormalize(vec)`，非 `None` 时 `vec_norm.save(str(path) + "_vecnormalize.pkl")`。注意拼接方式是字符串加后缀，不是 `with_suffix`。
- `def write_summary(self, payload: dict[str, Any]) -> Path:` → `self.summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")`，返回 `self.summary`。

### `_find_vecnormalize`

```python
def _find_vecnormalize(venv: Any) -> VecNormalize | None:
```

- 语义：沿 `venv.venv` 往上走，`seen < 10` 步内返回第一个 `isinstance(venv, VecNormalize)`；走不到返回 `None`。层数上限是防包装链成环。

### `restore_vecnormalize`

```python
def restore_vecnormalize(venv: VecEnv, path: str | Path) -> bool:
```

- 语义：把 `RunDir.save_model` 写出的观测统计还原进 **现有** venv。`PPO.load` 只还原网络权重，`VecNormalize` 的 running mean/var 单独存成 `<模型>_vecnormalize.pkl`；不还原它，策略就会拿到按 mean=0/var=1 归一化的输入，而训练时看到的是真统计——等于换了一套输入坐标系。
- `VecNormalize.load` 要求传入**未包装**的 `VecEnv`（它会新建包装器），这里不想再套一层，所以加载完只把统计量拷回现有包装器：`obs_rms`、`ret_rms`、`clip_obs`、`clip_reward`、`epsilon`。
- 返回：`bool`——文件不存在或找不到 `VecNormalize` 包装器都返回 `False`（不抛异常）。

### `format_duration`

```python
def format_duration(seconds: float) -> str:
```

- `< 60` → `f"{seconds:.1f}s"`；`< 3600` → `f"{seconds / 60:.1f}min"`；否则 `f"{seconds / 3600:.2f}h"`。

### `class Stopwatch`

```python
class Stopwatch:
    def __init__(self) -> None:
```

- `self.t0 = time.time()`；`@property elapsed` → `time.time() - self.t0`。

## 5. 内部结构与数据流

建环境：

```
scripts/train.py
  → make_vec_env(stages[0].spec, n_envs, cfg=cfg, seed=..., normalize=True, monitor=True, norm_reward=...)
      → make_env_fn(stair, cfg, seed + i)                        # n_envs 个零参闭包
      → DummyVecEnv(fns)
      → VecMonitor(venv)                                         # 在 VecNormalize 内侧
      → VecNormalize(venv, training=True, norm_obs=True, norm_reward=..., clip_obs=10.0)
  → build_ppo(venv, n_steps=..., batch_size=min(batch_size, n_steps * n_envs), ...)
  → model.learn(total_timesteps=stage.max_timesteps, callback=CallbackList([...]))
```

切地形：

```
CurriculumCallback._on_step() 返回 False
  → scripts/train.py 下一阶段
      → set_terrain(venv, stage.spec)            # 整段同高
          → venv.env_method("set_stair", stair) → Go2StairsEnv.set_stair → 重编译 MJCF
          → venv.reset()
      → set_terrain_per_env(venv, specs)         # --mixed-heights / --mixed-treads
          → venv.env_method("set_stair", spec, indices=[i])   # 逐子环境
          → venv.reset()
```

保存与恢复：

```
RunDir.save_model(model, path, venv)
  → model.save(str(path))
  → get_vec_normalize_env() or model.env → _find_vecnormalize → vec.save(str(path) + "_vecnormalize.pkl")

scripts/train.py --resume <zip>
  → env_config_for_model(zip) → find_run_dir → env_config_from_run  (或 _model_observation_dim → _obs_config_for_dim)
  → make_vec_env(flat_spec(), n_envs=1, cfg=cfg, seed=0)      # 只为建模型
  → load_ppo(zip, venv, device=...)
  → restore_vecnormalize(venv, <zip.stem>_vecnormalize.pkl)
  → model.tensorboard_log = str(run.tensorboard)              # 强制改写，PPO.load 会带回旧 run 的路径
```

## 6. 关键常量与默认值

| 名称 | 值 | 说明 |
| --- | --- | --- |
| `DEFAULT_NET_ARCH` | `(256, 256)` | actor 与 critic 共用的隐层 |
| `_ENV_CONFIG_TUPLE_FIELDS` | `frozenset({"kp", "kd", "gravity", "friction"})` | JSON 往返后需转回 tuple 的字段 |
| `make_vec_env.n_envs` | `8` | 子环境数（`DummyVecEnv`，单进程） |
| `make_vec_env.normalize` / `monitor` | `True` / `True` | 包装开关 |
| `make_vec_env.norm_reward` | `False` | 见 §3.2 |
| `VecNormalize` 内参 | `training=True, norm_obs=True, clip_obs=10.0` | `norm_reward` 来自参数 |
| `make_env_fn.seed` / `make_vec_env.seed` | `0` | 子环境用 `seed + i` |
| `build_ppo.n_steps` | `256` | rollout 长度 |
| `build_ppo.batch_size` | `512` | 调用方仍需 `min(512, n_steps*n_envs)` |
| `build_ppo.n_epochs` | `5` | |
| `build_ppo.learning_rate` | `3e-4` | |
| `build_ppo.gamma` / `gae_lambda` | `0.99` / `0.95` | |
| `build_ppo.clip_range` | `0.2` | |
| `build_ppo.ent_coef` / `vf_coef` | `0.002` / `0.5` | |
| `build_ppo.max_grad_norm` | `1.0` | |
| `build_ppo.target_kl` | `0.025` | `None` 可关掉 KL 早停 |
| `build_ppo.log_std_init` | `-0.7` | 归一化动作单位下 std ≈ 0.5 |
| `build_ppo.device` | `"cpu"` | |
| `build_ppo.verbose` | `1` | |
| `find_run_dir.max_up` | `4` | 向上搜索层数 |
| `_obs_config_candidates.history_len` | `10` | 默认 history 长度 |
| `_flag_names`（打印用） | `("base_lin_vel", "base_height", "foot_height", "gait_phase", "touch", "stair_probe", "drift")` | 歧义提示 |
| `_find_vecnormalize` 步数上限 | `10` | 包装链遍历上限 |
| `format_duration` 阈值 | `60` / `3600` | s / min / h |
| `ObsConfig` 默认维度 | `420`（42 × 10） | `EnvConfig()` 默认 |
| 课程配方维度 | `660`（66 × 10） | `experience.md` 里的 OOM traceback 是 `shape (512, 660)` |
| 单进程吞吐 | 约 250–560 向量步/s | 参考 run 的 tensorboard `time/fps = 417`，`summary.json` 的 `steps_per_second = 412.3` |
| `RunDir` 目录名 | `checkpoints` / `tensorboard` / `artifacts` | 属性名 `final_model` / `best_model` / `vecnormalize` / `summary` |

## 7. 与其它模块的关系

```
robotdog/tools/paths.py  (RUNS_DIR)
robotdog/tools/terrain.py (StairSpec)
        ↓
robotdog/env.py (EnvConfig, Go2StairsEnv)
        ↓
robotdog/training.py          ← 本模块
        ↑
robotdog/curriculum.py  (调 deterministic_success 时用 model.get_env()，不 import 本模块)
scripts/train.py        (唯一的驱动方)
```

- **向下依赖 `robotdog.env`**：本模块必须实例化 `Go2StairsEnv` 并读写 `EnvConfig`，这是训练层的定义。方向不能反：`env.py` 不知道 PPO、不知道 `VecNormalize`，否则工具层/环境层就被绑死在某个算法上，`robotdog/tools/` 也不能再声明「不 import robotdog.env」。
- **依赖 `robotdog/tools/paths.py` 的 `RUNS_DIR`**：run 目录布局属于工具层的路径约定，训练层只是使用者。这样 `scripts/` 与文档都以同一个常量为准。
- **被 `scripts/train.py` 与 `scripts/eval.py` 消费**：`train.py` 用全套；`eval.py` 只用 `make_vec_env` / `load_ppo` / `env_config_from_run`（不直接用 `RunDir`，它自己解析 `--run` / `--model` / `--vecnormalize`）。
- **`robotdog/curriculum.py` 不 import 本模块**：它只要求传入的对象有 `get_env()` 与 `predict()`，靠鸭子类型，避免训练层的两个文件互相成环。
- **`robotdog/render.py` 不依赖本模块**：它接的是单个 `Go2StairsEnv` 与任意 `Predictor`，所以渲染一张 GIF 不需要 `VecNormalize`——代价是调用方必须自己把观测归一化接上（见 `render.md`）。

## 8. 实测与经验教训

- 奖励归一化的增益与撤回记录：`experience.md` §4.2，以及 §5.1（探针报错回合的根因）。
- 消融表（A 组）：`train.md` §3.5 A。
- 观测归一化开关与权重必须成对：`experience.md` §5.3。
- 打开观测归一化却没喂对统计量的表现、以及 `pickle.loads(vecnormalize.pkl)` 返回的不是 dict 这件事：`experience.md` §5.4。
- `SubprocVecEnv` / 命名管道的沙箱限制：`experience.md` §5.8，`train.md` §1。
- 吞吐与步数口径：参考 run 的 tensorboard `time/fps ≈ 417`、`summary.json` 的 `steps_per_second = 412.3`（本仓 `runs/course22_ref/summary.json` 可核对）；`train.md` §1 说明「环境步」是跨 resume 累计的，引用步数时必须先确认口径（`train.md` §3.3）。
- `--resume` 静默忽略 `--net-arch` / `--learning-rate` / `--target-kl` / `--n-epochs`：`experience.md` §5.8，`train.md` §3.2。

## 9. 已知限制与坑

1. **并行 run 数必须 ≤ 3。** 超过之后一批 job 会瞬间 `exit code 1` 且没有 traceback，前台重跑同一命令正常。崩出来的 traceback 是：

   ```
   numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.29 MiB for an
   array with shape (512, 660) and data type float32
   ```

   位置在 `stable_baselines3\common\buffers.py` 的 `self.observations[batch_inds]`。失败的不是学习而是内存。见 `experience.md` §5.5。

2. **`summary.json` 的 `reproduce` 字段分两代。** 参考 run（course22）里的 `reproduce` 是**旧世代的有损回声**，只记了 5 个手挑参数，会静默丢掉 `--stage-top-cm` / `--mixed-heights` / `--resume` / `--obs-*` / `--gait-prior` / `--cfg`。course31 之后才改成原样回显真实 argv。引用旧 run 的 `reproduce` 前必须核对字段格式。见 `experience.md` §5.8。

3. **`_model_observation_dim` 全部异常都吞掉返回 `None`。** 结果是「读不到维度」与「zip 格式不认识」无法区分，调用方只会看到退回默认 `EnvConfig()` 的提示。

4. **`_obs_config_for_dim` 有歧义且只能返回第一个。** 判别歧义必须调 `_obs_config_candidates` 并检查长度；只有 `summary.json` 是精确的。

5. **反推路径恢复不了 `--cfg` 覆盖过的超参**（`episode_length_s`、`terminate_*`、`reward.*` 等），会静默用默认值。`env_config_for_model` 的提示里写了这一点，但它是 print，不是异常。

6. **`env_config_from_run` 遇到不认识的字段只打印并忽略。** 如果 `EnvConfig` 删掉了某个字段，旧 `summary.json` 能顺利加载但语义已经变了。

7. **`restore_vecnormalize` 只拷 5 个属性**（`obs_rms` / `ret_rms` / `clip_obs` / `clip_reward` / `epsilon`），不改 `training` / `norm_reward`。续训时 `training=True` 由 `make_vec_env` 保证；用 `eval.py` 的 `load_policy` 那条路径才显式设 `training=False`。

8. **`RunDir.save_model` 的归一化文件名是字符串拼接** `str(path) + "_vecnormalize.pkl"`，所以 `Path("model.zip")` 会得到 `model.zip_vecnormalize.pkl`。读取方（`scripts/train.py` / `scripts/eval.py`）用的是 `model_path.stem + "_vecnormalize.pkl"`，两套命名不要混。

## 10. 示例用法

从零训练一个课程 run（等价于 `scripts\train.py --run-name demo --stage all` 的核心部分）：

```python
from robotdog.curriculum import CurriculumCallback, default_curriculum
from robotdog.training import RunDir, build_ppo, make_vec_env
from stable_baselines3.common.callbacks import CallbackList

stages = default_curriculum()
run = RunDir("demo")
venv = make_vec_env(stages[0].spec, n_envs=8, seed=0, norm_reward=False)
model = build_ppo(venv, tensorboard_log=run.tensorboard)

for stage in stages:
    venv.env_method("set_stair", stage.spec)   # 等价于 training.set_terrain 的一半
    venv.reset()
    cb = CurriculumCallback(stage, cfg=model.get_env().envs[0].cfg)
    model.learn(total_timesteps=stage.max_timesteps,
                callback=CallbackList([cb]),
                tb_log_name="ppo", reset_num_timesteps=False)
    print(stage.name, cb.stop_reason)

run.save_model(model, run.final_model, venv)          # 同时写 model_final_vecnormalize.pkl
run.write_summary({"run_name": "demo", "stages": []})
```

从磁盘上的 checkpoint 恢复配置并续训：

```python
from pathlib import Path
from robotdog.training import (env_config_for_model, load_ppo, make_vec_env,
                               restore_vecnormalize, set_terrain)

zip_path = Path(r"runs\course22\checkpoints\medium_final.zip")
cfg = env_config_for_model(zip_path)          # 优先读同 run 的 summary.json
venv = make_vec_env(stages[0].spec, n_envs=8, cfg=cfg, seed=0)
model = load_ppo(zip_path, venv, device="cpu")
if not restore_vecnormalize(venv, zip_path.with_name(zip_path.stem + "_vecnormalize.pkl")):
    print("[恢复] ⚠ 观测归一化统计没加载上")
```

命令行（真实入口）：

```bat
.venv\Scripts\python.exe scripts\train.py --run-name demo --stage all --n-envs 8 ^
  --mixed-heights 8.0,11.0 --min-stage-frac 0.5 --norm-reward
```
