# `robotdog/training.py` 设计文档

## 1. 一句话定位

**训练层**的「发动机与工作台」：把 `env` + `VecNormalize` + PPO 组装起来，
提供 `runs/<名字>/` 目录管理、权重/归一化统计的成对保存与读回、以及加载策略的
一站式入口 `load_policy()`。

## 2. 职责边界

**做**：

* 建向量化环境（`make_env_fn` / `make_vec_env`）与换地形（`set_terrain`）。
* 建 PPO（`build_ppo`）与加载（`load_ppo` / `load_policy`）。
* `VecNormalize` 统计的保存、读回与沿包装链查找（`save_vecnormalize` /
  `restore_vecnormalize` / `_find_vecnormalize`）。
* 运行目录（`RunDir`）：检查点、tensorboard、`model_final.zip`、`summary.json`。
* 小工具：`format_duration`、`Stopwatch`、`mean_abs_action`。

**不做**：

* **不评估、不渲染**。成功率与 GIF 都在 `scripts/play.py`。
* **不决定练哪几级**。课程表在 `robotdog/curriculum.py`。
* **不解析 `summary.json` 里的配置**。那是 `robotdog/policy.py`
  （`env_config_for_model`），因为它要能在**没有 SB3** 的情况下跑。
* **不做多进程**。见 §3.1。

## 3. 关键设计决策与理由

### 3.1 `DummyVecEnv` 是唯一可行的并行方式

沙箱禁止 Windows 命名管道，`SubprocVecEnv` / `multiprocessing` 会
`PermissionError: [WinError 5]`。所以「并行 8 个环境」只能是 `DummyVecEnv(8)` ——
同一进程里顺序推进 8 个环境。`mj_step` 不释放 GIL，换线程池也没有收益。
实测 250~400 向量步/秒。

**副作用（必须记住）**：一台机器上并发 3 个以上的训练进程会把自己的内存耗光，
典型报错是 `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.29 MiB
for an array with shape (512, 660)`，位置在
`stable_baselines3/common/buffers.py:514` 的 `self.observations[batch_inds]`。

### 3.2 `VecMonitor` 放在 `VecNormalize` **里面**

```
DummyVecEnv → VecMonitor → VecNormalize
```

顺序反了的话 `info["episode"]["r"]` 会是**归一化后**的回报，报表数字跟训练日志、
跟历史 run 全都对不上。放里面，`VecMonitor` 拿到的是原始回报。

### 3.3 观测归一化必须开，奖励归一化默认也开

`norm_obs=True` 是硬性的：关节速度（±20 rad/s）与关节角（±3 rad）量纲差三个数量级，
不归一化时 critic 完全拟合不动。

`norm_reward` 默认 **True**，但它是「温和的稳定器」而不是解药。历史背景：早期版本的
奖励里侧向惩罚项主导每一步（实测 `straight_line` 一项占 72.7%，而唯一的前进奖励
`forward_vel` 只贡献 +0.356），挣扎中的策略每回合回报接近 -6000，critic 的
`explained_variance` 长期停在 0.01 附近，PPO 于是把策略压回「几乎不动」的极小残差。
打开奖励归一化后残差才开始增长，但增益有限（honest A/B：3 cm 成功率 0.474 → 0.539，
`mean|a|` 0.070 → 0.080，两侧在 11 cm 上都没成功）。

**唯一例外**：warm-start 一个**未归一化**训出来的权重时应当关掉，否则统计量对不上。

### 3.4 `log_std_init = -2.0`

归一化动作单位下 std ≈ 0.135。SB3 默认的 `0.0` 意味着 std = 1.0，即残差每一步都把
关节目标顶满 ±`action_scale`，对站立的四足来说就是纯噪声。但**也不能设得过小**：
残差恒为 0 时策略退化成「纯开环步态」，永远学不会爬楼梯（旧课程里出现过
`mean|a| = 0.011`、`|a| < 0.05` 占 99.9% 的退化）。

### 3.5 `set_terrain()` 刻意**复用** `VecNormalize` 包装器

`venv.env_method("set_stair", stair)` 后只 `venv.reset()`，不重建包装链。
这样上一阶段攒下的观测统计继续有效，课程才能干净地迁移；重建会让策略在换级瞬间
拿到一套完全不同的输入坐标系。

代价：每个子环境都要重编译 MJCF（约 250 ms × `n_envs`），所以换地形只能在
**阶段边界**做，不能插在 rollout 中间。

### 3.6 `save_model` 与 `save_vecnormalize` 的命名不一致（历史遗留）

* `RunDir.save_model(model, base, venv)` → `<base>.zip` + `<base>_vecnormalize.pkl`
  （用**字符串拼接** `str(base) + "_vecnormalize.pkl"`，所以 `base="model_final"` 得到
  `model_final_vecnormalize.pkl`）。
* `save_vecnormalize(model, base)` 同规则。
* **读取方**（`load_policy` / `play.py`）用的是 `model_path.stem + "_vecnormalize.pkl"`。

两者对 `model_final` 这类无后缀名一致；但如果你把权重命名成 `model.zip`，
拼接会得到 `model.zip_vecnormalize.pkl` 而读取方找的是 `model_vecnormalize.pkl`。
**给 base 传不带后缀的名字。**

### 3.7 `load_policy()` 为什么要自己读裸 pickle

它返回 `(PPO, ObsNorm | None)`。`ObsNorm` 来自同目录同名的 `*_vecnormalize.pkl`。
**不还原归一化统计，确定性评估会从 100% 掉到 0%** —— 策略会拿到按
`mean=0 / var=1` 归一化的输入，而训练时看到的是真统计，等于换了一套输入坐标系。

为什么不直接用 `VecNormalize.load()`：那会新建一个包装器（这里只需要统计量），
而且 `VecNormalize` 对象本身**不能随便访问属性**——多读一个不存在的名字就
`RecursionError: maximum recursion depth exceeded`。详见 `docs/robotdog/policy.md` §3.1。

## 4. 公开 API

```python
DEFAULT_NET_ARCH: tuple[int, ...] = (512, 256)

def make_env_fn(stair: StairSpec | None, cfg: EnvConfig | None = None,
                seed: int = 0) -> Callable[[], Go2StairsEnv]: ...
def make_vec_env(stair: StairSpec | None, n_envs: int = 8, *,
                 cfg: EnvConfig | None = None, seed: int = 0,
                 normalize: bool = True, monitor: bool = True,
                 norm_reward: bool = True) -> VecEnv: ...
def set_terrain(venv: VecEnv, stair: StairSpec) -> None: ...

def build_ppo(venv: VecEnv, *, seed: int = 0, n_steps: int = 256,
              batch_size: int = 512, n_epochs: int = 5,
              learning_rate: float = 1.5e-4, gamma: float = 0.99,
              gae_lambda: float = 0.95, clip_range: float = 0.2,
              ent_coef: float = 0.002, vf_coef: float = 0.5,
              max_grad_norm: float = 1.0, target_kl: float | None = 0.03,
              log_std_init: float = -2.0,
              net_arch: tuple[int, ...] = DEFAULT_NET_ARCH,
              tensorboard_log: str | Path | None = None,
              device: str = "cpu", verbose: int = 1) -> PPO: ...
def load_ppo(path: str | Path, venv: VecEnv, *, device: str = "cpu") -> PPO: ...
def load_policy(model_path: str | Path, *, cfg: EnvConfig | None = None,
                stair: StairSpec | None = None, device: str = "cpu",
                norm_reward: bool = False) -> tuple[PPO, ObsNorm | None]: ...

def save_vecnormalize(model: PPO, base: str | Path) -> Path | None: ...
def restore_vecnormalize(venv: VecEnv, path: str | Path) -> VecNormalize | None: ...

class RunDir:
    def __init__(self, name: str, root: str | Path | None = None) -> None: ...
    name: str
    root: Path                 # 默认 RUNS_DIR / name
    checkpoints: Path          # root/checkpoints
    tensorboard: Path          # root/tensorboard
    artifacts: Path            # root/artifacts
    @property
    def final_model(self) -> Path: ...        # root/model_final
    @property
    def vecnormalize(self) -> Path: ...       # root/model_final_vecnormalize.pkl
    @property
    def summary(self) -> Path: ...            # root/summary.json
    def save_model(self, model: PPO, base: str | Path,
                   venv: VecEnv | None = None) -> tuple[Path, Path | None]: ...
    def write_summary(self, payload: dict[str, Any]) -> Path: ...

def format_duration(seconds: float) -> str: ...      # 12.3s / 4.5min / 1.20h
class Stopwatch: ...                                  # .t0，.elapsed
def mean_abs_action(actions: np.ndarray) -> float: ...
```

`__all__` = `["DEFAULT_NET_ARCH", "RunDir", "Stopwatch", "build_ppo", "format_duration",
"load_policy", "load_ppo", "make_env_fn", "make_vec_env", "mean_abs_action",
"restore_vecnormalize", "save_vecnormalize", "set_terrain"]`

语义要点：

* `make_vec_env` 的 `n_envs < 1` 抛 `ValueError("n_envs must be >= 1")`。
  第 i 个子环境的 seed 是 `seed + i`（不是同一个 seed —— 那样 8 个环境会完全同步）。
* `load_ppo` 的 `path` 带不带 `.zip` 都可以：文件不存在且后缀不是 `.zip` 时会试
  `path.with_suffix(".zip")`。
* `load_policy` **自己建一个 1 环境的 `DummyVecEnv`**（用 `cfg or
  env_config_for_model(model_path)`），因为 PPO 必须挂在一个环境上；随后
  `model.set_parameters()` 而不是 `PPO.load()` —— 后者会用新模型覆盖策略，
  这里只需要把权重灌进去。
* `restore_vecnormalize` 只拷 5 个字段：`obs_rms`、`ret_rms`、`clip_obs`、
  `clip_reward`、`epsilon`。**不动** `training` / `norm_reward` —— 续训时
  `training=True` 由 `make_vec_env` 保证。
* `_find_vecnormalize(venv)` 沿 `.venv` 链下钻，**最多 10 层**（防环）。
* `RunDir.__init__` 会 `mkdir(parents=True, exist_ok=True)` 四个目录
  （`root` / `checkpoints` / `tensorboard` / `artifacts`）。
* `RunDir.write_summary` 用 `json.dumps(..., indent=2, ensure_ascii=False)`，所以
  中文标签在文件里是可读的。

## 5. 内部结构与数据流

```
scripts/train.py（StageTrainer 实现）
    make_env(stage)
        → make_vec_env(stage.spec, n_envs=8, cfg=…, norm_reward=True)
              DummyVecEnv(8) → VecMonitor → VecNormalize
    train_stage(stage, venv, warm_start=…)
        → build_ppo(venv, …, tensorboard_log=RunDir.tensorboard)
        → [warm_start 存在时] model.set_parameters(warm_start)
        → model.learn(total_timesteps=stage.timesteps, callback=[…])
              ProgressCallback  → EpisodeStats.add_infos(infos).format_line()
              CheckpointCallback → RunDir.save_model(model, checkpoints/<阶段>/<阶段>_<步数>_steps, venv)
        → RunDir.save_model(model, checkpoints/<阶段>_final, venv)

scripts/play.py
    load_policy(model_path, cfg=env_config_for_model(model_path))
        → make_vec_env(stair, n_envs=1, cfg=cfg)
        → build_ppo(venv) → model.set_parameters(model_path)
        → ObsNorm.from_path(<stem>_vecnormalize.pkl)
        → (PPO, ObsNorm | None)
```

`RunDir` 的目录布局：

```
runs/<名字>/
├── model_final.zip                  # 末级权重
├── model_final_vecnormalize.pkl     # 与它成对（必须一起搬）
├── summary.json                     # 配置 + 各级结果
├── checkpoints/
│   ├── <阶段>_final.zip             # 每级收尾
│   ├── <阶段>_vecnormalize.pkl
│   └── <阶段>/<阶段>_<步数>_steps.zip   # 周期检查点
├── tensorboard/
└── artifacts/
```

## 6. 关键常量与默认值

| 名称 | 值 | 说明 |
| --- | --- | --- |
| `DEFAULT_NET_ARCH` | `(512, 256)` | SB3 默认的 64×64 太窄；`pi` 与 `vf` 共用这个宽度 |
| 激活函数 | `nn.Tanh` | `policy_kwargs` 里写死 |
| `make_vec_env.n_envs` | `8` | 单进程 8 环境，实测 250~400 步/秒 |
| `VecNormalize.clip_obs` | `10.0` | 硬编码在 `make_vec_env` 里 |
| `n_steps` | `256` | × `n_envs 8` = **每次更新 2048 步** |
| `batch_size` | `512` | 2048 / 512 = 每轮 4 个 minibatch |
| `n_epochs` | `5` | |
| `learning_rate` | `1.5e-4` | 课程里更高台阶会降到 6e-5 |
| `gamma` / `gae_lambda` | `0.99` / `0.95` | |
| `clip_range` | `0.2` | |
| `ent_coef` / `vf_coef` / `max_grad_norm` | `0.002` / `0.5` / `1.0` | |
| `target_kl` | `0.03` | 触发时会 `Early stopping at step k/N due to reaching max kl` —— 正常现象 |
| `log_std_init` | `-2.0` | std ≈ 0.135 |
| `_find_vecnormalize` 下钻上限 | `10` 层 | 防环 |
| `restore_vecnormalize` 拷贝字段 | `obs_rms` / `ret_rms` / `clip_obs` / `clip_reward` / `epsilon` | 不改 `training` / `norm_reward` |
| `mean_abs_action` 健康区间 | `0.3 ~ 0.45` | 参考策略约 0.42；< 0.1 说明退化成纯先验 |

## 7. 与其它模块的关系

```
robotdog/env.py                    (Go2StairsEnv / EnvConfig)
        ▲
robotdog/training.py ──────────────► stable_baselines3 (PPO / DummyVecEnv / VecMonitor / VecNormalize)
        │        │
        │        └──► robotdog/policy.py  (ObsNorm / env_config_for_model)   ← 只读配置，不回头 import
        │
        └──► robotdog/tools/paths.py (RUNS_DIR)  /  robotdog/tools/terrain.py (StairSpec)
        ▲
scripts/train.py  /  scripts/play.py
```

* **向下**依赖 `env`（唯一的运行期依赖）、`policy`（读配置与归一化统计）、
  `tools.paths`（`RUNS_DIR`）、`tools.terrain`（`StairSpec` 类型）。
* **`policy.py` ↔ `training.py` 是单向的**：`training` import `policy`，`policy` 不
  import `training`。理由：`policy` 要能在没有 SB3 / torch 的环境下被调用
  （例如只读 `summary.json` 的脚本），所以重的依赖只许放在 `training` 这一侧。
* **向上**被 `scripts/train.py`（全量 API）与 `scripts/play.py`
  （`load_policy` / `make_vec_env`）消费。
* **`curriculum.py` 不依赖本模块**，两者只在 `scripts/train.py` 里相遇。

## 8. 实测与经验教训

* `experience.md` §4.2 —— 奖励归一化的诚实 A/B（0.474 → 0.539、0.070 → 0.080）。
* `experience.md` §5.3 —— 观测归一化没跟权重：5 cm 上成功率 100% → 0%。
* `experience.md` §5.4 —— `pickle.loads(vecnormalize.pkl)` 的三个坑。
* `experience.md` §5.5 —— 并行 run ≤ 3 与 OOM 的 traceback 原文。
* `train.md` §1.4 —— `DummyVecEnv → VecMonitor → VecNormalize` 的顺序与理由。
* `train.md` §3.2 —— `mean|a|` 是判断「有没有学起来」的关键量。
* `README.md` §4.1 —— `*_vecnormalize.pkl` 必须与 `.zip` 成对搬动。

## 9. 已知限制与坑

* **`load_policy` 会建一个 1 环境向量环境**。对 8 环境训出来的模型，
  `n_steps × n_envs = 256` 会让 SB3 抱怨
  `UserWarning: You have specified a mini-batch size of 512, but because the
  RolloutBuffer is of size n_steps * n_envs = 256 ... truncated mini-batch of size 256`。
  **这条警告无害** —— 推理不更新参数。
* **`make_vec_env` 的 seed 是 `seed + i`**。想复现某次 rollout 必须记住
  `seed` 与 `n_envs` —— 换 `n_envs` 会改变每个子环境拿到的种子。
* **`RunDir.root` 默认是 `RUNS_DIR / name`**，`RUNS_DIR` 由 `tools.paths` 从
  `__file__` 推出来（`parents[2]`）。移动这个文件会改 `RUNS_DIR`。
* **`set_terrain` 之后必须让 `learn()` 自己去收集 rollout**，不能在两次 `learn()`
  之间手工插入 `step()`，否则 buffer 里会混入两个地形的转移。
* **没有 `set_terrain_per_env`**。地形是**整个向量环境共用**的；要给每个子环境不同
  地形得自己调 `env_method`。
* **`load_ppo` 要求先有 `venv`**，而 `venv` 的观测维度必须与权重一致，否则
  `ValueError: Observation spaces do not match`。用 `policy.env_config_for_model()`
  先恢复配置再建环境。
* `format_duration` 输出两位小数（`1.20h`）；`curriculum.format_seconds` 是同一份
  逻辑的副本，两处改要一起改。

## 10. 示例用法

```python
from robotdog.env import EnvConfig
from robotdog.tools.terrain import demo_stairs_spec
from robotdog.training import (RunDir, build_ppo, load_policy, make_vec_env,
                               mean_abs_action, set_terrain)

stair = demo_stairs_spec(step_height=0.09, tread=0.50, n_up=8, top_landing=1.20)
cfg = EnvConfig()
cfg.gait_prior = True
cfg.obs.step_height = True

# --- 训练一段 ---
run = RunDir("demo")
venv = make_vec_env(stair, n_envs=8, cfg=cfg, norm_reward=True)
model = build_ppo(venv, tensorboard_log=run.tensorboard)
model.learn(total_timesteps=2048, tb_log_name="ppo")
run.save_model(model, run.checkpoints / "stair9_final", venv)

# --- 换到更高的一级（重编译 MJCF）---
set_terrain(venv, demo_stairs_spec(step_height=0.12, tread=0.50, n_up=8,
                                   top_landing=1.20))
venv.close()

# --- 读回策略并推理 ---
from robotdog.policy import env_config_for_model, stair_from_height

model_path = str(run.checkpoints / "stair9_final.zip")
cfg2 = env_config_for_model(model_path)          # 优先读同目录 summary.json
model2, norm = load_policy(model_path, cfg=cfg2, stair=stair_from_height(9.0))
venv2 = model2.get_env()                         # 保持引用，别让它被 GC

obs = venv2.reset()
for _ in range(500):
    z = norm.normalize(obs) if norm is not None else obs
    action, _ = model2.predict(z, deterministic=True)
    obs, _, done, infos = venv2.step(action)
    if done[0]:
        break
print("mean|a| =", mean_abs_action(action))       # 健康值约 0.3~0.45
venv2.close()
```
