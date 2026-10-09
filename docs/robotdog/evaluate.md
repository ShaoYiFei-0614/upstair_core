# `robotdog/evaluate.py` — 策略评估矩阵与 ONNX / TorchScript 导出

## 1. 一句话定位

训练层（第 9 步的上半）：在**训练地形 + 没见过的地形 + 随机地形**上跑固定回合数的 rollout，把成功率/摔倒率/速度/姿态/漂移聚合成可比的表格与 JSON 报表，并把「原始观测 → 裁剪后动作」这条推理路径折叠成一个自包含的 `torch.nn.Module`，导出 ONNX 与 TorchScript。它从不渲染动画。

## 2. 职责边界

**做：**

- 地形清单：训练时见过的 4 个（`trained_terrains`）、刻意偏离训练网格的 9 个（`unseen_terrains`）、按 `TerrainRandomization` 均匀抽样的若干个（`randomized_terrains`），以及组合入口 `all_terrains`。
- 单地形评估（`evaluate_terrain`）与批量评估（`evaluate_all`），逐回合 trace 可选。
- 聚合（`summarize` 按名字前缀分组）、中文表格（`format_table`）、直线性判语（`straightness_line`）、结束原因统计（`termination_breakdown`）。
- JSON 报表写出（`write_report`，`indent=2, ensure_ascii=False, encoding="utf-8"`）。
- 导出：把 `VecNormalize` 的 running mean/var 烘进推理模块（`build_inference_module`），导出 ONNX（`export_onnx`）+ 校验图合法（`check_onnx`）+ 数值比对（`verify_onnx_numeric`），导出 TorchScript（`export_torchscript`）+ 加载（`load_torchscript`）。

**不做：**

- 不加载模型、不解析 `--run` / `--model` / `--vecnormalize`，也不决定用哪个 `EnvConfig`——那是 `scripts/eval.py`（它调 `env_config_for_model`）与 `scripts/eval.py:load_policy` 的事。
- 不训练、不推进课程（`robotdog/curriculum.py`、`scripts/train.py`）。
- **不做渲染**：不写 PNG，不出 GIF。GIF 在 `robotdog/render.py`，PNG 帧转储在 `scripts/eval.py:render_video`。
- 不建向量化环境；它驱动的是**单个** `Go2StairsEnv`，归一化由调用方适配（`scripts/eval.py` 的 `SingleEnvPredict`）。
- 不对导出件做运行时性能基准测试（只做数值一致性）。

## 3. 关键设计决策与理由

### 3.1 训练/没见过的地形划分，几何上是可核对的

```python
_TRAINED_STEP_HEIGHTS = (0.05, 0.12, 0.19)
_TRAINED_TREAD = 0.50
```

- `trained_terrains()` = `train_flat`（`flat_spec(n_up=0, n_down=0, tread=0.50, width=2.0, start_x=1.50)`）+ `train_h050` / `train_h120` / `train_h190`（`demo_stairs_spec(step_height=h, tread=0.50, n_up=8, top_landing=1.20)`）。
- 名字里的数字是**毫米**：`f"train_h{int(h * 1000):03d}"` → `train_h050` / `train_h120` / `train_h190`。
- `train_h050` 与审计地形 `demo_stairs_spec(h, 0.50, n_up=8, top_landing=1.20)` **几何上完全相同**，所以它同时是「训练地形」和「外部审计基准」。
- `unseen_terrains()` 刻意同时覆盖**内插**（`unseen_h07_t50` 的 7 cm 落在 5 与 12 cm 两个训练阶段之间）与**外推**（`unseen_h23_t50` 的 23 cm 高于最难训练阶段），再加非 50 cm 踏面、课程从未训练的金字塔（先上后下）与窄楼梯：

| 名字 | `step_height` | `tread` | `n_up` | `top_landing` | 其它 |
| --- | --- | --- | --- | --- | --- |
| `unseen_h07_t50` | 0.07 | 0.50 | 8 | 1.20 | 内插 |
| `unseen_h10_t45` | 0.10 | 0.45 | 8 | 1.20 | 非训练踏面 |
| `unseen_h15_t40` | 0.15 | 0.40 | 8 | 1.20 | |
| `unseen_h17_t55` | 0.17 | 0.55 | 8 | 1.20 | |
| `unseen_h23_t50` | 0.23 | 0.50 | 8 | 1.20 | 外推 |
| `unseen_h09_t35` | 0.09 | 0.35 | 6 | 1.00 | 更陡 + 更短 |
| `unseen_shallow` | 0.035 | 0.60 | 10 | 1.20 | 很缓但很长 |
| `unseen_pyramid` | 0.08 | 0.50 | 6（`n_down=6`） | 1.20（`landing=1.20`） | 先上后下 |
| `unseen_narrow` | 0.06 | 0.50 | 8 | 1.20（`width=1.2`） | 窄一半 |

**被否决的替代方案：** 只报训练地形上的成功率。否决理由是这个数字对交付没有信息量——5 cm / 12 cm / 19 cm 就是训练目标，达标是必须的；判断「有没有过拟合到训练网格」必须看偏离网格的地形。

### 3.2 导出把观测归一化**折叠**进模型

训练用了 `VecNormalize(norm_obs=True)`，策略实际吃的是 `clip((obs - mean) / sqrt(var + eps), ±clip_obs)`。导出裸策略会让部署方必须自己复现这套统计量，**容易搞错**（漏掉就是喂随机输入，见 `experience.md` §5.3）。所以 `build_inference_module` 把归一化烘进图里：**ONNX/TorchScript 的输入是原始观测维度**，归一化在内部完成。

统计量来源 `_vecnormalize_stats(model)`：从 `model.get_vec_normalize_env()` 拿 `obs_rms.mean` / `obs_rms.var`（float32）与 `clip_obs`；没有包装器时返回 `(None, None, 10.0)`——此时模块跳过归一化（`self.normalize = mean is not None`），而不是假装有。

**被否决的替代方案：** 导出裸策略 + 另存 `vecnormalize.pkl` 让部署方自己接。否决理由是两份文件不同步时会静默出错，而且 SB3 的 pickle 里是 **VecNormalize 对象**不是 dict，第三方部署环境很难正确解包（`experience.md` §5.4）。

### 3.3 `verify_onnx_numeric` 才是有意义的证明

`onnx.checker.check_model` 只证明图是合法的，不证明它算的是同一个函数。所以导出后额外用 onnxruntime 跑随机观测与 PyTorch 比对：

- `n_samples: int = 8`，`tol: float = 1e-4`；
- 样本是 `rng.normal(0, 1.0, (n_samples, obs_dim))` **加上** `rng.normal(0, 20.0, (2, obs_dim))`——第二组大值是为了走**观测裁剪**那条分支（`clip_obs=10.0`），否则裁剪路径永远不被覆盖；
- 缺 onnxruntime 时返回 `{"available": False, ...}` 且 `result["skipped"] = f"onnxruntime 未安装: {exc}"`，不抛异常。

`export_torchscript` 同理：`torch.jit.trace(..., strict=False)` + `torch.jit.freeze`，并同时比 `trace_max_abs_diff` 与 `frozen_max_abs_diff`，`numeric_match = bool(np.abs(a - b).max() <= 1e-5)`。**冻结前后都测**，因为常量折叠可能改变数值。

### 3.4 写文件要绕开非 ASCII 路径（libtorch 的 C++ 限制）

`torch.jit.save` / `torch.jit.load` 是 C++ 实现，路径按 `const char*` 处理，Windows 用 ANSI 代码页（这里是 **GBK / 936**）解码，于是 `D:\项目\...` 这类 UTF-8 路径变成乱码，libtorch 报：

```
Parent directory ... does not exist
```

尽管目录明明存在。这与 `robotdog/go2_model.py` 记录的 MuJoCo `meshdir` 失败**同一根因**。

绕法：`_ascii_scratch_dir()` 找一个**可写的纯 ASCII** 目录（依次试 `DSH_ASCII_SCRATCH` → `tempfile.gettempdir()` → `TEMP` → `TMP` → `SystemRoot\Temp` → `LOCALAPPDATA\Temp`，用 `dsh_write_probe.tmp` 实测可写），`_write_via_ascii_path` 让 `payload_writer` 先写到那里，再把字节拷到真实目标；`load_torchscript` 反向——先把 `.pt` 拷进 ASCII 目录再 `torch.jit.load`，加载完删掉。

**被否决的替代方案：** 把整个导出目录搬到 ASCII 路径下。否决理由是交付件必须落在工作区里（用户要的是 `runs/<name>/artifacts/`），搬走等于改变产物位置；而且 `load_torchscript` 还得能读用户给的任意路径，绕不过去。

### 3.5 摔倒率必须读 `info["fallen"]`，不能读 `terminated`

`terminate_on_success` 打开时，机器人登顶也会让 `terminated=True`。所以 `bool(terminated)` 会把**每一次成功都算成摔倒**，`success_rate + fall_rate` 可以超过 1。源码注释写得很直白（`evaluate.py` line 238）：

```
NOTE: must use info["fallen"], not `terminated`. With terminate_on_success the env
also sets terminated=True when the robot reaches the crest, so `bool(terminated)`
would count every success as a fall (success_rate + fall_rate could exceed 1).
```

实际实现是 `fell = bool(info.get("fallen", bool(terminated) and not success))`——只在 `info` 里没有 `fallen` 时才退回旧口径。

### 3.6 所有姿态/漂移量都相对**本回合出生位姿**测量

`reset()` 自带 `reset_xy_noise=0.12` / `reset_yaw_noise=0.10` 的横向与朝向抖动，那不是策略的责任。所以「走直线」的基准是 `self._y0` / `self._yaw0`（`Go2StairsEnv` 里记录的出生位姿），不是楼梯中线。同理 `mean_height_gain` 用绝对机身 z 的变化，`mean_base_height` 用「机身下方的地形高度」作参照。

### 3.7 `straightness_line` 的阈值刻意粗糙

要衡量的是「机器人有没有守住自己的车道」，不是毫米级路径跟踪。所以三段判语：

- `y <= 0.10 and head <= 5.0` → `"✅ 基本直线"`
- `y <= 0.25 and head <= 12.0` → `"⚠️ 有可见漂移"`
- 否则 → `"❌ 明显走偏"`

### 3.8 `randomized_terrains` 用固定 `seed=12345`

`np.random.default_rng(seed)` 是**每条分支各自重建**的局部 rng，不碰全局 `np.random`。固定默认种子是为了让「同一命令两次运行的随机地形集合相同」，否则报表的数字没有可比性。

## 4. 公开 API

### `_TRAINED_STEP_HEIGHTS` / `_TRAINED_TREAD`

```python
_TRAINED_STEP_HEIGHTS = (0.05, 0.12, 0.19)
_TRAINED_TREAD = 0.50
```

模块级常量，单位米。

### `trained_terrains`

```python
def trained_terrains() -> list[tuple[str, StairSpec]]:
```

- 语义：返回 §3.1 的 4 个训练地形，顺序 `train_flat`、`train_h050`、`train_h120`、`train_h190`。
- 返回：`list[tuple[str, StairSpec]]`。异常：无。

### `unseen_terrains`

```python
def unseen_terrains() -> list[tuple[str, StairSpec]]:
```

- 语义：返回 §3.1 表格里的 9 个地形，顺序就是表里的顺序。
- 返回：`list[tuple[str, StairSpec]]`。异常：无。

### `randomized_terrains`

```python
def randomized_terrains(n: int, seed: int = 12345) -> list[tuple[str, StairSpec]]:
```

- 语义：从 `TerrainRandomization()` 均匀抽样 `n` 个地形。docstring：`Uniformly sampled terrain from :class:`TerrainRandomization`.` 名字 `f"rand_{i:02d}_h{int(spec.step_height * 1000):03d}_t{int(spec.tread * 100):03d}"`（高度取毫米、踏面取厘米）。
- 参数：`n` 为抽取个数；`seed` 固定保证可复现。
- 返回：`list[tuple[str, StairSpec]]`，长度 `n`（`n <= 0` 时为空列表）。异常：无。

### `all_terrains`

```python
def all_terrains(include_random: int = 0, seed: int = 12345,
                 only: str = "all") -> list[tuple[str, StairSpec]]:
```

- 语义：按 `only` 拼装：
  - `only in ("all", "trained")` 加训练地形；
  - `only in ("all", "unseen")` 加没见过的地形；
  - `only in ("all", "random") or include_random` 加随机地形，个数 `include_random or (6 if only == "random" else 0)`。
- 参数：`only` 取 `"all"` / `"trained"` / `"unseen"` / `"random"`（`scripts/eval.py` 用 `choices` 约束）；`include_random` 是**个数**不是布尔。
- 返回：`list[tuple[str, StairSpec]]`。异常：无（未知 `only` 会落到「都不加」）。

### `class TerrainResult`

```python
@dataclass
class TerrainResult:
    terrain: str
    step_height: float
    tread: float
    n_up: int
    n_down: int
    crest_height: float
    episodes: int = 0
    successes: int = 0
    falls: int = 0
    timeouts: int = 0
    success_rate: float = 0.0
    fall_rate: float = 0.0
    timeout_rate: float = 0.0
    mean_return: float = 0.0
    mean_episode_length: float = 0.0
    mean_forward_speed: float = 0.0   # m/s, over the whole episode
    mean_distance: float = 0.0        # m travelled in +x
    mean_height_gain: float = 0.0     # m climbed (change in absolute base z)
    mean_base_height: float = 0.0     # m of clearance above the terrain under the base
    mean_abs_roll: float = 0.0
    mean_abs_pitch: float = 0.0
    max_abs_roll: float = 0.0
    max_abs_pitch: float = 0.0
    mean_abs_y: float = 0.0           # m, mean |lateral offset|
    max_abs_y: float = 0.0
    final_abs_y: float = 0.0
    mean_abs_heading_deg: float = 0.0
    max_abs_heading_deg: float = 0.0
    mean_lateral_drift_per_m: float = 0.0
    mean_survival_fraction: float = 0.0   # episode steps / max steps
    steps: int = 0
```

- docstring 说明字段与用户要求的对应关系：`success_rate` / `fall_rate` / `timeout_rate` 对应「成功率 / 摔倒次数」，`mean_forward_speed` 与姿态各项对应「速度 / 姿态」。
- 前 6 个字段**无默认值**，必须给。
- `def as_dict(self) -> dict[str, Any]: return asdict(self)`。

### `class TerrainTrace`

```python
@dataclass
class TerrainTrace:
    terrain: str
    episodes: list[dict[str, Any]] = field(default_factory=list)
```

docstring：`Raw per-episode trace for one terrain, used for plotting / inspection.`

### `evaluate_terrain`

```python
def evaluate_terrain(model, spec: StairSpec, name: str = "terrain", *,
                     n_episodes: int = 20, seed: int = 0,
                     cfg: EnvConfig | None = None,
                     deterministic: bool = True,
                     collect_trace: bool = False,
                     max_episode_steps: int | None = None
                     ) -> tuple[TerrainResult, TerrainTrace]:
```

- 语义：`Go2StairsEnv(stair=spec, cfg=cfg, seed=seed)`；`max_steps = max_episode_steps or int(env.cfg.episode_length_s / env.cfg.control_dt)`；第 `ep` 回合 `env.reset(seed=seed + ep)`；每步 `model.predict(obs, deterministic=deterministic)`；记录 `env.base_rpy[0]` / `[1]`、`env.lateral_offset`、`env.heading_error`。
- 判据：`success = bool(info.get("is_success", False))`；`fell = bool(info.get("fallen", bool(terminated) and not success))`；`res.timeouts += int(truncated and not fell)`（见 §3.5）。
- 派生量：`heights.append(float(env.base_pos[2]) - float(env.terrain_height_under_base))`；`survival.append(steps / max_steps)`；`drift_per_m.append(ep_mean_abs_y / abs(travelled) if abs(travelled) > 1e-6 else 0.0)`。
- `collect_trace=True` 时每个回合产出 trace 条目，键：`episode`、`steps`、`return`（3 位）、`success`、`fell`、`timeout`、`distance`（3）、`height_gain`（3）、`max_abs_roll`（4）、`max_abs_pitch`（4）、`final_x`（3）、`mean_abs_y`（4）、`max_abs_y`（4）、`final_abs_y`（4）、`mean_abs_heading_deg`（`round(float(np.degrees(abs_heads[-1])), 3)`）。
- `res.steps = int(np.sum(lengths))`，其中 `n = max(res.episodes, 1)` 用于防零除。
- 返回：`(TerrainResult, TerrainTrace)`（trace 为空 dataclass 时 `episodes` 是空 list）。`env.close()` 在 `finally` 里。
- 异常：无显式抛出；`cfg` 与模型不匹配时由 `PPO.predict` 抛。

### `evaluate_all`

```python
def evaluate_all(model, terrains: Sequence[tuple[str, StairSpec]], *,
                 n_episodes: int = 20, seed: int = 0,
                 cfg: EnvConfig | None = None,
                 deterministic: bool = True,
                 progress: bool = True
                 ) -> tuple[list[TerrainResult], list[TerrainTrace]]:
```

- 语义：逐地形调 `evaluate_terrain(..., collect_trace=True)`。`progress=True` 时每个地形打两行：进度行 `f"  [{i:>2}/{len(terrains)}] {name:<22} 台阶={spec.step_height * 100:5.1f}cm 踏面={spec.tread * 100:4.0f}cm 级数={spec.n_up}..."`，结果行 `成功率=… 摔倒率=… 速度=…m/s (…s)`。
- 返回：`(results, traces)`，两个列表等长且与 `terrains` 同序。

### `summarize`

```python
def summarize(results: Sequence[TerrainResult]) -> dict[str, Any]:
```

- 语义：按组聚合，组名取地形名 `_` **之前**的词缀（`train_*` → `train`，`unseen_*` → `unseen`，`rand_*` → `rand`）。
- 每组键：`terrains`、`episodes`、`success_rate`（round 4）、`fall_rate`（4）、`timeout_rate`（4）、`mean_forward_speed`（4）、`mean_base_height`（4）、`mean_abs_pitch`（4）、`mean_abs_roll`（4）、`mean_return`（3）、`mean_abs_y`（4）、`max_abs_y`（4）、`mean_abs_heading_deg`（3）、`mean_lateral_drift_per_m`（4）、`worst_terrain`（`min(items, key=lambda i: i.success_rate).terrain`）、`best_terrain`（max 同形）。
- 返回：`dict[str, Any]`，键是组名。空输入 → 空 dict。

### `format_table`

```python
def format_table(results: Sequence[TerrainResult]) -> str:
```

- 语义：中文表头 + `"-" * len(head)` 分隔行，然后每行 22/6/6/5/5/8/8/7/8/8/8/7/8/8/10/8/8/7/8/8 宽度对齐：`地形`、`台阶`、`踏面`、`级数`、`回合`、`成功率`、`摔倒率`、`超时`、`速度`、`前进`、`爬升`、`离地`、`|roll|`、`|pitch|`、`最大pitch`、`|y|均`、`|y|最大`、`|y|末`、`航向°`、`漂移/m`。
- 返回：单个多行字符串（含末尾换行）。

### `straightness_line`

```python
def straightness_line(r: TerrainResult) -> str:
```

- 语义与阈值见 §3.7。
- 返回：三段判语之一。异常：无。

### `write_report`

```python
def write_report(path: str | Path, *, results, summary, meta,
                 traces=None, include_traces: bool = True) -> Path:
```

- 语义：`payload = {"meta": meta, "summary": summary, "results": [r.as_dict() for r in results]}`；`include_traces and traces` 为真时 `payload["traces"] = [asdict(t) for t in traces]`；`path.parent.mkdir(parents=True, exist_ok=True)`；`path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")`。
- 参数全部关键字。`meta` 常见键（由 `scripts/eval.py` 给）：`model`、`vecnormalize`、`episodes_per_terrain`、`deterministic`、`seed`、`only`、`n_random`、`elapsed_s`、`obs_dim`、`action_dim`、`num_parameters`。
- 返回：`Path`。
- 异常：父目录不可写时 `path.write_text` 抛 `PermissionError: [Errno 13] Permission denied`（见 §9）。

### `termination_breakdown`

```python
def termination_breakdown(results) -> dict[str, int]:
```

- 语义：`Counter` 累加四个键 `success`、`fall`、`timeout_survived`、`episodes`。
- 返回：`dict[str, int]`（`Counter` 的普通 dict 视图）。

### `_vecnormalize_stats`

```python
def _vecnormalize_stats(model) -> tuple[np.ndarray | None, np.ndarray | None, float]:
```

- docstring：「若模型带 VecNormalize, 取出 ``(mean, var, clip_obs)``。」
- 语义：`get_vec_normalize_env()` 拿包装器；缺失返回 `(None, None, 10.0)`，否则返回 float32 的 `mean` / `var` 与 `float(getattr(venv, "clip_obs", 10.0))`。

### `build_inference_module`

```python
def build_inference_module(model, *, epsilon: float = 1e-8):
```

- docstring：把**原始**观测映射到裁剪后动作的 `torch.nn.Module`；把 VecNormalize 的 running statistics 烘进去，导出件自包含。
- 内层 `class InferenceModule(nn.Module)`：持 `self.policy`；`self.normalize = mean is not None`；注册 buffer `obs_mean` 与 `obs_std = torch.as_tensor(np.sqrt(var + epsilon))`；`self.clip_obs`；`self.clip_action = 1.0`。`forward` 顺序：归一化 → `torch.clamp(x, -clip_obs, clip_obs)` → `self.policy.extract_features(x)` → `latent_pi, _ = self.policy.mlp_extractor(features)` → `actions = self.policy.action_net(latent_pi)` → `torch.clamp(actions, -self.clip_action, self.clip_action)`。
- 返回：`module`，已 `eval()` 且处于 CPU。异常：无（`mean` 为 `None` 时跳过归一化）。

### `export_onnx`

```python
def export_onnx(model, path, *, obs_dim: int | None = None, opset: int = 17) -> dict[str, Any]:
```

- 语义：`obs_dim = obs_dim or int(np.prod(model.observation_space.shape))`；`torch.onnx.export(module, (dummy,), str(path), input_names=["observation"], output_names=["action"], dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}}, opset_version=opset, do_constant_folding=True, dynamo=False)`。
- 返回：`dict`，键 `path`、`size_bytes`、`obs_dim`、`action_dim`、`opset`、`normalization_baked_in`、`reference_action`、`onnx_check`（`check_onnx` 的结果）、`numeric_match`（`verify_onnx_numeric` 的结果）。
- 异常：无显式抛出（导出失败由 torch 抛）。

### `check_onnx`

```python
def check_onnx(path) -> dict[str, Any]:
```

- 语义：`onnx.load` + `onnx.checker.check_model`。
- 返回：键 `ir_version`、`producer`（`f"{producer_name} {producer_version}".strip()`）、`inputs`、`outputs`，每个张量的形状写成 `[d.dim_param or d.dim_value for d in ...shape.dim]`。
- 异常：图非法时 `onnx.checker.ValidationError`。

### `verify_onnx_numeric`

```python
def verify_onnx_numeric(path, module, obs_dim, *, n_samples: int = 8,
                        tol: float = 1e-4) -> dict[str, Any]:
```

- docstring：用 onnxruntime 跑随机观测并与 PyTorch 比对；这才是「导出文件算的是同一个函数」的真正证明；`onnx.checker` 只证明图是合法的。
- 语义：样本构成见 §3.3；先在 `available: False` 下尝试 import onnxruntime，失败写 `result["skipped"] = f"onnxruntime 未安装: {exc}"`。
- 返回：键 `available`、`samples`、`max_abs_diff`、`tol`、`passed`（缺失时另有 `skipped`）。

### `_ascii_scratch_dir`

```python
def _ascii_scratch_dir() -> Path:
```

- 语义：找一个可写的**纯 ASCII** 目录，见 §3.4。候选顺序：`DSH_ASCII_SCRATCH` → `tempfile.gettempdir()` → `TEMP` → `TMP` → `SystemRoot\Temp` → `LOCALAPPDATA\Temp`；每个用 `dsh_write_probe.tmp` 实测可写后返回。
- 异常：全都不行时 `raise RuntimeError("找不到可写的纯 ASCII 临时目录; libtorch 的 C++ 文件写入无法处理非 ASCII 路径 (如 D:\\项目\\...)。请设置环境变量 DSH_ASCII_SCRATCH 指向一个纯 ASCII 目录。")`。

### `_write_via_ascii_path`

```python
def _write_via_ascii_path(payload_writer, target: Path) -> None:
```

- docstring：`先跑 ``payload_writer(ascii_path)``, 再把字节拷到 ``target``.`
- 语义：`scratch = _ascii_scratch_dir() / f"dsh_export_{os.getpid()}.tmp"`；调 `payload_writer(str(scratch))`；读取字节后 `finally` 里 `unlink`；`target.write_bytes(data)`。
- 异常：`_ascii_scratch_dir` 的 `RuntimeError`；`payload_writer` 自己的异常原样冒出。

### `load_torchscript`

```python
def load_torchscript(path: str | Path, *, device: str = "cpu"):
```

- docstring：`torch.jit.load` 同样是 C++，所以 `D:\项目\...` 下的 `.pt` 无法直接打开；先在纯 ASCII 目录放临时拷贝，加载完再删。
- 语义：路径本身 ASCII 时先直接 `torch.jit.load(str(path), map_location=device)`，`except (RuntimeError, ValueError)` 再走临时拷贝；否则直接走临时拷贝，临时文件名 `f"dsh_load_{os.getpid()}_{uuid.uuid4().hex[:8]}.pt"`。
- 返回：加载好的 `ScriptModule`。异常：两条路径都失败时最后一次的异常。

### `export_torchscript`

```python
def export_torchscript(model, path, *, obs_dim: int | None = None) -> dict[str, Any]:
```

- docstring：```torch.jit.trace`` the same inference module (raw obs -> action).`
- 语义：`traced = torch.jit.trace(module, (dummy,), strict=False)`；`traced = torch.jit.freeze(traced)`；`_write_via_ascii_path(traced.save, path)`；用 `batch = rng.normal(0, 1.5, (16, obs_dim))` 比对。
- 返回：键 `path`、`size_bytes`、`obs_dim`、`action_dim`、`normalization_baked_in`、`reference_action`、`trace_max_abs_diff`、`frozen_max_abs_diff`、`numeric_match`（`bool(np.abs(a - b).max() <= 1e-5)`）、`ascii_scratch_workaround`（`not str(path).isascii()`）。

## 5. 内部结构与数据流

评估：

```
scripts/eval.py
  → env_config_for_model(model_path)                     # robotdog.training
  → all_terrains(include_random=args.n_random, seed=args.seed, only=args.only)
      → trained_terrains() / unseen_terrains() / randomized_terrains(n, seed)
  → load_policy(model_path, vn_path, cfg=cfg) → (model, venv)
  → SingleEnvPredict(model, venv)                         # 手工归一化适配器
  → evaluate_all(predictor, terrains, n_episodes=…, seed=…, cfg=…, deterministic=…)
      → evaluate_terrain(model, spec, name, …, collect_trace=True)
          → Go2StairsEnv(stair=spec, cfg=cfg, seed=seed)
          → env.reset(seed=seed + ep) → model.predict(obs, deterministic=…) → env.step(action)
          → info["is_success"] / info["fallen"] / info["episode"]
          → TerrainResult + TerrainTrace
  → summarize(results)  /  format_table(results)  /  termination_breakdown(results)
  → write_report(out, results=…, summary=…, meta=…, traces=…)
```

导出：

```
export_onnx(model, path, obs_dim=…, opset=17)
  → _vecnormalize_stats(model) → (mean, var, clip_obs)
  → build_inference_module(model, epsilon=1e-8)           # 归一化烘进 forward
  → torch.onnx.export(module, (dummy,), …)
  → check_onnx(path)                                      # onnx.checker
  → verify_onnx_numeric(path, module, obs_dim)            # onnxruntime 比对

export_torchscript(model, path, obs_dim=…)
  → build_inference_module(model)
  → torch.jit.trace(module, (dummy,), strict=False) → torch.jit.freeze(…)
  → _write_via_ascii_path(traced.save, path)              # libtorch 只吃 ASCII 路径
  → trace/frozen 两次数值比对

load_torchscript(path, device="cpu")
  → str(path).isascii() ? torch.jit.load 直接读 : 拷到 _ascii_scratch_dir() 再读
```

## 6. 关键常量与默认值

| 名称 | 值 | 说明 |
| --- | --- | --- |
| `_TRAINED_STEP_HEIGHTS` | `(0.05, 0.12, 0.19)` | 米 |
| `_TRAINED_TREAD` | `0.50` | 米 |
| `train_flat` | `flat_spec(n_up=0, n_down=0, tread=0.50, width=2.0, start_x=1.50)` | |
| `train_h0xx` | `demo_stairs_spec(step_height=h, tread=0.50, n_up=8, top_landing=1.20)` | 名字用毫米：`train_h050` / `train_h120` / `train_h190` |
| 没见过的地形数 | `9` | |
| `unseen_pyramid` | `0.08 / 0.50 / n_up=6 / n_down=6 / landing=1.20` | 唯一带下坡的 |
| `unseen_narrow` | `0.06 / 0.50 / n_up=8 / width=1.2` | |
| `randomized_terrains.seed` | `12345` | |
| `all_terrains` 的 `only="random"` 默认个数 | `6` | |
| `evaluate_terrain.n_episodes` | `20` | `scripts/eval.py --episodes` 默认也是 20，但文档建议 ≥40 |
| `evaluate_terrain.seed` | `0` | 实际逐回合用 `seed + ep` |
| `evaluate_all.n_episodes` | `20` | |
| `export_onnx.opset` | `17` | `scripts/eval.py --onnx-opset` 默认 17 |
| `verify_onnx_numeric.n_samples` / `tol` | `8` / `1e-4` | 大值样本 `rng.normal(0, 20.0, (2, obs_dim))` |
| `export_torchscript` 比对样本 | `rng.normal(0, 1.5, (16, obs_dim))` | |
| `numeric_match` 阈值（TorchScript） | `1e-5` | |
| `build_inference_module.epsilon` | `1e-8` | |
| `clip_obs` 兜底 | `10.0` | `_vecnormalize_stats` 与 `build_inference_module` |
| `clip_action` | `1.0` | 动作空间是 `Box(-1.0, 1.0, (12,))` |
| `straightness_line` 阈值 | 横向 `0.10` / `0.25` m，航向 `5.0` / `12.0`° | |
| `write_report` 编码 | `indent=2, ensure_ascii=False, encoding="utf-8"` | |
| `drift_per_m` 分母下限 | `1e-6` | 前进量太小时置 0 |

## 7. 与其它模块的关系

```
robotdog/tools/terrain.py (StairSpec, TerrainRandomization, demo_stairs_spec, flat_spec)
robotdog/env.py           (EnvConfig, Go2StairsEnv)
        ↓
robotdog/evaluate.py       ← 本模块
        ↑
scripts/eval.py            (唯一的驱动方；还额外实现 PNG 帧转储与 SingleEnvPredict)
```

- **向下依赖 `robotdog.env`**：评估必须驱动真实环境并读 `info` / `env.base_pos` / `env.base_rpy` / `env.lateral_offset` / `env.heading_error` / `env.terrain_height_under_base`。这些量只有环境知道，把它们复制到评估层会立刻不同步。
- **不依赖 `robotdog.training`**：`env_config_from_run` / `env_config_for_model` 由 `scripts/eval.py` 调用后再把 `cfg` 传进来。这样 `evaluate.py` 可以在没有任何 `runs/` 目录的情况下工作（只给 `model` + `spec`）。
- **不依赖 `torch` / `onnx` 之外的训练栈**：导出路径只碰 `model.policy`（SB3 的 `ActorCriticPolicy`）与 `model.get_vec_normalize_env()`，不 import `stable_baselines3`。所以「评估」与「导出」可以在同一个文件里共存而不引入额外耦合。
- **被 `scripts/eval.py` 消费**：`--only` / `--episodes` / `--n-random` / `--deterministic|--stochastic` / `--export` / `--onnx-opset` / `--out` 全部翻译成本模块的参数；`--render-video` 单独走 `eval.py` 自己的 `render_video`（用 `robotdog.tools.png.write_png`），**不经过 `robotdog/render.py`**。
- **`scripts/render_checkpoint.py` 复用本模块的地形清单**：`--unseen` / `--trained` / `--random-terrains` 都来自 `unseen_terrains()` / `trained_terrains()` / `randomized_terrains()`。方向是脚本 → 评估模块，不是评估模块 → 渲染模块。
- **`robotdog/render.py` 不依赖本模块**：渲染只接 `Go2StairsEnv` + `Predictor`，所以两张产物（JSON 报表 vs GIF）互相独立。

## 8. 实测与经验教训

- `train_h050` 几何上与审计地形相同，参考 checkpoint 的评估结果与完整仓的官方报告一致（`train_flat` 100%、`train_h050` 100%、`train_h120` 0%、`train_h190` 0%）：`train.md` §3.1。
- 逐高度能力带（8.0/9.0/9.5/10.0/10.5/11.0/11.5 cm）与「11.0 cm 是墙」：`train.md` §3.1、`experience.md` §6。
- 三条交付策略在 6 档 × 40 回合上的均值只有 ±5~10 pt 的波动，统计上互不显著：`train.md` §3.1。
- 打开观测归一化却没跟上权重 ⇒ 策略退化成随机（导出必须折叠统计量的直接理由）：`experience.md` §5.3。
- `pickle.loads(vecnormalize.pkl)` 返回的是 VecNormalize 对象、多余属性访问会 `RecursionError`：`experience.md` §5.4。
- 10 回合的评估能给出 ±15 pt 的错觉，验收要 40 回合 × 6 档且共用种子：`experience.md` §5.7。
- 估值脚本把 cm 当成 m（`demo_stairs_spec(step_height=height)` 而不是 `height / 100.0`）导致在测 500/800/1100 cm 的台阶：`experience.md` §5.2。**本模块的所有地形表都用米**，与这条教训直接相关。
- `model.policy(tensor)` 的返回值不是二元组（`ValueError: too many values to unpack`），要用 `model.predict(z, deterministic=True)`：`experience.md` §5.8。

## 9. 已知限制与坑

1. **评估工作区外的 checkpoint 会在写报告时失败。** `--run` / `--model` 指向仓库外的权重时，`write_report` 的 `path.write_text(...)` 抛：

   ```
   PermissionError: [Errno 13] Permission denied
   ```

   把权重先拷进仓库再评估。

2. **回合数至少 40。** 10 回合能给出 ±15 pt 的错觉（原项目踩过：10 回合看是 90%，40 回合只有 65%）。本模块默认 `n_episodes = 20`，`scripts/eval.py` 的 docstring 直接写「⚠ 评估至少 40 回合」，默认值**不是**验收标准。

3. **导出路径碰不得非 ASCII。** `torch.jit.save` / `torch.jit.load` 的 `Parent directory ... does not exist` 见 §3.4。`export_torchscript` 的返回值里有 `ascii_scratch_workaround` 标记这次是否走了绕路；`DSH_ASCII_SCRATCH` 没设而机器上所有候选目录都不可写时抛 `RuntimeError`。

4. **`onnxruntime` 缺失不会让导出失败。** `verify_onnx_numeric` 只写 `{"available": False, "skipped": …}`，`numeric_match` 也就不存在。CI 里必须显式检查 `available`，否则「数值一致」这个保证会被静默跳过。

5. **`randomized_terrains` 的名字里高度是毫米、踏面是厘米。** `f"rand_{i:02d}_h{int(step_height * 1000):03d}_t{int(tread * 100):03d}"`，解析名字时要按这两种单位分别还原。

6. **`all_terrains` 的 `only` 传未知值不会报错**，会静默返回空列表；是 `scripts/eval.py` 的 `raise SystemExit("没有可评估的地形 (检查 --only / --n-random)")` 替你发现的。

7. **`straightness_line` 的阈值是刻意的粗判语**，不要当成路径跟踪指标用；精确数字在 `TerrainResult` 的 `mean_abs_y` / `max_abs_y` / `final_abs_y` / `mean_abs_heading_deg` 里。

8. **`mean_base_height` 与 `mean_height_gain` 口径不同**：前者是「机身相对其下方地形的离地高度」的均值，后者是「绝对 z 的变化」。爬楼梯成功时前者可能几乎不变（地形一起升了）。

9. **`evaluate_terrain` 的 `fell` 回退分支仍不安全。** `info.get("fallen", bool(terminated) and not success)` 只在环境**没有**提供 `fallen` 时才正确——如果未来某个环境版本忘了写这个键，成功率与摔倒率会重新互相污染。

10. **`TerrainResult` 前 6 个字段没有默认值**，手工构造时（比如从 JSON 里重建）必须全给，否则 `TypeError`。

## 10. 示例用法

在自己的脚本里评估一个 checkpoint：

```python
from robotdog.evaluate import (evaluate_all, format_table, summarize,
                               termination_breakdown, write_report)
from robotdog.training import env_config_for_model, load_ppo, make_vec_env
from robotdog.tools.terrain import flat_spec

zip_path = r"runs\course22\checkpoints\medium_final.zip"
cfg = env_config_for_model(zip_path)
venv = make_vec_env(flat_spec(), n_envs=1, cfg=cfg, seed=0)
model = load_ppo(zip_path, venv, device="cpu")

from robotdog.evaluate import unseen_terrains
results, traces = evaluate_all(model, unseen_terrains(), n_episodes=40, cfg=cfg)
print(format_table(results))
print(termination_breakdown(results))
write_report("runs/course22/artifacts/eval_report.json",
             results=results, summary=summarize(results),
             meta={"model": zip_path, "episodes_per_terrain": 40}, traces=traces)
```

只导出 ONNX 与 TorchScript：

```python
from robotdog.evaluate import export_onnx, export_torchscript

onnx_info = export_onnx(model, "runs/course22/artifacts/policy.onnx", obs_dim=cfg.obs.size)
print(onnx_info["normalization_baked_in"], onnx_info["numeric_match"]["passed"])

ts_info = export_torchscript(model, "runs/course22/artifacts/policy.pt", obs_dim=cfg.obs.size)
print(ts_info["frozen_max_abs_diff"], ts_info["ascii_scratch_workaround"])
```

命令行（真实入口，注意 `--episodes` 至少要 40）：

```bat
.venv\Scripts\python.exe scripts\eval.py --run runs\course22 --episodes 40 ^
  --only all --n-random 6 --out runs\course22\artifacts\eval_report.json
```
