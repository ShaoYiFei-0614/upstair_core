# `robotdog/render.py` — 回放 checkpoint 并导出动画 GIF

## 1. 一句话定位

训练层（第 9 步的下半，「肉眼版证据」）：把一个 checkpoint 在指定楼梯上回放若干回合，逐帧抓 MuJoCo 离屏渲染画面，叠上中文 HUD，编码成一张动画 GIF。JSON 报表告诉你策略**达到** 45% 成功率，本模块告诉你它**怎么**走的。

## 2. 职责边界

**做：**

- 回放与抓帧：`render_rollout` 驱动 `Go2StairsEnv`，每 `frame_stride` 个控制步保存一帧 `env.render(...)` 的 RGB 图，并记录每回合的步数/前进/爬升/速度/成功/摔倒/超时。
- 每回合与整个任务的统计：`EpisodeRender`（含 `status` 中文判语）、`RenderResult`（`successes` / `success_rate` / `summary()` / `episode_of_frame()`）。
- HUD 文字：`hud_lines` 生成文本行（可 `compact`），`overlay_hud` 用半透明底板 + TrueType/位图字体画到左上角。
- 输出：`save_gif`（动画 GIF）、`save_frames`（逐帧无依赖 PNG）、`contact_sheet`（横向拼图）、`copy_into`（把导出的模型搬进 `output/`）。
- 调色板：`_global_palette`，全片共用一套 256 色。

**不做：**

- **不做评估矩阵**：没有训练地形/没见过的地形清单，没有成功率聚合报表。那是 `robotdog/evaluate.py`。
- **不加载模型**：不碰 `PPO.load` / `VecNormalize.load` / `env_config_for_model`。它只接受一个已经能 `predict(obs, deterministic=...)` 的对象（`Predictor` 协议）。
- **不关闭环境**：`render_rollout` 明确**不**调 `env.close()`——见 §3.1。
- 不写 ONNX / TorchScript（`robotdog/evaluate.py`）。
- 不做 ffmpeg 视频（mp4/avi/webm）导出，也不做音轨。
- 不决定产物目录（`robotdog/paths.py` 的 `OUTPUT_GIFS_DIR` 等），不解析命令行（`scripts/render_checkpoint.py`）。
- 不定义地形几何，不算 `crest_height`。

## 3. 关键设计决策与理由

### 3.1 帧来自 MuJoCo 离屏渲染，且 `render_rollout` 故意不关环境

帧就是 `Go2StairsEnv.render(camera=..., width=..., height=...)` 的返回值，也就是 MuJoCo 的离屏 OpenGL 渲染：**没有显示窗口**，因此能过 RDP、能在后台 run 里跑。

`render_rollout` 的 docstring 写得很直白：

```
The environment is *not* closed here -- the caller owns it (the renderer
keeps a GL context alive and closing it early would force a recompile).
```

**被否决的替代方案：** 让 `render_rollout` 自己 `env.close()`「省事」。否决理由是渲染器持有 GL 上下文（`self._renderer` / `self._render_cam` / `self._scene_option`），提前关掉会强制下一次重编译模型（MJCF 编译约 250 ms），批量渲染多个地形时这个代价会累积。

### 3.2 GIF 容器：Pillow 自适应调色板 + 帧间差分，不需要 ffmpeg

模块 docstring 的第二条设计说明：

```
* The GIF container is written with Pillow (already present as a TensorBoard
  dependency).  Pillow gives us an adaptive palette plus frame differencing for
  free, which matters a lot: a 900-frame raw dump would be ~500 MB, the same
  animation is 2-6 MB as a GIF.
```

- **为什么不需要 ffmpeg**：GIF 是逐帧索引色容器，`PIL.Image.save(save_all=True, append_images=..., duration=..., loop=..., optimize=..., disposal=2)` 就能写；引入 ffmpeg 等于给「看一眼策略怎么走」这个需求配一个外部二进制依赖。
- **为什么不需要 Pillow 之外的库**：GIF 路径确实**要求** Pillow（`from PIL import Image`）；Pillow 随 TensorBoard 一起装，已经在环境里。没有 Pillow 时这条路径直接 `ImportError`。
- 逐帧 PNG 转储走的是**另一条**路径：`save_frames` → `robotdog/tools/png.py` 的 `write_png`（只用 `zlib` + `struct` 手写 PNG 容器），所以「不想装 Pillow 也要能 dump 帧」这个需求由 `png.py` 满足，由 `save_frames` 使用。
- 体积对照见上：900 帧原始转储约 **500 MB**，同一动画 GIF 是 **2–6 MB**。

### 3.3 全片共用一套调色板，而不是逐帧自适应

`_global_palette` 从**整个回合**里等间隔抽 `sample`（默认 16）帧，缩成 1/4 大小的缩略图，拼成 4 列的 mosaic，再 `mosaic.quantize(colors=256, method=Image.MEDIANCUT)`。

docstring 说明了理由：

```
A per-frame adaptive palette makes every frame carry its own colour table
*and* re-dithers the noisy MuJoCo background differently each time, which
both bloats the GIF and makes the sky shimmer.  One global palette fixes
both: the colour table is written once and the background stays stable.
```

- **否决方案：逐帧自适应调色板。** 两个后果：每帧都要带自己的颜色表（GIF 文件变大），以及 MuJoCo 那个带噪声的背景每帧被**重新抖动**成不同的颜色，观感上天空会「闪」。全片一套调色板同时修掉这两点。
- 抽 16 帧而不是全部帧：调色板只需要覆盖画面的颜色分布，抽帧把 `quantize` 的代价从 O(全部帧) 降到 O(16)，同时避免某一段（比如全黑的摔倒后画面）主导调色板。

### 3.4 缩小**先做**，HUD **后画**

`RenderOptions.scale` 的注释：

```
Box-filter factor applied to each frame before the HUD is drawn (so the
text is rasterised at the final size and stays sharp).
```

调用方 `scripts/render_checkpoint.py:labelled_frames` 按这个约定实现：`gif_frames` 先 `downscale(frame, opts.scale)` 再 `overlay_hud`，`strip_frames` 保留全分辨率另画一遍。这样 `--scale 2`（默认，半尺寸）时 HUD 文字是在 **GIF 的真实像素尺寸**上光栅化的，不会先画大字再缩成糊字。`save_gif(..., scale=1)` 因此不再二次缩放。

**被否决的替代方案：** 让 `save_gif` 自己的 `scale` 参数承担全部缩小。否决理由就是上面这条——`save_gif` 只能对已经贴好 HUD 的帧做 box filter，文字会糊。所以 `save_gif.scale` 只在「帧本来就是全分辨率、又不打算叠加 HUD」时才有用。

### 3.5 HUD 用内置位图字体链

`_FONT_CANDIDATES` 按顺序试：

```python
(r"C:\Windows\Fonts\msyh.ttc",       # 微软雅黑
 r"C:\Windows\Fonts\msyhbd.ttc",     # 微软雅黑 Bold
 r"C:\Windows\Fonts\simhei.ttf",     # 黑体
 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
```

都读不到（损坏字体文件会被 `except Exception: continue` 跳过）时退到 matplotlib 自带的 `fonts/ttf/DejaVuSans.ttf`，再不行 `ImageFont.load_default()`（Pillow 内置位图字体）。

- **为什么要字体链**：HUD 里有中文（`回合`、`登顶成功`、`摔倒`、`台阶`、`踏面`、`顶端`）。Pillow 的默认位图字体不含 CJK 字形，直接用会画出一堆方框。
- **为什么要退到位图字体**：`overlay_hud` 在算底板宽度时对 `draw.textbbox` 有 `except Exception` 回退（注释：`bitmap font has no textbbox`），退化成 `(0, 0, len(text) * size // 2, size)`。位图字体下底板宽度是**估算**的，所以窄字会略宽、宽字可能贴边——这是可接受的降级，不是 bug。
- `overlay_hud` 先铺一层 `fill=(0, 0, 0, 140)` 的半透明底板（注释：`Translucent plate so white text stays readable on any background.`），再画纯白文字；字号默认 `max(12, frame.shape[0] // 26)`（540 高 → 20 px）。

### 3.6 渲染相机强制六组 geom 可见

`robotdog/env.py` 的 `render()` 里，`mjv_defaultOption` 出厂的 `geomgroup = [1, 1, 1, 0, 0, 0]`，后三组默认隐藏；楼梯与触觉几何就在被隐藏的组里，不强制打开时 GIF 里只有一只悬在空中的狗。所以渲染前逐组 `self._scene_option.geomgroup[g] = 1`。失败的形态与修法记在 `experience.md` §1.2。

本模块只通过 `env.render(camera=..., ...)` 使用它，不自己碰 `MjvOption`。

### 3.7 两个帧集合、`compact` 阈值

`scripts/render_checkpoint.py:labelled_frames` 返回 `(gif_frames, strip_frames)`：

- `gif_frames`：缩小 → 贴 HUD。`compact = args.height // max(1, opts.scale) < 400`，即 GIF 最终高度不足 400 px 时丢掉姿态行与帧号行（`hud_lines` 的 `compact=True` 分支），否则五行文字会吃掉小 GIF 四分之一的画面。
- `strip_frames`：全分辨率 + 完整 HUD，供 `contact_sheet` 出静态拼图。

### 3.8 交付件只有 GIF

`output/gifs/<name>.gif` 是**唯一**交付件。`output/stills/`（静态拼图）、`output/frames/`（逐帧 PNG）、`output/reports/`（逐回合 JSON）、`output/exports/`（导出的模型）都是中间产物，分别由 `--no-strip` / `--save-frames` / `--no-report` 控制是否创建（`scripts/render_checkpoint.py` 的 `--no-strip` help 原文：「不写 output\stills\ 的静态拼图（第二十五步起 output\ 只交付 GIF）」）。批量重渲交付 GIF 时建议 `--no-strip --no-report`。

## 4. 公开 API

`__all__` 为 `["RenderOptions", "EpisodeRender", "RenderResult", "render_rollout", "save_gif", "save_frames", "contact_sheet", "overlay_hud", "hud_lines"]`。注意 `Predictor`、`_load_font`、`_global_palette`、`copy_into` **不在** `__all__` 里，但都是模块级公开名称（`copy_into` 被 CLI 使用）。

### `class Predictor(Protocol)`

```python
class Predictor(Protocol):  # pragma: no cover - structural typing only
    """Anything with SB3's ``predict(obs, deterministic=...)`` signature."""

    def predict(self, obs: Any, deterministic: bool = True): ...
```

结构化类型：任何带 `predict(obs, deterministic=...)` 的对象都满足，不需要继承。所以 `PPO`、`scripts/eval.py` 的 `SingleEnvPredict`、`scripts/render_checkpoint.py` 的 `PriorPolicy` 都能直接传。

### `class RenderOptions`

```python
@dataclass
class RenderOptions:
    """Frame-capture settings for :func:`render_rollout`."""

    width: int = 960
    height: int = 540
    fps: int = 25
    frame_stride: int = 2
    camera: str = "track"
    scale: int = 1
    overlay: bool = True
    title: str = ""
    max_frames: int = 0
    max_steps: int | None = None
```

- `frame_stride: int = 2` 注释：`Store one frame every ``frame_stride`` control steps.  Control runs at 50 Hz, so stride 2 with fps 25 gives real-time playback.`
- `scale: int = 1` 注释见 §3.4。
- `max_frames: int = 0` 注释：`Cap the number of stored frames per episode (0 = unlimited).  Guards against a policy that just stands still on a 20 s episode.`
- `max_steps: int | None = None`：`render_rollout` 里 `max_steps = opts.max_steps or int(env.cfg.episode_length_s / env.cfg.control_dt)`。`scripts/render_checkpoint.py` 用 `--max-seconds / cfg.control_dt` 填它。
- 单位：`width` / `height` 像素，`fps` 帧/秒，`camera` 取 `Go2StairsEnv.CAMERAS = ("side", "track", "chase", "front", "top")`。

### `class EpisodeRender`

```python
@dataclass
class EpisodeRender:
    """Outcome of one rendered episode."""

    episode: int
    steps: int
    frames: int
    distance: float = 0.0
    height_gain: float = 0.0
    forward_speed: float = 0.0
    final_x: float = 0.0
    success: bool = False
    fell: bool = False
    timeout: bool = False
```

- 前三个字段**无默认值**。
- `@property def status(self) -> str`：`"登顶成功"`（`success`）→ `"摔倒"`（`fell`）→ `"超时"`（`timeout`）→ `"未结束"`，按此优先级。
- `def as_dict(self) -> dict[str, Any]`：`d = dict(self.__dict__)`，再塞 `d["status"] = self.status`，返回。注意是**浅拷贝整份 `__dict__`**，所以 JSON 化时 `status` 一定会出现。
- `distance` / `height_gain` / `forward_speed` / `final_x` 在 `render_rollout` 里都 `round(..., 3)`（米、米、m/s、米）。

### `class RenderResult`

```python
@dataclass
class RenderResult:
    """All frames of a render job plus per-episode statistics."""

    frames: list[np.ndarray] = field(default_factory=list)
    episodes: list[EpisodeRender] = field(default_factory=list)
    episode_starts: list[int] = field(default_factory=list)
```

- `episode_starts` 注释：`Frame index at which each episode starts, so a multi-episode GIF can be labelled correctly during overlay.`
- `@property successes` → `sum(1 for e in self.episodes if e.success)`。
- `@property success_rate` → `self.successes / len(self.episodes) if self.episodes else 0.0`。
- `def summary(self) -> dict[str, Any]` → 键 `episodes`（`[e.as_dict() for e in self.episodes]`）、`total_frames`（`len(self.frames)`）、`successes`、`success_rate`（`round(..., 4)`）、`frames_per_episode`（`[e.frames for e in self.episodes]`）。
- `def episode_of_frame(self, index: int) -> int`：docstring `Which rendered episode a global frame index belongs to (0-based).` 实现是遍历 `episode_starts`、保留最后一个 `<= index` 的下标，遇到更大的就 `break`。**索引越界不会报错**，返回最后一个回合号。

### `render_rollout`

```python
def render_rollout(
    env: Go2StairsEnv,
    policy: Predictor,
    *,
    episodes: int = 1,
    seed: int = 0,
    opts: RenderOptions | None = None,
    deterministic: bool = True,
    stair: StairSpec | None = None,
) -> RenderResult:
```

- docstring：`Roll the policy out and collect frames + per-episode statistics.` + §3.1 的「不关环境」说明。
- 语义：
  1. `opts = opts or RenderOptions()`；`stair = stair or env.stair`；`max_steps = opts.max_steps or int(env.cfg.episode_length_s / env.cfg.control_dt)`。
  2. 每回合 `obs, _ = env.reset(seed=seed + ep)`，记 `x0 = float(env.base_pos[0])`、`z0 = float(env.base_pos[2])`，`result.episode_starts.append(len(result.frames))`。
  3. 主循环 `while not (terminated or truncated) and step < max_steps`：`policy.predict(obs, deterministic=deterministic)` → `env.step(action)` → `step += 1`；`if step % opts.frame_stride == 0`：若 `opts.max_frames and len(result.frames) - result.episode_starts[-1] >= opts.max_frames` 就 **break**，否则 `result.frames.append(env.render(opts.camera, width=opts.width, height=opts.height).copy())`。
  4. `dt = step * env.cfg.control_dt`，追加 `EpisodeRender(...)`，其中 `success=bool(info.get("is_success", False))`、`fell=bool(terminated) and not bool(info.get("is_success", False))`、`timeout=bool(truncated) and not bool(info.get("is_success", False))`。
- 参数：`episodes` 回放回合数；`seed` 起点种子，逐回合用 `seed + ep`；`opts` 缺省 `RenderOptions()`；`deterministic` 直接透传给 `policy.predict`；`stair` 只影响调用方的 HUD，本函数本身不使用它。
- 返回：`RenderResult`。
- 异常：无显式抛出。`env.render` 会抛 `ValueError(f"未知相机 {camera!r}, 可选 {self.CAMERAS}")`。

### `_FONT_CANDIDATES` 与 `_load_font`

```python
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _load_font(size: int):
```

docstring：`Best available TrueType font at ``size`` px, else Pillow's bitmap font.`
`from PIL import ImageFont` 在函数体内；逐个候选路径 `Path(path).exists()` 后 `ImageFont.truetype(path, size)`，失败 `continue`；再试 matplotlib 的 DejaVu；最后 `ImageFont.load_default()`。异常：只可能来自 `from PIL import ImageFont`（缺 Pillow 时 `ImportError`）。

### `hud_lines`

```python
def hud_lines(
    *,
    title: str,
    stair: StairSpec,
    episode: EpisodeRender | None,
    index: int,
    total: int,
    base_rpy: Sequence[float] | None = None,
    terrain_height: float | None = None,
    compact: bool = False,
) -> list[str]:
```

- docstring：`Build the text lines drawn on a frame (kept separate so it is testable).` + "``compact=True`` drops the least important lines, for small GIFs where five lines of text would eat a quarter of the picture."
- 全部参数只允许关键字。追加顺序：
  1. `title`（真值时）；
  2. 恒有：`f"台阶 {stair.step_height * 100:.1f}cm × {stair.n_up} 级  踏面 {stair.tread * 100:.0f}cm  顶端 {stair.crest_height:.2f}m"`；
  3. `episode is not None` 时：`f"回合 {episode.episode + 1}  {episode.status}  前进 {episode.distance:+.2f}m  速度 {episode.forward_speed:.2f}m/s"`；
  4. `base_rpy is not None and not compact` 时：`f"姿态 roll {np.degrees(base_rpy[0]):+.1f}°  pitch {np.degrees(base_rpy[1]):+.1f}°"`；
  5. `not compact` 时：`f"帧 {index + 1}/{total}"`。
- 返回：`list[str]`。
- **注意 `terrain_height` 参数目前没有被使用**（签名里有，函数体里没有读它）——传了也不会有额外一行。

### `overlay_hud`

```python
def overlay_hud(frame: np.ndarray, lines: Sequence[str], *,
                font_size: int | None = None) -> np.ndarray:
```

- docstring：`Draw ``lines`` into the top-left corner of an RGB frame.`
- 语义：`lines` 为空时**原样返回 `frame`**（不做拷贝）；否则 `Image.fromarray(np.ascontiguousarray(frame))` → `ImageDraw.Draw(img, "RGBA")` → `size = font_size or max(12, frame.shape[0] // 26)` → `_load_font(size)` → `pad = max(4, size // 3)` → `line_h = size + max(2, size // 4)`；用 `draw.textbbox((0, 0), text, font=font)` 逐行求最大宽度（异常时回退 `(0, 0, len(text) * size // 2, size)`）；底板 `plate = (pad, pad, pad + width + 2 * pad, pad + line_h * len(lines) + pad)`，`draw.rectangle(plate, fill=(0, 0, 0, 140))`；然后从 `y = pad + pad // 2` 起每行 `draw.text((pad + pad, y), text, font=font, fill=(255, 255, 255, 255))`，`y += line_h`。
- 返回：`np.asarray(img)`（RGB，`uint8`）。异常：缺 Pillow 时 `ImportError`。

### `save_frames`

```python
def save_frames(directory: str | Path, frames: Sequence[np.ndarray],
                *, prefix: str = "f", downscale_factor: int = 1) -> list[Path]:
```

- docstring：`Write every frame as a dependency-free PNG (see :mod:`robotdog.tools.png`).`
- 语义：`directory.mkdir(parents=True, exist_ok=True)`；逐帧 `img = downscale(frame, downscale_factor) if downscale_factor > 1 else frame`，写 `directory / f"{prefix}{i:04d}.png"`（`i` 从 0，四位补零），收集返回。
- 返回：`list[Path]`，与 `frames` 等长同序。
- 异常：`write_png` 的 `ValueError(f"需要 (H, W, 3) 的 RGB 数组, 收到 {arr.shape}")` / `ValueError(f"需要 uint8, 收到 {arr.dtype}")`。

### `_global_palette`

```python
def _global_palette(images: Sequence["Any"], sample: int = 16):
```

- docstring 见 §3.3。
- 语义：`len(images) <= sample` 时取全部；否则 `idx = np.linspace(0, len(images) - 1, sample).round().astype(int)` 抽 `sample` 帧。`cols = 4`，`rows = (len(picks) + cols - 1) // cols`；缩略图 `thumb_w = max(1, picks[0].width // 4)`、`thumb_h = max(1, picks[0].height // 4)`；`Image.new("RGB", (thumb_w * cols, thumb_h * rows))` 后逐张 `mosaic.paste(img.resize((thumb_w, thumb_h), Image.BOX), ((i % cols) * thumb_w, (i // cols) * thumb_h))`。
- 返回：`mosaic.quantize(colors=256, method=Image.MEDIANCUT)` 的结果（PIL 调色板图像）。
- 异常：`images` 为空时 `picks[0]` 触发 `IndexError`（调用方保证非空）。

### `save_gif`

```python
def save_gif(path: str | Path, frames: Sequence[np.ndarray], *,
             fps: int = 25, scale: int = 1, loop: int = 0,
             optimize: bool = True, dither: bool = True,
             palette_sample: int = 16) -> Path:
```

- docstring：`Encode ``frames`` (H, W, 3) uint8 into an animated GIF.` + "``scale`` box-filters by an integer factor before encoding -- a 960x540@25fps 20 s GIF is fine, but half-size is a third of the bytes and, for a README, just as readable.  All frames share one palette (see :func:`_global_palette`)."
- 语义：`if not frames: raise ValueError("没有帧可以写入 GIF")`；`path.parent.mkdir(parents=True, exist_ok=True)`；`duration = max(20, int(round(1000.0 / max(1, fps))))`（毫秒，下限 20）；逐帧 `Image.fromarray(np.ascontiguousarray(frame))`，`scale > 1` 时 `img.resize((max(1, img.width // scale), max(1, img.height // scale)), Image.BOX)`，再 `.convert("RGB")`；`palette = _global_palette(rgb, palette_sample)`；`mode = Image.FLOYDSTEINBERG if dither else Image.NONE`；`images = [img.quantize(palette=palette, dither=mode) for img in rgb]`；`images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=loop, optimize=optimize, disposal=2)`。
- 参数：`fps` 播放心率；`scale` 编码前 box filter 的整数因子；`loop=0` 表示无限循环；`optimize` 交给 Pillow 的帧优化；`dither=False` 用 `Image.NONE`（体积更小但渐变有色带）；`palette_sample` 抽多少帧建调色板。
- 返回：`path`（`Path`）。异常：空帧 `ValueError("没有帧可以写入 GIF")`；缺 Pillow `ImportError`。

### `contact_sheet`

```python
def contact_sheet(frames: Sequence[np.ndarray], *, n: int = 12,
                  scale: int = 4) -> np.ndarray:
```

- docstring：`Horizontal strip of ``n`` evenly spaced (downscaled) frames.`
- 语义：空帧 `raise ValueError("没有帧可以拼图")`；`len(frames) <= n` 取全部，否则 `idx = np.linspace(0, len(frames) - 1, n).round().astype(int)` 等间隔抽 `n` 帧；`return hstack([downscale(f, scale) for f in picked])`。
- 返回：横拼后的 `np.ndarray`（`hstack` 默认 `pad=0, background=255`）。异常：`ValueError("没有帧可以拼图")`。

### `copy_into`

```python
def copy_into(src: str | Path, dst: str | Path) -> Path:
```

- docstring：`Small shim used by the CLI to stage exported models into ``output/``.`
- 语义：`dst.parent.mkdir(parents=True, exist_ok=True)`；`shutil.copy2(src, dst)`；返回 `dst`。
- 异常：源不存在时 `FileNotFoundError`。

## 5. 内部结构与数据流

单条渲染链路（`scripts/render_checkpoint.py:render_one`）：

```
env_config_for_model(model_path)                          # robotdog.training
load_policy(model_path, vn_path, device=…, cfg=cfg)       # scripts.eval
  → SingleEnvPredict(model, venv)                         # Predictor
Go2StairsEnv(stair=stair, cfg=cfg, seed=seed)
render_rollout(env, predictor, episodes=…, seed=…, opts=…, stair=…, deterministic=…)
  → env.reset(seed=seed + ep)
  → policy.predict(obs, deterministic=…) → env.step(action)
  → env.render(opts.camera, width=opts.width, height=opts.height).copy()
  → RenderResult(frames=[…], episodes=[EpisodeRender, …], episode_starts=[…])
labelled_frames(result, title=…, stair=…, args=…, opts=…)
  → render_result.episode_of_frame(i)                     # 每帧属于哪个回合
  → hud_lines(title=…, stair=…, episode=…, index=i, total=…, compact=…)
  → downscale(frame, opts.scale) → overlay_hud(f, lines)  # gif 集：先缩后贴
  → overlay_hud(frame, lines)                             # strip 集：全分辨率
  → (gif_frames, strip_frames)
save_gif(OUTPUT_GIFS_DIR / f"{name}.gif", gif_frames, fps=…, scale=1, dither=…)
  → _global_palette(rgb, palette_sample)                  # 一套全片调色板
  → img.quantize(palette=…, dither=…) → images[0].save(…, disposal=2)
contact_sheet(strip_frames, n=12, scale=strip_scale) → write_png(OUTPUT_STILLS_DIR / …)
save_frames(OUTPUT_FRAMES_DIR / name, gif_frames)         # 仅 --save-frames
env.close()                                               # 由调用方负责
```

对比模式（`scripts/render_checkpoint.py:render_compare`）：

```
render_rollout(env, SingleEnvPredict(...), …)             # 左：训练后的策略
render_rollout(env, PriorPolicy(...), …)                  # 右：PriorPolicy 的 predict 恒返回 np.zeros(action_dim)
hstack([left_frame, right_frame], pad=4, background=30)
save_gif(OUTPUT_GIFS_DIR / f"{name}_vs_prior.gif", …)
```

`PriorPolicy.predict` 的实现是 `return np.zeros(self.action_dim, dtype=np.float32), None`——动作恒 0 意味着残差为零，只剩开环步态先验在控制，这正是训练的对照组。

## 6. 关键常量与默认值

| 名称 | 值 | 说明 |
| --- | --- | --- |
| `RenderOptions.width` / `height` | `960` / `540` | 像素 |
| `RenderOptions.fps` | `25` | 与 `frame_stride=2`、控制 50 Hz 一起给实时播放 |
| `RenderOptions.frame_stride` | `2` | 每 2 个控制步存一帧 |
| `RenderOptions.camera` | `"track"` | 可选 `Go2StairsEnv.CAMERAS = ("side", "track", "chase", "front", "top")` |
| `RenderOptions.scale` | `1` | CLI 默认把它设成 2（半尺寸） |
| `RenderOptions.overlay` | `True` | |
| `RenderOptions.title` | `""` | 空串时不画标题行 |
| `RenderOptions.max_frames` | `0` | 0 = 不限帧 |
| `RenderOptions.max_steps` | `None` | 缺省 = `episode_length_s / control_dt` |
| `hud_lines` 中文字号 | `max(12, frame.shape[0] // 26)` | 540 高 → 20 px |
| `overlay_hud` 内边距 | `pad = max(4, size // 3)` | |
| `overlay_hud` 行高 | `line_h = size + max(2, size // 4)` | |
| `overlay_hud` 底板色 | `(0, 0, 0, 140)` | RGBA 半透明 |
| `overlay_hud` 文字色 | `(255, 255, 255, 255)` | |
| `overlay_hud` 底板宽度回退 | `(0, 0, len(text) * size // 2, size)` | 位图字体无 `textbbox` 时 |
| `_FONT_CANDIDATES` | `msyh.ttc` / `msyhbd.ttc` / `simhei.ttf` / `DejaVuSans.ttf` | 另有 matplotlib 的 DejaVu 兜底 |
| `save_frames.prefix` | `"f"` | 文件名 `f0000.png` |
| `save_gif.fps` / `scale` / `loop` | `25` / `1` / `0` | `loop=0` = 无限循环 |
| `save_gif.optimize` / `dither` | `True` / `True` | |
| `save_gif.palette_sample` | `16` | 建调色板抽的帧数 |
| `save_gif` 帧时长 | `max(20, round(1000 / fps))` 毫秒 | 25 fps → 40 ms |
| `save_gif` GIF 处置方式 | `disposal=2` | 每帧前恢复背景 |
| `_global_palette` 列数 | `4` | mosaic 固定 4 列 |
| `_global_palette` 缩略图比例 | 宽高各 `// 4` | |
| `_global_palette` 颜色数 | `256`，`Image.MEDIANCUT` | |
| `contact_sheet.n` / `scale` | `12` / `4` | CLI 传 `scale=--strip-scale`（默认 2） |
| `copy_into` | `shutil.copy2` | 保留元数据 |
| 体积对照 | 900 帧原始 ≈ **500 MB** → 同一动画 GIF **2–6 MB** | 模块 docstring |
| `PriorPolicy` 动作 | `np.zeros(action_dim, dtype=np.float32)` | 纯先验对照组 |

## 7. 与其它模块的关系

```
robotdog/tools/png.py       (downscale, hstack, write_png)
robotdog/tools/terrain.py   (StairSpec)
robotdog/env.py             (Go2StairsEnv — 提供 render / base_pos / cfg)
        ↓
robotdog/render.py           ← 本模块
        ↑
scripts/render_checkpoint.py (唯一的驱动方)
scripts/eval.py              (间接：load_policy / SingleEnvPredict)
robotdog/paths.py            (OUTPUT_GIFS_DIR / OUTPUT_STILLS_DIR / OUTPUT_FRAMES_DIR / OUTPUT_REPORTS_DIR)
```

- **为什么在 `robotdog/` 顶层而不在 `robotdog/tools/`**：`docs/README.md` 的分层规则要求工具层**不 import `robotdog.env`**，而本模块必须驱动 `Go2StairsEnv`（读 `base_pos`、`cfg.control_dt`，调 `render`）。放进 `tools/` 会立刻让 `tools ↔ env` 成环。
- **为什么依赖 `robotdog/tools/png.py`**：`save_frames` 的逐帧 PNG 与 `contact_sheet` 的 `hstack` / `downscale` 都在那里，且那份 PNG 写入器不依赖 Pillow。方向是 渲染层 → 工具层，符合单向依赖。
- **为什么不依赖 `robotdog/evaluate.py`**：渲染只需要一个 `Predictor`，不需要评估矩阵。让 `evaluate.py` 依赖 `render.py`（或反之）会把「出报表」和「出动画」耦合成同一个可失败单元；现在两张产物各自独立。
- **被 `scripts/render_checkpoint.py` 消费**：`RenderOptions` / `render_rollout` / `hud_lines` / `overlay_hud` / `save_gif` / `save_frames` / `contact_sheet` 全部直接用；`copy_into` 在导出章节用来把模型搬进 `output/exports/`。
- **不依赖 `stable_baselines3`**：模型是调用方加载好之后以 `Predictor` 传进来的。所以本模块可以被 `PriorPolicy` 这种非 SB3 对象驱动——这正是「训练前后对比 GIF」能实现的原因。

## 8. 实测与经验教训

- 渲染相机必须强制打开 geomgroup，否则楼梯/触觉几何默认隐藏，GIF 里只剩悬空的狗：`experience.md` §1.2。
- 「策略 ≡ 纯先验」这个结论，肉眼看 GIF 与看数字是一致的——策略输出 `mean|a| = 0.0109`、`|a| < 0.05` 占 99.9% 时，动画就是开环步态在走：`experience.md` §4.1、`train.md` §3.3。
- 参考 checkpoint 的能力带（`train_flat` 100%、`train_h050` 100%、`train_h120` 0%、`train_h190` 0%）与逐高度 `max base_x` 曲线，是给 GIF 选地形时的对照基准：`train.md` §3.1、`train.md` §3.4。
- 11 cm 是墙不是奖励问题，也不是「抬不起来」——先验的几何能力够而策略没用它：`experience.md` §6.3、`experience.md` §6.4、`train.md` §3.6。渲染对比时，先验一侧在 11 cm 上的形态最能说明这一点。
- 摔倒率与成功率必须分开读，`terminated=True` 也包含登顶：`robotdog/evaluate.py` 的 `evaluate_terrain` 注释、`experience.md` §5.1（探针报错回合）。
- 从零训练的 run 在 5 cm 上也可能 0% 成功 / 100% 摔倒（6 回合/档、确定性）：`train.md` §3.4。GIF 只渲染 2 回合（`--episodes` 默认 2），**不能**当成功率证据。
- 参考的 `mean|a| = 0.42`，从零学不动的 run 只有 0.02~0.10：`train.md` §3.4。这个量在动画里表现为「身体有没有真的在动」。

## 9. 已知限制与坑

1. **`render_rollout` 不会关闭环境。** 忘了自己 `env.close()` 会留下 GL 上下文与渲染器；重复建环境会累积显存/句柄。

2. **一帧就是一整张 `uint8` 图，全存在内存里。** 960×540 的 RGB 帧约 1.5 MB；20 s × 50 Hz × `frame_stride=2` = 500 帧 ≈ 760 MB，`episodes > 1` 时线性增长，且 `disposal=2` 不影响抓帧期。长回合 + 多回合时用 `RenderOptions.max_frames` 或 `scale` 压住。

3. **`--episodes` 默认 2，不是统计证据。** 2 回合的 `RenderResult.success_rate` 只能是 0 / 0.5 / 1.0。要数字请用 `robotdog/evaluate.py`，并且 ≥40 回合（`experience.md` §5.7）。

4. **`save_gif` 空帧抛 `ValueError("没有帧可以写入 GIF")`**，`contact_sheet` 空帧抛 `ValueError("没有帧可以拼图")`。`scripts/render_checkpoint.py` 在调用前用 `if not gif_frames` 打印 `  [!] {name}: 一个帧都没有生成, 跳过`。

5. **GIF 路径需要 Pillow。** `from PIL import Image` 在 `save_gif` / `_global_palette` / `overlay_hud` / `_load_font` 内部；缺 Pillow 时 `ImportError`。逐帧 PNG（`save_frames`）不需要 Pillow。

6. **位图字体下 `overlay_hud` 的底板宽度是估算的。** `draw.textbbox` 失败时回退 `(0, 0, len(text) * size // 2, size)`，中文与 ASCII 混排时底板可能过宽或贴边。

7. **`hud_lines` 的 `terrain_height` 参数目前未使用。** 传了也不会多一行；不要以为它没生效是因为值不对。

8. **`episode_of_frame` 不校验越界。** 传 `index >= len(frames)` 返回最后一个回合号而不是报错。

9. **调色板抽帧可能漏掉稀有色。** `palette_sample=16` 时，只出现在 1~2 帧里的颜色（例如某一帧的红色警示）会被量化掉。需要保色时调大 `palette_sample`。

10. **`dither=False` 用 `Image.NONE`**，GIF 更小但天空渐变会出现色带；`--no-dither` 就是这条。

11. **`copy_into` 不做存在性检查**，源路径打错直接 `FileNotFoundError`；也不判断 `src`/`dst` 是否是同一个文件。

12. **`scale` 的两个含义容易搞混**：`RenderOptions.scale`（在 `env.render` 之后、贴 HUD 之前缩小）与 `save_gif(scale=...)`（对已贴好 HUD 的帧再缩小）是**两个独立的缩放点**。`scripts/render_checkpoint.py` 固定 `save_gif(..., scale=1)`，缩小交给前者。

## 10. 示例用法

最小可跑：直接驱动 `robotdog.render`（不经过 CLI）。

```python
import numpy as np
from robotdog.env import Go2StairsEnv
from robotdog.tools.terrain import demo_stairs_spec
from robotdog.training import env_config_for_model, load_ppo, make_vec_env
from robotdog.tools.terrain import flat_spec
from robotdog.render import (RenderOptions, render_rollout, save_gif,
                            hud_lines, overlay_hud, downscale, contact_sheet)

zip_path = r"runs\course22\checkpoints\medium_final.zip"
cfg = env_config_for_model(zip_path)

# 1. 模型需要一个向量化环境才能构造
venv = make_vec_env(flat_spec(), n_envs=1, cfg=cfg, seed=0)
model = load_ppo(zip_path, venv, device="cpu")

# 2. 渲染用的地形与选项
stair = demo_stairs_spec(step_height=0.10, tread=0.50, n_up=8, top_landing=1.20)
opts = RenderOptions(width=960, height=540, fps=25, frame_stride=2,
                     camera="track", scale=2, title="10 cm x 8")

# 3. 回放（注意 env 由我们自己关）
env = Go2StairsEnv(stair=stair, cfg=cfg, seed=0)
try:
    result = render_rollout(env, model, episodes=2, seed=0, opts=opts, stair=stair)
finally:
    env.close()

# 4. 先缩小、再贴 HUD，然后编码
gif_frames = []
for i, frame in enumerate(result.frames):
    ep = result.episode_of_frame(i)
    episode = result.episodes[ep] if ep < len(result.episodes) else None
    lines = hud_lines(title=opts.title, stair=stair, episode=episode,
                      index=i, total=len(result.frames))
    gif_frames.append(overlay_hud(downscale(frame, opts.scale), lines))

save_gif("output/gifs/demo_10cm.gif", gif_frames, fps=opts.fps, scale=1)
print(result.summary())
```

命令行（真实入口）：

```bat
.venv\Scripts\python.exe scripts\render_checkpoint.py --run course22 --stage medium ^
  --episodes 2 --camera track --width 960 --height 540 --scale 2 ^
  --no-strip --no-report
```

「训练到底有没有用」的并排对比（左＝策略，右＝动作恒 0 的纯开环先验）：

```bat
.venv\Scripts\python.exe scripts\render_checkpoint.py --run course22 --stage medium ^
  --episodes 1 --compare-prior --no-strip --no-report
```

产物：`output\gifs\course22_medium_..._vs_prior.gif`。
