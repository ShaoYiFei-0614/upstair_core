# `robotdog/tools/png.py` —— 无依赖 PNG 读写与图像拼接

## 1. 一句话定位

**工具层**（`robotdog/tools/`）里的图像 I/O 原语：用 `zlib` + `struct` 手写 PNG 容器，提供「把 `mujoco.Renderer` 给出的 `(H, W, 3)` uint8 数组写成 PNG」「读回来」「整数倍降采样」「横向拼接」四个操作。它解决的问题是：**渲染/验证脚本必须能在不装 Pillow 的环境里把一帧图像落盘**，否则「看一眼策略到底做了什么」这件事就被一个图像库的安装状态绑住了。

## 2. 职责边界

**做**：

* `write_png`：把 `(H, W, 3)` 的 uint8 RGB 数组写成 8-bit truecolour PNG，自动建父目录，返回最终路径。
* `read_png`：读回 PNG，解 `IHDR` / `IDAT` / `IEND`，实现全部 5 种行滤波（None / Sub / Up / Average / Paeth）。
* `downscale`：整数倍盒式（box-filter）降采样，用于把大帧压到 GIF/HUD 友好的尺寸。
* `hstack`：把多张图左右拼成一张对比图（评估脚本的 contact sheet 基础）。

**不做**：

* **不处理调色板（colour type 3）、灰度（0）、带 alpha（4/6）、16-bit（depth 16）、隔行（interlace）**：`write_png` 只写 colour type 2 / depth 8，`read_png` 遇到别的组合直接 `raise ValueError`。
* **不写/读非 PNG 格式**（JPEG、BMP、PPM 全不支持）。
* **不做有损压缩、不做缩放插值**（`downscale` 是均值盒式，没有双线性/双三次）。
* **不做 GIF 编码**（那是 `robotdog/render.py` 用 Pillow 做的），**不做 ffmpeg 调用**。
* **不做色彩空间转换**（sRGB/线性、gamma 一律不管）、不做文字/HUD 叠加（那是 `robotdog/render.py:overlay_hud` 用 Pillow 做的）。
* **不 import `robotdog.env`**，不知道 PPO / 课程 / 训练循环。
* 本模块**没有 `__all__`**（与同层其它模块不同），四个函数都是公开的。

## 3. 关键设计决策与理由

### 3.1 手写 PNG 容器，而不是依赖 Pillow / ffmpeg

源码 docstring 第 3-5 行给出理由：`The sandbox has no Pillow and installing one just to dump a frame is wasteful, so this module hand-rolls the PNG container (zlib + struct only).  It is shared by the verification scripts, the evaluation CLI and any future video export.`

拆开看是三条独立理由：

1. **依赖面**：`zlib` 与 `struct` 是 stdlib，PNG 的「容器层」本身非常简单 —— 8 字节 magic + 三个 chunk（`IHDR`/`IDAT`/`IEND`），每个 chunk 是「长度 + tag + payload + CRC32」。全部代码不到 20 行，而引入一个图像库要多一条安装/版本链。本模块被验证脚本、评估 CLI 共用，**这些脚本存在的意义就是「能在最小环境里跑起来核对结论」**，所以它们的依赖必须尽可能硬。
2. **格式恰好匹配**：`write_png` 只处理 truecolour 8-bit RGB，而 docstring 明说 `which is exactly what mujoco.Renderer returns`。渲染器给的就是 `(H, W, 3)` uint8，不需要任何转换、不需要 alpha、不需要调色板 —— 也就是说不需要库的「格式协商」能力。
3. **性能与文件大小**：帧转储走 `zlib.compress(raw, compress_level)`，压缩级别由调用方控制。**被否决的方案**：写成 PPM/BMP（更简单）—— 否决，因为它们没有压缩、体积太大（一张 640×480 的 PPM 约 0.9 MB），且无法被浏览器/文档直接嵌入。

**被否决的方案**：依赖 Pillow。真实的理由是**可用性分层**，不是「绝对不能用 Pillow」：当前仓库里 `robotdog/render.py:223, 285, 337, 368` 确实用了 Pillow（`from PIL import Image, ImageDraw, ImageFont`，docstring 说它 `already present as a TensorBoard dependency`），`README.md` 也把 `pillow 10.4.0` 列为依赖。区别在于**依赖方向**：`render.py` 是「可选的、给人看的产物层」，缺 Pillow 时整个渲染功能不工作是可以接受的；`png.py` 是「底层落盘能力」，评估与验证脚本要在 Pillow 缺席时照样能 dump 帧。所以这里是**能力分层**，而不是「仓库里没有 Pillow」。**另一个被否决的方案**：ffmpeg 子进程调用 —— 否决，因为本仓的运行环境禁止命名管道式的子进程输出捕获（见 `experience.md` §5.8），而且为一个 PNG 编解码拉起进程的代价远高于 20 行 zlib 代码。

> ⚠️ **文档与现实的偏差（如实记录）**：`png.py` 第 3 行的措辞是硬判断句（`The sandbox has no Pillow`），它描述的是本模块诞生的场景。当前仓库里 Pillow 是可用的（`robotdog/render.py` 直接 import，`README.md` 列为依赖）。因此这句话**不能**当作「装了 Pillow 就会坏」来读；它应当读作「本模块刻意不依赖 Pillow」。这一点在 §9 再次列出。

### 3.2 写侧只发最简单的形式，读侧支持全部 5 种滤波

`write_png` 的每一行都前置一个 `0x00` 滤波字节：

```python
raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
```

而 `read_png` 实现了 `0/1/2/3/4` 全部五种滤波，遇到别的值抛 `未知的 PNG 行滤波类型`。

理由不对称是**刻意**的：

* **写侧**：滤波是为了让压缩器看到更小的残差，但代价是实现 5 种预测器 + 逐行选优。本项目写出的图是**渲染帧**（大面积渐变背景 + 少量几何），在 `compress_level=6` 下 filter 0 的压缩率已经够用；写侧简单 = 正确性容易保证。
* **读侧**：`read_png` 的用途是「读回**自己写的**图」以及「读回别人写的图」（对比基线的拼图可能来自 Pillow 或别的工具）。Pillow 默认会选最优滤波，所以读侧必须完整。`read_png` 的 docstring 只说 `Read back a PNG written by :func:`write_png``，实现却更宽 —— 这是「读侧向前兼容」的保守选择。

**被否决的方案**：写侧也实现滤波 —— 否决，因为收益（文件小一点）小于风险（Paeth 预测器容易写错，而错的结果是**图像看起来还行但像素错了**，最容易骗过校验）；读侧不支持 5 种滤波 —— 否决，会让 `read_png` 在某些外部 PNG 上抛 `未知的 PNG 行滤波类型`，而这正是回归脚本要对齐的对象。

### 3.3 自己算 CRC32，不校验读入的 CRC

写侧：

```python
def chunk(tag: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))
```

`zlib.crc32` 在 Python 3 返回无符号值，`& 0xFFFFFFFF` 是把它锁成 32 位（并显式表态「网络字节序、无符号」）。读侧则**完全跳过 CRC 校验**（`pos += 12 + length` 直接跳过 4 字节 CRC）。

理由：读侧的用途是自检与拼图，`zlib.decompress` 已经会拦住绝大多数损坏（它有自己的 Adler-32 校验和），额外的 CRC 校验只是把「文件坏了」提前 10 行抛出，却要多写一个校验循环。**被否决的方案**：读侧校验 CRC —— 否决，收益是更早的报错，代价是更多代码路径，而本项目里 PNG 都是同一进程刚写出的。

### 3.4 数组约定固定为 `(H, W, 3)` uint8，越界用法显式报错

`write_png` 在真正写盘**之前**做两项检查：

```python
if arr.ndim != 3 or arr.shape[2] != 3:
    raise ValueError(f"需要 (H, W, 3) 的 RGB 数组, 收到 {arr.shape}")
if arr.dtype != np.uint8:
    raise ValueError(f"需要 uint8, 收到 {arr.dtype}")
```

理由：**形状错误必须比文件更早暴露**。若不检查，`arr[y].tobytes()` 会把 4 通道图的第一列当成 R 通道写进去，产出一张「斜了」的图 —— 这种错误在缩略图上很难看出来。同理 uint8 检查挡住了最常见的「`float32` 的 0~1 图像直接写盘」变成全黑/全白。

检查用**中文报错并回显实际形状/dtype**，因为这两条错误在调试时最常出现，报错本身要能直接告诉调用方「你给的是什么」。

注意顺序：`np.ascontiguousarray(rgb)` 在检查之前执行 —— 它保证后续 `arr[y].tobytes()` 得到的是**行主序的连续内存**（否则 `tobytes()` 会走一条慢路径或拿到错误的内存视图）。

### 3.5 `downscale` 是整数倍均值，且对退化输入返回原图

```python
if factor <= 1:
    return img
h, w = img.shape[:2]
h2, w2 = h // factor, w // factor
if h2 == 0 or w2 == 0:
    return img
crop = img[: h2 * factor, : w2 * factor]
return crop.reshape(h2, factor, w2, factor, 3).mean(axis=(1, 3)).astype(np.uint8)
```

三处细节都是决策：

1. **整数倍**：`h // factor`，不能整除的部分被 `crop` 丢掉。理由：均值盒式要求每个输出像素对应**恰好** `factor × factor` 个输入像素，否则要么引入权重（近似缩放）要么补齐边缘（改变尺寸语义）。渲染尺寸可被调用方控制，交给调用方选倍数更干净。
2. **`factor <= 1` 与 `h2/w2 == 0` 直接返回原图**，而不是抛异常。理由：调用点经常是「按配置降采样」，配置为 1 或图像比倍数还小时，返回原图是**有意义的行为**（GIF 仍然能生成），抛异常会让整条渲染链因为一个纯审美参数而失败。
3. **用 `reshape + mean(axis=(1, 3))` 而不是循环或切片相加**：一次归约完成，顺便让 numpy 用上 SIMD。`.astype(np.uint8)` 在最后做（中间保持均值精度），避免中间就截断。

**被否决的方案**：`stride_tricks` 视图 + `mean` —— 否决，视图方案在 `factor` 不能整除时行为更绕，且这里的内存开销本来就小（一帧几 MB）。**被否决的方案**：实现双线性缩放 —— 否决，`downscale` 的用途是压 GIF 体积，均值盒式在**降采样**场景下的抗锯齿效果反而优于双线性。

### 3.6 `downscale` 硬编码 3 通道，`hstack` 靠形状推断

`downscale` 里的 `.reshape(h2, factor, w2, factor, 3)` 把 `3` 写死。理由：本模块的数组约定全程是 RGB（见 §3.4），写死能让 reshape 一次到位；代价是灰度/alpha 图不能直接降采样。**被否决的方案**：`img.shape[2]` 动态取通道 —— 否决，会把 `(H, W)` 二维灰度图的 `shape[2]` 变成 `IndexError`，错得更晚。

`hstack` 反过来**不检查**通道数，只按 `im.shape[0]` / `im.shape[1]` 排布，最终 `out` 是 `(h, w, 3)`：拼接时若某张图不是 3 通道，赋值 `out[...] = im` 会由 numpy 自己广播或报错。理由：拼接是「摆位置」，形状一致性交给 numpy 的赋值语义（报错信息里带形状，足够定位）。

`hstack` 的两个参数决策：

* `pad: int = 0`（图之间的间隔像素数）与 `background: int = 255`（填充灰度值，默认白）。理由：对比图需要视觉分隔，而**白色背景**在深色渲染帧之间最容易辨认。
* 用 `np.full((h, w, 3), background, dtype=np.uint8)` 先铺底再贴图，而不是 `np.concatenate`：因为各图**高度可能不同**（`h = max(...)`），需要「顶部对齐 + 底部留白」的语义。docstring 写明 `all must share a height after cropping` —— 即调用方通常先 `downscale`/裁剪到同一高度，`hstack` 只是兜底。

**被否决的方案**：`hstack` 要求高度严格相等并抛异常 —— 否决，渲染脚本里不同相机的帧尺寸常有一两像素差异，硬失败会逼调用方到处写裁剪。

## 4. 公开 API

```python
def write_png(path: str | Path, rgb: np.ndarray, *, compress_level: int = 6) -> Path:
    """Write an ``(H, W, 3)`` uint8 array as a PNG file."""


def read_png(path: str | Path) -> np.ndarray:
    """Read back a PNG written by :func:`write_png` (8-bit truecolour RGB)."""


def downscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Box-filter downscale by an integer ``factor``."""


def hstack(images: list[np.ndarray], *, pad: int = 0,
           background: int = 255) -> np.ndarray:
    """Lay images out left-to-right (all must share a height after cropping)."""
```

### `write_png(path, rgb, *, compress_level=6) -> Path`

| 参数 | 类型 | 语义 |
| --- | --- | --- |
| `path` | `str \| Path`，位置参数 | 目标文件路径；**父目录会被自动创建**（`path.parent.mkdir(parents=True, exist_ok=True)`） |
| `rgb` | `np.ndarray` | 图像数据，必须是 `ndim == 3` 且 `shape[2] == 3`，`dtype` 必须是 `np.uint8`；内部先 `np.ascontiguousarray` |
| `compress_level` | `int`，**keyword-only**（`*` 之后），默认 `6` | 直接传给 `zlib.compress`（0~9） |

* **返回**：`Path`（由 `path` 归一化而来，即 `Path(path)`）。调用方可以直接 `write_png(...).parent`。
* **异常**：`ValueError(f"需要 (H, W, 3) 的 RGB 数组, 收到 {arr.shape}")`；`ValueError(f"需要 uint8, 收到 {arr.dtype}")`；`zlib.error`（`compress_level` 非法时由 `zlib.compress` 抛出）；`OSError`/`PermissionError`（写盘失败）。
* **写出的字节结构**：magic `\x89PNG\r\n\x1a\n` → `IHDR`（`struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)`：宽、高、bit depth 8、colour type 2（truecolour）、compression 0、filter 0、interlace 0）→ `IDAT`（`zlib.compress(raw, compress_level)`，`raw` 每行前置 `b"\x00"`）→ `IEND`（空 payload）。

### `read_png(path) -> np.ndarray`

| 参数 | 类型 | 语义 |
| --- | --- | --- |
| `path` | `str \| Path`，位置参数 | 要读的 PNG 文件 |

* **返回**：`np.ndarray`，形状 `(h, w, 3)`、`dtype = np.uint8`。
* **异常**：`ValueError("不是 PNG 文件")`（magic 不匹配）；`ValueError(f"仅支持 8-bit RGB, 收到 depth={depth} colour={colour}")`；`ValueError("缺少 IHDR")`；`ValueError(f"未知的 PNG 行滤波类型 {filt}")`；`FileNotFoundError`（`Path(path).read_bytes()`）；`zlib.error`（IDAT 损坏）。
* **解析细节**：从偏移 `pos = 8` 起逐 chunk（`(length,) = struct.unpack(">I", data[pos:pos+4])`、`tag = data[pos+4:pos+8]`、`payload = data[pos+8:pos+8+length]`，然后 `pos += 12 + length`）；`IHDR` 用 `struct.unpack(">IIBB", payload[:10])` 取 `w, h, depth, colour`；`IDAT` 累积到 `bytearray`；`IEND` 处 `break`。**CRC 不校验**。
* **滤波实现**：`0` 直通；`1` Sub（`line[i] = (int(line[i]) + int(line[i-3])) & 0xFF`）；`2` Up（`line = (line.astype(np.int32) + prev.astype(np.int32)).astype(np.uint8)`，整行向量化）；`3` Average（`(int(line[i]) + ((left + int(prev[i])) >> 1)) & 0xFF`，`i < 3` 时 `left = 0`）；`4` Paeth（`p = a + b - c`；`pa, pb, pc = abs(p-a), abs(p-b), abs(p-c)`；`pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)`，`i < 3` 时 `a = c = 0`）。每行处理后 `out[y] = line.reshape(w, 3)` 并 `prev = line`。

### `downscale(img, factor) -> np.ndarray`

| 参数 | 类型 | 语义 |
| --- | --- | --- |
| `img` | `np.ndarray` | `(H, W, 3)`（通道数硬编码为 3） |
| `factor` | `int` | 降采样倍数；`<= 1` 时原样返回 |

* **返回**：`(H // factor, W // factor, 3)` 的 uint8 数组，或 `factor <= 1` / 结果任一维为 0 时的**原数组本身**（注意：是同一个对象，不是副本）。
* **异常**：无显式 `raise`；`img` 不是 3 通道时由 `reshape` 抛 `ValueError`。

### `hstack(images, *, pad=0, background=255) -> np.ndarray`

| 参数 | 类型 | 语义 |
| --- | --- | --- |
| `images` | `list[np.ndarray]`，位置参数 | 要拼接的图，按列表顺序**从左到右** |
| `pad` | `int`，keyword-only，默认 `0` | 相邻图之间的间隔像素数（`background` 填充） |
| `background` | `int`，keyword-only，默认 `255` | 画布填充灰度值（默认白） |

* **返回**：`(max(H_i), sum(W_i) + pad * (len(images) - 1), 3)` 的 uint8 数组。每张图**顶部对齐**贴入：`out[: im.shape[0], x:x + im.shape[1]] = im`，`x += im.shape[1] + pad`。
* **异常**：`ValueError("没有图像")`（空列表）。
* **注意**：返回画布的通道数是写死的 `3`；若 `im` 不是 3 通道，赋值处由 numpy 报错。

本模块**没有 `__all__`**；`from robotdog.tools.png import write_png, read_png, downscale, hstack` 是标准用法。

## 5. 内部结构与数据流

写出一帧（主链路）：

```
mujoco.Renderer.render()
  → (H, W, 3) uint8 ndarray
  → write_png(path, rgb, compress_level=6)
      → path = Path(path)
      → path.parent.mkdir(parents=True, exist_ok=True)      # 自动建父目录
      → arr = np.ascontiguousarray(rgb)                     # 行主序连续内存
      → 形状/dtype 校验（失败 → ValueError）
      → raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
      → chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
      → chunk(b"IDAT", zlib.compress(raw, compress_level))
          → chunk() → struct.pack(">I", len) + tag + payload + struct.pack(">I", crc32)
      → chunk(b"IEND", b"")
      → path.write_bytes(b"\x89PNG\r\n\x1a\n" + ...)
      → return path
```

读回并做成对比图（评估脚本链）：

```
read_png(a) → np.ndarray (h, w, 3)
read_png(b) → np.ndarray (h, w, 3)
  → downscale(a, 2)              # (h//2, w//2, 3)；factor<=1 时原样返回
  → downscale(b, 2)
  → hstack([a2, b2], pad=8)      # 顶部对齐 + 白色间隔
  → write_png(out_path, sheet)   # 回到写链路
```

调用点（全仓）：

```
robotdog/render.py:31          from robotdog.tools.png import downscale, hstack, write_png
robotdog/render.py:324, 325, 408    ← 帧转储 / contact sheet / 单帧保存
scripts/render_checkpoint.py:54     from robotdog.tools.png import downscale, write_png
scripts/render_checkpoint.py:172, 230, 278, 319
scripts/eval.py:216                 from robotdog.tools.png import downscale, hstack, write_png
scripts/eval.py:237, 254, 256
```

## 6. 关键常量与默认值

| 项 | 真实值 | 位置 |
| --- | --- | --- |
| PNG magic | `b"\x89PNG\r\n\x1a\n"`（8 字节） | 写侧拼接、读侧 `startswith` |
| `write_png` 的 `compress_level` 默认 | `6` | 签名 keyword-only 参数 |
| `IHDR` 打包格式 | `">IIBBBBB"`（大端 2×u32 + 5×u8） | `struct.pack` |
| `IHDR` 固定字段 | bit depth `8`、colour type `2`、compression `0`、filter `0`、interlace `0` | `write_png` |
| 每行滤波字节（写侧） | `b"\x00"`（filter type 0，None） | `write_png` |
| CRC 掩码 | `0xFFFFFFFF` | `chunk()` |
| chunk 头部/尾部字节数 | 4（length）+ 4（tag）+ 4（CRC）= 12，故 `pos += 12 + length` | `read_png` |
| 读侧支持的滤波类型 | `0, 1, 2, 3, 4`（None/Sub/Up/Average/Paeth） | `read_png` |
| Sub / Average / Paeth 的 bpp 偏移 | `3` 字节（`line[i - 3]`，`i >= 3` 判定） | `read_png` |
| `read_png` 请求的 bit depth / colour type | `8` / `2` | `read_png` 校验 |
| `downscale` 的通道数 | 硬编码 `3` | `crop.reshape(h2, factor, w2, factor, 3)` |
| `downscale` 的 `factor` 短路阈值 | `<= 1` 返回原图 | `downscale` |
| `hstack` 的 `pad` 默认 | `0` | 签名 |
| `hstack` 的 `background` 默认 | `255` | 签名 |
| `hstack` 的输出通道数 | `3` | `np.full((h, w, 3), ...)` |
| 模块级 `__all__` | **不存在** | —— |

## 7. 与其它模块的关系

**上游**：stdlib `struct`、`zlib`、`pathlib.Path` + 第三方 `numpy`。**不依赖 mujoco**（虽然数组约定为 `mujoco.Renderer` 的输出格式）、**不依赖 Pillow**、**不依赖任何 robotdog 模块**。

**下游**：

| 位置 | 使用的符号 | 依赖方向为什么是这样 |
| --- | --- | --- |
| `robotdog/render.py:31` | `downscale, hstack, write_png` | 渲染层（`render.py` 在顶层，因为它驱动 `Go2StairsEnv`）需要落盘帧与拼图，于是向下依赖工具层；它自己用 Pillow 处理 GIF，与本模块**分工不重叠** |
| `scripts/render_checkpoint.py:54` | `downscale, write_png` | 脚本层用最小依赖落盘渲染结果 |
| `scripts/eval.py:216` | `downscale, hstack, write_png` | 评估脚本生成 contact sheet |

**为什么方向是「渲染层 → png 工具层」而不是反过来**：本模块只认识「数组 ↔ 文件」，不需要知道数组从哪来（MuJoCo 渲染、读回的旧图、甚至人工合成）。反过来若 `png.py` import `render.py`（为了知道渲染尺寸或调色板），就会把「图像容器」与「机器人渲染」耦合起来，任何渲染改动都会波及验证脚本 —— 而验证脚本的独立性正是它有价值的原因。

**与 `render.py` 的分工边界**（两个模块都碰图像，必须说清）：`png.py` 负责 **PNG 容器 + 像素数组运算**（无 Pillow）；`render.py` 负责 **GIF 编码 + HUD 文字叠加**（用 Pillow）。同一个渲染流程里两者接力：帧数组 → `write_png`（本模块）落盘/拼图 → `save_gif`（Pillow）编动画。

## 8. 实测与经验教训

* 「地形在视觉上不可见 / 狗在爬空气但 HUD 仍在报高度」这类只有**看渲染图**才能发现的物理坑，以及为什么必须把帧 dump 出来，见 `experience.md` §1.2、§1.1。
* 小样本评估得出的百分比会有 ±15 pt 量级的错觉，视觉对比图（contact sheet）是交叉验证手段之一，见 `experience.md` §5.7。
* 渲染产物（GIF 是交付物、stills/frames/reports 是工作目录）在训练/渲染流程中的位置，见 `train.md` §1、§2「产物」（本文档不重复其结论）。

## 9. 已知限制与坑

* **docstring 与现实不一致（务必注意）**：`robotdog/tools/png.py:3` 写 `The sandbox has no Pillow and installing one just to dump a frame is wasteful`，但当前仓库 `robotdog/render.py:223, 285, 337, 368` 确实 `from PIL import Image, ImageDraw, ImageFont`，且 `README.md` 把 `pillow 10.4.0` 列为依赖。正确读法是「本模块**刻意不依赖** Pillow」，不是「环境里没有 Pillow」。
* **`write_png` 不接受浮点图像**：传 `float32` 的 0~1 图像会得到
  `ValueError: 需要 uint8, 收到 float32`。
* **`write_png` 不接受灰度/alpha**：`(H, W)` 与 `(H, W, 4)` 都会得到
  `ValueError: 需要 (H, W, 3) 的 RGB 数组, 收到 (H, W)` 或 `收到 (H, W, 4)`。
* **`read_png` 拒绝一切非 8-bit truecolour**：读 Pillow 保存的调色板 PNG 或带 alpha 的 PNG 会得到
  `ValueError: 仅支持 8-bit RGB, 收到 depth=8 colour=6`。读到非 PNG 得到 `ValueError: 不是 PNG 文件`。
* **`read_png` 不校验 CRC**，也不检查「文件长度是否与 chunk 声明的长度一致」；截断的文件可能表现为 `zlib.error` 或行为异常，而不是明确的「文件损坏」。
* **`read_png` 依赖 `prev = line` 的原地语义**：`line` 是 `np.frombuffer(...).copy()`，每行重新绑定；若将来有人把 `.copy()` 去掉并写回 buffer，Sub/Average/Paeth 的结果会错。
* **`downscale` 会丢边**：尺寸不是 `factor` 的整数倍时，底部/右侧余数像素被丢弃（`crop = img[: h2*factor, : w2*factor]`），输出尺寸是 `h // factor` / `w // factor`。
* **`downscale` 可能返回入参本身**（`factor <= 1` 或结果维度为 0 时），调用方若就地把返回值当副本修改，会改到原数组。
* **`downscale` 硬编码 3 通道**：对 `(H, W)` 或 `(H, W, 4)` 输入会在 `reshape` 处抛 `ValueError`，而不是给出友好提示。
* **`hstack` 空列表**抛 `ValueError("没有图像")`；高度不齐时**不报错**，矮图下方留 `background` 色 —— 如果背景与图像颜色接近，问题会被视觉掩盖。
* **大帧的内存峰值**：`raw = b"".join(...)` 会先构造完整的未压缩字节串（640×480×3 ≈ 0.9 MB/帧，加每行 1 字节），随后 `zlib.compress` 再产出一份。批量转储时需要自己控制帧数，本模块没有流式写入路径。
* **没有 `__all__`**：`from robotdog.tools.png import *` 会把 `struct`、`zlib`、`Path`、`np` 一并导出。用显式导入。

## 10. 示例用法

```python
import numpy as np

from robotdog.tools.png import downscale, hstack, read_png, write_png

# 1) 写一张图（父目录自动创建，返回最终 Path）
img = np.zeros((480, 640, 3), dtype=np.uint8)
img[..., 0] = 200                                  # 偏红的一帧
out = write_png("output/stills/demo.png", img)     # -> Path('output/stills/demo.png')
print(out)

# 2) 读回来并核对形状（round-trip）
back = read_png(out)
assert back.shape == (480, 640, 3) and back.dtype == np.uint8

# 3) 降采样 + 横向拼接成对比图（pad=8 像素白缝）
small = downscale(back, 2)                          # (240, 320, 3)
sheet = hstack([small, small], pad=8)               # (240, 328, 3)
write_png("output/stills/demo_strip.png", sheet, compress_level=9)

# 4) 与 MuJoCo 渲染器直接对接（Renderer 返回的就是 write_png 期望的格式）
#    frame = renderer.render()
#    write_png(paths.OUTPUT_STILLS_DIR / "frame0.png", frame)
```
