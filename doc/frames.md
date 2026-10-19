# scripts/frames.py — 视频切片（粗 / 细）

对应代码：`scripts/frames.py`。视觉评审的规程在 `agent.md` §8 与
`.dsh/skills/visual-monitor/SKILL.md`；本文只讲这个工具的结构与取舍。

## 职责

把训练视频（`scripts/play.py` 的产物）切成**评审能直接读的 PNG**。两种模式，
对应视觉评审的两段：

- `--coarse N`：整段粗切片——均匀取 N 帧，但**强制含首 2 帧与末 4 帧**，拼成
  3×4 拼图，供「整体效果」（姿态 / 步态 / 结局与失败原因 / 指令跟踪）评审；
- `--window A:B`：窗口切片——把 `[A, B]` 秒内**视频原生的每一帧**摊开，供第一
  轮评审判定「需要细看」之后的短时间片分析。

两者都只做「读帧 → 缩放 → 烧角标 → 拼图 → 记账」。**不判事件、不插帧、不改
视频**：切片位置由人/评审员决定，工具不预设「可疑窗口」。

## 核心抽象

### 为什么末 4 帧必须在内

成功 / 摔倒 / 超时只发生在回合末尾。12 帧均匀采样时，末尾 1~2 帧经常刚好漏掉，
评审就只能写「结局不明」；首 2 帧同理（起始姿态）。这是**采样设计**，不是事件
检测——不含任何阈值或分类。

### 为什么窗口只认 25 fps

训练视频的 25 fps 来自 `play.py` 的 `FRAME_STRIDE=2`（50 Hz 控制 → 每 2 步一帧）。
更密的帧只能靠**重渲**（`--frame-stride 1`），本工具不插帧：低于 25 fps 的视频
直接报错并给出重渲命令。插帧出来的像素不是仿真状态，不能当接触证据。

### 为什么拼图单格 256、单帧 768

`read_image` 会把大图自动降采样。3 列 × 256 = 768 刚好压在阈值内；再大反而
丢掉每格细节。单帧 PNG 限宽 768 是沿用旧 `read_video.py` 的 token 预算（约 1k/张）。

### 预算

`--budget` 按 `--image-tokens`（默认 1000）估算**要读的图**：默认只数拼图
（单帧 PNG 是给引用用的，不一定逐张读）。超预算直接拒绝，而不是悄悄裁帧。

## 产物

```
<outdir>/f000.png ...        单帧（评审引用帧号用，限宽 768）
<outdir>/sheet_000.png ...   拼图（角标 fNNN t=1.24s）
<outdir>/manifest.json       video / fps / duration_ms / mode / window_ms /
                             sampled / sheet / images / image_tokens_estimate /
                             frames[]
```

## 已知的坑

- **旧视频可能不是 25 fps**：`play.video_max_frames=300` 会在 20 s 回合上把取帧
  间隔稀疏到 80 ms（12.5 fps）。评审前用 `.tmp/review_play.yaml`
  （`episodes_video: 1` + `video_max_frames: 500`）重渲；工具会直接拒绝 12.5 fps。
- **拼图不是证据的全部**：单格 256 宽只够看整体；接触/落点必须进窗口看单帧。
- **角标必须留半透明底板**：画面常是深色，白字直接压上去看不清。
- **读取器要关**：`imageio` 的 reader 在 `finally` 里关闭，否则 Windows 上文件句柄
  会拖住下一次覆盖。
- 字体顺序 `msyh.ttc` → `simhei.ttf` → 位图字体；前两个是 Windows 常见路径。

## 用法

```bat
:: 粗切片（整段）
.venv\Scripts\python.exe scripts\frames.py ^
    runs\<run>\videos\<批次>\<阶段>\<阶段>.mp4 .tmp\review\<run>\<阶段>\coarse --coarse 12

:: 细看窗口（第一轮评审判定需要时；视频必须是 25 fps）
.venv\Scripts\python.exe scripts\frames.py ^
    <同一个 mp4> .tmp\review\<run>\<阶段>\w1 --window 1.24:1.84
```
