r"""把训练视频切成评审用的帧序列：粗切片（整段概览）与窗口切片（25 fps 细看）。

用法::

    .venv\Scripts\python.exe scripts\frames.py <mp4> <outdir> --coarse 12
    .venv\Scripts\python.exe scripts\frames.py <mp4> <outdir> --window 1.24:1.84

两种模式都只做「取帧 + 拼图 + 记账」，不判事件、不插帧、不改视频：

* --coarse N：整段均匀取 N 帧，但强制包含首 2 帧与末 4 帧（结局必须在
  画面里；这是采样设计，不是事件检测），拼成拼图供「整体效果」评审；
* --window A:B：把 [A, B] 秒内视频原生的每一帧摊开（默认要求 25 fps
  = 40 ms/帧；低于 25 fps 直接拒绝，绝不插帧），供第一轮评审判定
  「需要细看」之后的短时间片分析。

设计细节与踩坑见 doc/frames.md。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robotdog.tools import console  # noqa: E402

#: 粗切片默认帧数；构成 = 首 2 + 均匀中间 + 末 4。
COARSE_FRAMES = 12
COARSE_HEAD = 2
COARSE_TAIL = 4
#: 粗切片默认拼图布局（列 × 行）。
COARSE_SHEET = (3, 4)
#: 窗口切片默认拼图布局。
WINDOW_SHEET = (3, 3)
#: 拼图单格宽度：3 列 × 256 = 768，正好压在 read_image 自动降采样之前。
TILE_WIDTH = 256
#: 单帧 PNG 的最大宽度（沿用旧 read_video.py 的预算）。
MAX_WIDTH = 768
#: 窗口切片要求的原生帧率。不做插帧：不达标就报错。
TARGET_FPS = 25.0
FPS_TOLERANCE = 0.5
#: 单个窗口最多摊开多少帧（0.4~1.0 s @25 fps = 10~25 帧）。
MAX_WINDOW_FRAMES = 40
#: token 预算：一张图按 1000 token 估（768 宽约 1k）。
IMAGE_TOKENS = 1000
DEFAULT_BUDGET = 30000

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _font(size: int):
    """角标用的字体；找不到就退回位图字体（字小但不会崩）。"""
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # pragma: no cover - 字体文件损坏
                continue
    return ImageFont.load_default()


def _parse_sheet(text: str) -> tuple[int, int]:
    """3x4 -> (3, 4)。"""
    try:
        cols, rows = (int(x) for x in text.lower().split("x"))
    except Exception as exc:  # noqa: BLE001 - 统一转成用户可读的报错
        raise SystemExit(
            f"--sheet 需要写成 <列>x<行>（如 3x4），收到 {text!r}") from exc
    if cols < 1 or rows < 1:
        raise SystemExit(f"--sheet 的行列都必须 >=1，收到 {text!r}")
    return cols, rows


def _parse_window(text: str) -> tuple[float, float]:
    """1.24:1.84 -> (1.24, 1.84) 秒。"""
    try:
        start, end = (float(x) for x in text.split(":"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"--window 需要写成 <起>:<止> 秒（如 1.24:1.84），收到 {text!r}") from exc
    if start < 0 or end <= start:
        raise SystemExit(f"--window 需要 0 <= 起 < 止，收到 {text!r}")
    return start, end


def coarse_indices(total: int, n: int) -> list[int]:
    """整段均匀取 n 帧，并强制包含首 COARSE_HEAD 帧与末 COARSE_TAIL 帧。

    末几帧必须在内：成功 / 摔倒 / 超时的结局只出现在回合末尾，均匀采样经常
    刚好漏掉。去重后按时间排序；总数尽量保持 n（中间点与首尾碰撞时从中间删）。
    """
    if total <= 0:
        raise SystemExit("视频里没有可读的帧")
    n = max(1, min(int(n), int(total)))
    head = min(COARSE_HEAD, n)
    tail = min(COARSE_TAIL, n - head)
    middle = n - head - tail
    picks: set[int] = set(range(head)) | set(range(total - tail, total))
    if middle > 0:
        lo, hi = head, total - tail - 1
        if hi <= lo:
            picks |= set(range(lo, hi + 1))
        else:
            span = hi - lo
            for k in range(middle):
                picks.add(round(lo + k * span / middle))
    out = sorted(p for p in picks if 0 <= p < total)
    if len(out) > n:
        head_set = set(range(head))
        tail_set = set(range(total - tail, total))
        keep = [p for p in out if p in head_set or p in tail_set]
        rest = [p for p in out if p not in head_set and p not in tail_set]
        while len(keep) + len(rest) > n and rest:
            rest.pop(len(rest) // 2)
        out = sorted(keep + rest)
    return out


def window_indices(total: int, fps: float | None, start: float, end: float, *,
                   target_fps: float = TARGET_FPS,
                   max_frames: int = MAX_WINDOW_FRAMES) -> list[int]:
    """把 [start, end] 秒内的原生帧摊开（按 target_fps 抽样，不插帧）。"""
    if not fps:
        raise SystemExit("视频没有 fps 元数据，无法按时间窗切帧")
    if float(fps) + FPS_TOLERANCE < float(target_fps):
        raise SystemExit(
            f"这段视频是 {float(fps):g} fps（帧间隔 {1000.0 / float(fps):.0f} ms），"
            f"达不到 {float(target_fps):g} fps 的细看要求；不插帧。\n"
            "请先用评审配置重渲（只改 play 段，不改代码）：\n"
            "  play.episodes_video: 1\n  play.video_max_frames: 500\n"
            "  .venv\\Scripts\\python.exe scripts\\play.py --run <run> --mode video "
            "--stage <级> --video-scale 1 --config .tmp\\review_play.yaml --batch <批次>\n"
            "示例配置见 .dsh\\skills\\visual-monitor\\SKILL.md 第 0 节。")
    i0 = max(0, int(round(start * float(fps))))
    i1 = min(int(total) - 1, int(round(end * float(fps))))
    if i1 < i0:
        raise SystemExit(
            f"窗口 {start:g}~{end:g} s 落在视频之外（时长 {int(total) / float(fps):.2f} s）")
    step = max(1, int(round(float(fps) / float(target_fps))))
    idxs = list(range(i0, i1 + 1, step))
    if len(idxs) > max_frames:
        raise SystemExit(
            f"窗口 {start:g}~{end:g} s 要摊开 {len(idxs)} 帧，超过上限 {max_frames}；"
            "请缩短窗口（建议 0.4~1.0 s）或显式提高 --max-window-frames")
    return idxs


def _load(reader, index: int, max_width: int):
    """读一帧并缩到 max_width 以内（宽高比不变）。"""
    from PIL import Image

    frame = np.asarray(reader.get_data(int(index)))
    img = Image.fromarray(frame)
    if max_width and img.width > max_width:
        img = img.resize(
            (int(max_width), round(img.height * max_width / img.width)), Image.LANCZOS)
    return img


def _caption(img, text: str, font):
    """在左上角烧一条半透明底板的角标。"""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img, "RGBA")
    box = draw.textbbox((0, 0), text, font=font)
    width, height = box[2] - box[0], box[3] - box[1]
    pad = max(3, height // 4)
    draw.rectangle((0, 0, width + 2 * pad, height + 2 * pad), fill=(0, 0, 0, 170))
    draw.text((pad, pad - box[1]), text, font=font, fill=(255, 255, 255, 255))
    return img


def _sheet(images, cols: int, rows: int, tile_width: int, font, captions):
    """把若干帧拼成一张 cols × rows 的拼图（单格宽 tile_width）。"""
    from PIL import Image

    if not images:
        raise SystemExit("没有可拼的帧")
    first = images[0]
    tile_h = max(1, round(tile_width * first.height / first.width))
    sheet = Image.new("RGB", (cols * tile_width, rows * tile_h), (8, 8, 8))
    for k, (img, text) in enumerate(zip(images, captions)):
        if k >= cols * rows:
            break
        tile = img.resize((tile_width, tile_h), Image.LANCZOS)
        if font is not None and text:
            tile = _caption(tile, text, font)
        sheet.paste(tile, ((k % cols) * tile_width, (k // cols) * tile_h))
    return sheet


def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(
        description="把训练视频切成评审用的帧（粗切片 / 25 fps 窗口）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("video", help="输入 MP4")
    ap.add_argument("outdir", help="输出目录（建议放 .tmp/review/<run>/<阶段>/...）")
    ap.add_argument("--coarse", type=int, default=None,
                    help=f"粗切片：整段取 N 帧（默认 {COARSE_FRAMES}；含首 2 末 4）")
    ap.add_argument("--window", default=None,
                    help="窗口切片：<起>:<止> 秒，摊开窗口内的原生帧")
    ap.add_argument("--fps", type=float, default=TARGET_FPS,
                    help="窗口切片要求的帧率；低于它直接报错，不插帧")
    ap.add_argument("--sheet", default=None,
                    help="拼图布局 <列>x<行>（粗切片默认 3x4，窗口默认 3x3）")
    ap.add_argument("--tile-width", type=int, default=TILE_WIDTH, help="拼图单格宽度")
    ap.add_argument("--max-width", type=int, default=MAX_WIDTH, help="单帧 PNG 最大宽度")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help="token 预算（按 --image-tokens 估算，超出直接拒绝）")
    ap.add_argument("--image-tokens", type=int, default=IMAGE_TOKENS,
                    help="一张图的 token 估算")
    ap.add_argument("--count-frames", action="store_true",
                    help="预算里把单帧 PNG 也算成图（默认只算拼图）")
    ap.add_argument("--no-caption", action="store_true", help="不烧角标")
    ap.add_argument("--max-window-frames", type=int, default=MAX_WINDOW_FRAMES,
                    help="单个窗口最多摊开多少帧")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    console.setup_utf8_console()
    args = parse_args(argv)
    if args.coarse is not None and args.window is not None:
        raise SystemExit("--coarse 与 --window 只能二选一")
    mode = "window" if args.window is not None else "coarse"

    import imageio

    src = Path(args.video)
    if not src.is_file():
        raise SystemExit(f"找不到视频：{src}")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    reader = imageio.get_reader(src)
    try:
        meta = reader.get_meta_data()
        fps = meta.get("fps")
        total = int(reader.count_frames())
        if mode == "coarse":
            n = COARSE_FRAMES if args.coarse is None else int(args.coarse)
            if n < 1:
                raise SystemExit("--coarse 必须 >=1")
            idxs = coarse_indices(total, n)
            cols, rows = (_parse_sheet(args.sheet) if args.sheet else COARSE_SHEET)
            window_ms = None
        else:
            start, end = _parse_window(str(args.window))
            idxs = window_indices(total, fps, start, end,
                                  target_fps=args.fps,
                                  max_frames=args.max_window_frames)
            cols, rows = (_parse_sheet(args.sheet) if args.sheet else WINDOW_SHEET)
            window_ms = [round(start * 1000), round(end * 1000)]

        per_sheet = cols * rows
        n_sheets = math.ceil(len(idxs) / per_sheet)
        n_images = n_sheets + (len(idxs) if args.count_frames else 0)
        if n_images * int(args.image_tokens) > int(args.budget):
            raise SystemExit(
                f"预计要用 {n_images} 张图 ≈{n_images * int(args.image_tokens)} token，"
                f"超过预算 {args.budget}；请减少 --coarse、缩短窗口，或显式提高 --budget")

        font = None if args.no_caption else _font(max(12, int(args.tile_width) // 12))
        frames_meta: list[dict] = []
        loaded: list[tuple[object, str]] = []
        for k, idx in enumerate(idxs):
            img = _load(reader, idx, int(args.max_width))
            name = f"f{k:03d}.png"
            img.save(out / name)
            t_ms = round(int(idx) * 1000.0 / float(fps)) if fps else None
            frames_meta.append({"file": name, "frame_index": int(idx), "t_ms": t_ms})
            text = name[:-4] if t_ms is None else f"{name[:-4]}  t={t_ms / 1000.0:.2f}s"
            loaded.append((img, text))

        images: list[str] = []
        for s in range(n_sheets):
            chunk = loaded[s * per_sheet:(s + 1) * per_sheet]
            sheet = _sheet([c[0] for c in chunk], cols, rows,
                           int(args.tile_width), font, [c[1] for c in chunk])
            sname = f"sheet_{s:03d}.png"
            sheet.save(out / sname)
            images.append(sname)

        manifest = {
            "video": str(src),
            "total_frames": total,
            "fps": float(fps) if fps else None,
            "duration_ms": round(total * 1000.0 / float(fps)) if fps else None,
            "mode": mode,
            "window_ms": window_ms,
            "sampled": len(idxs),
            "sheet": [cols, rows],
            "images": images,
            "image_tokens_estimate": n_images * int(args.image_tokens),
            "frames": frames_meta,
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False))
        return 0
    finally:
        reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
