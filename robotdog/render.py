"""Replay a trained checkpoint and export the rollout as an animated GIF.

This is the "look at what the policy actually does" half of step 9: the JSON
reports tell you *that* a policy reaches 45 % success, this module shows you
*how*.  It is deliberately separate from :mod:`robotdog.evaluate` (which never
renders) and from ``scripts/eval.py`` (which writes PNG frame dumps and a
contact sheet, but no animation).

Design notes
------------
* Frames come from ``Go2StairsEnv.render(camera=..., width=..., height=...)``,
  i.e. MuJoCo's offscreen OpenGL renderer -- no display window, works over RDP.
* The GIF container is written with Pillow (already present as a TensorBoard
  dependency).  Pillow gives us an adaptive palette plus frame differencing for
  free, which matters a lot: a 900-frame raw dump would be ~500 MB, the same
  animation is 2-6 MB as a GIF.
* The HUD overlay is optional and ASCII/CJK-safe: it picks Microsoft YaHei on
  Windows and falls back to DejaVu Sans, then to Pillow's bitmap font.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from robotdog.env import Go2StairsEnv
from robotdog.tools.png import downscale, hstack, write_png
from robotdog.tools.terrain import StairSpec

__all__ = [
    "RenderOptions",
    "EpisodeRender",
    "RenderResult",
    "render_rollout",
    "save_gif",
    "save_frames",
    "contact_sheet",
    "overlay_hud",
    "hud_lines",
]


class Predictor(Protocol):  # pragma: no cover - structural typing only
    """Anything with SB3's ``predict(obs, deterministic=...)`` signature."""

    def predict(self, obs: Any, deterministic: bool = True): ...


# --------------------------------------------------------------------------- #
# Options / results
# --------------------------------------------------------------------------- #


@dataclass
class RenderOptions:
    """Frame-capture settings for :func:`render_rollout`."""

    width: int = 960
    height: int = 540
    fps: int = 25
    #: Store one frame every ``frame_stride`` control steps.  Control runs at
    #: 50 Hz, so stride 2 with fps 25 gives real-time playback.
    frame_stride: int = 2
    camera: str = "track"
    #: Box-filter factor applied to each frame before the HUD is drawn (so the
    #: text is rasterised at the final size and stays sharp).
    scale: int = 1
    overlay: bool = True
    title: str = ""
    #: Cap the number of stored frames per episode (0 = unlimited).  Guards
    #: against a policy that just stands still on a 20 s episode.
    max_frames: int = 0
    max_steps: int | None = None


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

    @property
    def status(self) -> str:
        if self.success:
            return "登顶成功"
        if self.fell:
            return "摔倒"
        if self.timeout:
            return "超时"
        return "未结束"

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["status"] = self.status
        return d


@dataclass
class RenderResult:
    """All frames of a render job plus per-episode statistics."""

    frames: list[np.ndarray] = field(default_factory=list)
    episodes: list[EpisodeRender] = field(default_factory=list)
    #: Frame index at which each episode starts, so a multi-episode GIF can be
    #: labelled correctly during overlay.
    episode_starts: list[int] = field(default_factory=list)

    @property
    def successes(self) -> int:
        return sum(1 for e in self.episodes if e.success)

    @property
    def success_rate(self) -> float:
        return self.successes / len(self.episodes) if self.episodes else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "episodes": [e.as_dict() for e in self.episodes],
            "total_frames": len(self.frames),
            "successes": self.successes,
            "success_rate": round(self.success_rate, 4),
            "frames_per_episode": [e.frames for e in self.episodes],
        }

    def episode_of_frame(self, index: int) -> int:
        """Which rendered episode a global frame index belongs to (0-based)."""
        ep = 0
        for i, start in enumerate(self.episode_starts):
            if index >= start:
                ep = i
            else:
                break
        return ep


# --------------------------------------------------------------------------- #
# Rollout capture
# --------------------------------------------------------------------------- #


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
    """Roll the policy out and collect frames + per-episode statistics.

    The environment is *not* closed here -- the caller owns it (the renderer
    keeps a GL context alive and closing it early would force a recompile).
    """
    opts = opts or RenderOptions()
    stair = stair or env.stair
    max_steps = opts.max_steps or int(env.cfg.episode_length_s / env.cfg.control_dt)

    result = RenderResult()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        x0 = float(env.base_pos[0])
        z0 = float(env.base_pos[2])
        result.episode_starts.append(len(result.frames))

        step = 0
        terminated = truncated = False
        info: dict[str, Any] = {}
        while not (terminated or truncated) and step < max_steps:
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, _r, terminated, truncated, info = env.step(action)
            step += 1
            if step % opts.frame_stride == 0:
                if opts.max_frames and len(result.frames) - result.episode_starts[-1] >= opts.max_frames:
                    break
                result.frames.append(env.render(
                    opts.camera, width=opts.width, height=opts.height).copy())

        dt = step * env.cfg.control_dt
        result.episodes.append(EpisodeRender(
            episode=ep,
            steps=step,
            frames=len(result.frames) - result.episode_starts[-1],
            distance=round(float(env.base_pos[0]) - x0, 3),
            height_gain=round(float(env.base_pos[2]) - z0, 3),
            forward_speed=round((float(env.base_pos[0]) - x0) / dt, 3) if dt > 0 else 0.0,
            final_x=round(float(env.base_pos[0]), 3),
            success=bool(info.get("is_success", False)),
            fell=bool(terminated) and not bool(info.get("is_success", False)),
            timeout=bool(truncated) and not bool(info.get("is_success", False)),
        ))
    return result


# --------------------------------------------------------------------------- #
# HUD overlay
# --------------------------------------------------------------------------- #

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _load_font(size: int):
    """Best available TrueType font at ``size`` px, else Pillow's bitmap font."""
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # pragma: no cover - corrupt font file
                continue
    try:  # matplotlib ships DejaVu if the system has nothing
        import matplotlib

        cand = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        if cand.exists():
            return ImageFont.truetype(str(cand), size)
    except Exception:
        pass
    return ImageFont.load_default()


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
    """Build the text lines drawn on a frame (kept separate so it is testable).

    ``compact=True`` drops the least important lines, for small GIFs where
    five lines of text would eat a quarter of the picture.
    """
    lines: list[str] = []
    if title:
        lines.append(title)
    lines.append(
        f"台阶 {stair.step_height * 100:.1f}cm × {stair.n_up} 级"
        f"  踏面 {stair.tread * 100:.0f}cm  顶端 {stair.crest_height:.2f}m"
    )
    if episode is not None:
        lines.append(
            f"回合 {episode.episode + 1}  {episode.status}"
            f"  前进 {episode.distance:+.2f}m  速度 {episode.forward_speed:.2f}m/s"
        )
    if base_rpy is not None and not compact:
        lines.append(
            f"姿态 roll {np.degrees(base_rpy[0]):+.1f}°  "
            f"pitch {np.degrees(base_rpy[1]):+.1f}°"
        )
    if not compact:
        lines.append(f"帧 {index + 1}/{total}")
    return lines


def overlay_hud(frame: np.ndarray, lines: Sequence[str], *,
                font_size: int | None = None) -> np.ndarray:
    """Draw ``lines`` into the top-left corner of an RGB frame."""
    if not lines:
        return frame
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.ascontiguousarray(frame))
    draw = ImageDraw.Draw(img, "RGBA")
    size = font_size or max(12, frame.shape[0] // 26)
    font = _load_font(size)
    pad = max(4, size // 3)
    line_h = size + max(2, size // 4)

    # Translucent plate so white text stays readable on any background.
    width = 0
    for text in lines:
        try:
            box = draw.textbbox((0, 0), text, font=font)
        except Exception:  # pragma: no cover - bitmap font has no textbbox
            box = (0, 0, len(text) * size // 2, size)
        width = max(width, box[2] - box[0])
    plate = (pad, pad, pad + width + 2 * pad, pad + line_h * len(lines) + pad)
    draw.rectangle(plate, fill=(0, 0, 0, 140))

    y = pad + pad // 2
    for text in lines:
        draw.text((pad + pad, y), text, font=font, fill=(255, 255, 255, 255))
        y += line_h
    return np.asarray(img)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def save_frames(directory: str | Path, frames: Sequence[np.ndarray],
                *, prefix: str = "f", downscale_factor: int = 1) -> list[Path]:
    """Write every frame as a dependency-free PNG (see :mod:`robotdog.tools.png`)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, frame in enumerate(frames):
        img = downscale(frame, downscale_factor) if downscale_factor > 1 else frame
        written.append(write_png(directory / f"{prefix}{i:04d}.png", img))
    return written


def _global_palette(images: Sequence["Any"], sample: int = 16):
    """Quantise a mosaic of sampled frames into one shared 256-colour palette.

    A per-frame adaptive palette makes every frame carry its own colour table
    *and* re-dithers the noisy MuJoCo background differently each time, which
    both bloats the GIF and makes the sky shimmer.  One global palette fixes
    both: the colour table is written once and the background stays stable.
    """
    from PIL import Image

    if len(images) <= sample:
        picks = list(images)
    else:
        idx = np.linspace(0, len(images) - 1, sample).round().astype(int)
        picks = [images[int(i)] for i in idx]
    cols = 4
    rows = (len(picks) + cols - 1) // cols
    thumb_w = max(1, picks[0].width // 4)
    thumb_h = max(1, picks[0].height // 4)
    mosaic = Image.new("RGB", (thumb_w * cols, thumb_h * rows))
    for i, img in enumerate(picks):
        mosaic.paste(img.resize((thumb_w, thumb_h), Image.BOX),
                     ((i % cols) * thumb_w, (i // cols) * thumb_h))
    return mosaic.quantize(colors=256, method=Image.MEDIANCUT)


def save_gif(path: str | Path, frames: Sequence[np.ndarray], *,
             fps: int = 25, scale: int = 1, loop: int = 0,
             optimize: bool = True, dither: bool = True,
             palette_sample: int = 16) -> Path:
    """Encode ``frames`` (H, W, 3) uint8 into an animated GIF.

    ``scale`` box-filters by an integer factor before encoding -- a 960x540@25fps
    20 s GIF is fine, but half-size is a third of the bytes and, for a README,
    just as readable.  All frames share one palette (see
    :func:`_global_palette`).
    """
    if not frames:
        raise ValueError("没有帧可以写入 GIF")
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(20, int(round(1000.0 / max(1, fps))))

    rgb: list[Any] = []
    for frame in frames:
        img = Image.fromarray(np.ascontiguousarray(frame))
        if scale > 1:
            img = img.resize((max(1, img.width // scale), max(1, img.height // scale)),
                             Image.BOX)
        rgb.append(img.convert("RGB"))

    palette = _global_palette(rgb, palette_sample)
    mode = Image.FLOYDSTEINBERG if dither else Image.NONE
    images = [img.quantize(palette=palette, dither=mode) for img in rgb]

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop,
        optimize=optimize,
        disposal=2,
    )
    return path


def contact_sheet(frames: Sequence[np.ndarray], *, n: int = 12,
                  scale: int = 4) -> np.ndarray:
    """Horizontal strip of ``n`` evenly spaced (downscaled) frames."""
    if not frames:
        raise ValueError("没有帧可以拼图")
    if len(frames) <= n:
        picked = list(frames)
    else:
        idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
        picked = [frames[int(i)] for i in idx]
    return hstack([downscale(f, scale) for f in picked])


def copy_into(src: str | Path, dst: str | Path) -> Path:
    """Small shim used by the CLI to stage exported models into ``output/``."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst
