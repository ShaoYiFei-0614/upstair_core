"""播放 / 评估训练结果。

两种模式（渲染参数固定，不再由命令行暴露）：

* ``--mode gif``  —— 连续跑 ``play.episodes_gif`` 个回合（默认 10），逐帧渲染并叠加
  HUD，最后编码成一个 GIF，落在 ``output/gifs/<名字>.gif``。
* ``--mode eval`` —— 跑 ``play.episodes_eval`` 个回合（默认 100），只统计成功率 /
  摔倒率 / 超时率 / 平均前进。

用法::

    .venv\\Scripts\\python.exe scripts\\play.py --run course1 --mode eval
    .venv\\Scripts\\python.exe scripts\\play.py --run course1 --mode gif

``--run`` 可以是运行名、运行目录，或者直接的 ``.zip`` 路径。同名目录下的
``*_vecnormalize.pkl`` 会被自动加载；没有它时按原始观测推理（会提示）。
想换楼梯高度或相机，改 ``config.yaml`` 的 ``play:`` 段，不加命令行参数。

评估至少跑 40 回合，10 回合能给出 ±15 个百分点的错觉。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robotdog.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    config_to_play,
    load_config,
)
from robotdog.env import EnvConfig, Go2StairsEnv  # noqa: E402
from robotdog.policy import (  # noqa: E402
    ObsNorm,
    env_config_for_model,
    stair_from_height,
    step_height_from_model,
)
from robotdog.tools import console, paths  # noqa: E402
from robotdog.training import load_policy  # noqa: E402
from stable_baselines3.common.vec_env import VecEnv  # noqa: E402

#: 渲染时的每帧步数：50 Hz 控制下每 2 步取一帧 = 25 fps。
FRAME_STRIDE = 2

#: 评估模式用的相机（固定，不暴露到命令行）。
#: GIF 模式的相机不在这里 —— 它来自 ``config.yaml`` 的 ``play.camera``。
CAMERA_EVAL = "side"

#: 字体候选（HUD 用），按顺序取第一个存在的。
_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


# --------------------------------------------------------------------------- #
# 结果定位
# --------------------------------------------------------------------------- #


@dataclass
class PlayTarget:
    """一次播放需要的三样东西：模型、归一化统计、环境配置。"""

    model_path: Path
    norm_path: Path | None
    cfg: EnvConfig
    label: str


def _resolve_model(run: str) -> Path:
    """``--run`` 可以给运行名、运行目录或直接的 .zip 路径。"""
    raw = Path(run)
    if raw.suffix == ".zip":
        if not raw.is_file():
            raise SystemExit(f"找不到模型：{raw}")
        return raw
    root = raw if raw.is_dir() else paths.RUNS_DIR / run
    if not root.is_dir():
        raise SystemExit(f"找不到运行目录：{root}")
    for name in ("model_final.zip", "artifacts/best_stage.zip", "model_best.zip"):
        cand = root / name
        if cand.is_file():
            return cand
    zips = sorted(root.glob("*.zip"))
    if zips:
        return zips[0]
    # 没跑完 / 中途停掉的 run 只有阶段 checkpoint（``checkpoints/<阶段>_final.zip``）。
    # 取最后一个阶段的，好让「先训一级看一眼」这种用法也能直接 play。
    stage_zips = sorted((root / "checkpoints").glob("*_final.zip"))
    if stage_zips:
        return stage_zips[-1]
    raise SystemExit(f"{root} 下没有任何 .zip 权重")


def _resolve_norm(model_path: Path) -> Path | None:
    cand = model_path.with_name(model_path.stem + "_vecnormalize.pkl")
    if cand.is_file():
        return cand
    cand = model_path.with_suffix(".pkl")
    return cand if cand.is_file() else None


def resolve_target(run: str) -> PlayTarget:
    """从运行名 / 目录 / zip 解析出模型、归一化统计和它训练时的配置。"""
    model_path = _resolve_model(run)
    return PlayTarget(
        model_path=model_path,
        norm_path=_resolve_norm(model_path),
        cfg=env_config_for_model(model_path),
        label=model_path.stem,
    )


# --------------------------------------------------------------------------- #
# 策略适配
# --------------------------------------------------------------------------- #


class Runner:
    """把 PPO + 观测归一化统计包成 ``predict(obs) -> action``。"""

    def __init__(self, model, norm: ObsNorm | None) -> None:
        self.model = model
        self.norm = norm
        model.policy.set_training_mode(False)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        x = obs if self.norm is None else self.norm.normalize(obs)
        action, _ = self.model.predict(x, deterministic=True)
        return action


def open_runner(target: PlayTarget, *, stair, device: str = "cpu"
                ) -> tuple[Runner, Go2StairsEnv, "VecEnv"]:
    """加载权重并建好环境。

    返回 ``(Runner, 环境, 向量环境)``。向量环境必须一起返回并保活：``PPO`` 持有它，
    归一化统计也存在它上面。
    """
    model, norm = load_policy(
        target.model_path, cfg=target.cfg, stair=stair, device=device
    )
    if norm is None:
        print("⚠ 没有找到 *_vecnormalize.pkl，按原始观测推理（结果可能严重失真）")
    env = Go2StairsEnv(stair=stair, cfg=target.cfg, seed=0)
    return Runner(model, norm), env, model.env


# --------------------------------------------------------------------------- #
# GIF 写出（Pillow）
# --------------------------------------------------------------------------- #


def _load_font(size: int):
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # pragma: no cover - 字体文件损坏
                continue
    return ImageFont.load_default()


def draw_hud(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """把 ``lines`` 画到画面左上角的半透明底板上。"""
    if not lines:
        return frame
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.ascontiguousarray(frame))
    draw = ImageDraw.Draw(img, "RGBA")
    size = max(12, frame.shape[0] // 26)
    font = _load_font(size)
    pad = max(4, size // 3)
    line_h = size + max(2, size // 4)

    width = 0
    for text in lines:
        try:
            box = draw.textbbox((0, 0), text, font=font)
        except Exception:  # pragma: no cover - 位图字体没有 textbbox
            box = (0, 0, len(text) * size // 2, size)
        width = max(width, box[2] - box[0])
    draw.rectangle(
        (pad, pad, pad + width + 2 * pad, pad + line_h * len(lines) + pad),
        fill=(0, 0, 0, 140),
    )
    y = pad + pad // 2
    for text in lines:
        draw.text((pad + pad, y), text, font=font, fill=(255, 255, 255, 255))
        y += line_h
    return np.asarray(img)


def write_gif(path: Path, frames: Sequence[np.ndarray], *, fps: int,
              scale: int) -> Path:
    """把 RGB 帧编码成 GIF，全片共用一套调色板。

    先取若干帧拼成缩略 mosaic 求全局调色板，再逐帧用它量化。逐帧各自
    ``quantize()`` 会让同一只狗在不同帧里换颜色（GIF 只有 256 色，必须共用）。
    """
    if not frames:
        raise ValueError("没有帧可以写入 GIF")
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = []
    for frame in frames:
        img = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
        if scale > 1:
            img = img.resize(
                (max(1, img.width // scale), max(1, img.height // scale)), Image.BOX
            )
        rgb.append(img)

    sample = min(16, len(rgb))
    idx = np.linspace(0, len(rgb) - 1, sample).round().astype(int)
    picks = [rgb[int(i)] for i in idx]
    cols = 4
    rows = (len(picks) + cols - 1) // cols
    tw, th = max(1, picks[0].width // 4), max(1, picks[0].height // 4)
    mosaic = Image.new("RGB", (tw * cols, th * rows))
    for i, img in enumerate(picks):
        mosaic.paste(img.resize((tw, th), Image.BOX), ((i % cols) * tw, (i // cols) * th))
    palette = mosaic.quantize(colors=256, method=Image.MEDIANCUT)

    images = [img.quantize(palette=palette, dither=Image.FLOYDSTEINBERG) for img in rgb]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(20, int(round(1000.0 / max(1, fps)))),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return path


# --------------------------------------------------------------------------- #
# 回合
# --------------------------------------------------------------------------- #


@dataclass
class EpisodeOutcome:
    """一个回合的结果，两种模式都用它。"""

    episode: int
    steps: int
    distance: float = 0.0
    height_gain: float = 0.0
    speed: float = 0.0
    final_x: float = 0.0
    success: bool = False
    fell: bool = False
    timeout: bool = False
    frames: list[np.ndarray] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.success:
            return "成功"
        return "摔倒" if self.fell else "超时"


def run_episode(env: Go2StairsEnv, policy: Runner, *, episode: int, seed: int,
                capture: bool, camera: str, width: int, height: int) -> EpisodeOutcome:
    """跑一个回合；``capture=True`` 时每 ``FRAME_STRIDE`` 步取一帧。"""
    obs, _ = env.reset(seed=seed)
    x0, z0 = float(env.base_pos[0]), float(env.base_pos[2])
    max_steps = int(env.cfg.episode_length_s / env.cfg.control_dt)
    out = EpisodeOutcome(episode=episode, steps=0)
    info: dict[str, Any] = {}
    terminated = truncated = False

    step = 0
    while not (terminated or truncated) and step < max_steps:
        obs, _r, terminated, truncated, info = env.step(policy.predict(obs))
        step += 1
        if capture and step % FRAME_STRIDE == 0:
            out.frames.append(env.render(camera, width=width, height=height).copy())

    dt = step * env.cfg.control_dt
    out.steps = step
    out.distance = round(float(env.base_pos[0]) - x0, 3)
    out.height_gain = round(float(env.base_pos[2]) - z0, 3)
    out.speed = round(out.distance / dt, 3) if dt > 0 else 0.0
    out.final_x = round(float(env.base_pos[0]), 3)
    out.success = bool(info.get("is_success", False))
    out.fell = bool(terminated) and not out.success
    out.timeout = bool(truncated) and not out.success
    return out


def hud_for(out: EpisodeOutcome, total: int, stair, *, frame_index: int,
            frame_total: int) -> list[str]:
    """生成一帧上的 HUD 文本。"""
    return [
        f"回合 {out.episode + 1}/{total}  {out.status}",
        f"台阶 {stair.step_height * 100:.1f}cm × {stair.n_up} 级"
        f"  踏面 {stair.tread * 100:.0f}cm  顶端 {stair.crest_height:.2f}m",
        f"前进 {out.distance:+.2f}m  速度 {out.speed:.2f}m/s  帧 {frame_index}/{frame_total}",
    ]


# --------------------------------------------------------------------------- #
# 模式
# --------------------------------------------------------------------------- #


def mode_gif(env: Go2StairsEnv, policy: Runner, *, episodes: int, seed: int,
             name: str, play: dict[str, Any]) -> int:
    """模式 1：连续渲染 ``episodes`` 个回合，合成一个 GIF。"""
    camera = str(play["camera"])
    width, height = int(play["width"]), int(play["height"])
    fps, scale = int(play["fps"]), int(play["scale"])

    frames: list[np.ndarray] = []
    outcomes: list[EpisodeOutcome] = []
    for ep in range(episodes):
        out = run_episode(env, policy, episode=ep, seed=seed + ep, capture=True,
                          camera=camera, width=width, height=height)
        outcomes.append(out)
        total = len(out.frames)
        for i, frame in enumerate(out.frames):
            frames.append(draw_hud(frame, hud_for(out, episodes, env.stair,
                                                  frame_index=i + 1,
                                                  frame_total=total)))
        print(f"  回合 {ep + 1:>2}/{episodes}  {out.status}  "
              f"{out.steps:>4} 步  前进 {out.distance:+.2f}m  帧 {total}", flush=True)

    gif = paths.OUTPUT_GIFS_DIR / f"{name}.gif"
    write_gif(gif, frames, fps=fps, scale=scale)
    ok = sum(o.success for o in outcomes)
    size_mb = gif.stat().st_size / 1e6
    print(f"\nGIF: {gif}  ({len(frames)} 帧, {size_mb:.1f} MB, {fps} fps)")
    print(f"成功率: {ok}/{episodes} = {100.0 * ok / max(1, episodes):.1f}%")
    return 0


def mode_eval(env: Go2StairsEnv, policy: Runner, *, episodes: int, seed: int,
              model_path: Path | None = None) -> int:
    """模式 2：跑 ``episodes`` 个回合，只统计成功率。"""
    outcomes: list[EpisodeOutcome] = []
    t0 = time.time()
    for ep in range(episodes):
        out = run_episode(env, policy, episode=ep, seed=seed + ep, capture=False,
                          camera=CAMERA_EVAL, width=0, height=0)
        outcomes.append(out)
        if (ep + 1) % 10 == 0 or ep + 1 == episodes:
            ok = sum(o.success for o in outcomes)
            print(f"  {ep + 1:>3}/{episodes}  成功率 {100.0 * ok / (ep + 1):5.1f}%",
                  flush=True)

    n = len(outcomes)
    succ = sum(o.success for o in outcomes)
    fell = sum(o.fell for o in outcomes)
    tout = sum(o.timeout for o in outcomes)
    mean = lambda key: float(np.mean([getattr(o, key) for o in outcomes]))  # noqa: E731
    payload = {
        "model": str(model_path or ""),
        "episodes": n,
        "success_rate": round(succ / max(1, n), 4),
        "fall_rate": round(fell / max(1, n), 4),
        "timeout_rate": round(tout / max(1, n), 4),
        "mean_distance": round(mean("distance"), 3),
        "mean_height_gain": round(mean("height_gain"), 3),
        "mean_speed": round(mean("speed"), 3),
        "mean_steps": round(mean("steps"), 1),
        "wall_seconds": round(time.time() - t0, 1),
    }
    print(f"\n台阶 {env.stair.step_height * 100:.1f}cm × {env.stair.n_up} 级"
          f"  踏面 {env.stair.tread * 100:.0f}cm")
    print(f"成功率 {100.0 * succ / max(1, n):5.1f}%   "
          f"摔倒 {100.0 * fell / max(1, n):5.1f}%   "
          f"超时 {100.0 * tout / max(1, n):5.1f}%")
    print(f"平均前进 {payload['mean_distance']:.2f}m   "
          f"平均爬升 {payload['mean_height_gain']:.2f}m   "
          f"平均速度 {payload['mean_speed']:.2f}m/s   "
          f"平均步数 {payload['mean_steps']:.0f}")
    return 0


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(
        description="播放训练结果：渲染 GIF 或统计成功率（参数在 config.yaml）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run", required=True, help="运行名 / 运行目录 / .zip 路径")
    ap.add_argument("--mode", choices=("gif", "eval"), default="gif")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                    help="配置文件路径（渲染参数从这里读）")
    return ap.parse_args(argv)


def _default_height_cm(play: dict[str, Any], target: PlayTarget) -> float | None:
    """播放用哪条楼梯。

    优先 ``config.yaml`` 的 ``play.stage_height_cm``（显式覆盖）；否则读权重旁
    ``<模型>_env.json`` 里记的该 checkpoint 训练时的台阶高（**每个 checkpoint 的
    地形都不一样**，只有最后一级是 15 cm）；再没有就退回 :func:`stair_from_height`
    的默认（3 cm）。
    """
    if play.get("stage_height_cm") is not None:
        return float(play["stage_height_cm"])
    height = step_height_from_model(target.model_path)
    if height is not None:
        print(f"地形来自权重自带记录: 台阶 {height * 100:.1f}cm")
        return height * 100.0
    return None


def main(argv: Sequence[str] | None = None) -> int:
    console.setup_utf8_console()
    args = parse_args(argv)
    play = config_to_play(load_config(args.config))
    target = resolve_target(args.run)
    print(f"模型: {target.model_path}")
    print(f"归一化: {target.norm_path or '（无）'}")

    stair = stair_from_height(_default_height_cm(play, target))
    policy, env, _venv = open_runner(target, stair=stair, device=str(play.get("device", "cpu")))
    try:
        print(f"地形: 台阶 {env.stair.step_height * 100:.1f}cm × {env.stair.n_up} 级"
              f"  踏面 {env.stair.tread * 100:.0f}cm")
        print(f"观测: {env.observation_space.shape}  动作: {env.action_space.shape}")

        if args.mode == "gif":
            return mode_gif(
                env, policy,
                episodes=int(play["episodes_gif"]),
                seed=int(play["seed"]),
                name=target.label,
                play=play,
            )
        return mode_eval(
            env, policy,
            episodes=int(play["episodes_eval"]),
            seed=int(play["seed"]),
            model_path=target.model_path,
        )
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
