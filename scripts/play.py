"""播放 / 评估训练结果。

两种模式（渲染参数固定，不再由命令行暴露）：

* ``--mode gif``  —— 按课程阶段**逐个**跑 ``play.episodes_gif`` 个回合（默认 10），
  逐帧渲染并叠加 HUD（**含当前速度指令**），编码成 GIF。
  ⭐ 产物按批次 / 阶段分目录：``runs/<运行名>/gifs/<批次>/<阶段>/<阶段>.gif``。
  平地级与台阶级都渲染 —— 平地那几级是判断「跟不跟横向/偏航指令」的唯一画面证据。
* ``--mode eval`` —— 跑 ``play.episodes_eval`` 个回合（默认 100），只统计成功率 /
  摔倒率 / 超时率 / 平均前进。

用法::

    .venv\\Scripts\\python.exe scripts\\play.py --run cmdtrack --mode eval
    .venv\\Scripts\\python.exe scripts\\play.py --run cmdtrack --mode gif --batch batch1
    .venv\\Scripts\\python.exe scripts\\play.py --run cmdtrack --mode gif --stage stair9cm

``--run`` 可以是运行名、运行目录，或者直接的 ``.zip`` 路径。同名目录下的
``*_vecnormalize.pkl`` 会被自动加载；没有它时按原始观测推理（会提示）。
相机等渲染参数改 ``config.yaml`` 的 ``play:`` 段；要渲染哪些阶段用
``play.stages``（``auto`` | ``curriculum`` | 阶段名或高度列表）配 ``--stage``。

评估至少跑 40 回合，10 回合能给出 ±15 个百分点的错觉。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robotdog.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    command_for_stage,
    config_to_play,
    config_to_stages,
    device,
    load_config,
)
from robotdog.env import EnvConfig, Go2StairsEnv  # noqa: E402
# 运行时真的要构造它（``_spec_from_json`` 用 ``summary.json`` 的记录重建口径），
# 所以不能只放进 TYPE_CHECKING —— 那会让 GIF 模式在读到旧 summary 时抛 NameError。
from robotdog.curriculum import StageSpec  # noqa: E402

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
CAMERA_EVAL = "side"

# 渲染参数**固定**，不经 ``config.yaml``（相机、分辨率、帧率、缩放都是「看画面」
# 的事，不是「训练行为」的事，留成旋钮只会让人以为改它们有意义）：
#: GIF 用的相机：沿 x 跟随机器人的侧视图。
CAMERA_GIF = "track"
#: 渲染分辨率（mujoco 离屏渲染），GIF 编码前按 :data:`GIF_SCALE` 整数缩小。
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
#: GIF 帧率（HUD 里「帧 i/n」的基准；实际回放帧间隔由取帧步幅决定，见 run_episode）。
GIF_FPS = 25
#: GIF 编码前的整数缩小倍数；2 表示 960×540 → 480×270。
GIF_SCALE = 2


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


def read_json_or(path: Path | None, default: Any) -> Any:
    """读一个 JSON 文件；不存在 / 解析失败返回 ``default``。

    训练记录（``summary.json`` / ``plan.json``）是**可选数据**：老运行可能没有,
    半路夭折的运行可能写坏了 —— 读不到就按「没有记录」处理,而不是让回放崩掉。
    """
    if path is None or not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


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


def write_gif(path: Path, frames: Sequence[np.ndarray], *, frame_ms: int = 40) -> Path:
    """把 RGB 帧编码成 GIF（全片共用一套调色板）。

    先取若干帧拼缩略 mosaic 求全局调色板，再逐帧套用 —— 逐帧各自量化会让同一只
    狗在不同帧里换色。按「逐帧转 → 立刻量化成 P 模式 → 丢掉 RGB」流式处理，
    峰值内存只有单帧成本。
    """
    if not frames:
        raise ValueError("没有帧可以写入 GIF")
    from PIL import Image

    scale = GIF_SCALE
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_small(frame: np.ndarray) -> "Image.Image":
        img = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
        if scale > 1:
            img = img.resize(
                (max(1, img.width // scale), max(1, img.height // scale)), Image.BOX
            )
        return img

    sample = min(16, len(frames))
    idx = np.linspace(0, len(frames) - 1, sample).round().astype(int)
    picks = [to_small(frames[int(i)]) for i in np.unique(idx)]
    cols = 4
    rows = (len(picks) + cols - 1) // cols
    tw, th = max(1, picks[0].width // 4), max(1, picks[0].height // 4)
    mosaic = Image.new("RGB", (tw * cols, th * rows))
    for i, img in enumerate(picks):
        mosaic.paste(img.resize((tw, th), Image.BOX), ((i % cols) * tw, (i // cols) * th))
    palette = mosaic.quantize(colors=256, method=Image.MEDIANCUT)
    del picks, mosaic

    images = [to_small(frame).quantize(palette=palette, dither=Image.FLOYDSTEINBERG)
              for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(20, int(frame_ms)),
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
    success: bool = False
    fell: bool = False
    timeout: bool = False
    #: 本回合**钉住**的那条指令（GIF 的 HUD 要显示它，否则看画面的人无法判断
    #: 「它没动」是策略的问题还是指令本来就是 0）。
    cmd_vx: float = 0.0
    cmd_vy: float = 0.0
    cmd_yaw: float = 0.0
    #: 回合末的滤波实测值（与奖励里用的同一个量，见 ``robotdog/reward.py``）。
    act_vy: float = 0.0
    act_yaw: float = 0.0
    #: 本回合登上的最高台阶级数（0 = 未踏上任何一级；平地为 0）。
    steps_mounted: int = 0
    #: 帧间隔 [ms] = 取帧步幅 × 控制周期 —— GIF 按真实仿真时间回放
    #: （内存上限把取帧稀疏化时，帧间隔相应加大，播放不再被动加快）。
    frame_ms: int = 40
    frames: list[np.ndarray] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.success:
            return "成功"
        return "摔倒" if self.fell else "超时"


def run_episode(env: Go2StairsEnv, policy: Runner, *, episode: int, seed: int,
                capture: bool, camera: str, width: int, height: int,
                command: tuple[float, float, float] | None = None,
                max_frames: int = 0) -> EpisodeOutcome:
    """跑一个回合；``capture=True`` 时每 ``FRAME_STRIDE`` 步取一帧。

    ``max_frames`` 给定时按它反算稀疏步幅（渲染帧很大，不设上限曾把内存吃光）；
    ``0`` 表示不限制。

    ⭐ 回合开始后**必须把指令钉住**，且 :meth:`Go2StairsEnv.set_command_now` 要在
    ``reset()`` **之后**调 —— reset 会按量程重抽、覆盖之前写入的指令；钉住之后
    回放是确定的。``reset()`` 返回的 obs 是钉指令之前算的，要用
    :meth:`Go2StairsEnv._observation` 重取一份。
    """
    obs, _ = env.reset(seed=seed)
    cmd = env.cfg.command
    vx, vy, yaw = (command if command is not None
                   else (float(cmd.vx), float(cmd._now[1]), float(cmd._now[2])))
    env.set_command_now(vx=float(vx), vy=float(vy), yaw_rate=float(yaw), hold=True)
    obs = env._observation()
    x0, z0 = float(env.base_pos[0]), float(env.base_pos[2])
    max_steps = int(env.cfg.episode_length_s / env.cfg.control_dt)
    take = max(FRAME_STRIDE, -(-max_steps // max_frames)) if max_frames > 0 else FRAME_STRIDE
    # 帧间隔 = 取帧步幅 × 控制周期 → GIF 按真实仿真时间回放。
    frame_ms = max(20, int(round(take * env.cfg.control_dt * 1000)))
    out = EpisodeOutcome(episode=episode, steps=0, cmd_vx=float(vx),
                         cmd_vy=float(vy), cmd_yaw=float(yaw), frame_ms=frame_ms)
    info: dict[str, Any] = {}
    terminated = truncated = False

    step = 0
    while not (terminated or truncated) and step < max_steps:
        obs, _r, terminated, truncated, info = env.step(policy.predict(obs))
        step += 1
        if capture and step % take == 0:
            out.frames.append(env.render(camera, width=width, height=height).copy())

    dt = step * env.cfg.control_dt
    out.steps = step
    out.distance = round(float(env.base_pos[0]) - x0, 3)
    out.height_gain = round(float(env.base_pos[2]) - z0, 3)
    out.speed = round(out.distance / dt, 3) if dt > 0 else 0.0
    out.act_vy = round(float(env.filtered_lateral_vel), 3)
    out.act_yaw = round(float(env.filtered_yaw_rate), 3)
    out.steps_mounted = int(info.get("steps_mounted", 0))
    out.success = bool(info.get("is_success", False))
    out.fell = bool(terminated) and not out.success
    out.timeout = bool(truncated) and not out.success
    return out


def hud_for(out: EpisodeOutcome, total: int, stair, *, frame_index: int,
            frame_total: int) -> list[str]:
    """生成一帧上的 HUD 文本。

    指令那一行是**必须**的：没有它，看 GIF 的人无法区分「它站着不动」和
    「指令本来就是 0」，也无法判断横移/自转跟得准不准。括号里是回合**末**的
    滤波实测值（回合内的瞬时值会跳，写在 HUD 上反而误导）。

    平地阶段（``stair.step_height == 0``）打 ``台阶 0.0cm × 0 级`` 是纯噪声，
    所以那一行换成地形名。
    """
    if stair is None or float(stair.step_height) <= 0.0:
        terrain = "地形 平地（无台阶）  踏面 50cm"
    else:
        terrain = (f"台阶 {stair.step_height * 100:.1f}cm × {stair.n_up} 级"
                   f"  踏面 {stair.tread * 100:.0f}cm  顶端 {stair.crest_height:.2f}m")
    lines = [
        f"回合 {out.episode + 1}/{total}  {out.status}",
        terrain,
    ]
    if stair is not None and float(stair.step_height) > 0.0:
        lines.append(f"登踏 {out.steps_mounted}/{int(stair.n_up)} 级")
    lines += [
        f">指令 vx {out.cmd_vx:+.2f}  vy {out.cmd_vy:+.2f}  wz {out.cmd_yaw:+.2f} m/s, rad/s"
        f"（实测 vy {out.act_vy:+.2f}  wz {out.act_yaw:+.2f}）",
        f"前进 {out.distance:+.2f}m  速度 {out.speed:.2f}m/s  帧 {frame_index}/{frame_total}",
    ]
    return lines


# --------------------------------------------------------------------------- #
# 模式
# --------------------------------------------------------------------------- #


def mode_gif(env: Go2StairsEnv, policy: Runner, *, episodes: int, seed: int,
             name: str, play: dict[str, Any], gif_dir: Path | None = None,
             command: tuple[float, float, float] | None = None) -> None:
    """模式 1：连续渲染 ``episodes`` 个回合，合成一个 GIF。

    ``gif_dir`` 决定产物落点。**每个课程阶段一个子目录**（默认
    ``runs/<run>/gifs/<批次>/<阶段>/<阶段>.gif``）—— 批次在外层把两次训练分开，
    阶段在内层，否则十来个「地形 × 指令」的 GIF 平铺在一起根本认不出哪个是哪个。
    """
    camera = CAMERA_GIF
    width, height = FRAME_WIDTH, FRAME_HEIGHT
    max_frames = int(play.get("gif_max_frames", 0) or 0)

    frames: list[np.ndarray] = []
    outcomes: list[EpisodeOutcome] = []
    for ep in range(episodes):
        out = run_episode(env, policy, episode=ep, seed=seed + ep, capture=True,
                          camera=camera, width=width, height=height,
                          command=command, max_frames=max_frames)
        outcomes.append(out)
        total = len(out.frames)
        for i, frame in enumerate(out.frames):
            frames.append(draw_hud(frame, hud_for(out, episodes, env.stair,
                                                  frame_index=i + 1,
                                                  frame_total=total)))
        print(f"  回合 {ep + 1:>2}/{episodes}  {out.status}  "
              f"{out.steps:>4} 步  前进 {out.distance:+.2f}m  帧 {total}  "
              f"指令 vx{out.cmd_vx:+.2f}/vy{out.cmd_vy:+.2f}/wz{out.cmd_yaw:+.2f}  "
              f"实测 vy{out.act_vy:+.2f}/wz{out.act_yaw:+.2f}", flush=True)

    gif = (gif_dir or paths.OUTPUT_GIFS_DIR) / f"{name}.gif"
    frame_ms = outcomes[0].frame_ms if outcomes else 40
    write_gif(gif, frames, frame_ms=frame_ms)
    ok = sum(o.success for o in outcomes)
    size_mb = gif.stat().st_size / 1e6
    print(f"\nGIF: {gif}  ({len(frames)} 帧, {size_mb:.1f} MB, "
          f"{1000.0 / frame_ms:.1f} fps = 实时)")
    print(f"成功率: {ok}/{episodes} = {100.0 * ok / max(1, episodes):.1f}%")


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
    ap.add_argument("--stage", default=None,
                    help="gif 模式只渲染这一个课程阶段（名字或高度 cm）；"
                         "缺省渲染课程的全部级")
    ap.add_argument("--batch", default=None,
                    help="批次标签，覆盖 config.yaml 的 play.batch；"
                         "产物落 runs/<run>/gifs/<批次>/<阶段>/<阶段>.gif")
    ap.add_argument("--gifs-dir", default=None,
                    help="GIF 输出根目录（覆盖 runs/<run>/gifs）")
    return ap.parse_args(argv)


def _default_height_cm(play: dict[str, Any], target: PlayTarget) -> float | None:
    """播放用哪条楼梯。

    优先 ``config.yaml`` 的 ``play.stage_height_cm``（显式覆盖）；否则读权重旁
    ``<模型>_env.json`` 记录的该 checkpoint 训练时台阶高（每个 checkpoint 的地形
    都不一样）；都没有返回 ``None``，落到裸 ``StairSpec()`` 的默认（5 cm × 8 级）。
    """
    if play.get("stage_height_cm") is not None:
        return float(play["stage_height_cm"])
    height = step_height_from_model(target.model_path)
    if height is not None:
        print(f"地形来自权重自带记录: 台阶 {height * 100:.1f}cm")
        return height * 100.0
    return None


#: 渲染一个阶段所需的一切。``label`` 用来给产物目录命名，所以必须是文件系统
#: 友好且人类可读的。
RenderStage = namedtuple("RenderStage", "name label height_cm command spec")
#: ``spec`` 是**训练记录里的那一级**（``summary.json`` 的逐级下发值），用来把该级的
#: 指令量程与平地成功判据装回环境；``None`` = 没有记录，只能沿用 config.yaml 的基线。


def _run_root(run: str, target: PlayTarget) -> Path | None:
    """``--run`` 指向的那个运行目录（找不到返回 ``None``）。"""
    cand = Path(run)
    if cand.is_dir():
        return cand
    if cand.is_absolute() or "/" in run or "\\" in run:
        return cand if cand.is_dir() else None
    got = paths.RUNS_DIR / run
    return got if got.is_dir() else None


def _gif_dir_for(gifs_root: Path, stage: RenderStage) -> Path:
    """一个阶段一个子目录，阶段名就是 GIF 文件名。

    目录结构是 ``gifs/<批次>/<阶段>/<阶段>.gif``：**批次在外层**，这样两次训练
    的产物天然分开、可以并排比较；阶段在内层，避免十来个「地形 × 指令」的 GIF
    平铺在一个目录里认不出哪个是哪个。
    """
    return gifs_root / stage.label


def _stages_from_run(run_root: Path | None) -> list[dict[str, Any]]:
    """从运行目录 ``summary.json`` 读「训练时真正跑过哪些阶段、每级什么口径」。

    不按当前 config.yaml 对阶段名：改过课程再看老运行会静默漏渲染。训练记录里的
    名字、高度、逐级口径都是当时的真值。读不到返回空列表。
    """
    raw = read_json_or(None if run_root is None else run_root / "summary.json", {})
    return [st for st in (raw.get("stages", []) or []) if st.get("name")]


def _plan_entries(run_root: Path | None) -> dict[str, dict[str, Any]]:
    """读 ``runs/<run>/plan.json``：每一级**实际生效**的指令量程与奖励覆盖。

    它优先于 ``summary.json``（后者从阶段表抄来；这是唯一记了逐级
    ``reward_overrides`` 的地方）。旧运行没有这个文件，调用方必须能退回。
    """
    raw = read_json_or(None if run_root is None else run_root / "plan.json", None)
    if isinstance(raw, list):           # 早期形状：带 name 字段的列表
        return {str(x["name"]): x for x in raw if isinstance(x, dict) and x.get("name")}
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    return {}


def _spec_from_json(obj: Any) -> StageSpec | None:
    """按训练记录里的**逐级下发值**重建 :class:`StageSpec`（只用来查口径）。

    地形仍由权重自带的 ``env.json`` 决定，这里 ``spec`` 字段只是占位。
    返回 ``None`` 表示记录缺这些字段（老运行），调用方应退回当前课程并**明确
    告诉用户**这是推断值。
    """
    if "command_vy_max" not in obj:
        return None
    try:
        raw_vx = obj.get("command_vx", (0.0, 0.0))
        if isinstance(raw_vx, (int, float)):
            vx_range = (0.0, float(raw_vx))
        else:
            vx_range = (float(raw_vx[0]), float(raw_vx[1]))
        return StageSpec(
            name=str(obj.get("name") or obj.get("stage") or "?"),
            step_height=float(obj.get("step_height", 0.0)),
            spec=stair_from_height(float(obj.get("step_height", 0.0))),
            timesteps=int(obj.get("timesteps", 0)),
            learning_rate=float(obj.get("learning_rate", 0.0)),
            label=str(obj.get("label", "")),
            command_vy_max=float(obj.get("command_vy_max", 0.0)),
            command_yaw_max=float(obj.get("command_yaw_max", 0.0)),
            command_vx=vx_range,
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _same_command(a: StageSpec | None, b: StageSpec) -> bool:
    """两级的「回放口径」是否一致（三轴指令量程）。"""
    if a is None:
        return False
    return (abs(a.command_vy_max - b.command_vy_max) < 1e-9
            and abs(a.command_yaw_max - b.command_yaw_max) < 1e-9
            and all(abs(x - y) < 1e-9 for x, y in zip(a.command_vx, b.command_vx)))


def _stage_plan(cfg: dict[str, Any], play: dict[str, Any], target: PlayTarget,
                run: str, only: str | None) -> list[RenderStage]:
    """``gif`` 模式要渲染哪些阶段（地形 × 指令）。

    ⭐ **平地级必须在内** —— 它们带横向/偏航指令，是判断「跟不跟指令」的唯一
    画面证据，只渲染台阶级会漏掉指令跟踪这一半能力。

    ``play.stages``：``auto``（缺省）优先用运行目录的训练记录（summary.json +
    plan.json 的逐级下发值，都是当时的真值），没有记录退回当前课程并打印 ⚠；
    ``curriculum`` 强制全部级；其他 = 阶段名或高度列表。

    ⭐ **回放口径必须来自训练记录**：逐级量程是课程下发的，拿当前 config.yaml
    的基线去评老运行，成败判定与训练时不是同一件事（experience.md §17）。
    """
    stages = config_to_stages(cfg)
    by_name = {st.name: st for st in stages}
    by_height: dict[float, StageSpec] = {}
    for st in stages:
        by_height.setdefault(round(st.step_height, 6), st)
    spec = play.get("stages", "auto")

    # 每一级：(名字, 台阶高 m, 用于取口径的 StageSpec 或 None, 口径来源说明)
    picked: list[tuple[str, float, StageSpec | None, str]] = []

    def pick_one(key: str) -> tuple[str, float, StageSpec | None, str] | None:
        key = key.strip()
        for st in stages:
            if st.name == key or f"{st.step_height * 100:g}" == key:
                return (st.name, st.step_height, st, "当前课程同名级")
        if key.startswith("flat"):
            return (key, 0.0, None, "无（当前课程没有这一级，指令取 0）")
        got = re.fullmatch(r"(?:stair)?(\d+(?:\.\d+)?)", key)
        if got:
            h = float(got.group(1)) / 100.0
            return (key, h, None, "无（高度取自阶段名，指令取 0）")
        return None

    if only:
        hit = pick_one(only)
        if hit is None:
            raise SystemExit(f"--stage {only!r} 不在课程里；可选: "
                             + ", ".join(st.name for st in stages))
        picked = [hit]
    elif spec == "curriculum":
        picked = [(st.name, st.step_height, st, "当前课程") for st in stages]
    elif spec == "auto":
        run_root = _run_root(run, target)
        trained = _stages_from_run(run_root)
        have = [st for st in stages
                if run_root is not None
                and (run_root / "checkpoints" / f"{st.name}_final.zip").exists()]
        if trained:
            heads = ", ".join(f"{st['name']}({float(st.get('step_height', 0.0)) * 100:g}cm)"
                              for st in trained)
            print(f"阶段来自运行目录 summary.json: {heads}")
            # ``plan.json`` 是逐级下发值的权威记录（含 reward_overrides），
            # 用它覆盖同名级的字段；旧运行没有这个文件，就纯用 summary.json。
            plans = _plan_entries(run_root)
            if plans:
                print(f"  逐级下发值取自 plan.json（{len(plans)} 级）")
            for obj in trained:
                if plans.get(str(obj["name"])):
                    merged = dict(obj)
                    merged.update({k: v for k, v in plans[str(obj["name"])].items()
                                   if k != "stage"})
                    obj = merged
                name = str(obj["name"])
                height = float(obj.get("step_height", 0.0))
                recorded = _spec_from_json(obj)
                ref = by_name.get(name) or by_height.get(round(height, 6))
                if recorded is not None:
                    if _same_command(recorded, ref) if ref is not None else True:
                        picked.append((name, height, recorded,
                                       "训练记录（summary.json 逐级下发值）"))
                    else:
                        print(f"  ⚠ {name}: 当前 config.yaml 的口径与训练记录不同"
                              f"（训练 vy±{recorded.command_vy_max:g}"
                              f"/wz±{recorded.command_yaw_max:g}"
                              f"/vx{recorded.command_vx[0]:g}~"
                              f"{recorded.command_vx[1]:g}；"
                              f"配置 vy±{ref.command_vy_max:g}"
                              f"/wz±{ref.command_yaw_max:g}"
                              f"/vx{ref.command_vx[0]:g}~"
                              f"{ref.command_vx[1]:g}）"
                              f" —— **以训练记录为准**")
                        picked.append((name, height, recorded, "训练记录（同上级）"))
                else:
                    src = ("当前课程同名级（旧运行没有记口径）" if ref is not None
                           else "无（指令取 0）")
                    picked.append((name, height, ref, src))
        elif have:
            print("阶段来自运行目录的检查点: " + ", ".join(st.name for st in have))
            picked = [(st.name, st.step_height, st, "当前课程同名级") for st in have]
        else:
            picked = [(st.name, st.step_height, st, "当前课程") for st in stages]
    else:
        wanted = [str(x) for x in spec] if isinstance(spec, (list, tuple)) else [str(spec)]
        missing: list[str] = []
        for w in wanted:
            hit = pick_one(w)
            if hit is None:
                missing.append(w)
            else:
                picked.append(hit)
        if missing:
            raise SystemExit("play.stages 里有课程中不存在的阶段: " + ", ".join(missing))

    out: list[RenderStage] = []
    for name, height, ref, src in picked:
        # 逐级用该级自己的量程挑一条指令（量程为 0 的方向就是 0）。
        vx, vy, wz = command_for_stage(ref) if ref is not None else (0.0, 0.0, 0.0)
        label = name if height <= 0.0 else f"{name}_{height * 100:g}cm"
        if ref is None:
            print(f"  ⚠ {name}: {src}")
        out.append(RenderStage(name, label, height * 100.0, (vx, vy, wz), ref))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    console.setup_utf8_console()
    args = parse_args(argv)
    cfg = load_config(args.config)
    play = config_to_play(cfg)
    target = resolve_target(args.run)
    # 设备取自 ppo.device（与训练同一个入口），不是 play 段 —— play 段没有这个键。
    dev = device(cfg)
    print(f"模型: {target.model_path}")
    print(f"归一化: {target.norm_path or '（无）'}")

    if args.mode != "gif":
        stair = stair_from_height(_default_height_cm(play, target))
        policy, env, _venv = open_runner(target, stair=stair, device=dev)
        try:
            print(f"地形: 台阶 {env.stair.step_height * 100:.1f}cm × {env.stair.n_up} 级"
                  f"  踏面 {env.stair.tread * 100:.0f}cm")
            print(f"观测: {env.observation_space.shape}  动作: {env.action_space.shape}")
            return mode_eval(
                env, policy,
                episodes=int(play["episodes_eval"]),
                seed=int(play["seed"]),
                model_path=target.model_path,
            )
        finally:
            env.close()

    # ---- gif：一个批次一个根目录、一个阶段一个文件 ------------------------- #
    batch = (args.batch if args.batch is not None else str(play.get("batch", ""))).strip()
    batch = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]", "_", batch)
    if args.gifs_dir:
        gifs_root = Path(args.gifs_dir)
        if batch:
            gifs_root = gifs_root / batch
    else:
        run_root = _run_root(args.run, target)
        gifs_root = (run_root / "gifs") if run_root else paths.OUTPUT_GIFS_DIR
        if batch:
            gifs_root = gifs_root / batch
    if batch:
        print(f"批次: {batch}")

    plan = _stage_plan(cfg, play, target, args.run, args.stage)
    print(f"阶段: {len(plan)} 个 —— "
          + ", ".join(f"{s.name}({s.height_cm:g}cm)" for s in plan))
    episodes = int(play["episodes_gif"])
    seed = int(play["seed"])
    for i, stage in enumerate(plan):
        gif_dir = _gif_dir_for(gifs_root, stage)
        print(f"\n[{i + 1}/{len(plan)}] {stage.label}  "
              f"台阶 {stage.height_cm:g}cm  "
              f"指令 vx{stage.command[0]:+.2f}/vy{stage.command[1]:+.2f}"
              f"/wz{stage.command[2]:+.2f}")
        stair = stair_from_height(stage.height_cm)
        policy, env, _venv = open_runner(target, stair=stair, device=dev)
        try:
            # ⭐ 把这一级的**训练口径**装回环境：指令量程与平地成功判据都是课程逐级
            # 下发的，不回灌的话回放会拿 config.yaml 的基线去判成功/失败，GIF 的
            # 结局标签就是假的（曾把「跟指令」的级按「前进 N 米」评成 20%）。
            if stage.spec is not None:
                env.set_commands(vy_max=stage.spec.command_vy_max,
                                 yaw_max=stage.spec.command_yaw_max,
                                 vx_min=stage.spec.command_vx[0],
                                 vx_max=stage.spec.command_vx[1])
            print(f"  观测: {env.observation_space.shape}  动作: {env.action_space.shape}")
            mode_gif(env, policy, episodes=episodes, seed=seed,
                     name=stage.label, play=play, gif_dir=gif_dir,
                     command=stage.command)
        finally:
            env.close()
    print(f"\n全部 GIF 输出到: {gifs_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
