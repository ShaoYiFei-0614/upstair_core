"""把 checkpoint 回放成 GIF —— 「训练效果」的肉眼版证据。

`scripts/eval.py` 给出的是数字（成功率/速度/姿态），这个脚本给出的是画面：
加载一个 checkpoint，让它在指定楼梯上跑几个回合，把每帧离屏渲染出来，
叠上 HUD 文字，编码成动画 GIF，全部写到 ``output/``。

用法示例
--------
::

    # ① 课程每一档各出一段 GIF（最直观地看到「越训越强」）
    .venv\\Scripts\\python.exe scripts\\render_checkpoint.py --run course8 --all-stages --episodes 3

    # ② 只看某一档，换追尾视角、前半速
    .venv\\Scripts\\python.exe scripts\\render_checkpoint.py --run course8 --stage easy3 --n-up 5 --camera chase

    # ③ 最终模型在**没见过**的楼梯上（第九步的泛化性）
    .venv\\Scripts\\python.exe scripts\\render_checkpoint.py --run course8 --unseen --episodes 2

    # ④ 平地行走 / 指定台阶
    .venv\\Scripts\\python.exe scripts\\render_checkpoint.py --run course8 --flat
    .venv\\Scripts\\python.exe scripts\\render_checkpoint.py --run course8 --step-height-cm 6 --n-up 8

    # ⑤ 只要 GIF（第二十五步起 ``output`` 的交付件只有 GIF 一种）
    .venv\\Scripts\\python.exe scripts\\render_checkpoint.py --run course8 --all-stages --no-strip --no-report

产物
----
::

    output/gifs/<名字>.gif            动画（默认半尺寸，可直接贴进 README）——唯一交付件
    output/stills/<名字>_strip.png     12 帧横向拼图（静态预览；--no-strip 不写）
    output/frames/<名字>/f0000.png …   逐帧 PNG（只在 --save-frames 时写）
    output/reports/render_<名字>.json  每回合的步数/前进距离/成功与否（--no-report 不写）

后面三个都是**中间产物**：`output/` 的交付物只有 `gifs/`，所以这些目录只在真的写
东西时才被创建；批量重渲交付 GIF 时建议直接加 `--no-strip --no-report`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import robotdog  # noqa: F401  (installs the UTF-8 console wrapper)
from robotdog import paths
from robotdog.env import Go2StairsEnv
from robotdog.evaluate import (
    randomized_terrains,
    trained_terrains,
    unseen_terrains,
)
from robotdog.png import downscale, write_png
from robotdog.render import (
    RenderOptions,
    contact_sheet,
    hud_lines,
    overlay_hud,
    render_rollout,
    save_frames,
    save_gif,
)
from robotdog.terrain import StairSpec, flat_spec
from robotdog.training import env_config_for_model, format_duration


# --------------------------------------------------------------------------- #
# Checkpoint / terrain resolution
# --------------------------------------------------------------------------- #


def load_run_summary(run: str | Path) -> dict:
    """Read ``runs/<run>/summary.json`` (empty dict when absent)."""
    root = Path(run)
    if not root.is_absolute():
        root = paths.RUNS_DIR / run
    summary = root / "summary.json"
    if not summary.exists():
        return {}
    return json.loads(summary.read_text(encoding="utf-8"))


def run_dir_of(run: str | Path) -> Path:
    root = Path(run)
    return root if root.is_absolute() else paths.RUNS_DIR / root


def stage_terrains(summary: dict) -> list[tuple[str, StairSpec]]:
    """Rebuild each sub-stage's terrain from ``summary.json``.

    ``n_up`` is recorded by :meth:`StageSpec.result`; older runs that predate
    that field fall back to the ``--n-up-fallback`` value.
    """
    out: list[tuple[str, StairSpec]] = []
    for stage in summary.get("stages", []):
        name = str(stage.get("stage", "stage"))
        height = float(stage.get("step_height", 0.05))
        n_up = int(stage.get("n_up", 0) or 0)
        out.append((name, StairSpec(
            step_height=height,
            tread=float(stage.get("tread", 0.50)),
            n_up=n_up,
            n_down=0,
        )))
    return out


def resolve_checkpoint(args: argparse.Namespace, stage: str | None = None) -> tuple[Path, Path]:
    """Return ``(model.zip, vecnormalize.pkl)`` for the requested checkpoint."""
    if args.checkpoint:
        model = Path(args.checkpoint)
        if not model.is_absolute():
            model = paths.WORKSPACE / model
    else:
        root = run_dir_of(args.run)
        if stage:
            model = root / "checkpoints" / f"{stage}_final.zip"
        else:
            model = root / f"model_{args.model}.zip"
    if not model.exists():
        raise SystemExit(f"找不到 checkpoint: {model}")
    if args.vecnormalize:
        vn = Path(args.vecnormalize)
        if not vn.is_absolute():
            vn = paths.WORKSPACE / vn
    else:
        vn = model.with_name(model.name[: -len(model.suffix)] + "_vecnormalize.pkl")
    return model, vn


def terrain_name(stair: StairSpec, prefix: str, tag: str = "") -> str:
    """``<run>_<tag>_<height>cm_x<n>``; the tag keeps experimental renders from
    silently overwriting a delivered GIF (same idea as ``eval.py --export-name``)."""
    stem = f"{prefix}_{tag}" if tag else prefix
    if stair.step_height <= 0 or stair.n_up == 0:
        return f"{stem}_flat"
    return f"{stem}_{stair.step_height * 100:.1f}cm_x{stair.n_up}"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


class PriorPolicy:
    """动作恒 0 = 只剩开环 trot 先验在控制（训练的对照组）。"""

    def __init__(self, action_dim: int = 12) -> None:
        self.action_dim = action_dim

    def predict(self, obs, deterministic: bool = True):
        return np.zeros(self.action_dim, dtype=np.float32), None


def labelled_frames(result, *, title: str, stair: StairSpec,
                    args: argparse.Namespace, opts: RenderOptions):
    """Build ``(gif_frames, strip_frames)`` from a rollout result.

    Two frame sets on purpose:

    * ``gif_frames`` is downscaled **first** and labelled **after**, so the HUD
      text is rasterised at the GIF's real pixel size and stays crisp.
    * ``strip_frames`` keeps full resolution for the static preview.
    """
    gif_frames: list[np.ndarray] = []
    strip_frames: list[np.ndarray] = []
    compact = args.height // max(1, opts.scale) < 400
    for i, frame in enumerate(result.frames):
        ep = result.episode_of_frame(i)
        episode = result.episodes[ep] if ep < len(result.episodes) else None
        for f in ([frame] if opts.scale == 1 else [downscale(frame, opts.scale)]):
            lines = hud_lines(
                title=title, stair=stair, episode=episode, index=i,
                total=len(result.frames), compact=compact,
            )
            gif_frames.append(overlay_hud(f, lines))
        lines = hud_lines(title=title, stair=stair, episode=episode,
                          index=i, total=len(result.frames))
        strip_frames.append(overlay_hud(frame, lines))
    return gif_frames, strip_frames


def _options_for(args: argparse.Namespace, title: str, cfg, *,
                 width: int | None = None) -> RenderOptions:
    return RenderOptions(
        width=width or args.width,
        height=args.height,
        fps=args.fps,
        frame_stride=args.stride,
        camera=args.camera,
        scale=max(1, args.scale),
        overlay=not args.no_overlay,
        title=title,
        max_steps=(int(args.max_seconds / cfg.control_dt) if args.max_seconds else None),
    )


def render_one(model_path: Path, vn_path: Path, stair: StairSpec, name: str,
               title: str, args: argparse.Namespace) -> dict:
    """Load a checkpoint, roll it out, and write the GIF + strip + JSON."""
    from scripts.eval import SingleEnvPredict, load_policy

    cfg = env_config_for_model(model_path)
    model, venv = load_policy(model_path, vn_path, device=args.device, cfg=cfg)
    predictor = SingleEnvPredict(model, venv)

    opts = _options_for(args, title, cfg)
    env = Go2StairsEnv(stair=stair, cfg=cfg, seed=args.seed)
    try:
        result = render_rollout(env, predictor, episodes=args.episodes,
                                seed=args.seed, opts=opts, stair=stair,
                                deterministic=not args.stochastic)
        gif_frames, strip_frames = labelled_frames(
            result, title=title, stair=stair, args=args, opts=opts)
    finally:
        env.close()

    if not gif_frames:
        print(f"  [!] {name}: 一个帧都没有生成, 跳过")
        return {"name": name, "skipped": True}

    paths.ensure_output_dirs()
    gif_path = save_gif(paths.OUTPUT_GIFS_DIR / f"{name}.gif", gif_frames,
                        fps=args.fps, scale=1, dither=not args.no_dither)
    strip_path = None
    if not args.no_strip:
        strip_path = paths.OUTPUT_STILLS_DIR / f"{name}_strip.png"
        paths.ensure_dir(strip_path.parent)
        write_png(strip_path, contact_sheet(strip_frames, n=12,
                                            scale=max(1, args.strip_scale)))

    frame_dir = None
    if args.save_frames:
        written = save_frames(paths.OUTPUT_FRAMES_DIR / name, gif_frames)
        frame_dir = str(written[0].parent)

    report = {
        "name": name,
        "title": title,
        "checkpoint": str(model_path),
        "vecnormalize": str(vn_path),
        "terrain": {
            "step_height_cm": round(stair.step_height * 100, 3),
            "tread_cm": round(stair.tread * 100, 2),
            "n_up": int(stair.n_up),
            "crest_height_m": round(float(stair.crest_height), 4),
        },
        "camera": args.camera,
        "fps": args.fps,
        "frame_stride": args.stride,
        "gif": str(gif_path.relative_to(paths.WORKSPACE)),
        "gif_bytes": gif_path.stat().st_size,
        "strip": None if strip_path is None else str(strip_path.relative_to(paths.WORKSPACE)),
        "frames_dir": frame_dir,
        **result.summary(),
    }
    if not args.no_report:
        report_path = paths.OUTPUT_REPORTS_DIR / f"render_{name}.json"
        paths.ensure_dir(report_path.parent)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")

    ok = f"{result.successes}/{len(result.episodes)} 回合成功"
    print(f"  [OK] {name}: {len(gif_frames)} 帧, {ok}, GIF {gif_path.stat().st_size / 1024:.0f} KB"
          f" -> {gif_path.relative_to(paths.WORKSPACE)}")
    return report


def render_compare(model_path: Path, vn_path: Path, stair: StairSpec, name: str,
                   args: argparse.Namespace) -> dict:
    """左＝训练后的策略, 右＝纯开环先验（动作恒 0）, 同一地形同一种子。

    这是「训练到底有没有用」最直观的证据：两边同时起步、同样的扰动，
    先验往往在几秒内摔下去，策略则继续向上爬。
    """
    from scripts.eval import SingleEnvPredict, load_policy
    from robotdog.png import hstack

    cfg = env_config_for_model(model_path)
    model, venv = load_policy(model_path, vn_path, device=args.device, cfg=cfg)
    half = max(160, args.width // 2)

    def run(predictor, title: str):
        opts = _options_for(args, title, cfg, width=half)
        env = Go2StairsEnv(stair=stair, cfg=cfg, seed=args.seed)
        try:
            res = render_rollout(env, predictor, episodes=args.episodes,
                                 seed=args.seed, opts=opts, stair=stair,
                                 deterministic=not args.stochastic)
            gif_frames, strip_frames = labelled_frames(
                res, title=title, stair=stair, args=args, opts=opts)
        finally:
            env.close()
        return res, gif_frames, strip_frames

    res_p, gif_p, strip_p = run(SingleEnvPredict(model, venv),
                                f"{args.run} · 训练后策略")
    res_z, gif_z, strip_z = run(PriorPolicy(), f"{args.run} · 纯先验(动作=0)")

    if not gif_p or not gif_z:
        print(f"  [!] {name}: 有一边没有帧, 跳过")
        return {"name": name, "skipped": True}

    # 两边的回合长度不同：短的一边定格在最后一帧，长的一边继续播。
    n = max(len(gif_p), len(gif_z))
    frames = [hstack([gif_p[min(i, len(gif_p) - 1)], gif_z[min(i, len(gif_z) - 1)]],
                     pad=4, background=30) for i in range(n)]
    strips = [hstack([strip_p[min(i, len(strip_p) - 1)], strip_z[min(i, len(strip_z) - 1)]],
                     pad=4, background=30) for i in range(n)]

    paths.ensure_output_dirs()
    gif_path = save_gif(paths.OUTPUT_GIFS_DIR / f"{name}_vs_prior.gif", frames,
                        fps=args.fps, scale=1, dither=not args.no_dither)
    strip_path = None
    if not args.no_strip:
        strip_path = paths.OUTPUT_STILLS_DIR / f"{name}_vs_prior_strip.png"
        paths.ensure_dir(strip_path.parent)
        write_png(strip_path, contact_sheet(strips, n=10,
                                            scale=max(1, args.strip_scale)))

    def _side(res) -> dict:
        eps = res.episodes
        n = max(len(eps), 1)
        return {
            "successes": res.successes,
            "episodes": len(eps),
            "total_frames": len(res.frames),
            "mean_distance": round(sum(e.distance for e in eps) / n, 3),
            "mean_forward_speed": round(sum(e.forward_speed for e in eps) / n, 3),
            "mean_steps": round(sum(e.steps for e in eps) / n, 1),
            "outcomes": [e.status for e in eps],
        }

    report = {
        "name": name,
        "mode": "compare_prior",
        "checkpoint": str(model_path),
        "terrain": {
            "step_height_cm": round(stair.step_height * 100, 3),
            "tread_cm": round(stair.tread * 100, 2),
            "n_up": int(stair.n_up),
            "crest_height_m": round(float(stair.crest_height), 4),
        },
        "camera": args.camera,
        "fps": args.fps,
        "frame_stride": args.stride,
        "gif": str(gif_path.relative_to(paths.WORKSPACE)),
        "gif_bytes": gif_path.stat().st_size,
        "strip": None if strip_path is None else str(strip_path.relative_to(paths.WORKSPACE)),
        "total_frames": n,
        "policy": _side(res_p),
        "prior": _side(res_z),
    }
    if not args.no_report:
        report_path = paths.OUTPUT_REPORTS_DIR / f"render_{name}_vs_prior.json"
        paths.ensure_dir(report_path.parent)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")

    print(f"  [OK] {name}_vs_prior: {n} 帧, GIF {gif_path.stat().st_size / 1024:.0f} KB"
          f" -> {gif_path.relative_to(paths.WORKSPACE)}")
    print(f"       策略 {res_p.successes}/{len(res_p.episodes)} 成功 "
          f"前进 {report['policy']['mean_distance']:.2f} m  |  "
          f"先验 {res_z.successes}/{len(res_z.episodes)} 成功 "
          f"前进 {report['prior']['mean_distance']:.2f} m")
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="把 checkpoint 渲染成 GIF（输出到 output/）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run", default="course8",
                   help="runs/<名字>，例如 course8；也可直接给绝对路径")
    p.add_argument("--checkpoint", default="",
                   help="直接指定 .zip，覆盖 --run/--stage/--model")
    p.add_argument("--vecnormalize", default="", help="直接指定 _vecnormalize.pkl")
    p.add_argument("--model", default="best", choices=["best", "final"],
                   help="用 model_best.zip 还是 model_final.zip")
    p.add_argument("--stage", default="",
                   help="用 checkpoints/<stage>_final.zip，例如 easy3")
    p.add_argument("--all-stages", action="store_true",
                   help="课程里每一档都渲染一段（读 summary.json）")
    p.add_argument("--n-up-ladder", default="",
                   help="逗号分隔的级数梯级, 按档位顺序对应, 例如 3,4,5,6,7,8。"
                        "summary.json 里没有 n_up 的旧 run 必须给这个")
    p.add_argument("--n-up-fallback", type=int, default=5,
                   help="既没有 n_up 也没有 --n-up-ladder 时的级数")

    # 地形选择（互斥的一组开关，都不给 = 从 summary.json 取最好那一档）
    p.add_argument("--step-height-cm", type=float, default=None, help="台阶高（厘米）")
    p.add_argument("--name-tag", default="",
                   help="产物名里插一段标记（<run>_<tag>_<高>cm_x<级>），"
                        "用来做实验渲染而不覆盖交付 GIF")
    p.add_argument("--n-up", type=int, default=None, help="上楼梯级数")
    p.add_argument("--tread", type=float, default=0.50, help="踏面宽度（米）")
    p.add_argument("--flat", action="store_true", help="平地")
    p.add_argument("--unseen", action="store_true", help="9 个训练时没见过的楼梯")
    p.add_argument("--trained", action="store_true", help="3 个训练地形")
    p.add_argument("--random-terrains", type=int, default=0, help="额外随机地形个数")

    # 渲染
    p.add_argument("--episodes", type=int, default=2, help="每个地形渲染几个回合")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--stride", type=int, default=2,
                   help="每多少个控制步存一帧（控制 50Hz, stride=2 即实时）")
    p.add_argument("--camera", default="track",
                   choices=list(Go2StairsEnv.CAMERAS))
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--scale", type=int, default=2,
                   help="GIF 缩小倍数（2 = 半尺寸）；HUD 文字在缩小后才画, 所以仍然清晰")
    p.add_argument("--strip-scale", type=int, default=2,
                   help="静态拼图的缩小倍数（拼图用全分辨率帧, 所以这是独立的）")
    p.add_argument("--no-dither", action="store_true",
                   help="GIF 调色板不做抖动（体积更小, 但渐变会有色带）")
    p.add_argument("--no-overlay", action="store_true", help="不画 HUD 文字")
    p.add_argument("--save-frames", action="store_true", help="同时逐帧存 PNG")
    p.add_argument("--no-strip", action="store_true",
                   help="不写 output\\stills\\ 的静态拼图（第二十五步起 output\\ 只交付 GIF）")
    p.add_argument("--no-report", action="store_true",
                   help="不写 output\\reports\\render_*.json（逐回合统计，只是中间产物）")
    p.add_argument("--compare-prior", action="store_true",
                   help="左右并排对比：左边=训练后策略，右边=纯开环先验（动作恒 0）。"
                        "这是「训练有没有用」最直观的证据，建议配 --episodes 1")
    p.add_argument("--stochastic", action="store_true", help="用随机动作而不是确定性动作")
    p.add_argument("--max-seconds", type=float, default=0.0, help="每回合最多渲染多少秒（0=整个回合）")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def parse_ladder(text: str) -> list[int]:
    if not text.strip():
        return []
    return [int(x) for x in text.replace(" ", "").split(",") if x]


def stage_n_ups(summary: dict, args: argparse.Namespace) -> list[int]:
    """One ``n_up`` per sub-stage: recorded value, else the explicit ladder."""
    ladder = parse_ladder(args.n_up_ladder)
    recorded = [int(s.get("n_up", 0) or 0) for s in summary.get("stages", [])]
    out: list[int] = []
    for i, n in enumerate(recorded):
        if n > 0:
            out.append(n)
        elif i < len(ladder):
            out.append(ladder[i])
        elif ladder:
            out.append(ladder[-1])
        else:
            out.append(args.n_up_fallback)
    return out


def select_terrains(args: argparse.Namespace, summary: dict) -> list[tuple[str, StairSpec]]:
    """Decide which terrains to render, from the CLI flags."""
    if args.flat:
        return [("flat", flat_spec())]
    if args.trained:
        return trained_terrains()
    if args.unseen:
        return unseen_terrains()
    if args.random_terrains:
        return randomized_terrains(args.random_terrains, seed=args.seed)
    if args.step_height_cm is not None:
        n_up = args.n_up if args.n_up is not None else 8
        spec = StairSpec(step_height=args.step_height_cm / 100.0,
                         tread=args.tread, n_up=n_up, n_down=0)
        return [(f"{args.step_height_cm:.1f}cm_x{n_up}", spec)]

    stages = stage_terrains(summary)
    if stages:
        n_ups = stage_n_ups(summary, args)
        rebuilt: list[tuple[str, StairSpec]] = []
        for (name, spec), n in zip(stages, n_ups):
            if n != spec.n_up:
                spec = StairSpec(step_height=spec.step_height, tread=spec.tread,
                                 n_up=n, n_down=0)
            rebuilt.append((name, spec))
        stages = rebuilt
        # --stage 同时决定 checkpoint 和地形，否则 easy3 会被渲染到别的档位上
        if args.stage:
            for name, spec in stages:
                if name == args.stage:
                    return [(name, spec)]
        wanted = str(summary.get("best_stage") or "")
        for name, spec in stages:
            if name == wanted and spec.n_up > 0:
                return [(name, spec)]
        for name, spec in reversed(stages):
            if spec.n_up > 0:
                return [(name, spec)]
        return [stages[-1]]

    n_up = args.n_up if args.n_up is not None else 8
    return [(f"5.0cm_x{n_up}", StairSpec(step_height=0.05, tread=args.tread,
                                         n_up=n_up, n_down=0))]


def select_all_stages(summary: dict, args: argparse.Namespace) -> list[tuple[str, StairSpec]]:
    """Every sub-stage terrain, in curriculum order (for ``--all-stages``)."""
    stages = stage_terrains(summary)
    n_ups = stage_n_ups(summary, args)
    out: list[tuple[str, StairSpec]] = []
    for (name, spec), n in zip(stages, n_ups):
        if n != spec.n_up:
            spec = StairSpec(step_height=spec.step_height, tread=spec.tread,
                             n_up=n, n_down=0)
        out.append((name, spec))
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths.ensure_output_dirs()
    import time

    # 这个坑已经踩过两次：只给 --checkpoint（或忘了给 --run）时，产物名会沿用
    # 默认的 --run 值，于是别的 run 的交付 GIF 被静默覆盖，或者生成一个名字与
    # 内容都不对的「新」文件。默认值无从判断，只能看命令行里有没有显式出现。
    argv_given = sys.argv[1:] if argv is None else list(argv)
    if not any(a == "--run" or a.startswith("--run=") for a in argv_given):
        print(f"[!] 命令行里没有显式给 --run，产物会沿用默认的 {args.run!r}：")
        print(f"    GIF / 静态图 / 报告都会写成 {args.run}_* ，可能覆盖别的 run 的交付物。")
        print("    配合 --checkpoint 用别的 run 的模型时尤其危险，请补上 --run <那个 run 的名字>。")

    t0 = time.time()
    summary = load_run_summary(args.run)
    if summary:
        print(f"[run] {args.run}: {len(summary.get('stages', []))} 档, "
              f"总计 {summary.get('total_timesteps', 0):,} 步")
    else:
        print(f"[run] {args.run}: 没有 summary.json, 将用默认配置推断")

    out_parts = ["gifs/"]
    if not args.no_strip:
        out_parts.append("stills/")
    if args.save_frames:
        out_parts.append("frames/")
    if not args.no_report:
        out_parts.append("reports/")
    print(f"[输出] {paths.OUTPUT_DIR.relative_to(paths.WORKSPACE)}\\  "
          f"({' '.join(out_parts)})"
          + ("  [--no-strip/--no-report ⇒ 只交付 GIF]" if args.no_strip and args.no_report else ""))

    # 旧 run 的 summary.json 里没有记录 n_up（那是后加的字段）。级数直接决定地形
    # 难度，猜错就会把 easy5 渲染到 5 级而不是 7 级，所以必须显式提醒。
    if summary and not parse_ladder(args.n_up_ladder):
        # 平地段本来就没有台阶，n_up=0 是正确记录而不是缺失，所以只对
        # step_height > 0 的台阶档位报警（否则 course12 会为 flat 误报一次）。
        missing = [str(s.get("stage")) for s in summary.get("stages", [])
                   if (s.get("step_height") or 0) > 0 and not (s.get("n_up") or 0)]
        if missing:
            print(f"[!] {args.run}/summary.json 没有记录 n_up（{', '.join(missing)}）——"
                  f"地形级数会回退到 {args.n_up_fallback}。")
            print(f"    该 run 是用 --sub-n-up N --sub-steps M 训的，请显式给 "
                  f"--n-up-ladder 例如 --n-up-ladder 3,4,5,6,7,8")

    reports: list[dict] = []
    if args.all_stages:
        stages = select_all_stages(summary, args)
        if not stages:
            raise SystemExit("--all-stages 需要 runs/<run>/summary.json 里有 stages 列表")
        for name, spec in stages:
            if spec.n_up <= 0 and spec.step_height <= 0:
                spec = flat_spec()
            try:
                model_path, vn_path = resolve_checkpoint(args, name)
            except SystemExit as exc:
                print(f"  [跳过] {name}: {exc}")
                continue
            title = f"{args.run} · 课程档位 {name}"
            if args.compare_prior:
                reports.append(render_compare(model_path, vn_path, spec, name, args))
            else:
                reports.append(render_one(model_path, vn_path, spec, name, title, args))
    else:
        for label, spec in select_terrains(args, summary):
            name = terrain_name(spec, args.run, args.name_tag)
            model_path, vn_path = resolve_checkpoint(args, args.stage or None)
            stage_txt = f" · {args.stage}" if args.stage else ""
            title = f"{args.run}{stage_txt}"
            if args.compare_prior:
                reports.append(render_compare(model_path, vn_path, spec, name, args))
            else:
                reports.append(render_one(model_path, vn_path, spec, name, title, args))

    ok = [r for r in reports if not r.get("skipped")]
    total_frames = sum(r.get("total_frames", 0) for r in ok)
    print()
    print(f"[完成] {len(ok)} 段 GIF, 共 {total_frames} 帧, 用时 {format_duration(time.time() - t0)}")
    for r in ok:
        if r.get("mode") == "compare_prior":
            note = (f"策略 {r['policy']['successes']}/{r['policy']['episodes']} vs "
                    f"先验 {r['prior']['successes']}/{r['prior']['episodes']}")
        else:
            note = f"成功 {r.get('successes', 0)}/{len(r.get('episodes', []))} 回合"
        print(f"  {r['gif']:<44} {r['gif_bytes'] / 1024:7.0f} KB  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
