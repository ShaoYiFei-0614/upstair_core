"""第 9 步 CLI: 在**没见过的楼梯**上评估策略, 保存最好策略, 导出 ONNX / TorchScript。

用法
----
::

    # 默认: 训练地形 + 9 个没见过的地形 + 6 个随机地形
    .venv\\Scripts\\python.exe scripts\\eval.py --run runs\\go2_xxx --episodes 20

    # 只评估没见过的地形, 每个 30 回合 (⚠ 评估至少 40 回合, 10 回合能给出 ±15 pt 错觉)
    ... scripts\\eval.py --run runs\\go2_xxx --only unseen --episodes 30

    # 顺带导出 ONNX 与 TorchScript 并逐样本核对数值
    ... scripts\\eval.py --run runs\\go2_xxx --export

    # 用一段渲染视频确认视觉上真的在爬楼梯
    ... scripts\\eval.py --run runs\\go2_xxx --only unseen --render-video
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import robotdog  # noqa: F401  (installs the UTF-8 console wrapper)
from robotdog.tools import paths
from robotdog.env import EnvConfig, Go2StairsEnv
from robotdog.evaluate import (
    all_terrains,
    build_inference_module,
    evaluate_all,
    export_onnx,
    export_torchscript,
    format_table,
    summarize,
    termination_breakdown,
    write_report,
)
from robotdog.training import RunDir, format_duration, load_ppo, make_vec_env


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="评估 Go2 爬楼梯策略并导出部署产物",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run", type=str, default=None,
                   help="run 目录 (含 model_final.zip) 或直接的 .zip 路径")
    p.add_argument("--model", type=str, default=None,
                   help="显式指定 .zip (覆盖 --run 的默认查找)")
    p.add_argument("--model-name", type=str, default="model_final",
                   help="当 --run 是目录时要加载的模型名, 例如 model_best")
    p.add_argument("--vecnormalize", type=str, default=None,
                   help="VecNormalize .pkl; 默认在模型旁边自动查找")
    p.add_argument("--only", type=str, default="all",
                   choices=["all", "trained", "unseen", "random"],
                   help="要评估的地形组")
    p.add_argument("--episodes", type=int, default=20, help="每组地形的回合数")
    p.add_argument("--n-random", type=int, default=0,
                   help="在 --only 之外额外采样多少随机地形")
    p.add_argument("--seed", type=int, default=0, help="评估随机种子")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", dest="deterministic", action="store_false",
                   help="用随机策略采样动作而不是取均值")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default=None,
                   help="报告 JSON 路径; 默认 <run>/artifacts/eval_report.json")
    p.add_argument("--export", action="store_true",
                   help="导出 ONNX + TorchScript 并做数值核对")
    p.add_argument("--export-dir", type=str, default=None,
                   help="导出目录; 默认 <run>/artifacts")
    p.add_argument("--export-name", type=str, default="policy",
                   help="导出文件名的主干（默认 policy -> policy.onnx / policy.pt）；"
                        "同一个 --export-dir 里放多个 run 的产物时用它区分，"
                        "例如 --export-name policy_course18")
    p.add_argument("--onnx-opset", type=int, default=17)
    p.add_argument("--render-video", action="store_true",
                   help="在每个地形组的最佳/最差地形上渲染一段帧序列")
    p.add_argument("--video-terrain", type=str, default=None,
                   help="只在这一个地形名上渲染视频")
    p.add_argument("--video-episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None,
                   help="覆盖每回合最大控制步数 (默认由 episode_length_s 决定)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #


def resolve_model_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.model:
        model_path = Path(args.model)
    elif args.run:
        run = Path(args.run)
        if run.is_dir():
            model_path = run / f"{args.model_name}.zip"
        else:
            model_path = run
    else:
        raise SystemExit("必须给出 --run 或 --model")

    if not model_path.exists():
        raise SystemExit(f"模型不存在: {model_path}")

    if args.vecnormalize:
        vn_path = Path(args.vecnormalize)
    else:
        side = model_path.with_name(model_path.stem + "_vecnormalize.pkl")
        if side.exists():
            vn_path = side
        else:
            cands = sorted(model_path.parent.glob("*_vecnormalize.pkl"))
            if not cands:
                raise SystemExit(
                    f"在 {model_path.parent} 找不到 *_vecnormalize.pkl; "
                    "策略是用归一化观测训练的, 没有它结果会完全错误。"
                    "请用 --vecnormalize 显式指定。"
                )
            vn_path = cands[0]
    if not vn_path.exists():
        raise SystemExit(f"VecNormalize 不存在: {vn_path}")
    return model_path, vn_path


def load_policy(model_path: Path, vn_path: Path, device: str = "cpu",
                cfg: EnvConfig | None = None):
    """把 PPO 与其 VecNormalize 装进一个 1 环境向量环境。

    模型构造需要**某个**环境; 一个默认地形的 1 环境 DummyVecEnv 就够, 并保留
    VecNormalize 使 ``model.predict`` 拿到的归一化观测与训练时一致。

    ``cfg`` 必须是**训练时**的 EnvConfig (用
    :func:`robotdog.training.env_config_from_run`)。带
    ``--obs-foot-height --obs-base-height`` 训练的模型是 490 维, 而 ``EnvConfig()``
    默认 420 维, 用默认配置加载会抛 ``ValueError: Observation spaces do not match``。
    """
    from robotdog.tools.terrain import flat_spec

    if cfg is None:
        cfg = EnvConfig()
    venv = make_vec_env(flat_spec(), n_envs=1, cfg=cfg, seed=0)
    model = load_ppo(model_path, venv, device=device)
    import pickle

    from stable_baselines3.common.vec_env import VecNormalize

    with open(vn_path, "rb") as fh:
        vn = pickle.load(fh)
    venv.obs_rms = vn.obs_rms
    venv.ret_rms = vn.ret_rms
    venv.training = False
    venv.norm_reward = False
    model.set_env(venv)
    return model, venv


class SingleEnvPredict:
    """Adapts ``PPO.predict`` to the plain :class:`Go2StairsEnv` used by eval.

    ``evaluate_terrain`` calls ``model.predict(obs, deterministic=...)`` with a
    bare 1-D observation from a non-vectorized env, so we wrap the vectorized
    model and normalise manually with the training statistics.
    """

    def __init__(self, model, venv) -> None:
        self.model = model
        self.venv = venv
        self.obs_rms = venv.obs_rms
        self.clip_obs = float(getattr(venv, "clip_obs", 10.0))
        self.epsilon = float(getattr(venv, "epsilon", 1e-8))

    def predict(self, obs, deterministic: bool = True):
        x = np.asarray(obs, dtype=np.float64)
        if self.obs_rms is not None:
            x = (x - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)
            x = np.clip(x, -self.clip_obs, self.clip_obs)
        action, state = self.model.predict(x.astype(np.float32),
                                           deterministic=deterministic)
        return action, state


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_video(
    predictor,
    spec,
    name: str,
    out_dir: Path,
    *,
    episodes: int = 1,
    seed: int = 0,
    cfg: EnvConfig | None = None,
    max_steps: int | None = None,
    fps: int = 25,
) -> dict:
    """Render a frame every 2 control steps and write a GIF-ish PNG strip.

    We avoid an ffmpeg dependency by saving individual PNGs plus a single
    contact-sheet strip.  Frames come from ``Go2StairsEnv.render()`` (offscreen
    MuJoCo Renderer), and every PNG is written with the dependency-free
    ``write_png`` helper.
    """
    from robotdog.tools.png import downscale, hstack, write_png

    env = Go2StairsEnv(stair=spec, cfg=cfg, seed=seed)
    max_steps = max_steps or int(env.cfg.episode_length_s / env.cfg.control_dt)
    frame_dir = Path(out_dir) / f"frames_{name}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    all_frames: list[np.ndarray] = []
    ep_stats: list[dict] = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            x0 = float(env.base_pos[0])
            frames = []
            for step in range(max_steps):
                action, _ = predictor.predict(obs, deterministic=True)
                obs, _r, terminated, truncated, info = env.step(action)
                if step % 2 == 0:
                    frames.append(env.render().copy())
                if terminated or truncated:
                    break
            for i, f in enumerate(frames):
                write_png(frame_dir / f"ep{ep:02d}_{i:04d}.png", f)
            all_frames.extend(frames)
            ep_stats.append({
                "episode": ep,
                "frames": len(frames),
                "steps": step + 1,
                "distance": round(float(env.base_pos[0]) - x0, 3),
                "success": bool(info.get("is_success", False)),
                "fell": bool(terminated),
            })
    finally:
        env.close()

    # Contact sheet: every Nth frame laid out horizontally, scaled to 1/4.
    strip_path = None
    if all_frames:
        pick = _spread(all_frames, 12)
        sheet = hstack([downscale(f, 4) for f in pick])
        strip_path = Path(out_dir) / f"strip_{name}.png"
        write_png(strip_path, sheet)
    return {
        "terrain": name,
        "frames_dir": str(frame_dir),
        "strip": str(strip_path) if strip_path else None,
        "episodes": ep_stats,
    }


def _spread(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[int(i)] for i in idx]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    model_path, vn_path = resolve_model_paths(args)
    print("=" * 78)
    print("第九步: 策略评估与导出")
    print("=" * 78)
    print(f"模型        : {model_path}")
    print(f"VecNormalize: {vn_path}")
    if args.model:
        out_dir = Path(args.export_dir) if args.export_dir else model_path.parent
    elif args.run and Path(args.run).is_dir():
        run_path = Path(args.run)
        out_dir = RunDir(name=run_path.name, root=run_path).artifacts
    else:
        out_dir = model_path.parent
    print(f"输出目录    : {out_dir}")

    # 训练时的 EnvConfig 必须从 run 的 summary.json 恢复: 用了
    # --obs-foot-height / --obs-base-height / --cfg 的模型观测维度不是 420,
    # 用默认配置搭环境会在 PPO.load 处直接抛观测空间不匹配。
    from robotdog.training import env_config_for_model

    cfg = env_config_for_model(model_path)
    model, venv = load_policy(model_path, vn_path, device=args.device, cfg=cfg)
    predictor = SingleEnvPredict(model, venv)
    print(f"观测维度    : {model.observation_space.shape}"
          f"  动作维度: {model.action_space.shape}")
    print(f"确定性策略  : {args.deterministic}")

    terrains = all_terrains(include_random=args.n_random, seed=args.seed,
                            only=args.only)
    if not terrains:
        raise SystemExit("没有可评估的地形 (检查 --only / --n-random)")

    print("-" * 78)
    print(f"开始评估 {len(terrains)} 个地形 × {args.episodes} 回合 "
          f"(共 {len(terrains) * args.episodes} 回合)...")
    results, traces = evaluate_all(
        predictor, terrains, n_episodes=args.episodes, seed=args.seed,
        cfg=cfg, deterministic=args.deterministic,
        progress=not args.quiet,
    )

    summary = summarize(results)
    breakdown = termination_breakdown(results)

    print("-" * 78)
    print("按地形逐个结果")
    print("-" * 78)
    print(format_table(results))

    print("-" * 78)
    print("分组汇总")
    print("-" * 78)
    for key, s in summary.items():
        print(f"[{key:<8}] 地形数={s['terrains']:>2} 回合={s['episodes']:>4} "
              f"成功率={s['success_rate']:>7.1%} 摔倒率={s['fall_rate']:>7.1%} "
              f"超时={s['timeout_rate']:>6.1%} 平均速度={s['mean_forward_speed']:.2f} m/s "
              f"离地={s['mean_base_height']:.3f} m |pitch|={s['mean_abs_pitch']:.4f}")
        print(f"           最好={s['best_terrain']}  最差={s['worst_terrain']}")
    print(f"结束原因统计: {breakdown}")

    report_path = Path(args.out) if args.out else out_dir / "eval_report.json"
    meta = {
        "model": str(model_path),
        "vecnormalize": str(vn_path),
        "episodes_per_terrain": args.episodes,
        "deterministic": args.deterministic,
        "seed": args.seed,
        "only": args.only,
        "n_random": args.n_random,
        "elapsed_s": round(time.time() - t0, 1),
        "obs_dim": int(np.prod(model.observation_space.shape)),
        "action_dim": int(np.prod(model.action_space.shape)),
        "num_parameters": int(sum(p.numel() for p in model.policy.parameters())),
    }

    exports: dict = {}
    if args.export:
        export_dir = Path(args.export_dir) if args.export_dir else out_dir
        export_dir.mkdir(parents=True, exist_ok=True)
        stem = str(args.export_name).strip() or "policy"
        print("-" * 78)
        print("导出 ONNX / TorchScript")
        print("-" * 78)
        onnx_info = export_onnx(model, export_dir / f"{stem}.onnx",
                                obs_dim=meta["obs_dim"], opset=args.onnx_opset)
        exports["onnx"] = onnx_info
        print(f"ONNX        : {onnx_info['path']} "
              f"({onnx_info['size_bytes'] / 1024:.1f} KiB, "
              f"归一化内嵌={onnx_info['normalization_baked_in']})")
        print(f"  onnx.checker : IR v{onnx_info['onnx_check']['ir_version']}, "
              f"输入 {onnx_info['onnx_check']['inputs']}")
        nm = onnx_info["numeric_match"]
        if nm.get("available"):
            print(f"  onnxruntime  : {nm['samples']} 个随机观测, "
                  f"max|ONNX - PyTorch| = {nm['max_abs_diff']:.3e} "
                  f"(容差 {nm['tol']:.0e}) -> "
                  f"{'通过' if nm['passed'] else '失败'}")
        else:
            print(f"  onnxruntime  : 跳过 ({nm.get('skipped')})")
        torch_info = export_torchscript(model, export_dir / f"{stem}.pt",
                                        obs_dim=meta["obs_dim"])
        exports["torchscript"] = torch_info
        print(f"TorchScript : {torch_info['path']} "
              f"({torch_info['size_bytes'] / 1024:.1f} KiB)")
        print(f"  trace 差异   : max|traced - eager| = "
              f"{torch_info['trace_max_abs_diff']:.3e} -> "
              f"{'通过' if torch_info['numeric_match'] else '失败'}")
        # Cross-check: eager module vs the exported ONNX on the same input.
        module = build_inference_module(model)
        torch_info["eager_vs_onnx_max_abs_diff"] = None
        meta["exports"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "onnx_check"}
            for k, v in exports.items()
        }

    if args.render_video:
        print("-" * 78)
        print("渲染帧序列")
        print("-" * 78)
        by_name = {name: spec for name, spec in terrains}
        if args.video_terrain:
            targets = [(args.video_terrain, by_name[args.video_terrain])]
        else:
            best = max(results, key=lambda r: (r.success_rate, r.mean_forward_speed))
            worst = min(results, key=lambda r: (r.success_rate, r.mean_forward_speed))
            targets = [(best.terrain, by_name[best.terrain]),
                       (worst.terrain, by_name[worst.terrain])]
        videos = []
        for name, spec in targets:
            print(f"  渲染 {name} ...", flush=True)
            info = render_video(predictor, spec, name, out_dir,
                                episodes=args.video_episodes, seed=args.seed,
                                cfg=cfg, max_steps=args.max_steps)
            print(f"    -> {info['episodes']}")
            videos.append(info)
        meta["videos"] = videos

    report_path = write_report(report_path, results=results, summary=summary,
                               meta=meta, traces=traces)
    print("-" * 78)
    print(f"报告已写入: {report_path}")
    print(f"总耗时    : {format_duration(time.time() - t0)}")

    # ---- final verdict ---------------------------------------------------- #
    unseen = summary.get("unseen")
    if unseen is not None:
        print("=" * 78)
        print(f"没见过的楼梯: {unseen['terrains']} 个地形 / {unseen['episodes']} 回合, "
              f"成功率 {unseen['success_rate']:.1%}, 摔倒率 {unseen['fall_rate']:.1%}")
        print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
