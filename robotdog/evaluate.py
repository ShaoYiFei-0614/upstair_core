"""第九步: 策略评估与导出 (ONNX / TorchScript)。

## 评估内容 (用户规格)
"测试没见过的楼梯。看成功率、速度、姿态、摔倒次数。保存最好策略, 导出 ONNX 或
TorchScript。"

* **没见过的楼梯**: :func:`unseen_terrains` 生成的台阶高度/踏面宽度都不在训练
  课程里出现过 (训练用 5 / 12 / 19 cm 台阶 + 50 cm 踏面)。
* **成功率 / 速度 / 姿态 / 摔倒次数**: 见 :class:`TerrainResult`。
* **导出**: :func:`export_onnx` / :func:`export_torchscript`。

## 为什么导出时要内嵌观测归一化
训练用了 ``VecNormalize(norm_obs=True)``, 策略实际吃的是
``clip((obs - mean)/sqrt(var + eps), -clip_obs, clip_obs)``。
如果导出裸策略, 部署方必须自己复现这套统计量, 很容易搞错。所以导出时把归一化
**折叠进模型**: ONNX/TorchScript 接受**原始 420 维观测**, 内部自己做归一化。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from robotdog.env import EnvConfig, Go2StairsEnv
from robotdog.terrain import StairSpec, TerrainRandomization, demo_stairs_spec, flat_spec

# --------------------------------------------------------------------------- #
# Evaluation terrains
# --------------------------------------------------------------------------- #

# Step heights / tread widths used during training (see default_curriculum).
_TRAINED_STEP_HEIGHTS = (0.05, 0.12, 0.19)
_TRAINED_TREAD = 0.50


def trained_terrains() -> list[tuple[str, StairSpec]]:
    return [
        ("train_flat", flat_spec(n_up=0, n_down=0, tread=_TRAINED_TREAD,
                                 width=2.0, start_x=1.50)),
        *[
            (f"train_h{int(h * 1000):03d}", demo_stairs_spec(
                step_height=h, tread=_TRAINED_TREAD, n_up=8, top_landing=1.20))
            for h in _TRAINED_STEP_HEIGHTS
        ],
    ]


def unseen_terrains() -> list[tuple[str, StairSpec]]:
    """Step heights and tread widths deliberately off the training grid.

    Includes both **interpolation** (7 cm sits between the 5 and 12 cm stages)
    and **extrapolation** (23 cm is above the hardest training stage), plus
    tread widths other than the trained 50 cm, plus a pyramid (up then down)
    which the curriculum never trained at all.
    """
    out: list[tuple[str, StairSpec]] = [
        ("unseen_h07_t50", demo_stairs_spec(step_height=0.07, tread=0.50, n_up=8,
                                            top_landing=1.20)),
        ("unseen_h10_t45", demo_stairs_spec(step_height=0.10, tread=0.45, n_up=8,
                                            top_landing=1.20)),
        ("unseen_h15_t40", demo_stairs_spec(step_height=0.15, tread=0.40, n_up=8,
                                            top_landing=1.20)),
        ("unseen_h17_t55", demo_stairs_spec(step_height=0.17, tread=0.55, n_up=8,
                                            top_landing=1.20)),
        ("unseen_h23_t50", demo_stairs_spec(step_height=0.23, tread=0.50, n_up=8,
                                            top_landing=1.20)),
        ("unseen_h09_t35", demo_stairs_spec(step_height=0.09, tread=0.35, n_up=6,
                                            top_landing=1.00)),
        ("unseen_shallow", demo_stairs_spec(step_height=0.035, tread=0.60, n_up=10,
                                            top_landing=1.20)),
        ("unseen_pyramid", demo_stairs_spec(step_height=0.08, tread=0.50, n_up=6,
                                            n_down=6, landing=1.20)),
        ("unseen_narrow", demo_stairs_spec(step_height=0.06, tread=0.50, n_up=8,
                                           width=1.2, top_landing=1.20)),
    ]
    return out


def randomized_terrains(n: int, seed: int = 12345) -> list[tuple[str, StairSpec]]:
    """Uniformly sampled terrain from :class:`TerrainRandomization`."""
    rand = TerrainRandomization()
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        spec = rand.sample_spec(rng)
        out.append((f"rand_{i:02d}_h{int(spec.step_height * 1000):03d}"
                    f"_t{int(spec.tread * 100):03d}", spec))
    return out


def all_terrains(include_random: int = 0, seed: int = 12345,
                 only: str = "all") -> list[tuple[str, StairSpec]]:
    groups: list[tuple[str, StairSpec]] = []
    if only in ("all", "trained"):
        groups += trained_terrains()
    if only in ("all", "unseen"):
        groups += unseen_terrains()
    if only in ("all", "random") or include_random:
        groups += randomized_terrains(include_random or (6 if only == "random" else 0),
                                      seed=seed)
    return groups


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@dataclass
class TerrainResult:
    """Per-terrain evaluation metrics.

    ``success_rate`` / ``fall_rate`` / ``timeout_rate`` are the user's
    "成功率 / 摔倒次数"; ``mean_forward_speed`` and the posture fields are
    "速度 / 姿态".
    """

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
    mean_forward_speed: float = 0.0  # m/s, over the whole episode
    mean_distance: float = 0.0  # m travelled in +x
    mean_height_gain: float = 0.0  # m climbed (change in absolute base z)
    mean_base_height: float = 0.0  # m of clearance above the terrain under the base
    mean_abs_roll: float = 0.0
    mean_abs_pitch: float = 0.0
    max_abs_roll: float = 0.0
    max_abs_pitch: float = 0.0
    # --- step 11: 走直线 (straight-line) metrics -------------------------- #
    # All measured relative to the pose the episode spawned in, because reset
    # itself adds lateral/yaw jitter (reset_xy_noise / reset_yaw_noise) that the
    # robot is not responsible for.
    mean_abs_y: float = 0.0  # m, mean |lateral offset|
    max_abs_y: float = 0.0  # m, worst lateral excursion
    final_abs_y: float = 0.0  # m, |lateral offset| at the final step
    mean_abs_heading_deg: float = 0.0  # deg, mean |heading error|
    max_abs_heading_deg: float = 0.0  # deg
    mean_lateral_drift_per_m: float = 0.0  # m of |y| per m travelled in +x
    mean_survival_fraction: float = 0.0  # episode steps / max steps
    steps: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TerrainTrace:
    """Raw per-episode trace for one terrain, used for plotting / inspection."""

    terrain: str
    episodes: list[dict[str, Any]] = field(default_factory=list)


def evaluate_terrain(
    model,
    spec: StairSpec,
    name: str = "terrain",
    *,
    n_episodes: int = 20,
    seed: int = 0,
    cfg: EnvConfig | None = None,
    deterministic: bool = True,
    collect_trace: bool = False,
    max_episode_steps: int | None = None,
) -> tuple[TerrainResult, TerrainTrace]:
    """Roll out ``n_episodes`` episodes on one terrain and aggregate metrics."""
    env = Go2StairsEnv(stair=spec, cfg=cfg, seed=seed)
    max_steps = max_episode_steps or int(env.cfg.episode_length_s / env.cfg.control_dt)
    res = TerrainResult(
        terrain=name,
        step_height=float(spec.step_height),
        tread=float(spec.tread),
        n_up=int(spec.n_up),
        n_down=int(spec.n_down),
        crest_height=float(spec.crest_height),
    )
    trace = TerrainTrace(terrain=name)

    returns: list[float] = []
    lengths: list[float] = []
    speeds: list[float] = []
    distances: list[float] = []
    gains: list[float] = []
    heights: list[float] = []
    rolls: list[float] = []
    pitches: list[float] = []
    max_rolls: list[float] = []
    max_pitches: list[float] = []
    survival: list[float] = []
    abs_ys: list[float] = []
    max_abs_ys: list[float] = []
    final_abs_ys: list[float] = []
    abs_heads: list[float] = []
    max_abs_heads: list[float] = []
    drift_per_m: list[float] = []

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            x0 = float(env.base_pos[0])
            z0 = float(env.base_pos[2])
            ep_return = 0.0
            max_roll = max_pitch = 0.0
            steps = 0
            terminated = truncated = False
            info: dict[str, Any] = {}
            ep_rolls: list[float] = []
            ep_pitches: list[float] = []
            ep_abs_y: list[float] = []
            ep_abs_head: list[float] = []
            while not (terminated or truncated) and steps < max_steps:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)
                steps += 1
                rpy = env.base_rpy
                ep_rolls.append(abs(float(rpy[0])))
                ep_pitches.append(abs(float(rpy[1])))
                max_roll = max(max_roll, abs(float(rpy[0])))
                max_pitch = max(max_pitch, abs(float(rpy[1])))
                ep_abs_y.append(abs(env.lateral_offset))
                ep_abs_head.append(abs(env.heading_error))

            success = bool(info.get("is_success", False))
            # NOTE: must use info["fallen"], not `terminated`.  With
            # terminate_on_success the env also sets terminated=True when the
            # robot reaches the crest, so `bool(terminated)` would count every
            # success as a fall (success_rate + fall_rate could exceed 1).
            fell = bool(info.get("fallen", bool(terminated) and not success))
            res.episodes += 1
            res.successes += int(success)
            res.falls += int(fell)
            res.timeouts += int(truncated and not fell)
            returns.append(ep_return)
            lengths.append(float(steps))
            dt = steps * env.cfg.control_dt
            travelled = float(env.base_pos[0]) - x0
            distances.append(travelled)
            speeds.append(travelled / dt if dt > 0 else 0.0)
            gains.append(float(env.base_pos[2]) - z0)
            heights.append(float(env.base_pos[2])
                           - float(env.terrain_height_under_base))
            rolls.append(float(np.mean(ep_rolls)) if ep_rolls else 0.0)
            pitches.append(float(np.mean(ep_pitches)) if ep_pitches else 0.0)
            max_rolls.append(max_roll)
            max_pitches.append(max_pitch)
            survival.append(steps / max_steps)
            ep_mean_abs_y = float(np.mean(ep_abs_y)) if ep_abs_y else 0.0
            abs_ys.append(ep_mean_abs_y)
            max_abs_ys.append(float(np.max(ep_abs_y)) if ep_abs_y else 0.0)
            final_abs_ys.append(abs(env.lateral_offset))
            abs_heads.append(float(np.mean(ep_abs_head)) if ep_abs_head else 0.0)
            max_abs_heads.append(float(np.max(ep_abs_head)) if ep_abs_head else 0.0)
            drift_per_m.append(ep_mean_abs_y / abs(travelled) if abs(travelled) > 1e-6
                               else 0.0)
            if collect_trace:
                trace.episodes.append({
                    "episode": ep,
                    "steps": steps,
                    "return": round(ep_return, 3),
                    "success": success,
                    "fell": fell,
                    "timeout": bool(truncated and not fell),
                    "distance": round(travelled, 3),
                    "height_gain": round(gains[-1], 3),
                    "max_abs_roll": round(max_roll, 4),
                    "max_abs_pitch": round(max_pitch, 4),
                    "final_x": round(float(env.base_pos[0]), 3),
                    "mean_abs_y": round(ep_mean_abs_y, 4),
                    "max_abs_y": round(max_abs_ys[-1], 4),
                    "final_abs_y": round(final_abs_ys[-1], 4),
                    "mean_abs_heading_deg": round(float(np.degrees(abs_heads[-1])), 3),
                })
    finally:
        env.close()

    n = max(res.episodes, 1)
    res.success_rate = res.successes / n
    res.fall_rate = res.falls / n
    res.timeout_rate = res.timeouts / n
    res.mean_return = float(np.mean(returns)) if returns else 0.0
    res.mean_episode_length = float(np.mean(lengths)) if lengths else 0.0
    res.mean_forward_speed = float(np.mean(speeds)) if speeds else 0.0
    res.mean_distance = float(np.mean(distances)) if distances else 0.0
    res.mean_height_gain = float(np.mean(gains)) if gains else 0.0
    res.mean_base_height = float(np.mean(heights)) if heights else 0.0
    res.mean_abs_roll = float(np.mean(rolls)) if rolls else 0.0
    res.mean_abs_pitch = float(np.mean(pitches)) if pitches else 0.0
    res.max_abs_roll = float(np.max(max_rolls)) if max_rolls else 0.0
    res.max_abs_pitch = float(np.max(max_pitches)) if max_pitches else 0.0
    res.mean_abs_y = float(np.mean(abs_ys)) if abs_ys else 0.0
    res.max_abs_y = float(np.max(max_abs_ys)) if max_abs_ys else 0.0
    res.final_abs_y = float(np.mean(final_abs_ys)) if final_abs_ys else 0.0
    res.mean_abs_heading_deg = float(np.degrees(np.mean(abs_heads))) if abs_heads else 0.0
    res.max_abs_heading_deg = float(np.degrees(np.max(max_abs_heads))) if max_abs_heads else 0.0
    res.mean_lateral_drift_per_m = float(np.mean(drift_per_m)) if drift_per_m else 0.0
    res.mean_survival_fraction = float(np.mean(survival)) if survival else 0.0
    res.steps = int(np.sum(lengths))
    return res, trace


def evaluate_all(
    model,
    terrains: Sequence[tuple[str, StairSpec]],
    *,
    n_episodes: int = 20,
    seed: int = 0,
    cfg: EnvConfig | None = None,
    deterministic: bool = True,
    progress: bool = True,
) -> tuple[list[TerrainResult], list[TerrainTrace]]:
    results: list[TerrainResult] = []
    traces: list[TerrainTrace] = []
    for i, (name, spec) in enumerate(terrains, start=1):
        if progress:
            print(f"  [{i:>2}/{len(terrains)}] {name:<22} "
                  f"台阶={spec.step_height * 100:5.1f}cm 踏面={spec.tread * 100:4.0f}cm "
                  f"级数={spec.n_up}...", end="", flush=True)
        t0 = time.time()
        res, trace = evaluate_terrain(
            model, spec, name, n_episodes=n_episodes, seed=seed, cfg=cfg,
            deterministic=deterministic, collect_trace=True,
        )
        results.append(res)
        traces.append(trace)
        if progress:
            print(f" 成功率={res.success_rate:5.1%} 摔倒率={res.fall_rate:5.1%} "
                  f"速度={res.mean_forward_speed:5.2f}m/s "
                  f"({time.time() - t0:.1f}s)")
    return results, traces


# --------------------------------------------------------------------------- #
# Aggregation / reporting
# --------------------------------------------------------------------------- #


def summarize(results: Sequence[TerrainResult]) -> dict[str, Any]:
    """Group-wise aggregate, keyed by the terrain-name prefix before ``_``."""
    groups: dict[str, list[TerrainResult]] = {}
    for r in results:
        key = r.terrain.split("_")[0]
        groups.setdefault(key, []).append(r)

    out: dict[str, Any] = {}
    for key, items in groups.items():
        n = max(sum(i.episodes for i in items), 1)
        out[key] = {
            "terrains": len(items),
            "episodes": sum(i.episodes for i in items),
            "success_rate": round(sum(i.successes for i in items) / n, 4),
            "fall_rate": round(sum(i.falls for i in items) / n, 4),
            "timeout_rate": round(sum(i.timeouts for i in items) / n, 4),
            "mean_forward_speed": round(
                float(np.mean([i.mean_forward_speed for i in items])), 4
            ),
            "mean_base_height": round(
                float(np.mean([i.mean_base_height for i in items])), 4
            ),
            "mean_abs_pitch": round(float(np.mean([i.mean_abs_pitch for i in items])), 4),
            "mean_abs_roll": round(float(np.mean([i.mean_abs_roll for i in items])), 4),
            "mean_return": round(float(np.mean([i.mean_return for i in items])), 3),
            "mean_abs_y": round(float(np.mean([i.mean_abs_y for i in items])), 4),
            "max_abs_y": round(float(np.max([i.max_abs_y for i in items])), 4),
            "mean_abs_heading_deg": round(
                float(np.mean([i.mean_abs_heading_deg for i in items])), 3
            ),
            "mean_lateral_drift_per_m": round(
                float(np.mean([i.mean_lateral_drift_per_m for i in items])), 4
            ),
            "worst_terrain": min(items, key=lambda i: i.success_rate).terrain,
            "best_terrain": max(items, key=lambda i: i.success_rate).terrain,
        }
    return out


def format_table(results: Sequence[TerrainResult]) -> str:
    head = (
        f"{'地形':<22}{'台阶':>6}{'踏面':>6}{'级数':>5}{'回合':>5}"
        f"{'成功率':>8}{'摔倒率':>8}{'超时':>7}{'速度':>8}{'前进':>8}"
        f"{'爬升':>8}{'离地':>7}{'|roll|':>8}{'|pitch|':>8}{'最大pitch':>10}"
        f"{'|y|均':>8}{'|y|最大':>8}{'|y|末':>7}{'航向°':>8}{'漂移/m':>8}"
    )
    lines = [head, "-" * len(head)]
    for r in results:
        lines.append(
            f"{r.terrain:<22}{r.step_height * 100:>5.0f}c{r.tread * 100:>5.0f}c"
            f"{r.n_up:>5d}{r.episodes:>5d}"
            f"{r.success_rate:>8.1%}{r.fall_rate:>8.1%}{r.timeout_rate:>7.1%}"
            f"{r.mean_forward_speed:>7.2f}m{r.mean_distance:>7.2f}m"
            f"{r.mean_height_gain:>7.2f}m{r.mean_base_height:>6.2f}m"
            f"{r.mean_abs_roll:>8.4f}{r.mean_abs_pitch:>8.4f}{r.max_abs_pitch:>10.4f}"
            f"{r.mean_abs_y:>7.3f}m{r.max_abs_y:>7.3f}m{r.final_abs_y:>6.3f}m"
            f"{r.mean_abs_heading_deg:>8.2f}{r.mean_lateral_drift_per_m:>8.3f}"
        )
    return "\n".join(lines)


def straightness_line(r: TerrainResult) -> str:
    """One-line straightness verdict for a single terrain result.

    The thresholds are deliberately coarse: the thing being measured is "does
    the robot hold its lane", not a millimetre-accurate path tracker.
    """
    y = r.mean_abs_y
    head = r.mean_abs_heading_deg
    if y <= 0.10 and head <= 5.0:
        verdict = "✅ 基本直线"
    elif y <= 0.25 and head <= 12.0:
        verdict = "⚠️ 有可见漂移"
    else:
        verdict = "❌ 明显走偏"
    return (f"{verdict}  平均|y|={y:.3f}m 最大|y|={r.max_abs_y:.3f}m "
            f"航向偏差={head:.2f}° 每米漂移={r.mean_lateral_drift_per_m:.3f}m")


def write_report(path: str | Path, *, results: Sequence[TerrainResult],
                 summary: dict[str, Any], meta: dict[str, Any],
                 traces: Sequence[TerrainTrace] | None = None,
                 include_traces: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "meta": meta,
        "summary": summary,
        "results": [r.as_dict() for r in results],
    }
    if traces is not None and include_traces:
        payload["traces"] = [asdict(t) for t in traces]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def termination_breakdown(results: Sequence[TerrainResult]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in results:
        c["success"] += r.successes
        c["fall"] += r.falls
        c["timeout_survived"] += r.timeouts
        c["episodes"] += r.episodes
    return dict(c)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def _vecnormalize_stats(model) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Pull ``(mean, var, clip_obs)`` out of the model's VecNormalize, if any."""
    getter = getattr(model, "get_vec_normalize_env", None)
    venv = getter() if callable(getter) else None
    if venv is None or getattr(venv, "obs_rms", None) is None:
        return None, None, 10.0
    return (
        np.asarray(venv.obs_rms.mean, dtype=np.float32),
        np.asarray(venv.obs_rms.var, dtype=np.float32),
        float(getattr(venv, "clip_obs", 10.0)),
    )


def build_inference_module(model, *, epsilon: float = 1e-8):
    """A ``torch.nn.Module`` mapping **raw** observation -> clipped action.

    Bakes in VecNormalize's running statistics so the exported artifact is
    self-contained: feed it the same 420-dim observation the env produces.
    """
    import torch
    import torch.nn as nn

    policy = model.policy
    mean, var, clip_obs = _vecnormalize_stats(model)

    class InferenceModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy = policy
            self.normalize = mean is not None
            if self.normalize:
                self.register_buffer("obs_mean", torch.as_tensor(mean))
                self.register_buffer(
                    "obs_std", torch.as_tensor(np.sqrt(var + epsilon))
                )
            self.clip_obs = clip_obs
            self.clip_action = 1.0

        def forward(self, obs: "torch.Tensor") -> "torch.Tensor":
            x = obs
            if self.normalize:
                x = (x - self.obs_mean) / self.obs_std
                x = torch.clamp(x, -self.clip_obs, self.clip_obs)
            features = self.policy.extract_features(x)
            latent_pi, _ = self.policy.mlp_extractor(features)
            actions = self.policy.action_net(latent_pi)
            return torch.clamp(actions, -self.clip_action, self.clip_action)

    module = InferenceModule()
    module.eval()
    return module


def export_onnx(
    model,
    path: str | Path,
    *,
    obs_dim: int | None = None,
    opset: int = 17,
) -> dict[str, Any]:
    """Export to ONNX. Input is the raw observation, output is the action."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obs_dim = obs_dim or int(np.prod(model.observation_space.shape))
    module = build_inference_module(model)
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        reference = module(dummy).cpu().numpy()

    torch.onnx.export(
        module,
        (dummy,),
        str(path),
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    info: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "obs_dim": obs_dim,
        "action_dim": int(reference.shape[-1]),
        "opset": opset,
        "normalization_baked_in": bool(module.normalize),
        "reference_action": [round(float(v), 6) for v in reference[0]],
    }
    info["onnx_check"] = check_onnx(path)
    info["numeric_match"] = verify_onnx_numeric(path, module, obs_dim)
    return info


def check_onnx(path: str | Path) -> dict[str, Any]:
    import onnx

    model_proto = onnx.load(str(path))
    onnx.checker.check_model(model_proto)
    return {
        "ir_version": model_proto.ir_version,
        "producer": f"{model_proto.producer_name} {model_proto.producer_version}".strip(),
        "inputs": [
            {"name": i.name,
             "shape": [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]}
            for i in model_proto.graph.input
        ],
        "outputs": [
            {"name": o.name,
             "shape": [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]}
            for o in model_proto.graph.output
        ],
    }


def verify_onnx_numeric(path: str | Path, module, obs_dim: int,
                        *, n_samples: int = 8, tol: float = 1e-4) -> dict[str, Any]:
    """Run onnxruntime on random observations and compare with PyTorch.

    This is the real proof that the exported file computes the same function;
    ``onnx.checker`` alone only proves the graph is well-formed.
    """
    import torch
    result: dict[str, Any] = {"available": False}
    try:
        import onnxruntime as ort
    except ImportError as exc:
        result["skipped"] = f"onnxruntime 未安装: {exc}"
        return result

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    # Sample realistic observations (std 1) plus a few large ones to exercise
    # the observation clipping path.
    batch = np.concatenate([
        rng.normal(0, 1.0, (n_samples, obs_dim)),
        rng.normal(0, 20.0, (2, obs_dim)),
    ]).astype(np.float32)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: batch})[0]
    with torch.no_grad():
        torch_out = module(torch.as_tensor(batch)).cpu().numpy()
    diff = np.abs(onnx_out - torch_out)
    result.update({
        "available": True,
        "samples": int(batch.shape[0]),
        "max_abs_diff": float(diff.max()),
        "tol": tol,
        "passed": bool(diff.max() <= tol),
    })
    return result


def _ascii_scratch_dir() -> Path:
    """A writable **pure-ASCII** directory, needed to appease libtorch.

    ``torch.jit.save`` is implemented in C++ and takes its destination as a
    ``const char*``; on Windows that is decoded through the ANSI code page
    (GBK/936 here), so a UTF-8 path such as ``D:\\项目\\...`` turns into
    mojibake and libtorch reports ``Parent directory ... does not exist`` even
    though it plainly does.  This is the *same* root cause as the MuJoCo
    ``meshdir`` asset-loading failure documented in
    :mod:`robotdog.go2_model`.
    """
    import os
    import tempfile

    candidates = [
        os.environ.get("DSH_ASCII_SCRATCH"),
        tempfile.gettempdir(),
        os.environ.get("TEMP"),
        os.environ.get("TMP"),
        os.environ.get("SystemRoot", r"C:\Windows") + r"\Temp",
        os.environ.get("LOCALAPPDATA", "") + r"\Temp",
    ]
    for cand in candidates:
        if not cand or not str(cand).isascii():
            continue
        p = Path(cand)
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / "dsh_write_probe.tmp"
            probe.write_bytes(b"ok")
            probe.unlink()
        except OSError:
            continue
        return p
    raise RuntimeError(
        "找不到可写的纯 ASCII 临时目录; libtorch 的 C++ 文件写入无法处理"
        "非 ASCII 路径 (如 D:\\项目\\...)。请设置环境变量 "
        "DSH_ASCII_SCRATCH 指向一个纯 ASCII 目录。"
    )


def _write_via_ascii_path(payload_writer, target: Path) -> None:
    """Run ``payload_writer(ascii_path)`` then copy the bytes to ``target``."""
    import os

    scratch = _ascii_scratch_dir() / f"dsh_export_{os.getpid()}.tmp"
    try:
        payload_writer(str(scratch))
        data = scratch.read_bytes()
    finally:
        try:
            scratch.unlink()
        except OSError:
            pass
    target.write_bytes(data)


def load_torchscript(path: str | Path, *, device: str = "cpu"):
    """Load a TorchScript module, working around libtorch's ASCII-only paths.

    ``torch.jit.load`` is C++ too, so on this machine a ``.pt`` sitting under
    ``D:\\项目\\...`` cannot be opened directly.  We stage a temporary copy in a
    pure-ASCII directory, load it, and delete the copy.
    """
    import os
    import uuid

    import torch

    path = Path(path)
    if str(path).isascii():
        try:
            return torch.jit.load(str(path), map_location=device)
        except (RuntimeError, ValueError):
            pass
    data = path.read_bytes()
    scratch = _ascii_scratch_dir() / f"dsh_load_{os.getpid()}_{uuid.uuid4().hex[:8]}.pt"
    try:
        scratch.write_bytes(data)
        return torch.jit.load(str(scratch), map_location=device)
    finally:
        try:
            scratch.unlink()
        except OSError:
            pass


def export_torchscript(
    model,
    path: str | Path,
    *,
    obs_dim: int | None = None,
) -> dict[str, Any]:
    """``torch.jit.trace`` the same inference module (raw obs -> action)."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obs_dim = obs_dim or int(np.prod(model.observation_space.shape))
    module = build_inference_module(model)
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        reference = module(dummy).cpu().numpy()
        traced = torch.jit.trace(module, (dummy,), strict=False)
        traced = torch.jit.freeze(traced)
        traced_out = traced(dummy).cpu().numpy()
    _write_via_ascii_path(traced.save, path)

    rng = np.random.default_rng(0)
    batch = rng.normal(0, 1.5, (16, obs_dim)).astype(np.float32)
    with torch.no_grad():
        a = module(torch.as_tensor(batch)).cpu().numpy()
        b = traced(torch.as_tensor(batch)).cpu().numpy()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "obs_dim": obs_dim,
        "action_dim": int(reference.shape[-1]),
        "normalization_baked_in": bool(module.normalize),
        "reference_action": [round(float(v), 6) for v in reference[0]],
        "trace_max_abs_diff": float(np.abs(a - b).max()),
        "frozen_max_abs_diff": float(np.abs(traced_out - reference).max()),
        "numeric_match": bool(np.abs(a - b).max() <= 1e-5),
        "ascii_scratch_workaround": not str(path).isascii(),
    }
