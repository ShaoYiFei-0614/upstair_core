"""把 GRU 触觉估计器 checkpoint（.pt）导出成 numpy .npz（Level2 隐式臂用）。

隐式臂的 critic 特权块在环境的 8 个 worker 里逐控制步前向。worker 若各自
import torch + 反序列化整个模型，常驻内存会乘 8（本机 16.8 GB，见
experience.md 的 OOM 记录），所以训练前先把权重落成纯 numpy 的 .npz，
在线前向由 robotdog/tactile_estimate_obs.py 用 numpy 手写（对拍见
tests/test_tactile_estimate_obs.py）。

用法::

    .venv/Scripts/python.exe scripts/export_tactile_estimator.py \
        --checkpoint .tmp/tactile_est_v7/ckpts/C3c_s0.pt \
        --output .tmp/level2/est_C3c_s0.npz

.npz 的键与 checkpoint 的 state_dict 键同名（gru.weight_ih_l0 等），外加
feature_mean / feature_std / n_features / hidden / layers / window /
force_ref_n / meta / source_checkpoint。只导出 force 与 area 两个头需要的量，
但**不改动** checkpoint 本身。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="估计器 .pt 路径")
    p.add_argument("--output", required=True, help="导出的 .npz 路径")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    src = Path(args.checkpoint)
    if not src.is_file():
        raise SystemExit(f"找不到估计器 checkpoint：{src}")
    import torch          # 只有导出脚本 import；训练侧不依赖 torch

    blob = torch.load(src, map_location="cpu", weights_only=False)
    needed = ["config", "state_dict", "feature_mean", "feature_std", "n_features"]
    missing = [k for k in needed if k not in blob]
    if missing:
        raise SystemExit(f"checkpoint 缺字段 {missing}：{src}")
    cfg = blob["config"]
    sd = blob["state_dict"]
    for head, dim in (("force_head", 12), ("area_head", 4)):
        for suffix in ("weight", "bias"):
            key = f"{head}.{suffix}"
            if key not in sd:
                raise SystemExit(
                    f"checkpoint 缺 {key}：隐式块需要 force 与 area 两个头。"
                    f"现有键：{sorted(sd)[:8]} …"
                )
    n_features = int(blob["n_features"])
    window = int(cfg.get("window", 20))
    hidden = int(cfg.get("hidden", 128))
    layers = int(cfg.get("layers", 1))
    force_ref_n = float(cfg.get("force_ref_n", 149.17))
    if str(cfg.get("arch", "shared")) != "shared":
        raise SystemExit(
            f"只支持 arch=shared 的估计器（当前 {cfg.get('arch')}）；"
            "独立骨干版本的 state_dict 键名不同，需要先扩导出逻辑。"
        )

    out = {}
    for i in range(layers):
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            key = f"gru.{name}_l{i}"
            if key not in sd:
                raise SystemExit(f"checkpoint 缺 {key}")
            out[key] = sd[key].detach().cpu().numpy().astype(np.float64)
    for head in ("force_head", "area_head"):
        for suffix in ("weight", "bias"):
            key = f"{head}.{suffix}"
            out[key] = sd[key].detach().cpu().numpy().astype(np.float64)
    out["feature_mean"] = np.asarray(blob["feature_mean"], dtype=np.float64)
    out["feature_std"] = np.asarray(blob["feature_std"], dtype=np.float64)
    out["n_features"] = np.asarray(n_features)
    out["hidden"] = np.asarray(hidden)
    out["layers"] = np.asarray(layers)
    out["window"] = np.asarray(window)
    out["force_ref_n"] = np.asarray(force_ref_n)
    out["meta"] = np.asarray(json.dumps(blob.get("meta", {}), ensure_ascii=False))
    out["source_checkpoint"] = np.asarray(str(src.resolve()))

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)
    print(f"[export] {src} -> {dst}")
    print(f"[export] n_features={n_features} window={window} hidden={hidden} "
          f"layers={layers} force_ref_n={force_ref_n} "
          f"n_actor={n_features - 12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
