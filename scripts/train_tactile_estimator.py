"""训练 GRU 足底触觉状态估计器。

用法::

    .venv\\Scripts\\python.exe scripts/train_tactile_estimator.py \
        --data-dir .tmp/tactile_estimator/data \
        --out .tmp/tactile_estimator/estimator.pt

输入是 ``scripts/collect_tactile_data.py`` 产出的 ``train_*.npz``（留出集
``test_*.npz`` 不参与训练，只在评估脚本里用）。标准化统计**只从训练集**算，
随权重一起存档；按**回合**切 val（同一回合不会同时出现在 train/val）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from robotdog.tactile_estimator import (
    EstimatorConfig, contact_metrics, force_metrics, load_dataset,
    make_dataset_windows, merge_windows, regression_metrics, split_episodes,
    train_estimator,
)


def collect_train_windows(data_dir: Path, cfg: EstimatorConfig, stride: int):
    files = sorted(Path(data_dir).glob("train_*.npz"))
    if not files:
        raise SystemExit(f"{data_dir} 里没有 train_*.npz")
    parts = []
    for i, f in enumerate(files):
        data = load_dataset(f)
        w = make_dataset_windows(data, cfg, stride)
        w["episode"] = w["episode"] + i * 100000
        parts.append(w)
        print(f"  {f.name:28s} windows={len(w['x'])}")
    return merge_windows(parts), [str(f) for f in files]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".tmp/tactile_estimator/data")
    ap.add_argument("--out", default=".tmp/tactile_estimator/estimator.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stride-train", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = EstimatorConfig(window=args.window, hidden=args.hidden,
                          layers=args.layers, batch=args.batch, lr=args.lr,
                          epochs=args.epochs, val_frac=args.val_frac,
                          seed=args.seed)
    t0 = time.time()
    print("[1/2] 读训练窗口")
    train_all, files = collect_train_windows(Path(args.data_dir), cfg,
                                             args.stride_train)
    n = int(train_all["x"].shape[0])
    tr_mask, val_mask = split_episodes(train_all["episode"], cfg.val_frac,
                                       cfg.seed)
    train = {k: v[tr_mask] for k, v in train_all.items() if k != "episode"}
    val = {k: v[val_mask] for k, v in train_all.items() if k != "episode"}
    print(f"  train windows={train['x'].shape[0]}  val windows={val['x'].shape[0]}"
          f"  features={train['x'].shape[-1]}  window={cfg.window}")

    print("[2/2] 训练")
    est, history = train_estimator(train, val, cfg, device=args.device)
    out = Path(args.out)
    est.save(out, extra_meta={
        "train_files": files,
        "train_windows": int(train["x"].shape[0]),
        "val_windows": int(val["x"].shape[0]),
        "history": history,
    })

    # 训练/验证集上的快速指标（详细留出集指标由 eval 脚本给）
    val_pred = est.predict(val["x"])
    fm = force_metrics(val_pred["force"], val["force"], cfg.force_ref_n)
    cm = contact_metrics(val_pred["contact"], val["contact"],
                         cfg.contact_threshold)
    am = regression_metrics(val_pred["area"], val["area"], val["contact"])
    pm = regression_metrics(val_pred["cop"], val["cop"],
                            np.repeat(val["contact"][:, :, None], 2, axis=2))
    summary = {"val_force": fm, "val_contact": cm, "val_area": am,
               "val_cop": pm, "seconds": time.time() - t0}
    (out.with_suffix(".json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"val force RMSE {fm['rmse_all_n']:.1f} N ({fm['rmse_all_bw']:.3f} BW)  "
          f"MAE {fm['mae_all_n']:.1f} N  | contact acc {cm['accuracy']:.3f} "
          f"F1 {cm['f1']:.3f} | area RMSE {am['rmse']:.3f} "
          f"| cop RMSE {pm['rmse']:.3f}")
    print(f"saved: {out}  ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
