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
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from robotdog.tactile_estimator import (
    EstimatorConfig, arch_spec, contact_metrics, force_metrics, load_dataset,
    make_dataset_windows, merge_windows, regression_metrics, slice_features,
    split_episodes, train_estimator,
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


def build_parser() -> argparse.ArgumentParser:
    """CLI 参数默认值全部来自 EstimatorConfig()（默认值单源）。

    改 EstimatorConfig 的超参默认值即改 CLI 默认值；测试
    test_cli_defaults_match_config 断言两者一致。
    """
    d = EstimatorConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".tmp/tactile_estimator/data")
    ap.add_argument("--out", default=".tmp/tactile_estimator/estimator.pt")
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--window", type=int, default=d.window)
    ap.add_argument("--hidden", type=int, default=d.hidden)
    ap.add_argument("--layers", type=int, default=d.layers)
    ap.add_argument("--batch", type=int, default=d.batch)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--stride-train", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=d.val_frac)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--device", default="cpu")
    # ---- v3：力矩特征 / 热启动 / 解耦训练 ----
    ap.add_argument("--features", type=int, default=0,
                    help="只取特征前 N 列（0=全宽；65=53 观测+12 力矩，53=纯观测基线）")
    ap.add_argument("--init-from", default="",
                    help="热启动 checkpoint（要求 n_features 一致；缺失的头随机初始化）")
    ap.add_argument("--freeze-trunk", action="store_true",
                    help="冻结主干与非 F_bar 头，只训练 F_bar 头（解耦臂，需要 --w-forcenorm > 0）")
    # ---- 消费端对齐（默认全 0，与历史配方逐位等价）----
    ap.add_argument("--w-forcenorm", type=float, default=d.w_forcenorm)
    ap.add_argument("--w-delta", type=float, default=d.w_delta)
    ap.add_argument("--w-peak", type=float, default=d.w_peak)
    ap.add_argument("--w-magrel", type=float, default=d.w_magrel)
    ap.add_argument("--w-fbar-rel", type=float, default=d.w_fbar_rel)
    # ---- v4：支撑相加权 / 解耦头容量 ----
    ap.add_argument("--w-stance", type=float, default=d.w_stance,
                    help="支撑相(phase>2.5)帧的额外损失权重倍率（0=不加权）")
    ap.add_argument("--head-hidden", type=int, default=d.head_hidden,
                    help="F_bar 头隐层宽度（0=单层线性；>0 时 Linear-GELU-Linear）")
    # ---- v5：架构 / 训练调度 ----
    ap.add_argument("--arch", default=d.arch, choices=("shared", "indep"),
                    help="shared=四头共享一个 GRU；indep=每任务独立 GRU（容量见 --task-caps）")
    ap.add_argument("--task-caps", default=d.task_caps,
                    help='indep 的每任务容量，如 "force:2x128,contact:1x128,area:1x64,cop:1x64"')
    ap.add_argument("--lr-gamma", type=float, default=d.lr_gamma,
                    help="每 epoch 学习率指数衰减系数（1.0=不衰减）")
    ap.add_argument("--early-stop", type=int, default=d.early_stop,
                    help="连续多少 epoch val 无改善就停（0=不早停；仍然保存 best-val）")
    # 早停窗口统一由 --early-stop 指定；EstimatorConfig.patience 保留字段只为
    # 兼容旧 checkpoint 的 config（不再有 CLI，也不会被训练循环读取）。
    # ---- v7 Wave C：双频输出与训练端接触去抖 ----
    ap.add_argument("--bifreq-k", type=int, default=d.bifreq_k,
                    help=">0：F_bar 头目标改为因果 k 帧均值（低频幅值通道；需要 --w-forcenorm>0）")
    ap.add_argument("--label-clip-mg", type=float, default=d.label_clip_mg,
                    help="训练侧标签限幅（mg，默认 10=1492 N；0=不限幅，复现历史行为）")
    ap.add_argument("--debounce-contact", action="store_true",
                    help="训练侧接触标签用 3 帧>=2 帧因果多数票去抖（评估仍按原始 contact）")
    ap.add_argument("--physics-weight", type=float, default=d.physics_weight)
    ap.add_argument("--lookahead", type=int, default=d.lookahead_contact,
                    help="接触前瞻帧数（0/1/2）")
    ap.add_argument("--w-lookahead", type=float, default=d.w_lookahead)
    return ap


def config_from_args(args: argparse.Namespace) -> EstimatorConfig:
    """把 CLI 参数合并回 EstimatorConfig（未传即用 config 默认）。"""
    return EstimatorConfig(window=args.window, hidden=args.hidden,
                           layers=args.layers, batch=args.batch, lr=args.lr,
                           epochs=args.epochs, val_frac=args.val_frac,
                           seed=args.seed,
                           w_forcenorm=args.w_forcenorm, w_delta=args.w_delta,
                           w_peak=args.w_peak, w_magrel=args.w_magrel,
                           w_fbar_rel=args.w_fbar_rel,
                           w_stance=args.w_stance,
                           head_hidden=args.head_hidden,
                           physics_weight=args.physics_weight,
                           lookahead_contact=args.lookahead,
                           w_lookahead=args.w_lookahead,
                           arch=args.arch, task_caps=args.task_caps,
                           lr_gamma=args.lr_gamma, early_stop=args.early_stop,
                           bifreq_k=args.bifreq_k,
                           debounce_contact=bool(args.debounce_contact),
                           label_clip_mg=args.label_clip_mg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = build_parser().parse_args()
    cfg = config_from_args(args)
    print(f"架构: arch={cfg.arch} task_caps={cfg.task_caps or '(默认 1x64)'} "
          f"lr_gamma={cfg.lr_gamma} early_stop={cfg.early_stop}")
    print("消费端权重: " + ", ".join(
        f"{k}={getattr(cfg, k)}" for k in
        ("w_forcenorm", "w_delta", "w_peak", "w_magrel", "w_fbar_rel",
         "physics_weight", "lookahead_contact", "w_lookahead",
         "bifreq_k", "debounce_contact")))
    t0 = time.time()
    print("[1/2] 读训练窗口")
    train_all, files = collect_train_windows(Path(args.data_dir), cfg,
                                             args.stride_train)
    train_all = slice_features(train_all, args.features)
    n = int(train_all["x"].shape[0])
    tr_mask, val_mask = split_episodes(train_all["episode"], cfg.val_frac,
                                       cfg.seed)
    train = {k: v[tr_mask] for k, v in train_all.items() if k != "episode"}
    val = {k: v[val_mask] for k, v in train_all.items() if k != "episode"}
    print(f"  train windows={train['x'].shape[0]}  val windows={val['x'].shape[0]}"
          f"  features={train['x'].shape[-1]}  window={cfg.window}")

    print("[2/2] 训练")
    est, history = train_estimator(
        train, val, cfg, device=args.device,
        init_from=(args.init_from or None), freeze_trunk=bool(args.freeze_trunk))
    out = Path(args.out)
    data_dir = str(Path(args.data_dir).resolve())
    data_sha = {Path(f).name: sha256_file(Path(f)) for f in files}
    est.save(out, extra_meta={
        # v5 provenance：ckpt 必须能自证「哪批数据、什么切分、最优轮」
        "data_dir": data_dir,
        "data_sha256": data_sha,
        "split_seed": int(cfg.seed),
        "val_frac": float(cfg.val_frac),
        "val_episode_ids": sorted({int(e) for e in np.unique(
            train_all["episode"][val_mask])}),
        "arch": cfg.arch,
        "task_caps": cfg.task_caps,
        "arch_spec": {k: list(v) for k, v in
                      (arch_spec(cfg.arch, cfg.task_caps,
                                 cfg.layers, cfg.hidden).items())},
        "stride_train": int(args.stride_train),
        "device": str(args.device),
        "n_params": int(sum(p.numel() for p in est.model.parameters())),
        "best_epoch": int(getattr(est, "best_epoch", 0)),
        "best_val_loss": getattr(est, "best_val_loss", None),
        "stopped_early": bool(getattr(est, "stopped_early", False)),
        "val_empty": bool(getattr(est, "val_empty", False)),
        "epochs_run": int(getattr(est, "epochs_run", len(history))),
        "train_files": files,
        "train_windows": int(train["x"].shape[0]),
        "val_windows": int(val["x"].shape[0]),
        "history": history,
        "features": int(train["x"].shape[-1]),
        "init_from": (args.init_from or None),
        "freeze_trunk": bool(args.freeze_trunk),
        "init_info": getattr(est, "init_info", None),
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
