"""拟合消费端 F̄ 通道的后验仿射标定 y' = a*y + b。

用法::

    .venv/Scripts/python.exe scripts/calibrate_tactile_estimator.py \
        --checkpoint .tmp/tactile_est_v3/c0T.pt \
        --data-dir .tmp/tactile_est_v3/data \
        --out .tmp/tactile_est_v3/calibration_c0T.json

三条硬约束（评审 B1/B2/B3）：

1. **只用训练地形的 val 回合**：文件顺序、`episode + i*100000` 偏移、`split_episodes`
   的 val_frac/seed 全部与 `scripts/train_tactile_estimator.py` 一致 —— 标定系数是在
   训练时**没见过**的回合上拟合的，留出地形（stair16_5/stair18/rough_s1/rough_s2）
   按名拒绝，json 里写 `val_episode_ids` 与 `assert_no_holdout=true`。
2. **两种拟合方向都给出**：`ls`（t ≈ a*y+b，RMSE 最优、收缩型）与 `inverse`
   （y ≈ α*t+β 反演，a=var(t)/cov(y,t)，斜率修正型）。P3 的 slope 判据只看 inverse，
   ls 用于回答「RMSE 能不能更好」。
3. **掩码统一**：拟合与评估都用 `contact & phase > 2.5`（支撑相接触帧）。

标定结果只是报告口径：scripts/eval_tactile_estimator.py --calibration 施加它，
不改任何权重。产物 json 带 ckpt_sha256/features/stride，eval 侧会校验一致。
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from robotdog.tactile_estimator import (
    TactileEstimator, apply_clustered_affine, apply_standardizer, assign_clusters,
    fit_affine, fit_kmeans, fit_standardizer, load_dataset, make_dataset_windows,
    merge_windows, slice_features, split_episodes, window_embedding,
)

# 留出地形：标定里出现任何一个 ⇒ 直接拒绝（防循环论证）。
# v5：7 个留出地形（插值 10.5/13.5、外推 16.5/18/19.5、rough seed1/2）
HOLDOUT_NAMES = ("stair16_5", "stair18", "stair19_5", "stair10_5",
                 "stair13_5", "rough_s1", "rough_s2")
CHANNELS = ("vector_z", "vector_mag", "head")
METHODS = ("ls", "inverse")
MASK_DESC = "stance(phase>2.5) & contact"


def is_holdout_name(name: str) -> bool:
    """精确名或 `<留出名>_<策略后缀>` 都算留出（R4：防 rough_s1_flatpol 漏网）。"""
    return any(name == h or name.startswith(h + "_") for h in HOLDOUT_NAMES)


def channel_values(pred: dict, ref: float) -> dict:
    """把预测转成消费端可标定的三个通道（形状 (n, 4)，未加掩码）。"""
    vals = {
        "vector_z": pred["force"][:, :, 2] / ref,
        "vector_mag": np.linalg.norm(pred["force"], axis=2) / ref,
    }
    if "forcenorm" in pred:
        vals["head"] = pred["forcenorm"]
    return vals


def sha256_file(path: Path, limit: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(limit)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default=".tmp/tactile_est_v3/data")
    ap.add_argument("--out", default=".tmp/tactile_est_v3/calibration.json")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--features", type=int, default=0,
                    help="特征列数（0=按 checkpoint 的 n_features 自动切前缀）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-samples", type=int, default=100)
    ap.add_argument("--a-min", type=float, default=None,
                    help="inverse 标定的斜率 a 下限（限幅；b 重新取均值）")
    ap.add_argument("--a-max", type=float, default=None,
                    help="inverse 标定的斜率 a 上限（v4：1.0/1.3/1.6/2.0 扫描）")
    ap.add_argument("--cluster-k", type=int, default=0,
                    help="v6 聚类条件标定：在训练地形 val 回合的支撑相窗口上做 "
                         "k-means（k=0 关闭，保持与全局标定逐位一致）")
    ap.add_argument("--cluster-seed", type=int, default=0,
                    help="k-means 初始化种子（k-means++ 是随机的，必须固定）")
    ap.add_argument("--cluster-iters", type=int, default=50)
    args = ap.parse_args(argv)

    ckpt = Path(args.checkpoint)
    est = TactileEstimator.load(ckpt)
    ref = float(est.cfg.force_ref_n)
    n_feat = int(args.features or est.n_features)
    if n_feat != int(est.n_features):
        raise SystemExit(f"--features={n_feat} 与权重 n_features={est.n_features} "
                         f"不一致：标定与评估必须同列")
    files = sorted(Path(args.data_dir).glob("train_*.npz"))
    if not files:
        raise SystemExit(f"{args.data_dir} 里没有 train_*.npz")
    names = [f.stem.split("_", 1)[1] if "_" in f.stem else f.stem for f in files]
    leaked = [n for n in names if is_holdout_name(n)]
    if leaked:
        raise SystemExit(f"标定数据里出现留出地形 {leaked}：这会循环论证，拒绝执行")
    # 数据宽度守卫（评审 R2 同款）：更宽=显式切前缀，更窄=报错
    data_features = 0
    for f0 in files:
        w0 = int(load_dataset(f0).get("meta", {}).get("n_features", 0) or 0)
        if w0 and w0 < n_feat:
            raise SystemExit(f"{f0.name} 的特征宽度 {w0} < 标定列数 {n_feat}")
        data_features = max(data_features, w0)
    if data_features > n_feat:
        print(f"[info] 数据比权重宽 {data_features - n_feat} 列：按前 {n_feat} 列前缀标定")

    # 与训练脚本完全一致地拼窗口（同文件顺序、同 episode 偏移、同 stride）
    parts = []
    for i, f in enumerate(files):
        data = load_dataset(f)
        # 标定拟合用原始（未限幅）标签
        w = slice_features(make_dataset_windows(
            data, dataclasses.replace(est.cfg, label_clip_mg=0.0), args.stride), n_feat)
        w["episode"] = w["episode"] + i * 100000
        w["_case"] = np.array([i] * len(w["x"]))
        parts.append(w)
    wins = merge_windows(parts)
    tr_mask, val_mask = split_episodes(wins["episode"], est.cfg.val_frac,
                                      est.cfg.seed)
    if not val_mask.any():
        raise SystemExit("val 回合为空：val_frac 需 > 0")
    ep = np.asarray(wins["episode"])
    val_episode_ids = {names[i]: sorted(int(e) for e in np.unique(ep[val_mask & (wins["_case"] == i)]))
                       for i in range(len(files))}
    val_episode_ids = {k: v for k, v in val_episode_ids.items() if v}

    pred = est.predict(wins["x"])
    vals = channel_values(pred, ref)
    contact = np.asarray(wins["contact"]) > 0.5
    phase = np.asarray(wins["phase"])
    # val_mask 是「窗口」级 (N,)，contact/phase 是「足」级 (N,4) ⇒ 需要显式广播
    mask = contact & (phase > 2.5) & np.asarray(val_mask)[:, None]
    n_frames = int(mask.sum())
    if n_frames == 0:
        raise SystemExit("val 回合里没有支撑相接触帧，无法标定")
    t = np.asarray(wins["forcenorm"], dtype=np.float64)[mask]
    label_source = "sensor" if "force_norm" in load_dataset(files[0]) else "proxy"

    channels: dict[str, dict] = {}
    for k in CHANNELS:
        if k not in vals:
            continue
        y = np.asarray(vals[k], dtype=np.float64)[mask]
        if y.size < int(args.min_samples):
            print(f"  跳过通道 {k}：样本 {y.size} < {args.min_samples}")
            continue
        channels[k] = {
            "ls": fit_affine(y, t, method="ls"),
            "inverse": fit_affine(y, t, method="inverse",
                                  a_min=args.a_min, a_max=args.a_max),
        }
        ls, inv = channels[k]["ls"], channels[k]["inverse"]
        print(f"  {k:10s} ls a={ls['a']:.4f} b={ls['b']:+.4f} rmse "
              f"{ls['rmse_before']:.4f}->{ls['rmse_after']:.4f} slope_after={ls['slope_after']:.3f} | "
              f"inverse a={inv['a']:.4f} slope_after={inv['slope_after']:.3f} "
              f"rmse {inv['rmse_after']:.4f}")

    if not channels:
        raise SystemExit("没有任何通道可标定（样本不足）")

    # ------------------------------------------------------------------
    # v6 聚类条件标定（--cluster-k > 0）：全局一个仿射在「平地 / 低台阶 /
    # 高台阶」之间被折中掉，聚类条件标定按「窗口嵌入所属档位」给不同 (a,b)。
    # 嵌入 = [mean(特征), std(特征)]（20x65 窗口折成 130 维），只用策略可见特征，
    # 不碰地形真值 ⇒ 推理时可复现（这是可部署的「隐式地形档位」）。
    # ------------------------------------------------------------------
    clustered = None
    cluster_k = int(args.cluster_k)
    if cluster_k > 0:
        t_all = np.asarray(wins["forcenorm"], dtype=np.float64)
        stance_win = ((contact & (phase > 2.5))
                      & np.asarray(val_mask)[:, None]).any(axis=1)
        if int(stance_win.sum()) < cluster_k:
            raise SystemExit(f"支撑相 val 窗口 {int(stance_win.sum())} 少于 k={cluster_k}")
        emb = window_embedding(wins["x"])
        x_fit = emb[stance_win]
        scaler = fit_standardizer(x_fit)
        z_fit = apply_standardizer(x_fit, scaler)
        km = fit_kmeans(z_fit, cluster_k, seed=int(args.cluster_seed),
                        iters=int(args.cluster_iters))
        lab_fit = assign_clusters(z_fit, km["centers"])
        lab_win = np.full(int(emb.shape[0]), -1, dtype=np.int64)
        lab_win[stance_win] = lab_fit
        per_win = lab_win[:, None]
        clusters_channels: dict[str, dict[str, list]] = {}
        for k in CHANNELS:
            if k not in vals:
                continue
            y_all = np.asarray(vals[k], dtype=np.float64)
            per_method: dict[str, list] = {}
            for method in METHODS:
                fits = []
                for c in range(cluster_k):
                    m_c = mask & (per_win == c)
                    if int(m_c.sum()) < 2:
                        # 小簇：退化为全局系数，避免 a=1,b=0 这种静默错误
                        g = channels[k][method]
                        fits.append({"a": float(g["a"]), "b": float(g["b"]),
                                     "n": int(m_c.sum()), "cluster": c,
                                     "fallback_global": True})
                        continue
                    fit = fit_affine(y_all[m_c], t_all[m_c], method=method,
                                     a_min=args.a_min, a_max=args.a_max)
                    fit["cluster"] = c
                    fit["fallback_global"] = False
                    fits.append(fit)
                per_method[method] = fits
            clusters_channels[k] = per_method
            inv = per_method["inverse"]
            print("  cluster " + str(k) + ": " + ", ".join(
                "c%d a=%.3f(n=%d%s)" % (ft["cluster"], ft["a"], ft["n"],
                                        ",fb" if ft.get("fallback_global") else "")
                for ft in inv))
        clustered = {
            "k": cluster_k,
            "embedding": "mean_std",
            "scaler_mean": [float(v) for v in scaler["mean"]],
            "scaler_std": [float(v) for v in scaler["std"]],
            "centers": [[float(v) for v in row] for row in km["centers"]],
            "counts": [int(v) for v in km["counts"]],
            "inertia": float(km["inertia"]),
            "n_iter": int(km["n_iter"]),
            "seed": int(args.cluster_seed),
            "iters": int(args.cluster_iters),
            "n_windows_stance": int(stance_win.sum()),
            "channels": clusters_channels,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_by_case = {}
    for i, f in enumerate(files):
        meta_by_case[names[i]] = dict(load_dataset(f).get("meta", {}))
    payload = {
        "checkpoint": str(ckpt),
        "ckpt_sha256": sha256_file(ckpt),
        "features": n_feat,
        "data_features": data_features,
        "stride": int(args.stride),
        # seed 仅作记录：拟合是闭式最小二乘、无随机性（R6）
        "seed": int(args.seed),
        "seed_used": False,
        "cases": names,
        # 标定拟合源（评审 B4：必须能从 json 直接看出用的是哪个数据目录）
        "data_dir": str(args.data_dir),
        # 键必须与 ckpt meta.data_sha256 一致（含 train_ 前缀与 .npz 后缀），
        # 否则 eval 的绑定校验会永远不匹配（评审 N3 的落地形态）
        "case_sha256": {f.name: sha256_file(f) for f in files},
        "case_meta": {k: {kk: v[kk] for kk in ("policy", "vx", "episodes")
                          if kk in v} for k, v in meta_by_case.items()},
        "label_source": label_source,
        "mask": MASK_DESC,
        "val_frac": float(est.cfg.val_frac),
        "split_seed": int(est.cfg.seed),
        "val_episode_ids": val_episode_ids,
        "assert_no_holdout": True,
        "holdout_names": list(HOLDOUT_NAMES),
        "n_windows": int(wins["x"].shape[0]),
        "n_windows_val": int(val_mask.sum()),
        "n_frames": n_frames,
        "a_bounds": {"min": args.a_min, "max": args.a_max},
        "channels": channels,
    }
    if clustered is not None:
        payload["clustered"] = clustered
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"val 回合 {sum(len(v) for v in val_episode_ids.values())} 个 / "
          f"{n_frames} 帧；json_sha256={sha256_file(out)[:16]}  saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
