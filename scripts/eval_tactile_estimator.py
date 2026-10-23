"""评估 GRU 足底触觉状态估计器（留出地形 + 训练地形对照）。

用法::

    .venv\\Scripts\\python.exe scripts/eval_tactile_estimator.py \
        --checkpoint .tmp/tactile_estimator/estimator.pt \
        --data-dir .tmp/tactile_estimator/data \
        --out .tmp/tactile_estimator/report.md

对每个数据集（train / test 都跑，用于看 in-sample vs out-of-sample）计算：
三维力 RMSE/MAE（**世界系**，N 与 %体重、全量与仅接触帧、**分轴**）、接触
检测准确率/精确率/召回率/F1、面积与 CoP 的 RMSE/R²。

**足体系不单独报告**：`make_dataset_windows` 会丢弃采集 npz 里的 `force_foot`，
且数据集未保存每步足体旋转矩阵 `foot_rot`，无法把世界系预测旋到足体系；
世界系指标已覆盖交付口径（见 `doc/tactile_estimator.md` §5）。

产物：`<out>`（markdown）、`<out>.json`（机器可读）、`<out>.axes.txt`（分轴表）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dataclasses

import numpy as np
import torch

from robotdog.tactile_estimator import (  # noqa: E402
    TactileEstimator, apply_affine, apply_clustered_affine, apply_standardizer,
    assign_clusters, contact_latency, contact_metrics, fit_affine, force_metrics,
    load_dataset, make_dataset_windows, per_foot_constant_baseline,
    regression_metrics, slice_features, stance_force_metrics, window_embedding,
)


def _eval_cfg(cfg):
    """评估口径：训练侧的标签限幅不能作用于真值（否则 R² 是跟被截断的标签比）。"""
    return dataclasses.replace(cfg, label_clip_mg=0.0)


def _r2(y: np.ndarray, t: np.ndarray) -> float:
    t = np.asarray(t, dtype=np.float64)
    ss = float(np.sum((t - t.mean()) ** 2))
    if ss <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((np.asarray(y, dtype=np.float64) - t) ** 2)) / ss


def decomp_triple(y: np.ndarray, t: np.ndarray, mask: np.ndarray) -> dict:
    """bias / slope / scatter 三元组（与 fit_affine 同一闭式解）。

    bias = mean(y)-mean(t)（均值偏移）、slope = cov(y,t)/var(t)（幅度收缩/放大）、
    scatter = 仿射标定后的 R²（不可由仿射修正的那部分散布）。
    """
    y = np.asarray(y, dtype=np.float64)[mask]
    t = np.asarray(t, dtype=np.float64)[mask]
    if y.size < 2 or float(np.var(t)) <= 0.0:
        return {"bias": float("nan"), "slope": float("nan"),
                "scatter": float("nan"), "n": int(y.size)}
    fit = fit_affine(y, t)
    return {"bias": float(y.mean() - t.mean()),
            "slope": float(np.cov(y, t, bias=True)[0, 1] / np.var(t)),
            "scatter": float(_r2(apply_affine(y, fit["a"], fit["b"]), t)),
            # scatter 是**评估集 in-sample 仿射**（oracle 上界），不是「训练回合
            # 拟合、留出集评估」的标定后残差；低 n case 会偏乐观（评审 N4）。
            "scatter_kind": "affine_oracle_in_sample",
            "a_ls": float(fit["a"]), "b_ls": float(fit["b"]),
            "n": int(y.size)}


def cluster_bootstrap_r2(y: np.ndarray, t: np.ndarray, episode: np.ndarray,
                         n_boot: int = 2000, seed: int = 0) -> dict:
    """按回合整群重采样的 R² 置信区间（窗口 50% 重叠 + 回合内自相关 ⇒ 不能按窗口重采样）。"""
    eps = np.unique(episode)
    out = {"n_episodes": int(eps.size), "n_boot": int(n_boot)}
    if eps.size < 2:
        out.update({"degenerate": True, "ci_low": float("nan"),
                    "ci_high": float("nan"), "note": "单回合，无法估计群间方差"})
        return out
    rng = np.random.default_rng(int(seed))
    idx_by_ep = {float(e): np.flatnonzero(episode == e) for e in eps}
    vals = []
    for _ in range(int(n_boot)):
        pick = rng.choice(eps, size=eps.size, replace=True)
        idx = np.concatenate([idx_by_ep[float(e)] for e in pick])
        vals.append(_r2(y[idx], t[idx]))
    arr = np.asarray(vals, dtype=np.float64)
    out.update({"degenerate": False,
                "ci_low": float(np.nanpercentile(arr, 2.5)),
                "ci_high": float(np.nanpercentile(arr, 97.5)),
                "boot_sd": float(np.nanstd(arr, ddof=1))})
    return out


# 评估必须覆盖的留出地形（P1 判据只取 stair16_5/stair18；缺文件会静默少行，评审 B6）
# v5：7 个留出地形（插值 10.5/13.5、外推 16.5/18/19.5、rough seed1/2）
REQUIRED_CASES = ("stair16_5", "stair18", "stair19_5", "stair10_5",
                  "stair13_5", "rough_s1", "rough_s2")

# 斜率限幅扫描网格（S4 的判据）。并集 8 值：多出来的值是「描出曲线」用的，
# **只有 PREREG_A 里的 4 个进判定**（预注册在 task_book rev2，事后扩网格不上判据）。
CLAMP_GRID = (1.0, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.5)
PREREG_A = (1.0, 1.3, 1.6, 2.0)


def sha256_file(path: Path, limit: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(limit)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def check_calibration(cal: dict, ckpt: Path, n_features: int, stride: int) -> dict:
    """标定 json 必须来自同一个 checkpoint / 同列特征 / 同 stride（评审 B4）。

    否则 smoke 或 c0T 的系数能被 t3 无警告消费，标定表就不再是「该权重的后验」。
    """
    sha = sha256_file(ckpt)
    bad = []
    # ckpt_sha256 必须存在：缺键不能静默跳过（评审 B8）
    if not cal.get("ckpt_sha256"):
        bad.append("标定 json 缺 ckpt_sha256（无法证明是这个权重）")
    elif cal["ckpt_sha256"] != sha:
        bad.append(f"ckpt_sha256 {cal['ckpt_sha256'][:16]} != 实际 {sha[:16]}")
    if int(cal.get("features", -1)) != int(n_features):
        bad.append(f"features {cal.get('features')} != {n_features}")
    if int(cal.get("stride", -1)) != int(stride):
        bad.append(f"stride {cal.get('stride')} != {stride}")
    # case_sha256 绑定（评审 N3）：7 个 case 中途重采过，旧标定不能静默套用
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    ck_sha = dict((ck.get("meta") or {}).get("data_sha256") or {})
    cal_sha = {}
    for k, v in (cal.get("case_sha256") or {}).items():
        key = k if k.startswith("train_") else f"train_{k if k.endswith('.npz') else k + '.npz'}"
        cal_sha[key] = v
    if ck_sha:
        # ckpt 有 v5 provenance 时，标定必须指向同批数据（缺键/不一致都阻断）
        if not cal_sha:
            bad.append("标定 json 缺 case_sha256（无法证明拟合数据与权重训练数据同批）")
        else:
            diff = {k for k in set(cal_sha) | set(ck_sha)
                    if cal_sha.get(k) != ck_sha.get(k)}
            if diff:
                bad.append(f"case_sha256 与 ckpt data_sha256 不一致"
                           f"（{len(diff)} 个文件）")
    else:
        cal["_case_sha_note"] = "ckpt 无 data_sha256（历史权重）：跳过逐文件绑定校验"
    if bad:
        raise SystemExit("标定文件与本次评估不匹配：" + "；".join(bad))
    cal["_ckpt_sha256"] = sha
    return cal


def latency_metrics(est: TactileEstimator, data: dict) -> dict[str, float]:
    """接触起始帧的检出时延（stride=1 全帧覆盖，按回合求平均）。"""
    wins = slice_features(make_dataset_windows(data, _eval_cfg(est.cfg), 1),
                          est.n_features)
    pred = est.predict(wins["x"])
    prob, tgt, ep = pred["contact"], wins["contact"], wins["episode"]
    rows = []
    for e in np.unique(ep):
        m = ep == e
        if int(m.sum()) < 8:      # 太短的回合没有统计意义
            continue
        rows.append(contact_latency(prob[m], tgt[m],
                                    threshold=est.cfg.contact_threshold))
    rows = [r for r in rows if r["n_onsets"] > 0]
    if not rows:
        return {"n_onsets": 0.0}
    keys = [k for k in rows[0] if k != "n_onsets"]
    agg = {"n_onsets": float(sum(r["n_onsets"] for r in rows))}
    for k in keys:
        vals = [r[k] for r in rows if np.isfinite(r[k])]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    return agg


def eval_dataset(est: TactileEstimator, data: dict, stride: int = 2,
                 latency: bool = False, n_features: int = 0,
                 calibration: dict | None = None, bootstrap: int = 0,
                 bootstrap_seed: int = 0):
    cfg = est.cfg
    ref = float(cfg.force_ref_n)
    # 特征宽度默认对齐 checkpoint（v3 数据集 65 列、旧权重 53 列 ⇒ 自动切前缀）；
    # --features 可显式覆盖（例如 53 的纯观测量臂 vs 65 的力矩臂）。
    wins = slice_features(make_dataset_windows(data, _eval_cfg(cfg), stride),
                          n_features or est.n_features)
    pred = est.predict(wins["x"])
    mask = wins["contact"] > 0.5
    mask2 = np.repeat(mask[:, :, None], 2, axis=2)
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    out = {
        "n_windows": int(wins["x"].shape[0]),
        # 只报世界系：模型预测的就是世界系；足体系需要每步足体旋转矩阵（数据集没存），
        # 拿世界系预测去比足体系标签会得到一个「看起来也是力误差」的错误数字。
        "force_world": force_metrics(pred["force"], wins["force"], cfg.force_ref_n),
        "contact": contact_metrics(pred["contact"], wins["contact"],
                                   cfg.contact_threshold),
        "area": regression_metrics(pred["area"], wins["area"], mask),
        "cop": regression_metrics(pred["cop"], wins["cop"], mask2),
    }
    if mask3.any():
        # 只看「接触足」的三维力误差：把 4 足维度摊平后按足-元素取掩码
        err = (pred["force"] - wins["force"])[mask3]
        out["force_world_contact_only"] = {
            "rmse_n": float(np.sqrt(np.mean(err ** 2))),
            "mae_n": float(np.mean(np.abs(err))),
            "rmse_bw": float(np.sqrt(np.mean(err ** 2)) / ref),
            "mae_bw": float(np.mean(np.abs(err)) / ref),
            "p95_n": float(np.percentile(np.abs(err), 95)),
        }
    # 支撑相（phase=STANCE=3）的三维力误差：区分「落地冲击帧」与「稳态支撑帧」
    ph = np.asarray(wins["phase"])
    stance = mask & (ph > 2.5)
    stance3 = np.repeat(stance[:, :, None], 3, axis=2)
    if stance3.any():
        err_s = (pred["force"] - wins["force"])[stance3]
        out["force_world_stance_only"] = {
            "rmse_n": float(np.sqrt(np.mean(err_s ** 2))),
            "mae_n": float(np.mean(np.abs(err_s))),
            "rmse_bw": float(np.sqrt(np.mean(err_s ** 2)) / ref),
            "p95_n": float(np.percentile(np.abs(err_s), 95)),
        }
    # 消费端口径：接触足上的 F̄ 幅度质量（R² / 斜率 / 均值比 / p95 比）。
    # v1 权重没有 F̄ 头，用「世界系竖直分量」和「合力模长」两个可替代量做对照。
    tgt_fn = wins["forcenorm"]
    # v7 Wave C 双频：带头且 cfg.bifreq_k>0 时，F̄ 头学到的是「因果 k 帧均值」，
    # 必须与同口径目标比，否则会把低频头当高频头评（口径错配）。
    bifreq_k = int(getattr(est.cfg, "bifreq_k", 0) or 0)
    tgt_fn_head = (wins["forcenorm_mean"]
                   if (bifreq_k > 0 and "forcenorm_mean" in wins) else tgt_fn)
    contact = wins["contact"]
    vals = {
        "vector_z": pred["force"][:, :, 2] / ref,
        "vector_mag": np.linalg.norm(pred["force"], axis=2) / ref,
    }
    if "forcenorm" in pred:
        vals["head"] = pred["forcenorm"]
    # 拟合域与报告域必须一致：contact & phase>2.5（支撑相，排除落地冲击帧）。
    consumer = {"label_source": "sensor" if "force_norm" in data else "proxy",
                "mask": "stance(phase>2.5) & contact",
                "head_target": ("causal_mean_k" if bifreq_k > 0 else "instant"),
                "bifreq_k": bifreq_k,
                "debounce_contact": bool(getattr(est.cfg, "debounce_contact", False))}
    for key, y in vals.items():
        tgt_use = tgt_fn_head if key == "head" else tgt_fn
        consumer[key] = stance_force_metrics(y, tgt_use, contact, force_ref_n=ref,
                                             phase=ph)
    for key in ("delta", "peak"):
        if key in pred:
            consumer[key] = stance_force_metrics(pred[key], wins[key], contact,
                                                 force_ref_n=ref, phase=ph)
    # 后验仿射标定（y' = a*y + b，来自 scripts/calibrate_tactile_estimator.py）：
    # 只改报告口径、不改权重；标定后仍用同一套 stance+contact 掩码与真值比较。
    if calibration:
        channels = dict(calibration.get("channels") or {})
        calib: dict[str, dict[str, float]] = {}
        for key, y in vals.items():
            c = channels.get(key) or {}
            for method in ("ls", "inverse"):
                fit = c.get(method) if isinstance(c.get(method), dict) else None
                if not fit:
                    continue
                m = stance_force_metrics(apply_affine(y, fit["a"], fit["b"]), tgt_fn,
                                         contact, force_ref_n=ref, phase=ph)
                m["a"], m["b"] = float(fit["a"]), float(fit["b"])
                m["method"] = method
                calib[f"{key}.{method}"] = m
        consumer["calibrated"] = calib
        # 斜率限幅扫描（S4）：只改 a、b 按 mean_t - a*mean_y 重算（与 fit_affine 同式），
        # 因此「斜率换 R²」的折中曲线可复算。a_raw 来自 inverse 拟合。
        sweeps: dict[str, list[dict[str, float]]] = {}
        for key, y in vals.items():
            c = channels.get(key) or {}
            fit = c.get("inverse") if isinstance(c.get("inverse"), dict) else None
            if not fit or "mean_y" not in fit or "mean_t" not in fit:
                continue
            mean_y, mean_t = float(fit["mean_y"]), float(fit["mean_t"])
            a_raw = float(fit["a"])
            cells = []
            for a in CLAMP_GRID:
                b = mean_t - a * mean_y
                m = stance_force_metrics(apply_affine(y, a, b), tgt_fn, contact,
                                         force_ref_n=ref, phase=ph)
                cells.append({"a": float(a), "b": float(b), "a_raw": a_raw,
                              "clamped": bool(abs(a - a_raw) > 1e-12),
                              "r2": float(m["r2"]), "slope": float(m["slope"]),
                              "mean_ratio": float(m["mean_ratio"]),
                              "p95_ratio": float(m["p95_ratio"]),
                              "n": float(m["n"])})
            sweeps[key] = cells
        consumer["clamp_sweep"] = sweeps
        consumer["clamp_grid"] = [float(a) for a in CLAMP_GRID]
        consumer["clamp_prereg"] = [float(a) for a in PREREG_A]
        consumer["calibration_file"] = str(calibration.get("_path", ""))
        consumer["calibration_mask"] = str(calibration.get("mask", ""))
        # ---- v6 聚类条件标定：每个窗口按最近簇心选自己的 (a,b) ----
        # 簇心/标准化统计量都来自标定 json（训练地形 val 回合），评估侧只做
        # 同式嵌入 + 最近中心分配 ⇒ 不是用留出地形拟合，也不含地形真值。
        cl = calibration.get("clustered") or None
        if cl:
            kk = int(cl.get("k") or 0)
            scaler = {"mean": np.asarray(cl["scaler_mean"], dtype=np.float64),
                      "std": np.asarray(cl["scaler_std"], dtype=np.float64)}
            centers = np.asarray(cl["centers"], dtype=np.float64)
            emb = window_embedding(wins["x"])
            if emb.shape[1] != scaler["mean"].size:
                raise SystemExit(
                    f"聚类标定的嵌入宽度 {scaler['mean'].size} 与本次窗口 "
                    f"{emb.shape[1]} 不一致（window/features 改变了？）")
            lab = assign_clusters(apply_standardizer(emb, scaler), centers)
            calib_c: dict[str, dict[str, float]] = {}
            for key, y in vals.items():
                per = (cl.get("channels") or {}).get(key) or {}
                for method in ("ls", "inverse"):
                    fits = per.get(method)
                    if not isinstance(fits, list) or len(fits) != kk:
                        continue
                    y_c = apply_clustered_affine(y, lab, fits, kk)
                    m = stance_force_metrics(y_c, tgt_fn, contact,
                                             force_ref_n=ref, phase=ph)
                    m["method"] = method
                    m["cluster_k"] = kk
                    m["cluster_counts"] = [int(v) for v in cl.get("counts") or []]
                    calib_c[f"{key}.{method}"] = m
            consumer["calibrated_cluster"] = calib_c
    out["consumer"] = consumer
    # ---- v5：掩码后样本量 + bias/slope/scatter 三元组 + 回合级 bootstrap ----
    n_contact = int(contact.sum())
    n_stance = int(stance.sum())
    out["case_stats"] = {
        "n_windows": int(wins["x"].shape[0]),
        "n_contact": n_contact,
        "n_stance": n_stance,
        "stance_ratio": (float(n_stance) / float(n_contact)) if n_contact else 0.0,
        "n_episodes": int(np.unique(wins["episode"]).size),
        "low_n": bool(n_stance < 300),
    }
    out["decomp"] = {key: decomp_triple(y, tgt_fn, stance)
                     for key, y in vals.items()}
    if bootstrap and bootstrap > 0:
        ep_w = np.repeat(wins["episode"][:, None], 4, axis=1)[stance]
        out["bootstrap"] = cluster_bootstrap_r2(
            vals["vector_mag"][stance], tgt_fn[stance], ep_w,
            n_boot=int(bootstrap), seed=int(bootstrap_seed))
    # 旧口径（v2 及更早的「接触足」）：只掩 contact，含落地/离地帧。
    # 保留它是为了让 v3 数字能与历史文档对比，**不作为主判据**（掩码里近零力的
    # 帧会把 R² 抬高，见 doc/tactile_estimator.md 的口径说明）。
    legacy = {"label_source": consumer["label_source"], "mask": "contact"}
    for key, y in vals.items():
        legacy[key] = stance_force_metrics(y, tgt_fn, contact, force_ref_n=ref)
    out["consumer_contact"] = legacy
    # 弱基线：每足每轴常数（只用本数据集均值）—— GRU 到底赢多少
    out["constant_baseline"] = per_foot_constant_baseline(wins["force"])
    if latency:
        out["contact_latency"] = latency_metrics(est, data)
    return out, wins, pred


def fmt_force(m: dict) -> str:
    return (f"RMSE {m['rmse_all_n']:6.1f} N ({m['rmse_all_bw']:.3f} BW)  "
            f"MAE {m['mae_all_n']:6.1f} N  p95 {m['p95_all_n']:6.1f} N  "
            f"R2 {m['r2_all']:+.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=".tmp/tactile_estimator/estimator.pt")
    ap.add_argument("--data-dir", default=".tmp/tactile_estimator/data")
    ap.add_argument("--out", default=".tmp/tactile_estimator/report.md")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--latency", action="store_true",
                    help="额外跑 stride=1 全帧覆盖，统计触地起始帧检出时延")
    ap.add_argument("--features", type=int, default=0,
                    help="特征列数（0=按 checkpoint 的 n_features 自动切前缀；53=纯观测基线）")
    ap.add_argument("--calibration", default="",
                    help="后验仿射标定 json（只改报告口径，不改权重）")
    ap.add_argument("--allow-missing-cases", action="store_true",
                    help="允许缺少数留出地形（默认缺失 REQUIRED_CASES 直接报错）")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="回合级 bootstrap 次数（0=不算；建议 2000）")
    ap.add_argument("--bootstrap-seed", type=int, default=0,
                    help="bootstrap 随机种子（固定后才可复现 CI）")
    args = ap.parse_args()
    est = TactileEstimator.load(args.checkpoint)
    cfg = est.cfg
    ckpt = Path(args.checkpoint)
    n_feat = int(args.features or est.n_features)
    if n_feat != int(est.n_features):
        raise SystemExit(
            f"--features={n_feat} 与权重 n_features={est.n_features} 不一致："
            f"53 列纯观测臂必须用 53 列权重；65 列数据只切前 53 列会静默丢掉力矩列")
    cal = None
    if args.calibration:
        cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        cal["_path"] = str(Path(args.calibration))
        cal = check_calibration(cal, ckpt, n_feat, args.stride)
    files = sorted(Path(args.data_dir).glob("*.npz"))
    if not files:
        raise SystemExit(f"{args.data_dir} 里没有 npz")
    present = {f.stem.split("_", 1)[1] if "_" in f.stem else f.stem for f in files}
    missing = [c for c in REQUIRED_CASES if c not in present]
    if missing and not args.allow_missing_cases:
        raise SystemExit(f"缺少留出地形 {missing}（确实只评部分数据集请加 "
                         f"--allow-missing-cases）；当前 {args.data_dir} 只有 {sorted(present)}")
    # 数据宽度守卫（评审 R2）：读 meta 而不是从文件名猜。
    # 评估列数已强制等于权重宽度；数据更宽（65 vs 53）等价于「只用本体感觉前缀」，
    # 是 c0 这类纯观测臂的既定设计，必须显式提示而不是静默；数据更窄一律报错。
    prefix_dropped = 0
    for f0 in files:
        m0 = load_dataset(f0).get("meta", {})
        w0 = int(m0.get("n_features", 0) or 0)
        if w0 and w0 < n_feat:
            raise SystemExit(f"{f0.name} 的特征宽度 {w0} < 评估列数 {n_feat}："
                             f"无法用更宽的权重评估更窄的数据")
        if w0 > n_feat:
            prefix_dropped = max(prefix_dropped, w0 - n_feat)
    if prefix_dropped:
        print(f"[info] 数据比权重宽 {prefix_dropped} 列：按前 {n_feat} 列前缀评估"
              f"（多出的力矩/额外列未使用）")
    rows = []
    header = [
        "# 足底触觉 GRU 估计器 · 评估报告",
        "",
        f"- 权重：`{args.checkpoint}`（window={cfg.window}, hidden={cfg.hidden}, "
        f"layers={cfg.layers}, force_ref={cfg.force_ref_n:.2f} N）",
        "- 标签：`TactileSensor` 的足底三维力 / 接触 / 面积 / CoP（仿真测量）",
        "- 输入：策略可见观测（53 维本体感觉 + 指令 + step_height）"
        f"（+ 可选 12 维归一化关节力矩），窗口 {cfg.window} 步"
        f"（{cfg.window * 0.02:.2f} s @50 Hz）；本报告用特征列数 = {n_feat}",
        "",
        "| 数据集 | split | 回合/窗口 | 三维力（世界系，全量） | 接触 Acc/F1 | "
        "面积 RMSE/R² | CoP RMSE/R² | 支撑相 F̄ R²（斜率/均值比, n_stance） | 弱基线 R² |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        "",
        "> 主表数字**全部未标定**（标定只作用于下面「后验仿射标定」小节）；"
        "「支撑相 F̄」= `contact & phase>2.5`（排除落地冲击帧）上的 vector_mag。",
    ]
    for f in files:
        data = load_dataset(f)
        meta = data.get("meta", {})
        res, _wins, _pred = eval_dataset(est, data, args.stride,
                                        latency=args.latency,
                                        n_features=args.features,
                                        calibration=cal,
                                        bootstrap=args.bootstrap,
                                        bootstrap_seed=args.bootstrap_seed)
        name = meta.get("name", f.stem)
        split = meta.get("split", f.stem.split("_")[0])
        fm = res["force_world"]
        cm = res["contact"]
        am = res["area"]
        pm = res["cop"]
        cons = res["consumer"]
        # 主口径固定为 vector_mag（|F|/mg，所有 checkpoint 都有的通道、可直接跨版本比较）；
        # head 只在带 F̄ 头的权重上存在，单列报告。consumer_best 仅作参考，必须记录用了哪个候选，
        # 否则「候选数不同的两次实验取 max」会系统性偏袒候选更多的权重（评审 B2）。
        cands = {"vector_z": cons["vector_z"], "vector_mag": cons["vector_mag"]}
        if "head" in cons:
            cands["head"] = cons["head"]
        primary = cons["vector_mag"]
        best_name = max(cands, key=lambda k: cands[k]["r2"]
                        if np.isfinite(cands[k]["r2"]) else -9.0)
        res["consumer_best"] = dict(cands[best_name], candidate=best_name)
        rows.append({"name": name, "split": split, "file": str(f),
                     "meta": meta, **res})
        head_txt = ""
        if "head" in cons:
            hd = cons["head"]
            # 同一格里追加，避免数据行比表头多一列（评审 B7）
            head_txt = (f"；F̄头 {hd['r2']:+.2f} "
                        f"({hd['slope']:.2f}/{hd['mean_ratio']:.2f})")
        header.append(
            f"| {name} | {split} | {meta.get('episodes', '?')}/{res['n_windows']} | "
            f"{fmt_force(fm)} | {cm['accuracy']:.3f}/{cm['f1']:.3f} | "
            f"{am['rmse']:.3f}/{am['r2']:+.2f} | {pm['rmse']:.3f}/{pm['r2']:+.2f} | "
            f"{primary['r2']:+.2f} ({primary['slope']:.2f}/{primary['mean_ratio']:.2f}, "
            f"n={res['case_stats']['n_stance']})"
            f"{head_txt} | "
            f"{res['constant_baseline']['r2_all']:+.2f} |")
        print(f"{split:5s} {name:18s} force {fmt_force(fm)} | contact "
              f"acc {cm['accuracy']:.3f} F1 {cm['f1']:.3f} | area {am['rmse']:.3f} "
              f"| cop {pm['rmse']:.3f} | stance F-bar(vector_mag) R2 "
              f"{primary['r2']:+.2f} ratio {primary['mean_ratio']:.2f} "
              f"n={res['case_stats']['n_stance']}{head_txt}")
    header += ["",
               "足体系三维力不单独报告：窗口化丢弃 `force_foot`、且未保存每步 `foot_rot`，",
               "无法把世界系预测旋到足体系；世界系指标已覆盖交付口径（doc §5）。",
               "",
               "## 消费端口径（支撑相接触足 F̄ 幅度）",
               "",
               "掩码 = `contact & phase>2.5`（STANCE），与标定拟合域一致。"
               "R² 的分母是支撑相帧间方差，才是「能不能拿去用」的指标。**主口径固定为 `vector_mag`**"
               "（= |F|/mg，所有 checkpoint 都有、可跨版本比较）；`vector_z` 是 Fz/mg；"
               "`head` 是消费端对齐权重专用的 F̄ 头，只在带头的权重上出现。"
               "报告里的 `consumer_best` 只是参考，且带 `candidate` 字段说明取了哪个候选——"
               "不同实验候选数不同时**不要**拿它做跨版本比较。",
               "",
               "| 数据集 | 标签源 | vector_z R²/均值比 | vector_mag R²/均值比 | "
               "head R²/斜率/均值比 | 支撑相三维力 RMSE |",
               "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        cons = row["consumer"]
        vz, vm, hd = cons["vector_z"], cons["vector_mag"], cons.get("head")
        st = row.get("force_world_stance_only") or {}
        head_cell = (f"{hd['r2']:+.2f}/{hd['slope']:.2f}/{hd['mean_ratio']:.2f}"
                     if hd else "—")
        header.append(
            f"| {row['name']} | {cons['label_source']} | "
            f"{vz['r2']:+.2f}/{vz['mean_ratio']:.2f} | "
            f"{vm['r2']:+.2f}/{vm['mean_ratio']:.2f} | {head_cell} | "
            f"{st.get('rmse_n', float('nan')):.1f} N |")
    header += ["",
               "## 旧口径对照（只掩 contact，含落地/离地帧）",
               "",
               "该口径把近零力的触地/离地帧也算进 R² 的分母，数字会被抬高；v2 及更早"
               "文档里的「stance F̄」用的就是它，**不能**与上表支撑相口径直接比较。",
               "",
               "| 数据集 | vector_z R²/斜率/均值比 | vector_mag R²/斜率/均值比 | "
               "head R²/斜率/均值比 |",
               "| --- | --- | --- | --- |"]
    for row in rows:
        lg = row.get("consumer_contact") or {}
        cells = []
        for k in ("vector_z", "vector_mag", "head"):
            m = lg.get(k)
            cells.append(f"{m['r2']:+.2f}/{m['slope']:.2f}/{m['mean_ratio']:.2f}"
                         if m else "—")
        header.append(f"| {row['name']} | " + " | ".join(cells) + " |")
    header += ["",
               "## 样本量与误差结构（支撑相口径）",
               "",
               "R² 是 bias（均值偏移）/ slope（幅度收缩）/ scatter（仿射后残差）的混合："
               "仿射标定只能修前两项。n_stance < 300 标 LOW_N，该 case 的 R² 波动大，"
               "不能单独用来判定一个小效应。",
               "",
               "| 数据集 | n_stance | stance/contact | 回合 | vector_mag bias/slope/scatter | "
               "head bias/slope/scatter | R² 95% CI（回合级 bootstrap） |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        st = row.get("case_stats") or {}
        dcp = row.get("decomp") or {}

        def _dc(chan: str) -> str:
            d = dcp.get(chan)
            if not d:
                return "—"
            return (f"{d['bias']:+.3f}/{d['slope']:.2f}/{d['scatter']:+.2f}"
                    f" (n={d['n']})")

        boot = row.get("bootstrap") or {}
        if boot and not boot.get("degenerate"):
            ci = (f"[{boot['ci_low']:+.2f}, {boot['ci_high']:+.2f}] "
                  f"sd {boot.get('boot_sd', float('nan')):.2f} "
                  f"({boot['n_boot']}×{boot['n_episodes']} 回合)")
        else:
            ci = "—（未做或单回合）"
        flag = " LOW_N" if st.get("low_n") else ""
        header.append(
            f"| {row['name']} | {st.get('n_stance', 0)}{flag} | "
            f"{(st.get('stance_ratio') or 0.0):.2f} | {st.get('n_episodes', '?')} | "
            f"{_dc('vector_mag')} | {_dc('head')} | {ci} |")
    if cal:
        header += ["",
                   "## 后验仿射标定（y' = a*y + b）",
                   "",
                   "系数在**训练地形的 val 回合**上拟合（与训练同文件顺序、同 "
                   "`episode + i*100000` 偏移、同 `split_episodes` seed）；"
                   "拟合域与评估域都是 `contact & phase>2.5`。标定只改报告口径、不改权重。",
                   "`ls` = 最小二乘 t≈a*y+b（RMSE 最优、收缩型，标定后斜率≈r²）；"
                   "`inverse` = 衰减偏差修正 a=var(t)/cov(y,t)（标定后斜率恒为 1、"
                   "RMSE 可能变差）。P3 的 slope 判据只看 inverse。",
                   "",
                   "| 数据集 | 通道 | 方法 | a | b | 标定后 R²/斜率/均值比 | p95 比 |",
                   "| --- | --- | --- | --- | --- | --- | --- |"]
        for row in rows:
            calib = (row.get("consumer") or {}).get("calibrated") or {}
            for key, m in calib.items():
                chan, _, method = key.partition(".")
                header.append(
                    f"| {row['name']} | {chan} | {method or '-'} | {m['a']:.3f} | "
                    f"{m['b']:+.3f} | "
                    f"{m['r2']:+.2f}/{m['slope']:.2f}/{m['mean_ratio']:.2f} | "
                    f"{m['p95_ratio']:.2f} |")
        header += ["",
                   "## 斜率限幅扫描（S4：只动 a，b 按 mean_t − a·mean_y 重算）",
                   "",
                   "对每个通道的 inverse 拟合，把 a 换成网格值并重算 b；判定只用**预注册**的 "
                   "1.0 / 1.3 / 1.6 / 2.0，其余值是画曲线用的（事后扩网格不进判据）。",
                   "",
                   "| 数据集 | 通道 | a | 斜率 | R² | 均值比 | p95 比 | 预注册 |",
                   "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in rows:
            sw = (row.get("consumer") or {}).get("clamp_sweep") or {}
            for chan, cells in sw.items():
                for cell in cells:
                    header.append(
                        f"| {row['name']} | {chan} | {cell['a']:.2f} | "
                        f"{cell['slope']:.2f} | {cell['r2']:+.2f} | "
                        f"{cell['mean_ratio']:.2f} | {cell['p95_ratio']:.2f} | "
                        f"{'yes' if cell['a'] in PREREG_A else ''} |")
    if args.latency:
        header += ["",
                   "## 触地起始帧检出时延（stride=1，按回合平均）",
                   "",
                   "| 数据集 | n_onsets | 起始帧召回 | 起始帧均值 p | "
                   "前一帧均值 p | 后一帧均值 p |",
                   "| --- | --- | --- | --- | --- | --- |"]
        for row in rows:
            lt = row.get("contact_latency") or {}
            if not lt or lt.get("n_onsets", 0) == 0:
                continue
            header.append(
                f"| {row['name']} | {int(lt['n_onsets'])} | "
                f"{lt['onset_recall']:.3f} | {lt['onset_prob_mean']:.3f} | "
                f"{lt['prev_prob_mean']:.3f} | {lt['next_prob_mean']:.3f} |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    # 分轴表由本脚本产出（不依赖任何 .tmp 一次性脚本）
    axes_lines = ["# 分轴力误差（世界系；all = 全体窗口，contact = 仅接触足窗口）",
                  "name split n rmse_x mae_x r2_x rmse_y mae_y r2_y rmse_z mae_z r2_z"]
    for row in rows:
        fm = row["force_world"]
        parts = [f"{row['name']} {row['split']} {int(row.get('n_windows', 0))}"]
        for axis in ("x", "y", "z"):
            parts.append(f"{fm[f'rmse_{axis}_n']:.2f} {fm[f'mae_{axis}_n']:.2f} "
                         f"{fm[f'r2_{axis}']:+.3f}")
        axes_lines.append(" ".join(parts))
        fmo = (row.get("force_world_contact_only") or {})
        if fmo:
            axes_lines.append(f"  contact-only {row['name']}: "
                              f"rmse {fmo['rmse_n']:.2f} N  "
                              f"mae {fmo['mae_n']:.2f} N  "
                              f"p95 {fmo['p95_n']:.2f} N")
        fms = (row.get("force_world_stance_only") or {})
        if fms:
            axes_lines.append(f"  stance-only  {row['name']}: "
                              f"rmse {fms['rmse_n']:.2f} N  "
                              f"mae {fms['mae_n']:.2f} N  "
                              f"p95 {fms['p95_n']:.2f} N")
        cb = row.get("constant_baseline") or {}
        best = row.get("consumer_best") or {}
        if best:
            axes_lines.append(f"  consumer     {row['name']}: "
                              f"[{best.get('candidate', '?')}] "
                              f"F-bar r2 {best['r2']:+.3f}  slope {best['slope']:.3f}  "
                              f"mean-ratio {best['mean_ratio']:.3f}  "
                              f"p95-ratio {best['p95_ratio']:.3f}  "
                              f"const-baseline-r2 {cb.get('r2_all', float('nan')):+.3f}")
        lt = row.get("contact_latency") or {}
        if lt.get("n_onsets", 0):
            axes_lines.append(f"  onset-latency {row['name']}: "
                              f"n {int(lt['n_onsets'])}  "
                              f"recall {lt['onset_recall']:.3f}  "
                              f"p(onset) {lt['onset_prob_mean']:.3f}  "
                              f"p(prev) {lt['prev_prob_mean']:.3f}  "
                              f"p(next) {lt['next_prob_mean']:.3f}")
    axes_out = out.with_suffix(".axes.txt")
    axes_out.write_text("\n".join(axes_lines) + "\n", encoding="utf-8")
    print(f"report: {out}")
    print(f"axes:   {axes_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
