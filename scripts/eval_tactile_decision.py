
# -*- coding: utf-8 -*-
"""Wave 0 靶子：决策量 + 时移容差（零训练，只读已有 checkpoint 与留出数据）。

口径（与 blockers 文档 7.5 一致）：
- 掩码一律取消费帧（label end frame）：contact>0.5 且 phase>2.5。
- 时移定义：delta>0 表示消费端「用旧估计」——把 pred(t-delta) 与 truth(t) 配对（delta=1 即
  消费时刻只有上一帧的估计可用）；delta<0 表示消费端拿到了未来的估计。
- delta_star = 最大 |delta| 使滞后/超前两侧 R2 都不低于 max(0.45, 0.8*R2(0))。

输出：--out 前缀写 .md / .json；多个 --checkpoint 时逐臂一节。
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotdog.tactile_estimator import TactileEstimator, load_dataset, make_dataset_windows  # noqa: E402

REQUIRED_CASES = ("stair10_5", "stair13_5", "stair16_5", "stair18", "stair19_5", "rough_s1", "rough_s2")
GROUPS = {
    "stairs_interp": ("stair10_5", "stair13_5"),
    "stairs_legacy": ("stair16_5", "stair18"),
    "stairs_extrap": ("stair16_5", "stair18", "stair19_5"),
    "rough": ("rough_s1", "rough_s2"),
}
DELTAS = (-3, -2, -1, 0, 1, 2, 3)
LOAD_BEARING = 0.5
PHASE_STANCE = 2.5
R2_MARGIN = 0.8
R2_FLOOR = 0.45


def _r2(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    sst = float(((true - true.mean()) ** 2).sum())
    if sst <= 1e-12 or pred.size < 2:
        return None
    return 1.0 - float(((pred - true) ** 2).sum()) / sst


def _rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    vals, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(cnt > 1):
        sums = np.zeros(len(vals))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    return ranks


def spearman_frames(pred, true, contact, phase):
    """逐帧「接触足之间的受力排序」Spearman，返回 (mean, n_frames)。

    只在该帧至少有 2 只足处于 contact 且处于支撑相时统计（否则排序无信息）。
    """
    c = (contact > 0.5) & (phase > PHASE_STANCE)
    vals = []
    for i in np.flatnonzero(c.sum(axis=1) >= 2):
        sub = c[i]
        p, t = pred[i][sub], true[i][sub]
        if float(t.std()) <= 1e-9 or float(p.std()) <= 1e-9:
            continue
        rp, rt = _rankdata(p), _rankdata(t)
        vals.append(float(np.corrcoef(rp, rt)[0, 1]))
    return (round(float(np.mean(vals)), 4) if vals else None), len(vals)


def load_bearing_f1(pred_mag, true_mag, contact):
    """只在真值接触的足-帧上算「是否承重(>50%BW)」的 F1。"""
    c = contact > 0.5
    pt = (pred_mag > LOAD_BEARING)[c]
    tt = (true_mag > LOAD_BEARING)[c]
    tp = float(np.sum(pt & tt)); fp = float(np.sum(pt & ~tt)); fn = float(np.sum(~pt & tt))
    if tp + fp + fn == 0:
        return None
    return 2 * tp / (2 * tp + fp + fn)


def _segments(ep):
    return [np.flatnonzero(ep == e) for e in np.unique(ep)]


def tolerance(pred_mag, true_mag, contact, phase, ep):
    """按消费帧掩码、滞后为正，返回 {delta: r2}（逐回合配对后池化）。"""
    res = {}
    for dl in DELTAS:
        ps, ts = [], []
        for idx in _segments(ep):
            L = len(idx)
            if L <= abs(dl) + 1:
                continue
            if dl >= 0:
                src, lab = idx[: L - dl], idx[dl:]
            else:
                lab, src = idx[: L + dl], idx[-dl:]
            m = (contact[lab] > 0.5) & (phase[lab] > PHASE_STANCE)
            if not m.any():
                continue
            ps.append(pred_mag[src][m]); ts.append(true_mag[lab][m])
        if not ps:
            res[dl] = None
            continue
        p = np.concatenate(ps); t = np.concatenate(ts)
        res[dl] = _r2(p, t)
    return {k: (round(v, 4) if v is not None else None) for k, v in res.items()}


def delta_star(curve):
    """最大 |delta| 使两侧 R2 都 >= max(R2_FLOOR, R2_MARGIN*R2(0))。"""
    c0 = curve.get(0)
    if c0 is None or c0 <= 0:
        return 0, None
    thr = max(R2_FLOOR, R2_MARGIN * c0)
    d = 0
    for k in (1, 2, 3):
        left, right = curve.get(-k), curve.get(k)
        if left is None or right is None or min(left, right) < thr:
            break
        d = k
    return d, round(thr, 4)


def label_autocorr(true_mag, contact, phase, ep, lags=(1, 2, 5, 10)):
    out = {}
    for k in lags:
        aa, bb = [], []
        for idx in _segments(ep):
            if len(idx) <= k:
                continue
            m = (contact[idx[:-k]] > 0.5) & (phase[idx[:-k]] > PHASE_STANCE)
            aa.append(true_mag[idx[:-k]][m].ravel()); bb.append(true_mag[idx[k:]][m].ravel())
        if not aa:
            continue
        a = np.concatenate(aa); b = np.concatenate(bb)
        out[str(k)] = round(float(np.corrcoef(a, b)[0, 1]), 4) if a.size > 10 else None
    return out


def contact_prev_agree(contact, ep):
    same, tot = 0, 0
    for idx in _segments(ep):
        c = contact[idx] > 0.5
        if len(c) < 2:
            continue
        same += int((c[1:] == c[:-1]).sum()); tot += c[1:].size
    return round(same / max(1, tot), 4)


def onset_recall(pred_contact, true_contact, tol_frames=2):
    """真值接触起点在 ±tol 帧内是否被预测检出（逐足）。"""
    hit, n = 0, 0
    for f in range(true_contact.shape[1]):
        c = (true_contact[:, f] > 0.5).astype(np.int8)
        onset = np.flatnonzero(np.diff(c) == 1) + 1
        for o in onset:
            n += 1
            lo, hi = max(0, o - tol_frames), min(len(c), o + tol_frames + 1)
            if (pred_contact[lo:hi, f] > 0.5).any():
                hit += 1
    return (round(hit / n, 4) if n else None), n


def eval_case(est, path, features=0):
    data = load_dataset(str(path))
    wins = make_dataset_windows(data, est.cfg, 1)
    if features and features != wins["x"].shape[-1]:
        wins["x"] = wins["x"][..., :features]
    pred = est.predict(wins["x"])
    ref = float(est.cfg.force_ref_n)
    mag = np.linalg.norm(pred["force"], axis=2) / ref
    fz = pred["force"][:, :, 2] / ref
    true_mag = np.asarray(wins["forcenorm"], dtype=np.float64)
    contact = np.asarray(wins["contact"]); phase = np.asarray(wins["phase"]); ep = np.asarray(wins["episode"])
    m = (contact > 0.5) & (phase > PHASE_STANCE)
    curve = tolerance(mag, true_mag, contact, phase, ep)
    dstar, thr = delta_star(curve)
    sp, n_sp = spearman_frames(mag, true_mag, contact, phase)
    f1 = load_bearing_f1(mag, true_mag, contact)
    cm = np.where(contact > 0.5, mag, -1.0)
    ctm = np.where(contact > 0.5, true_mag, -1.0)
    frame_ok = m.any(axis=1)
    argmax_hit = float(np.mean((cm.argmax(1) == ctm.argmax(1))[frame_ok])) if frame_ok.any() else None
    tot_p = mag[m].sum(axis=0); tot_t = true_mag[m].sum(axis=0)
    total_relerr = float(np.mean(np.abs(tot_p - tot_t) / np.maximum(1e-9, tot_t)))
    rec, n_on = onset_recall(pred["contact"], contact)
    return {
        "case": Path(path).stem.replace("test_", "", 1),
        "n_windows": int(len(ep)),
        "n_stance": int(m.sum()),
        "n_stance_frames": int(frame_ok.sum()),
        "stance_share_of_contact": round(float(m.sum() / max(1, (contact > 0.5).sum())), 4),
        "vector_mag_r2": curve.get(0),
        "tolerance": {str(k): v for k, v in curve.items()},
        "delta_star": dstar,
        "delta_thr": thr,
        "robust_ratio_1": (round(curve[1] / curve[0], 4)
                           if curve.get(0) is not None and abs(curve[0]) >= 0.05 and curve.get(1) is not None else None),
        "decision": {
            "load_bearing_f1": None if f1 is None else round(float(f1), 4),
            "argmax_foot_hit": None if argmax_hit is None else round(argmax_hit, 4),
            "rank_spearman": sp,
            "rank_spearman_frames": n_sp,
            "total_force_rel_err": round(total_relerr, 4),
            "onset_recall_pm2": rec,
            "n_onsets": n_on,
        },
        "vector_z_r2": _r2(fz[m], true_mag[m]),
        "label_autocorr": label_autocorr(true_mag, contact, phase, ep),
        "contact_prev_agree": contact_prev_agree(contact, ep),
    }


def group_means(cases):
    out = {}
    for g, names in GROUPS.items():
        rows = [c for c in cases if c["case"] in names]
        if not rows:
            continue
        def mean_of(key):
            vals = [r["decision"][key] for r in rows if r["decision"].get(key) is not None]
            return round(float(np.mean(vals)), 4) if vals else None
        out[g] = {
            "cases": [r["case"] for r in rows],
            "vector_mag_r2": round(float(np.mean([r["vector_mag_r2"] for r in rows if r["vector_mag_r2"] is not None])), 4),
            "r2_lag1": round(float(np.mean([r["tolerance"]["1"] for r in rows if r["tolerance"].get("1") is not None])), 4),
            "r2_lag2": round(float(np.mean([r["tolerance"]["2"] for r in rows if r["tolerance"].get("2") is not None])), 4),
            "delta_star": int(min(r["delta_star"] for r in rows)),
            "load_bearing_f1": mean_of("load_bearing_f1"),
            "argmax_foot_hit": mean_of("argmax_foot_hit"),
            "rank_spearman": mean_of("rank_spearman"),
            "total_force_rel_err": mean_of("total_force_rel_err"),
            "onset_recall_pm2": mean_of("onset_recall_pm2"),
            "n_stance": int(sum(r["n_stance"] for r in rows)),
        }
    return out


def _fmt(v, spec="{:+.3f}"):
    return "—" if v is None else spec.format(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Wave0 靶子：决策量 + 时移容差（零训练）")
    ap.add_argument("--checkpoint", action="append", required=True, help="可重复；每个 checkpoint 一节")
    ap.add_argument("--data-dir", default=".tmp/tactile_est_v6/arms/holdout")
    ap.add_argument("--out", default=".tmp/tactile_est_v7/wave0/report_decision.md")
    ap.add_argument("--features", type=int, default=0)
    ap.add_argument("--allow-missing-cases", action="store_true")
    args = ap.parse_args(argv)

    files = {}
    for p in sorted(glob.glob(os.path.join(args.data_dir, "test_*.npz"))):
        files[Path(p).stem.replace("test_", "", 1)] = p
    missing = [c for c in REQUIRED_CASES if c not in files]
    if missing and not args.allow_missing_cases:
        raise SystemExit("缺少必需留出 case：%s" % ", ".join(missing))
    extra = [c for c in files if c not in REQUIRED_CASES]
    if extra:
        print("[info] 忽略非留出 npz：%s" % ", ".join(extra))

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    for ck in args.checkpoint:
        est = TactileEstimator.load(ck)
        rows = [eval_case(est, files[c], features=args.features) for c in REQUIRED_CASES if c in files]
        tag = Path(ck).stem
        all_rows[tag] = {"checkpoint": ck, "cases": rows, "groups": group_means(rows)}
        print("[done] %s" % tag, flush=True)

    payload = {"required_cases": list(REQUIRED_CASES), "mask": "contact>0.5 & phase>2.5 (consumer frame)",
               "deltas": list(DELTAS), "load_bearing_thr": LOAD_BEARING, "runs": all_rows}
    Path(str(out_path).replace(".md", ".json")).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# Wave 0 靶子基线：决策量 + 时移容差（零训练）", "",
             "口径：掩码取消费帧 contact>0.5 & phase>2.5；delta>0 = 估计滞后；主指标 vector_mag（未标定）。", ""]
    for tag, blk in all_rows.items():
        lines += ["## %s" % tag, "",
                  "| case | n_stance | R2(0) | R2(+1) | R2(+2) | R2(-1) | delta* | R2(+1)/R2(0) | 承重F1 | argmax | 排序rho | SFz 相对误差 | onset@±2 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in blk["cases"]:
            t = r["tolerance"]; d = r["decision"]
            lines.append("| %s | %d | %s | %s | %s | %s | %d | %s | %s | %s | %s | %s | %s |" % (
                r["case"], r["n_stance"], _fmt(r["vector_mag_r2"]), _fmt(t.get("1")), _fmt(t.get("2")),
                _fmt(t.get("-1")), r["delta_star"], _fmt(r["robust_ratio_1"], "{:.2f}"),
                _fmt(d["load_bearing_f1"], "{:.3f}"), _fmt(d["argmax_foot_hit"], "{:.3f}"),
                _fmt(d["rank_spearman"], "{:.3f}"), _fmt(d["total_force_rel_err"], "{:.3f}"),
                _fmt(d["onset_recall_pm2"], "{:.3f}")))
        lines += ["", "| 组 | R2(0) | R2(+1) | R2(+2) | delta* | 承重F1 | argmax | 排序rho | SFz 相对误差 | onset |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for g, v in blk["groups"].items():
            lines.append("| %s | %s | %s | %s | %d | %s | %s | %s | %s | %s |" % (
                g, _fmt(v["vector_mag_r2"]), _fmt(v["r2_lag1"]), _fmt(v["r2_lag2"]), v["delta_star"],
                _fmt(v["load_bearing_f1"], "{:.3f}"), _fmt(v["argmax_foot_hit"], "{:.3f}"),
                _fmt(v["rank_spearman"], "{:.3f}"), _fmt(v["total_force_rel_err"], "{:.3f}"),
                _fmt(v["onset_recall_pm2"], "{:.3f}")))
        lines += ["", "标签自相关：" + json.dumps(blk["cases"][0]["label_autocorr"], ensure_ascii=False),
                  "相邻帧接触一致率：" + str(blk["cases"][0]["contact_prev_agree"]), ""]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
