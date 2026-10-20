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
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from robotdog.tactile_estimator import (
    TactileEstimator, contact_metrics, force_metrics, load_dataset,
    make_dataset_windows, regression_metrics,
)


def eval_dataset(est: TactileEstimator, data: dict, stride: int = 2):
    cfg = est.cfg
    wins = make_dataset_windows(data, cfg, stride)
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
        ref = float(cfg.force_ref_n)
        out["force_world_contact_only"] = {
            "rmse_n": float(np.sqrt(np.mean(err ** 2))),
            "mae_n": float(np.mean(np.abs(err))),
            "rmse_bw": float(np.sqrt(np.mean(err ** 2)) / ref),
            "mae_bw": float(np.mean(np.abs(err)) / ref),
            "p95_n": float(np.percentile(np.abs(err), 95)),
        }
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
    args = ap.parse_args()
    est = TactileEstimator.load(args.checkpoint)
    cfg = est.cfg
    files = sorted(Path(args.data_dir).glob("*.npz"))
    if not files:
        raise SystemExit(f"{args.data_dir} 里没有 npz")
    rows = []
    header = [
        "# 足底触觉 GRU 估计器 · 评估报告",
        "",
        f"- 权重：`{args.checkpoint}`（window={cfg.window}, hidden={cfg.hidden}, "
        f"layers={cfg.layers}, force_ref={cfg.force_ref_n:.2f} N）",
        "- 标签：`TactileSensor` 的足底三维力 / 接触 / 面积 / CoP（仿真测量）",
        "- 输入：策略可见观测（53 维本体感觉 + 指令 + step_height），"
        f"窗口 {cfg.window} 步（{cfg.window * 0.02:.2f} s @50 Hz）",
        "",
        "| 数据集 | split | 回合/窗口 | 三维力（世界系，全量） | 接触 Acc/F1 | "
        "面积 RMSE/R² | CoP RMSE/R² |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for f in files:
        data = load_dataset(f)
        meta = data.get("meta", {})
        res, _wins, _pred = eval_dataset(est, data, args.stride)
        name = meta.get("name", f.stem)
        split = meta.get("split", f.stem.split("_")[0])
        fm = res["force_world"]
        cm = res["contact"]
        am = res["area"]
        pm = res["cop"]
        rows.append({"name": name, "split": split, "file": str(f),
                     "meta": meta, **res})
        header.append(
            f"| {name} | {split} | {meta.get('episodes', '?')}/{res['n_windows']} | "
            f"{fmt_force(fm)} | {cm['accuracy']:.3f}/{cm['f1']:.3f} | "
            f"{am['rmse']:.3f}/{am['r2']:+.2f} | {pm['rmse']:.3f}/{pm['r2']:+.2f} |")
        print(f"{split:5s} {name:18s} force {fmt_force(fm)} | contact "
              f"acc {cm['accuracy']:.3f} F1 {cm['f1']:.3f} | area {am['rmse']:.3f} "
              f"| cop {pm['rmse']:.3f}")
    header += ["",
               "足体系三维力不单独报告：窗口化丢弃 `force_foot`、且未保存每步 `foot_rot`，",
               "无法把世界系预测旋到足体系；世界系指标已覆盖交付口径（doc §5）。"]
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
    axes_out = out.with_suffix(".axes.txt")
    axes_out.write_text("\n".join(axes_lines) + "\n", encoding="utf-8")
    print(f"report: {out}")
    print(f"axes:   {axes_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
