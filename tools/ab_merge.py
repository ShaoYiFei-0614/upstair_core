"""Merge the per-job A/B caches into one and print the controlled comparison.

⚠ Why this exists: `tools/ab_pair.py` is read-modify-write on a single JSON
cache, so running several of them **concurrently** silently drops rows (each
process writes back the snapshot it loaded). I lost 5 rows that way, so every
concurrent job now gets `--cache .tmp/ab_<tag>.json`, and this script unions
them back together. Per (label, height) the outcome vectors are bit-identical
across files (same seed), so a plain union is safe.

Usage:
    python tools/ab_merge.py [--base "c27 final"] [--out .tmp/ab_final.json]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import OrderedDict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HEIGHTS = [0.080, 0.090, 0.095, 0.100, 0.105, 0.110]
SEED = 21000


def load_all() -> dict:
    merged: dict = {}
    srcs = []
    for p in sorted((ROOT / ".tmp").glob("ab_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[跳过] {p.name} 解析失败")
            continue
        n = 0
        for label, by_h in (d.get("runs") or {}).items():
            tgt = merged.setdefault(label, {})
            for k, v in by_h.items():
                if k in tgt and len(tgt[k]) != len(v):
                    print(f"[冲突] {label}@{k}: {len(tgt[k])} vs {len(v)} 回合")
                tgt[k] = v
                n += 1
        srcs.append(f"{p.name}({n})")
    print(f"[合并] {len(srcs)} 个缓存: {', '.join(srcs)}")
    return merged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="合并 A/B 缓存并做配对检验")
    ap.add_argument("--base", default="c22 final")
    ap.add_argument("--out", default=".tmp/ab_final.json")
    ap.add_argument("--heights", default="")
    args = ap.parse_args(argv)

    runs = load_all()
    # ⚠ `--heights` 收的是**厘米**（和 ab_pair.py 一样，如 `--heights 11` ⇒ "11.0"）。
    # 旧写法 `float(x)/100:.1f` 会把 11 变成 0.11 再四舍五入成 "0.1"，永远匹配不到行。
    keys = [f"{float(x):.1f}" for x in args.heights.split(",") if x.strip()] \
        if args.heights.strip() else [f"{h*100:.1f}" for h in HEIGHTS]
    labels = [lb for lb in runs if all(k in runs[lb] for k in keys)]
    skipped = [lb for lb in runs if lb not in labels]

    print()
    print(f"=== 40 回合确定性 x {len(keys)} 档 (paired, SEED={SEED}), "
          f"只列 {len(labels)} 个满行的标签 ===")
    print(f"{'checkpoint':<40} " + "".join(f"{k:>7}" for k in keys)
          + f"{'  均值':>8}{'  最差':>8}")
    print("-" * (42 + 7 * len(keys) + 16))
    means = {}
    for lb in labels:
        rs = [sum(runs[lb][k]) / len(runs[lb][k]) for k in keys]
        means[lb] = sum(rs) / len(rs)
        print(f"{lb:<40} " + "".join(f"{r*100:>6.1f}%" for r in rs)
              + f"{means[lb]*100:>7.1f}%{min(rs)*100:>7.1f}%")

    base = next((lb for lb in labels if lb.startswith(args.base)), None)
    if base is None:
        print(f"\n[跳过配对] 找不到基线前缀 {args.base!r}")
        return 1
    print()
    print(f"=== 配对检验 (同一 seed 的同一回合; 基线 = {base}) ===")
    print("    变体赢 = 基线败而变体成; 基线赢 = 反过来。p 越接近 1 表示变体越差。")
    print(f"{'checkpoint':<40} " + "".join(f"{k:>9}" for k in keys)
          + f"{'  变体赢/基线赢':>16}{'     n':>7}{'     p':>8}")
    print("-" * (42 + 9 * len(keys) + 32))
    out = {"base": base, "keys": keys, "means": means, "paired": {}}
    for lb in labels:
        if lb == base:
            continue
        cells, wt, lt, nt = [], 0, 0, 0
        for k in keys:
            a = runs[lb][k]
            b = runs[base][k]
            w = int(sum(1 for x, y in zip(a, b) if x == 1 and y == 0))
            l = int(sum(1 for x, y in zip(a, b) if x == 0 and y == 1))
            wt += w
            lt += l
            nt += len(a)
            cells.append(f"{w:>3}/{l:<3}")
        disc = wt + lt
        p = 1.0 if disc == 0 else float(
            sum(math.comb(disc, i) for i in range(wt, disc + 1)) / 2 ** disc)
        delta = (means[lb] - means[base]) * 100.0
        flag = "  <<< 显著更差" if (disc >= 8 and p > 0.95) else (
            "  <<< 建议采纳" if (disc >= 8 and p < 0.05) else "")
        out["paired"][lb] = {"variant_wins": wt, "base_wins": lt, "n": nt,
                             "discordant": disc, "p_one_sided": p,
                             "mean_delta_pt": delta}
        print(f"{lb:<40} " + "".join(f"{c:>9}" for c in cells)
              + f"{wt:>10}/{lt:<6}{nt:>7}{p:>8.3f}"
              + f"  Δ均值{delta:+6.1f}pt{flag}")

    if skipped:
        print(f"\n[未满行, 不参检] {len(skipped)} 个: "
              + ", ".join(f"{lb}({len(runs[lb])}档)" for lb in skipped))
    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    print(f"\n-> {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
