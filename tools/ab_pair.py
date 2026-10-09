# -*- coding: utf-8 -*-
"""增量式 A/B 配对比较：把任意一批 checkpoint 对上 course22 基线。

为什么要有这个脚本（而不是每次复制一份 frontier*.py）
------------------------------------------------------
`m11915` 要求「逐一尝试修改与验证」—— 每个候选改动都要重训一次再对比基线,
一共 7 个 run。每个 run 单跑一次 40 回合 × 6 档高度的成本是 ~16 分钟 CPU,
如果每次都重算基线就是纯浪费。所以这里:

  * **缓存**: 每个 (label, 高度) 的逐回合成败存进 `.tmp/ab_pair.json`, 再跑只补
    没算过的行;
  * **配对**: 所有 checkpoint 都跑同一批种子 (`SEED + k`), 所以「同一回合基线成、
    变体败」可以直接计数, 比两条独立比例灵敏得多 (见 README §11.21 E);
  * **单侧精确二项检验**: H0 = 变体不优于基线, 只看不一致的回合;
  * **六档高度**: 低档 (8.0/9.0/9.5) 用来抓**灾难性遗忘** —— 只盯 11.0 cm 会看不见
    变体把本来会爬的高度弄坏 (course17 的教训)。

用法
----
    # 加一个新变体, 顺便把没算过的基线行补齐
    .venv\\Scripts\\python.exe tools\\ab_pair.py "c30 foot_lift2.5=runs/course30/model_final.zip=runs/course30"

    # 只补算缓存里缺的行, 不打印新东西
    .venv\\Scripts\\python.exe tools\\ab_pair.py --list

    # 丢掉缓存重来
    .venv\\Scripts\\python.exe tools\\ab_pair.py --fresh "base=runs/course22/checkpoints/medium_final.zip=runs/course22"

行格式: ``<label>=<checkpoint .zip>=<run 目录>``。``<run 目录>`` 用来取那个 run 的
``EnvConfig`` (obs 组合 / 奖励权重 / 终止阈值都随 run 走, 必须用它自己的)。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "robotdog").is_dir())
sys.path.insert(0, str(ROOT))

from robotdog.tools.console import setup_utf8_console  # noqa: E402
from robotdog.env import Go2StairsEnv  # noqa: E402
from robotdog.tools.terrain import StairSpec  # noqa: E402
from robotdog.training import env_config_from_run  # noqa: E402
from scripts.eval import load_policy  # noqa: E402

setup_utf8_console()

CACHE = ROOT / ".tmp" / "ab_pair.json"
BASE_LABEL = "c22 final (基线)"
BASE_ROW = f"{BASE_LABEL}=runs/course22/checkpoints/medium_final.zip=runs/course22"

#: 6 档。低三档抓灾难性遗忘, 高三档是这次要攻的前沿。
DEFAULT_HEIGHTS = [0.080, 0.090, 0.095, 0.100, 0.105, 0.110]
SEED = 21000


def spec_for(h: float) -> StairSpec:
    return StairSpec(step_height=h, tread=0.50, width=2.00, n_up=8, n_down=0,
                     start_x=1.50, landing=1.20, top_landing=1.20)


def parse_row(text: str) -> tuple[str, Path, Path]:
    parts = text.split("=")
    if len(parts) != 3:
        raise SystemExit(f"行格式错误 (要 label=ckpt=rundir): {text!r}")
    label, ckpt, rundir = (p.strip() for p in parts)
    return label, ROOT / ckpt, ROOT / rundir


def per_episode(model, venv, cfg, spec, n_eps: int) -> list[int]:
    """逐回合成败 (1/0), 顺序与 seed 一一对应 —— 配对比较要用它。"""
    env = Go2StairsEnv(stair=spec, cfg=cfg, seed=SEED)
    orm = getattr(venv, "obs_rms", None)
    co = float(getattr(venv, "clip_obs", 10.0))
    ep = float(getattr(venv, "epsilon", 1e-8))
    ms = int(cfg.episode_length_s / cfg.control_dt)
    res: list[int] = []
    try:
        for k in range(n_eps):
            obs, _ = env.reset(seed=SEED + k)
            term = trunc = False
            info: dict = {}
            n = 0
            while not (term or trunc) and n < ms:
                x = np.asarray(obs, dtype=np.float64)
                if orm is not None:
                    x = (x - orm.mean) / np.sqrt(orm.var + ep)
                    x = np.clip(x, -co, co)
                a, _ = model.predict(x.astype(np.float32), deterministic=True)
                obs, _r, term, trunc, info = env.step(a)
                n += 1
            res.append(int(bool(info.get("is_success"))))
    finally:
        env.close()
    return res


def load_cache() -> dict:
    if CACHE.is_file():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("[缓存] 解析失败, 重新开始")
    return {"episodes": 0, "runs": {}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="增量式 A/B 配对比较")
    ap.add_argument("rows", nargs="*", help="label=ckpt.zip=rundir")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--heights", default="",
                    help="逗号分隔的 cm, 如 8,9.5,10.5; 默认 6 档")
    ap.add_argument("--fresh", action="store_true", help="忽略并覆盖缓存")
    ap.add_argument("--list", action="store_true", help="只打印缓存现状")
    ap.add_argument("--base", default="",
                    help="换一个基线标签 (前缀匹配); 默认 c22 final。"
                         "只用缓存重算, 不跑新回合 —— 用来把同一个变体表"
                         "换成对『纯续训对照』的配对检验。")
    ap.add_argument("--cache", default="",
                    help="换一个缓存文件。⚠ 并发跑多个 ab_pair 时**必须**各用"
                         "一个缓存文件: 这个脚本是 read-modify-write 的, 同时"
                         "跑会把别人刚算出来的行盖掉 (踩过一次, 丢了 5 行)。")
    args = ap.parse_args(argv)
    if args.cache.strip():
        global CACHE
        CACHE = (ROOT / args.cache).resolve()
        CACHE.parent.mkdir(parents=True, exist_ok=True)

    heights = ([float(x) / 100.0 for x in args.heights.split(",") if x.strip()]
               if args.heights.strip() else DEFAULT_HEIGHTS)

    cache = {"episodes": args.episodes, "runs": {}} if args.fresh else load_cache()
    if int(cache.get("episodes", 0)) != args.episodes and cache["runs"]:
        print(f"[缓存] 回合数从 {cache.get('episodes')} 变成 {args.episodes}, 清空重算")
        cache = {"episodes": args.episodes, "runs": {}}
    cache["episodes"] = args.episodes
    runs: dict = cache["runs"]
    before = {lb: len(by_h) for lb, by_h in runs.items()}

    rows = list(args.rows)
    if not args.base and all(not r.startswith(BASE_LABEL) for r in rows) \
            and BASE_LABEL not in runs:
        rows = [BASE_ROW] + rows

    if args.list:
        print(f"缓存: {CACHE}  回合数={args.episodes}")
        for label, by_h in runs.items():
            have = sorted(by_h)
            print(f"  {label:<38} {len(have)} 档: {', '.join(have)}")
        return 0

    if not rows:
        rows = [BASE_ROW]

    # ---- 1) 补算缺失的行 ---------------------------------------------------
    todo = []
    for text in rows:
        label, ckpt, rundir = parse_row(text)
        if not ckpt.is_file():
            print(f"[缺文件] {label}: {ckpt}")
            continue
        by_h = runs.setdefault(label, {})
        todo.append((label, ckpt, rundir, by_h))

    for label, ckpt, rundir, by_h in todo:
        missing = [h for h in heights if f"{h*100:.1f}" not in by_h]
        if not missing:
            print(f"[缓存命中] {label}", flush=True)
            continue
        cfg = env_config_from_run(rundir)
        model, venv = load_policy(
            ckpt, ckpt.with_name(ckpt.stem + "_vecnormalize.pkl"), "cpu", cfg)
        for h in missing:
            res = per_episode(model, venv, cfg, spec_for(h), args.episodes)
            by_h[f"{h*100:.1f}"] = res
            print(f"  {label:<38} {h*100:>5.1f}cm -> {np.mean(res)*100:>5.1f}%",
                  flush=True)
            CACHE.parent.mkdir(exist_ok=True)
            CACHE.write_text(json.dumps(cache, ensure_ascii=False),
                             encoding="utf-8")

    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    # ---- 2) 打印成功率表 ---------------------------------------------------
    labels = [lb for lb in runs if all(f"{h*100:.1f}" in runs[lb] for h in heights)]
    keys = [f"{h*100:.1f}" for h in heights]
    print()
    print(f"=== {args.episodes} 回合确定性 x {len(heights)} 档 (paired, SEED={SEED}) ===")
    print(f"{'checkpoint':<40} " + "".join(f"{k:>7}" for k in keys)
          + f"{'  均值':>8}{'  最差':>8}")
    print("-" * (42 + 7 * len(keys) + 16))
    for lb in labels:
        by_h = runs[lb]
        rs = [float(np.mean(by_h[k])) for k in keys]
        print(f"{lb:<40} " + "".join(f"{r*100:>6.1f}%" for r in rs)
              + f"{np.mean(rs)*100:>7.1f}%{min(rs)*100:>7.1f}%")

    # ---- 3) 配对二项检验 ---------------------------------------------------
    if args.base:
        base_label = next((lb for lb in labels if lb.startswith(args.base)), None)
        if base_label is None:
            print(f"\n[跳过配对] 缓存里找不到基线行前缀 {args.base!r}")
            return 1
    else:
        base_label = next((lb for lb in labels if lb.startswith("c22 final")), None)
    if base_label is None:
        print("\n[跳过配对] 缓存里没有基线行 (c22 final)")
        return 0
    print()
    print(f"=== 配对检验 (同一 seed 的同一回合; 基线 = {base_label}) ===")
    print("    变体赢 = 基线败而变体成; 基线赢 = 反过来。p 越接近 1 表示变体越差。")
    print(f"{'checkpoint':<40} " + "".join(f"{k:>9}" for k in keys)
          + f"{'  变体赢/基线赢':>16}{'     n':>7}{'     p':>8}")
    print("-" * (42 + 9 * len(keys) + 32))
    summary = {}
    for lb in labels:
        if lb == base_label:
            continue
        cells, wt, lt, nt = [], 0, 0, 0
        for k in keys:
            a = np.array(runs[lb][k])
            b = np.array(runs[base_label][k])
            w = int(((a == 1) & (b == 0)).sum())
            l = int(((a == 0) & (b == 1)).sum())
            wt += w
            lt += l
            nt += a.size
            cells.append(f"{w:>3}/{l:<3}")
        disc = wt + lt
        p = 1.0 if disc == 0 else float(
            sum(math.comb(disc, i) for i in range(wt, disc + 1)) / 2 ** disc)
        summary[lb] = {"variant_wins": wt, "base_wins": lt, "n": nt,
                       "discordant": disc, "p_one_sided": p}
        flag = "  <<< 显著更差" if (disc >= 8 and p > 0.95) else (
            "  <<< 建议" if (disc >= 8 and p < 0.05) else "")
        print(f"{lb:<40} " + "".join(f"{c:>9}" for c in cells)
              + f"{wt:>10}/{lt:<6}{nt:>7}{p:>8.3f}{flag}")

    cache["paired"] = summary
    # ⚠ 防并发覆盖: 如果本次把已有的行写少了, 说明有另一个进程在同时写同一个
    # 缓存文件 (read-modify-write 竞争), 立刻喊出来 —— 我丢过 5 行就是这个原因。
    lost = {lb: (before[lb], len(by_h)) for lb, by_h in runs.items()
            if lb in before and len(by_h) < before[lb]}
    CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    if lost:
        print(f"[警告] 缓存里有行数变少的标签 (疑似并发覆盖): {lost}")
    print(f"\n缓存 -> {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
