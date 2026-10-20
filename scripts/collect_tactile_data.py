"""采集足底触觉数据集（训练 + 留出地形），供 GRU 状态估计器使用。

用法::

    .venv\\Scripts\\python.exe scripts/collect_tactile_data.py --out .tmp/tactile_estimator/data

每个「地形 × 策略」一个 npz：特征 = 策略可见的``_observation()``（默认 53 维），
标签 = ``TactileState`` 的三维力（世界系/足体系）、接触、面积、CoP。

**地形分配**（留出集不参与训练）：

* train：平地、15 cm 楼梯、2–5 cm 粗糙地面（seed 0，两个策略各采一段）；
* test：16.5 cm 楼梯、18 cm 楼梯、粗糙地面 seed 1 / seed 2。

策略用的是本仓已有权重（平地 ``ab_f_ctl``、15 cm ``crawl24``、16.5/18 cm ``hi165/hi18``），
确定性策略 ⇒ 数据集可由命令行逐位复现。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from robotdog.env import Go2StairsEnv
from robotdog.policy import env_config_for_model, stair_from_height
from robotdog.tactile import TactileConfig
from robotdog.tactile_estimator import save_dataset
from robotdog.tools.rough import rough_spec
from robotdog.tools.terrain import flat_spec
from robotdog.training import load_policy

FLAT_POLICY = "runs/ab_f_ctl/checkpoints/flat_final.zip"
STAIR15_POLICY = "runs/crawl24/checkpoints/stair15_final.zip"
STAIR165_POLICY = "runs/hi165/checkpoints/stair16_5_final.zip"
STAIR18_POLICY = "runs/hi18/checkpoints/stair18_final.zip"

# (名字, terrain spec 工厂, 策略, 指令 vx, 回合数, split)
CASES = [
    ("flat", lambda: flat_spec(), FLAT_POLICY, 0.45, 16, "train"),
    ("stair15", lambda: stair_from_height(15.0, tread=0.30, n_up=8),
     STAIR15_POLICY, 0.45, 16, "train"),
    ("rough_s0_flatpol", lambda: rough_spec(0.02, 0.05, seed=0),
     FLAT_POLICY, 0.35, 10, "train"),
    ("rough_s0_stairpol", lambda: rough_spec(0.02, 0.05, seed=0),
     STAIR15_POLICY, 0.35, 10, "train"),
    ("stair16_5", lambda: stair_from_height(16.5, tread=0.30, n_up=8),
     STAIR165_POLICY, 0.42, 8, "test"),
    ("stair18", lambda: stair_from_height(18.0, tread=0.30, n_up=8),
     STAIR18_POLICY, 0.42, 8, "test"),
    ("rough_s1", lambda: rough_spec(0.02, 0.05, seed=1),
     STAIR15_POLICY, 0.35, 8, "test"),
    ("rough_s2", lambda: rough_spec(0.02, 0.05, seed=2),
     STAIR15_POLICY, 0.35, 8, "test"),
]

MAX_STEPS = 700


def collect(name: str, stair, policy_path: str, vx: float, episodes: int,
            seed: int = 0) -> tuple[dict[str, np.ndarray], dict]:
    cfg = env_config_for_model(policy_path)
    cfg.tactile = TactileConfig(enabled=True)
    env = Go2StairsEnv(stair=stair, cfg=cfg, seed=seed)
    model, norm = load_policy(policy_path, cfg=cfg, stair=stair)
    feats, fw, ff, ct, ar, cp, ph, ep = [], [], [], [], [], [], [], []
    steps = ok_eps = fall_eps = 0
    max_base_x = -np.inf
    for e in range(int(episodes)):
        obs, _ = env.reset(seed=1000 * seed + e + 1)
        env.set_command_now(vx=float(vx), hold=True)
        obs = env._observation()
        done = trunc = False
        n = 0
        info = {}
        while not (done or trunc) and n < MAX_STEPS:
            x = obs if norm is None else norm.normalize(obs)
            act, _ = model.predict(x, deterministic=True)
            obs, _r, done, trunc, info = env.step(act)
            st = env.tactile
            feats.append(np.asarray(obs, dtype=np.float32))
            fw.append(st.force_vec.copy())
            ff.append(st.force_vec_foot.copy())
            ct.append((st.force_n > 1e-6).astype(np.uint8))
            ar.append(st.area_norm.copy())
            cp.append(st.cop_norm.copy())
            ph.append(st.phase.copy())
            ep.append(e)
            n += 1
            steps += 1
            max_base_x = max(max_base_x, float(env.base_pos[0]))
        # 「到达」只是采集覆盖度的粗判据（不参与训练/评估）：
        # 楼梯 = 判成功或越过顶端前 0.6m；平地/rough = 前进超过起点 0.2m。
        # rough 上的真实穿越没有 env 级判据（成功判据是平地指令跟踪），
        # 所以 summary 里同时记 `max_base_x` 供核对。
        reached = bool(info.get("is_success", False))
        if stair.n_up > 0:
            reached = reached or (env.base_pos[0] > stair.up_end_x - 0.6)
        else:
            reached = reached or (max_base_x > stair.start_x + 0.2)
        ok_eps += int(reached)
        fall_eps += int(done and not reached)
    env.close()
    data = {
        "features": np.stack(feats),
        "force_world": np.stack(fw),
        "force_foot": np.stack(ff),
        "contact": np.stack(ct),
        "area_norm": np.stack(ar),
        "cop_norm": np.stack(cp),
        "phase": np.stack(ph),
        "episode": np.asarray(ep, dtype=np.int32),
    }
    meta = {
        "name": name,
        "policy": policy_path,
        "vx": float(vx),
        "episodes": int(episodes),
        "steps": int(steps),
        "max_base_x": float(max_base_x) if np.isfinite(max_base_x) else float("nan"),
        "success_episodes": int(ok_eps),
        "fall_episodes": int(fall_eps),
        "step_height_m": float(stair.step_height),
        "tread_m": float(stair.tread),
        "n_up": int(stair.n_up),
        "rough_amp_hi_m": float(getattr(stair, "amp_hi", 0.0)),
        "rough_seed": int(getattr(stair, "rough_seed", -1)),
        "force_ref_n": 149.17,
        "source": "tactile simulator (TactileSensor, mj_contactForce)",
    }
    return data, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".tmp/tactile_estimator/data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="只采指定名字（逗号分隔）")
    args = ap.parse_args()
    out = Path(args.out)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    t0 = time.time()
    summary = []
    for name, spec_fn, policy_path, vx, episodes, split in CASES:
        if only and name not in only:
            continue
        if not Path(policy_path).is_file():
            print(f"SKIP {name}: 缺权重 {policy_path}")
            continue
        data, meta = collect(name, spec_fn(), policy_path, vx, episodes,
                             seed=args.seed)
        meta["split"] = split
        path = save_dataset(out / f"{split}_{name}.npz", meta=meta, **data)
        contact_rate = float(np.mean(np.asarray(data["contact"]).reshape(-1)))
        fmax = float(np.abs(data["force_world"]).max())
        summary.append({**meta, "file": str(path), "contact_rate": contact_rate,
                        "max_force_n": fmax})
        print(f"{split:5s} {name:18s} steps={meta['steps']:6d} "
              f"eps={episodes:2d} ok={meta['success_episodes']:2d} "
              f"fall={meta['fall_episodes']:2d} contact={contact_rate:.2f} "
              f"|F|max={fmax:6.1f} N")
    (out / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"total {len(summary)} datasets, {time.time() - t0:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
