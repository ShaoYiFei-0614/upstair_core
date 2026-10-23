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
from robotdog.tactile_estimator import (
    DEFAULT_FORCE_REF_N, TORQUE_FEATURE_NAMES, TORQUE_REF_NM, save_dataset,
    torque_features,
)
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

# v4 数据多样性臂（--include-extra 才采）：**全部是训练地形**，
# 只动「同地形换策略 / rough 换 seed」，不碰留出集的 16.5/18cm 与 rough_s1/s2。
# 目的：分离「模型容量不够」与「训练分布太窄」——rough 只见过 seed 0、
# 楼梯只见过与自身高度配对的策略。
EXTRA_CASES = [
    ("rough_s3_flatpol", lambda: rough_spec(0.02, 0.05, seed=3),
     FLAT_POLICY, 0.35, 10, "train"),
    ("rough_s4_stairpol", lambda: rough_spec(0.02, 0.05, seed=4),
     STAIR15_POLICY, 0.35, 10, "train"),
    ("stair15_stair165pol", lambda: stair_from_height(15.0, tread=0.30, n_up=8),
     STAIR165_POLICY, 0.42, 8, "train"),
    ("stair15_stair18pol", lambda: stair_from_height(15.0, tread=0.30, n_up=8),
     STAIR18_POLICY, 0.42, 8, "train"),
    ("flat_stair165pol", lambda: flat_spec(), STAIR165_POLICY, 0.45, 10, "train"),
    ("rough_s3_stairpol", lambda: rough_spec(0.02, 0.05, seed=3),
     STAIR15_POLICY, 0.35, 10, "train"),
]

# ---------------------------------------------------------------------------
# v5 数据：低/中台阶高度阶梯。策略可用性已由 probe 探针验证（4.5/6/7.5/9/12cm
# 均能登顶；其中 4.5/6/7.5/9 用各自匹配策略，12 用 asym12）。
# 留出：插值 10.5/13.5、外推 16.5/18/19.5（16.5/18 与 v2-v4 一致，保持可比）。
# ---------------------------------------------------------------------------
STAIR4_5_POLICY = "runs/asym_upstairs/checkpoints/stair4_5_final.zip"
STAIR6_POLICY = "runs/ab_6_ctl/checkpoints/stair6_final.zip"
STAIR7_5_POLICY = "runs/asym_upstairs/checkpoints/stair7_5_final.zip"
STAIR9_POLICY = "runs/asym_upstairs/checkpoints/stair9_final.zip"
STAIR10_5_POLICY = "runs/crawl21/checkpoints/stair10_5_final.zip"
STAIR12_POLICY = "runs/asym12/checkpoints/stair12_final.zip"
STAIR13_5_POLICY = "runs/crawl23/checkpoints/stair13_5_final.zip"
STAIR19_5_POLICY = "runs/hi195/checkpoints/stair19_5_final.zip"

# (名字, terrain spec 工厂, 策略, 指令 vx, 回合数, split[, 回合种子偏移])
CASES_V5_TRAIN = [
    ("stair4_5", lambda: stair_from_height(4.5, tread=0.30, n_up=8),
     STAIR4_5_POLICY, 0.45, 8, "train"),
    ("stair6", lambda: stair_from_height(6.0, tread=0.30, n_up=8),
     STAIR6_POLICY, 0.45, 8, "train"),
    ("stair7_5", lambda: stair_from_height(7.5, tread=0.30, n_up=8),
     STAIR7_5_POLICY, 0.45, 8, "train"),
    ("stair9", lambda: stair_from_height(9.0, tread=0.30, n_up=8),
     STAIR9_POLICY, 0.45, 8, "train"),
    ("stair12", lambda: stair_from_height(12.0, tread=0.30, n_up=8),
     STAIR12_POLICY, 0.42, 8, "train"),
]

# A3 臂：每个新高度再配一条「高度不匹配」策略（策略多样性，不与高度因子混）
CASES_V5_POL2 = [
    ("stair4_5_pol6", lambda: stair_from_height(4.5, tread=0.30, n_up=8),
     STAIR6_POLICY, 0.45, 8, "train"),
    ("stair6_pol4_5", lambda: stair_from_height(6.0, tread=0.30, n_up=8),
     STAIR4_5_POLICY, 0.45, 8, "train"),
    ("stair7_5_pol9", lambda: stair_from_height(7.5, tread=0.30, n_up=8),
     STAIR9_POLICY, 0.45, 8, "train"),
    ("stair9_pol7_5", lambda: stair_from_height(9.0, tread=0.30, n_up=8),
     STAIR7_5_POLICY, 0.45, 8, "train"),
    ("stair12_pol13_5", lambda: stair_from_height(12.0, tread=0.30, n_up=8),
     STAIR13_5_POLICY, 0.42, 8, "train"),
]

# 留出：只采集不训练（main 里的归集守卫会拒绝同名训练 case）
CASES_V5_TEST = [
    ("stair10_5", lambda: stair_from_height(10.5, tread=0.30, n_up=8),
     STAIR10_5_POLICY, 0.42, 8, "test"),
    ("stair13_5", lambda: stair_from_height(13.5, tread=0.30, n_up=8),
     STAIR13_5_POLICY, 0.42, 8, "test"),
    ("stair19_5", lambda: stair_from_height(19.5, tread=0.30, n_up=8),
     STAIR19_5_POLICY, 0.42, 8, "test"),
]

# A1 臂（「同类型加倍」对照）：同一批训练 case 再采 4 个回合（不同初始扰动）。
# 只加「数量」不加「类型」，与 A2（加高度类型）配对，用来分离数量 vs 类型。
A1_EXTRA = [
    ("flat_x4", lambda: flat_spec(), FLAT_POLICY, 0.45, 4, "train", 100),
    ("stair15_x4", lambda: stair_from_height(15.0, tread=0.30, n_up=8),
     STAIR15_POLICY, 0.45, 4, "train", 100),
    ("rough_s0_flatpol_x4", lambda: rough_spec(0.02, 0.05, seed=0),
     FLAT_POLICY, 0.35, 4, "train", 100),
    ("rough_s0_stairpol_x4", lambda: rough_spec(0.02, 0.05, seed=0),
     STAIR15_POLICY, 0.35, 4, "train", 100),
    ("rough_s3_flatpol_x4", lambda: rough_spec(0.02, 0.05, seed=3),
     FLAT_POLICY, 0.35, 4, "train", 100),
    ("rough_s4_stairpol_x4", lambda: rough_spec(0.02, 0.05, seed=4),
     STAIR15_POLICY, 0.35, 4, "train", 100),
    ("rough_s3_stairpol_x4", lambda: rough_spec(0.02, 0.05, seed=3),
     STAIR15_POLICY, 0.35, 4, "train", 100),
    ("stair15_stair165pol_x4", lambda: stair_from_height(15.0, tread=0.30, n_up=8),
     STAIR165_POLICY, 0.42, 4, "train", 100),
    ("stair15_stair18pol_x4", lambda: stair_from_height(15.0, tread=0.30, n_up=8),
     STAIR18_POLICY, 0.42, 4, "train", 100),
    ("flat_stair165pol_x4", lambda: flat_spec(), STAIR165_POLICY, 0.45, 4, "train", 100),
]

# ---------------------------------------------------------------------------
# v6 数据扩展：**类型**（新几何 tread、新速度 vx、新 rough seed/幅度）与
# **数量**（同分布再加 8 回合，seed_offset=200 与 A1 的 100 不重叠）。
# 两者分开成 --include-v6-type / --include-v6-qty，便于把「类型 vs 数量」
# 做成两个独立臂（任务书 E-P3）。全部是训练地形，不碰 7 个留出。
# 可行性由 .tmp/tactile_est_v6/tools/probe_v6.py 逐 case 先探（1 回合登顶）。
# ---------------------------------------------------------------------------
CASES_V6_TYPE = [
    ("stair9_tread28", lambda: stair_from_height(9.0, tread=0.28, n_up=8),
     STAIR9_POLICY, 0.45, 8, "train"),
    ("stair15_tread32", lambda: stair_from_height(15.0, tread=0.32, n_up=8),
     STAIR15_POLICY, 0.45, 8, "train"),
    ("stair9_vx30", lambda: stair_from_height(9.0, tread=0.30, n_up=8),
     STAIR9_POLICY, 0.30, 8, "train"),
    # v6：vx 0.55 在 15cm 上第 101 步摔（probe 实测），改 0.50（388 步登顶、0 摔）
    ("stair15_vx50", lambda: stair_from_height(15.0, tread=0.30, n_up=8),
     STAIR15_POLICY, 0.50, 8, "train"),
    ("rough_s5_stairpol", lambda: rough_spec(0.03, 0.06, seed=5),
     STAIR15_POLICY, 0.35, 8, "train"),
    ("rough_s6_flatpol", lambda: rough_spec(0.03, 0.06, seed=6),
     FLAT_POLICY, 0.35, 8, "train"),
]

# 数量臂：同地形、同策略、同指令，只换初始扰动种子（seed_offset=200）。
CASES_V6_QTY = [
    ("stair9_x8", lambda: stair_from_height(9.0, tread=0.30, n_up=8),
     STAIR9_POLICY, 0.45, 8, "train", 200),
    ("stair15_x8", lambda: stair_from_height(15.0, tread=0.30, n_up=8),
     STAIR15_POLICY, 0.45, 8, "train", 200),
]

# ---------------------------------------------------------------------------
# v7 Wave A 数据：**10 个零摔策略 × 12 地形条件 × 2 回合 = 240 回合**。
# 设计动机（用户 2026-10-23 评审）：数据扩展方向无条件执行，扩展剂量要按
# 「策略宽度 × 每策略地形覆盖」补足；有摔倒的策略整条剔除（先 probe 再采）。
#
# 12 条件 = 6 平整（flat + 4.5/6/9/12/15cm）+ 6 rough（振幅带 × seed）。
# 注意：本仓 rough 是「平地走廊 + 随机方块高度场」（RoughSpec 强制 n_up=0、
# step_height=0），与楼梯几何互斥 ⇒ 不存在「某高度的 rough 楼梯」；rough 维度
# 只能拆成振幅带（2–3 / 3–4 / 4–5 / 2–5 cm）与 seed 两个因子。
# 留出不动：10.5/13.5/16.5/18/19.5cm + rough seed 1/2（不同种子）。
# 10 个策略均来自 v7 探针（20 策略 × 5 高度 × 1 回合）里 5/5 通过且 0 摔者。
# ---------------------------------------------------------------------------
V7_POLICIES = [
    ("l0v2", "runs/l0_v2/checkpoints/stair15_final.zip"),
    ("l1all", "runs/l1_all/checkpoints/stair15_final.zip"),
    ("l1stv2", "runs/l1_st_v2/checkpoints/stair15_final.zip"),
    ("asym15", "runs/asym15_ladder/checkpoints/stair15_final.zip"),
    ("crawl24", "runs/crawl24/checkpoints/stair15_final.zip"),
    ("ab15", "runs/ab_15_ctl/checkpoints/stair15_final.zip"),
    ("hi165", "runs/hi165/checkpoints/stair16_5_final.zip"),
    ("hi195", "runs/hi195/checkpoints/stair19_5_final.zip"),
    ("lr15_9e5", "runs/lr15_9e5/checkpoints/stair15_final.zip"),
    ("lr15_7e5", "runs/lr15_7e5/checkpoints/stair15_final.zip"),
]

# (条件名, terrain spec 工厂, 指令 vx)
V7_CONDITIONS = [
    ("flat", lambda: flat_spec(), 0.45),
    ("stair4_5", lambda: stair_from_height(4.5, tread=0.30, n_up=8), 0.45),
    ("stair6", lambda: stair_from_height(6.0, tread=0.30, n_up=8), 0.45),
    ("stair9", lambda: stair_from_height(9.0, tread=0.30, n_up=8), 0.45),
    ("stair12", lambda: stair_from_height(12.0, tread=0.30, n_up=8), 0.42),
    ("stair15", lambda: stair_from_height(15.0, tread=0.30, n_up=8), 0.42),
    ("rough_a2", lambda: rough_spec(0.02, 0.03, seed=0), 0.35),
    ("rough_a3", lambda: rough_spec(0.03, 0.04, seed=3), 0.35),
    ("rough_a4", lambda: rough_spec(0.04, 0.05, seed=4), 0.35),
    ("rough_s5", lambda: rough_spec(0.02, 0.05, seed=5), 0.35),
    ("rough_s6", lambda: rough_spec(0.02, 0.05, seed=6), 0.35),
    ("rough_s7", lambda: rough_spec(0.02, 0.05, seed=7), 0.35),
]

# 表驱动展开：每格 2 回合（同条件两回合只差初始扰动，seed 规则 env.reset(1000*(seed+offset)+e+1)）
CASES_V7 = [
    (f"{pname}_{cname}", spec_fn, policy, vx, 2, "train")
    for pname, policy in V7_POLICIES
    for cname, spec_fn, vx in V7_CONDITIONS
]

# 策略漂移评测集（只评不训）：5 个「有摔/未通过」策略 × 3 高度 × 1 回合。
# 用途：容量选择是否在**未见策略**上更好（描述性，不做 gate）；摔倒不剔除，
# 用摔倒前的帧，meta 里 fall_episodes 会记录。
V7_POLSHIFT_POLICIES = [
    ("l1pl", "runs/l1_pl/checkpoints/stair15_final.zip"),
    ("crawl21", STAIR10_5_POLICY),
    ("ab6", STAIR6_POLICY),
    ("asym12", STAIR12_POLICY),
    ("crawl23", STAIR13_5_POLICY),
]
V7_POLSHIFT_HEIGHTS = [4.5, 9.0, 15.0]
CASES_POLSHIFT = [
    (f"polshift_{pname}_h{str(h).replace('.', '')}",
     (lambda h=h: stair_from_height(h, tread=0.30, n_up=8)), policy, 0.45, 1, "polshift")
    for pname, policy in V7_POLSHIFT_POLICIES
    for h in V7_POLSHIFT_HEIGHTS
]

MAX_STEPS = 700


def spec_key(s) -> tuple:
    """地形几何键：楼梯=(高度, 踏面, 阶数)；rough=(seed, amp_hi)；平地=("flat",)。

    粗糙地形高度恒为 0、踏面字段无意义，必须用 seed/幅度区分，否则所有 rough
    训练 case 都会被误判成留出几何（v6 实现时踩过）。留出守卫用它做第二层检查。
    """
    hs = float(getattr(s, "step_height", 0.0))
    nh = int(getattr(s, "n_up", 0))
    seed = int(getattr(s, "rough_seed", -1))
    if nh == 0 and seed >= 0:
        return ("rough", seed, round(float(getattr(s, "amp_hi", 0.0)), 4))
    if nh == 0:
        return ("flat",)
    return ("stair", round(hs, 4), round(float(getattr(s, "tread", 0.0)), 4), nh)


def collect(name: str, stair, policy_path: str, vx: float, episodes: int,
            seed: int = 0, seed_offset: int = 0,
            privileged: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    cfg = env_config_for_model(policy_path)
    # 显式钉住 force_ref_n：F̄ / ΔF̄ / 峰值三个标签都是「除以体重」的量，
    # 必须和估计器的 force_ref_n 用同一个 mg，否则训练/评估口径会差一个常数。
    cfg.tactile = TactileConfig(enabled=True, force_ref_n=DEFAULT_FORCE_REF_N)
    # 特征列必须是 **actor 可见的 53 维**：部分历史策略的 config 带
    # privileged_foot_contact（critic 特权块），env._observation() 会返回 57 维，
    # 直接拿它当特征会把「足端接触真值」漏进估计器输入（数据污染）。
    # 按 cfg.obs.actor_index 取 actor 切片，并硬校验列数。
    actor_idx = list(cfg.obs.actor_index)
    n_actor = int(cfg.obs.actor_size)
    n_full = int(cfg.obs.size)
    if n_actor != 53:
        raise SystemExit(f"actor 观测应为 53 维（估计器特征契约），实际 {n_actor}；"
                         f"policy={policy_path} cfg.obs={cfg.obs}")

    def _actor_obs(raw: np.ndarray) -> np.ndarray:
        vec = np.asarray(raw, dtype=np.float32).reshape(-1)
        if vec.size == n_actor:
            return vec
        if vec.size != n_full:
            raise SystemExit(f"观测维数 {vec.size} 既不是 actor {n_actor} 也不是全帧 {n_full}")
        return vec[actor_idx]

    env = Go2StairsEnv(stair=stair, cfg=cfg, seed=seed)
    model, norm = load_policy(policy_path, cfg=cfg, stair=stair)
    feats, fw, ff, ct, ar, cp, ph, ep = [], [], [], [], [], [], [], []
    fn, dl, pk = [], [], []
    tq = []            # 原始 12 维执行器力矩（未归一化）
    # D1 上限探测用的特权输入（真机上可由感知/状态估计得到，但不在策略观测里）：
    # 足端表面高度、离地间隙、垂向速度、接触真值、足下地形高度、机身高度。
    pz, pclr, pvz, pct, pth, pbz = [], [], [], [], [], []
    n_obs = 0          # 策略可见观测维数（默认 53），力矩列追加在其后
    steps = ok_eps = fall_eps = 0
    max_base_x = -np.inf
    prev_feet = None
    for e in range(int(episodes)):
        obs, _ = env.reset(seed=1000 * (seed + seed_offset) + e + 1)
        env.set_command_now(vx=float(vx), hold=True)
        obs = env._observation()   # 策略要它训练时的完整宽度（可能含 critic 特权块）
        done = trunc = False
        n = 0
        info = {}
        while not (done or trunc) and n < MAX_STEPS:
            x = obs if norm is None else norm.normalize(obs)
            act, _ = model.predict(x, deterministic=True)
            obs, _r, done, trunc, info = env.step(act)
            st = env.tactile
            # 特征 = 策略可见观测 + 12 维归一化关节力矩（真机同样可得；
            # 归一化尺度的单源在 tactile_estimator.TORQUE_REF_NM）
            obs_vec = _actor_obs(obs)
            tau = np.asarray(env.data.actuator_force, dtype=np.float32).reshape(-1)
            n_obs = int(obs_vec.size)
            if n_obs != 53:
                raise SystemExit(f"特征观测 {n_obs} 维，应为 53（policy={policy_path}）")
            feats.append(np.concatenate([obs_vec, torque_features(tau)]))
            tq.append(tau)
            fw.append(st.force_vec.copy())
            ff.append(st.force_vec_foot.copy())
            ct.append((st.force_n > 1e-6).astype(np.uint8))
            ar.append(st.area_norm.copy())
            cp.append(st.cop_norm.copy())
            ph.append(st.phase.copy())
            # 消费端标签（与 tactile.py 的字段语义一致，已按体重归一化）
            fn.append(st.force_norm.copy())
            dl.append(st.delta_force_norm.copy())
            pk.append(st.impact_peak.copy())
            ep.append(e)
            if privileged:
                feet = np.asarray(env.foot_pos, dtype=np.float32)
                if prev_feet is None:
                    fvz = np.zeros(4, dtype=np.float32)
                else:
                    fvz = ((feet[:, 2] - prev_feet[:, 2])
                           / max(1e-6, float(cfg.control_dt))).astype(np.float32)
                prev_feet = feet.copy()
                pz.append(feet[:, 2] - float(env.meta.foot_radius))
                pclr.append(np.asarray(env.foot_clearance, dtype=np.float32))
                pvz.append(fvz)
                pct.append(np.asarray(env._foot_contact_now, dtype=np.uint8).reshape(-1))
                pth.append(np.asarray(
                    env._terrain_height(feet[:, 0], feet[:, 1]),
                    dtype=np.float32).reshape(-1))
                pbz.append(np.asarray([env.base_pos[2]], dtype=np.float32))
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
        "force_norm": np.stack(fn),
        "delta_force_norm": np.stack(dl),
        "impact_peak": np.stack(pk),
        "torque": np.stack(tq),
        "episode": np.asarray(ep, dtype=np.int32),
    }
    if privileged:
        data.update({
            "priv_foot_z": np.stack(pz),
            "priv_clearance": np.stack(pclr),
            "priv_foot_vz": np.stack(pvz),
            "priv_contact_true": np.stack(pct),
            "priv_terrain_h": np.stack(pth),
            "priv_base_z": np.stack(pbz),
        })
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
        "force_ref_n": float(DEFAULT_FORCE_REF_N),
        "source": "tactile simulator (TactileSensor, mj_contactForce)",
        "n_features": int(n_obs + len(TORQUE_FEATURE_NAMES)),
        "n_obs": int(n_obs),
        "feature_names": [f"obs_{i}" for i in range(n_obs)]
                         + list(TORQUE_FEATURE_NAMES),
        "torque_ref_nm": [float(v) for v in TORQUE_REF_NM],
        "torque_source": "env.data.actuator_force / TORQUE_REF_NM",
        "privileged": bool(privileged),
        "seed": int(seed),
        "seed_offset": int(seed_offset),
        "reset_seed_base": int(1000 * (seed + seed_offset)),
        "reset_seed_rule": "env.reset(seed=reset_seed_base + episode + 1)",
    }
    return data, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".tmp/tactile_estimator/data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="只采指定名字（逗号分隔）")
    ap.add_argument("--include-extra", action="store_true",
                    help="额外采 EXTRA_CASES（v4 数据多样性臂用的训练地形）")
    ap.add_argument("--include-v5", action="store_true",
                    help="v5 高度阶梯（4.5/6/7.5/9/12cm）与插值/外推留出（10.5/13.5/19.5cm）")
    ap.add_argument("--include-pol2", action="store_true",
                    help="v5 A3 臂：每个新高度再配一条不匹配策略（训练）")
    ap.add_argument("--include-a1", action="store_true",
                    help="v5 A1 臂：同类型训练 case 再加 4 回合（数量对照，训练）")
    ap.add_argument("--include-v6-type", action="store_true",
                    help="v6 类型扩展：新 tread/新 vx/新 rough seed（训练）")
    ap.add_argument("--include-v6-qty", action="store_true",
                    help="v6 数量扩展：stair9/stair15 各再 8 回合（seed_offset=200）")
    ap.add_argument("--include-v7", action="store_true",
                    help="v7 Wave A：10 策略 × 12 条件（训练），回合数由 --v7-episodes 决定")
    ap.add_argument("--v7-episodes", type=int, default=2,
                    help="v7 每格回合数（默认 2；补量时用 3，reset seed 规则不变 ⇒ 前 2 回合可复现）")
    ap.add_argument("--include-polshift", action="store_true",
                    help="v7 策略漂移评测集：5 个被剔策略 × 3 高度 × 1 回合（split=polshift，只评不训）")
    ap.add_argument("--strict", action="store_true",
                    help="有 case 因缺策略被跳过时直接失败（v6：数据集不允许静默变小）")
    ap.add_argument("--privileged", action="store_true",
                    help="落盘 D1 上限探测用的特权输入（足端高度/间隙/垂速/接触/地形高）")
    args = ap.parse_args()
    out = Path(args.out)
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    t0 = time.time()
    summary = []
    # 每条 case 记一个 family（写进 meta.case_family，L2 用它核「数据域」）：
    cases: list[tuple[tuple, str]] = [(c, "base") for c in CASES]
    if args.include_extra:
        cases += [(c, "extra") for c in EXTRA_CASES]
    if args.include_v5:
        cases += [(c, "v5") for c in CASES_V5_TRAIN]
        cases += [(c, "v5") for c in CASES_V5_TEST]
    if args.include_pol2:
        cases += [(c, "v5_pol2") for c in CASES_V5_POL2]
    if args.include_a1:
        cases += [(c, "a1") for c in A1_EXTRA]
    if args.include_v6_type:
        cases += [(c, "v6_type") for c in CASES_V6_TYPE]
    if args.include_v6_qty:
        cases += [(c, "v6_qty") for c in CASES_V6_QTY]
    if args.include_v7:
        nv = max(1, int(args.v7_episodes))
        cases += [((c[0], c[1], c[2], c[3], nv, c[5]), "v7") for c in CASES_V7]
    if args.include_polshift:
        cases += [(c, "polshift") for c in CASES_POLSHIFT]
    holdouts = ("stair16_5", "stair18", "rough_s1", "rough_s2",
                "stair10_5", "stair13_5", "stair19_5")
    # 双层守卫（评审 C：只查名字前缀时，换个名字的地形会静默泄漏）：
    # (a) 训练 case 不得与留出地形同名/前缀；(b) 训练 case 的地形 spec
    # （高度, 踏面, 阶数）不得等于任何留出 spec（留出高度集合是硬事实）。
    holdout_specs = set()
    for c in CASES + CASES_V5_TEST:
        if c[0] in holdouts or any(c[0].startswith(h + "_") for h in holdouts):
            holdout_specs.add(spec_key(c[1]()))
    leaked, spec_leaked = [], []
    for case, fam in cases:
        name, spec_fn, policy_path, vx, episodes, split = case[:6]
        if split != "train":
            continue
        if any(name == h or name.startswith(h + "_") for h in holdouts):
            leaked.append(name)
        key = spec_key(spec_fn())
        if key in holdout_specs:
            spec_leaked.append((name, key))
    bad_split = [c[0] for c, _f in cases
                 if c[5] == "test" and not any(
                     c[0] == h or c[0].startswith(h + "_") for h in holdouts)]
    if bad_split:
        raise SystemExit(f"test CASE 不在留出名单里 {bad_split}：拒绝采集")
    if leaked or spec_leaked:
        raise SystemExit(f"训练 CASE 混入留出地形 名字={leaked} 几何={spec_leaked}：拒绝采集")
    skipped: list[dict] = []
    for case, fam in cases:
        name, spec_fn, policy_path, vx, episodes, split = case[:6]
        seed_offset = int(case[6]) if len(case) > 6 else 0
        if only and name not in only:
            continue
        if not Path(policy_path).is_file():
            print(f"SKIP {name}: 缺权重 {policy_path}")
            skipped.append({"name": name, "policy": policy_path,
                            "family": fam, "reason": "missing_policy"})
            continue
        data, meta = collect(name, spec_fn(), policy_path, vx, episodes,
                             seed=args.seed, seed_offset=seed_offset,
                             privileged=args.privileged)
        meta["split"] = split
        meta["case_family"] = fam
        priv = {k: v for k, v in data.items() if k.startswith("priv_")}
        main_arrays = {k: v for k, v in data.items() if not k.startswith("priv_")}
        path = save_dataset(out / f"{split}_{name}.npz", meta=meta,
                            extra=priv or None, **main_arrays)
        contact_rate = float(np.mean(np.asarray(data["contact"]).reshape(-1)))
        fmax = float(np.abs(data["force_world"]).max())
        summary.append({**meta, "file": str(path), "contact_rate": contact_rate,
                        "max_force_n": fmax})
        print(f"{split:5s} {name:18s} steps={meta['steps']:6d} "
              f"eps={episodes:2d} ok={meta['success_episodes']:2d} "
              f"fall={meta['fall_episodes']:2d} contact={contact_rate:.2f} "
              f"|F|max={fmax:6.1f} N")
    payload = {"datasets": summary, "cases_skipped": skipped,
               "n_datasets": len(summary)}
    (out / "collection_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"total {len(summary)} datasets, {time.time() - t0:.1f}s -> {out}")
    if skipped:
        print(f"[warn] {len(skipped)} 个 case 因缺策略被跳过：{[s['name'] for s in skipped]}")
        if args.strict:
            raise SystemExit("--strict：存在被跳过的 case，数据集不完整")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
