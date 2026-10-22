"""L1-stance 奖励修正 + warm-start 回放选取的契约测试（unittest，不依赖 pytest）。

对应 2026-10-21 的改动（doc/introduce_tactile.md §9）：
1. `follow_zero_baseline`：零指令通道按「完美跟踪=0 分」计，堵掉 l1_st 的免费奖励；
2. `stance_min_vx`：支撑奖励按前进速度门控，堵掉「蹲稳不动」；
3. `tactile_slip`：支撑相足端水平滑移惩罚（负权重，有界）；
4. 三项默认中性 ⇒ 逐步奖励逐位不变（eq_check 另测）；
5. `replay_success_rate`：训练循环的 warm-start 选取口径（确定性、不钉指令）。

跑法：.venv\\Scripts\\python.exe -m unittest tests.test_stance_reward -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robotdog.env import CommandConfig, EnvConfig, Go2StairsEnv, ObsConfig
from robotdog.policy import save_env_json
from robotdog.reward import compute_reward
from robotdog.tactile import STANCE, SWING, TactileConfig
from robotdog.tools.terrain import flat_spec
from robotdog.training import (
    build_ppo,
    make_vec_env,
    replay_success_rate,
    save_vecnormalize,
    warm_select_stages,
)


def _cfg(*, command: CommandConfig | None = None, **reward_kw) -> EnvConfig:
    """构造测试配置：``command`` 必须传给 ``EnvConfig``（不是 reward）。"""
    cfg = EnvConfig(obs=ObsConfig(history_len=1, gait_phase=True,
                                  step_height=True, vel_filter=True),
                    tactile=TactileConfig(enabled=True),
                    command_wander_std=0.0,
                    command=command if command is not None else CommandConfig())
    for k, v in reward_kw.items():
        setattr(cfg.reward, k, v)
    return cfg


def _env(cfg: EnvConfig) -> Go2StairsEnv:
    env = Go2StairsEnv(stair=flat_spec(), cfg=cfg, seed=0)
    env.reset(seed=0)
    return env


class ZeroBaselineTest(unittest.TestCase):
    def test_zero_command_channels_stop_paying(self) -> None:
        # 零横向/零偏航级别（vy_max=yaw_max=0）：这两条通道没有任务内容，
        # 旧实现里「站着不动」每步白给 lateral_vel + yaw_rate ≈ 10 分。
        cmd = CommandConfig(vy_max=0.0, yaw_max=0.0)
        legacy = _env(_cfg(command=cmd, follow_zero_baseline=False))
        base = _env(_cfg(command=cmd, follow_zero_baseline=True))
        zero = np.zeros(12)
        gaps = []
        for _ in range(30):
            _, _, term0, trunc0, info0 = legacy.step(zero)
            _, _, term1, trunc1, info1 = base.step(zero)
            gaps.append(info0["reward_terms"]["command_follow"]
                        - info1["reward_terms"]["command_follow"])
            if term0 or term1 or trunc0 or trunc1:
                legacy.reset(seed=0); base.reset(seed=0)
        # 零横向/零偏航通道每步白给 lateral_vel + yaw_rate ≈ 10/步
        self.assertGreater(float(np.mean(gaps)), 3.0,
                           msg="零指令通道的免费奖励没有被扣掉")

    def test_nonzero_lateral_command_keeps_track_reward(self) -> None:
        """横向指令量程 >0 时**不**扣基线（只有偏航通道是零指令）。"""
        cmd = CommandConfig(vy_max=0.20, yaw_max=0.0)
        legacy = _env(_cfg(command=cmd, follow_zero_baseline=False))
        base = _env(_cfg(command=cmd, follow_zero_baseline=True))
        zero = np.zeros(12)
        gaps = []
        for _ in range(20):
            _, _, term0, trunc0, info0 = legacy.step(zero)
            _, _, term1, trunc1, info1 = base.step(zero)
            gaps.append(info0["reward_terms"]["command_follow"]
                        - info1["reward_terms"]["command_follow"])
            if term0 or term1 or trunc0 or trunc1:
                legacy.reset(seed=0); base.reset(seed=0)
        # 只剩偏航通道被扣 1.0 ⇒ 每步差 ≈ yaw_rate = 4
        self.assertGreater(float(np.mean(gaps)), 3.0)
        self.assertLess(float(np.mean(gaps)), 5.0)


class StanceGateTest(unittest.TestCase):
    def test_stance_reward_requires_forward_speed(self) -> None:
        free = _env(_cfg(tactile_stance=1.0, stance_min_vx=0.0))
        gated = _env(_cfg(tactile_stance=1.0, stance_min_vx=10.0))
        zero = np.zeros(12)
        vals_free, vals_gated = [], []
        for _ in range(30):
            _, _, t0, tr0, i0 = free.step(zero)
            _, _, t1, tr1, i1 = gated.step(zero)
            vals_free.append(i0["reward_terms"]["tactile_stance"])
            vals_gated.append(i1["reward_terms"]["tactile_stance"])
            if t0 or tr0 or t1 or tr1:
                break
        self.assertGreater(max(vals_free), 0.0)
        self.assertAlmostEqual(max(vals_gated), 0.0, places=9)


class SlipPenaltyTest(unittest.TestCase):
    def test_slip_term_is_bounded_negative(self) -> None:
        env = _env(_cfg(tactile_slip=-3.0, slip_ref=0.005))
        rng = np.random.default_rng(0)
        vals = []
        for _ in range(60):
            _, _, done, trunc, info = env.step(rng.uniform(-1, 1, 12))
            vals.append(info["reward_terms"]["tactile_slip"])
            if done or trunc:
                env.reset(seed=0)
        self.assertLessEqual(max(vals), 0.0)
        self.assertGreaterEqual(min(vals), -3.0 - 1e-9)
        self.assertLess(min(vals), -1e-6, msg="随机动作下支撑相应有滑移")

    def test_slip_key_absent_only_when_all_tactile_weights_zero(self) -> None:
        # 触觉三项全 0 ⇒ 整段跳过，键不存在；但只把 slip 置 0、stance>0 时，
        # 键**存在且恒为 0.0**（有意新增项，不是 bug）。
        env = _env(_cfg())
        _, _, _, _, info = env.step(np.zeros(12))
        self.assertNotIn("tactile_slip", info["reward_terms"])
        env2 = _env(_cfg(tactile_stance=1.0))
        _, _, _, _, info2 = env2.step(np.zeros(12))
        self.assertIn("tactile_slip", info2["reward_terms"])
        self.assertEqual(info2["reward_terms"]["tactile_slip"], 0.0)


class ValidationTest(unittest.TestCase):
    def test_slip_requires_tactile_enabled(self) -> None:
        cfg = EnvConfig(tactile=TactileConfig(enabled=False))
        cfg.reward.tactile_slip = -1.0
        with self.assertRaises(ValueError):
            Go2StairsEnv(stair=flat_spec(), cfg=cfg, seed=0)


class ReplayHelperTest(unittest.TestCase):
    def test_replay_success_rate_runs(self) -> None:
        cfg = EnvConfig(command_wander_std=0.0)
        venv = make_vec_env(flat_spec(), n_envs=1, cfg=cfg, seed=0,
                            normalize=False, monitor=True, norm_reward=False)
        try:
            model = build_ppo(venv, seed=0, net_arch=(32,), n_steps=16,
                              batch_size=16, n_epochs=1, verbose=0)
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp) / "model"
                model.save(str(base))
                save_vecnormalize(model, base)
                save_env_json(base, cfg, step_height=0.0, stage="flat")
                stats = replay_success_rate(str(base) + ".zip", episodes=2, seed=0)
        finally:
            venv.close()
        self.assertEqual(stats["episodes"], 2.0)
        self.assertTrue(0.0 <= stats["success_rate"] <= 1.0)
        self.assertTrue(0.0 <= stats["fall_rate"] <= 1.0)
        self.assertGreater(stats["mean_steps"], 0.0)


class NeutralAndBoundaryTest(unittest.TestCase):
    """补 reviewer 指出的缺口：默认键集、死区边界、门控中间比例。"""

    def test_default_env_has_no_tactile_keys(self) -> None:
        # 默认 EnvConfig：tactile 关、新奖励权重全 0 ⇒ 不产生 tactile_* 键，
        # 且 command_follow 仍是旧公式（follow_zero_baseline=False）。
        env = Go2StairsEnv(stair=flat_spec(), cfg=EnvConfig(), seed=0)
        try:
            env.reset(seed=0)
            keys = set()
            for _ in range(5):
                _, _, term, trunc, info = env.step(np.zeros(12))
                keys.update(info["reward_terms"])
                if term or trunc:
                    env.reset(seed=0)
            self.assertFalse([k for k in keys if k.startswith("tactile_")])
        finally:
            env.close()

    def test_zero_baseline_deadband_boundary(self) -> None:
        # 死区 0.05：|vy_cmd|=0.04 落在死区内（扣基线），|vy_cmd|=0.06 在死区外（不扣）。
        def one_step(vy: float, baseline: bool) -> float:
            cfg = EnvConfig(obs=ObsConfig(history_len=1, gait_phase=True,
                                          step_height=True, vel_filter=True),
                            command_wander_std=0.0,
                            command=CommandConfig(vy_max=0.0, yaw_max=0.0))
            cfg.reward.follow_zero_baseline = baseline
            # 本用例只量 vy 通道：把偏航死区设为 0，避免零偏航通道再扣 1 份
            cfg.reward.zero_cmd_eps_wz = 0.0
            env = Go2StairsEnv(stair=flat_spec(), cfg=cfg, seed=0)
            try:
                env.reset(seed=0)
                cmd = [0.0, vy, 0.0]
                env._command[:] = cmd
                env._cmd_hold[:] = cmd
                env._cmd_age = -(10 ** 9)
                _, _, _, _, info = env.step(np.zeros(12))
                return float(info["reward_terms"]["command_follow"])
            finally:
                env.close()

        gap_in = one_step(0.04, True) - one_step(0.04, False)
        gap_out = one_step(0.06, True) - one_step(0.06, False)
        # 死区内：基线与关基线差一个满额跟踪项；死区外：两者完全一致
        self.assertLess(gap_in, -1.0)
        self.assertAlmostEqual(gap_out, 0.0, places=9)

    def test_stance_gate_scales_with_vx_reference(self) -> None:
        # 同一状态（前进速度 >0.2 m/s）、只改 stance_min_vx：
        # 参考速度小 ⇒ 门控≈1；参考速度极大 ⇒ 门控≈0。
        def gate(min_vx: float) -> float:
            cfg = _cfg(stance_min_vx=min_vx, tactile_stance=1.0)
            env = _env(cfg)
            try:
                env.reset(seed=0)
                # 直接构造状态：机身系前进速度 0.4 m/s + 一个支撑相样本
                env.data.qvel[0:3] = [0.4, 0.0, 0.0]
                env.data.qvel[3:6] = [0.0, 0.0, 0.0]
                env._tactile.phase[:] = STANCE
                env._tactile.area_norm[:] = 0.2
                env._tactile.margin[:] = 0.3
                env._tactile.cop_jump[:] = 0.0
                _, terms, _ = compute_reward(env, np.zeros(12), False)
                return float(terms.get("tactile_stance", 0.0))
            finally:
                env.close()

        slow = gate(0.01)
        fast = gate(1.0e6)
        self.assertGreater(slow, 0.0)
        self.assertLess(abs(fast), 0.02 * slow)


class ZeroBaselineYawDeadbandTest(unittest.TestCase):
    """偏航通道的死区镜像用例（vy 用例把 eps_wz 置 0，覆盖不到这条分支）。"""

    def test_yaw_deadband_mirrors_vy(self) -> None:
        def one_step(wz: float, baseline: bool) -> float:
            cfg = EnvConfig(obs=ObsConfig(history_len=1, gait_phase=True,
                                          step_height=True, vel_filter=True),
                            command_wander_std=0.0,
                            command=CommandConfig(vy_max=0.0, yaw_max=0.0))
            cfg.reward.follow_zero_baseline = baseline
            cfg.reward.zero_cmd_eps_vy = 0.0   # 本用例只量偏航通道
            env = Go2StairsEnv(stair=flat_spec(), cfg=cfg, seed=0)
            try:
                env.reset(seed=0)
                cmd = [0.0, 0.0, float(wz)]
                env._command[:] = cmd
                env._cmd_hold[:] = cmd
                env._cmd_age = -(10 ** 9)
                _, _, _, _, info = env.step(np.zeros(12))
                return float(info["reward_terms"]["command_follow"])
            finally:
                env.close()

        gap_in = one_step(0.06, True) - one_step(0.06, False)   # |wz| < 0.10
        gap_out = one_step(0.14, True) - one_step(0.14, False)  # |wz| > 0.10
        self.assertLess(gap_in, -1.0)
        self.assertAlmostEqual(gap_out, 0.0, places=9)


class SlipMaskTest(unittest.TestCase):
    """滑移项只结算支撑相：非 STANCE 足的位移不计入。"""

    def test_only_stance_feet_count(self) -> None:
        cfg = EnvConfig(obs=ObsConfig(history_len=1, gait_phase=True,
                                      step_height=True, vel_filter=True),
                        command_wander_std=0.0)
        cfg.tactile.enabled = True
        cfg.reward.tactile_slip = -3.0
        cfg.reward.slip_ref = 0.005
        env = Go2StairsEnv(stair=flat_spec(), cfg=cfg, seed=0)
        try:
            env.reset(seed=0)
            # 上一步足端整体后移 5cm（远超 slip_ref=5mm）
            env._prev_foot_pos = np.asarray(env.foot_pos).copy() - 0.05
            env._tactile.phase[:] = SWING
            _, t_swing, _ = compute_reward(env, np.zeros(12), False)
            self.assertEqual(float(t_swing.get("tactile_slip", 0.0)), 0.0)
            env._tactile.phase[:] = STANCE
            env._tactile.area_norm[:] = 0.2
            env._tactile.margin[:] = 0.3
            env._tactile.cop_jump[:] = 0.0
            _, t_stance, _ = compute_reward(env, np.zeros(12), False)
            self.assertLess(float(t_stance["tactile_slip"]), 0.0)
        finally:
            env.close()


class WarmSelectStagesTest(unittest.TestCase):
    """B1 回归：warm-select 只对「有下一级」的级生效（1-based 1..n-1）。"""

    def test_stage_indexes_exclude_last(self) -> None:
        self.assertEqual(warm_select_stages(16), list(range(1, 16)))
        self.assertEqual(len(warm_select_stages(16)), 15)
        self.assertNotIn(16, warm_select_stages(16))
        self.assertEqual(warm_select_stages(2), [1])
        self.assertEqual(warm_select_stages(1), [])
        self.assertEqual(warm_select_stages(0), [])


if __name__ == "__main__":
    unittest.main()