"""环境契约 / 审计回归测试（unittest，不依赖 pytest）。

把 2026-10-18 审计（``doc/audit_report.md`` B1–B8）里"已经确认并修掉"的 bug
钉成可执行契约。跑法::

    .venv\\Scripts\\python.exe -m unittest tests.test_env_contract -v

设计原则：每条测试都对着**一个曾经真实出错的行为**，不做无意义的 API 冒泡；
需要随机性时把噪声项置 0，断言用精确相等或明确的物理量（不是"跑通就行"）。
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from robotdog.env import EnvConfig, Go2StairsEnv, ObsConfig
from robotdog.policy import env_config_from_summary
from robotdog.tools.terrain import demo_stairs_spec, flat_spec


def make_env(stair=None, cfg: EnvConfig | None = None, seed: int = 0) -> Go2StairsEnv:
    return Go2StairsEnv(stair=stair if stair is not None else flat_spec(),
                        cfg=cfg, seed=seed)


class ObsShapeTest(unittest.TestCase):
    """B5：观测维数必须是"现算 + 自查"的，文档/注释写错不能被信。"""

    def test_documented_dims(self) -> None:
        # 42（本体感知）+ 6（结构块：机身系线速度 3 + 三轴指令 3）
        # + gait_phase 2 + step_height 1 + vel_filter 2 [+ gait_cmd 2]
        cases = [
            ((False, False, False, False), 48),
            ((False, False, False, True), 50),
            ((True, True, True, False), 53),   # config.yaml 的当前配置
            ((True, True, True, True), 55),
        ]
        for (gp, sh, vf, gc), expected in cases:
            cfg = EnvConfig(obs=ObsConfig(history_len=1, gait_phase=gp,
                                          step_height=sh, vel_filter=vf,
                                          gait_cmd=gc))
            env = make_env(cfg=cfg)
            self.assertEqual(cfg.obs.size, expected)
            self.assertEqual(env.observation_space.shape, (expected,))
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (expected,))


class CommandStateTest(unittest.TestCase):
    """B1/B2/B3：指令写入口的返回、观测刷新、hold 语义。"""

    def test_set_command_now_returns_and_refreshes_obs(self) -> None:
        cfg = EnvConfig(obs=ObsConfig(history_len=1, gait_phase=True,
                                      step_height=True, vel_filter=True),
                        command_wander_std=0.0)
        env = make_env(cfg=cfg, seed=3)
        env.reset()
        before = env._hist[-1].copy()
        cmd = env.set_command_now(vx=0.9, vy=0.2, yaw_rate=0.4, hold=True)
        # B1：旧实现 return 在函数外面（其实是缩进错位进了下一个函数），返回 None。
        self.assertIsInstance(cmd, list)
        np.testing.assert_allclose(cmd, [0.9, 0.2, 0.4])
        # B3：写完指令后，_hist 里"已经算好待取"的那一帧必须立刻同步。
        np.testing.assert_allclose(env._hist[-1], env._obs_frame())
        self.assertFalse(np.allclose(before, env._hist[-1]),
                         "写入指令后观测没变 —— 刷新逻辑失效")

    def test_hold_command_survives_wander(self) -> None:
        # B2：旧条件 ``_cmd_age < command_hold_steps`` 恒为真（_cmd_age 总被
        # 重置回 [0, hold_steps)），于是每步都会把 hold 的指令覆盖成 hold+wander。
        cfg = EnvConfig(obs=ObsConfig(history_len=1),
                        command_wander_std=0.05,
                        command_hold_steps=10)
        env = make_env(cfg=cfg, seed=4)
        env.reset()
        target = [0.8, 0.1, 0.3]
        env.set_command_now(vx=target[0], vy=target[1], yaw_rate=target[2],
                            hold=True)
        for i in range(40):
            _, _, terminated, truncated, _ = env.step(np.zeros(env.num_action))
            np.testing.assert_allclose(env._command, target, atol=1e-12,
                                       err_msg=f"第 {i} 步 hold 指令被改写")
            if terminated or truncated:
                env.reset()
                env.set_command_now(vx=target[0], vy=target[1],
                                    yaw_rate=target[2], hold=True)

    def test_envs_do_not_share_command_state(self) -> None:
        # B3 的孪生问题：8 个 env 共享同一个 cfg 时 _command 就是共享 list。
        from robotdog.training import make_env_fn

        cfg = EnvConfig(obs=ObsConfig(history_len=1), command_wander_std=0.0)
        e0 = make_env_fn(flat_spec(), cfg, seed=1)()
        e1 = make_env_fn(flat_spec(), cfg, seed=2)()
        e0.reset()
        e1.reset()
        self.assertIsNot(e0.cfg, e1.cfg,
                         "make_env_fn 没有给每个 env 复制 cfg —— 指令状态会串台")
        before = list(e1._command)
        e0.set_command_now(vx=0.9, vy=0.2, yaw_rate=0.4, hold=True)
        self.assertEqual(list(e1._command), before)


class ResetSpawnTest(unittest.TestCase):
    """B4：reset 的播种高度必须取**播种点**的地形高度，不能取 qpos0 处的高度。"""

    def test_spawn_on_stairs_uses_analytic_height(self) -> None:
        stair = demo_stairs_spec(step_height=0.06, tread=0.30, n_up=8)
        cfg = EnvConfig(spawn_back=-0.90, reset_xy_noise=0.0,
                        reset_joint_noise=0.0,
                        obs=ObsConfig(history_len=1))
        env = make_env(stair=stair, cfg=cfg, seed=5)
        env.reset()
        x = float(env.base_pos[0])
        ground = float(stair.height_at(x))
        self.assertGreater(ground, 0.10, "测试前提：播种点应落在台阶上")
        self.assertAlmostEqual(float(env.base_pos[2]), ground + cfg.reset_z,
                               places=9)


class GaitPhaseTest(unittest.TestCase):
    """B6：条件化分支的相位口径必须与 GaitParams 内部约定一致（秒制）。"""

    def test_gait_cmd_and_uncond_agree_at_same_freq(self) -> None:
        base = dict(history_len=1)
        uncond_cfg = EnvConfig(gait_speed_track=False,
                               obs=ObsConfig(gait_cmd=False, **base))
        cond_cfg = EnvConfig(gait_speed_track=False,
                             obs=ObsConfig(gait_cmd=True, **base))
        uncond = make_env(cfg=uncond_cfg, seed=6)
        cond = make_env(cfg=cond_cfg, seed=6)
        uncond.reset()
        cond.reset()
        f = float(cond.cfg.gait.freq)
        cond._gait_freq_cmd = f
        cond._gait_swing_cmd = 1.0
        t = 0.137
        uncond._gait_phase = t
        cond._gait_phase = t
        np.testing.assert_allclose(cond._gait_action(), uncond._gait_action(),
                                   atol=1e-12)


class YawRateTest(unittest.TestCase):
    """B7：偏航角速度三种口径的数值契约（旧口径是双重旋转）。"""

    def test_three_modes(self) -> None:
        cfg = EnvConfig(obs=ObsConfig(history_len=1))
        env = make_env(cfg=cfg, seed=7)
        env.reset()
        pitch = 0.3
        # 只留 pitch：qpos 四元数 (w, x, y, z) = (cos(p/2), 0, sin(p/2), 0)
        env.data.qpos[3:7] = [math.cos(pitch / 2), 0.0,
                              math.sin(pitch / 2), 0.0]
        env.data.qvel[:] = 0.0
        env.data.qvel[3:6] = [0.5, 0.2, 0.3]   # 机体系角速度
        mujoco.mj_forward(env.model, env.data)

        cfg.yaw_rate_mode = "body_z"
        self.assertAlmostEqual(env.yaw_rate, 0.3, places=12)
        cfg.yaw_rate_mode = "world"
        self.assertAlmostEqual(env.yaw_rate, 0.3 / math.cos(pitch), places=12)
        # legacy：R(yaw)^T·ω 的 z 分量；yaw=0 时 = sinθ·ωx + cosθ·ωz。
        cfg.yaw_rate_mode = "legacy"
        expected = math.sin(pitch) * 0.5 + math.cos(pitch) * 0.3
        self.assertAlmostEqual(env.yaw_rate, expected, places=12)
        self.assertGreater(abs(expected - 0.3), 0.1,
                           "测试前提：legacy 与 body_z 的差应显著")

    def test_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            make_env(cfg=EnvConfig(yaw_rate_mode="body-z"))


class GuidanceHoldTest(unittest.TestCase):
    """B8：楼梯级引导是训练期的指令成形；``hold=True`` 默认沿用（复现训练条件），
    ``stair_guidance_under_hold=False`` 时必须让 hold 真正钉住 vy/wz。"""

    def _pinned_env(self, *, under_hold: bool) -> Go2StairsEnv:
        stair = demo_stairs_spec(step_height=0.06, tread=0.30, n_up=8)
        cfg = EnvConfig(obs=ObsConfig(history_len=1),
                        command_wander_std=0.0,
                        stair_guidance=True,
                        stair_guidance_under_hold=under_hold)
        env = make_env(stair=stair, cfg=cfg, seed=8)
        env.reset()
        env.set_command_now(vx=0.5, vy=0.12, yaw_rate=0.30, hold=True)
        return env

    def test_default_reproduces_training_guidance(self) -> None:
        env = self._pinned_env(under_hold=True)
        env.step(np.zeros(env.num_action))
        self.assertFalse(np.allclose(env._command[1:], [0.12, 0.30]),
                         "默认行为应是训练条件：楼梯级 vy/wz 由引导生成")

    def test_strict_hold_pins_vy_wz_on_stairs(self) -> None:
        env = self._pinned_env(under_hold=False)
        for i in range(20):
            _, _, terminated, truncated, _ = env.step(np.zeros(env.num_action))
            np.testing.assert_allclose(
                env._command[1:], [0.12, 0.30], atol=1e-12,
                err_msg=f"第 {i} 步 strict hold 的 vy/wz 被引导改写")
            if terminated or truncated:
                env.reset()
                env.set_command_now(vx=0.5, vy=0.12, yaw_rate=0.30, hold=True)


class ActionDimTest(unittest.TestCase):
    def test_num_action_is_12_in_both_modes(self) -> None:
        for mode in ("residual", "gait_params"):
            env = make_env(cfg=EnvConfig(action_mode=mode))
            self.assertEqual(env.num_action, 12)
            self.assertEqual(env.action_space.shape, (12,))


class LegacyCheckpointCompatTest(unittest.TestCase):
    """复现旧 checkpoint（2026-10-14 之前的 ``_env.json``）时的字段回填。

    ``output/sub9_v5`` 这一代 checkpoint 的 ``_env.json`` 里没有
    ``gait_freq_flat`` / ``gait_freq_stair`` / ``gait_stair`` —— 当年的相位时钟就是
    ``cfg.gait.freq``（trot_natural 2.0 Hz）。若让它们落到今天的默认值
    （平地 1.67 / 楼梯按台阶高插值），``gait_phase`` 两维观测的频率就错了。
    2026-10-18 实测：sub9_v5 的 9cm 权重在楼梯上从 2/3 成功掉到 0/3，回填后恢复。
    """

    def _legacy_payload(self, *, modern: bool) -> dict:
        base = EnvConfig(obs=ObsConfig(history_len=1, gait_phase=True,
                                        step_height=True, vel_filter=True))
        env_cfg = asdict(base)
        gait = dict(env_cfg["gait"])
        gait["freq"] = 2.0                      # trot_natural 的频率
        env_cfg["gait"] = gait
        env_cfg.pop("gait_stair", None)
        if modern:
            env_cfg["gait_freq_flat"] = 1.67
            env_cfg["gait_freq_stair"] = 1.25
        else:
            # 模拟旧权重：字段根本不存在（而不是值为 None）
            env_cfg.pop("gait_freq_flat", None)
            env_cfg.pop("gait_freq_stair", None)
            env_cfg.pop("gait_freq_stair_fixed", None)
        return {"env_config": env_cfg,
                "obs_config": asdict(base.obs),
                "reward_config": asdict(base.reward)}

    def _load(self, payload: dict):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return env_config_from_summary(path)

    def test_legacy_json_backfills_gait_freq(self) -> None:
        cfg = self._load(self._legacy_payload(modern=False))
        self.assertEqual(cfg.obs.size, 53)
        self.assertAlmostEqual(cfg.gait_freq_flat, 2.0)          # = cfg.gait.freq
        self.assertAlmostEqual(cfg.gait_freq_stair, 2.0)
        # 决定观测相位频率的是 gait_freq_stair_fixed（_gait_freq_eff 的楼梯分支）
        self.assertAlmostEqual(cfg.gait_freq_stair_fixed, 2.0)

    def test_modern_json_keeps_its_own_gait_freq(self) -> None:
        cfg = self._load(self._legacy_payload(modern=True))
        self.assertAlmostEqual(cfg.gait_freq_flat, 1.67)
        self.assertAlmostEqual(cfg.gait_freq_stair, 1.25)
        self.assertEqual(cfg.gait_freq_stair_fixed, 0.0)  # 不覆盖新权重的降频设计


if __name__ == "__main__":
    unittest.main()