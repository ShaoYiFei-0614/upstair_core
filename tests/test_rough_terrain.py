"""随机粗糙地面（robotdog/tools/rough.py）的几何/高度一致性测试。"""

from __future__ import annotations

import unittest

import numpy as np

from robotdog.env import EnvConfig, Go2StairsEnv, ObsConfig
from robotdog.tactile import TactileConfig
from robotdog.tools.rough import RoughSpec, rough_spec
from robotdog.tools.terrain import build_terrain_xml, flat_spec


class RoughGeometryTest(unittest.TestCase):
    def test_heights_range_and_cache(self):
        spec = rough_spec(0.02, 0.05, seed=0)
        h = spec.heights()
        self.assertEqual(h.shape, (spec.n_x, spec.n_y))
        self.assertGreaterEqual(float(h.min()), 0.0)
        self.assertLessEqual(float(h.max()), spec.amp_hi + 1e-9)
        np.testing.assert_array_equal(h, spec.heights())  # 缓存逐位相同

    def test_height_at_matches_cells(self):
        spec = rough_spec(0.02, 0.05, seed=3)
        h = spec.heights()
        rng = np.random.default_rng(0)
        for _ in range(50):
            x = float(rng.uniform(spec.rough_x0, spec.rough_x1))
            y = float(rng.uniform(-spec.corridor_half_width + 1e-3,
                                  spec.corridor_half_width - 1e-3))
            i = min(spec.n_x - 1, int((x - spec.rough_x0) // spec.cell))
            j = min(spec.n_y - 1,
                    int((y + spec.corridor_half_width) // spec.cell))
            self.assertAlmostEqual(spec.height_at(x, y), float(h[i, j]), places=12)

    def test_outside_is_flat(self):
        spec = rough_spec(0.02, 0.05, seed=0)
        self.assertEqual(spec.height_at(spec.rough_x0 - 0.1), 0.0)
        self.assertEqual(spec.height_at(spec.rough_x1 + 0.1), 0.0)
        self.assertEqual(spec.height_at(3.0, spec.corridor_half_width + 0.2), 0.0)

    def test_xml_and_geom_detection(self):
        spec = rough_spec(0.02, 0.05, seed=0)
        xml = build_terrain_xml(spec)
        self.assertIn("stair_rough_", xml)
        self.assertEqual(xml.count("<geom"), len(spec.step_blocks()))
        self.assertEqual(build_terrain_xml(flat_spec()), "")  # 平地仍不注入几何

    def test_env_compiles_and_steps(self):
        cfg = EnvConfig(obs=ObsConfig(history_len=1, gait_phase=True,
                                      step_height=True, vel_filter=True),
                        tactile=TactileConfig(enabled=True))
        env = Go2StairsEnv(stair=rough_spec(0.02, 0.05, seed=0), cfg=cfg, seed=0)
        try:
            self.assertEqual(len(env.meta.stair_geom_ids),
                             len(env.stair.step_blocks()))
            obs, _ = env.reset(seed=0)
            self.assertEqual(obs.shape, (cfg.obs.size,))
            for _ in range(20):
                obs, _r, term, trunc, info = env.step(np.zeros(env.num_action))
                if term or trunc:
                    obs, _ = env.reset(seed=1)
            self.assertTrue(np.all(np.isfinite(env.foot_clearance)))
            self.assertTrue(np.all(np.isfinite(env._terrain_height(
                env.foot_pos[:, 0], env.foot_pos[:, 1]))))
        finally:
            env.close()


class RoughSpecValidationTest(unittest.TestCase):
    def test_bad_ranges_raise(self):
        with self.assertRaises(ValueError):
            rough_spec(amp_lo=0.05, amp_hi=0.02)
        with self.assertRaises(ValueError):
            rough_spec(cell=0.0)
        with self.assertRaises(ValueError):
            rough_spec(rough_x0=3.0, rough_x1=2.0)


class RoughTaperAndSpawnTest(unittest.TestCase):
    """taper 边界与「地形高度必须带 y」协议（reviewer 门禁后补）。"""

    def test_taper_first_last_columns_are_zero(self):
        for taper in (1, 2, 3, 10):
            spec = rough_spec(0.02, 0.05, seed=2, taper_cells=taper)
            h = spec.heights()
            np.testing.assert_allclose(h[:, 0], 0.0, atol=1e-15)
            np.testing.assert_allclose(h[:, -1], 0.0, atol=1e-15)
            # 出口/入口与区域外（高度 0）连续，没有垂直落差
            self.assertEqual(spec.height_at(spec.rough_x0 - 1e-6), 0.0)
            self.assertEqual(spec.height_at(spec.rough_x1 + 1e-6), 0.0)

    def test_taper_clamped_and_validated(self):
        with self.assertRaises(ValueError):
            rough_spec(taper_cells=0)
        with self.assertRaises(ValueError):
            rough_spec(smooth_passes=-1)
        # taper_cells 大于半宽时被夹到 n_x//2，不抛 IndexError
        spec = rough_spec(rough_x0=2.0, rough_x1=2.24, taper_cells=99)
        self.assertGreaterEqual(spec.n_x, 2)
        self.assertTrue(np.all(np.isfinite(spec.heights())))

    def test_env_terrain_height_uses_y(self):
        spec = rough_spec(0.02, 0.05, seed=7)
        env = Go2StairsEnv(stair=spec, cfg=EnvConfig(), seed=0)
        rng = np.random.default_rng(1)
        for _ in range(20):
            x = float(rng.uniform(spec.rough_x0 + 0.1, spec.rough_x1 - 0.1))
            y = float(rng.uniform(-spec.corridor_half_width + 1e-2,
                                  spec.corridor_half_width - 1e-2))
            got = float(np.ravel(env._terrain_height(x, y))[0])
            self.assertAlmostEqual(got, float(spec.height_at(x, y)), places=12)
        env.close()

    def test_spawn_on_rough_uses_analytic_height(self):
        spec = rough_spec(0.02, 0.05, seed=3)
        cfg = EnvConfig(spawn_back=-0.10, reset_xy_noise=0.0,
                        reset_joint_noise=0.0)
        env = Go2StairsEnv(stair=spec, cfg=cfg, seed=5)
        env.reset(seed=5)
        x, y, z = env.base_pos
        self.assertAlmostEqual(float(z),
                               float(spec.height_at(x, y)) + cfg.reset_z,
                               places=9)
        self.assertGreater(float(spec.height_at(x, y)), 0.0)
        env.close()

    def test_foot_clearance_finite_on_rough(self):
        spec = rough_spec(0.02, 0.05, seed=4)
        env = Go2StairsEnv(stair=spec, cfg=EnvConfig(spawn_back=0.10), seed=0)
        env.reset(seed=0)
        for _ in range(40):
            _, _, term, trunc, _ = env.step(np.zeros(env.num_action))
            clr = env.foot_clearance
            self.assertTrue(np.all(np.isfinite(clr)))
            self.assertTrue(np.all(clr > -1e-9))
            if term or trunc:
                env.reset()
        env.close()


if __name__ == "__main__":
    unittest.main()
