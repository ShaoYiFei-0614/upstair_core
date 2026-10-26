"""估计触觉块（robotdog/tactile_estimate_obs.py）与 torch 估计器的最小对拍测试。

覆盖：
1) numpy 手写 GRU 前向 == TactileEstimator.predict（同一权重、同一窗口）；
2) reset/update 语义（形状、有限性）；
3) 配置默认值。

权重缺失时自动 skip。运行：.venv/Scripts/python.exe -m unittest tests.test_tactile_estimate_obs
"""
import os
import unittest

import numpy as np

from robotdog.tactile_estimate_obs import (
    EST_BLOCK_SIZE,
    EstimateObsConfig,
    TactileEstimateObserver,
)

NPZ = "artifacts/tactile_est_C3c_s0.npz"
PT = ".tmp/tactile_est_v7/ckpts/C3c_s0.pt"


class TactileEstimateObsTest(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(NPZ):
            self.skipTest("缺少 %s" % NPZ)
        self.obs = TactileEstimateObserver.from_npz(NPZ)

    def test_block_size_constant(self):
        self.assertEqual(EST_BLOCK_SIZE, 8)
        self.assertEqual(self.obs.n_actor + 12, self.obs.n_features)

    def test_config_defaults(self):
        cfg = EstimateObsConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.stride, 1)
        self.assertGreater(cfg.force_ref_n, 0.0)

    def test_parity_with_torch(self):
        if not os.path.exists(PT):
            self.skipTest("缺少 %s" % PT)
        try:
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest("无 torch: %s" % exc)
        from robotdog.tactile_estimator import TactileEstimator

        est = TactileEstimator.load(PT)
        rng = np.random.default_rng(0)
        window = rng.normal(size=(3, self.obs.window, self.obs.n_actor)).astype(np.float32)
        tau = rng.normal(size=(3, self.obs.window, 12)).astype(np.float32)
        batch = np.concatenate([window, tau], axis=-1)  # (B,T,65)
        ref = est.predict(batch)
        got = np.concatenate([self.obs.predict_window(batch[i]) for i in range(batch.shape[0])])
        want = np.concatenate([
            np.linalg.norm(ref["force"], axis=-1).reshape(3, 4) / est.cfg.force_ref_n,
            ref["area"].reshape(3, 4),
        ], axis=-1).reshape(-1)
        self.assertEqual(got.shape, want.shape)
        self.assertEqual(got.shape, (3 * EST_BLOCK_SIZE,))
        self.assertLess(float(np.max(np.abs(got - want))), 1e-4)

    def test_reset_and_update_semantics(self):
        frame = np.zeros(self.obs.n_actor, dtype=np.float32)
        tau = np.zeros(12, dtype=np.float32)
        first = self.obs.reset(frame, tau)
        self.assertEqual(first.shape, (EST_BLOCK_SIZE,))
        self.assertTrue(np.all(np.isfinite(first)))
        frame2 = frame.copy()
        frame2[0] = 1.0
        blk = self.obs.update(frame2, tau)
        self.assertEqual(blk.shape, (EST_BLOCK_SIZE,))
        self.assertTrue(np.all(np.isfinite(blk)))


if __name__ == "__main__":
    unittest.main()
