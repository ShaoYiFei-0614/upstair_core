"""GRU 足底触觉估计器的契约测试（窗口化 / 指标 / 存档）。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from robotdog.tactile_estimator import (
    EstimatorConfig, TactileEstimator, TactileGRU, contact_metrics,
    feature_stats, force_metrics, gather_windows, load_dataset,
    make_dataset_windows, merge_windows, regression_metrics, save_dataset,
    split_episodes, train_estimator, window_starts,
)


class WindowTest(unittest.TestCase):
    def test_window_starts(self):
        np.testing.assert_array_equal(window_starts(25, 20, 5), [19, 24])
        self.assertEqual(window_starts(10, 20, 2).size, 0)

    def test_gather_windows(self):
        f = np.arange(20, dtype=np.float32).reshape(10, 2)
        w = gather_windows(f, np.array([3]), 3)
        self.assertEqual(w.shape, (1, 3, 2))
        np.testing.assert_array_equal(w[0], f[1:4])


class SplitTest(unittest.TestCase):
    def test_split_never_shares_episode(self):
        episode = np.repeat(np.arange(10), 5)
        tr, va = split_episodes(episode, 0.2, seed=0)
        self.assertTrue(va.any() and tr.any())
        self.assertEqual(set(episode[tr]) & set(episode[va]), set())


class MetricTest(unittest.TestCase):
    def test_force_metrics_perfect(self):
        tgt = np.zeros((7, 4, 3))
        m = force_metrics(tgt, tgt, 100.0)
        self.assertEqual(m["rmse_all_n"], 0.0)
        self.assertEqual(m["rmse_all_bw"], 0.0)
        self.assertEqual(m["r2_all"], 1.0)

    def test_force_metrics_known_error(self):
        tgt = np.zeros((2, 4, 3))
        pred = np.zeros((2, 4, 3))
        pred[:, :, 2] = 10.0
        m = force_metrics(pred, tgt, 100.0)
        self.assertAlmostEqual(m["rmse_z_n"], 10.0)
        self.assertAlmostEqual(m["rmse_z_bw"], 0.1)
        self.assertAlmostEqual(m["mae_all_n"], 10.0 / 3.0)

    def test_contact_metrics(self):
        prob = np.array([[0.9, 0.1, 0.9, 0.2]])
        tgt = np.array([[1.0, 0.0, 1.0, 1.0]])
        m = contact_metrics(prob, tgt, 0.5)
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertAlmostEqual(m["recall"], 2.0 / 3.0)
        self.assertAlmostEqual(m["precision"], 1.0)
        perfect = contact_metrics(tgt, tgt, 0.5)
        self.assertEqual(perfect["f1"], 1.0)

    def test_regression_metrics_r2(self):
        tgt = np.arange(10, dtype=np.float64)
        self.assertAlmostEqual(regression_metrics(tgt, tgt)["r2"], 1.0)
        const = np.zeros(10)
        self.assertLess(regression_metrics(tgt, const)["r2"], 0.0)


class ModelTest(unittest.TestCase):
    def test_forward_shapes(self):
        cfg = EstimatorConfig(window=5, hidden=16, epochs=1)
        model = TactileGRU(n_features=53, cfg=cfg)
        out = model(torch.zeros(3, 5, 53))
        self.assertEqual(tuple(out["force"].shape), (3, 12))
        self.assertEqual(tuple(out["contact"].shape), (3, 4))
        self.assertEqual(tuple(out["area"].shape), (3, 4))
        self.assertEqual(tuple(out["cop"].shape), (3, 8))

    def test_feature_stats_std_floor(self):
        x = np.zeros((2, 3, 4), dtype=np.float32)
        mean, std = feature_stats(x)
        np.testing.assert_allclose(mean, 0.0)
        np.testing.assert_allclose(std, 1.0)   # 标准差下限 1e-6 ⇒ 置 1

    def test_checkpoint_roundtrip(self):
        cfg = EstimatorConfig(window=5, hidden=16, epochs=1, force_ref_n=149.17)
        model = TactileGRU(53, cfg)
        est = TactileEstimator(model, cfg, np.zeros(53, np.float32),
                               np.ones(53, np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "est.pt"
            est.save(path, extra_meta={"k": 1})
            loaded = TactileEstimator.load(path)
            self.assertEqual(loaded.n_features, 53)
            out = loaded.predict(np.zeros((2, 5, 53), np.float32))
            self.assertEqual(out["force"].shape, (2, 4, 3))
            self.assertEqual(out["contact"].shape, (2, 4))
            self.assertEqual(out["area"].shape, (2, 4))
            self.assertEqual(out["cop"].shape, (2, 4, 2))
            self.assertTrue(np.all(np.isfinite(out["force"])))


class DatasetWindowTest(unittest.TestCase):
    """窗口化/合并/存档（reviewer 门禁后补）。"""

    def _data(self):
        n, F = 30, 5
        ep = np.concatenate([np.zeros(20, dtype=np.int32), np.ones(10, dtype=np.int32)])
        return {
            "features": np.random.default_rng(0).standard_normal((n, F)).astype(np.float32),
            "force_world": np.zeros((n, 4, 3), dtype=np.float32),
            "force_foot": np.ones((n, 4, 3), dtype=np.float32),
            "contact": np.ones((n, 4), dtype=np.uint8),
            "area_norm": np.full((n, 4), 0.2, dtype=np.float32),
            "cop_norm": np.zeros((n, 4, 2), dtype=np.float32),
            "phase": np.zeros((n, 4), dtype=np.int8),
            "episode": ep,
        }

    def test_make_dataset_windows_drops_force_foot(self):
        cfg = EstimatorConfig(window=4)
        w = make_dataset_windows(self._data(), cfg, stride=4)
        self.assertNotIn("force_foot", w)
        # 回合 0：20 步 → 终点 3,7,11,15,19；回合 1：10 步 → 3,7,9
        self.assertEqual(int(w["x"].shape[0]), 8)
        self.assertEqual(int(w["x"].shape[1]), 4)
        # 按回合切：窗口内不会跨回合（最后一帧的 episode 即窗口所属回合）
        self.assertEqual(int(w["force"].shape[0]), 8)

    def test_window_starts_includes_last_frame(self):
        starts = window_starts(23, 20, 4)
        self.assertEqual(int(starts[-1]), 22)
        self.assertEqual(int(starts[0]), 19)

    def test_merge_windows(self):
        a = {"x": np.zeros((2, 3, 1), dtype=np.float32),
             "contact": np.zeros((2, 4), dtype=np.float32)}
        b = {"x": np.ones((1, 3, 1), dtype=np.float32),
             "contact": np.ones((1, 4), dtype=np.float32)}
        m = merge_windows([a, b])
        self.assertEqual(m["x"].shape, (3, 3, 1))
        self.assertEqual(float(m["x"][2, 0, 0]), 1.0)

    def test_save_load_dataset_roundtrip(self):
        data = self._data()
        meta = {"name": "synthetic", "split": "train"}
        with tempfile.TemporaryDirectory() as tmp:
            p = save_dataset(Path(tmp) / "d.npz", meta=meta, **data)
            back = load_dataset(p)
            self.assertEqual(back["meta"], meta)
            for k, v in data.items():
                self.assertEqual(tuple(back[k].shape), tuple(np.asarray(v).shape))
            np.testing.assert_array_equal(back["contact"], data["contact"])

    def test_train_estimator_val_loss_finite(self):
        cfg = EstimatorConfig(window=4, hidden=8, epochs=2, batch=32, lr=1e-3)
        rng = np.random.default_rng(1)
        n, F = 80, 6
        ep = np.repeat(np.arange(8, dtype=np.int32), 10)
        feats = rng.standard_normal((n, F)).astype(np.float32)
        fw = rng.standard_normal((n, 4, 3)).astype(np.float32)
        ct = (rng.random((n, 4)) > 0.4).astype(np.uint8)
        area = rng.random((n, 4)).astype(np.float32)
        cop = rng.uniform(-1, 1, (n, 4, 2)).astype(np.float32)
        data = {"features": feats, "force_world": fw, "force_foot": fw,
                "contact": ct, "area_norm": area, "cop_norm": cop,
                "phase": np.zeros((n, 4), dtype=np.int8), "episode": ep}
        w = make_dataset_windows(data, cfg, stride=2)
        tr, va = split_episodes(w["episode"], cfg.val_frac, cfg.seed)
        train = {k: v[tr] for k, v in w.items() if k != "episode"}
        val = {k: v[va] for k, v in w.items() if k != "episode"}
        _est, hist = train_estimator(train, val, cfg, verbose=False)
        self.assertTrue(np.isfinite(hist[-1]["val_loss"]))
        self.assertTrue(np.isfinite(hist[-1]["train_loss"]))


if __name__ == "__main__":
    unittest.main()
