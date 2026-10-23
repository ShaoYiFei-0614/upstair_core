"""v7 Wave C 契约测试：双频幅值目标（因果均值）与训练侧接触去抖。"""

import numpy as np
import torch
import unittest

from robotdog.tactile_estimator import (
    EstimatorConfig, causal_mean_labels, debounce_contact_labels,
    make_dataset_windows, multi_task_loss, TactileGRU,
)


def _episode():
    return np.array([0] * 5 + [1] * 4, dtype=np.int64)


class CausalMeanTest(unittest.TestCase):
    def test_ramp_matches_manual(self):
        v = np.arange(9, dtype=np.float32).reshape(9, 1)
        out = causal_mean_labels(v, 3, _episode())
        # 回合 0：0; 0.5; 1; 2; 3（起点截断）；回合 1 重新起算
        np.testing.assert_allclose(out[:5, 0], [0.0, 0.5, 1.0, 2.0, 3.0], rtol=1e-6)
        np.testing.assert_allclose(out[5:, 0], [5.0, 5.5, 6.0, 7.0], rtol=1e-6)

    def test_no_cross_episode_leak(self):
        v = np.ones((9, 2), dtype=np.float32)
        v[5:] = 100.0
        out = causal_mean_labels(v, 4, _episode())
        # 回合 1 的第一帧只能看到自己，不能吃到回合 0 的 1.0
        self.assertAlmostEqual(float(out[5, 0]), 100.0, places=5)

    def test_k1_identity(self):
        rng = np.random.default_rng(0)
        v = rng.normal(size=(9, 4)).astype(np.float32)
        np.testing.assert_array_equal(causal_mean_labels(v, 1, _episode()), v)


class DebounceTest(unittest.TestCase):
    def test_majority_vote(self):
        c = np.array([[1], [0], [1], [0], [0], [1], [1], [1], [0]], dtype=np.float32)
        out = debounce_contact_labels(c, _episode(), window=3, votes=2)
        # 因果口径：回合开头只有 1 帧可用（1 票 < 2 票）⇒ 0；之后按最近 3 帧
        # 的多数票；回合 1 重新起算。
        self.assertEqual(out[:, 0].tolist(),
                         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    def test_votes1_identity(self):
        c = np.array([[1], [0]], dtype=np.float32)
        np.testing.assert_array_equal(
            debounce_contact_labels(c, np.array([0, 0]), votes=1), c)


class ConfigContractTest(unittest.TestCase):
    def test_defaults_off(self):
        cfg = EstimatorConfig()
        self.assertEqual(cfg.bifreq_k, 0)
        self.assertFalse(cfg.debounce_contact)

    def test_bifreq_requires_head(self):
        with self.assertRaises(ValueError):
            EstimatorConfig(bifreq_k=5)

    def test_bifreq_range(self):
        with self.assertRaises(ValueError):
            EstimatorConfig(bifreq_k=200, w_forcenorm=0.5)

    def test_roundtrip_keeps_fields(self):
        cfg = EstimatorConfig(bifreq_k=5, w_forcenorm=0.5, debounce_contact=True)
        again = EstimatorConfig(**cfg.to_dict())
        self.assertEqual(again.bifreq_k, 5)
        self.assertTrue(again.debounce_contact)


def _toy_data(cfg, n=120):
    rng = np.random.default_rng(0)
    contact = (rng.random((n, 4)) > 0.4).astype(np.float32)
    force_norm = (contact * rng.uniform(0.2, 1.2, size=(n, 4))).astype(np.float32)
    force_world = np.zeros((n, 4, 3), dtype=np.float32)
    force_world[:, :, 2] = force_norm * 149.17
    # 接触抖一抖，评估去抖效果
    flip = contact.copy()
    flip[::7] = 1.0 - flip[::7]
    return {
        "features": rng.normal(size=(n, 65)).astype(np.float32),
        "force_world": force_world, "contact": contact,
        "area_norm": contact * 0.5, "cop_norm": rng.uniform(-0.5, 0.5, (n, 4, 2)),
        "phase": np.where(contact > 0.5, 3.0, 0.0).astype(np.float32),
        "episode": np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=np.int64),
        "force_norm": force_norm,
        "delta_force_norm": np.zeros_like(force_norm),
        "impact_peak": force_norm.copy(),
    }


class WindowLabelsTest(unittest.TestCase):
    def test_offsets_identity(self):
        cfg = EstimatorConfig(window=4)
        data = _toy_data(cfg)
        w = make_dataset_windows(data, cfg, 2)
        np.testing.assert_array_equal(w["contact_train"], w["contact"])
        np.testing.assert_array_equal(w["forcenorm_mean"], w["forcenorm"])

    def test_debounce_changes_train_labels_only(self):
        cfg = EstimatorConfig(window=4, debounce_contact=True)
        data = _toy_data(cfg)
        data["contact"] = data["contact"].copy()
        # 制造单帧抖动
        data["contact"][[3, 10]] = 1.0 - data["contact"][[3, 10]]
        w = make_dataset_windows(data, cfg, 2)
        self.assertFalse(np.array_equal(w["contact_train"], w["contact"]))
        # contact 仍是原始标签（评估口径）：窗口终点逐帧对齐
        ends = np.concatenate([np.flatnonzero(data["episode"] == e)[3::2]
                               for e in np.unique(data["episode"])])
        np.testing.assert_array_equal(w["contact"], data["contact"][ends])

    def test_mean_label_smoother(self):
        cfg = EstimatorConfig(window=4, bifreq_k=5, w_forcenorm=0.5)
        data = _toy_data(cfg)
        w = make_dataset_windows(data, cfg, 2)
        self.assertLess(float(np.std(w["forcenorm_mean"])),
                        float(np.std(w["forcenorm"])) + 1e-6)


class LossTargetTest(unittest.TestCase):
    def _out(self, b=2):
        m = TactileGRU(65, EstimatorConfig(window=4, bifreq_k=5, w_forcenorm=0.5))
        x = torch.zeros(b, 4, 65)
        return m(x), m

    def test_head_uses_mean_target(self):
        cfg = EstimatorConfig(window=4, bifreq_k=5, w_forcenorm=0.5)
        out, _m = self._out()
        b = out["force"].shape[0]
        contact = torch.ones(b, 4)
        raw = torch.zeros(b, 4)
        mean = torch.ones(b, 4)
        targets = {"force": torch.zeros(b, 4, 3), "contact": contact,
                   "contact_train": contact, "area": torch.zeros(b, 4),
                   "cop": torch.zeros(b, 4, 2), "phase": torch.full((b, 4), 3.0),
                   "forcenorm": raw, "forcenorm_mean": mean,
                   "delta": raw, "peak": raw}
        loss_a, terms_a = multi_task_loss(out, targets, cfg)
        targets2 = dict(targets)
        targets2["forcenorm_mean"] = raw
        loss_b, terms_b = multi_task_loss(out, targets2, cfg)
        self.assertNotAlmostEqual(float(terms_a["forcenorm"]),
                                  float(terms_b["forcenorm"]), places=5)

    def test_contact_train_overrides_mask(self):
        cfg = EstimatorConfig(window=4, w_contact=1.0, w_force=0.0,
                              w_area=0.0, w_cop=0.0)
        out, _m = self._out()
        b = out["force"].shape[0]
        ones = torch.ones(b, 4)
        zeros = torch.zeros(b, 4)
        base = {"force": torch.zeros(b, 4, 3), "contact": ones,
                "area": zeros, "cop": torch.zeros(b, 4, 2),
                "phase": torch.full((b, 4), 3.0), "forcenorm": zeros,
                "forcenorm_mean": zeros, "delta": zeros, "peak": zeros}
        t1 = dict(base, contact_train=ones)
        t2 = dict(base, contact_train=zeros)
        l1, _ = multi_task_loss(out, t1, cfg)
        l2, _ = multi_task_loss(out, t2, cfg)
        self.assertNotAlmostEqual(float(l1), float(l2), places=5)



class LabelClipTest(unittest.TestCase):
    def test_clip_bounds_targets(self):
        cfg = EstimatorConfig(window=4, label_clip_mg=2.0)
        data = _toy_data(cfg)
        data["force_norm"] = data["force_norm"].copy()
        data["force_norm"][5] = 100.0          # 仿真尖峰
        data["force_world"] = data["force_world"].copy()
        data["force_world"][5, :, 2] = 100.0 * 149.17
        w = make_dataset_windows(data, cfg, 2)
        self.assertLessEqual(float(w["forcenorm"].max()), 2.0 + 1e-6)
        ends = np.concatenate([np.flatnonzero(data["episode"] == e)[3::2]
                               for e in np.unique(data["episode"])])
        np.testing.assert_allclose(w["forcenorm"][ends == 5][0],
                                   np.minimum(data["force_norm"][5], 2.0))
        self.assertLessEqual(float(np.abs(w["force"]).max()), 2.0 * 149.17 + 1e-3)

    def test_zero_clip_is_identity(self):
        cfg = EstimatorConfig(window=4, label_clip_mg=0.0)
        data = _toy_data(cfg)
        w = make_dataset_windows(data, cfg, 2)
        ends = np.concatenate([np.flatnonzero(data["episode"] == e)[3::2]
                               for e in np.unique(data["episode"])])
        np.testing.assert_array_equal(w["force"], data["force_world"][ends])
        np.testing.assert_array_equal(w["forcenorm"], data["force_norm"][ends])

    def test_mean_uses_clipped_values(self):
        cfg = EstimatorConfig(window=4, label_clip_mg=1.0, bifreq_k=3,
                              w_forcenorm=0.5)
        data = _toy_data(cfg)
        data["force_norm"] = data["force_norm"].copy()
        data["force_norm"][10:13] = 50.0
        w = make_dataset_windows(data, cfg, 2)
        self.assertLessEqual(float(w["forcenorm_mean"].max()), 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
