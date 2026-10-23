"""GRU 足底触觉估计器的契约测试（窗口化 / 指标 / 存档）。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from robotdog.tactile_estimator import (
    ACTUATOR_NAMES, TORQUE_FEATURE_NAMES, TORQUE_REF_NM, EstimatorConfig,
    TactileEstimator, TactileGRU, apply_affine, contact_latency,
    contact_metrics, consumer_label_arrays, feature_stats, fit_affine,
    force_metrics, freeze_except_forcenorm, gather_windows,
    init_model_from_checkpoint, load_dataset, make_dataset_windows,
    merge_windows, multi_task_loss, per_foot_constant_baseline,
    physics_penalty, regression_metrics, relative_magnitude_penalty,
    save_dataset, slice_features, split_episodes, stance_force_metrics,
    torque_features, train_estimator, window_starts,
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

class ConsumerLabelTest(unittest.TestCase):
    """F̄ / ΔF̄ / 落地峰值标签：真标签优先，缺字段时按世界系力给代理。"""

    def _base(self):
        rng = np.random.default_rng(3)
        n = 12
        ep = np.concatenate([np.zeros(6, dtype=np.int32), np.ones(6, dtype=np.int32)])
        fw = rng.random((n, 4, 3)).astype(np.float32) * 100.0
        return {
            "features": rng.standard_normal((n, 4)).astype(np.float32),
            "force_world": fw,
            "force_foot": fw,
            "contact": np.ones((n, 4), dtype=np.uint8),
            "area_norm": np.full((n, 4), 0.2, dtype=np.float32),
            "cop_norm": np.zeros((n, 4, 2), dtype=np.float32),
            "phase": np.zeros((n, 4), dtype=np.int8),
            "episode": ep,
        }

    def test_proxy_when_sensor_labels_missing(self):
        cfg = EstimatorConfig(window=3, force_ref_n=100.0)
        data = self._base()
        fn, dl, pk, src = consumer_label_arrays(data, cfg)
        self.assertEqual(src, "proxy")
        np.testing.assert_allclose(
            fn, np.linalg.norm(data["force_world"], axis=2) / 100.0, rtol=1e-5)
        # 回合首步的 delta 等于当帧 F̄（tactile.py 里上一帧力记 0）
        self.assertAlmostEqual(float(dl[0, 0]), float(fn[0, 0]), places=6)
        self.assertAlmostEqual(float(dl[6, 0]), float(fn[6, 0]), places=6)
        np.testing.assert_allclose(dl[1], fn[1] - fn[0], rtol=1e-5)
        # 峰值是因果滚动最大值：不早于当前帧、不晚于 5 帧窗口
        self.assertGreaterEqual(float(pk[1, 0]), float(fn[1, 0]) - 1e-6)
        self.assertAlmostEqual(float(pk[0, 0]), float(fn[0, 0]), places=6)

    def test_sensor_labels_preferred(self):
        cfg = EstimatorConfig(window=3, force_ref_n=100.0)
        data = self._base()
        n = data["features"].shape[0]
        data["force_norm"] = np.full((n, 4), 0.8, dtype=np.float32)
        data["delta_force_norm"] = np.full((n, 4), 0.1, dtype=np.float32)
        data["impact_peak"] = np.full((n, 4), 1.2, dtype=np.float32)
        fn, dl, pk, src = consumer_label_arrays(data, cfg)
        self.assertEqual(src, "sensor")
        np.testing.assert_allclose(fn, 0.8)
        np.testing.assert_allclose(dl, 0.1)
        np.testing.assert_allclose(pk, 1.2)


class ConsumerLossTest(unittest.TestCase):
    """多任务损失：默认权重与历史配方等价，新增项按开关生效。"""

    def _batch(self, cfg, n=6):
        torch.manual_seed(0)
        model = TactileGRU(n_features=5, cfg=cfg)
        out = model(torch.randn(n, cfg.window, 5))
        tgt = {
            "force": torch.rand(n, 4, 3) * 0.5,
            "contact": (torch.rand(n, 4) > 0.5).float(),
            "area": torch.rand(n, 4) * 0.4,
            "cop": torch.rand(n, 4, 2) * 2 - 1,
        }
        return out, tgt

    def test_default_equals_legacy_formula(self):
        cfg = EstimatorConfig(window=4, hidden=8)
        out, tgt = self._batch(cfg)
        total, terms = multi_task_loss(out, tgt, cfg)
        bce = torch.nn.BCEWithLogitsLoss()
        mask = (tgt["contact"] > 0.5).float()
        denom = mask.sum().clamp(min=1.0)
        legacy = (cfg.w_force * torch.mean((out["force"].reshape(-1, 4, 3) - tgt["force"]) ** 2)
                  + cfg.w_contact * bce(out["contact"], tgt["contact"])
                  + cfg.w_area * (((out["area"] - tgt["area"]) ** 2) * mask).sum() / denom
                  + cfg.w_cop * (torch.mean((out["cop"].reshape(-1, 4, 2) - tgt["cop"]) ** 2, dim=2)
                                 * mask).sum() / denom)
        self.assertAlmostEqual(float(total.detach()), float(legacy.detach()), places=5)
        self.assertEqual(set(terms), {"force", "contact", "area", "cop"})

    def test_optional_terms_appear(self):
        cfg = EstimatorConfig(window=4, hidden=8, w_forcenorm=0.5, w_delta=0.2,
                              w_peak=0.2, w_magrel=0.1, physics_weight=0.1,
                              lookahead_contact=1, w_lookahead=0.3)
        out, tgt = self._batch(cfg)
        n = tgt["force"].shape[0]
        tgt.update({"forcenorm": torch.rand(n, 4), "delta": torch.rand(n, 4),
                    "peak": torch.rand(n, 4), "contact_next": torch.rand(n, 4)})
        total, terms = multi_task_loss(out, tgt, cfg)
        for key in ("forcenorm", "delta", "peak", "magrel", "physics", "lookahead"):
            self.assertIn(key, terms)
        self.assertTrue(np.isfinite(float(total)))

    def test_physics_penalty_uses_time_average(self):
        # 单帧可以偏离 1（实测 222.5 N vs mg=149.17 N），只要时间平均守住体重
        self.assertAlmostEqual(float(physics_penalty(torch.ones(5))), 0.0, places=6)
        z = torch.tensor([0.0, 2.0, 0.0, 2.0, 1.0])
        self.assertAlmostEqual(float(physics_penalty(z)), 0.0, places=6)
        # 整体幅值收缩（v1 的 0.80 均值比）必须被罚
        self.assertGreater(float(physics_penalty(torch.full((5,), 0.8))), 0.0)
        # mask 只用于筛样本，仍是平均口径
        self.assertAlmostEqual(float(physics_penalty(z, torch.ones(5))), 0.0, places=6)

    def test_relative_magnitude_penalty(self):
        mag = torch.full((4, 4), 0.6)
        mask = torch.ones(4, 4)
        self.assertAlmostEqual(float(relative_magnitude_penalty(mag, mag, mask)), 0.0,
                               places=6)
        worse = relative_magnitude_penalty(mag * 0.5, mag, mask)
        self.assertGreater(float(worse), 0.0)


class HeadGatingTest(unittest.TestCase):
    """消费端头默认不建：参数两与历史 checkpoint 逐键一致。"""

    def test_default_model_has_no_extra_heads(self):
        cfg = EstimatorConfig(window=4, hidden=8)
        model = TactileGRU(5, cfg)
        for name in ("forcenorm_head", "delta_head", "peak_head", "lookahead_head"):
            self.assertIsNone(getattr(model, name))
        out = model(torch.zeros(2, 4, 5))
        self.assertEqual(set(out), {"force", "contact", "area", "cop"})
        est = TactileEstimator(model, cfg, np.zeros(5, np.float32),
                               np.ones(5, np.float32))
        pred = est.predict(np.zeros((2, 4, 5), np.float32))
        self.assertEqual(set(pred), {"force", "contact", "area", "cop"})

    def test_enabled_heads_shapes(self):
        cfg = EstimatorConfig(window=4, hidden=8, w_forcenorm=1.0, w_delta=1.0,
                              w_peak=1.0, lookahead_contact=2, w_lookahead=1.0)
        model = TactileGRU(5, cfg)
        out = model(torch.zeros(3, 4, 5))
        self.assertEqual(tuple(out["forcenorm"].shape), (3, 4))
        self.assertEqual(tuple(out["delta"].shape), (3, 4))
        self.assertEqual(tuple(out["peak"].shape), (3, 4))
        self.assertEqual(tuple(out["contact_next"].shape), (3, 8))

    def test_lookahead_requires_weight(self):
        with self.assertRaises(ValueError):
            EstimatorConfig(window=4, lookahead_contact=2, w_lookahead=0.0)
        with self.assertRaises(ValueError):
            EstimatorConfig(window=4, lookahead_contact=5, w_lookahead=1.0)
        with self.assertRaises(ValueError):
            EstimatorConfig(window=4, w_forcenorm=-1.0)


class ConsumerMetricTest(unittest.TestCase):
    def test_stance_force_metrics_perfect(self):
        tgt = np.linspace(0.3, 0.9, 40).reshape(10, 4).astype(np.float32)
        contact = np.ones((10, 4), dtype=np.float32)
        m = stance_force_metrics(tgt, tgt, contact, force_ref_n=100.0)
        self.assertAlmostEqual(m["r2"], 1.0, places=6)
        self.assertAlmostEqual(m["slope"], 1.0, places=6)
        self.assertAlmostEqual(m["mean_ratio"], 1.0, places=6)
        self.assertAlmostEqual(m["p95_ratio"], 1.0, places=6)
        self.assertEqual(m["n"], 40.0)

    def test_stance_force_metrics_detects_shrinkage(self):
        tgt = np.linspace(0.2, 1.0, 40).reshape(10, 4).astype(np.float32)
        contact = np.ones((10, 4), dtype=np.float32)
        shrunk = (tgt * 0.7).astype(np.float32)     # 向均值收缩的典型失真
        m = stance_force_metrics(shrunk, tgt, contact, force_ref_n=100.0)
        self.assertAlmostEqual(m["slope"], 0.7, places=5)
        self.assertAlmostEqual(m["mean_ratio"], 0.7, places=5)
        self.assertLess(m["r2"], 1.0)

    def test_stance_force_metrics_empty(self):
        m = stance_force_metrics(np.zeros((3, 4)), np.zeros((3, 4)),
                                 np.zeros((3, 4)), force_ref_n=100.0)
        self.assertEqual(m["n"], 0.0)

    def test_constant_baseline_r2(self):
        fw = np.zeros((50, 4, 3), dtype=np.float32)
        fw[:, :, 2] = np.linspace(50, 150, 50)[:, None]
        base = per_foot_constant_baseline(fw)
        self.assertLess(base["r2_all"], 1.0)
        self.assertEqual(base["rmse_all_n"] > 0.0, True)

    def test_contact_latency_one_frame_delay(self):
        n = 10
        tgt = np.zeros((n, 4), dtype=np.float32)
        tgt[5, :] = 1.0
        prob = np.zeros((n, 4), dtype=np.float32)
        prob[6, :] = 0.9          # 比标签晚 1 帧
        out = contact_latency(prob, tgt, threshold=0.5)
        self.assertEqual(int(out["n_onsets"]), 4)
        self.assertAlmostEqual(out["onset_recall"], 0.0)
        self.assertAlmostEqual(out["prev_prob_mean"], 0.0)
        self.assertAlmostEqual(out["next_recall"], 1.0)

    def test_contact_latency_perfect(self):
        tgt = np.zeros((8, 4), dtype=np.float32)
        tgt[3:, :] = 1.0
        out = contact_latency(tgt, tgt, threshold=0.5)
        self.assertAlmostEqual(out["onset_recall"], 1.0)
        self.assertAlmostEqual(out["onset_prob_mean"], 1.0)


class WindowTargetsTest(unittest.TestCase):
    """窗口化产出的消费端目标：形状正确、前瞻接触越界取 0。"""

    def _data(self):
        rng = np.random.default_rng(5)
        n = 14
        ep = np.concatenate([np.zeros(8, dtype=np.int32), np.ones(6, dtype=np.int32)])
        ct = np.zeros((n, 4), dtype=np.uint8)
        ct[::3] = 1
        fw = rng.random((n, 4, 3)).astype(np.float32) * 80
        return {
            "features": rng.standard_normal((n, 3)).astype(np.float32),
            "force_world": fw, "force_foot": fw, "contact": ct,
            "area_norm": np.full((n, 4), 0.2, dtype=np.float32),
            "cop_norm": np.zeros((n, 4, 2), dtype=np.float32),
            "phase": np.ones((n, 4), dtype=np.int8),
            "episode": ep,
        }

    def test_targets_shapes_and_no_lookahead(self):
        cfg = EstimatorConfig(window=3)
        w = make_dataset_windows(self._data(), cfg, stride=2)
        for key in ("forcenorm", "delta", "peak", "phase"):
            self.assertIn(key, w)
        self.assertEqual(w["forcenorm"].shape, w["contact"].shape)
        self.assertEqual(tuple(w["contact_next"].shape), (w["x"].shape[0], 0))
        np.testing.assert_allclose(w["phase"], np.ones_like(w["phase"]))

    def test_lookahead_boundary_zero(self):
        cfg = EstimatorConfig(window=3, lookahead_contact=2, w_lookahead=1.0)
        data = self._data()
        w = make_dataset_windows(data, cfg, stride=2)
        self.assertEqual(tuple(w["contact_next"].shape), (w["x"].shape[0], 8))
        # 每个回合最后一个窗口的「后 1 帧」越界 ⇒ 对应 4 列全 0
        ep = w["episode"]
        last_win = {int(e): int(np.flatnonzero(ep == e)[-1]) for e in np.unique(ep)}
        for idx in last_win.values():
            np.testing.assert_array_equal(w["contact_next"][idx, :4], np.zeros(4))
        # 每个窗口的 2 组前瞻列必须逐帧等于「序列里后 k 帧」的真实接触；越界取 0
        ep_arr = data["episode"]
        offset = 0
        for e in np.unique(ep_arr):
            idx = np.flatnonzero(ep_arr == e)
            ends = idx[window_starts(idx.size, cfg.window, 2)]
            for j, end in enumerate(ends):
                for k in range(1, cfg.lookahead_contact + 1):
                    got = w["contact_next"][offset + j,
                                            4 * (k - 1):4 * k].astype(np.uint8)
                    # 不跨回合：回合末越界（含跨进下一回合）一律 0
                    want = (data["contact"][end + k] if end + k <= idx[-1]
                            else np.zeros(4, dtype=np.uint8))
                    np.testing.assert_array_equal(got, want)
            offset += ends.size
        self.assertEqual(offset, w["x"].shape[0])


class HeadCheckpointTest(unittest.TestCase):
    def test_roundtrip_preserves_enabled_heads(self):
        cfg = EstimatorConfig(window=4, hidden=8, w_forcenorm=1.0, w_peak=1.0,
                              lookahead_contact=1, w_lookahead=1.0)
        model = TactileGRU(5, cfg)
        est = TactileEstimator(model, cfg, np.zeros(5, np.float32),
                               np.ones(5, np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "est.pt"
            est.save(path, extra_meta={"tag": "v2"})
            back = TactileEstimator.load(path)
        self.assertEqual(back.cfg.lookahead_contact, 1)
        pred = back.predict(np.zeros((2, 4, 5), np.float32))
        for key in ("force", "contact", "area", "cop", "forcenorm", "peak",
                    "contact_next"):
            self.assertIn(key, pred)
        self.assertEqual(pred["contact_next"].shape, (2, 4))
        self.assertTrue(np.all((pred["contact_next"] >= 0) & (pred["contact_next"] <= 1)))


class MaskedConsumerLossTest(unittest.TestCase):
    """接触掩码：非接触帧的 F̄/Δ/峰值标签不进损失（均值收缩的直接防线）。"""

    def _targets(self, contact):
        b = contact.shape[0]
        z = np.zeros((b, 4), np.float32)
        fn = np.where(contact > 0.5, 0.0, 1e3).astype(np.float32)  # 非接触帧给垃圾值
        return {"force": torch.zeros(b, 4, 3), "contact": torch.from_numpy(contact),
                "area": torch.from_numpy(z), "cop": torch.from_numpy(
                    np.zeros((b, 4, 2), np.float32)),
                "forcenorm": torch.from_numpy(fn),
                "delta": torch.from_numpy(z), "peak": torch.from_numpy(z)}

    def _out(self, b):
        return {"force": torch.zeros(b, 4, 3),
                "contact": torch.zeros(b, 4),
                "area": torch.zeros(b, 4),
                "cop": torch.zeros(b, 4, 2),
                "forcenorm": torch.zeros(b, 4)}

    def test_forcenorm_loss_ignores_non_contact_frames(self):
        contact = np.zeros((6, 4), np.float32)
        contact[:, 0] = 1.0     # 只有第 0 足接触，其余足给 1e3 的垃圾标签
        cfg = EstimatorConfig(w_force=0.0, w_contact=0.0, w_area=0.0, w_cop=0.0,
                              w_forcenorm=1.0)
        targets = self._targets(contact)
        out = self._out(6)
        _, terms = multi_task_loss(out, targets, cfg)
        self.assertLess(float(terms["forcenorm"]), 1e-6)

    def test_fbar_rel_penalises_masked_shrinkage(self):
        contact = np.ones((6, 4), np.float32)
        cfg = EstimatorConfig(w_force=0.0, w_contact=0.0, w_area=0.0, w_cop=0.0,
                              w_fbar_rel=1.0)
        targets = self._targets(contact)
        out = self._out(6)
        out["forcenorm"] = torch.full((6, 4), 0.45)   # 预测只有真值 0.9 的一半
        total, terms = multi_task_loss(out, targets, cfg)
        self.assertGreater(float(terms["fbar_rel"]), 0.2)
        self.assertGreater(float(total), 0.2)


class CliDefaultsTest(unittest.TestCase):
    """CLI 默认值必须来自 EstimatorConfig（默认值单源）。"""

    def test_cli_defaults_match_config(self):
        import importlib.util
        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "train_tactile_estimator", root / "scripts" / "train_tactile_estimator.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        args = mod.build_parser().parse_args([])
        cfg = mod.config_from_args(args)
        self.assertEqual(cfg, EstimatorConfig())
        # v3 新参：默认等价于历史配方（全宽特征、不热启动、不冻结）
        self.assertEqual(args.features, 0)
        self.assertEqual(args.init_from, "")
        self.assertFalse(args.freeze_trunk)

    def test_eval_script_exposes_v3_flags(self):
        # eval 脚本的 argparse 在 main() 内联，这里退化为「参数名出现在源码里」+
        # 「默认值与契约一致」两条（可执行 --help 由 L0 门禁覆盖）。
        root = Path(__file__).resolve().parent.parent
        text = (root / "scripts" / "eval_tactile_estimator.py").read_text(
            encoding="utf-8")
        self.assertIn('"--features"', text)
        self.assertIn('"--calibration"', text)
        self.assertIn("default=0", text)
        self.assertIn('default=""', text)


class TorqueFeatureTest(unittest.TestCase):
    """12 维关节力矩特征：顺序、归一化、维数契约。"""

    def test_ref_and_names_layout(self):
        self.assertEqual(len(TORQUE_REF_NM), 12)
        self.assertEqual(len(ACTUATOR_NAMES), 12)
        self.assertEqual(len(TORQUE_FEATURE_NAMES), 12)
        # 腿部顺序必须与 go2 的 12 个位置执行器一致（FL/FR/RL/RR × hip/thigh/calf）
        self.assertEqual(ACTUATOR_NAMES[0], "FL_hip")
        self.assertEqual(ACTUATOR_NAMES[2], "FL_calf")
        self.assertEqual(ACTUATOR_NAMES[-1], "RR_calf")
        # 每腿 3 个：hip/thigh 23.7、calf 45.43（镜像 go2.xml 的 ctrlrange）
        for leg in range(4):
            self.assertEqual(TORQUE_REF_NM[3 * leg], 23.7)
            self.assertEqual(TORQUE_REF_NM[3 * leg + 1], 23.7)
            self.assertEqual(TORQUE_REF_NM[3 * leg + 2], 45.43)
        self.assertGreater(float(np.abs(np.asarray(TORQUE_REF_NM)).max()), 1e-6)  # 防 0/0

    def test_normalisation_hits_unit_scale(self):
        tau = np.asarray(TORQUE_REF_NM, np.float32).copy()
        feat = torque_features(tau)
        self.assertEqual(feat.shape, (12,))
        np.testing.assert_allclose(feat, np.ones(12), atol=1e-6)
        np.testing.assert_array_equal(np.abs(torque_features(-tau)),
                                      np.ones(12, np.float32))

    def test_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            torque_features(np.zeros(11, np.float32))
        with self.assertRaises(ValueError):
            torque_features(np.zeros((2, 12), np.float32))


class SliceFeaturesTest(unittest.TestCase):
    """--features 切片语义：0/全宽原样、53 列是 65 列的前缀、越界报错。"""

    def _windows(self):
        x = np.arange(2 * 3 * 65, dtype=np.float32).reshape(2, 3, 65)
        return {"x": x, "force": np.zeros((2, 4, 3), np.float32)}

    def test_zero_keeps_everything(self):
        w = self._windows()
        out = slice_features(w, 0)
        self.assertIs(out["x"], w["x"])

    def test_prefix_slice_is_view_and_matches(self):
        w = self._windows()
        out = slice_features(w, 53)
        self.assertEqual(out["x"].shape, (2, 3, 53))
        np.testing.assert_array_equal(out["x"], w["x"][..., :53])
        self.assertFalse(out["x"].flags["OWNDATA"])   # 视图，不复制
        self.assertEqual(out["force"].shape, (2, 4, 3))

    def test_too_wide_raises(self):
        with self.assertRaises(ValueError):
            slice_features(self._windows(), 66)


class InitFreezeTest(unittest.TestCase):
    """--init-from / --freeze-trunk 的契约（缺失键、维数不一致、只训 F̄ 头）。"""

    def _save(self, path, n_features, cfg=None):
        cfg = cfg or EstimatorConfig(window=4, hidden=8)
        model = TactileGRU(n_features, cfg)
        est = TactileEstimator(model, cfg, np.zeros(n_features, np.float32),
                               np.ones(n_features, np.float32))
        est.save(path, extra_meta={"tag": "v3"})
        return model

    def test_same_width_loads_without_missing(self):
        cfg = EstimatorConfig(window=4, hidden=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wide.pt"
            self._save(path, 65, cfg)
            model = TactileGRU(65, cfg)
            info = init_model_from_checkpoint(model, path)
        self.assertEqual((info["missing"], info["unexpected"]), ([], []))

    def test_width_mismatch_is_rejected(self):
        cfg = EstimatorConfig(window=4, hidden=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "narrow.pt"
            self._save(path, 53, cfg)
            with self.assertRaises(ValueError):
                init_model_from_checkpoint(TactileGRU(65, cfg), path)

    def test_missing_head_keys_are_the_only_missing(self):
        head_cfg = EstimatorConfig(window=4, hidden=8, w_forcenorm=0.5)
        base_cfg = EstimatorConfig(window=4, hidden=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nohead.pt"
            self._save(path, 65, base_cfg)
            model = TactileGRU(65, head_cfg)
            info = init_model_from_checkpoint(model, path)
        self.assertTrue(info["missing"])
        self.assertTrue(all(k.startswith("forcenorm_head") for k in info["missing"]))
        self.assertEqual(info["unexpected"], [])

    def test_freeze_keeps_only_forcenorm_head_trainable(self):
        cfg = EstimatorConfig(window=4, hidden=8, w_forcenorm=0.5)
        model = TactileGRU(65, cfg)
        trainable = freeze_except_forcenorm(model)
        self.assertTrue(trainable)
        self.assertTrue(all(n.startswith("forcenorm_head") for n in trainable))
        self.assertFalse(model.gru.weight_ih_l0.requires_grad)
        self.assertFalse(model.force_head.weight.requires_grad)
        self.assertTrue(model.forcenorm_head.weight.requires_grad)

    def test_freeze_without_head_is_a_config_error(self):
        model = TactileGRU(65, EstimatorConfig(window=4, hidden=8))
        self.assertEqual(freeze_except_forcenorm(model), [])

    def test_train_estimator_requires_head_for_freeze(self):
        cfg = EstimatorConfig(window=3, hidden=8, epochs=1, batch=2)
        n, f = 4, 65
        data = {
            "x": np.zeros((n, cfg.window, f), np.float32),
            "force": np.zeros((n, 4, 3), np.float32),
            "contact": np.ones((n, 4), np.float32),
            "area": np.zeros((n, 4), np.float32),
            "cop": np.zeros((n, 4, 2), np.float32),
        }
        with self.assertRaises(ValueError):
            train_estimator(data, None, cfg, freeze_trunk=True, verbose=False)


class CalibrationMathTest(unittest.TestCase):
    """后验仿射 y' = a*y + b：闭式可复原、常数退化安全、长度契约。"""

    def test_apply_affine(self):
        y = np.array([0.0, 1.0, 2.0], np.float32)
        np.testing.assert_allclose(apply_affine(y, 2.0, 0.5), [0.5, 2.5, 4.5])

    def test_fit_affine_recovers_known_line(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0.6, 0.1, 4000)
        t = 1.4 * y - 0.2
        fit = fit_affine(y, t)
        self.assertAlmostEqual(fit["a"], 1.4, places=6)
        self.assertAlmostEqual(fit["b"], -0.2, places=6)
        self.assertEqual(fit["n"], 4000)
        self.assertLess(fit["rmse_after"], fit["rmse_before"])

    def test_fit_affine_constant_prediction_falls_back(self):
        y = np.full(10, 0.3)
        t = np.linspace(0.1, 0.5, 10)
        fit = fit_affine(y, t)
        self.assertEqual(fit["a"], 1.0)
        self.assertAlmostEqual(fit["b"], float(t.mean() - 0.3), places=6)

    def test_fit_affine_size_mismatch(self):
        with self.assertRaises(ValueError):
            fit_affine(np.zeros(3), np.zeros(4))

    def test_identity_calibration_is_lossless(self):
        y = np.array([1.2, -0.4, 0.9], np.float32)
        np.testing.assert_array_equal(apply_affine(y, 1.0, 0.0), y)



class AffineDirectionTest(unittest.TestCase):
    """两种标定方向（评审 B2）：ls 收缩、inverse 修正衰减。"""

    def _shrink(self, s=0.7, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        t = rng.normal(0.6, 0.1, n)
        y = s * t + rng.normal(0.0, 0.02, n)
        return y, t

    def test_ls_slope_after_equals_r2_and_shrinks(self):
        y, t = self._shrink()
        fit = fit_affine(y, t, method="ls")
        self.assertLess(fit["slope_after"], 1.0)
        self.assertLess(fit["rmse_after"], fit["rmse_before"])
        # 有噪声时 ls 的 a=cov(y,t)/var(y) 略小于 1/s（收缩未被完全补偿）
        self.assertLess(fit["a"], 1.0 / 0.7)
        self.assertGreater(fit["a"], 1.0)

    def test_inverse_slope_after_is_one(self):
        y, t = self._shrink()
        fit = fit_affine(y, t, method="inverse")
        self.assertAlmostEqual(fit["slope_after"], 1.0, places=6)
        # inverse 的目标就是把斜率拉回 1，RMSE 允许比 ls 差
        self.assertGreater(fit["rmse_after"], 0.0)

    def test_invalid_method_rejected(self):
        with self.assertRaises(ValueError):
            fit_affine(np.zeros(3), np.zeros(3), method="ols")


class StanceMaskPhaseTest(unittest.TestCase):
    """消费端掩码 = contact & phase>2.5（评审 B3/R1）。"""

    def test_phase_excludes_landing_frames(self):
        n = 6
        tgt = np.full((n, 4), 0.5, np.float32)
        tgt[1] = 2.0                     # 落地冲击帧，幅值极高
        contact = np.ones((n, 4), np.float32)
        phase = np.full((n, 4), 3.0, np.float32)
        phase[1] = 2.0                   # LANDING
        pred = tgt.copy()
        m_all = stance_force_metrics(pred, tgt, contact, force_ref_n=100.0)
        m_st = stance_force_metrics(pred, tgt, contact, force_ref_n=100.0, phase=phase)
        self.assertEqual(int(m_all["n"]), n * 4)
        self.assertEqual(int(m_st["n"]), (n - 1) * 4)
        self.assertLess(float(m_all["mean_target_n"]), float(m_st["mean_target_n"]) * 1.0 + 1e9)

    def test_phase_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            stance_force_metrics(np.zeros((3, 4)), np.zeros((3, 4)),
                                 np.zeros((3, 4)), force_ref_n=100.0,
                                 phase=np.zeros((3, 3)))


def _load_script(name: str, filename: str):
    import importlib.util
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(name, root / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CalibrateGuardTest(unittest.TestCase):
    """标定/评估的守卫：留出地形识别（含策略后缀）、标定文件与权重匹配。"""

    def test_holdout_prefix_match(self):
        cal = _load_script("calibrate_tactile_estimator",
                           "calibrate_tactile_estimator.py")
        self.assertTrue(cal.is_holdout_name("stair18"))
        self.assertTrue(cal.is_holdout_name("rough_s1_flatpol"))
        self.assertTrue(cal.is_holdout_name("stair16_5_x"))
        self.assertFalse(cal.is_holdout_name("flat"))
        self.assertFalse(cal.is_holdout_name("rough_s0_stairpol"))
        self.assertFalse(cal.is_holdout_name("stair15"))

    def test_eval_rejects_mismatched_calibration(self):
        ev = _load_script("eval_tactile_estimator", "eval_tactile_estimator.py")
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "w.pt"
            cfg = EstimatorConfig(window=4, hidden=8)
            est = TactileEstimator(TactileGRU(5, cfg), cfg, np.zeros(5, np.float32),
                                   np.ones(5, np.float32))
            est.save(ckpt)
            good = {"ckpt_sha256": ev.sha256_file(ckpt), "features": 5, "stride": 2}
            ev.check_calibration(dict(good), ckpt, 5, 2)      # 一致 ⇒ 通过
            for bad in ({"ckpt_sha256": "0" * 64}, {"features": 53}, {"stride": 3}):
                merged = dict(good)
                merged.update(bad)
                with self.assertRaises(SystemExit):
                    ev.check_calibration(dict(merged), ckpt, 5, 2)
            # v5 provenance：ckpt 带 data_sha256 时，标定必须绑定同批数据
            sha = {"train_a.npz": "a" * 64}
            est.save(ckpt, extra_meta={"data_sha256": sha})
            good5 = dict(good, ckpt_sha256=ev.sha256_file(ckpt),
                         case_sha256=dict(sha))
            ev.check_calibration(dict(good5), ckpt, 5, 2)     # 绑定一致 ⇒ 通过
            with self.assertRaises(SystemExit):               # 缺 case_sha256
                ev.check_calibration({k: v for k, v in good5.items()
                                      if k != "case_sha256"}, ckpt, 5, 2)
            with self.assertRaises(SystemExit):               # sha 不一致
                ev.check_calibration(dict(good5, case_sha256={"train_a.npz": "b" * 64}),
                                     ckpt, 5, 2)


class StanceWeightingTest(unittest.TestCase):
    """v4 支撑相加权：w_stance=0 与历史逐位等价；>0 时支撑相帧代价更高。"""

    @staticmethod
    def _out(win_force):
        return {"force": torch.tensor(win_force, dtype=torch.float32),
                "contact": torch.zeros(2, 4),
                "area": torch.zeros(2, 4),
                "cop": torch.zeros(2, 8)}

    @staticmethod
    def _targets(phase, contact=1.0):
        return {"force": torch.zeros(2, 4, 3),
                "contact": torch.full((2, 4), float(contact)),
                "area": torch.zeros(2, 4),
                "cop": torch.zeros(2, 4, 2),
                "phase": torch.tensor(phase, dtype=torch.float32)}

    def test_zero_weight_matches_legacy(self):
        # w_stance=0 ⇒ phase 完全不参与，与「不传 phase」逐位同值
        f = np.zeros((2, 12), np.float32)
        f[0] = 0.1
        t = self._targets([[3.0] * 4, [0.0] * 4])
        t_no = {k: v for k, v in t.items() if k != "phase"}
        l0, _ = multi_task_loss(self._out(f), t, EstimatorConfig())
        l1, _ = multi_task_loss(self._out(f), t_no, EstimatorConfig())
        self.assertAlmostEqual(float(l0), float(l1), places=7)

    def test_stance_error_costs_more(self):
        cfg = EstimatorConfig(w_stance=3.0)
        f_st = np.zeros((2, 12), np.float32)
        f_st[0] = 0.1                      # 误差全在支撑相窗口
        f_sw = np.zeros((2, 12), np.float32)
        f_sw[1] = 0.1                      # 误差全在摆动相窗口
        t = self._targets([[3.0] * 4, [0.0] * 4])
        l_st, _ = multi_task_loss(self._out(f_st), t, cfg)
        l_sw, _ = multi_task_loss(self._out(f_sw), t, cfg)
        self.assertGreater(float(l_st), float(l_sw))

    def test_stance_weight_applies_to_consumer_heads(self):
        cfg3 = EstimatorConfig(w_forcenorm=1.0, w_stance=3.0)
        cfg0 = EstimatorConfig(w_forcenorm=1.0)
        out = self._out(np.zeros((2, 12), np.float32))
        out["forcenorm"] = torch.zeros(2, 4)
        t = self._targets([[3.0] * 4, [0.0] * 4])
        t["forcenorm"] = torch.zeros(2, 4)
        t["forcenorm"][0] = 0.2            # 误差全在支撑相 ⇒ 加权后更大
        l3, _ = multi_task_loss(out, t, cfg3)
        l0, _ = multi_task_loss(out, t, cfg0)
        self.assertGreater(float(l3), float(l0))


class HeadCapacityTest(unittest.TestCase):
    """v4：head_hidden>0 时 F_bar 头是 MLP，参数名仍是 forcenorm_head.*。"""

    def test_head_hidden_builds_mlp(self):
        m = TactileGRU(5, EstimatorConfig(w_forcenorm=0.5, head_hidden=32))
        keys = [k for k in m.state_dict() if k.startswith("forcenorm_head")]
        self.assertEqual(sorted(keys), ["forcenorm_head.0.bias",
                                        "forcenorm_head.0.weight",
                                        "forcenorm_head.2.bias",
                                        "forcenorm_head.2.weight"])
        out = m(torch.zeros(2, 3, 5))
        self.assertEqual(tuple(out["forcenorm"].shape), (2, 4))

    def test_head_hidden_zero_keeps_linear(self):
        m = TactileGRU(5, EstimatorConfig(w_forcenorm=0.5))
        self.assertEqual([k for k in m.state_dict()
                          if k.startswith("forcenorm_head")],
                         ["forcenorm_head.weight", "forcenorm_head.bias"])

    def test_head_hidden_validation(self):
        with self.assertRaises(ValueError):
            EstimatorConfig(head_hidden=-1)
        with self.assertRaises(ValueError):
            EstimatorConfig(w_stance=-0.1)


class AffineClampTest(unittest.TestCase):
    """v4：a 限幅只改斜率，b 重新取均值；a_raw 保留供审计。"""

    def test_inverse_clamped(self):
        rng = np.random.default_rng(0)
        t = rng.normal(0.8, 0.2, 500)
        y = 0.4 * t + 0.3 + rng.normal(0, 0.02, 500)    # 斜率 0.4 ⇒ inverse a≈2.5
        raw = fit_affine(y, t, method="inverse")
        self.assertGreater(raw["a_raw"], 2.0)
        cl = fit_affine(y, t, method="inverse", a_max=1.5)
        self.assertAlmostEqual(cl["a"], 1.5, places=9)
        self.assertEqual(cl["a_raw"], raw["a_raw"])
        self.assertTrue(cl["clipped"])
        self.assertAlmostEqual(cl["b"], float(t.mean()) - 1.5 * float(y.mean()),
                               places=9)
        self.assertAlmostEqual(cl["mean_y"], float(y.mean()), places=9)
        self.assertAlmostEqual(cl["mean_t"], float(t.mean()), places=9)

    def test_ls_min_clamp(self):
        rng = np.random.default_rng(1)
        t = rng.normal(0.8, 0.2, 300)
        # ls 拟合的是 t≈a*y：y 对 t 的斜率 3 ⇒ a≈1/3 < 1
        y = 3.0 * t + 0.1 + rng.normal(0, 0.01, 300)
        cl = fit_affine(y, t, method="ls", a_min=1.0)
        self.assertGreaterEqual(cl["a"], 1.0)
        self.assertTrue(cl["clipped"])


class V4CliAndCasesTest(unittest.TestCase):
    """v4 CLI 单源 + 采集 CASE 不混入留出地形。"""

    HOLDOUT = ("stair16_5", "stair18", "rough_s1", "rough_s2")

    def test_train_cli_flags(self):
        mod = _load_script("train_tactile_estimator", "train_tactile_estimator.py")
        cfg = mod.config_from_args(
            mod.build_parser().parse_args(["--w-stance", "3.0",
                                           "--head-hidden", "64"]))
        self.assertEqual(cfg.w_stance, 3.0)
        self.assertEqual(cfg.head_hidden, 64)
        d = mod.config_from_args(mod.build_parser().parse_args([]))
        self.assertEqual((d.w_stance, d.head_hidden), (0.0, 0))

    def test_extra_cases_never_holdout(self):
        mod = _load_script("collect_tactile_data", "collect_tactile_data.py")
        self.assertTrue(mod.EXTRA_CASES)
        for name, _spec, _pol, _vx, _eps, split in mod.EXTRA_CASES:
            self.assertEqual(split, "train", name)
            self.assertFalse(any(name == h or name.startswith(h + "_")
                                 for h in self.HOLDOUT), name)
        for name, _spec, _pol, _vx, _eps, split in mod.CASES:
            if split == "train":
                self.assertFalse(any(name == h or name.startswith(h + "_")
                                     for h in self.HOLDOUT), name)

    def test_eval_clamp_grid(self):
        ev = _load_script("eval_tactile_estimator", "eval_tactile_estimator.py")
        self.assertIn(1.0, ev.CLAMP_GRID)
        self.assertEqual(tuple(sorted(ev.CLAMP_GRID)), ev.CLAMP_GRID)

    def test_calibrate_clamp_flags(self):
        """--a-max 的语义：把 a 限到上限后 b 重新取 mean_t - a*mean_y。

        这条测的是 fit_affine 的限幅路径（calibrate/eval 都只经由它实现
        --a-max / S4 扫描），不是 CLI 解析；旧版断言 max(1.0, 1.5) 恒真，
        等于没测（fresh 评审 n4）。
        """
        rng = np.random.default_rng(0)
        t = rng.uniform(0.2, 1.2, 2000)          # 真值
        y = 0.6 * t + 0.1 + rng.normal(0, 0.01, 2000)   # 收缩型预测

        inv = fit_affine(y, t, method="inverse", a_max=0.8)
        self.assertTrue(inv["clipped"])
        self.assertAlmostEqual(inv["a"], 0.8, places=12)
        self.assertGreater(inv["a_raw"], 1.5)     # 未限幅的 inverse 约 1/0.6
        self.assertAlmostEqual(inv["b"], float(t.mean()) - 0.8 * float(y.mean()),
                               places=10)

        # ls 拟合的是 t ≈ a*y + b，对收缩预测 a ≈ cov(y,t)/var(y) ≈ 1/0.6
        ls = fit_affine(y, t, method="ls")
        self.assertFalse(ls["clipped"])
        self.assertAlmostEqual(ls["a"], 1.0 / 0.6, delta=0.02)
        self.assertAlmostEqual(ls["b"],
                               float(t.mean()) - ls["a"] * float(y.mean()), places=9)

        # 限幅对 a 的作用与方向无关：ls 给上限时同样被夹住并重算 b
        ls_clip = fit_affine(y, t, method="ls", a_max=1.5)
        self.assertTrue(ls_clip["clipped"])
        self.assertAlmostEqual(ls_clip["a"], 1.5, places=12)
        self.assertAlmostEqual(ls_clip["b"],
                               float(t.mean()) - 1.5 * float(y.mean()), places=10)

        # 下限同理
        inv_lo = fit_affine(y, t, method="inverse", a_min=2.0)
        self.assertTrue(inv_lo["clipped"])
        self.assertAlmostEqual(inv_lo["a"], 2.0, places=12)

        mod = _load_script("calibrate_tactile_estimator",
                           "calibrate_tactile_estimator.py")
        self.assertTrue(callable(mod.main))


if __name__ == "__main__":
    unittest.main()

