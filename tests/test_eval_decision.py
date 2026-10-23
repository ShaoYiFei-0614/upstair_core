
# -*- coding: utf-8 -*-
"""Wave0 靶子（决策量 + 时移容差）的数学契约测试。"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ed = _load("eval_tactile_decision", "scripts/eval_tactile_decision.py")


class MathTest(unittest.TestCase):
    def test_r2_exact(self):
        self.assertAlmostEqual(ed._r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0, places=12)
        y = np.array([0.1, -0.2, 0.3, 0.0])
        self.assertAlmostEqual(ed._r2(np.full_like(y, y.mean()), y), 0.0, places=12)

    def test_load_bearing_f1(self):
        true = np.array([[0.6, 0.2]])
        pred = np.array([[0.6, 0.6]])
        contact = np.ones((1, 2))
        self.assertAlmostEqual(ed.load_bearing_f1(pred, true, contact), 2.0 / 3.0, places=12)

    def test_spearman_subset(self):
        contact = np.ones((1, 4)); phase = np.full((1, 4), 3.0)
        pred = np.array([[3.0, 1.0, 4.0, 2.0]])
        true = np.array([[3.0, 1.0, 4.0, 2.0]])
        rho, n = ed.spearman_frames(pred, true, contact, phase)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(rho, 1.0, places=9)
        # 只有 1 只足在接触 → 不统计
        c2 = np.zeros((1, 4)); c2[0, 0] = 1
        self.assertEqual(ed.spearman_frames(pred, true, c2, phase)[1], 0)


class ToleranceConventionTest(unittest.TestCase):
    def _series(self, n=60, seed=0):
        rng = np.random.RandomState(seed)
        base = np.cumsum(rng.randn(n))
        return base

    def test_leading_estimator_peaks_at_plus_one(self):
        """pred(t)=true(t+1)（估计超前 1 帧）⇒ 消费端用 t-1 帧的估计时最准 ⇒ curve[+1]≈1。"""
        n = 60
        true = self._series(n)
        pred = np.empty(n); pred[:-1] = true[1:]; pred[-1] = true[-1]
        ep = np.zeros(n, dtype=np.int64)
        contact = np.ones(n); phase = np.full(n, 3.0)
        curve = ed.tolerance(pred, true, contact, phase, ep)
        self.assertGreater(curve[1], 0.9)
        self.assertLess(curve[0], 0.9)
        self.assertGreater(curve[0], curve[-1])

    def test_perfect_estimator_peaks_at_zero(self):
        n = 60
        true = self._series(n, seed=1)
        ep = np.zeros(n, dtype=np.int64)
        contact = np.ones(n); phase = np.full(n, 3.0)
        curve = ed.tolerance(true.copy(), true, contact, phase, ep)
        self.assertAlmostEqual(curve[0], 1.0, places=9)
        self.assertLess(curve[1], 0.95)
        self.assertLess(curve[-1], 0.95)

    def test_no_cross_episode_pairing(self):
        """两个回合成一条直线：若不跨回合配对，滞后 1 帧只损失边缘点。"""
        a = np.linspace(0, 1, 30); b = np.linspace(0, 1, 30)
        true = np.concatenate([a, b])
        ep = np.array([0] * 30 + [1] * 30)
        contact = np.ones(60); phase = np.full(60, 3.0)
        curve = ed.tolerance(true.copy(), true, contact, phase, ep)
        self.assertAlmostEqual(curve[0], 1.0, places=9)

    def test_delta_star(self):
        curve = {0: 0.8, 1: 0.75, 2: 0.6, 3: 0.1, -1: 0.7, -2: 0.65, -3: 0.2}
        self.assertEqual(ed.delta_star(curve)[0], 1)
        self.assertEqual(ed.delta_star({0: -0.1})[0], 0)


class ContactMetricsTest(unittest.TestCase):
    def test_onset_recall_exact(self):
        true_c = np.zeros((10, 1)); true_c[4:, 0] = 1
        hit, n = ed.onset_recall(true_c, true_c, tol_frames=0)
        self.assertEqual((hit, n), (1.0, 1))
        miss, n2 = ed.onset_recall(np.zeros((10, 1)), true_c, tol_frames=2)
        self.assertEqual((miss, n2), (0.0, 1))

    def test_label_autocorr_and_agree(self):
        true = np.tile(np.array([0.2, 0.4, 0.6, 0.8]), (30, 1))
        ep = np.repeat([0, 1, 2], 10)
        contact = np.ones_like(true); phase = np.full_like(true, 3.0)
        ac = ed.label_autocorr(true, contact, phase, ep, lags=(1,))
        self.assertGreater(ac["1"], 0.0)
        c = np.zeros((20, 4)); c[::2] = 1
        self.assertAlmostEqual(ed.contact_prev_agree(c, np.zeros(20, dtype=np.int64)), 0.0, places=9)


class RequiredCasesTest(unittest.TestCase):
    def test_required_cases_match_calibrate_holdouts(self):
        cal = _load("calibrate_estimator", "scripts/calibrate_tactile_estimator.py")
        self.assertEqual(set(ed.REQUIRED_CASES), set(cal.HOLDOUT_NAMES))
        self.assertEqual(len(ed.REQUIRED_CASES), 7)


if __name__ == "__main__":
    unittest.main()
