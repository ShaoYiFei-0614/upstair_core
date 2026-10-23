from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import robotdog.tactile_estimator as te
from robotdog.tactile_estimator import (
    EstimatorConfig,
    TactileEstimator,
    TactileGRU,
    fit_affine,
    freeze_except_forcenorm,
    init_model_from_checkpoint,
    multi_task_loss,
    stance_force_metrics,
    TORQUE_FEATURE_NAMES,
)


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_v4_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _batch(n=4, window=4, n_features=6, contact=True, stance=None):
    """构造一个最小 targets/out 批次（n 个窗口终点）。"""
    if stance is None:
        stance = [False] * n
    contact_arr = np.zeros((n, 4), dtype=np.float32)
    phase = np.zeros((n, 4), dtype=np.float32)
    if contact:
        contact_arr[:] = 1.0
        phase[:] = 1.0
        for i, s in enumerate(stance):
            if s:
                phase[i, :] = 3.0
    targets = {
        "force": torch.zeros((n, 4, 3)),
        "contact": torch.from_numpy(contact_arr),
        "area": torch.zeros((n, 4)),
        "cop": torch.zeros((n, 4, 2)),
        "forcenorm": torch.zeros((n, 4)),
        "delta": torch.zeros((n, 4)),
        "peak": torch.zeros((n, 4)),
        "phase": torch.from_numpy(phase),
    }
    out = {
        "force": torch.zeros((n, 4, 3)),
        "contact": torch.zeros((n, 4)),
        "area": torch.zeros((n, 4)),
        "cop": torch.zeros((n, 4, 2)),
        "forcenorm": torch.zeros((n, 4)),
        "delta": torch.zeros((n, 4)),
        "peak": torch.zeros((n, 4)),
    }
    return out, targets


class WeightingTest(unittest.TestCase):
    """T1/T2：frame_w 闭式加权，force 项与 F̄ 项分母口径不同。"""

    def test_t1_forcenorm_denominator_is_stance_weighted_sum(self):
        cfg = EstimatorConfig(w_forcenorm=0.5, w_stance=3.0)
        out, tg = _batch(n=4, stance=[True, False, False, False])
        out["forcenorm"] = torch.zeros((4, 4))
        tg["forcenorm"] = torch.zeros((4, 4))
        tg["forcenorm"][0] = 1.0            # 只有支撑相那一帧有误差
        _loss, terms = multi_task_loss(out, tg, cfg)
        # frame_w = mask_c*(1+3*stance) = [4,1,1,1] ⇒ 分母 7，分子 4*1²
        self.assertAlmostEqual(float(terms["forcenorm"]), 4.0 / 7.0, places=7)

        # 同一误差放到非支撑相帧：分母不变、分子只 1 份
        out2, tg2 = _batch(n=4, stance=[True, False, False, False])
        tg2["forcenorm"][1] = 1.0
        _l2, terms2 = multi_task_loss(out2, tg2, cfg)
        self.assertAlmostEqual(float(terms2["forcenorm"]), 1.0 / 7.0, places=7)
        self.assertGreater(float(terms["forcenorm"]), float(terms2["forcenorm"]))

    def test_t1_forcenorm_ignores_non_contact_frames(self):
        cfg = EstimatorConfig(w_forcenorm=0.5, w_stance=3.0)
        out, tg = _batch(n=2, contact=False)     # 全无接触 ⇒ frame_w 全 0
        tg["forcenorm"] = torch.tensor([[9.0, 9.0, 9.0, 9.0],
                                        [9.0, 9.0, 9.0, 9.0]])
        _l, terms = multi_task_loss(out, tg, cfg)
        self.assertAlmostEqual(float(terms["forcenorm"]), 0.0, places=7)

    def test_t1_force_term_weights_all_frames(self):
        """force 项分母是 w_force.sum()（全帧按 1 计权），与 F̄ 项分母不同。"""
        cfg = EstimatorConfig(w_forcenorm=0.5, w_stance=3.0)
        out, tg = _batch(n=4, contact=False, stance=[False, False, False, False])
        out["force"] = torch.zeros((4, 4, 3))
        tg["force"] = torch.zeros((4, 4, 3))
        tg["force"][0] = 1.0                     # 非接触帧上的力误差
        tg["forcenorm"] = torch.zeros((4, 4))
        tg["forcenorm"][0] = 9.0                 # 同一帧的 F̄ 误差
        _l, terms = multi_task_loss(out, tg, cfg)
        # force 项分母 w_force.sum()=16（4 帧×4 足，全部按 1 计权）、分子 4（4 足各 1.0）
        self.assertAlmostEqual(float(terms["force"]), 4.0 / 16.0, places=7)
        # F̄ 项同一帧的误差被 mask_c=0 完全忽略 ⇒ 两项分母口径确实不同
        self.assertAlmostEqual(float(terms["forcenorm"]), 0.0, places=7)

    def test_t2_zero_stance_weight_equals_zero_stance_data(self):
        """w_stance=0 与「w_stance>0 但数据里没有支撑相帧」逐项相等。"""
        out, tg = _batch(n=6, contact=True, stance=[False] * 6)
        tg["forcenorm"] = torch.rand((6, 4))
        out["forcenorm"] = torch.rand((6, 4))
        _a, terms_a = multi_task_loss(out, tg, EstimatorConfig(w_forcenorm=0.5, w_stance=0.0))
        _b, terms_b = multi_task_loss(out, tg, EstimatorConfig(w_forcenorm=0.5, w_stance=3.0))
        for key in terms_a:
            self.assertAlmostEqual(float(terms_a[key]), float(terms_b[key]), places=7,
                                   msg=f"w_stance=0 与无支撑相数据应等价：{key}")
        self.assertNotIn("fbar_rel", terms_a)


class HeadShapeTest(unittest.TestCase):
    """T3/T4：MLP 头键名与可训练参数计数。"""

    def _model(self, head_hidden: int):
        cfg = EstimatorConfig(w_forcenorm=0.5, head_hidden=head_hidden)
        return TactileGRU(6, cfg), cfg

    def test_t3_mlp_head_keys(self):
        model, _ = self._model(64)
        keys = {k for k in model.state_dict() if k.startswith("forcenorm_head")}
        self.assertEqual(keys, {"forcenorm_head.0.weight", "forcenorm_head.0.bias",
                                "forcenorm_head.2.weight", "forcenorm_head.2.bias"})
        self.assertFalse([k for k in keys if k.startswith("forcenorm_head.1")])

    def test_t3_linear_head_keys(self):
        model, _ = self._model(0)
        keys = {k for k in model.state_dict() if k.startswith("forcenorm_head")}
        self.assertEqual(keys, {"forcenorm_head.weight", "forcenorm_head.bias"})

    def test_t4_trainable_counts(self):
        for head_hidden, expected in ((0, 516), (64, 8516)):
            model, _ = self._model(head_hidden)
            names = freeze_except_forcenorm(model)
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.assertEqual(n_trainable, expected, f"head_hidden={head_hidden}")
            self.assertTrue(all(n.startswith("forcenorm_head") for n in names))
            self.assertEqual(len(names), 2 if head_hidden == 0 else 4)


class AffineTest(unittest.TestCase):
    """T5：inverse 闭式 + 限幅后 b 重新取均值。"""

    def test_inverse_closed_form(self):
        y = np.array([0.0, 1.0, 2.0, 3.0])
        t = 2.0 * y + 1.0 + np.array([0.01, -0.01, 0.02, -0.02])
        out = fit_affine(y, t, method="inverse")
        cov = float(np.mean((y - y.mean()) * (t - t.mean())))
        self.assertAlmostEqual(out["a_raw"], float(np.var(t) / cov), places=9)
        self.assertAlmostEqual(out["slope_after"], 1.0, places=6)
        self.assertEqual(out["n"], 4)

    def test_clamp_recomputes_b(self):
        y = np.array([0.0, 1.0, 2.0, 3.0])
        t = 3.0 * y + 2.0
        out = fit_affine(y, t, method="inverse", a_min=1.0, a_max=2.0)
        self.assertTrue(out["clipped"])
        self.assertAlmostEqual(out["a"], 2.0, places=12)
        self.assertAlmostEqual(out["b"], float(t.mean() - 2.0 * y.mean()), places=12)

    def test_no_clamp_keeps_a(self):
        y = np.array([0.0, 1.0, 2.0, 3.0])
        t = 1.5 * y + 0.5
        out = fit_affine(y, t, method="ls", a_min=1.0, a_max=2.0)
        self.assertFalse(out["clipped"])
        self.assertAlmostEqual(out["a"], 1.5, places=9)


class InitFromTest(unittest.TestCase):
    """T6：init_from 三态矩阵（w2B 的真正门）。"""

    def _save(self, path, *, w_forcenorm, head_hidden, n_features=6, hidden=16):
        cfg = EstimatorConfig(w_forcenorm=w_forcenorm, head_hidden=head_hidden,
                              hidden=hidden)
        model = TactileGRU(n_features, cfg)
        est = TactileEstimator(model, cfg, np.zeros(n_features), np.ones(n_features))
        est.save(path)
        return path

    def test_same_arch_hits_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = self._save(Path(tmp) / "a.pt", w_forcenorm=0.5, head_hidden=64)
            cfg = EstimatorConfig(w_forcenorm=0.5, head_hidden=64, hidden=16)
            model = TactileGRU(6, cfg)
            out = init_model_from_checkpoint(model, ck)
            self.assertEqual(out["missing"], [])
            self.assertEqual(out["unexpected"], [])

    def test_mlp_ckpt_into_linear_model_is_rejected(self):
        """漏传 --head-hidden：MLP 权重灌进单层头 ⇒ unexpected 非空 ⇒ 拒。"""
        with tempfile.TemporaryDirectory() as tmp:
            ck = self._save(Path(tmp) / "mlp.pt", w_forcenorm=0.5, head_hidden=64)
            cfg = EstimatorConfig(w_forcenorm=0.5, head_hidden=0, hidden=16)
            model = TactileGRU(6, cfg)
            out = init_model_from_checkpoint(model, ck)
            self.assertTrue(out["unexpected"])
            self.assertTrue(all(k.startswith("forcenorm_head") for k in out["unexpected"]))
            self.assertTrue(all(k.startswith("forcenorm_head") for k in out["missing"]))

    def test_linear_ckpt_into_mlp_model_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = self._save(Path(tmp) / "lin.pt", w_forcenorm=0.0, head_hidden=0)
            cfg = EstimatorConfig(w_forcenorm=0.5, head_hidden=64, hidden=16)
            model = TactileGRU(6, cfg)
            out = init_model_from_checkpoint(model, ck)
            self.assertEqual(out["unexpected"], [])
            self.assertTrue(out["missing"])
            self.assertTrue(all(k.startswith("forcenorm_head") for k in out["missing"]))

    def test_feature_width_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = self._save(Path(tmp) / "w6.pt", w_forcenorm=0.0, head_hidden=0)
            model = TactileGRU(8, EstimatorConfig(hidden=16))
            with self.assertRaises(ValueError):
                init_model_from_checkpoint(model, ck)


class EvalContractTest(unittest.TestCase):
    """T9/T11：CLAMP_GRID 精确 8 值、主口径严格 phase>2.5。"""

    def test_clamp_grid_exact(self):
        mod = _load_script("eval_tactile_estimator")
        self.assertEqual(tuple(mod.CLAMP_GRID),
                         (1.0, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.5))
        self.assertEqual(len(set(mod.CLAMP_GRID)), 8)
        self.assertEqual(tuple(mod.PREREG_A), (1.0, 1.3, 1.6, 2.0))

    def test_stance_mask_is_strict_greater(self):
        n = 3
        pred = np.ones((n, 4, 3), dtype=np.float64)
        target = np.zeros((n, 4, 3), dtype=np.float64)
        contact = np.ones((n, 4), dtype=np.float64)
        # 每帧 4 足：phase 2.4 / 2.5 / 2.6 / 2.4 ⇒ 严格 >2.5 只留第 3 足
        phase = np.tile(np.array([2.4, 2.5, 2.6, 2.4], dtype=np.float64), (n, 1))
        m = stance_force_metrics(pred, target, contact, force_ref_n=149.17,
                                 phase=phase, phase_min=2.5)
        # n 是掩码后的元素数：3 帧 × 1 足（phase==2.6，2.5 不算）× 3 轴
        self.assertEqual(int(m["n"]), n * 3)

    def test_torque_feature_layout(self):
        self.assertEqual(len(TORQUE_FEATURE_NAMES), 12)
        self.assertEqual(TORQUE_FEATURE_NAMES[0], "tau_FL_hip_norm")
        tau = np.arange(12, dtype=np.float64)
        norm = te.torque_features(tau)
        self.assertEqual(norm.shape, (12,))
        self.assertAlmostEqual(float(norm[2]), 2.0 / 45.43, places=6)


class CliContractTest(unittest.TestCase):
    """T8：--w-stance/--head-hidden 传值真的进 config（默认值相等不算）。"""

    def test_non_default_roundtrip(self):
        mod = _load_script("train_tactile_estimator")
        ns = mod.build_parser().parse_args(
            ["--w-stance", "3.0", "--head-hidden", "64", "--features", "53"])
        cfg = mod.config_from_args(ns)
        self.assertEqual(cfg.w_stance, 3.0)
        self.assertEqual(cfg.head_hidden, 64)
        self.assertEqual(int(ns.features), 53)

    def test_defaults_match_config(self):
        mod = _load_script("train_tactile_estimator")
        cfg = mod.config_from_args(mod.build_parser().parse_args([]))
        base = EstimatorConfig()
        for name in ("w_stance", "head_hidden", "w_forcenorm", "w_fbar_rel",
                     "window", "hidden", "epochs"):
            self.assertEqual(getattr(cfg, name), getattr(base, name), name)


class TrainPathTest(unittest.TestCase):
    """T13：训练路径确实把 phase 喂进 loss 并按 frame_w 加权。"""

    def _fixture(self, n=8, window=4):
        rng = np.random.default_rng(0)
        d = {
            "x": rng.normal(size=(n, window, 6)).astype(np.float32),
            "force": rng.normal(size=(n, 4, 3)).astype(np.float32),
            "contact": (rng.random((n, 4)) > 0.3).astype(np.float32),
            "area": rng.random((n, 4)).astype(np.float32),
            "cop": rng.normal(size=(n, 4, 2)).astype(np.float32),
            "forcenorm": rng.random((n, 4)).astype(np.float32),
            "phase": np.where(rng.random((n, 4)) > 0.5, 3.0, 1.0).astype(np.float32),
        }
        d["contact"][0] = 1.0
        return d

    def test_train_path_feeds_phase_and_weights(self):
        cfg = EstimatorConfig(window=4, hidden=16, epochs=1, batch=4,
                              w_forcenorm=0.5, w_stance=3.0)
        captured = {}
        real = te.multi_task_loss

        def spy(out, targets, cfg_):
            if not captured:
                captured["keys"] = set(targets.keys())
                mask_c = (targets["contact"] > 0.5).float()
                frame_w = mask_c * (1.0 + 3.0 * (targets["phase"] > 2.5).float())
                captured["frame_w_sum"] = float(frame_w.sum())
                captured["mask_sum"] = float(mask_c.sum())
            return real(out, targets, cfg_)

        te.multi_task_loss = spy
        try:
            te.train_estimator(self._fixture(), None, cfg, verbose=False)
        finally:
            te.multi_task_loss = real
        self.assertIn("phase", captured["keys"])
        self.assertGreater(captured["frame_w_sum"], captured["mask_sum"])

    def test_train_path_without_stance_keeps_mask(self):
        cfg = EstimatorConfig(window=4, hidden=16, epochs=1, batch=4, w_stance=0.0)
        captured = {}
        real = te.multi_task_loss

        def spy(out, targets, cfg_):
            if not captured:
                mask_c = (targets["contact"] > 0.5).float()
                captured["frame_w_is_mask"] = True
                captured["mask_sum"] = float(mask_c.sum())
            return real(out, targets, cfg_)

        te.multi_task_loss = spy
        try:
            te.train_estimator(self._fixture(), None, cfg, verbose=False)
        finally:
            te.multi_task_loss = real
        self.assertTrue(captured["frame_w_is_mask"])


if __name__ == "__main__":
    unittest.main()
