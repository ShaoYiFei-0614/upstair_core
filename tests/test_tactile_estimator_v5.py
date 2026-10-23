from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import robotdog.tactile_estimator as te
from robotdog.tactile_estimator import (
    ARCH_PARAM_TARGETS, ARCH_SPECS, HEAD_DIMS, TASKS, EstimatorConfig,
    TactileEstimator, TactileGRU, arch_spec, make_dataset_windows,
    parse_task_caps, save_dataset, train_estimator,
)


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_v5_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _caps_str(arch: str) -> str:
    spec = ARCH_SPECS[arch]
    return ",".join(f"{task}:{spec[task][0]}x{spec[task][1]}" for task in TASKS)


def _cfg_for(arch: str) -> EstimatorConfig:
    if arch in ("S1", "S3"):
        layers = ARCH_SPECS[arch]["force"][0]
        return EstimatorConfig(arch="shared", hidden=128, layers=layers)
    return EstimatorConfig(arch="indep", task_caps=_caps_str(arch))


def _synthetic(episodes=4, steps=14, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    n = episodes * steps
    contact = (rng.random((n, 4)) > 0.4).astype(np.uint8)
    data = {
        "features": rng.normal(size=(n, n_features)).astype(np.float32),
        "force_world": rng.normal(size=(n, 4, 3)).astype(np.float32),
        "force_foot": rng.normal(size=(n, 4, 3)).astype(np.float32),
        "contact": contact,
        "area_norm": rng.random((n, 4)).astype(np.float32),
        "cop_norm": rng.uniform(-1, 1, (n, 4, 2)).astype(np.float32),
        "phase": rng.integers(0, 4, (n, 4)).astype(np.float32),
        "episode": np.repeat(np.arange(episodes), steps).astype(np.int64),
        "force_norm": np.abs(rng.normal(size=(n, 4))).astype(np.float32),
        "delta_force_norm": rng.normal(size=(n, 4)).astype(np.float32),
        "impact_peak": np.abs(rng.normal(size=(n, 4))).astype(np.float32),
    }
    return data


class ArchParamCountTest(unittest.TestCase):
    """L3：六条预注册臂的参数总量必须与常数表逐位一致（输入 65）。"""

    def test_param_targets(self):
        for arch, target in ARCH_PARAM_TARGETS.items():
            cfg = _cfg_for(arch)
            model = TactileGRU(65, cfg)
            n = sum(p.numel() for p in model.parameters())
            self.assertEqual(n, target, f"{arch} 参数量 {n} != {target}")

    def test_arch_spec_matches_config(self):
        cfg = _cfg_for("I2")
        spec = arch_spec("indep", cfg.task_caps)
        self.assertEqual(spec, ARCH_SPECS["I2"])
        self.assertEqual(arch_spec("shared"), ARCH_SPECS["S1"])


class TaskCapsTest(unittest.TestCase):
    def test_roundtrip(self):
        caps = parse_task_caps(_caps_str("I2"))
        self.assertEqual(caps["force"], (2, 128))
        self.assertEqual(caps["contact"], (1, 128))
        self.assertEqual(caps["area"], (1, 64))
        self.assertEqual(caps["cop"], (1, 64))

    def test_errors(self):
        for bad in ("force:1x64,contact:1x64,area:1x64",          # 缺 cop
                    "force:1x64,contact:1x64,area:1x64,cop:1x64,force:1x64",
                    "force:1x64,contact:1x64,area:1x64,cop:1x0",
                    "force:1x64,contact:1x64,area:1x64,depth:1x64"):
            with self.assertRaises(ValueError):
                parse_task_caps(bad)

    def test_config_rejects_bad_arch(self):
        with self.assertRaises(ValueError):
            EstimatorConfig(arch="mixture")
        with self.assertRaises(ValueError):
            EstimatorConfig(arch="indep", task_caps="force:1x64")


class IndepModelTest(unittest.TestCase):
    def test_no_shared_trunk(self):
        cfg = _cfg_for("I1")
        model = TactileGRU(5, cfg)
        self.assertIsNone(model.gru)
        self.assertEqual(sorted(model.grus.keys()), sorted(TASKS))
        for task in TASKS:
            head = getattr(model, f"{task}_head")
            self.assertEqual(head.out_features, HEAD_DIMS[task])
            self.assertIsNone(getattr(model, "forcenorm_head", None))

    def test_forward_shapes(self):
        cfg = _cfg_for("I1")
        model = TactileGRU(5, cfg)
        out = model(torch.zeros(3, 6, 5))
        self.assertEqual(set(out), set(TASKS))
        self.assertEqual(tuple(out["force"].shape), (3, 12))
        self.assertEqual(tuple(out["contact"].shape), (3, 4))
        self.assertEqual(tuple(out["area"].shape), (3, 4))
        self.assertEqual(tuple(out["cop"].shape), (3, 8))

    def test_shared_still_has_trunk(self):
        model = TactileGRU(5, EstimatorConfig())
        self.assertIsNotNone(model.gru)
        self.assertFalse(hasattr(model, "grus"))


class PrivExtraWhitelistTest(unittest.TestCase):
    def _save(self, extra, features=None):
        n = 4
        if features is None:
            features = np.zeros((n, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            save_dataset(Path(td) / "d.npz",
                         features=features,
                         force_world=np.zeros((n, 4, 3), dtype=np.float32),
                         force_foot=np.zeros((n, 4, 3), dtype=np.float32),
                         contact=np.zeros((n, 4), dtype=np.uint8),
                         area_norm=np.zeros((n, 4), dtype=np.float32),
                         cop_norm=np.zeros((n, 4, 2), dtype=np.float32),
                         phase=np.zeros((n, 4), dtype=np.float32),
                         episode=np.zeros(n, dtype=np.int64),
                         meta={},
                         extra=extra)

    def test_priv_prefix_required(self):
        self._save({"priv_foot_z": np.zeros((4, 4), dtype=np.float32)})
        with self.assertRaises(ValueError):
            self._save({"foot_z": np.zeros((4, 4), dtype=np.float32)})

    def test_length_contract(self):
        with self.assertRaises(ValueError):
            self._save({"priv_foot_z": np.zeros((3, 4), dtype=np.float32)})

    def test_priv_columns_do_not_collide_with_labels(self):
        # priv_* 只作为独立数组落盘，不与 force_world 等标签同名 ⇒ 不会混进特征
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.npz"
            n = 4
            save_dataset(p, features=np.zeros((n, 3), dtype=np.float32),
                         force_world=np.zeros((n, 4, 3), dtype=np.float32),
                         force_foot=np.zeros((n, 4, 3), dtype=np.float32),
                         contact=np.zeros((n, 4), dtype=np.uint8),
                         area_norm=np.zeros((n, 4), dtype=np.float32),
                         cop_norm=np.zeros((n, 4, 2), dtype=np.float32),
                         phase=np.zeros((n, 4), dtype=np.float32),
                         episode=np.zeros(n, dtype=np.int64), meta={},
                         extra={"priv_foot_z": np.zeros((n, 4), dtype=np.float32)})
            d = te.load_dataset(p)
            self.assertEqual(d["features"].shape, (n, 3))
            self.assertIn("priv_foot_z", d)
            self.assertEqual(d["meta"].get("n_features", 3), 3)


class EvalStatsTest(unittest.TestCase):
    def test_decomp_identity(self):
        mod = _load_script("eval_tactile_estimator")
        rng = np.random.default_rng(0)
        t = rng.normal(1.0, 0.3, 400).astype(np.float32)
        y = (0.6 * t + 0.2).astype(np.float32)
        mask = np.ones(t.shape, dtype=bool)
        d = mod.decomp_triple(y, t, mask)
        self.assertAlmostEqual(d["slope"], 0.6, places=3)
        # bias = mean(y)-mean(t) = 0.2 + (0.6-1)*mean(t) ≈ -0.2（不是截距 0.2）
        self.assertAlmostEqual(d["bias"], 0.2 - 0.4 * t.mean(), places=2)
        self.assertAlmostEqual(d["scatter"], 1.0, places=2)
        self.assertEqual(d["n"], 400)

    def test_bootstrap_degenerate_single_episode(self):
        mod = _load_script("eval_tactile_estimator")
        y = np.arange(20, dtype=np.float32)
        t = np.arange(20, dtype=np.float32)
        out = mod.cluster_bootstrap_r2(y, t, np.zeros(20), n_boot=10, seed=0)
        self.assertTrue(out["degenerate"])
        self.assertEqual(out["n_episodes"], 1)

    def test_bootstrap_ci_covers_point(self):
        mod = _load_script("eval_tactile_estimator")
        rng = np.random.default_rng(1)
        ep = np.repeat(np.arange(6), 30)
        t = rng.normal(size=180).astype(np.float32)
        y = (t + rng.normal(scale=0.1, size=180)).astype(np.float32)
        out = mod.cluster_bootstrap_r2(y, t, ep, n_boot=200, seed=0)
        self.assertFalse(out["degenerate"])
        self.assertEqual(out["n_episodes"], 6)
        self.assertLessEqual(out["ci_low"], out["ci_high"])
        self.assertGreater(out["ci_low"], 0.5)

    def test_required_cases_and_grid(self):
        mod = _load_script("eval_tactile_estimator")
        self.assertEqual(len(mod.REQUIRED_CASES), 7)
        self.assertEqual(set(mod.REQUIRED_CASES), {
            "stair10_5", "stair13_5", "stair16_5", "stair18", "stair19_5",
            "rough_s1", "rough_s2"})
        self.assertEqual(len(mod.CLAMP_GRID), 8)
        cal = _load_script("calibrate_tactile_estimator")
        self.assertEqual(len(cal.HOLDOUT_NAMES), 7)


class TrainLoopTest(unittest.TestCase):
    def _train(self, **kw):
        cfg = EstimatorConfig(window=4, hidden=8, epochs=3, batch=8,
                              val_frac=0.4, seed=0, **kw)
        data = _synthetic(episodes=4, steps=14, n_features=5)
        wins = make_dataset_windows(data, cfg, stride=2)
        mask = np.zeros(wins["x"].shape[0], dtype=bool)
        mask[::2] = True
        train = {k: (v[mask] if hasattr(v, "shape") and v.shape[:1] == mask.shape else v)
                 for k, v in wins.items()}
        val = {k: (v[~mask] if hasattr(v, "shape") and v.shape[:1] == mask.shape else v)
               for k, v in wins.items()}
        est, history = train_estimator(train, val, cfg, verbose=False)
        return est, history, cfg

    def test_early_stop_and_best_val_meta(self):
        est, hist, _ = self._train(early_stop=1)
        self.assertTrue(hasattr(est, "best_epoch"))
        self.assertIsNotNone(est.best_val_loss)
        self.assertLessEqual(est.best_epoch, est.epochs_run)
        self.assertTrue(all("lr" in row for row in hist))
        self.assertTrue(any(row["epoch"] == est.best_epoch for row in hist))

    def test_lr_gamma_decays(self):
        _est, hist, _ = self._train(lr_gamma=0.5)
        lrs = [row["lr"] for row in hist]
        self.assertAlmostEqual(lrs[0], 1e-3, places=9)
        self.assertAlmostEqual(lrs[1], 5e-4, places=9)
        self.assertAlmostEqual(lrs[2], 2.5e-4, places=9)

    def test_no_early_stop_runs_all_epochs(self):
        est, _hist, cfg = self._train()
        self.assertEqual(est.epochs_run, cfg.epochs)
        self.assertFalse(est.stopped_early)


class Window40Test(unittest.TestCase):
    def test_window40_does_not_cross_episode(self):
        cfg = EstimatorConfig(window=40, hidden=8, epochs=1, batch=8)
        data = _synthetic(episodes=4, steps=60, n_features=5)
        wins = make_dataset_windows(data, cfg, stride=2)
        ends = te.window_starts(60, 40, 2)
        ep = np.repeat(np.arange(4), 60)
        for i, e in enumerate(ends):
            ep_block = ep[e - 39:e + 1]
            self.assertEqual(len(set(ep_block.tolist())), 1,
                             f"窗口 {i} 跨了回合：{sorted(set(ep_block.tolist()))}")
        self.assertEqual(wins["x"].shape[1], 40)


class EvalStatsInReportTest(unittest.TestCase):
    def test_eval_dataset_has_case_stats(self):
        mod = _load_script("eval_tactile_estimator")
        cfg = EstimatorConfig(window=4, hidden=8, epochs=1, batch=8)
        model = TactileGRU(5, cfg)
        est = TactileEstimator(model, cfg, np.zeros(5, dtype=np.float32),
                               np.ones(5, dtype=np.float32))
        data = _synthetic(episodes=3, steps=20, n_features=5)
        res, _w, _p = mod.eval_dataset(est, data, stride=4, bootstrap=20)
        self.assertIn("n_stance", res["case_stats"])
        self.assertIn("vector_mag", res["decomp"])
        self.assertEqual(res["decomp"]["vector_mag"]["n"], res["case_stats"]["n_stance"])
        self.assertIn("bootstrap", res)


if __name__ == "__main__":
    unittest.main()