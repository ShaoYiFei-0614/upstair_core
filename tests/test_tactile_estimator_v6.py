"""v6 契约测试：4x2x64 容量、聚类条件标定、k=0 逐位一致、best-val/早停硬语义。

这些测试针对评审 C 指出的「假绿」点：参数自证只覆盖旧六臂、CLI 用源码文本匹配、
早停/最优轮从不 assert、留出守卫只查名字、聚类标定可能覆盖全局通道。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import robotdog.tactile_estimator as te
from robotdog.tactile_estimator import (
    ARCH_PARAM_TARGETS, EstimatorConfig, TactileEstimator, TactileGRU,
    apply_clustered_affine, apply_standardizer, arch_spec, assign_clusters,
    fit_kmeans, fit_standardizer, make_dataset_windows, save_dataset,
    stance_force_metrics, train_estimator, window_embedding,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_v6_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic(episodes=4, steps=60, n_features=65, seed=0):
    rng = np.random.default_rng(seed)
    n = episodes * steps
    contact = (rng.random((n, 4)) > 0.35).astype(np.uint8)
    return {
        "features": rng.normal(size=(n, n_features)).astype(np.float32),
        "force_world": rng.normal(size=(n, 4, 3)).astype(np.float32) * 40.0,
        "force_foot": rng.normal(size=(n, 4, 3)).astype(np.float32) * 40.0,
        "contact": contact,
        "area_norm": rng.random((n, 4)).astype(np.float32),
        "cop_norm": rng.uniform(-1, 1, (n, 4, 2)).astype(np.float32),
        "phase": rng.integers(0, 4, (n, 4)).astype(np.float32),
        "episode": np.repeat(np.arange(episodes), steps).astype(np.int64),
        "force_norm": np.abs(rng.normal(size=(n, 4))).astype(np.float32),
        "delta_force_norm": rng.normal(size=(n, 4)).astype(np.float32),
        "impact_peak": np.abs(rng.normal(size=(n, 4))).astype(np.float32),
        "_torque": rng.normal(size=(n, 12)).astype(np.float32),
    }


def _cfg(**kw) -> EstimatorConfig:
    base = dict(window=4, hidden=8, layers=1, epochs=2, batch=8, val_frac=0.3,
                lr=1e-2, seed=0, w_forcenorm=0.5)
    base.update(kw)
    return EstimatorConfig(**base)


def _tiny_est(cfg=None, n_features=65, seed=0):
    cfg = cfg or _cfg()
    torch.manual_seed(seed)
    model = TactileGRU(n_features, cfg)
    mean = np.zeros(n_features, dtype=np.float32)
    std = np.ones(n_features, dtype=np.float32)
    return TactileEstimator(model, cfg, mean, std)


class Arch2x64Test(unittest.TestCase):
    """T1/T2：4x2x64 的容量与非共享必须可自证。"""

    def test_param_count(self):
        cfg = EstimatorConfig(arch="indep",
                              task_caps="force:2x64,contact:2x64,area:2x64,cop:2x64")
        model = TactileGRU(65, cfg)
        n = sum(p.numel() for p in model.parameters())
        self.assertEqual(n, ARCH_PARAM_TARGETS["I2x64"])
        self.assertEqual(n, 202268)
        self.assertEqual(arch_spec("indep", cfg.task_caps),
                         {t: (2, 64) for t in ("force", "contact", "area", "cop")})
        # 单任务（含 force 头 Linear(64,12)）应是 50892：arm 命名混淆时能立刻抓到
        sub = sum(p.numel() for p in model.grus["force"].parameters())
        sub += sum(p.numel() for p in model.force_head.parameters())
        self.assertEqual(sub, 50892)

    def test_no_sharing(self):
        cfg = EstimatorConfig(arch="indep",
                              task_caps="force:2x64,contact:2x64,area:2x64,cop:2x64")
        model = TactileGRU(65, cfg)
        self.assertIsNone(model.gru)
        ptrs = {id(model.grus[t]) for t in ("force", "contact", "area", "cop")}
        self.assertEqual(len(ptrs), 4)
        firsts = [next(model.grus[t].parameters()).data_ptr()
                  for t in ("force", "contact", "area", "cop")]
        self.assertEqual(len(set(firsts)), 4)


class ClusterMathTest(unittest.TestCase):
    """T5/T6：k-means 确定性、k>n 拒绝、分配必须用标准化后的嵌入。"""

    def test_deterministic(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(60, 6))
        a = fit_kmeans(x, 3, seed=7)
        b = fit_kmeans(x, 3, seed=7)
        self.assertTrue(np.array_equal(a["centers"], b["centers"]))
        self.assertEqual(a["counts"], b["counts"])
        self.assertEqual(sum(a["counts"]), 60)

    def test_k_too_large(self):
        x = np.zeros((4, 3))
        with self.assertRaises(ValueError):
            fit_kmeans(x, 5, seed=0)

    def test_assign_uses_scaler(self):
        x = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)
        scaler = fit_standardizer(x)
        z = apply_standardizer(x, scaler)
        centers = np.array([[1.0, 0.0], [-1.0, 0.0]])
        # z = [-1, +1]：第一点离 [-1,0]（索引 1）近、第二点离 [1,0]（索引 0）近
        self.assertEqual(list(assign_clusters(z, centers)), [1, 0])
        # 未标准化的输入会把两点都判给同一个中心 ⇒ 证明 scaler 真的在生效
        self.assertEqual(list(assign_clusters(x, centers)), [0, 0])

    def test_window_embedding_shape(self):
        x = np.zeros((5, 4, 3), dtype=np.float32)
        x[:, :, 0] = 2.0
        emb = window_embedding(x)
        self.assertEqual(emb.shape, (5, 6))
        self.assertAlmostEqual(float(emb[0, 0]), 2.0, places=6)
        self.assertAlmostEqual(float(emb[0, 3]), 0.0, places=6)

    def test_clustered_affine(self):
        y = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
        fits = [{"a": 2.0, "b": 0.0}, {"a": 1.0, "b": 1.0}]
        out = apply_clustered_affine(y, np.array([0, 1]), fits, 2)
        self.assertTrue(np.allclose(out, np.array([[2.0, 2.0], [3.0, 3.0]])))
        with self.assertRaises(ValueError):
            apply_clustered_affine(y, np.array([0, 1]), fits[:1], 2)


class CollectSpecGuardTest(unittest.TestCase):
    """T8：留出守卫的第二层（几何键），rough 与楼梯不能互相误判。"""

    def test_spec_key(self):
        mod = _load_script("collect_tactile_data")
        from robotdog.policy import stair_from_height
        from robotdog.tools.rough import rough_spec
        from robotdog.tools.terrain import flat_spec
        k_tread = mod.spec_key(stair_from_height(9.0, tread=0.28, n_up=8))
        # step_height_m 是米（stair_from_height 的入参是厘米，几何键统一用米）
        self.assertEqual(k_tread, ("stair", 0.09, 0.28, 8))
        self.assertNotEqual(k_tread, mod.spec_key(stair_from_height(9.0, tread=0.30, n_up=8)))
        self.assertEqual(mod.spec_key(flat_spec()), ("flat",))
        self.assertEqual(mod.spec_key(rough_spec(0.02, 0.05, seed=1)),
                         ("rough", 1, 0.05))
        # 同高度同踏面的训练 case 与留出完全同名几何 ⇒ 守卫必须能识别
        self.assertEqual(mod.spec_key(stair_from_height(16.5, tread=0.30, n_up=8)),
                         ("stair", 0.165, 0.30, 8))
        # 留出名单与 eval/calibrate 的名单同源（三处漂移会被这条抓住）
        ev = _load_script("eval_tactile_estimator")
        ca = _load_script("calibrate_tactile_estimator")
        self.assertEqual(set(ev.REQUIRED_CASES), set(ca.HOLDOUT_NAMES))
        self.assertEqual(len(ca.HOLDOUT_NAMES), 7)


class CalibrateScriptTest(unittest.TestCase):
    """T4/T7：k=0 逐位一致、聚类块只在 k>0 出现、eval 施加聚类不覆盖全局。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        data_dir = self.dir / "data"
        data_dir.mkdir()
        for i, name in enumerate(("flat", "stair15")):
            d = _synthetic(seed=i)
            save_dataset(data_dir / f"train_{name}.npz", meta={"n_features": 65,
                                                               "n_obs": 53},
                         features=d["features"], force_world=d["force_world"],
                         force_foot=d["force_foot"], contact=d["contact"],
                         area_norm=d["area_norm"], cop_norm=d["cop_norm"],
                         phase=d["phase"], episode=d["episode"],
                         force_norm=d["force_norm"],
                         delta_force_norm=d["delta_force_norm"],
                         impact_peak=d["impact_peak"])
        self.data_dir = data_dir
        self.est = _tiny_est()
        self.ckpt = self.dir / "est.pt"
        self.est.save(self.ckpt)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, out: Path, extra: list[str]):
        mod = _load_script("calibrate_tactile_estimator")
        argv = ["--checkpoint", str(self.ckpt), "--data-dir", str(self.data_dir),
                "--out", str(out), "--stride", "1", "--features", "65",
                "--min-samples", "2"] + extra
        self.assertEqual(mod.main(argv), 0)
        return json.loads(out.read_text(encoding="utf-8")), out.read_bytes()

    def test_k0_byte_identical_and_no_cluster_keys(self):
        a, ba = self._run(self.dir / "a.json", [])
        b, bb = self._run(self.dir / "b.json", ["--cluster-k", "0"])
        self.assertEqual(hashlib.sha256(ba).hexdigest(),
                         hashlib.sha256(bb).hexdigest())
        self.assertNotIn("clustered", a)
        self.assertNotIn("cluster_k", a)

    def test_cluster_block(self):
        payload, _ = self._run(self.dir / "k.json", ["--cluster-k", "2"])
        cl = payload["clustered"]
        self.assertEqual(cl["k"], 2)
        self.assertEqual(len(cl["centers"]), 2)
        self.assertEqual(len(cl["scaler_mean"]), 130)
        self.assertEqual(sum(cl["counts"]), cl["n_windows_stance"])
        for chan, per in cl["channels"].items():
            for method in ("ls", "inverse"):
                self.assertEqual(len(per[method]), 2, f"{chan}.{method}")
        # 全局通道还在，且未被覆盖（数值型 dict）
        self.assertIsInstance(payload["channels"]["vector_mag"]["ls"]["a"], float)

    def test_eval_applies_cluster(self):
        payload, _ = self._run(self.dir / "k.json", ["--cluster-k", "2"])
        payload["_path"] = "k.json"
        ev = _load_script("eval_tactile_estimator")
        data = {k: v for k, v in _synthetic(seed=9, episodes=3).items()
                if not k.startswith("_")}
        data.pop("_torque", None)
        out, wins, pred = ev.eval_dataset(self.est, data, stride=1,
                                          calibration=payload)
        cons = out["consumer"]
        self.assertIn("calibrated", cons)
        self.assertIn("calibrated_cluster", cons)
        key = "vector_mag.ls"
        self.assertIn(key, cons["calibrated_cluster"])
        # 手算：按窗口簇标签施加 per-cluster 仿射后，用同一主口径算 R²
        ref = float(self.est.cfg.force_ref_n)
        y = np.linalg.norm(pred["force"], axis=2) / ref
        scaler = {"mean": np.asarray(payload["clustered"]["scaler_mean"]),
                  "std": np.asarray(payload["clustered"]["scaler_std"])}
        emb = window_embedding(wins["x"])
        lab = assign_clusters(apply_standardizer(emb, scaler),
                              np.asarray(payload["clustered"]["centers"]))
        fits = payload["clustered"]["channels"]["vector_mag"]["ls"]
        y_c = apply_clustered_affine(y, lab, fits, 2)
        manual = stance_force_metrics(y_c, wins["forcenorm"], wins["contact"],
                                      force_ref_n=ref, phase=wins["phase"])
        self.assertAlmostEqual(
            cons["calibrated_cluster"][key]["r2"], manual["r2"], places=10)
        # 全局通道必须仍是「全局 a/b」的结果，不能被 cluster 版本替换
        self.assertNotIn("cluster_k", cons["calibrated"][key])


class BestValStopTest(unittest.TestCase):
    """T11：早停与最优轮必须真触发（lr=0 让 val 恒定不变差、不改善）。"""

    def test_early_stop_fires(self):
        cfg = _cfg(epochs=8, lr=0.0, early_stop=2, lr_gamma=0.5)
        data = _synthetic(episodes=4, steps=60)
        wins = make_dataset_windows(data, cfg, 1)
        tr, va = wins, wins
        est, hist = train_estimator(tr, va, cfg, verbose=False)
        self.assertTrue(est.stopped_early)
        self.assertEqual(est.epochs_run, 3)
        self.assertEqual(len(hist), 3)
        self.assertGreaterEqual(est.best_epoch, 1)
        self.assertEqual(hist[0]["lr"], 0.0)
        self.assertFalse(est.val_empty)

    def test_best_epoch_is_argmin(self):
        cfg = _cfg(epochs=4, lr_gamma=0.9)
        data = _synthetic(episodes=5, steps=60)
        wins = make_dataset_windows(data, cfg, 1)
        est, hist = train_estimator(wins, wins, cfg, verbose=False)
        vals = [row["val_loss"] for row in hist]
        self.assertAlmostEqual(est.best_val_loss, min(vals), places=9)
        self.assertEqual(est.best_epoch, int(np.argmin(vals)) + 1)
        lrs = [row["lr"] for row in hist]
        self.assertAlmostEqual(lrs[1], lrs[0] * 0.9, places=12)


class DecompStrictTest(unittest.TestCase):
    """T13：decomp 恒等式要 exact（places=12），否则 scatter 的断言不可证伪。"""

    def test_exact_affine(self):
        ev = _load_script("eval_tactile_estimator")
        t = np.linspace(-1, 1, 40)
        y = 0.6 * t + 0.2
        m = np.ones((40, 1), dtype=bool)
        d = ev.decomp_triple(y.reshape(-1, 1), t.reshape(-1, 1), m)
        self.assertAlmostEqual(d["slope"], 0.6, places=12)
        self.assertAlmostEqual(d["scatter"], 1.0, places=12)
        self.assertAlmostEqual(d["a_ls"], 1 / 0.6, places=9)


if __name__ == "__main__":
    unittest.main()
