"""rollup_waveA 的判据守卫测试（评审 B1/B2/B5/B6/N8/N9 反例）。

覆盖：A-P5 必须排除基线臂且用 max；缺 case / 混入非 test 行必须 SystemExit；
A-P4 必须 INCONCLUSIVE；MDE 缺失必须阻断（除非显式 --allow-no-mde）。
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_rollup():
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "rollup_waveA", root / ".tmp" / "tactile_est_v5" / "tools" / "rollup_waveA.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(name, r2=0.0, f1=0.9, slope=1.0, ratio=1.0, split="test"):
    return {"name": name, "split": split,
            "consumer": {"vector_mag": {"r2": r2, "slope": slope,
                                        "mean_ratio": ratio}},
            "contact": {"f1": f1},
            "case_stats": {"n_stance": 500},
            "decomp": {"vector_mag": {"slope": slope}},
            "bootstrap": {"ci_low": r2 - 0.1, "ci_high": r2 + 0.1}}


class RollupGuardTest(unittest.TestCase):
    def setUp(self):
        self.mod = _load_rollup()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, arm, rows):
        (self.dir / f"report_{arm}.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    def _sigma(self):
        p = self.dir / "d3_sigma.json"
        p.write_text(json.dumps({"primary": {"sigma_r2_mean": 0.05,
                                             "mde_1.5sqrt2": 0.106}}),
                     encoding="utf-8")
        return str(p)

    def _run(self, *extra):
        return self.mod.main([
            "--reports-dir", str(self.dir), "--out", str(self.dir / "rollup.json"),
            *extra])

    def test_p5_excludes_baseline_and_uses_max(self):
        # A0 = F1 0.95；A1 掉到 0.75（下降 0.20）⇒ 必须 FAIL
        for arm in ("A0", "A1", "A2", "A3"):
            rows = [_row(n, f1=0.95 if arm == "A0" else 0.90)
                    for n in self.mod.HOLDOUT]
            if arm == "A1":
                rows[0] = _row(self.mod.HOLDOUT[0], f1=0.75)
            self._write(arm, rows)
        self._run("--sigma", self._sigma())
        out = json.loads((self.dir / "rollup.json").read_text(encoding="utf-8"))
        p5 = next(c for c in out["criteria"]["checks"] if c["id"] == "A-P5_contact_guard")
        self.assertAlmostEqual(p5["value"], 0.20, places=6)   # 不是 0（自比）也不是 min
        self.assertEqual(p5["status"], "FAIL")
        self.assertEqual(set(p5["detail"]["arms_compared"]), {"A1", "A2", "A3"})
        self.assertNotIn("A0", next(iter(p5["detail"]["per_case"].values())))

    def test_p4_is_inconclusive(self):
        for arm in ("A0", "A1", "A2", "A3"):
            self._write(arm, [_row(n, r2=0.0) for n in self.mod.HOLDOUT])
        self._run("--sigma", self._sigma())
        out = json.loads((self.dir / "rollup.json").read_text(encoding="utf-8"))
        p4 = next(c for c in out["criteria"]["checks"] if c["id"] == "A-P4_policies")
        self.assertEqual(p4["status"], "INCONCLUSIVE")

    def test_missing_case_blocks(self):
        for arm in ("A0", "A1", "A2", "A3"):
            rows = [_row(n) for n in self.mod.HOLDOUT if n != "stair13_5"]
            self._write(arm, rows)
        with self.assertRaises(SystemExit):
            self._run("--sigma", self._sigma())

    def test_non_test_rows_block(self):
        for arm in ("A0", "A1", "A2", "A3"):
            rows = [_row(n) for n in self.mod.HOLDOUT]
            rows[0] = _row(rows[0]["name"], split="train")
            self._write(arm, rows)
        with self.assertRaises(SystemExit):
            self._run("--sigma", self._sigma())

    def test_missing_mde_blocks_without_flag(self):
        for arm in ("A0", "A1", "A2", "A3"):
            self._write(arm, [_row(n) for n in self.mod.HOLDOUT])
        with self.assertRaises(SystemExit):
            self._run("--sigma", str(self.dir / "nope.json"))
        # 显式放行（只用于调试）时不应崩
        self.assertEqual(self._run("--sigma", str(self.dir / "nope.json"),
                                   "--allow-no-mde"), 0)

    def test_p2_uses_a2_minus_a1(self):
        # A0=0.0, A1=0.5（数量臂很强）, A2=0.6 ⇒ A2-A1=0.1，而不是 A1-A0=0.5
        for arm, val in (("A0", 0.0), ("A1", 0.5), ("A2", 0.6), ("A3", 0.6)):
            self._write(arm, [_row(n, r2=val) for n in self.mod.HOLDOUT])
        self._run("--sigma", self._sigma())
        out = json.loads((self.dir / "rollup.json").read_text(encoding="utf-8"))
        p2 = next(c for c in out["criteria"]["checks"] if c["id"] == "A-P2_quantity")
        self.assertAlmostEqual(p2["value"], 0.1, places=6)

    def test_noise_flag(self):
        for arm in ("A0", "A1", "A2", "A3"):
            self._write(arm, [_row(n, r2=0.5) for n in self.mod.HOLDOUT])
        self._run("--sigma", self._sigma())
        out = json.loads((self.dir / "rollup.json").read_text(encoding="utf-8"))
        p1 = next(c for c in out["criteria"]["checks"] if c["id"] == "A-P1_types")
        self.assertAlmostEqual(p1["value"], 0.0, places=9)
        self.assertTrue(p1["noise"])          # 单 seed 且 |Δ|<0.02 ⇒ 噪声标记


if __name__ == "__main__":
    unittest.main()
