"""回归：`_env.json` 反序列化必须重建 `TactileConfig`（Level1 权重可回放）。

2026-10-20 的 bug：`env_config_from_summary` 漏重建 tactile 嵌套 dataclass ⇒
`cfg.tactile` 是 dict ⇒ `_validate_tactile()` 抛 `AttributeError: dict has no attribute
"enabled"`，所有 `tactile.enabled=true` 的权重无法 play/eval。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from robotdog.env import EnvConfig, ObsConfig
from robotdog.policy import env_config_from_summary
from robotdog.tactile import TactileConfig


class TactileEnvJsonTest(unittest.TestCase):
    def test_env_json_restores_tactile_config(self) -> None:
        cfg = EnvConfig(obs=ObsConfig(history_len=1, gait_phase=True,
                                        step_height=True, vel_filter=True))
        cfg.tactile = TactileConfig(enabled=True)
        payload = {"env_config": asdict(cfg), "obs_config": asdict(cfg.obs)}
        with tempfile.TemporaryDirectory() as tmp:
            env_json = Path(tmp) / "model_final_env.json"
            env_json.write_text(json.dumps(payload, ensure_ascii=False),
                                encoding="utf-8")
            # 与 play.py 一致：传 <模型>_env.json（而不是 summary.json）
            restored = env_config_from_summary(env_json)
        self.assertIsInstance(restored.tactile, TactileConfig)
        self.assertTrue(restored.tactile.enabled)
        self.assertEqual(restored.tactile.n_taxels, 20)
        # 触觉只用于奖励/评测：默认观测里不带 tactile 块
        self.assertEqual(restored.obs.size, 53)


if __name__ == "__main__":
    unittest.main()