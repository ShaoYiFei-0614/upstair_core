"""``run.log_reward_terms`` 训练期奖励明细记录的契约测试（unittest）。

钉住三件事：
1. ``RewardTermsMeter`` 的口径：按**步**取均值、只记点名的键、没出现的键不记、
   ``drain()`` 取走即清零、没有 ``reward_terms`` 的 info 不计步数；
2. ``config_to_run`` 的默认与校验（默认空列表 = 关闭；非字符串列表直接报错）；
3. 键名列表可以包含暂时不存在的项（该窗口只是不记它，不报错）。

跑法：`.venv\\Scripts\\python.exe -m unittest tests.test_reward_terms_log -v`
"""

from __future__ import annotations

import unittest

from robotdog.config import config_to_run
from robotdog.curriculum import RewardTermsMeter


class MeterTest(unittest.TestCase):
    def test_mean_over_steps_and_filter_keys(self) -> None:
        meter = RewardTermsMeter(keys=("forward_vel", "alive"))
        n = meter.add_infos([
            {"reward_terms": {"forward_vel": -2.0, "alive": 0.15, "torque": -1.0}},
            {"reward_terms": {"forward_vel": -4.0, "alive": 0.15}},
        ])
        self.assertEqual(n, 2)
        means = meter.drain()
        self.assertAlmostEqual(means["forward_vel"], -3.0)
        self.assertAlmostEqual(means["alive"], 0.15)
        self.assertNotIn("torque", means)          # 没点名的不记
        self.assertEqual(meter.drain(), {})         # 取走即清零
        self.assertEqual(meter.n_steps, 0)

    def test_infos_without_terms_are_skipped(self) -> None:
        meter = RewardTermsMeter(keys=("alive",))
        meter.add_infos([{"episode": {"r": 1.0}}, {"reward_terms": {"alive": 0.3}}])
        self.assertAlmostEqual(meter.drain()["alive"], 0.3)

    def test_missing_key_is_simply_absent(self) -> None:
        meter = RewardTermsMeter(keys=("forward_vel", "termination"))
        meter.add_infos([{"reward_terms": {"forward_vel": -1.0}}])
        means = meter.drain()
        self.assertEqual(set(means), {"forward_vel"})

    def test_one_shot_terms_are_amortized_over_steps(self) -> None:
        # termination 只在摔倒那一步出现：摊到窗口步数上，量级仍是负的大数
        meter = RewardTermsMeter(keys=("termination",))
        meter.add_infos([{"reward_terms": {"termination": -5000.0}}]
                        + [{"reward_terms": {}} for _ in range(9)])
        self.assertAlmostEqual(meter.drain()["termination"], -500.0)


class RunConfigTest(unittest.TestCase):
    def test_default_is_empty(self) -> None:
        run = config_to_run({"run": {"name": "x"}})
        self.assertEqual(run["log_reward_terms"], [])

    def test_list_accepted(self) -> None:
        run = config_to_run({"run": {"name": "x",
                                     "log_reward_terms": ["forward_vel", "alive"]}})
        self.assertEqual(run["log_reward_terms"], ["forward_vel", "alive"])

    def test_bad_type_rejected(self) -> None:
        for bad in ("forward_vel", [1, 2], {"a": 1}):
            with self.assertRaises(SystemExit,
                                   msg=f"log_reward_terms={bad!r} 应被拒绝"):
                config_to_run({"run": {"name": "x", "log_reward_terms": bad}})


if __name__ == "__main__":
    unittest.main()
