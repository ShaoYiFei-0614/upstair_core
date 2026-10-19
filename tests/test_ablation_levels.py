"""消融接口契约：Level 0→3 只能靠配置切换，不许靠代码分支。

对应 `doc/ablation.md` 与 `ablations/level{0,1,2,3}.yaml`。钉住四件事：

1. **Actor/Critic 输入维拼接**：L0 是 53/53 对称；L1–L3 是 57 帧、actor 53 / critic 57，
   策略类随 `obs.privileged_foot_contact` 自动切换（同一份 `build_ppo`）。
2. **reward 项开关**：L2/L3 打开 `contact_rewards` + `foot_scuff`，L0/L1 保持关闭。
3. **课程采样器**：L3 的 `curriculum` 展开成两级，高度/指令区间/奖励覆盖逐级不同；
   L0–L2 是单级 15cm。
4. **同一训练脚本**：以上全部通过 `scripts/train.py --config ablations/levelN.yaml` 生效（本测试只验
   配置→EnvConfig/StageSpec/策略装配这一段，训练链路由 `.tmp/ablation_smoke` 冒烟覆盖）。

跑法：`.venv\\Scripts\\python.exe -m unittest tests.test_ablation_levels -v`
"""

from __future__ import annotations

import unittest
from pathlib import Path

from robotdog.asymmetric import AsymmetricActorCriticPolicy
from robotdog.config import config_to_env, config_to_stages, load_config
from robotdog.policy import stair_from_height
from robotdog.training import build_ppo, make_vec_env

CONFIGS = Path("ablations")

EXPECTED = {
    0: dict(size=53, frame=53, actor=53, priv=0, policy="ActorCriticPolicy",
            stages=1, heights=[15.0], contact=False, scuff=0.0),
    1: dict(size=57, frame=57, actor=53, priv=4, policy="AsymmetricActorCriticPolicy",
            stages=1, heights=[15.0], contact=False, scuff=0.0),
    2: dict(size=57, frame=57, actor=53, priv=4, policy="AsymmetricActorCriticPolicy",
            stages=1, heights=[15.0], contact=True, scuff=-8.0),
    3: dict(size=57, frame=57, actor=53, priv=4, policy="AsymmetricActorCriticPolicy",
            stages=2, heights=[10.5, 15.0], contact=True, scuff=-8.0),
}


class AblationConfigTest(unittest.TestCase):
    def test_levels_map_to_config_only(self) -> None:
        for level, want in EXPECTED.items():
            with self.subTest(level=level):
                raw = load_config(CONFIGS / f"level{level}.yaml")
                cfg = config_to_env(raw)
                stages = config_to_stages(raw)

                self.assertEqual(cfg.obs.size, want["size"])
                self.assertEqual(cfg.obs.frame_size, want["frame"])
                self.assertEqual(cfg.obs.actor_size, want["actor"])
                self.assertEqual(cfg.obs.privileged_frame_size, want["priv"])
                self.assertEqual(cfg.reward.contact_rewards, want["contact"])
                self.assertAlmostEqual(cfg.reward.foot_scuff, want["scuff"])

                self.assertEqual(len(stages), want["stages"])
                self.assertEqual([round(s.step_height * 100, 1) for s in stages],
                                 want["heights"])

    def test_reward_switches_only_change_additional_terms(self) -> None:
        raw0 = load_config(CONFIGS / "level0.yaml")
        raw2 = load_config(CONFIGS / "level2.yaml")
        r0 = config_to_env(raw0).reward
        r2 = config_to_env(raw2).reward
        self.assertFalse(r0.contact_rewards)
        self.assertTrue(r2.contact_rewards)
        self.assertAlmostEqual(r2.foot_scuff, -8.0)
        # 其余基线权重逐项不变（reward 开关只加项，不改旧项）
        for name in ("forward_vel", "foot_lift", "probe_lift", "air_time",
                     "orientation", "torque", "action_rate", "success"):
            self.assertEqual(getattr(r0, name), getattr(r2, name), name)

    def test_curriculum_sampler_is_per_stage(self) -> None:
        raw = load_config(CONFIGS / "level3.yaml")
        stages = config_to_stages(raw)
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0].command_vx, (0.2, 0.6))
        self.assertEqual(stages[1].command_vx, (0.25, 0.7))
        self.assertEqual(stages[0].command_vy_max, 0.0)
        self.assertEqual(stages[1].command_vy_max, 0.15)
        self.assertEqual(stages[0].command_yaw_max, 0.0)
        self.assertEqual(stages[1].command_yaw_max, 0.4)
        # 逐级学习率/奖励覆盖也必须各 2 份
        self.assertEqual(len({s.learning_rate for s in stages}), 2)
        self.assertNotEqual(stages[0].reward_overrides, stages[1].reward_overrides)


class PolicyAssemblyTest(unittest.TestCase):
    def test_policy_class_and_dims_follow_config(self) -> None:
        for level, want in EXPECTED.items():
            with self.subTest(level=level):
                raw = load_config(CONFIGS / f"level{level}.yaml")
                cfg = config_to_env(raw)
                stair = stair_from_height(15.0, tread=0.30, n_up=8)
                venv = make_vec_env(stair, n_envs=1, cfg=cfg, seed=0)
                try:
                    model = build_ppo(
                        venv, seed=0, net_arch=(64, 32), n_steps=16, batch_size=16,
                        n_epochs=1, verbose=0,
                        actor_obs_index=(cfg.obs.actor_index
                                         if cfg.obs.privileged_frame_size > 0 else None),
                    )
                    pol = model.policy
                    self.assertEqual(type(pol).__name__, want["policy"])
                    self.assertEqual(pol.mlp_extractor.policy_net[0].in_features,
                                     want["actor"])
                    self.assertEqual(pol.mlp_extractor.value_net[0].in_features,
                                     want["size"])
                    self.assertEqual(venv.observation_space.shape[0], want["size"])
                finally:
                    venv.close()


if __name__ == "__main__":
    unittest.main()
