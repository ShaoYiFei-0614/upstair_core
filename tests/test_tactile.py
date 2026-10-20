"""足底触觉（TactileStep）契约测试（unittest，不依赖 pytest）。

钉住四件事：

1. 默认关闭逐位兼容：观测 53 维、info 里没有 tactile、逐步奖励里没有
   tactile_* 键 —— 现有权重/评测不受影响。
2. 测量物理性：接触时 F̄/Ā/CoP 有值且量纲正确（面积 <= 足底面积、CoP 在
   [-1,1]、力非负），相位机按 swing→prelanding→landing→stance 推进。
3. 观测块一致：obs.tactile 打开后每帧末尾 16 维与 info["tactile"] 同源；
   obs.privileged_tactile 只进 critic（不进 actor_index）。
4. 三个相位策略：符号正确（prelanding/impact 为负、stance 为正）、门控正确
   （只在对应相位结算）、量级有界；未开 tactile 却配权重 ⇒ 构造期报错。

跑法：.venv\\Scripts\\python.exe -m unittest tests.test_tactile -v
"""

from __future__ import annotations

import unittest

import numpy as np

from robotdog.env import EnvConfig, Go2StairsEnv, ObsConfig
from robotdog.tactile import (LANDING, PHASE_NAMES, PRELANDING, STANCE, SWING,
                              TactileConfig)
from robotdog.tools.terrain import demo_stairs_spec, flat_spec

#: 当前 config.yaml 的观测开关组合（53 维，history=1）。
BASE_OBS = dict(history_len=1, gait_phase=True, step_height=True, vel_filter=True)


def base_obs(**kw) -> ObsConfig:
    return ObsConfig(**{**BASE_OBS, **kw})


def make_env(cfg: EnvConfig, stair=None) -> Go2StairsEnv:
    return Go2StairsEnv(stair=stair if stair is not None else
                        demo_stairs_spec(step_height=0.06), cfg=cfg, seed=0)


def tactile_env(**reward_kw) -> Go2StairsEnv:
    cfg = EnvConfig(tactile=TactileConfig(enabled=True), obs=base_obs())
    for k, v in reward_kw.items():
        setattr(cfg.reward, k, v)
    return make_env(cfg)


class DefaultOffTest(unittest.TestCase):
    """1. 默认关闭 —— 与不带触觉的旧实现逐位一致。"""

    def test_no_info_no_terms_obs_unchanged(self) -> None:
        env = make_env(EnvConfig(obs=base_obs()))
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (53,))
        for _ in range(30):
            obs, _r, term, trunc, info = env.step(np.zeros(12))
            self.assertNotIn("tactile", info)
            self.assertFalse([k for k in info["reward_terms"]
                              if k.startswith("tactile")])
            if term or trunc:
                break

    def test_off_equals_missing_config(self) -> None:
        # 显式 enabled=False 与根本没写 tactile 段等价（同一个默认对象）
        a = make_env(EnvConfig(obs=base_obs()))
        b = make_env(EnvConfig(obs=base_obs(), tactile=TactileConfig(enabled=False)))
        oa, _ = a.reset(seed=0)
        ob, _ = b.reset(seed=0)
        np.testing.assert_array_equal(oa, ob)


class MeasurementTest(unittest.TestCase):
    """2. 测量的物理性（量纲、上下界、相位机）。"""

    def test_quantities_and_phase_machine(self) -> None:
        env = tactile_env()
        env.reset(seed=0)
        phases = set()
        touched = 0
        for _ in range(300):
            _o, _r, term, trunc, info = env.step(np.zeros(12))
            tac = info["tactile"]
            phases.update(tac["phase"])          # info 里是相位**名**（字符串）
            fn = np.asarray(tac["force_n"])
            area = np.asarray(tac["area"])
            cop = np.asarray(tac["cop_norm"])
            self.assertTrue(np.all(fn >= 0.0))
            self.assertTrue(np.all(area >= 0.0))
            self.assertTrue(np.all(np.abs(cop) <= 1.0 + 1e-9))
            self.assertLessEqual(float(area.max()),
                                 env.cfg.tactile.sole_area + 1e-9)
            if fn.sum() > 0:
                touched += 1
            if term or trunc:
                break
        self.assertGreater(touched, 0)
        self.assertEqual(phases, set(PHASE_NAMES))

    def test_phase_names_and_order(self) -> None:
        self.assertEqual(PHASE_NAMES, ("swing", "prelanding", "landing", "stance"))
        env = tactile_env()
        env.reset(seed=5)
        name_to_id = {name: i for i, name in enumerate(PHASE_NAMES)}
        seen = []
        for _ in range(200):
            _o, _r, term, trunc, info = env.step(np.zeros(12))
            seen.extend(name_to_id[name]
                        for name in info["tactile"]["phase"][:1])
            if term or trunc:
                break
        seq = [p for i, p in enumerate(seen) if i == 0 or p != seen[i - 1]]
        for a, b in zip(seq, seq[1:]):
            if b == LANDING:
                self.assertIn(a, (SWING, PRELANDING))

    def test_actor_features_shape_and_range(self) -> None:
        env = tactile_env()
        env.reset(seed=0)
        for _ in range(30):
            _o, _r, term, trunc, info = env.step(np.zeros(12))
            feat = env.tactile.actor_features()
            self.assertEqual(feat.shape, (16,))
            self.assertTrue(np.all(np.isfinite(feat)))
            if term or trunc:
                break


class ObservationBlockTest(unittest.TestCase):
    """3. 观测块：actor 16 维可进 obs，特权 16 维只给 critic。"""

    def test_tactile_in_obs_matches_info(self) -> None:
        cfg = EnvConfig(tactile=TactileConfig(enabled=True),
                        obs=base_obs(tactile=True))
        env = make_env(cfg)
        obs, _ = env.reset(seed=0)
        self.assertEqual(cfg.obs.size, 53 + 16)
        self.assertEqual(obs.shape, (69,))
        for _ in range(30):
            obs, _r, term, trunc, info = env.step(np.zeros(12))
            np.testing.assert_allclose(obs[-16:],
                                       env.tactile.actor_features(), atol=1e-6)
            if term or trunc:
                break

    def test_privileged_tactile_not_in_actor_index(self) -> None:
        cfg = EnvConfig(tactile=TactileConfig(enabled=True),
                        obs=base_obs(privileged_tactile=True))
        env = make_env(cfg)
        env.reset(seed=0)
        self.assertEqual(cfg.obs.size, 69)
        self.assertEqual(cfg.obs.actor_size, 53)
        self.assertEqual(len(cfg.obs.actor_index), 53)
        self.assertNotIn(True, [i >= 53 for i in cfg.obs.actor_index])

    def test_requires_enabled_sensor(self) -> None:
        with self.assertRaises(ValueError):
            make_env(EnvConfig(obs=base_obs(tactile=True)))
        with self.assertRaises(ValueError):
            make_env(EnvConfig(obs=base_obs(privileged_tactile=True)))


class PhasePolicyTest(unittest.TestCase):
    """4. 三个相位条件策略：符号、门控、有界、构造期校验。"""

    def _run(self, env, n=200):
        env.reset(seed=0)
        rows = []
        for _ in range(n):
            _o, _r, term, trunc, info = env.step(np.zeros(12))
            # 注意：state 对象在 step 之间复用，必须深拷贝后存
            tac = {k: (np.array(v, copy=True) if isinstance(v, (list, tuple, np.ndarray))
                       else v) for k, v in info["tactile"].items()}
            rows.append((dict(info["reward_terms"]), tac))
            if term or trunc:
                break
        return rows

    def test_signs_and_bounds(self) -> None:
        w_pre, w_imp, w_sta = -0.5, -1.0, 0.8
        env = tactile_env(tactile_prelanding=w_pre, tactile_impact=w_imp,
                          tactile_stance=w_sta)
        rows = self._run(env)
        vals = {k: [] for k in ("tactile_prelanding", "tactile_impact",
                                "tactile_stance")}
        for terms, _tac in rows:
            for k in vals:
                vals[k].append(float(terms[k]))
        pre = np.asarray(vals["tactile_prelanding"])
        imp = np.asarray(vals["tactile_impact"])
        sta = np.asarray(vals["tactile_stance"])
        self.assertTrue(np.all(pre <= 0.0) and np.all(pre >= w_pre - 1e-12))
        self.assertTrue(np.all(imp <= 0.0) and np.all(imp >= 6.0 * w_imp - 1e-12))
        self.assertTrue(np.all(sta >= 0.0) and np.all(sta <= 2.0 * w_sta + 1e-12))
        self.assertLess(float(pre.min()), -1e-9)   # 至少有一帧非零

    def test_gated_by_phase(self) -> None:
        env = tactile_env(tactile_prelanding=-0.5, tactile_impact=-1.0,
                          tactile_stance=0.8)
        rows = self._run(env)
        for terms, tac in rows:
            names = list(tac["phase"])
            if PHASE_NAMES[PRELANDING] not in names:
                self.assertEqual(terms["tactile_prelanding"], 0.0)
            if PHASE_NAMES[LANDING] not in names:
                self.assertEqual(terms["tactile_impact"], 0.0)
            if PHASE_NAMES[STANCE] not in names:
                self.assertEqual(terms["tactile_stance"], 0.0)

    def test_zero_weight_keeps_zero(self) -> None:
        env = tactile_env()   # 权重全 0（默认）
        rows = self._run(env)
        for terms, _tac in rows:
            # 三个权重都为 0 时整段跳过：键根本不进 terms（与旧实现逐位一致）
            self.assertNotIn("tactile_prelanding", terms)
            self.assertNotIn("tactile_impact", terms)
            self.assertNotIn("tactile_stance", terms)

    def test_reward_weight_requires_sensor(self) -> None:
        cfg = EnvConfig(obs=base_obs())
        cfg.reward.tactile_impact = -1.0
        with self.assertRaises(ValueError):
            make_env(cfg)


class FlatAndStairTest(unittest.TestCase):
    """平地/楼梯两种地形都不能崩（触觉在平地也应有接触）。"""

    def test_flat_touch(self) -> None:
        cfg = EnvConfig(tactile=TactileConfig(enabled=True), obs=base_obs())
        env = Go2StairsEnv(stair=flat_spec(), cfg=cfg, seed=0)
        env.reset(seed=0)
        touched = 0
        for _ in range(60):
            _o, _r, term, trunc, info = env.step(np.zeros(12))
            if float(np.sum(info["tactile"]["force_n"])) > 0:
                touched += 1
            if term or trunc:
                break
        self.assertGreater(touched, 0)


class ForceVectorTest(unittest.TestCase):
    """三维力向量的方向约定（mj_contactForce 作用在 geom1 上）。

    站立时世界系合力的 z 分量必须 ≈ 体重；每足 z 分量必须与标量法向力一致。
    符号写反时这条会给出 −m·g（experience.md §23.74 坑 1）。
    """

    def test_world_force_sum_equals_weight(self) -> None:
        cfg = EnvConfig(obs=base_obs(), tactile=TactileConfig(enabled=True),
                        gait_prior=False)
        env = make_env(cfg, flat_spec())
        env.reset()
        info = {}
        for _ in range(40):
            _, _, term, trunc, info = env.step(np.zeros(12))
            if term or trunc:
                env.reset()
        tac = info["tactile"]
        fv = np.asarray(tac["force_vec"], dtype=float)
        fn = np.asarray(tac["force_n"], dtype=float)
        fvf = np.asarray(tac["force_vec_foot"], dtype=float)
        ref = float(env._tactile_sensor._force_ref)
        self.assertEqual(fv.shape, (4, 3))
        self.assertEqual(fvf.shape, (4, 3))
        self.assertAlmostEqual(float(fv[:, 2].sum()) / ref, 1.0, delta=0.10)
        np.testing.assert_allclose(fv[:, 2], fn, atol=2.0)


if __name__ == "__main__":
    unittest.main()
