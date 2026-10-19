"""非对称 actor-critic（critic 特权观测）契约测试（unittest，不依赖 pytest）。

对应变更简报 .tmp/change_asym_ac.md 与 doc/env.md / doc/training.md。钉住三件事：

1. 前向兼容：开关默认关时观测维数/布局与旧实现逐位一致，策略类仍是 SB3
   MlpPolicy；旧 53 维 checkpoint 仍能 load_policy + predict。
2. actor 看不到特权块：actor 网络第一层输入维数 = actor_size，只有 critic
   的第一层吃全量（含 4 维足端接触）。
3. 特权块本身可用：每帧末尾 4 维 ∈ {0,1}、随步变化、info 里报同一份。

跑法：.venv\\Scripts\\python.exe -m unittest tests.test_asymmetric -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th

from robotdog.asymmetric import AsymmetricActorCriticPolicy, AsymmetricMlpExtractor
from robotdog.env import EnvConfig, Go2StairsEnv, ObsConfig
from robotdog.policy import env_config_for_model, observation_dim
from robotdog.tools.terrain import flat_spec
from robotdog.training import (
    build_ppo,
    load_policy,
    make_vec_env,
    restore_vecnormalize_compat,
    set_parameters_compat,
)

#: config.yaml 当前的观测开关组合（53 维）。
BASE_OBS = dict(history_len=1, gait_phase=True, step_height=True, vel_filter=True)


def _env(cfg: EnvConfig | None = None, seed: int = 0) -> Go2StairsEnv:
    return Go2StairsEnv(stair=flat_spec(), cfg=cfg, seed=seed)


def _priv_cfg(history_len: int = 1) -> EnvConfig:
    obs = ObsConfig(**{**BASE_OBS, "history_len": history_len,
                       "privileged_foot_contact": True})
    return EnvConfig(obs=obs, command_wander_std=0.0)


class PrivilegedObsTest(unittest.TestCase):
    """观测布局：默认 53 维；开启后帧末尾 +4 维，但 actor 维数不变。"""

    def test_default_off_is_old_layout(self) -> None:
        cfg = EnvConfig(obs=ObsConfig(**BASE_OBS))
        self.assertEqual(cfg.obs.size, 53)
        self.assertEqual(cfg.obs.privileged_frame_size, 0)
        self.assertEqual(cfg.obs.actor_frame_size, 53)
        self.assertEqual(cfg.obs.actor_size, 53)
        self.assertEqual(cfg.obs.actor_index, tuple(range(53)))
        env = _env(cfg)
        self.assertEqual(env.observation_space.shape, (53,))
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (53,))
        _, _, _, _, info = env.step(np.zeros(12))
        # info 里始终带逐足接触（只读诊断，不参与奖励/终止）
        self.assertEqual(len(info["foot_contact"]), 4)

    def test_enabled_appends_privileged_block(self) -> None:
        cfg_on = _priv_cfg()
        cfg_off = EnvConfig(obs=ObsConfig(**BASE_OBS), command_wander_std=0.0)
        self.assertEqual(cfg_on.obs.size, 57)
        self.assertEqual(cfg_on.obs.frame_size, 57)
        self.assertEqual(cfg_on.obs.privileged_frame_size, 4)
        self.assertEqual(cfg_on.obs.actor_frame_size, 53)
        self.assertEqual(cfg_on.obs.actor_size, 53)
        env_on, env_off = _env(cfg_on), _env(cfg_off)
        self.assertEqual(env_on.observation_space.shape, (57,))
        obs_on, _ = env_on.reset(seed=0)
        obs_off, _ = env_off.reset(seed=0)
        self.assertEqual(obs_on.shape, (57,))
        # 追加而非插入：前 53 维与关闭特权块时逐位相同（同一 seed ⇒ 同一状态）
        np.testing.assert_allclose(obs_on[:53], obs_off, atol=1e-6)
        priv = obs_on[53:]
        self.assertEqual(priv.shape, (4,))
        self.assertTrue(np.all((priv == 0.0) | (priv == 1.0)))

    def test_contact_channel_changes_during_walk(self) -> None:
        env = _env(_priv_cfg())
        env.reset(seed=0)
        masks = []
        for _ in range(40):
            obs, _, term, trunc, info = env.step(np.zeros(12))
            # info 报的接触必须与观测末尾 4 维同源（同一个 _foot_contact_now）
            np.testing.assert_allclose(obs[-4:], np.asarray(info["foot_contact"]),
                                       atol=1e-6)
            masks.append(tuple(info["foot_contact"]))
            if term or trunc:
                env.reset()
        self.assertTrue(all(len(m) == 4 for m in masks))
        self.assertTrue(all(v in (0.0, 1.0) for m in masks for v in m))
        # 先验步态下 40 步内必出现「有脚离地」的帧（否则说明没接上接触扫描）
        self.assertTrue(any(v == 0.0 for m in masks for v in m))

    def test_history_strip_index(self) -> None:
        cfg = _priv_cfg(history_len=2)
        self.assertEqual(cfg.obs.frame_size, 57)
        self.assertEqual(cfg.obs.size, 114)
        self.assertEqual(cfg.obs.actor_size, 106)
        idx = cfg.obs.actor_index
        self.assertEqual(len(idx), 106)
        # 帧优先条带：第 0 帧取 0..52、第 1 帧取 57..109
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[52], 52)
        self.assertEqual(idx[53], 57)
        self.assertEqual(idx[105], 109)
        got = np.arange(114)[list(idx)]
        expected = np.concatenate([np.arange(53), np.arange(57, 110)])
        np.testing.assert_array_equal(got, expected)


class AsymmetricPolicyTest(unittest.TestCase):
    """策略层：非对称类只在开户时构造；actor/critic 首层维数正确、能学能存。"""

    def _venv(self, cfg: EnvConfig, n_envs: int = 2):
        return make_vec_env(flat_spec(), n_envs=n_envs, cfg=cfg, seed=0,
                            normalize=False, monitor=False)

    def test_default_path_is_unchanged(self) -> None:
        cfg = EnvConfig(obs=ObsConfig(**BASE_OBS))
        venv = self._venv(cfg)
        try:
            model = build_ppo(venv, seed=0, net_arch=(32,), n_steps=16,
                              batch_size=16, n_epochs=1, verbose=0)
            self.assertNotIsInstance(model.policy, AsymmetricActorCriticPolicy)
            self.assertIsNone(getattr(model.policy, "_actor_obs_index", None))
        finally:
            venv.close()

    def test_asymmetric_policy_dims_and_learn(self) -> None:
        cfg = _priv_cfg()
        venv = self._venv(cfg)
        try:
            model = build_ppo(venv, seed=0, net_arch=(32, 16), n_steps=32,
                              batch_size=32, n_epochs=1, verbose=0,
                              actor_obs_index=cfg.obs.actor_index)
            pol = model.policy
            self.assertIsInstance(pol, AsymmetricActorCriticPolicy)
            self.assertIsInstance(pol.mlp_extractor, AsymmetricMlpExtractor)
            # actor 第一层只吃 actor 维；critic 第一层吃全量（含特权块）
            self.assertEqual(pol.mlp_extractor.policy_net[0].in_features,
                             cfg.obs.actor_size)
            self.assertEqual(pol.mlp_extractor.value_net[0].in_features,
                             cfg.obs.size)
            obs = venv.reset()
            action, _ = model.predict(obs, deterministic=True)
            self.assertEqual(action.shape, (2, 12))
            model.learn(total_timesteps=64)  # 一次完整 update（缓冲/前向/反传）
        finally:
            venv.close()

    def test_save_load_roundtrip(self) -> None:
        cfg = _priv_cfg()
        venv = self._venv(cfg)
        try:
            model = build_ppo(venv, seed=0, net_arch=(32, 16), n_steps=16,
                              batch_size=16, n_epochs=1, verbose=0,
                              actor_obs_index=cfg.obs.actor_index)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "m.zip"
                model.save(str(path))
                venv2 = self._venv(cfg)
                try:
                    model2 = build_ppo(venv2, seed=0, net_arch=(32, 16),
                                       n_steps=16, batch_size=16, n_epochs=1,
                                       verbose=0,
                                       actor_obs_index=cfg.obs.actor_index)
                    model2.set_parameters(str(path))  # 非持久 buffer 不进 state_dict
                    obs = venv2.reset()
                    a1, _ = model.predict(obs, deterministic=True)
                    a2, _ = model2.predict(obs, deterministic=True)
                    np.testing.assert_allclose(a1, a2, atol=1e-6)
                finally:
                    venv2.close()
        finally:
            venv.close()


class VecNormalizeCompatTest(unittest.TestCase):
    """57 维观测过 VecNormalize（obs_rms/clip_obs 形状）必须正常。"""

    def test_vecnormalize_57_dims(self) -> None:
        cfg = _priv_cfg()
        venv = make_vec_env(flat_spec(), n_envs=2, cfg=cfg, seed=0,
                            normalize=True, monitor=True, norm_reward=False)
        try:
            obs = venv.reset()
            self.assertEqual(obs.shape, (2, 57))
            model = build_ppo(venv, seed=0, net_arch=(32,), n_steps=16,
                              batch_size=16, n_epochs=1, verbose=0,
                              actor_obs_index=cfg.obs.actor_index)
            action, _ = model.predict(obs, deterministic=True)
            self.assertEqual(action.shape, (2, 12))
        finally:
            venv.close()


class AsymmetricLoadChainTest(unittest.TestCase):
    """冒烟 run 的 57 维 checkpoint：_env.json → load_policy → predict。

    这是 reviewer 指出的「自描述加载链路无自动化测试」的补丁：验证
    `env_config_for_model` 能还原 57 维 cfg 且 `load_policy` 用非对称类重建。
    """

    MODEL = Path("runs/smoke_asym_ac_1/checkpoints/stair3_final.zip")

    def test_env_json_to_load_policy(self) -> None:
        if not self.MODEL.is_file():
            self.skipTest(f"缺少冒烟权重 {self.MODEL}")
        cfg = env_config_for_model(self.MODEL)
        self.assertEqual(cfg.obs.size, 57)
        self.assertEqual(cfg.obs.frame_size, 57)
        self.assertEqual(cfg.obs.privileged_frame_size, 4)
        self.assertEqual(cfg.obs.actor_size, 53)
        model, _ = load_policy(self.MODEL)
        self.assertIsInstance(model.policy, AsymmetricActorCriticPolicy)
        pol = model.policy
        self.assertEqual(pol.mlp_extractor.policy_net[0].in_features, 53)
        self.assertEqual(pol.mlp_extractor.value_net[0].in_features, 57)
        action, _ = model.predict(np.zeros(57, dtype=np.float32),
                                  deterministic=True)
        self.assertEqual(action.shape, (12,))
        self.assertTrue(np.all(np.isfinite(action)))


class WarmStartCompatTest(unittest.TestCase):
    """对称 AC（53 维）→ 非对称 AC（57 维）的 warm-start 兼容迁移。

    语义：actor 权重原样搬（两边 actor 都只看 53 维）；critic 首层左侧旧列搬、
    新增 4 个特权列补零 ⇒ 初始 value 与旧权重逐位一致。观测归一化统计按
    actor_index 放回、新增维取 mean 0 / var 1。
    """

    def _venv(self, cfg: EnvConfig):
        return make_vec_env(flat_spec(), n_envs=2, cfg=cfg, seed=0,
                            normalize=True, monitor=True, norm_reward=False)

    def test_expand_state_dict_and_obs_stats(self) -> None:
        cfg_old = EnvConfig(obs=ObsConfig(**BASE_OBS))
        cfg_new = _priv_cfg()
        venv_old = self._venv(cfg_old)
        venv_new = self._venv(cfg_new)
        try:
            old = build_ppo(venv_old, seed=0, net_arch=(32, 16), n_steps=16,
                            batch_size=16, n_epochs=1, verbose=0)
            new = build_ppo(venv_new, seed=0, net_arch=(32, 16), n_steps=16,
                            batch_size=16, n_epochs=1, verbose=0,
                            actor_obs_index=cfg_new.obs.actor_index)
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp) / "old"
                old.save(str(base))
                # 观测统计打上与维度有关的假值，验证扩维映射
                venv_old.obs_rms.mean[:] = np.arange(53, dtype=np.float64)
                venv_old.obs_rms.var[:] = np.arange(53, dtype=np.float64) + 1.0
                # count 模拟「已经训过很多步」的统计量；B1 的判据就靠它
                venv_old.obs_rms.count = 40_000.0
                venv_old.ret_rms.count = 12_345.0
                npath = Path(str(base) + "_vecnormalize.pkl")
                venv_old.save(str(npath))

                note = set_parameters_compat(new, str(base) + ".zip",
                                             old_obs_dim=53, new_obs_dim=57,
                                             actor_index=cfg_new.obs.actor_index)
                self.assertEqual(note, "53->57")
                self.assertTrue(th.equal(
                    new.policy.mlp_extractor.policy_net[0].weight,
                    old.policy.mlp_extractor.policy_net[0].weight))
                w_new = new.policy.mlp_extractor.value_net[0].weight
                w_old = old.policy.mlp_extractor.value_net[0].weight
                self.assertEqual(tuple(w_new.shape), (32, 57))
                self.assertTrue(th.equal(w_new[:, :53], w_old))
                self.assertTrue(th.equal(w_new[:, 53:], th.zeros_like(w_new[:, 53:])))
                v_new = new.policy.predict_values(th.zeros(1, 57))
                v_old = old.policy.predict_values(th.zeros(1, 53))
                np.testing.assert_allclose(v_new.detach().numpy(),
                                           v_old.detach().numpy(), atol=1e-6)

                restored = restore_vecnormalize_compat(
                    venv_new, npath, old_obs_dim=53, new_obs_dim=57,
                    actor_index=cfg_new.obs.actor_index)
                self.assertIsNotNone(restored)
                idx = list(cfg_new.obs.actor_index)
                np.testing.assert_allclose(restored.obs_rms.mean[idx],
                                           np.arange(53, dtype=np.float64))
                np.testing.assert_allclose(restored.obs_rms.var[idx],
                                           np.arange(53, dtype=np.float64) + 1.0)
                np.testing.assert_allclose(restored.obs_rms.mean[53:], 0.0)
                np.testing.assert_allclose(restored.obs_rms.var[53:], 1.0)
                # B1：count 必须一起恢复，否则第一次 reset/step 就把旧统计整份覆盖
                self.assertGreaterEqual(restored.obs_rms.count, 40_000.0)
                self.assertEqual(restored.ret_rms.count, 12_345.0)
                venv_new.reset()
                venv_new.step(np.zeros((2, 12)))
                self.assertGreaterEqual(restored.obs_rms.count, 40_000.0)
                np.testing.assert_allclose(restored.obs_rms.mean[idx],
                                           np.arange(53, dtype=np.float64),
                                           atol=0.05)
                # policy_kwargs 必须带 actor_obs_index（checkpoint 自描述）
                saved = Path(tmp) / "new.zip"
                new.save(str(saved))
                import json as _json
                import zipfile as _zipfile
                with _zipfile.ZipFile(saved) as _zf:
                    data = _json.loads(_zf.read("data").decode("utf-8"))
                self.assertIn("actor_obs_index", _json.dumps(data))
                self.assertEqual(observation_dim(saved), 57)
        finally:
            venv_old.close()
            venv_new.close()


class WarmStartFailFastTest(unittest.TestCase):
    """迁移失败必须响亮（不静默降级）。"""

    def test_arch_mismatch_raises(self) -> None:
        venv_old = make_vec_env(flat_spec(), n_envs=1,
                                cfg=EnvConfig(obs=ObsConfig(**BASE_OBS)),
                                seed=0, normalize=False, monitor=False)
        venv_new = make_vec_env(flat_spec(), n_envs=1, cfg=_priv_cfg(), seed=0,
                                normalize=False, monitor=False)
        try:
            old = build_ppo(venv_old, seed=0, net_arch=(32, 16), n_steps=16,
                            batch_size=16, n_epochs=1, verbose=0)
            new = build_ppo(venv_new, seed=0, net_arch=(64, 32), n_steps=16,
                            batch_size=16, n_epochs=1, verbose=0,
                            actor_obs_index=_priv_cfg().obs.actor_index)
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp) / "old"
                old.save(str(base))
                with self.assertRaises(ValueError):
                    set_parameters_compat(new, str(base) + ".zip",
                                          old_obs_dim=53, new_obs_dim=57)
        finally:
            venv_old.close()
            venv_new.close()

    def test_narrowing_raises(self) -> None:
        cfg = _priv_cfg()
        venv = make_vec_env(flat_spec(), n_envs=1, cfg=cfg, seed=0,
                            normalize=False, monitor=False)
        try:
            model = build_ppo(venv, seed=0, net_arch=(32, 16), n_steps=16,
                              batch_size=16, n_epochs=1, verbose=0,
                              actor_obs_index=cfg.obs.actor_index)
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp) / "m57"
                model.save(str(base))
                with self.assertRaises(ValueError):
                    set_parameters_compat(model, str(base) + ".zip",
                                          old_obs_dim=57, new_obs_dim=53)
        finally:
            venv.close()

    def test_history_len2_migration_uses_actor_index(self) -> None:
        cfg53 = EnvConfig(obs=ObsConfig(**{**BASE_OBS, "history_len": 2}))
        cfg57 = EnvConfig(obs=ObsConfig(**{**BASE_OBS, "history_len": 2,
                                          "privileged_foot_contact": True}))
        self.assertEqual(cfg53.obs.size, 106)
        self.assertEqual(cfg57.obs.size, 114)
        venv_old = make_vec_env(flat_spec(), n_envs=1, cfg=cfg53, seed=0,
                                normalize=False, monitor=False)
        venv_new = make_vec_env(flat_spec(), n_envs=1, cfg=cfg57, seed=0,
                                normalize=False, monitor=False)
        try:
            old = build_ppo(venv_old, seed=0, net_arch=(32, 16), n_steps=16,
                            batch_size=16, n_epochs=1, verbose=0)
            new = build_ppo(venv_new, seed=0, net_arch=(32, 16), n_steps=16,
                            batch_size=16, n_epochs=1, verbose=0,
                            actor_obs_index=cfg57.obs.actor_index)
            idx = list(cfg57.obs.actor_index)
            self.assertEqual(idx[53], 57)   # 第 2 帧的 actor 维从 57 开始
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp) / "old"
                old.save(str(base))
                note = set_parameters_compat(
                    new, str(base) + ".zip", old_obs_dim=106, new_obs_dim=114,
                    actor_index=cfg57.obs.actor_index)
                self.assertEqual(note, "106->114")
                w_new = new.policy.mlp_extractor.value_net[0].weight
                w_old = old.policy.mlp_extractor.value_net[0].weight
                self.assertTrue(th.equal(w_new[:, idx], w_old))
                mask = np.zeros(114, dtype=bool)
                mask[idx] = True
                self.assertTrue(th.equal(w_new[:, ~th.as_tensor(mask)],
                                        th.zeros_like(w_new[:, ~th.as_tensor(mask)])))
                self.assertTrue(th.equal(
                    new.policy.mlp_extractor.policy_net[0].weight,
                    old.policy.mlp_extractor.policy_net[0].weight))
        finally:
            venv_old.close()
            venv_new.close()

    def test_actor_index_length_mismatch_raises(self) -> None:
        cfg = _priv_cfg()
        venv = make_vec_env(flat_spec(), n_envs=1, cfg=cfg, seed=0,
                            normalize=True, monitor=True, norm_reward=False)
        venv_old = make_vec_env(flat_spec(), n_envs=1,
                                cfg=EnvConfig(obs=ObsConfig(**BASE_OBS)), seed=0,
                                normalize=True, monitor=True, norm_reward=False)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "v.pkl"
                venv_old.save(str(path))
                with self.assertRaises(ValueError):
                    restore_vecnormalize_compat(venv, path, old_obs_dim=53,
                                                new_obs_dim=57,
                                                actor_index=list(range(52)))
        finally:
            venv.close()
            venv_old.close()


class LegacyCheckpointTest(unittest.TestCase):
    """旧 53 维对称 AC 权重在默认配置下照旧可加载。"""

    MODEL = Path("runs/ab_6_r1/model_final.zip")

    def test_old_checkpoint_still_loads(self) -> None:
        if not self.MODEL.is_file():
            self.skipTest(f"缺少历史权重 {self.MODEL}")
        cfg = env_config_for_model(self.MODEL)
        self.assertEqual(cfg.obs.size, 53)
        self.assertEqual(cfg.obs.privileged_frame_size, 0)
        model, _ = load_policy(self.MODEL)
        self.assertNotIsInstance(model.policy, AsymmetricActorCriticPolicy)
        action, _ = model.predict(np.zeros(53, dtype=np.float32), deterministic=True)
        self.assertEqual(action.shape, (12,))


if __name__ == "__main__":
    unittest.main()
