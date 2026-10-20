"""向量化环境装配的契约测试（unittest，不依赖 pytest）。

把多进程支持（``env.vec_env``）的三个接口钉成可执行契约：

* **默认 ``auto``**：训练（``n_envs > 1``，默认 8）解析成 ``SubprocVecEnv``；
  单环境（评估/遥控/加载权重）解析成 ``DummyVecEnv``；显式 ``dummy`` 可强制单进程；
* ``make_env_fn`` 是**模块级** ``functools.partial``，spawn 子进程能重新 import；
* 每个子环境拿到独立的 ``cfg`` 深拷贝（指令状态不串台）；
* ``build_ppo(device="cuda")`` 在没有 CUDA 时直接报错，而不是静默回退 CPU。

跑法::

    .venv\\Scripts\\python.exe -X utf8 tests\\test_vec_env.py -v

⚠ Windows 的 ``spawn`` 会重新 import 主模块。用 ``python -m unittest`` 跑时，
主模块是 ``unittest.__main__``，子进程会递归再跑一遍测试；因此本文件里的
``SubprocVecEnv`` 用例在 ``-m unittest`` 下会 skip，并提示改用上面的脚本方式。
全量 unittest discover 仍覆盖其余 6 条契约。
"""

from __future__ import annotations

import functools
import inspect
import sys
import unittest
from pathlib import Path

import cloudpickle
import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from robotdog.config import (  # noqa: E402
    config_to_env,
    load_config,
    vec_env as cfg_vec_env,
)
from robotdog.env import EnvConfig, ObsConfig  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from robotdog.training import (  # noqa: E402
    build_ppo,
    make_env_fn,
    make_vec_env,
    resolve_vec_env,
)
from robotdog.tools.terrain import demo_stairs_spec  # noqa: E402


def _cfg53() -> EnvConfig:
    """与 config.yaml 同维的观测配置（history_len=1 时 53 维）。"""
    return EnvConfig(obs=ObsConfig(history_len=1, gait_phase=True,
                                   step_height=True, vel_filter=True))


def _spawn_block_reason() -> str | None:
    """``-m unittest`` 下 spawn 会重导入 ``unittest.__main__``，不能在这里 spawn。"""
    if sys.platform != "win32":
        return None
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    if spec is not None and spec.name in ("unittest.__main__",):
        return ("spawn 会重导入 unittest.__main__；"
                "请用 python tests/test_vec_env.py 跑 subproc 用例")
    return None


def _is_pipe_unavailable(exc: BaseException) -> bool:
    """区分「环境不允许命名管道」与真正的装配 bug。"""
    if isinstance(exc, (PermissionError, BrokenPipeError, ConnectionResetError)):
        return True
    if isinstance(exc, OSError):
        code = getattr(exc, "winerror", None)
        return code in (5, 109, 232)
    return False


def _close(venv) -> None:
    if venv is None:
        return
    try:
        venv.close()
    except Exception:  # noqa: BLE001 - 清理失败不应掩盖断言结果
        pass


class DefaultPathTest(unittest.TestCase):
    def test_default_is_auto(self) -> None:
        # 默认 auto：训练（n_envs>1，默认 8）走 subproc，单环境走 dummy。
        self.assertEqual(make_vec_env.__kwdefaults__["vec_env"], "auto")
        self.assertEqual(make_vec_env.__kwdefaults__["start_method"], "spawn")
        import train as train_script  # scripts/train.py，有 main 保护
        sig = inspect.signature(train_script.Trainer.__init__)
        self.assertEqual(sig.parameters["vec_env"].default, "auto")

    def test_auto_resolution(self) -> None:
        self.assertEqual(resolve_vec_env(8, "auto"), "subproc")   # 训练默认 8 进程
        self.assertEqual(resolve_vec_env(2, "auto"), "subproc")
        self.assertEqual(resolve_vec_env(1, "auto"), "dummy")     # 评估/遥控
        self.assertEqual(resolve_vec_env(8, "dummy"), "dummy")    # 显式降级
        self.assertEqual(resolve_vec_env(1, "subproc"), "subproc")  # 显式覆盖
        self.assertEqual(resolve_vec_env(8, " SUBPROC "), "subproc")

    def test_unknown_vec_env_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            make_vec_env(None, 1, vec_env="thread")
        self.assertIn("必须是 'auto' / 'dummy' / 'subproc'", str(caught.exception))
        with self.assertRaises(ValueError):
            resolve_vec_env(8, "thread")

    def test_config_vec_env_default_and_invalid(self) -> None:
        self.assertEqual(cfg_vec_env({"env": {}}), "auto")
        self.assertEqual(cfg_vec_env({"env": {"vec_env": " SUBPROC "}}), "subproc")
        self.assertEqual(cfg_vec_env({"env": {"vec_env": "auto"}}), "auto")
        cfg = config_to_env(load_config("config.yaml"))
        self.assertEqual(cfg.obs.size, 53)
        self.assertFalse(hasattr(cfg, "vec_env"))
        with self.assertRaises(SystemExit) as caught:
            cfg_vec_env({"env": {"vec_env": "thread"}})
        self.assertIn("必须是 'auto' / 'dummy' / 'subproc'", str(caught.exception))


class EnvFactoryTest(unittest.TestCase):
    def test_make_env_fn_is_module_level_partial(self) -> None:
        f = make_env_fn(None, None, 0)
        self.assertIsInstance(f, functools.partial)
        self.assertIs(f.func, __import__("robotdog.training", fromlist=["_make_env"])._make_env)
        self.assertIsNone(f.func.__closure__)
        self.assertEqual(f.func.__module__, "robotdog.training")

    def test_cloudpickle_roundtrip_builds_env(self) -> None:
        cfg = _cfg53()
        f = cloudpickle.loads(cloudpickle.dumps(make_env_fn(None, cfg, 0)))
        env = f()
        try:
            env.reset()
            self.assertEqual(env.observation_space.shape, (53,))
            self.assertEqual(env.action_space.shape, (12,))
        finally:
            env.close()

    def test_each_env_gets_own_cfg(self) -> None:
        cfg = EnvConfig()
        e0 = make_env_fn(demo_stairs_spec(0.06), cfg, seed=1)()
        e1 = make_env_fn(demo_stairs_spec(0.06), cfg, seed=2)()
        try:
            e0.reset()
            e1.reset()
            self.assertIsNot(e0.cfg, cfg)
            self.assertIsNot(e0.cfg, e1.cfg)
            before = list(e1._command)
            e0.set_command_now(vx=0.9, vy=0.2, yaw_rate=0.4, hold=True)
            self.assertEqual(list(e1._command), before)
        finally:
            e0.close()
            e1.close()


class AutoImplTest(unittest.TestCase):
    """``auto`` 的两种解析必须落到正确的实现类。"""

    def test_auto_picks_impl_by_n_envs(self) -> None:
        reason = _spawn_block_reason()
        if reason is not None:
            self.skipTest(reason)
        one = two = None
        try:
            one = make_vec_env(None, n_envs=1, cfg=_cfg53(), seed=0,
                                normalize=False, monitor=False)
            self.assertIsInstance(one, DummyVecEnv)
            try:
                two = make_vec_env(None, n_envs=2, cfg=_cfg53(), seed=0,
                                    normalize=False, monitor=False)
            except Exception as exc:  # noqa: BLE001
                if _is_pipe_unavailable(exc):
                    self.skipTest(f"当前环境不允许子进程管道：{exc!r}")
                raise
            self.assertIsInstance(two, SubprocVecEnv)
        finally:
            _close(one)
            _close(two)


class VecEnvShapeTest(unittest.TestCase):
    def test_dummy_and_subproc_shapes(self) -> None:
        reason = _spawn_block_reason()
        if reason is not None:
            self.skipTest(reason)
        for arm in ("dummy", "subproc"):
            venv = None
            try:
                try:
                    venv = make_vec_env(
                        demo_stairs_spec(0.06), n_envs=2,
                        cfg=_cfg53(), seed=0, vec_env=arm)
                except Exception as exc:  # noqa: BLE001
                    # 只把「构造期命名管道被拒绝」当环境限制；reset/step
                    # 之后的 BrokenPipe 必须暴露成真实装配缺陷。
                    if arm == "subproc" and _is_pipe_unavailable(exc):
                        self.skipTest(f"当前环境不允许子进程管道：{exc!r}")
                    raise
                obs = venv.reset()
                self.assertEqual(obs.shape, (2, 53), arm)
                self.assertEqual(obs.dtype, np.float32, arm)
                out = venv.step(np.zeros((2, 12), dtype=np.float32))
                self.assertEqual(out[0].shape, (2, 53), arm)
                self.assertEqual(out[1].shape, (2,), arm)
                self.assertTrue(np.isfinite(out[0]).all(), arm)
                self.assertTrue(np.isfinite(out[1]).all(), arm)
            finally:
                _close(venv)

    def test_live_env_cfg_readable_from_vecenv(self) -> None:
        reason = _spawn_block_reason()
        if reason is not None:
            self.skipTest(reason)
        for arm in ("dummy", "subproc"):
            venv = None
            try:
                try:
                    venv = make_vec_env(
                        demo_stairs_spec(0.06), n_envs=2,
                        cfg=_cfg53(), seed=0, vec_env=arm)
                except Exception as exc:  # noqa: BLE001
                    if arm == "subproc" and _is_pipe_unavailable(exc):
                        self.skipTest(f"当前环境不允许子进程管道：{exc!r}")
                    raise
                cfgs = venv.get_attr("cfg")
                self.assertEqual(len(cfgs), 2, arm)
                self.assertEqual(cfgs[0].obs.size, 53, arm)

                # 课程下发契约：广播命令两个 worker 都变；indices=[0] 的
                # 奖励覆盖只能改 0 号，1 号保持基线（证明 cfg 未共享）。
                base_foot_lift = [c.reward.foot_lift for c in cfgs]
                venv.env_method("set_commands", vy_max=0.11, yaw_max=0.33)
                cmds = venv.get_attr("cfg")
                self.assertEqual([c.command.vy_max for c in cmds],
                                 [0.11, 0.11], arm)
                self.assertEqual([c.command.yaw_max for c in cmds],
                                 [0.33, 0.33], arm)
                venv.env_method("set_reward_overrides",
                                {"foot_lift": 0.77}, indices=[0])
                overridden = venv.get_attr("cfg")
                self.assertAlmostEqual(overridden[0].reward.foot_lift, 0.77,
                                       places=12, msg=arm)
                self.assertEqual(overridden[1].reward.foot_lift,
                                 base_foot_lift[1], arm)
            finally:
                _close(venv)


class PpoDeviceTest(unittest.TestCase):
    def test_build_ppo_rejects_cuda_without_cuda(self) -> None:
        if th.cuda.is_available():
            self.skipTest("当前环境有 CUDA，跳过「无 CUDA 必须报错」用例")
        venv = make_vec_env(None, n_envs=1, cfg=EnvConfig(), seed=0)
        try:
            with self.assertRaisesRegex(
                    RuntimeError, r"torch\.cuda\.is_available\(\) 为 False"):
                build_ppo(venv, device="cuda", verbose=0)
            model = build_ppo(venv, device="cpu", verbose=0)
            self.assertEqual(model.device.type, "cpu")
        finally:
            _close(venv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
