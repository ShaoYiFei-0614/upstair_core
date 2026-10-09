"""把训练产物（PPO + ``*_vecnormalize.pkl``）包成一个可调用的策略。

``scripts/play.py`` 用它来推理。这里**同时**提供 PPO 与 ONNX 两条后端，因为它们
在 SDK 侧的使用方式完全不同：PPO 要一个能匹配观测维度的环境，ONNX 只要一个
``(1, obs_dim)`` 的 float32 张量。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from robotdog.env import EnvConfig, ObsConfig, RewardConfig
from robotdog.tools.terrain import StairSpec


class Policy(Protocol):
    """任何能给出动作的东西：PPO、ONNX、或者一个手写的开环步态。"""

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        ...


# --------------------------------------------------------------------------- #
# VecNormalize 统计
# --------------------------------------------------------------------------- #


@dataclass
class ObsNorm:
    """``VecNormalize.obs_rms`` 的一份只读拷贝。

    直接 ``pickle.loads`` 出来的 ``VecNormalize`` 对象不能随便访问属性：SB3 的
    ``BaseVecEnv.__getattr__`` 会递归枚举所有属性，多读一个不存在的名字就
    ``RecursionError``。所以这里立刻把 mean/var/epsilon/clip 拷出来，然后把原对象
    丢掉（``load_obs_norm`` 里就是这么做的）。
    """

    mean: np.ndarray
    var: np.ndarray
    epsilon: float = 1e-8
    clip: float = 10.0

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        z = (np.asarray(obs, dtype=np.float32) - self.mean) / np.sqrt(
            self.var + self.epsilon
        )
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    @classmethod
    def from_path(cls, path: str | Path) -> ObsNorm:
        import pickle

        with open(path, "rb") as fh:
            raw = pickle.load(fh)
        try:
            mean = np.asarray(raw.obs_rms.mean, dtype=np.float64)
            var = np.asarray(raw.obs_rms.var, dtype=np.float64)
            eps = float(getattr(raw, "epsilon", 1e-8))
            clip = float(getattr(raw, "clip_obs", 10.0))
        finally:
            del raw
        return cls(mean=mean, var=var, epsilon=eps, clip=clip)


# --------------------------------------------------------------------------- #
# 配置恢复
# --------------------------------------------------------------------------- #

#: ``EnvConfig`` 里是元组而不是列表的字段（JSON 往返后全变 list，重建要转回）。
_TUPLE_FIELDS = frozenset({"kp", "kd", "gravity", "friction"})


def _rebuild(cfg_cls: Any, payload: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cfg_cls)}
    return cfg_cls(**{k: v for k, v in payload.items() if k in known})


def env_config_from_summary(summary_path: str | Path) -> EnvConfig:
    """从 ``runs/<name>/summary.json`` 恢复训练时使用的 :class:`EnvConfig`。"""
    from robotdog.gait import GaitParams

    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    env_kw = dict(data.get("env_config") or {})
    for key in _TUPLE_FIELDS:
        if isinstance(env_kw.get(key), list):
            env_kw[key] = tuple(env_kw[key])
    if data.get("obs_config"):
        env_kw["obs"] = _rebuild(ObsConfig, data["obs_config"])
    if data.get("reward_config"):
        env_kw["reward"] = _rebuild(RewardConfig, data["reward_config"])
    if isinstance(env_kw.get("gait"), dict):
        env_kw["gait"] = GaitParams.from_dict(env_kw["gait"])
    known = {f.name for f in fields(EnvConfig)}
    return EnvConfig(**{k: v for k, v in env_kw.items() if k in known})


def observation_dim(model_path: str | Path) -> int | None:
    """从 SB3 的 ``.zip`` 里读出保存的观测维度（不解包、不加载模型）。

    SB3 2.9 把 ``data`` 存成 JSON，并用自定义编码器把对象写进
    ``":serialized:"`` 的 base64 pickle；SB3 < 2.0 直接 pickle 整个 ``data``。
    """
    import base64
    import pickle
    import zipfile

    def _shape(obj: Any) -> int | None:
        shape = getattr(obj, "shape", None)
        if shape is None and isinstance(obj, dict):
            shape = obj.get("shape")
        try:
            return int(shape[0]) if shape is not None else None
        except (TypeError, IndexError, ValueError):
            return None

    try:
        with zipfile.ZipFile(model_path) as zf:
            raw = zf.read("data")
    except Exception:  # noqa: BLE001 - 探测函数
        return None
    if raw[:1] == b"{":
        try:
            space = json.loads(raw.decode("utf-8")).get("observation_space")
            if isinstance(space, dict) and ":serialized:" in space:
                space = pickle.loads(base64.b64decode(space[":serialized:"]))
            return _shape(space)
        except Exception:  # noqa: BLE001
            return None
    try:
        return _shape(pickle.loads(raw).get("observation_space"))
    except Exception:  # noqa: BLE001
        return None


def env_config_for_model(model_path: str | Path) -> EnvConfig:
    """恢复能加载 ``model_path`` 的 :class:`EnvConfig`。

    首选模型同目录的 ``summary.json``（精确）。没有时按 ``.zip`` 里保存的观测
    维度反推是哪几个观测块（会打一行提示）。必须有这条退路，是因为
    ``EnvConfig()`` 默认 460 维，而开了 ``--obs-base-height --obs-gait-phase``
    训练出来是 474 维，直接用默认配置加载会抛
    ``ValueError: Observation spaces do not match``。
    """
    model_path = Path(model_path)
    summary = model_path.parent / "summary.json"
    if summary.is_file():
        return env_config_from_summary(summary)

    env_kw = _config_from_env_json(model_path)
    if env_kw is not None:
        return env_kw

    dim = observation_dim(model_path)
    if dim is None:
        print(f"[配置] {model_path.name} 里读不到观测维度，退回 EnvConfig() 默认值")
        return EnvConfig()
    obs = _obs_config_for_dim(dim)
    if obs is None:
        raise SystemExit(
            f"模型观测维度 {dim} 无法用 EnvConfig 的观测块还原。"
            f"请把模型放回它的 runs/<name>/ 目录（读 summary.json），"
            f"或显式指定观测开关。"
        )
    print(f"[配置] 模型旁没有 summary.json，已从观测维度 {dim} 反推: "
          f"frame={obs.frame_size} history={obs.history_len}")
    return EnvConfig(obs=obs)


def _config_from_env_json(model_path: Path) -> EnvConfig | None:
    """``train.py`` 会把完整配置一起存成 ``<模型>_env.json``，优先读它。"""
    cand = model_path.with_name(model_path.stem + "_env.json")
    if not cand.is_file():
        return None
    return env_config_from_summary(cand)


def _obs_config_for_dim(total: int, history_len: int = 10) -> ObsConfig | None:
    """反推是哪几个观测块拼出了 ``total`` 维。

    ``base_lin_vel_body``(3) / ``base_height``(1) / ``gait_phase``(2) /
    ``step_height``(1) 四个开关组合不多，直接枚举。``base_lin_vel_body`` 默认开，
    所以枚举顺序把它放前面，使最简配置优先命中。
    """
    from itertools import product

    flags = ("base_lin_vel_body", "base_height", "gait_phase", "step_height")
    for combo in product((False, True), repeat=len(flags)):
        cand = ObsConfig(history_len=history_len, **dict(zip(flags, combo)))
        if cand.size == total:
            return cand
    return None


# --------------------------------------------------------------------------- #
# Stair geometry
# --------------------------------------------------------------------------- #


def stair_from_height(height_cm: float | None, *, tread: float = 0.50,
                      n_up: int = 8) -> StairSpec | None:
    """由台阶高度（cm）造一条楼梯；``None`` 表示沿用配置里的地形。"""
    from robotdog.tools.terrain import demo_stairs_spec, flat_spec

    if height_cm is None:
        return None
    if height_cm <= 0.0:
        return flat_spec(n_up=0, n_down=0, step_height=0.0, tread=tread)
    return demo_stairs_spec(
        step_height=height_cm / 100.0, tread=tread, n_up=n_up, top_landing=1.20
    )


__all__ = [
    "ObsNorm",
    "Policy",
    "env_config_for_model",
    "env_config_from_summary",
    "observation_dim",
    "stair_from_height",
]
