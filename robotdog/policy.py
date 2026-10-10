"""把训练产物（PPO + ``*_vecnormalize.pkl``）包成一个可调用的策略。

``scripts/play.py`` 用它来推理。这里**同时**提供 PPO 与 ONNX 两条后端，因为它们
在 SDK 侧的使用方式完全不同：PPO 要一个能匹配观测维度的环境，ONNX 只要一个
``(1, obs_dim)`` 的 float32 张量。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from robotdog.env import CommandConfig, EnvConfig, ObsConfig, RewardConfig
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
    if isinstance(env_kw.get("command"), dict):
        # 指令是嵌套 dataclass（含当前值 ``_now``）。JSON 往返后它是个普通
        # dict，直接塞给 EnvConfig 会让 ``env._command`` 变成 dict，
        # ``set_command_now`` 当场 AttributeError。必须重建。
        env_kw["command"] = _rebuild(CommandConfig, env_kw["command"])
    if isinstance(env_kw.get("gait"), dict):
        env_kw["gait"] = GaitParams.from_dict(env_kw["gait"])
    known = {f.name for f in fields(EnvConfig)}
    return EnvConfig(**{k: v for k, v in env_kw.items() if k in known})


def _zip_data(model_path: str | Path) -> Any:
    """读 SB3 ``.zip`` 的 ``data`` 条目并解成对象（失败返回 ``None``）。

    SB3 2.9 把 ``data`` 存成 JSON，并用自定义编码器把对象写进
    ``":serialized:"`` 的 base64 pickle；SB3 < 2.0 直接 pickle 整个 ``data``。
    """
    import base64
    import pickle
    import zipfile

    try:
        with zipfile.ZipFile(model_path) as zf:
            raw = zf.read("data")
    except Exception:  # noqa: BLE001 - 探测函数
        return None
    if raw[:1] == b"{":
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return None
    try:
        return pickle.loads(raw)
    except Exception:  # noqa: BLE001
        return None


def _deserialize(value: Any) -> Any:
    """把 SB3 的 ``":serialized:"`` 字段解回对象；普通值原样返回。"""
    import base64
    import pickle

    if isinstance(value, dict) and ":serialized:" in value:
        try:
            return pickle.loads(base64.b64decode(value[":serialized:"]))
        except Exception:  # noqa: BLE001
            return None
    return value


def observation_dim(model_path: str | Path) -> int | None:
    """从 SB3 的 ``.zip`` 里读出保存的观测维度（不解包、不加载模型）。"""

    def _shape(obj: Any) -> int | None:
        shape = getattr(obj, "shape", None)
        if shape is None and isinstance(obj, dict):
            shape = obj.get("shape")
        try:
            return int(shape[0]) if shape is not None else None
        except (TypeError, IndexError, ValueError):
            return None

    data = _zip_data(model_path)
    if data is None:
        return None
    try:
        return _shape(_deserialize(data.get("observation_space")))
    except Exception:  # noqa: BLE001
        return None


def policy_arch(model_path: str | Path) -> tuple[int, ...] | None:
    """从 checkpoint 里读出策略网络的隐藏层宽度，例如 ``(512, 256)``。

    重建模型时必须用**训练时那一套**结构：``load_policy`` 若按默认架构
    ``(512, 256)`` 建网再 ``set_parameters``，一个用 ``(64, 64)`` 训出来的权重会
    直接 ``RuntimeError: Error(s) in loading state_dict for ActorCriticPolicy: size
    mismatch for mlp_extractor.policy_net.0.weight: copying a param with shape
    torch.Size([64, 480]) ... the shape in current model is torch.Size([512, 480])``。
    ``policy_kwargs.net_arch`` 是普通 JSON（``{"pi": [...], "vf": [...]}``），不需要
    解 pickle。
    """
    data = _zip_data(model_path)
    if not isinstance(data, dict):
        return None
    kwargs = _deserialize(data.get("policy_kwargs"))
    if not isinstance(kwargs, dict):
        return None
    arch = kwargs.get("net_arch")
    if isinstance(arch, dict):
        arch = arch.get("pi") or arch.get("vf")
    if isinstance(arch, (list, tuple)) and arch:
        try:
            return tuple(int(w) for w in arch)
        except (TypeError, ValueError):
            return None
    return None


def env_config_for_model(model_path: str | Path) -> EnvConfig:
    """恢复能加载 ``model_path`` 的 :class:`EnvConfig`。

    首选模型同目录的 ``summary.json``（精确）。没有时按 ``.zip`` 里保存的观测
    维度反推是哪几个观测块（会打一行提示）。必须有这条退路，是因为
    三条路径，按精确度降序：模型同目录的 ``summary.json``（精确）→
    ``<stem>_env.json``（精确，``train.py`` 每个 checkpoint 各存一份）→
    按 ``.zip`` 里保存的观测维度反推是哪几个观测块（**启发式**，会打一行提示）。

    ``EnvConfig()`` 的默认观测是 10 帧 × 48 维 = 480 维，而本仓 ``config.yaml``
    用 ``history_len: 1`` 只有 48 维；直接用默认配置加载本仓权重会抛
    ``ValueError: Observation spaces do not match``，所以反推这条退路必须保留。

    反推**不可能**打开 ``obs.base_height``：那个开关在 :class:`ObsConfig` 里
    被硬性禁止（含地形真值，破坏盲走），所以 480/474 这类旧维度要么被无歧义
    还原，要么走 :meth:`_obs_config_for_dim` 返回 ``None`` 的报错分支。
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


def env_json_path(model_path: str | Path) -> Path:
    """``<模型>_env.json`` 的路径（与 ``<模型>_vecnormalize.pkl`` 同规矩）。"""
    model_path = Path(model_path)
    return model_path.with_name(model_path.stem + "_env.json")


def save_env_json(model_path: str | Path, cfg: EnvConfig, *,
                  step_height: float | None = None,
                  stage: str = "") -> Path:
    """把 ``cfg`` 写进 ``<模型>_env.json``，并在 report 段记下该权重的地形。

    **为什么需要它**：``summary.json`` 记的是**整条课程**的配置，而每个
    checkpoint 是在不同的台阶上训出来的。play 想默认用对地形，就必须知道这个
    权重当时的台阶高——只靠「最后一级」是错的（中间 checkpoint 全都会用错地形）。
    """
    payload = {
        "env_config": asdict(cfg),
        "obs_config": asdict(cfg.obs),
        "reward_config": asdict(cfg.reward),
        "report": {"step_height": step_height, "stage": stage},
    }
    target = env_json_path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    return target


def step_height_from_model(model_path: str | Path) -> float | None:
    """读 ``<模型>_env.json`` 的 ``report.step_height``（单位 m）；没有则 ``None``。"""
    cand = env_json_path(model_path)
    if not cand.is_file():
        return None
    try:
        report = json.loads(cand.read_text(encoding="utf-8")).get("report") or {}
    except Exception:  # noqa: BLE001 - 探测函数
        return None
    value = report.get("step_height")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _obs_config_for_dim(total: int, history_len: int = 10) -> ObsConfig | None:
    """反推是哪几个观测块拼出了 ``total`` 维。

    ``base_lin_vel_body``(3) / ``gait_phase``(2) / ``step_height``(1) / ``command``(2)
    四个开关组合不多，直接枚举。``base_lin_vel_body`` 与 ``command`` 默认开，所以
    枚举顺序把它们放前面，使最简配置优先命中。

    **``base_height`` 故意不在枚举里**：它在 :class:`ObsConfig` 的 ``__post_init__``
    里被硬性禁止（含地形真值，会破坏盲走），所以任何会打开它的组合都不可构造。
    归档 checkpoint 里那些 480/474 维的权重本来就与本仓 48 维观测不兼容，
    不必为它们保留反推能力。
    """
    from itertools import product

    flags = ("base_lin_vel_body", "gait_phase", "step_height", "command", "vel_filter")
    for combo in product((False, True), repeat=len(flags)):
        cand = ObsConfig(history_len=history_len, **dict(zip(flags, combo)))
        if cand.size == total:
            return cand
    return None


# --------------------------------------------------------------------------- #
# 楼梯几何
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
    "env_json_path",
    "observation_dim",
    "policy_arch",
    "save_env_json",
    "stair_from_height",
    "step_height_from_model",
]
