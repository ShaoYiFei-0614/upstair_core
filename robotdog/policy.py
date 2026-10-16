"""把训练产物（PPO + ``*_vecnormalize.pkl``）包成一个可调用的策略。

``scripts/play.py`` 用它来推理。这里**同时**提供 PPO 与 ONNX 两条后端，因为它们
在 SDK 侧的使用方式完全不同：PPO 要一个能匹配观测维度的环境，ONNX 只要一个
``(1, obs_dim)`` 的 float32 张量。
"""

from __future__ import annotations

import base64
import json
import pickle
import zipfile
from dataclasses import asdict, dataclass, fields
from itertools import product
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from robotdog.env import CommandConfig, EnvConfig, ObsConfig, RewardConfig
from robotdog.gait import GaitParams
from robotdog.tools.terrain import StairSpec, demo_stairs_spec, flat_spec


class Policy(Protocol):
    """任何能给出动作的东西：PPO、ONNX、或者一个手写的开环步态。"""

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        ...


# --------------------------------------------------------------------------- #
# VecNormalize 统计
# --------------------------------------------------------------------------- #


@dataclass
class ObsNorm:
    """``VecNormalize.obs_rms`` 的只读拷贝。

    直接反 pickle 出的 ``VecNormalize`` 不能随便访问属性（SB3 的
    ``__getattr__`` 会递归枚举，多读一个不存在的名字就 ``RecursionError``），
    所以立刻拷出 mean/var/epsilon/clip、丢掉原对象，此后只用这份拷贝。
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
    if isinstance(env_kw.get("gait_stair"), dict):
        # 同样必须重建：JSON 往返后它是普通 dict，回放楼梯级时
        # ``_active_gait().freq`` 会当场 AttributeError（2026-10-15 实测踩到）。
        env_kw["gait_stair"] = GaitParams.from_dict(env_kw["gait_stair"])
    known = {f.name for f in fields(EnvConfig)}
    return EnvConfig(**{k: v for k, v in env_kw.items() if k in known})


def _zip_data(model_path: str | Path) -> Any:
    """读 SB3 ``.zip`` 的 ``data`` 条目并解成对象（失败返回 ``None``）。

    SB3 2.9 把 ``data`` 存成 JSON，并用自定义编码器把对象写进
    ``":serialized:"`` 的 base64 pickle；SB3 < 2.0 直接 pickle 整个 ``data``。
    """
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
    """从 checkpoint 里读策略网络隐藏层宽度（如 ``(512, 256)``）。

    重建模型必须用**训练时那一套**结构，否则 ``set_parameters`` 直接
    size mismatch。``policy_kwargs.net_arch`` 是普通 JSON，不需要解 pickle。
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
    """恢复能加载 ``model_path`` 的 :class:`EnvConfig`（三层退路，能力递减）。

    1. 同目录 ``<stem>_env.json`` —— 训练时逐 checkpoint 写的完整配置，**唯一
       精确来源**；
    2. 运行目录 ``summary.json`` —— 整条课程的最后一份配置，中途 checkpoint
       用它不精确；
    3. 按观测维度反推观测开关组合（成功时打印提示，调用方必须知道是推断值）。

    **反推在结构上不可能恢复不进观测的字段**（``gait_prior`` / ``action_scale``
    等）—— 用默认配置回放一个在另一套配置下训出的权重，会得到**相反**的评测
    结论（experience.md §20.7）。必须精确时用第 1 层。
    """
    model_path = Path(model_path)
    env_kw = _config_from_env_json(model_path)
    if env_kw is not None:
        return env_kw

    summary = model_path.parent / "summary.json"
    if summary.is_file():
        return env_config_from_summary(summary)

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

    每个 checkpoint 训练时的台阶高都不一样（只有最后一级是 15 cm），回放想默认
    用对地形必须逐权重记录 —— 只靠「最后一级」是错的。
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


def _obs_config_for_dim(total: int, history_len: int = 1) -> ObsConfig | None:
    """按总维度反推观测开关组合。

    机身系线速度与三轴指令是**结构固定块**（无开关，恒在），
    反推只枚举三个可选开关。``base_height`` 同样不在枚举里（构造期即被禁止）。
    枚举顺序让「本仓最常用」的组合优先命中 —— ``itertools.product`` 是最后
    一个位置翻转最快，所以把 ``gait_phase`` 放最前（恒 False 持续最久）、
    ``vel_filter`` 居中：51 维会先命中 vel+step 组合而非 gait+step。
    ``history_len`` 与帧大小是乘积关系、单靠总维数不可分离，默认取本仓最常用的 1。
    """
    flags = ("gait_phase", "vel_filter", "step_height")
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
