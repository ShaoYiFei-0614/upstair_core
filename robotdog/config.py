"""读取 ``config.yaml``，把它变成 :class:`~robotdog.env.EnvConfig` 与课程阶段表。

**为什么有这个模块**：训练与播放的超参集中在仓根 ``config.yaml`` 一处，脚本不再
暴露一堆命令行开关。这里只做三件事——读 YAML、按代码默认值补全缺失的键、把已知
的键写进 dataclass 字段；不认识的键直接报错（打错一个字母要立刻看得见，而不是被
静默忽略）。

用法::

    from robotdog.config import load_config, config_to_env, config_to_stages

    raw = load_config()                       # 默认 <仓根>/config.yaml
    cfg = config_to_env(raw)                  # EnvConfig（含 trot 先验）
    stages = config_to_stages(raw)            # 六级 StageSpec

配置文件的完整键与含义见仓根 ``config.yaml`` 的注释，以及 ``README.md`` §4。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from robotdog.curriculum import (
    DEFAULT_N_UP,
    DEFAULT_TREAD,
    STAGE_HEIGHTS_CM,
    StageSpec,
)
from robotdog.env import EnvConfig
from robotdog.gait import GaitParams, load_gait
from robotdog.tools import paths
from robotdog.tools.terrain import demo_stairs_spec, flat_spec

#: 默认配置文件位置（仓根）。
DEFAULT_CONFIG_PATH: Path = paths.WORKSPACE / "config.yaml"

#: 配置文件省略 ``curriculum.timesteps`` 时的每级预算（平地 : 台阶 = 1 : 2.5）。
DEFAULT_STAGE_TIMESTEPS: list[int] = [100_000] + [250_000] * (len(STAGE_HEIGHTS_CM) - 1)

#: 配置文件省略 ``curriculum.learning_rate`` 时的统一学习率。
DEFAULT_LR: float = 1.5e-4


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """读 ``config.yaml``；``path=None`` 时用仓根的默认文件。

    文件不存在、不是合法 YAML、顶层不是映射，都会 ``SystemExit`` 并给出中文原因。
    """
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not target.is_file():
        raise SystemExit(f"找不到配置文件：{target}")
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:                      # 语法错误
        raise SystemExit(f"配置文件解析失败（{target}）：\n{exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SystemExit(f"配置文件顶层必须是映射，实际是 {type(data).__name__}")
    return data


def _section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    """取出一个子段；缺失 = 空段（全部走默认值）。"""
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SystemExit(f"config.yaml 的 {name}: 必须是映射（键值对）")
    return dict(value)


def _timesteps(value: Any, n_stages: int, *, where: str) -> list[int]:
    """把 ``timesteps`` 归一化成 ``n_stages`` 个正整数。

    允许给一个标量（每级相同）或一个列表。列表长度不等时按比例重新采样到
    ``n_stages``（保持总量，余数补给最难的一级），这样改课程级数时不必手改预算。
    """
    if isinstance(value, (int, float)):
        raw = float(value)
        if raw < 0:
            raise SystemExit(f"{where}: 步数不能为负（{value}）")
        return [max(1, int(round(raw)))] * n_stages
    if not isinstance(value, (list, tuple)) or not value:
        raise SystemExit(f"{where}: 期望数字或非空列表，实际是 {type(value).__name__}")

    numbers = [float(v) for v in value]
    if any(v < 0 for v in numbers):
        raise SystemExit(f"{where}: 步数不能为负（{numbers}）")
    if len(numbers) == n_stages:
        resampled = numbers
    else:
        total = sum(numbers)
        if total <= 0:
            raise SystemExit(f"{where}: 之和必须为正")
        unit = total / n_stages                      # 均值，保持总量
        resampled = [w * unit for w in _spread(numbers, n_stages)]

    out = [max(1, int(round(v))) for v in resampled]
    out[-1] += max(0, int(round(sum(numbers))) - sum(out))   # 余数给最难的一级
    return out


def _spread(weights: list[float], n: int) -> list[float]:
    """把 ``weights`` 重新采样成 ``n`` 个权重（线性插值，保持总量）。"""
    if n == 1:
        return [sum(weights)]
    m = len(weights)
    if m == n:
        return weights
    out = []
    for i in range(n):
        pos = i * (m - 1) / (n - 1)
        lo = int(pos)
        hi = min(lo + 1, m - 1)
        frac = pos - lo
        out.append(weights[lo] * (1 - frac) + weights[hi] * frac)
    return out


def _floats(value: Any, n_stages: int, *, where: str, default: float) -> list[float]:
    """把 ``learning_rate`` 之类的标量/列表归一化成 ``n_stages`` 个浮点。"""
    if value is None:
        return [default] * n_stages
    if isinstance(value, (int, float)):
        return [float(value)] * n_stages
    if isinstance(value, (list, tuple)) and value:
        values = [float(v) for v in value]
        if len(values) == n_stages:
            return values
        if len(values) == 1:
            return values * n_stages
        return _spread(values, n_stages)
    raise SystemExit(f"{where}: 期望数字或列表，实际是 {type(value).__name__}")


# --------------------------------------------------------------------------- #
# 环境配置
# --------------------------------------------------------------------------- #


def config_to_env(raw: Mapping[str, Any]) -> EnvConfig:
    """把 ``env:`` + ``reward:`` 两段写进 :class:`EnvConfig`。

    没写的键保持 dataclass 默认值，所以配置文件只需要列出你想改的项。
    未知的键（打错的字段名）会报错并列出现有字段。
    """
    env = _section(raw, "env")
    obs_raw = env.pop("obs", None)       # 嵌套子段，单独处理
    env.pop("n_envs", None)              # 属于运行参数，不属于 EnvConfig
    gait_file = env.pop("gait_file", None)
    cfg = EnvConfig()

    _apply(cfg, env, where="env")
    _apply(cfg.reward, _section(raw, "reward"), where="reward")

    if obs_raw is not None:
        if not isinstance(obs_raw, Mapping):
            raise SystemExit("config.yaml 的 env.obs 必须是映射")
        _apply(cfg.obs, dict(obs_raw), where="env.obs")

    if cfg.gait_prior:
        cfg.gait = _load_gait(gait_file)
    return cfg


def _load_gait(gait_file: Any) -> GaitParams:
    """读先验文件；路径相对仓根解析。"""
    if gait_file is None:
        return GaitParams.default_trot()
    path = Path(str(gait_file))
    if not path.is_absolute():
        path = paths.WORKSPACE / path
    if not path.is_file():
        raise SystemExit(f"找不到步态先验文件：{path}")
    return load_gait(path)


def _apply(target: Any, values: Mapping[str, Any], *, where: str) -> None:
    """把 ``values`` 写进 dataclass ``target``，未知键报错。"""
    valid = set(getattr(target, "__dataclass_fields__", {}))
    unknown = sorted(k for k in values if k not in valid)
    if unknown:
        raise SystemExit(
            f"config.yaml 的 {where}: 无法识别的键 {unknown}；"
            f"可用键 = {sorted(valid)}"
        )
    for key, value in values.items():
        setattr(target, key, value)


# --------------------------------------------------------------------------- #
# 课程
# --------------------------------------------------------------------------- #


def config_to_stages(raw: Mapping[str, Any]) -> list[StageSpec]:
    """把 ``curriculum:`` 段展开成阶段表（默认就是平地 + 3/6/9/12/15 cm）。

    阶段名由高度决定：平地是 ``flat``，其余是 ``stair<整数厘米>``（``3.5`` 会写成
    ``stair3.5``）。步数与学习率按 ``heights_cm`` 的长度对齐。
    """
    cur = _section(raw, "curriculum")
    tread = float(cur.get("tread", DEFAULT_TREAD))
    n_up = int(cur.get("n_up", DEFAULT_N_UP))

    heights = cur.get("heights_cm")
    if heights is None:
        heights = [h * 100.0 for h in STAGE_HEIGHTS_CM]
    if not isinstance(heights, (list, tuple)) or not heights:
        raise SystemExit("config.yaml 的 curriculum.heights_cm 必须是非空列表")
    heights = [float(h) for h in heights]
    if any(h < 0 for h in heights):
        raise SystemExit("curriculum.heights_cm: 台阶高不能为负")
    if heights != sorted(heights):
        raise SystemExit(f"curriculum.heights_cm 必须按高度升序：{heights}")

    n = len(heights)
    budgets = _timesteps(cur.get("timesteps", DEFAULT_STAGE_TIMESTEPS), n,
                         where="curriculum.timesteps")
    lrs = _floats(cur.get("learning_rate"), n,
                  where="curriculum.learning_rate", default=DEFAULT_LR)

    stages: list[StageSpec] = []
    for h_cm, steps, lr in zip(heights, budgets, lrs):
        steps = int(steps)
        if h_cm <= 0.0:
            spec = flat_spec(n_up=0, n_down=0, step_height=0.0, tread=tread)
            name, label = "flat", "平地"
        else:
            spec = demo_stairs_spec(step_height=h_cm / 100.0, tread=tread, n_up=n_up)
            name = f"stair{h_cm:g}"
            label = f"{h_cm:g}cm 台阶"
        stages.append(
            StageSpec(
                name=name,
                step_height=float(spec.step_height),
                spec=spec,
                timesteps=steps,
                learning_rate=float(lr),
                label=label,
                description=f"{label} 上行走并到达顶层平台",
            )
        )
    return stages


# --------------------------------------------------------------------------- #
# PPO 与播放
# --------------------------------------------------------------------------- #


def config_to_ppo(raw: Mapping[str, Any]) -> dict[str, Any]:
    """``ppo:`` 段原样返回（键的合法性由 :func:`robotdog.training.build_ppo` 校验）。"""
    return _section(raw, "ppo")


def config_to_run(raw: Mapping[str, Any]) -> dict[str, Any]:
    """``run:`` 段，补上默认值。"""
    run = _section(raw, "run")
    run.setdefault("name", "course1")
    run.setdefault("seed", 0)
    run.setdefault("save_every", 50_000)
    run.setdefault("log_every", 20)
    return run


def config_to_play(raw: Mapping[str, Any]) -> dict[str, Any]:
    """``play:`` 段，补上渲染默认值。"""
    play = _section(raw, "play")
    play.setdefault("episodes_gif", 10)
    play.setdefault("episodes_eval", 100)
    play.setdefault("seed", 0)
    play.setdefault("camera", "track")
    play.setdefault("width", 960)
    play.setdefault("height", 540)
    play.setdefault("fps", 25)
    play.setdefault("scale", 2)
    return play


def n_envs(raw: Mapping[str, Any]) -> int:
    """``env.n_envs``，缺省 8。"""
    return int(_section(raw, "env").get("n_envs", 8))


def device(raw: Mapping[str, Any]) -> str:
    """``ppo.device``，缺省 ``cpu``。"""
    return str(_section(raw, "ppo").get("device", "cpu"))
