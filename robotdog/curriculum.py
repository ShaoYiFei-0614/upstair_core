"""课程学习: 固定六级阶梯 (平地 + 3/6/9/12/15 cm 台阶)。

课程是**串行**的: 每个阶段只对应一种台阶几何 (无高度带、阶段内无随机化),
在该阶段的步数预算上训练, 保存模型, 然后进入下一阶段。下一阶段从上一阶段的
权重 warm-start (``PPO.set_parameters``), 所以策略会迁移。

阶段之间切换地形必须重新编译 MJCF 并 reset, 因此切换只发生在两个 ``train_stage``
调用之间, 不会打断 rollout。真正的 PPO 训练留在 :mod:`robotdog.training` /
``scripts/train.py``, 这里只负责"训哪几级、按什么顺序、训多少步"。
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from robotdog.tools.terrain import StairSpec

# --------------------------------------------------------------------------- #
# 固定课程梯度
# --------------------------------------------------------------------------- #

#: 固定的台阶高度梯度 (cm): 三级平地预热（直走 → 小指令 → 全量程指令），
#: 再 3 / 6 / 9 cm。`config.yaml` 的 `curriculum.heights_cm` 优先；这里只是兜底默认。
#: 同一高度重复出现是合法的，`config_to_stages()` 会用 `_unique_name()` 去重命名。
STAGE_HEIGHTS_CM: tuple[float, ...] = (0.0, 0.0, 0.0, 3.0, 6.0, 9.0)

#: 所有台阶阶段共用的地形尺寸。
DEFAULT_TREAD: float = 0.50
DEFAULT_N_UP: int = 8



@dataclass
class StageSpec:
    """课程的一级。

    ``spec`` 是该阶段**唯一**的地形; 阶段内不再按高度或踏面铺开。
    """

    name: str
    step_height: float          # 台阶高 [m], 平地 = 0.0
    spec: StairSpec
    timesteps: int              # 该阶段的训练步数预算
    learning_rate: float        # 该阶段 PPO 的学习率
    label: str = ""             # 中文标签, 用于日志与报表
    description: str = ""       # 一句话说明这一级在练什么
    command_vy_max: float = 0.0     # 该阶段横向指令上界 [m/s]
    command_yaw_max: float = 0.0    # 该阶段偏航指令上界 [rad/s]
    # 该阶段前向指令的采样区间 ``(vx_min, vx_max)`` [m/s]；``(0, 0)`` = 用 env 的
    # 定值基线 ``env.command.vx``。
    command_vx: tuple[float, float] = (0.0, 0.0)
    reward_overrides: dict[str, float] = field(default_factory=dict)
    # 该阶段的奖励覆盖, 来自 ``config.yaml`` 的
    # ``curriculum.reward_overrides``; 空 = 用 ``reward:`` 段的基线值。

    @property
    def is_flat(self) -> bool:
        """平地段: 没有上行台阶 (``step_height == 0.0`` 且 ``n_up == 0``)。"""
        return bool(self.spec.n_up == 0 and self.spec.n_down == 0)

    def summary(self) -> str:
        """一行文本, 便于日志。表格请用 :func:`format_curriculum`。"""
        s = self.spec
        cmd = ""
        if self.command_vy_max or self.command_yaw_max:
            cmd = (f" 指令 vy±{self.command_vy_max:g} wz±{self.command_yaw_max:g}")
        vx_lo, vx_hi = self.command_vx
        if vx_hi:
            cmd += (f" vx={vx_lo:g}" if vx_lo == vx_hi
                    else f" vx=[{vx_lo:g},{vx_hi:g}]")
        rew = ""
        if self.reward_overrides:
            items = " ".join(f"{k}={v:g}" for k, v in sorted(
                self.reward_overrides.items()))
            rew = f" 奖励覆盖 {items}"
        return (
            f"{self.name:<8} {self.label:<10} 台阶={self.step_height * 100:5.1f}cm "
            f"踏面={s.tread * 100:4.0f}cm 级数={s.n_up:<2d} "
            f"顶端={s.crest_height:5.2f}m 预算={self.timesteps:>9,} "
            f"lr={self.learning_rate:g}{cmd}{rew}"
        )
# --------------------------------------------------------------------------- #
# 回合统计
# --------------------------------------------------------------------------- #


@dataclass
class EpisodeStats:
    """滑动窗口内的已完成回合统计。

    数据来自 ``VecMonitor`` 放在 ``info["episode"]`` 上的 ``r`` / ``l``; 成功与摔倒
    来自环境自己的 ``info["is_success"]`` / ``info["fallen"]``。
    """

    window: int = 50
    _success: deque[float] = field(default_factory=deque, repr=False)
    _fall: deque[float] = field(default_factory=deque, repr=False)
    _returns: deque[float] = field(default_factory=deque, repr=False)
    _lengths: deque[float] = field(default_factory=deque, repr=False)
    n_total: int = 0

    def __post_init__(self) -> None:
        self._success = deque(self._success, maxlen=self.window)
        self._fall = deque(self._fall, maxlen=self.window)
        self._returns = deque(self._returns, maxlen=self.window)
        self._lengths = deque(self._lengths, maxlen=self.window)

    def add(self, *, success: bool, fallen: bool, ret: float, length: float) -> None:
        """记一个已结束的回合 (``ret``/``length`` 应为 ``nan`` 表示未知)。"""
        self._success.append(1.0 if success else 0.0)
        self._fall.append(1.0 if fallen else 0.0)
        self._returns.append(float(ret))
        self._lengths.append(float(length))
        self.n_total += 1

    def add_info(self, info: dict[str, Any]) -> bool:
        """从环境 ``info`` 里取一条回合记录, 成功时返回 ``True``。

        只有 ``VecMonitor`` 打了 ``episode`` 键的 ``info`` 才是回合结束; 其它
        信息字典会被忽略, 所以调用方可以无脑遍历。
        """
        ep = info.get("episode")
        if not isinstance(ep, dict):
            return False
        self.add(
            success=bool(info.get("is_success", False)),
            fallen=bool(info.get("fallen", False)),
            ret=float(ep.get("r", float("nan"))),
            length=float(ep.get("l", float("nan"))),
        )
        return True

    def add_infos(self, infos: Iterable[dict[str, Any]]) -> int:
        """批量吃一批 ``info``, 返回新增的回合数。"""
        n = 0
        for info in infos:
            n += int(self.add_info(info))
        return n

    @property
    def n_window(self) -> int:
        return len(self._success)

    @property
    def success_rate(self) -> float:
        return float(np.mean(self._success)) if self._success else 0.0

    @property
    def fall_rate(self) -> float:
        return float(np.mean(self._fall)) if self._fall else 0.0

    @property
    def mean_return(self) -> float:
        return _nanmean(self._returns)

    @property
    def mean_length(self) -> float:
        return _nanmean(self._lengths)

    def as_dict(self) -> dict[str, float]:
        return {
            "episodes_total": self.n_total,
            "episodes_window": self.n_window,
            "success_rate": round(self.success_rate, 4),
            "fall_rate": round(self.fall_rate, 4),
            "mean_return": round(self.mean_return, 2),
            "mean_length": round(self.mean_length, 1),
        }

    def format_line(self) -> str:
        """一行实时进度, 供训练回调打印。"""
        return (
            f"回合={self.n_total:>5d} 成功率={self.success_rate:6.1%} "
            f"摔倒率={self.fall_rate:6.1%} 均回报={self.mean_return:9.2f} "
            f"均步数={self.mean_length:6.1f}"
        )


def _nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else 0.0


# --------------------------------------------------------------------------- #
# 阶段结果
# --------------------------------------------------------------------------- #


@dataclass
class StageResult:
    """一个阶段训练结束后的实测结果。"""

    name: str
    label: str = ""
    step_height: float = 0.0
    timesteps: int = 0
    mean_return: float = 0.0
    success_rate: float = 0.0
    fall_rate: float = 0.0
    wall_seconds: float = 0.0
    model_path: str = ""
    # 非合同字段, 只为报表好读。
    mean_length: float = 0.0
    episodes: int = 0
    obs_size: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "step_height": float(self.step_height),
            "step_height_cm": round(float(self.step_height) * 100.0, 2),
            "timesteps": int(self.timesteps),
            "mean_return": round(float(self.mean_return), 3),
            "success_rate": round(float(self.success_rate), 4),
            "fall_rate": round(float(self.fall_rate), 4),
            "mean_length": round(float(self.mean_length), 1),
            "episodes": int(self.episodes),
            "wall_seconds": round(float(self.wall_seconds), 2),
            "obs_size": int(self.obs_size),
            "model_path": str(self.model_path),
            "extras": dict(self.extras),
        }


# --------------------------------------------------------------------------- #
# 报表
# --------------------------------------------------------------------------- #


def format_curriculum(
    stages: Sequence[StageSpec],
    results: Sequence[StageResult] | None = None,
) -> str:
    """把阶段表 (和可选的实测结果) 排成可打印的定宽表格。"""
    by_name = {r.name: r for r in (results or [])}
    head = (f"{'阶段':<8} {'标签':<10} {'台阶':>6} {'踏面':>5} {'级数':>4} "
            f"{'顶端':>7} {'预算':>10} {'学习率':>8}")
    if results:
        head += f" {'成功率':>7} {'摔倒率':>7} {'均回报':>9} {'用时':>8} {'模型':<28}"
    lines = [head, "-" * len(head)]
    for st in stages:
        row = (f"{st.name:<8} {st.label:<10} {st.step_height * 100:5.1f}c "
               f"{st.spec.tread * 100:4.0f}c {st.spec.n_up:>4d} "
               f"{st.spec.crest_height:6.2f}m {st.timesteps:>10,} "
               f"{st.learning_rate:>8g}")
        r = by_name.get(st.name)
        if results:
            if r is None:
                row += f" {'-':>7} {'-':>7} {'-':>9} {'-':>8} {'(未训练)':<28}"
            else:
                name = Path(r.model_path).name if r.model_path else "-"
                row += (f" {r.success_rate:6.1%} {r.fall_rate:6.1%} "
                        f"{r.mean_return:9.2f} {format_seconds(r.wall_seconds):>8} "
                        f"{name:<28}")
        lines.append(row)
    total = sum(s.timesteps for s in stages)
    lines.append("-" * len(head))
    lines.append(f"合计 {len(stages)} 级, 总预算 {total:,} 步")
    return "\n".join(lines)


def write_summary(
    path: str | Path,
    stages: Sequence[StageSpec],
    results: Sequence[StageResult] | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """写 ``summary.json``: 课程定义 + 每级结果 + 元信息 (中文不转义)。

    ``meta`` 里的键会**平铺到顶层**，再补上 ``stages`` / ``results`` /
    ``total_timesteps`` 三个由本函数计算的键。

    ``stages`` 里每一级除了地形与预算，还**逐级**记下该级的指令量程
    （``command_vx`` / ``command_vy_max`` / ``command_yaw_max``）。这些是**课程
    逐级下发的值**，``env.json`` 里只有 ``config.yaml`` 的基线（``vy_max=0``），
    回放时如果拿基线去评，判出来的成功/失败与训练时不是同一件事 —— 曾经因此把
    「指令跟随」的级按「前进 N 米」去评，GIF 上打出一堆假的「摔倒/超时」。
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dict(meta or {})
    payload["stages"] = [
        {
            "name": s.name,
            "label": s.label,
            "step_height": float(s.step_height),
            "timesteps": int(s.timesteps),
            "learning_rate": float(s.learning_rate),
            "is_flat": bool(s.is_flat),
            "command_vx": [float(v) for v in s.command_vx],
            "command_vy_max": float(s.command_vy_max),
            "command_yaw_max": float(s.command_yaw_max),
            "spec": _spec_as_dict(s.spec),
        }
        for s in stages
    ]
    payload["results"] = [r.as_dict() for r in (results or [])]
    payload["total_timesteps"] = int(sum(s.timesteps for s in stages))
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _spec_as_dict(spec: StairSpec) -> dict[str, Any]:
    d = dict(vars(spec))
    d["friction"] = list(d.get("friction", ()))
    d["crest_height"] = float(spec.crest_height)
    d["up_end_x"] = float(spec.up_end_x)
    return d


def format_seconds(seconds: float) -> str:
    """把秒数写成 ``12.3s`` / ``4.5min`` / ``1.20h``。"""
    s = float(seconds)
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s / 60:.1f}min"
    return f"{s / 3600:.2f}h"


__all__ = [
    "STAGE_HEIGHTS_CM",
    "EpisodeStats",
    "StageResult",
    "StageSpec",
    "format_curriculum",
    "format_seconds",
    "write_summary",
]
