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
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from robotdog.tools.terrain import StairSpec, demo_stairs_spec, flat_spec

# --------------------------------------------------------------------------- #
# 固定课程梯度
# --------------------------------------------------------------------------- #

#: 固定的台阶高度梯度 (cm): 先平地, 再 3 cm 逐级到 15 cm。
STAGE_HEIGHTS_CM: tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0)

#: 所有台阶阶段共用的地形尺寸。
DEFAULT_TREAD: float = 0.50
DEFAULT_N_UP: int = 8
DEFAULT_TOP_LANDING: float = 1.20
DEFAULT_WIDTH: float = 2.0
DEFAULT_START_X: float = 1.50

#: 平地阶段与台阶阶段的保底步数预算 (用作 ``total_timesteps`` 的权重基准)。
_FLAT_BASE_STEPS: int = 100_000
_STAIR_BASE_STEPS: int = 250_000


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

    @property
    def is_flat(self) -> bool:
        """平地段: 没有上行台阶 (``step_height == 0.0`` 且 ``n_up == 0``)。"""
        return bool(self.spec.n_up == 0 and self.spec.n_down == 0)

    def summary(self) -> str:
        """一行文本, 便于日志。表格请用 :func:`format_curriculum`。"""
        s = self.spec
        return (
            f"{self.name:<8} {self.label:<10} 台阶={self.step_height * 100:5.1f}cm "
            f"踏面={s.tread * 100:4.0f}cm 级数={s.n_up:<2d} "
            f"顶端={s.crest_height:5.2f}m 预算={self.timesteps:>9,} "
            f"lr={self.learning_rate:g}"
        )


def default_curriculum(
    *,
    tread: float = DEFAULT_TREAD,
    n_up: int = DEFAULT_N_UP,
    total_timesteps: int | None = None,
) -> list[StageSpec]:
    """返回固定的六级课程 (平地 -> 3 -> 6 -> 9 -> 12 -> 15 cm)。

    **步数预算的取法**: 平地段给得最少。开环 trot 先验本身就能在平地上走,
    平地阶段几乎不产生学习信号 —— 实测里它的成功率很快就顶住, 继续训只是
    空转; 台阶阶段才是学习真正发生的地方, 所以台阶步数明显大于平地。

    ``total_timesteps`` 给出时按 1:2.5 的权重 (平地 : 每级台阶) 分配,
    并保证每级至少 1 步; 余数补给最后的 (最难的) 一级。
    """
    tread_m = float(tread)
    n_up_i = int(n_up)
    stair_kw = dict(tread=tread_m, n_up=n_up_i, top_landing=DEFAULT_TOP_LANDING,
                    width=DEFAULT_WIDTH, start_x=DEFAULT_START_X)
    flat = flat_spec(n_up=0, n_down=0, step_height=0.0, tread=tread_m,
                     width=DEFAULT_WIDTH, start_x=DEFAULT_START_X)

    n_stair = len(STAGE_HEIGHTS_CM) - 1
    if total_timesteps is None:
        budgets = [_FLAT_BASE_STEPS] + [_STAIR_BASE_STEPS] * n_stair
    else:
        total = max(len(STAGE_HEIGHTS_CM), int(total_timesteps))
        weights = [_FLAT_BASE_STEPS] + [_STAIR_BASE_STEPS] * n_stair
        wsum = float(sum(weights))
        budgets = [max(1, int(total * w / wsum)) for w in weights]
        budgets[-1] += total - sum(budgets)       # 余数给最难的一级
    # 台阶越高, 学习率越小: 高立面需要更细的更新, 免得把已学会的步态踢掉。
    lrs = [1.5e-4, 1.5e-4, 1.2e-4, 1.0e-4, 8e-5, 6e-5]

    stages: list[StageSpec] = [
        StageSpec(
            name="flat",
            step_height=0.0,
            spec=flat,
            timesteps=budgets[0],
            learning_rate=lrs[0],
            label="平地",
            description="平地行走, 站稳并跟上前进速度指令",
        )
    ]
    for i, h_cm in enumerate(STAGE_HEIGHTS_CM[1:]):
        h = float(h_cm) / 100.0
        stages.append(
            StageSpec(
                name=f"stair{int(h_cm)}",
                step_height=h,
                spec=demo_stairs_spec(step_height=h, **stair_kw),
                timesteps=budgets[i + 1],
                learning_rate=lrs[i + 1],
                label=f"{h_cm:g}cm 台阶",
                description=f"{h_cm:g} cm 立面 x {n_up_i} 级, 学习抬腿跨越",
            )
        )
    return stages


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
# 训练接口
# --------------------------------------------------------------------------- #


class StageTrainer(Protocol):
    """一个阶段的实际训练者。

    只要提供同名方法即可, 无需继承 (结构化子类型)。真正的 PPO 代码在
    ``scripts/train.py`` 里实现这个接口, 本模块不 import SB3。
    """

    def make_env(self, stage: StageSpec, *, warm_start: str | None = None) -> Any:
        """为该阶段构建 (或复用) 一个向量化环境。返回任何 ``train_stage`` 能用的对象。"""
        ...

    def train_stage(
        self,
        stage: StageSpec,
        venv: Any,
        *,
        warm_start: str | None = None,
        tensorboard_log: str | Path | None = None,
        **kwargs: Any,
    ) -> tuple[Any, EpisodeStats]:
        """用 ``stage.timesteps`` 训练 ``stage``, 返回 ``(模型, 回合统计)``。"""
        ...

    def save_model(self, model: Any, stage: StageSpec, venv: Any) -> Path:
        """保存该阶段的权重, 返回 ``.zip`` 路径; 不支持则抛 ``NotImplementedError``。"""
        ...

    def close(self) -> None:
        """释放训练者持有的资源; 可选。"""
        ...


# --------------------------------------------------------------------------- #
# 课程驱动器
# --------------------------------------------------------------------------- #


@dataclass
class Curriculum:
    """按顺序跑完 ``stages``, 记录每级的 :class:`StageResult`。"""

    stages: list[StageSpec]
    results: list[StageResult] = field(default_factory=list)
    model_path: str | None = None      # 最近一次训练产出的模型路径
    summary_path: Path | None = None   # :meth:`run` 用 ``write_summary`` 写的文件

    # ---------------------------------------------------------------- 输入
    @classmethod
    def default(cls, *, total_timesteps: int | None = None) -> Curriculum:
        """用 :func:`default_curriculum` 建一个课程。"""
        return cls(default_curriculum(total_timesteps=total_timesteps))

    def names(self) -> list[str]:
        return [s.name for s in self.stages]

    def __iter__(self):
        return iter(self.stages)

    def __len__(self) -> int:
        return len(self.stages)

    # ---------------------------------------------------------------- 执行
    def run(
        self,
        trainer: StageTrainer,
        *,
        resume_from: str | None = None,
        tensorboard_log: str | Path | None = None,
        summary_path: str | Path | None = None,
        meta: dict[str, Any] | None = None,
        verbose: int = 1,
        **train_kwargs: Any,
    ) -> list[StageResult]:
        """顺序训练每一级; 下一级从上一级的权重 warm-start。

        ``train_stage`` 返回的模型可以选择**同一个实例** (在线续训) 或新实例;
        后者更安全, 因为 ``PPO.set_parameters`` 会把上一级冻结的对数标准差一起
        搬过来, 换实例能保证没有残留状态。预算由 ``stage.timesteps`` 决定,
        没有任何中途毕业条件。
        """
        self.results = []
        warm = resume_from
        for i, stage in enumerate(self.stages, start=1):
            if verbose:
                print(f"\n[课程] >>> 第 {i}/{len(self.stages)} 级 "
                      f"'{stage.name}' ({stage.label})", flush=True)
                print(f"[课程]     {stage.summary()} -- {stage.description}",
                      flush=True)
            t0 = time.time()
            venv = trainer.make_env(stage, warm_start=warm)
            try:
                model, stats = trainer.train_stage(
                    stage, venv, warm_start=warm,
                    tensorboard_log=tensorboard_log, **train_kwargs,
                )
                obs_size = _obs_size_of(venv)
                try:
                    saved = trainer.save_model(model, stage, venv)
                except NotImplementedError:
                    saved = None
                path = str(saved) if saved else _model_path_of(model)
            finally:
                close = getattr(venv, "close", None)
                if callable(close):
                    close()
            wall = time.time() - t0
            path = path or warm or ""
            result = StageResult(
                name=stage.name,
                label=stage.label,
                step_height=stage.step_height,
                timesteps=int(stage.timesteps),
                mean_return=float(stats.mean_return),
                success_rate=float(stats.success_rate),
                fall_rate=float(stats.fall_rate),
                wall_seconds=wall,
                model_path=str(path),
                mean_length=float(stats.mean_length),
                episodes=int(stats.n_total),
                obs_size=obs_size,
            )
            self.results.append(result)
            if path:
                self.model_path = str(path)
            warm = self.model_path
            if verbose:
                print(f"[课程] <<< '{stage.name}' 完成: 成功率={result.success_rate:.1%} "
                      f"摔倒率={result.fall_rate:.1%} 均回报={result.mean_return:.2f} "
                      f"用时={format_seconds(wall)}", flush=True)

        if summary_path is not None:
            payload = dict(meta or {})
            payload.setdefault("stage_heights_cm", list(STAGE_HEIGHTS_CM))
            payload.setdefault("n_stages", len(self.stages))
            payload.setdefault("total_timesteps", sum(s.timesteps for s in self.stages))
            payload["resume_from"] = resume_from
            payload["final_model"] = self.model_path
            self.summary_path = write_summary(
                summary_path, self.stages, self.results, payload
            )
        return self.results

    # ---------------------------------------------------------------- 输出
    def table(self) -> str:
        """:meth:`run` 已产生的阶段结果表格 (没跑过则只有阶段定义)。"""
        if not self.results:
            return format_curriculum(self.stages)
        return format_curriculum(self.stages, self.results)

    def total_timesteps(self) -> int:
        return int(sum(s.timesteps for s in self.stages))


def _model_path_of(model: Any) -> str:
    """模型对象的落盘路径; 拿不到就返回空串。"""
    for attr in ("model_path", "path"):
        val = getattr(model, attr, None)
        if val:
            return str(val)
    return ""


def _obs_size_of(venv: Any) -> int:
    """向量环境的观测维度; 拿不到就返回 0。"""
    space = getattr(venv, "observation_space", None)
    shape = getattr(space, "shape", None)
    try:
        return int(shape[0]) if shape else 0
    except (TypeError, IndexError, ValueError):
        return 0


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
    path: str | Path | dict[str, Any],
    stages: Sequence[StageSpec] | str | Path,
    results: Sequence[StageResult] | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """写 ``summary.json``: 课程定义 + 每级结果 + 元信息 (中文不转义)。

    前两个参数也接受 ``(payload, path)`` 的顺序 (老脚本习惯先攒 payload)。
    """
    if isinstance(path, dict):
        path, meta = stages, path
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
    "Curriculum",
    "EpisodeStats",
    "StageResult",
    "StageSpec",
    "StageTrainer",
    "default_curriculum",
    "format_curriculum",
    "format_seconds",
    "write_summary",
]
