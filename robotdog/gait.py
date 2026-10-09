"""开环步态先验 —— 让残差 RL 在这里能学起来的热启动。

为什么需要它
------------
单进程 8 环境（沙箱禁用 Windows 命名管道，`SubprocVecEnv` 用不了，见
`robotdog/training.py`）下 PPO 从零学不出步态：两次标定（平地 120k / 150k 步）
都以 0% 成功率、位移约 0.02 m 收场，学到的策略比「零动作站桩」还差
（回合回报 466 vs 496）。原因是平台期而非权重 bug —— 随机关节抖动不产生净推进，
前进速度奖励在出现**协调的周期步态**之前是平的，50 Hz 采样的 MLP 撞不上它。

所以离线先搜好一个固定的开环 trot，再让策略学**残差**：

    target = home + gait_gain * gait.radians(t) + action_scale * action

即常规的残差 RL（Johannink et al., "Residual Reinforcement Learning", 2019）。
观测/动作接口不变：动作仍是 12 个关节目标角偏移。先验是固定的、可检查的轨迹，
不是隐藏控制器；`EnvConfig.gait_prior` 默认 `False`，纯残差写法依然可用。

帧约定
------
幅值与偏置都是**归一化动作单位**（与 `action` 同尺度），乘 `action_scale` 才是弧度，
于是 `action_scale` 变了先验依然成立；加载时强制 `|off| + amp <= 1`。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

LEG_ORDER = ("FL", "FR", "RL", "RR")
# Trot: the diagonal pairs (FL, RR) and (FR, RL) move together.
TROT_PHASE = (0.0, math.pi, math.pi, 0.0)
CALF_SHAPES = ("sin", "dip", "half")

# 默认先验 = 「速度目标」搜索的赢家：自然前进速度已是 0.55 m/s（实测 0.550 m/s
# 均值、8 s 走 4.39 m、100% 存活、机身 0.268 m，标称站姿 0.27 m）。
_FALLBACK_TROT = dict(
    freq=6.0,
    amp_th=0.24914502679724,
    amp_ca=0.5518579715236906,
    ph_th=1.6754541487333945,
    ph_ca=-2.146505768979483,
    off_th=0.25539290883226595,
    off_ca=0.09025271647195035,
    calf_shape="sin",
    action_scale=0.90,
)

# 「距离目标」搜索的赢家（1.702 m/s、6 s 走 10.19 m、100% 存活），留作速度快照。
FAST_TROT = dict(
    freq=4.0,
    amp_th=0.6093323116494958,
    amp_ca=0.9305071165813157,
    ph_th=1.533544334735976,
    ph_ca=2.1658126114417198,
    off_th=0.3906676883505043,
    off_ca=0.0694928834186843,
    calf_shape="sin",
    action_scale=0.70,
)


@dataclass
class GaitParams:
    """A periodic joint-target trajectory, in normalised action units."""

    freq: float = 2.0  # Hz, gait cycle frequency
    amp_th: float = 0.0  # thigh swing amplitude
    amp_ca: float = 0.0  # calf swing amplitude
    ph_th: float = 0.0  # thigh phase offset (rad)
    ph_ca: float = 0.0  # calf phase offset (rad)
    off_th: float = 0.0  # thigh constant offset
    off_ca: float = 0.0  # calf constant offset
    calf_shape: str = "sin"  # sin | dip | half
    action_scale: float = 0.70  # rad per normalised unit
    leg_phase: tuple[float, float, float, float] = field(
        default_factory=lambda: tuple(TROT_PHASE)
    )

    # ---------------------------------------------------------------- shape
    def _calf_wave(self, x: np.ndarray) -> np.ndarray:
        if self.calf_shape == "sin":
            return np.sin(x)
        if self.calf_shape == "dip":
            return -0.5 * (1.0 - np.cos(x))
        if self.calf_shape == "half":
            return -np.maximum(0.0, np.sin(x))
        raise ValueError(f"unknown calf_shape {self.calf_shape!r}")

    def leg_phases(self, t: float) -> np.ndarray:
        return 2.0 * np.pi * self.freq * t + np.asarray(self.leg_phase, dtype=np.float64)

    def actions(self, t: float) -> np.ndarray:
        """Normalised action units, shape ``(12,)``, in actuator order."""
        p = self.leg_phases(t)
        thigh = self.off_th + self.amp_th * np.sin(p + self.ph_th)
        calf = self.off_ca + self.amp_ca * self._calf_wave(p + self.ph_ca)
        a = np.zeros((4, 3), dtype=np.float64)  # per leg: hip, thigh, calf
        a[:, 1] = thigh
        a[:, 2] = calf
        return np.clip(a, -1.0, 1.0).ravel()

    def targets(self, t: float, home: np.ndarray) -> np.ndarray:
        """Absolute joint target angles (rad), using this gait's own scale."""
        return np.asarray(home, dtype=np.float64) + self.radians(t)

    def radians(self, t: float) -> np.ndarray:
        """Joint-angle offsets in radians, shape ``(12,)``, in actuator order.

        The environment uses this (not :meth:`actions`) so that the prior's
        amplitude is independent of ``EnvConfig.action_scale``, which only
        controls how much authority the *policy's* residual action has.  Mixing
        the two would silently multiply the policy's exploration noise by the
        prior's scale (measured: it made the robot fall within 2 s).
        """
        return self.action_scale * self.actions(t)

    @property
    def period(self) -> float:
        return 1.0 / self.freq if self.freq > 0 else float("inf")

    # -------------------------------------------------------------- control
    def normalize(self) -> GaitParams:
        """Clamp to a physically meaningful region; keep ``|off| + amp <= 1``."""
        self.freq = float(np.clip(self.freq, 0.0, 10.0))
        for name in ("amp_th", "amp_ca", "off_th", "off_ca"):
            setattr(self, name, float(np.clip(getattr(self, name), -1.0, 1.0)))
        for amp, off in (("amp_th", "off_th"), ("amp_ca", "off_ca")):
            a, o = getattr(self, amp), getattr(self, off)
            if abs(o) + a > 1.0:
                s = 1.0 / max(abs(o) + a, 1e-9)
                setattr(self, off, o * s)
                setattr(self, amp, a * s)
        self.ph_th = float(math.atan2(math.sin(self.ph_th), math.cos(self.ph_th)))
        self.ph_ca = float(math.atan2(math.sin(self.ph_ca), math.cos(self.ph_ca)))
        if self.calf_shape not in CALF_SHAPES:
            raise ValueError(f"calf_shape must be one of {CALF_SHAPES}")
        self.leg_phase = tuple(float(v) for v in self.leg_phase)
        if len(self.leg_phase) != 4:
            raise ValueError("leg_phase must have 4 entries (FL, FR, RL, RR)")
        return self

    @property
    def is_static(self) -> bool:
        return self.amp_th == 0.0 and self.amp_ca == 0.0

    # ------------------------------------------------------------- (de)serial
    def to_dict(self) -> dict:
        d = asdict(self)
        d["leg_phase"] = list(d["leg_phase"])
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: dict) -> GaitParams:
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in d.items() if k in known}
        if "leg_phase" in kw:
            kw["leg_phase"] = tuple(float(v) for v in kw["leg_phase"])
        if isinstance(kw.get("calf_shape"), int):
            kw["calf_shape"] = CALF_SHAPES[int(kw["calf_shape"])]
        return cls(**kw).normalize()

    @classmethod
    def from_json(cls, path: str | Path) -> GaitParams:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_search_json(cls, path: str | Path, rank: int = 0) -> GaitParams:
        """Load entry ``rank`` of a ``scripts/search_trot.py`` result file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = data.get("top") or []
        if not rows:
            raise ValueError(f"{path} has no 'top' entries")
        entry = rows[rank]
        d = dict(entry["gait"])
        d["action_scale"] = float(entry["action_scale"])
        return cls.from_dict(d)

    @classmethod
    def default_trot(cls) -> GaitParams:
        """Best known trot at the commanded speed (0.550 m/s on flat)."""
        return cls.from_dict(dict(_FALLBACK_TROT))

    @classmethod
    def fast_trot(cls) -> GaitParams:
        """Fastest searched trot (1.702 m/s) -- useful as a sanity baseline."""
        return cls.from_dict(dict(FAST_TROT))

    @classmethod
    def static(cls) -> GaitParams:
        """A prior that does nothing (equivalent to plain ``residual`` mode)."""
        return cls(freq=0.0, amp_th=0.0, amp_ca=0.0).normalize()

    # ----------------------------------------------------------------- report
    def summary(self) -> str:
        return (f"{self.freq:.2f} Hz, ±{self.amp_th:.3f} thigh / ±{self.amp_ca:.3f} calf "
                f"(norm. units), shape={self.calf_shape}, scale={self.action_scale:.2f} rad, "
                f"period={self.period * 1000:.0f} ms")


def load_gait(path: str | Path | None) -> GaitParams:
    """Load a gait from a JSON file, a search result, or return the default."""
    if path is None:
        return GaitParams.default_trot()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"gait file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "top" in data:  # a search_trot.py result
        return GaitParams.from_search_json(path)
    return GaitParams.from_dict(data)
