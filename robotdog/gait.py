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
# trot：对角腿对（FL, RR）与（FR, RL）同步摆动。
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
@dataclass
class GaitParams:
    """一条周期性的关节目标轨迹，单位为归一化动作单位。"""

    freq: float = 2.0  # Hz，步态周期频率
    amp_th: float = 0.0  # 大腿摆动幅值
    amp_ca: float = 0.0  # 小腿摆动幅值
    ph_th: float = 0.0  # 大腿相位偏置（rad）
    ph_ca: float = 0.0  # 小腿相位偏置（rad）
    off_th: float = 0.0  # 大腿常量偏置
    off_ca: float = 0.0  # 小腿常量偏置
    calf_shape: str = "sin"  # 小腿波形：sin | dip | half
    action_scale: float = 0.70  # 每归一化单位对应的弧度
    leg_phase: tuple[float, float, float, float] = field(
        default_factory=lambda: tuple(TROT_PHASE)
    )

    # ---------------------------------------------------------------- 形状
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
        """归一化动作单位，形状 ``(12,)``，按执行器顺序排列。"""
        p = self.leg_phases(t)
        thigh = self.off_th + self.amp_th * np.sin(p + self.ph_th)
        calf = self.off_ca + self.amp_ca * self._calf_wave(p + self.ph_ca)
        a = np.zeros((4, 3), dtype=np.float64)  # 每条腿：hip, thigh, calf
        a[:, 1] = thigh
        a[:, 2] = calf
        return np.clip(a, -1.0, 1.0).ravel()

    def targets(self, t: float, home: np.ndarray) -> np.ndarray:
        """绝对关节目标角（rad），使用本步态自己的缩放系数。"""
        return np.asarray(home, dtype=np.float64) + self.radians(t)

    def radians(self, t: float) -> np.ndarray:
        """关节角偏置（弧度），形状 ``(12,)``，按执行器顺序排列。

        环境用的是这个（而不是 :meth:`actions`），这样先验的
        幅值就与 ``EnvConfig.action_scale`` 无关：后者只
        控制*策略*残差动作有多少权限。把两者混在一起，会在
        背地里把策略的探索噪声乘上先验的缩放系数
        （实测：它让机器人在 2 s 内摔倒）。
        """
        return self.action_scale * self.actions(t)

    @property
    def period(self) -> float:
        return 1.0 / self.freq if self.freq > 0 else float("inf")

    # -------------------------------------------------------------- 控制
    def normalize(self) -> GaitParams:
        """裁剪到物理上有意义的范围；保持 ``|off| + amp <= 1``。"""
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

    # ------------------------------------------------------------- （反）序列化
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
    def from_search_json(cls, path: str | Path, rank: int = 0) -> GaitParams:
        """读取 ``scripts/search_trot.py`` 结果文件里的第 ``rank`` 条记录。"""
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
        """已知最优的 trot（指令速度下；平地 0.550 m/s）。"""
        return cls.from_dict(dict(_FALLBACK_TROT))

    @classmethod
    def static(cls) -> GaitParams:
        """什么都不做的先验（等价于纯 ``residual`` 模式）。"""
        return cls(freq=0.0, amp_th=0.0, amp_ca=0.0).normalize()

    # ----------------------------------------------------------------- 报表
    def summary(self) -> str:
        return (f"{self.freq:.2f} Hz, ±{self.amp_th:.3f} thigh / ±{self.amp_ca:.3f} calf "
                f"(norm. units), shape={self.calf_shape}, scale={self.action_scale:.2f} rad, "
                f"period={self.period * 1000:.0f} ms")


def load_gait(path: str | Path | None) -> GaitParams:
    """从 JSON 文件或搜索结果加载步态；都没有时返回默认步态。"""
    if path is None:
        return GaitParams.default_trot()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"gait file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "top" in data:  # search_trot.py 的结果文件
        return GaitParams.from_search_json(path)
    return GaitParams.from_dict(data)
