"""Open-loop gait priors -- the warm start that makes residual RL work here.

Why this module exists
----------------------
With 8 CPU environments in a single process (this sandbox forbids Windows
named pipes, so ``SubprocVecEnv`` is unavailable -- see ``robotdog/training.py``)
PPO never discovered a gait from scratch.  Two calibration runs, 120k and 150k
steps on flat, both ended at 0% success and ~0.02 m travelled; the learned
policy scored *worse* than a zero-action stance (466 vs 496 return per episode).

The reason is a plateau, not a weight bug: random joint jitter produces no net
propulsion, so the forward-velocity reward is flat until a *coordinated*
periodic gait appears.  Nothing in a 350k-parameter MLP sampled at 50 Hz finds
that by accident.

So we search for a fixed open-loop trot offline and let the policy learn the
*residual* on top of it:

    target = home + gait_gain * gait.radians(t) + action_scale * action

This is ordinary residual RL (Johannink et al., "Residual Reinforcement
Learning", 2019).  Everything about the observation/action interface the user
asked for is unchanged -- the action is still 12 joint-target offsets, and the
result is still a joint-space position command.  The prior is a fixed,
inspectable trajectory, not a hidden controller, and ``gait_prior`` defaults to
``False`` so the plain formulation is still available.

Measured by ``scripts/search_trot.py`` on flat ground with the real
``Go2StairsEnv`` (deterministic reset, zero action noise):

* speed objective, target 0.55 m/s -> **0.550 m/s**, 4.39 m in 8 s, 100%
  survival, body height 0.268 m (this is ``GaitParams.default_trot()``)
* distance objective -> **1.702 m/s**, 10.19 m in 6 s, 100% survival
  (``GaitParams.fast_trot()``)

Frame conventions
-----------------
Amplitudes and offsets are expressed in **normalised action units** (the same
units as ``action``), so multiplying by ``action_scale`` yields radians.  This
keeps the gait valid if ``action_scale`` changes: ``|off| + amp <= 1`` is
enforced when loading.
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

# Default prior = the winner of the *speed* objective search
# (``artifacts/trot_gait.json``), i.e. the gait whose natural forward speed is
# already the commanded 0.55 m/s: 0.550 m/s mean, 4.39 m in 8 s, 100% survival,
# body height 0.268 m (nominal stance is 0.27 m).
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

# The distance-objective winner, kept because it is the cleanest proof that the
# model can locomote fast: 1.702 m/s, 10.19 m in 6 s, 100% survival.
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
