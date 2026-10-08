"""Tactile (contact) sensing and a forward stair rangefinder for the Go2.

Why this module exists
----------------------
The original specification (§四) said "do not use contact-sensor data", so the
observation carried no touch information at all and the only two contact *facts*
in the environment were the boolean masks in ``env._foot_contact_mask`` (used by
the ``air_time`` shaping) and ``env._body_contact`` (used by ``body_collision``
and the termination test).  Contact **forces** were never read.

That is enough to *punish* a collision but it cannot say "a riser is right in
front of your front-left foot, lift it".  This module adds exactly that:

1. :meth:`TouchSensor.feet` -- per-foot **normal force** and **slip ratio** read
   from MuJoCo's own contact solver via ``mj_contactForce``.
2. ``riser_graze`` -- a foot pushing against a **vertical** stair face, i.e. the
   literal "撞到台阶立面" event.
3. :meth:`TouchSensor.forward_probe` -- a **rangefinder**: two ``mj_ray`` casts
   look ahead of the robot and report how much the terrain rises and how far
   away the step is.

Verified MuJoCo facts this file relies on
-----------------------------------------
* ``data.contact[i].frame[0:3]`` **is** the contact normal and points from
  ``geom[0]`` to ``geom[1]`` (``mjdata.h``: *"normal is in [0-2], points from
  geom[0] to geom[1]"*).
* ``mj_contactForce(m, d, i, res)`` returns ``res`` **in the contact frame**:
  ``res[0]`` is the normal force, ``res[1:3]`` the two tangential (friction)
  components, ``res[3:6]`` the torque.
* Measured on flat ground, standing still: the four feet carry
  ``28.94 / 29.56 / 43.94 / 46.73 N`` whose **sum is 149.18 N == 100.0 % of the
  15.2064 kg body weight**.  That identity is used as a regression assertion in
  ``scripts/verifications/step11_touch.py``.

Self-occlusion of the rangefinder
---------------------------------
The probes start at the torso and shoot forwards, straight through the volume
the front knees sweep.  Casting with "all groups" therefore hits the robot's own
legs.  The fix is MuJoCo's ``geomgroup`` mask: the terrain and floor geoms live
in :data:`TERRAIN_GEOM_GROUP` while every robot geom sits in groups 2 (visual)
and 3 (collision), so a mask that enables only the terrain group can never see
the robot.  Geom groups do not affect physics -- but MuJoCo **does** hide groups
3-5 from offscreen renders by default, which is why the terrain group is 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotdog.terrain import TERRAIN_GEOM_GROUP

#: ``geomgroup`` argument for ``mj_ray``: enable the terrain group only.
TERRAIN_GEOM_MASK = np.array(
    [1 if g == TERRAIN_GEOM_GROUP else 0 for g in range(6)], dtype=np.uint8
)


@dataclass
class TouchConfig:
    """Geometry and thresholds of the tactile sensor suite."""

    # --- forward stair probe (the rangefinder) ----------------------------- #
    #: The probe is a **fan of vertical rays**, not a slanted beam.  A vertical
    #: ray measures the terrain height directly, so the reading is immune to the
    #: torso pitching through the trot cycle -- and, unlike a slanted ray, it
    #: never "sees past" a tread it has already climbed over.
    probe_forward: float = 0.34  # m, first ray, ahead of the base centre
    probe_span: float = 0.60  # m, how far the fan reaches past the first ray
    probe_rays: int = 9  # >= 2; spacing = probe_span/(probe_rays-1)
    probe_ray_top: float = 1.00  # m, rays start this high and shoot down
    probe_rise_eps: float = 0.015  # m, rise smaller than this is "still flat"
    probe_horizontal_height: float = 0.045  # m, side ray that pinpoints a riser
    probe_step_clip: float = 0.24  # m, clamp on the reported rise
    probe_step_ref: float = 0.030  # m, rise that earns full trigger credit
    probe_trig_dist: float = 0.62  # m, trigger ramps up inside this distance
    probe_range_obs: float = 1.20  # m, distance normalisation for the obs
    # --- foot tactile ------------------------------------------------------ #
    force_norm_ref: float = 75.0  # N, per-foot force that saturates the obs
    impact_force_target: float = 95.0  # N, hard-landing threshold
    slip_safe: float = 0.55  # tangential/normal ratio that costs nothing
    slip_loaded_force: float = 10.0  # N, below this a foot is not "loaded"
    slip_max: float = 8.0  # cap on the tangential/normal ratio (see feet())
    riser_max_normal_z: float = 0.60  # |n_z| below this == a vertical face
    riser_min_force: float = 5.0  # N, contact must push this hard


@dataclass
class TouchReading:
    """One control step worth of tactile + rangefinder data."""

    foot_normal_force: np.ndarray = field(
        default_factory=lambda: np.zeros(4))
    foot_tangential_force: np.ndarray = field(
        default_factory=lambda: np.zeros(4))
    foot_slip: np.ndarray = field(default_factory=lambda: np.zeros(4))
    foot_contact: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=bool))
    riser_graze: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=bool))
    #: a downhill-facing ray normal at the contact (0 = flat tread, 1|numeric)
    probe_hit: bool = False
    probe_range: float = 0.0  # horizontal distance to the step (m)
    probe_height_ahead: float = 0.0  # terrain height at the first rising ray (m)
    probe_step: float = 0.0  # rise of the NEAREST step (m, clipped)
    probe_slope: float = 0.0  # rise / horizontal distance ahead
    probe_far_step: float = 0.0  # tallest rise anywhere in the fan (m, clipped)

    # -------------------------------------------------------------- derived
    @property
    def total_normal_force(self) -> float:
        return float(self.foot_normal_force.sum())

    def loaded(self, cfg: TouchConfig) -> np.ndarray:
        """Feet that currently carry weight (used to gate the slip penalty)."""
        return self.foot_normal_force > cfg.slip_loaded_force

    def riser_graze_any(self) -> bool:
        return bool(self.riser_graze.any())

    # ------------------------------------------------------------ obs blocks
    def foot_obs(self, cfg: TouchConfig) -> np.ndarray:
        """9 values: 4 normalised forces, 4 slip ratios, 1 riser flag."""
        force = np.clip(self.foot_normal_force / cfg.force_norm_ref, 0.0, 1.0)
        slip = np.clip(self.foot_slip / 2.0, 0.0, 1.0)
        riser = np.array([1.0 if self.riser_graze.any() else 0.0])
        return np.concatenate([force, slip, riser])

    def probe_obs(self, cfg: TouchConfig) -> np.ndarray:
        """3 values: normalised rise, proximity, hit flag.

        ``rise``      -- ``clip(step / probe_step_clip, 0, 1)``; how tall the
                         obstacle ahead is relative to the tallest step the robot
                         is ever expected to climb.
        ``proximity`` -- ``1`` when the obstacle is at the sensor, ``0`` at (or
                         beyond) ``probe_range_obs``.  Measured **from the cast
                         origin**, not from the base: the sensor sits
                         ``probe_forward`` in front of the torso, so a
                         base-relative distance could never fall below that
                         offset and the term would saturate at ~0.45.
        ``hit``       -- the vertical fan found a tread that stands at least
                         ``probe_rise_eps`` above the one under the base.
        """
        rise = float(np.clip(self.probe_step / cfg.probe_step_clip, 0.0, 1.0))
        if self.probe_hit:
            prox = 1.0 - float(np.clip(self.ahead_of_sensor(cfg)
                                       / cfg.probe_range_obs, 0.0, 1.0))
        else:
            prox = 0.0
        return np.array([rise, prox, 1.0 if self.probe_hit else 0.0])

    def probe_obs_full(self, cfg: TouchConfig) -> np.ndarray:
        """5 values: the 3 above plus two quantities that were already computed.

        ``probe_slope``    -- ``clip(step / ahead, 0, 5) / 5``; how *steep* the
                              obstacle is right at the sensor.  This is the
                              "is this a climbable tread edge or a wall" feature,
                              and it is what distinguishes a 10 cm riser from a
                              40 cm one at the same distance.
        ``probe_far_step`` -- the tallest rise anywhere in the fan, normalised by
                              the same clip as ``rise``.  ``rise`` only reports the
                              *nearest* step; this one reports the tallest one
                              still in view.
        """
        base = self.probe_obs(cfg)
        slope = float(np.clip(self.probe_slope / 5.0, 0.0, 1.0))
        far = float(np.clip(self.probe_far_step / cfg.probe_step_clip, 0.0, 1.0))
        return np.concatenate([base, np.array([slope, far])])

    def ahead_of_sensor(self, cfg: TouchConfig) -> float:
        """Distance from the rangefinder to the obstacle (m).

        :attr:`probe_range` is measured from the **base centre** (that is the
        number worth printing in a log); the reward and observation need it
        relative to the sensor, which is mounted ``probe_forward`` ahead.
        """
        if not self.probe_hit:
            return float("inf")
        return max(float(self.probe_range) - cfg.probe_forward, 0.0)

    def trigger(self, cfg: TouchConfig) -> float:
        """The "a step is close AND tall" gate the lift reward multiplies.

        Smooth by construction (a product of two clipped ramps) so that the
        reward keeps a usable gradient instead of switching on and off.
        """
        if not self.probe_hit:
            return 0.0
        rise = float(np.clip(self.probe_step / cfg.probe_step_ref, 0.0, 1.0))
        prox = float(np.clip(1.0 - self.ahead_of_sensor(cfg) / cfg.probe_trig_dist,
                             0.0, 1.0))
        return rise * prox

    def as_dict(self) -> dict[str, float]:
        d = {
            "foot_force_sum": self.total_normal_force,
            "riser_graze": float(self.riser_graze.any()),
            "probe_hit": float(self.probe_hit),
            "probe_range": float(self.probe_range),
            "probe_step": float(self.probe_step),
            "probe_slope": float(self.probe_slope),
            "probe_far_step": float(self.probe_far_step),
        }
        for k in range(4):
            d[f"foot_force_{k}"] = float(self.foot_normal_force[k])
            d[f"foot_slip_{k}"] = float(self.foot_slip[k])
        return d


def cast_ray(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    geomgroup: np.ndarray | None = TERRAIN_GEOM_MASK,
    max_range: float | None = None,
) -> tuple[float, int]:
    """Cast one ray; return ``(distance, geom_id)``.

    ``distance`` is ``inf`` and ``geom_id`` is ``-1`` when nothing is hit (or
    when the hit is farther than ``max_range``).  ``direction`` is normalised
    here, so the returned distance is always in metres.
    """
    pnt = np.asarray(origin, dtype=np.float64).reshape(3)
    vec = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return float("inf"), -1
    vec = vec / norm
    geomid = np.array([-1], dtype=np.int32)
    dist = float(mujoco.mj_ray(
        model, data, pnt, vec, geomgroup, 1, -1, geomid
    ))
    if dist < 0.0:
        return float("inf"), -1
    if max_range is not None and dist > max_range:
        return float("inf"), -1
    return dist, int(geomid[0])


class TouchSensor:
    """Reads contact forces and rangefinder distances out of one MuJoCo data.

    Held by :class:`robotdog.env.Go2StairsEnv` (one instance per environment).
    """

    def __init__(self, model, data, meta, cfg: TouchConfig | None = None,
                 *, probe_enabled: bool = True) -> None:
        self.model = model
        self.data = data
        self.meta = meta
        self.cfg = cfg if cfg is not None else TouchConfig()
        #: The ray fan is the expensive half (~60 us per control step, roughly
        #: half a physics step).  The environment switches it off when neither
        #: the observation nor the ``stair_probe_lift`` reward reads it.
        self.probe_enabled = bool(probe_enabled)
        self.terrain_geoms: set[int] = {int(meta.floor_geom_id)} | {
            int(g) for g in meta.stair_geom_ids
        }
        self._foot_index = {
            int(g): k for k, g in enumerate(meta.foot_geom_ids)
        }
        self._force6 = np.zeros(6, dtype=np.float64)

    # ------------------------------------------------------------------ feet
    def feet(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(normal, tangential, slip, riser_graze)`` per foot.

        A foot may touch the terrain with more than one contact (the sphere
        against two tread edges, say); forces are summed and the riser flag is
        OR-ed.
        """
        cfg = self.cfg
        n = 4
        normal = np.zeros(n)
        tangential = np.zeros(n)
        contact = np.zeros(n, dtype=bool)
        riser = np.zeros(n, dtype=bool)

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self._foot_index:
                k = self._foot_index[g1]
                other = g2
            elif g2 in self._foot_index:
                k = self._foot_index[g2]
                other = g1
            else:
                continue
            if other not in self.terrain_geoms:
                continue

            mujoco.mj_contactForce(self.model, self.data, i, self._force6)
            fn = abs(float(self._force6[0]))
            ft = float(np.hypot(self._force6[1], self._force6[2]))
            normal[k] += fn
            tangential[k] += ft
            contact[k] = True

            # Vertical face?  ``frame[0:3]`` is the contact normal, so a normal
            # with a small world-z component means the foot is being pushed
            # sideways by a riser rather than supported by a tread.
            frame = np.asarray(c.frame, dtype=np.float64).reshape(3, 3)
            if (abs(float(frame[0][2])) < cfg.riser_max_normal_z
                    and fn > cfg.riser_min_force):
                riser[k] = True

        # The ratio is unbounded by construction: a foot that is barely loaded
        # but being scrubbed along a riser edge can read tangential/normal in
        # the hundreds.  That value is meaningless (anything above a few is
        # already "completely sliding") and it feeds both the observation and a
        # squared reward term, so cap it here at the source.
        slip = np.clip(tangential / np.maximum(normal, 1e-6), 0.0, cfg.slip_max)
        return normal, tangential, slip, riser

    # --------------------------------------------------------- the rangefinder
    def forward_probe(
        self, base_pos: np.ndarray, yaw: float, terrain_height: float
    ) -> tuple[bool, float, float, float, float]:
        """Look ahead of the torso.  Returns ``(hit, range, height, step, far_step)``.

        The probe is a **fan of vertical rays** shot downwards from
        ``probe_ray_top`` metres above the current tread, spread from
        ``probe_forward`` to ``probe_forward + probe_span`` ahead of the base.

        Why vertical rays instead of the obvious slanted beam?  A slanted ray
        has to descend to the height of the next tread before it can find it,
        so it is blind to a 5 cm step until it is almost on top of it; and once
        the torso has climbed onto a tread the beam flies over that tread and
        reports the *next* riser, so the distance reading jumps backwards.  A
        vertical ray simply reports "the ground here is this high", which is
        what the policy actually needs, and it does not care about the torso
        pitching through the trot cycle (the heading uses **yaw only** for the
        same reason).

        The flat side ray is then used only to sharpen the distance: it flies
        straight at the riser and stops on the vertical face, which is far more
        precise than the fan spacing -- but it is only trusted when it reports
        something *nearer* than the fan, because once the sensor origin is past
        a riser the flat ray happily skips ahead to the next one.
        """
        cfg = self.cfg
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        x0, y0 = float(base_pos[0]), float(base_pos[1])
        top = float(terrain_height) + cfg.probe_ray_top

        n = max(int(cfg.probe_rays), 2)
        span = max(float(cfg.probe_span), 0.0)
        spacing = span / float(n - 1)

        hit = False
        step = 0.0
        range_m = float(cfg.probe_range_obs)
        height_ahead = float(terrain_height)
        far_step = 0.0
        first_rise_dist = float("inf")

        for k in range(n):
            d = cfg.probe_forward + k * spacing
            origin = np.array([x0 + c * d, y0 + s * d, top])
            dist, _ = cast_ray(self.model, self.data, origin,
                               np.array([0.0, 0.0, -1.0]))
            if not np.isfinite(dist):
                continue
            h = top - float(dist)
            rise = h - float(terrain_height)
            if rise > far_step:
                far_step = rise
            if rise <= cfg.probe_rise_eps:
                continue
            if not hit:
                # The riser sits somewhere between this ray and the previous
                # one; stepping back half the spacing is a decent estimate.
                first_rise_dist = max(d - 0.5 * spacing, float(cfg.probe_forward))
                hit = True
                step = rise
                height_ahead = h

        if hit:
            range_m = first_rise_dist
            # Sharpen the distance with the flat ray, but never let it push the
            # reading *outwards* (that is the "flew over the tread" artefact).
            flat_origin = np.array([
                x0 + c * cfg.probe_forward,
                y0 + s * cfg.probe_forward,
                float(terrain_height) + cfg.probe_horizontal_height,
            ])
            flat_dist, _ = cast_ray(
                self.model, self.data, flat_origin, np.array([c, s, 0.0]))
            if np.isfinite(flat_dist):
                flat_range = float(cfg.probe_forward) + float(flat_dist)
                range_m = min(range_m, flat_range)

        height_ahead = float(height_ahead if hit else terrain_height)
        step = float(np.clip(step, 0.0, cfg.probe_step_clip))
        far_step = float(np.clip(far_step, 0.0, cfg.probe_step_clip))
        return hit, float(range_m), height_ahead, step, far_step

    # ------------------------------------------------------------------ read
    def read(self, *, base_pos: np.ndarray, yaw: float,
             terrain_height: float) -> TouchReading:
        normal, tangential, slip, riser = self.feet()
        if self.probe_enabled:
            hit, rng, height_ahead, step, far_step = self.forward_probe(
                base_pos, yaw, terrain_height
            )
        else:
            hit = False
            rng = height_ahead = step = far_step = 0.0
        if not hit:
            rng = self.cfg.probe_range_obs
        ahead = max(rng - self.cfg.probe_forward, 1e-6)
        return TouchReading(
            foot_normal_force=normal,
            foot_tangential_force=tangential,
            foot_slip=slip,
            foot_contact=normal > 0.0,
            riser_graze=riser,
            probe_hit=hit,
            probe_range=float(rng),
            probe_height_ahead=float(height_ahead),
            probe_step=float(step),
            probe_slope=float(np.clip(step / ahead, 0.0, 5.0)) if hit else 0.0,
            probe_far_step=float(far_step),
        )


def empty_reading() -> TouchReading:
    """A "nothing touched, nothing ahead" reading.

    Used before the first :meth:`reset` so that ``_obs_frame`` is well defined.
    """
    return TouchReading()
