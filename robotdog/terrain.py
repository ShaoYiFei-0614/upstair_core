"""Parametric stair terrain for MuJoCo, built from solid box geoms.

Design notes
------------
* Every step is a **solid block resting on the ground plane** (not a thin slab):
  half-size ``(tread/2, width/2, top_z/2)`` centred at ``z = top_z/2``. Solid
  blocks cannot be caught on from underneath and keep contact geometry simple.
* Static-static overlap between neighbouring blocks is harmless in MuJoCo --
  contacts are only generated for pairs where at least one body is dynamic.
* Crisp box edges are deliberate: the robot must learn to clear a sharp
  riser, whereas a heightfield would round the edge over one cell.

Layouts (walking direction is +x)
---------------------------------
``up``      : flat ground -> n_up risers -> flat top landing.
``pyramid`` : flat ground -> n_up risers -> flat crest -> n_down risers -> ground.
              This directly provides both 上楼 and 下楼 without needing a pit.

The analytic height profile is exposed through :func:`height_at` so that reward
terms, spawn placement and evaluation metrics all agree with the geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

DEFAULT_STAIR_FRICTION: tuple[float, float, float] = (0.9, 0.005, 0.0001)
DEFAULT_STAIR_RGBA = "0.62 0.60 0.55 1"
DEFAULT_EDGE_RGBA = "0.95 0.72 0.15 1"

#: MuJoCo geom group reserved for terrain (stairs **and** the ground floor).
#:
#: The forward rangefinder in :mod:`robotdog.touch` must be able to look ahead
#: without hitting the robot's own front legs, which sweep exactly through the
#: probed volume.  ``mj_ray`` takes a *geom group mask*, so putting all terrain
#: geometry in one dedicated group and casting with a mask that enables only
#: that group makes the probe blind to the robot by construction.
#:
#: **Group 1, and the value is load-bearing twice over:**
#:
#: 1. The Unitree MJCF assigns ``group="2"`` to its visual geoms and
#:    ``group="3"`` to every collision geom (torso boxes, knee cylinders and the
#:    four foot spheres), so 2 and 3 are taken; group 0 is what any geom without
#:    an explicit group gets.
#: 2. MuJoCo's ``mjv_defaultOption`` ships ``geomgroup = [1, 1, 1, 0, 0, 0]`` --
#:    **groups 3, 4 and 5 are hidden by default.**  Stamping the stairs with
#:    group 4 therefore makes them invisible in every offscreen render while
#:    leaving the physics untouched: the dog is seen climbing bare air while the
#:    HUD happily reports the height gain.  (This really happened; see
#:    ``scripts/verifications/step11_touch.py`` test 1.)  Group 1 sits inside the
#:    default-visible range, so the staircase shows up without having to touch
#:    the renderer.
#:
#: :meth:`robotdog.env.Go2StairsEnv.render` additionally forces *all six* groups
#: on, so changing this constant cannot silently hide the terrain again.
TERRAIN_GEOM_GROUP = 1

#: How far (in treads) the flat crest / top landing reaches *back* into the last
#: ascending step.  It must be > 0, and the reason is a genuine MuJoCo trap:
#:
#: The crest sits at exactly the same height as the last tread, so the two boxes
#: are coplanar.  When they merely *abut* (crest starts exactly at ``up_end_x``)
#: their touching faces are a zero-height seam.  A foot sphere whose centre lands
#: on that shared plane is at **zero** penetration from both vertical faces but
#: tens of millimetres from the top faces, and MuJoCo resolves a sphere-inside-box
#: contact along the *least penetrating* axis -- so both boxes push the foot
#: horizontally, in opposite directions.  The two pushes cancel, nothing supports
#: the foot in z, and it wedges: pinned in x, sunk 30-40 mm below the tread, with
#: ``touch.riser_graze`` latched on every step because the contact normal is
#: horizontal.  Measured on a 8 cm x 8 staircase (``.tmp/stall_probe.py``): both
#: front feet frozen at ``x = 5.500 == up_end_x`` for the remaining 700 control
#: steps, i.e. the robot climbs the whole staircase and then stands 0.114 m short
#: of the finish line until the 20 s cap -- which is exactly why tall-stair
#: training plateaued at 20-40 %.
#:
#: Overlapping the crest into the last step buries its rear face inside that step,
#: so the coincident faces disappear and only the real risers survive.  The solid
#: *union* is unchanged, so the analytic profile, the downward raycast and every
#: rendered pixel are identical.
CREST_SEAM_OVERLAP = 1.0  # in treads


@dataclass
class StairSpec:
    """Geometry of one staircase.

    Heights follow the user's curriculum: 5 cm risers with 50 cm treads first,
    then 3-8 cm, 8-15 cm and 15-22 cm bands during curriculum learning.
    """

    step_height: float = 0.05  # riser height [m]  (台阶高度)
    tread: float = 0.50  # tread depth along +x [m]  (台阶宽度)
    width: float = 2.00  # lateral extent along y [m]
    n_up: int = 8  # number of ascending risers
    n_down: int = 0  # ascending only when 0; >0 makes a pyramid
    start_x: float = 1.50  # world x where the first riser begins
    landing: float = 1.20  # flat crest length between the up and down flights [m]
    # Flat top platform when n_down == 0 [m].  Must be > 0 for the terrain to be
    # succeedable at all: ``StairSpec.height_at`` returns 0 past
    # ``up_end_x + top_landing``, and the success test requires the base to be
    # past ``up_end_x`` *and* sitting at crest height.  With the old default of
    # 0.0 a bare ``StairSpec()`` was a 0.40 m cliff at the crest, so
    # ``Go2StairsEnv()`` could never report a success.
    top_landing: float = 1.20
    center_y: float = 0.0
    friction: tuple[float, float, float] = DEFAULT_STAIR_FRICTION
    rgba: str = DEFAULT_STAIR_RGBA
    name_prefix: str = "stair"

    # ---- derived geometry -------------------------------------------------- #

    @property
    def crest_height(self) -> float:
        return self.n_up * self.step_height

    @property
    def up_end_x(self) -> float:
        """x where the last riser's tread ends (= start of the crest)."""
        return self.start_x + self.n_up * self.tread

    @property
    def down_end_x(self) -> float:
        """x where the descending flight returns to ground level."""
        return self.up_end_x + self.landing + self.n_down * self.tread

    @property
    def span_end_x(self) -> float:
        """x where solid staircase geometry ends (start of bare ground again)."""
        if self.n_down > 0:
            return self.down_end_x
        if self.n_up == 0:
            return self.up_end_x
        return self.up_end_x + self.top_landing

    @property
    def max_height(self) -> float:
        return self.crest_height

    def step_blocks(self) -> list[tuple[str, tuple[float, float, float],
                                       tuple[float, float, float]]]:
        """Return ``[(name, (cx, cy, cz), (hx, hy, hz)), ...]`` for every block.

        The full staircase is emitted front-to-back so that the raycast and the
        analytic profile in :func:`height_at` line up exactly.
        """
        blocks: list[tuple[str, tuple[float, float, float],
                           tuple[float, float, float]]] = []
        hx, hy = self.tread / 2.0, self.width / 2.0

        # Ascending flight: block i spans x in [start_x + i*tread, +tread],
        # top surface at z = (i+1) * step_height.
        for i in range(self.n_up):
            top = (i + 1) * self.step_height
            cx = self.start_x + (i + 0.5) * self.tread
            blocks.append(
                (f"{self.name_prefix}_up{i}",
                 (cx, self.center_y, top / 2.0), (hx, hy, top / 2.0))
            )

        # Flat crest between the flights.  ``ov`` reaches back into the last
        # ascending step so the two coplanar boxes overlap instead of abutting --
        # see CREST_SEAM_OVERLAP for the wedged-foot trap that avoids.
        ov = CREST_SEAM_OVERLAP * self.tread
        if self.n_down > 0:
            top = self.crest_height
            cx = self.up_end_x + (self.landing - ov) / 2.0
            blocks.append(
                (f"{self.name_prefix}_crest",
                 (cx, self.center_y, top / 2.0),
                 ((self.landing + ov) / 2.0, hy, top / 2.0))
            )

            # Descending flight: block j has its top surface one riser lower
            # than the previous, ending flush with the ground.
            for j in range(self.n_down):
                top = self.crest_height - (j + 1) * self.step_height
                cx = self.up_end_x + self.landing + (j + 0.5) * self.tread
                if top <= 1e-9:
                    continue
                blocks.append(
                    (f"{self.name_prefix}_down{j}",
                     (cx, self.center_y, top / 2.0), (hx, hy, top / 2.0))
                )
        elif self.top_landing > 0.0 and self.n_up > 0:
            # Ascending-only staircase: finish on a flat platform so the robot
            # does not walk off a 40 cm cliff after the last riser.  Requires
            # ``n_up > 0``: for a flat stage the crest height is 0, so a "top
            # landing" would be a degenerate zero-height block.
            top = self.crest_height
            cx = self.up_end_x + (self.top_landing - ov) / 2.0
            blocks.append(
                (f"{self.name_prefix}_top",
                 (cx, self.center_y, top / 2.0),
                 ((self.top_landing + ov) / 2.0, hy, top / 2.0))
            )
        return blocks

    def to_xml(self, indent: str = "    ") -> str:
        """MJCF ``<geom>`` elements for the whole staircase.

        Every block is stamped with :data:`TERRAIN_GEOM_GROUP`.  That is what
        lets the forward rangefinder in :mod:`robotdog.touch` cast a ray that is
        blind to the robot's own legs -- the robot's geoms sit in groups 2 and
        3.  Geom groups have no effect on physics, but they *do* decide whether
        the block is rendered at all, so see the constant's docstring before
        changing it.
        """
        out: list[str] = []
        fr = " ".join(f"{v:g}" for v in self.friction)
        for name, (cx, cy, cz), (hx, hy, hz) in self.step_blocks():
            out.append(
                f'{indent}<geom name="{name}" type="box" '
                f'pos="{cx:.6f} {cy:.6f} {cz:.6f}" '
                f'size="{hx:.6f} {hy:.6f} {hz:.6f}" '
                f'condim="3" friction="{fr}" rgba="{self.rgba}" '
                f'group="{TERRAIN_GEOM_GROUP}"/>'
            )
        return "\n".join(out)

    def height_at(self, x: float) -> float:
        """Analytic terrain height [m] at world x (0 outside the staircase)."""
        if x < self.start_x:
            return 0.0
        if x < self.up_end_x:
            i = int((x - self.start_x) // self.tread)
            return (i + 1) * self.step_height
        if self.n_down == 0:
            # Ascending-only: crest height across the optional top platform,
            # then bare ground (the geometry genuinely ends there).
            return self.crest_height if x < self.up_end_x + self.top_landing else 0.0
        if x < self.up_end_x + self.landing:
            return self.crest_height
        j = int((x - self.up_end_x - self.landing) // self.tread)
        if j >= self.n_down:
            return 0.0
        return max(0.0, self.crest_height - (j + 1) * self.step_height)


@dataclass
class TerrainRandomization:
    """Per-episode stair randomization (used from the domain-randomization step).

    Sampling happens at ``reset()``; the spec is rebuilt rather than patched so
    that the analytic profile stays exact.
    """

    height_range: tuple[float, float] = (0.03, 0.08)
    tread_range: tuple[float, float] = (0.30, 0.60)
    width_range: tuple[float, float] = (1.2, 3.0)
    n_up_range: tuple[int, int] = (4, 9)
    start_x: float = 1.20
    landing: float = 1.20


def sample_spec(rng: np.random.Generator, rand: TerrainRandomization,
                base: StairSpec | None = None) -> StairSpec:
    """Draw a randomized :class:`StairSpec` (pyramid: up then down)."""
    base = base or StairSpec()
    n_up = int(rng.integers(rand.n_up_range[0], rand.n_up_range[1] + 1))
    return replace(
        base,
        step_height=float(rng.uniform(*rand.height_range)),
        tread=float(rng.uniform(*rand.tread_range)),
        width=float(rng.uniform(*rand.width_range)),
        n_up=n_up,
        n_down=n_up,
        start_x=rand.start_x,
        landing=rand.landing,
    )


# --------------------------------------------------------------------------- #
# Convenience constructors matching the user's requested curriculum
# --------------------------------------------------------------------------- #

#: Flat ground only -- the first curriculum stage.
def flat_spec(**kw) -> StairSpec:
    kw.setdefault("n_up", 0)
    kw.setdefault("n_down", 0)
    return StairSpec(**kw)


#: 5 cm risers / 50 cm treads -- the user's explicitly requested first staircase.
def demo_stairs_spec(
    step_height: float = 0.05,
    tread: float = 0.50,
    n_up: int = 8,
    n_down: int = 0,
    **kw,
) -> StairSpec:
    """Explicit parameters so callers can override every field."""
    kw.setdefault("top_landing", 1.20 if n_down == 0 else 0.0)
    return StairSpec(
        step_height=step_height,
        tread=tread,
        n_up=n_up,
        n_down=n_down,
        **kw,
    )


#: 5 cm / 50 cm up-and-over pyramid (both climbing and descending).
def demo_pyramid_spec(
    step_height: float = 0.05,
    tread: float = 0.50,
    n_up: int = 8,
    n_down: int = 8,
    **kw,
) -> StairSpec:
    return StairSpec(
        step_height=step_height,
        tread=tread,
        n_up=n_up,
        n_down=n_down,
        **kw,
    )


def build_terrain_xml(spec: StairSpec) -> str:
    """MJCF fragment to inject into ``<worldbody>``."""
    if spec.n_up == 0 and spec.n_down == 0:
        return ""
    return "\n" + spec.to_xml()


# --------------------------------------------------------------------------- #
# Raycast helpers (verification + locomotion metrics)
# --------------------------------------------------------------------------- #

def raycast_heights(model, data, spec: StairSpec, xs: np.ndarray,
                    y: float | None = None, z_high: float = 6.0) -> np.ndarray:
    """Downward-raycast terrain height at each x, using MuJoCo's own collision.

    Returns ``nan`` where no geometry was hit.
    """
    y = spec.center_y if y is None else y
    geomid = np.zeros(1, dtype=np.int32)
    out = np.empty(len(xs), dtype=float)
    for k, x in enumerate(xs):
        pnt = np.array([float(x), y, z_high], dtype=float)
        dist = mujoco_mj_ray(model, data, pnt, geomid)
        out[k] = (z_high - dist) if dist >= 0 else np.nan
    return out


def mujoco_mj_ray(model, data, pnt: np.ndarray, geomid: np.ndarray) -> float:
    import mujoco

    return float(
        mujoco.mj_ray(
            model, data, pnt, np.array([0.0, 0.0, -1.0]),
            None, 1, -1, geomid,
        )
    )
