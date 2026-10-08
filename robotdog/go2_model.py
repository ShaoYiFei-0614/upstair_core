"""Compose MuJoCo scenes from the Unitree Go2 MJCF plus custom terrain.

The vendored `models/unitree_go2/go2.xml` is never modified. Instead this module
reads it, injects terrain geometry into the worldbody, and compiles the result
as a single self-contained MJCF string.

Element injection points (all inside the Go2 document):
    <asset>      -> extra textures/materials for the ground       (append)
    <worldbody>  -> terrain geoms, lights                          (append)
    <option>     -> timestep / integrator / gravity overrides      (edit)

`<include>` is deliberately not used: MuJoCo resolves relative asset
directories against the *main* file, and compiling from a string has no main
file directory to resolve against.

Mesh assets and non-ASCII paths
-------------------------------
This workspace lives at ``D:\\项目\\RobortDog\\upstair``. MuJoCo's asset reader
resolves narrow (``const char*``) filenames through the Windows ANSI codepage,
so pointing ``meshdir`` at a Chinese path fails with::

    Error opening file 'D:/<mojibake>/models/unitree_go2/assets/base_0.obj'

``mujoco.MjVfs`` exposes only ``close()`` to Python, so custom VFS callbacks are
not available. The supported fix is the ``assets`` mapping accepted by
``MjModel.from_xml_string``, which serves file contents from memory and never
touches the filesystem. We therefore keep the original relative
``meshdir="assets"`` and always pass :func:`go2_assets`.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from robotdog.paths import GO2_ASSETS_DIR, GO2_XML
from robotdog.terrain import TERRAIN_GEOM_GROUP

# --------------------------------------------------------------------------- #
# Go2 constants (mirrored from models/unitree_go2/go2.xml, verified at load)
# --------------------------------------------------------------------------- #

LEGS: tuple[str, str, str, str] = ("FL", "FR", "RL", "RR")
JOINT_SUFFIXES: tuple[str, str, str] = ("hip", "thigh", "calf")

# Actuator order as declared in go2.xml:
#   FL_hip, FL_thigh, FL_calf, FR_hip, ..., RR_hip, RR_thigh, RR_calf
ACTUATOR_NAMES: tuple[str, ...] = tuple(f"{leg}_{s}" for leg in LEGS for s in JOINT_SUFFIXES)
JOINT_NAMES: tuple[str, ...] = tuple(f"{name}_joint" for name in ACTUATOR_NAMES)
FOOT_GEOM_NAMES: tuple[str, ...] = LEGS

# Home / nominal standing pose, from go2.xml <keyframe name="home">.
BASE_HEIGHT_HOME = 0.27
HOME_JOINT_ANGLES: tuple[float, ...] = tuple(
    angle for _ in LEGS for angle in (0.0, 0.9, -1.8)
)

# Total robot mass [kg]: 6.921 base + 4*(0.678 hip + 1.152 thigh + 0.241352 calf)
NOMINAL_MASS = 15.206408

DEFAULT_TIMESTEP = 0.002
DEFAULT_CONTROL_DECIMATION = 10  # -> 50 Hz control, 500 Hz physics

# Geom groups used by the Go2 MJCF.  ``2`` holds visual-only meshes that never
# collide; ``3`` holds every colliding primitive (torso box/cylinder/sphere,
# hip/thigh/calf shapes and the four foot spheres), so ``group == 3`` is exactly
# "this geom can touch the terrain".  Terrain lives in ``TERRAIN_GEOM_GROUP``.
VISUAL_GEOM_GROUP = 2
COLLISION_GEOM_GROUP = 3


# --------------------------------------------------------------------------- #
# MJCF text assembly
# --------------------------------------------------------------------------- #

_OUTER = re.compile(r"\s*<mujoco\b[^>]*>(?P<body>.*)</mujoco>\s*$", re.DOTALL)
_OPTION = re.compile(r"<option\b[^>]*/>")
_COMPILER = re.compile(r"<compiler\b[^>]*/>")
_MESHDIR = re.compile(r'meshdir\s*=\s*"[^"]*"')


@dataclass
class SceneOptions:
    """Overrides layered onto the Go2 model when a scene is compiled."""

    timestep: float = DEFAULT_TIMESTEP
    integrator: str = "implicitfast"
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    # Solver/contact stability for a ~15 kg quadruped on thin stair edges.
    iterations: int = 50
    ls_iterations: int = 20
    solver: str = "Newton"
    # Restitution-like softness for stiff foot-box contacts.
    add_ground_texture: bool = True
    extra_visual: str = ""
    nconmax: int | None = None
    njmax: int | None = None


def _go2_inner_xml() -> str:
    """Return go2.xml contents with the outer <mujoco> wrapper removed.

    ``meshdir`` is deliberately left as the original relative ``"assets"``:
    the mesh bodies are supplied in memory by :func:`go2_assets`, so no
    directory lookup happens at all (see the module docstring).
    """
    text = GO2_XML.read_text(encoding="utf-8")
    match = _OUTER.match(text)
    if not match:
        raise ValueError(f"could not parse {GO2_XML}: missing <mujoco> wrapper")
    inner = match.group("body")

    if not _MESHDIR.search(inner):
        # Defensive: without meshdir MuJoCo would look in the CWD.
        inner, n = _COMPILER.subn(
            lambda m: m.group(0)[:-2] + ' meshdir="assets"/>', inner, count=1
        )
        if n == 0:
            raise ValueError(f"could not parse {GO2_XML}: missing <compiler/>")
    return inner


@functools.lru_cache(maxsize=1)
def go2_assets() -> dict[str, bytes]:
    """Mesh file contents keyed by basename, for MuJoCo's in-memory VFS.

    Read once and cached: the 16 OBJ files total ~27 MB and every worker
    process pays this cost a single time.

    Keys must be *bare basenames*: MuJoCo rejects a mapping that contains the
    same file name twice ("Repeated file name in assets dict"), so do not also
    register ``assets/<name>`` aliases.
    """
    meshes = sorted(GO2_ASSETS_DIR.glob("*.obj"))
    if not meshes:
        raise FileNotFoundError(
            f"no .obj meshes in {GO2_ASSETS_DIR}; run scripts/download_go2.py"
        )
    return {path.name: path.read_bytes() for path in meshes}


def _option_xml(opts: SceneOptions) -> str:
    g = " ".join(f"{v:g}" for v in opts.gravity)
    out = (
        f'<option cone="elliptic" impratio="100" timestep="{opts.timestep:g}" '
        f'integrator="{opts.integrator}" gravity="{g}" '
        f'iterations="{opts.iterations}" ls_iterations="{opts.ls_iterations}" '
        f'solver="{opts.solver}"/>'
    )
    if opts.nconmax is not None or opts.njmax is not None:
        attrs = ""
        if opts.nconmax is not None:
            attrs += f' nconmax="{opts.nconmax}"'
        if opts.njmax is not None:
            attrs += f' njmax="{opts.njmax}"'
        out += f"<size{attrs}/>"
    return out


_GROUND_ASSETS = """
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.25 0.32 0.40" rgb2="0.14 0.19 0.25" markrgb="0.85 0.85 0.85"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="8 8" reflectance="0.05"/>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.5 0.7"
      rgb2="0.05 0.05 0.08" width="512" height="3072"/>
    <material name="stairmat" rgba="0.62 0.60 0.55 1" reflectance="0.02"/>
    <material name="edgemat" rgba="0.95 0.72 0.15 1"/>
"""

_GROUND_VISUAL = """
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20" offwidth="1280" offheight="720"/>
  </visual>
"""


def build_scene_xml(
    terrain_xml: str = "",
    *,
    opts: SceneOptions | None = None,
    floor_geom: str | None = None,
    include_lights: bool = True,
) -> str:
    """Assemble a compilable MJCF string: Go2 + floor/terrain + lighting.

    Parameters
    ----------
    terrain_xml:
        Raw XML inserted as the last children of ``<worldbody>``. Use it for
        stairs, heightfields, obstacles, and any per-episode geometry.
    floor_geom:
        Optional ``<geom .../>`` string for a ground plane. Pass ``None`` for no
        floor (terrain supplies its own).
    """
    opts = opts or SceneOptions()
    inner = _go2_inner_xml()
    inner = _OPTION.sub(_option_xml(opts), inner, count=1)

    if opts.add_ground_texture:
        inner = inner.replace("</asset>", _GROUND_ASSETS + "  </asset>", 1)

    worldbody_additions: list[str] = []
    if floor_geom is not None:
        worldbody_additions.append(floor_geom)
    if include_lights:
        worldbody_additions.append(
            '<light pos="0 0 3" dir="0 0 -1" directional="true" castshadow="false"/>'
            '<light pos="3 -3 3" dir="-1 1 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>'
        )
    if terrain_xml:
        worldbody_additions.append(terrain_xml)

    if worldbody_additions:
        inner = inner.replace(
            "</worldbody>", "\n".join(worldbody_additions) + "\n  </worldbody>", 1
        )

    visual = _GROUND_VISUAL if opts.add_ground_texture else ""
    if opts.extra_visual:
        visual = visual.replace("</visual>", opts.extra_visual + "</visual>", 1)

    return (
        '<mujoco model="go2_stairs">\n'
        + inner
        + visual
        + "\n</mujoco>\n"
    )


def default_floor_geom(z: float = 0.0, size: float = 60.0) -> str:
    """A textured infinite plane at height ``z``.

    The plane carries ``group=TERRAIN_GEOM_GROUP`` so that the forward
    rangefinder in :mod:`robotdog.touch` sees the ground but never the robot.
    Pick that group with care -- MuJoCo hides groups 3-5 by default, so a bad
    choice makes the whole world vanish from the renders.
    """
    return (
        f'<geom name="floor" type="plane" pos="0 0 {z:g}" size="{size:g} {size:g} 0.05" '
        f'condim="3" friction="0.9 0.005 0.0001" material="groundplane" '
        f'group="{TERRAIN_GEOM_GROUP}"/>'
    )


# --------------------------------------------------------------------------- #
# Compiled-model metadata
# --------------------------------------------------------------------------- #


@dataclass
class Go2Meta:
    """Resolved ids/indices for the compiled Go2 scene."""

    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    foot_geom_names: tuple[str, ...]

    joint_ids: np.ndarray
    actuator_ids: np.ndarray
    foot_geom_ids: np.ndarray
    foot_radius: float

    qpos_joint_adr: np.ndarray
    qvel_joint_adr: np.ndarray
    dof_joint_adr: np.ndarray

    base_body_id: int
    imu_site_id: int

    joint_ranges: np.ndarray  # (12, 2)
    ctrl_ranges: np.ndarray  # (12, 2)
    home_qpos: np.ndarray  # (19,)
    home_joint_angles: np.ndarray  # (12,)
    total_mass: float
    floor_geom_id: int
    stair_geom_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    # Collision-geom split of the robot (all ``geom_group == 3``).
    #
    # ``trunk_geom_ids`` is the three collision primitives welded to ``base``
    # (box + cylinder + sphere): a contact here means the torso itself hit the
    # terrain, i.e. a genuine fall.
    #
    # ``leg_geom_ids`` is every other robot collision primitive (hips, thighs,
    # calves) minus the four foot spheres.  A contact here is a shin/knee
    # *grazing* a riser while the robot is still upright and exploring -- it must
    # be penalised, not turned into an instant death sentence.
    trunk_geom_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    leg_geom_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    nq: int = 0
    nv: int = 0
    nu: int = 0

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


def _resolve_id(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx < 0:
        raise KeyError(f"{objtype} named {name!r} not found in compiled model")
    return int(idx)


def extract_meta(model: mujoco.MjModel) -> Go2Meta:
    """Read back every id/index the environment needs from a compiled model."""
    joint_ids = np.array(
        [_resolve_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES], dtype=int
    )
    actuator_ids = np.array(
        [_resolve_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES], dtype=int
    )
    foot_geom_ids = np.array(
        [_resolve_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOM_NAMES], dtype=int
    )
    # The foot is a sphere; geom_xpos is its centre, so true clearance needs it.
    foot_radius = float(model.geom_size[foot_geom_ids[0], 0])

    base_body_id = _resolve_id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    imu_site_id = _resolve_id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")

    # A free joint always occupies the first 7 qpos / 6 qvel entries.
    qpos_joint_adr = np.array([model.jnt_qposadr[j] for j in joint_ids], dtype=int)
    qvel_joint_adr = np.array([model.jnt_dofadr[j] for j in joint_ids], dtype=int)
    dof_joint_adr = qvel_joint_adr

    joint_ranges = np.array([model.jnt_range[j] for j in joint_ids], dtype=float)
    ctrl_ranges = np.array([model.actuator_ctrlrange[a] for a in actuator_ids], dtype=float)

    home_joint_angles = np.asarray(HOME_JOINT_ANGLES, dtype=float)
    home_qpos = np.zeros(model.nq, dtype=float)
    home_qpos[2] = BASE_HEIGHT_HOME
    home_qpos[3] = 1.0  # quaternion w
    home_qpos[qpos_joint_adr] = home_joint_angles

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    stair_ids = [
        g
        for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("stair_")
    ]

    # Exclude the robot's own 12 bodies from the mass sum where relevant.
    total_mass = float(model.body_mass[1:].sum())

    # --- split the robot's collision geoms into trunk / legs -----------------
    # The Go2 MJCF puts visual-only meshes in group 2 and every colliding
    # primitive in group 3, so ``group == 3`` is exactly "can generate a
    # contact with the terrain".
    terrain_ids = {int(floor_id)} | {int(g) for g in stair_ids}
    trunk_ids: list[int] = []
    leg_ids: list[int] = []
    foot_set = {int(g) for g in foot_geom_ids}
    for g in range(model.ngeom):
        if int(model.geom_group[g]) != COLLISION_GEOM_GROUP:
            continue
        if g in terrain_ids or g in foot_set:
            continue
        if int(model.geom_bodyid[g]) == base_body_id:
            trunk_ids.append(g)
        else:
            leg_ids.append(g)

    return Go2Meta(
        joint_names=JOINT_NAMES,
        actuator_names=ACTUATOR_NAMES,
        foot_geom_names=FOOT_GEOM_NAMES,
        joint_ids=joint_ids,
        actuator_ids=actuator_ids,
        foot_geom_ids=foot_geom_ids,
        foot_radius=foot_radius,
        qpos_joint_adr=qpos_joint_adr,
        qvel_joint_adr=qvel_joint_adr,
        dof_joint_adr=dof_joint_adr,
        base_body_id=base_body_id,
        imu_site_id=imu_site_id,
        joint_ranges=joint_ranges,
        ctrl_ranges=ctrl_ranges,
        home_qpos=home_qpos,
        home_joint_angles=home_joint_angles,
        total_mass=total_mass,
        floor_geom_id=int(floor_id),
        stair_geom_ids=np.array(stair_ids, dtype=int),
        trunk_geom_ids=np.array(trunk_ids, dtype=int),
        leg_geom_ids=np.array(leg_ids, dtype=int),
        nq=model.nq,
        nv=model.nv,
        nu=model.nu,
    )


def compile_scene(
    terrain_xml: str = "",
    *,
    opts: SceneOptions | None = None,
    floor_geom: str | None = None,
    write_to: Path | None = None,
) -> tuple[mujoco.MjModel, Go2Meta, str]:
    """Build + compile a scene, returning (model, meta, xml_text)."""
    if floor_geom is None:
        floor_geom = default_floor_geom()
    xml = build_scene_xml(terrain_xml, opts=opts, floor_geom=floor_geom)

    if write_to is not None:
        write_to.parent.mkdir(parents=True, exist_ok=True)
        write_to.write_text(xml, encoding="utf-8")

    try:
        model = mujoco.MjModel.from_xml_string(xml, assets=go2_assets())
    except Exception as exc:  # surface the generated XML for debugging
        if write_to is None:
            from robotdog.paths import GENERATED_DIR

            GENERATED_DIR.mkdir(parents=True, exist_ok=True)
            dump = GENERATED_DIR / "failed_scene.xml"
            dump.write_text(xml, encoding="utf-8")
            raise RuntimeError(f"MJCF compile failed; dumped to {dump}") from exc
        raise

    meta = extract_meta(model)
    return model, meta, xml


def home_qpos(meta: Go2Meta, *, x: float = 0.0, y: float = 0.0, z: float | None = None,
              yaw: float = 0.0) -> np.ndarray:
    """Home keyframe qpos, optionally relocated/oriented (yaw about world Z)."""
    qpos = meta.home_qpos.copy()
    qpos[0] = x
    qpos[1] = y
    if z is not None:
        qpos[2] = z
    half = yaw / 2.0
    qpos[3:7] = (np.cos(half), 0.0, 0.0, np.sin(half))
    return qpos


# --------------------------------------------------------------------------- #
# Frame conventions
# --------------------------------------------------------------------------- #

def quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """(w,x,y,z) -> 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quat_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray:
    """(w,x,y,z) -> (roll, pitch, yaw) about the world X/Y/Z axes (ZYX order)."""
    w, x, y, z = quat_wxyz
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=float)
