"""由 Unitree Go2 的 MJCF 加上自定义地形组装出 MuJoCo 场景。

vendored 的 `models/unitree_go2/go2.xml` 永不被修改。本模块改为读取它，
把地形几何注入 worldbody，再把结果编译成一个自包含的 MJCF 字符串。

元素注入点（都在 Go2 文档内部）：
    <asset>      -> 地面额外需要的 texture/material          （追加）
    <worldbody>  -> 地形 geom、灯光                          （追加）
    <option>     -> timestep / integrator / gravity 覆盖     （修改）

`<include>` 被刻意弃用：MuJoCo 把相对资源目录解析到*主*文件上，而从字符串
编译时根本没有主文件目录可解析。

网格资源与非 ASCII 路径
-----------------------
本工作区位于 ``D:\\项目\\RobortDog\\upstair_core``。MuJoCo 的资源读取器通过
Windows ANSI 代码页解析窄字符（``const char*``）文件名，所以把 ``meshdir``
指向中文路径会失败并报出::

    Error opening file 'D:/<mojibake>/models/unitree_go2/assets/base_0.obj'

``mujoco.MjVfs`` 只向 Python 暴露 ``close()``，所以自定义 VFS 回调不可用。
受支持的修法是 ``MjModel.from_xml_string`` 接受的 ``assets`` 映射：它直接从
内存提供文件内容，完全不碰文件系统。所以我们保留原本的相对路径
``meshdir="assets"``，并始终传入 :func:`go2_assets`。
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from robotdog.tools.paths import GO2_ASSETS_DIR, GO2_XML
from robotdog.tools.terrain import TERRAIN_GEOM_GROUP

# --------------------------------------------------------------------------- #
# Go2 常量（镜像自 models/unitree_go2/go2.xml，加载时校验）
# --------------------------------------------------------------------------- #

LEGS: tuple[str, str, str, str] = ("FL", "FR", "RL", "RR")
JOINT_SUFFIXES: tuple[str, str, str] = ("hip", "thigh", "calf")

# go2.xml 里声明的执行器顺序：
#   FL_hip, FL_thigh, FL_calf, FR_hip, ..., RR_hip, RR_thigh, RR_calf
ACTUATOR_NAMES: tuple[str, ...] = tuple(f"{leg}_{s}" for leg in LEGS for s in JOINT_SUFFIXES)
JOINT_NAMES: tuple[str, ...] = tuple(f"{name}_joint" for name in ACTUATOR_NAMES)
FOOT_GEOM_NAMES: tuple[str, ...] = LEGS

# home / 标称站姿，取自 go2.xml 的 <keyframe name="home">。
BASE_HEIGHT_HOME = 0.27
HOME_JOINT_ANGLES: tuple[float, ...] = tuple(
    angle for _ in LEGS for angle in (0.0, 0.9, -1.8)
)
DEFAULT_TIMESTEP = 0.002

# MuJoCo 几何分组。Go2 的 MJCF 里 ``3`` 组装着**所有会碰撞的图元** —— 躯干盒/柱/球、
# 髋/大腿/小腿的碰撞体, 以及四个脚的球体; 所以 ``geom_group == 3`` 恰好等价于
# 「这个几何体可能接触地形」。地形本身在 :data:`TERRAIN_GEOM_GROUP` 组里,
# 于是 :func:`extract_meta` 用这个常量把「脚」从其它碰撞体里筛出来。
COLLISION_GEOM_GROUP = 3


# --------------------------------------------------------------------------- #
# MJCF 文本装配
# --------------------------------------------------------------------------- #

_OUTER = re.compile(r"\s*<mujoco\b[^>]*>(?P<body>.*)</mujoco>\s*$", re.DOTALL)
_OPTION = re.compile(r"<option\b[^>]*/>")
_COMPILER = re.compile(r"<compiler\b[^>]*/>")
_MESHDIR = re.compile(r'meshdir\s*=\s*"[^"]*"')


@dataclass
class SceneOptions:
    """编译场景时叠加到 Go2 模型上的覆盖项。"""

    timestep: float = DEFAULT_TIMESTEP
    integrator: str = "implicitfast"
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    # 为在薄楼梯边缘上的约 15 kg 四足提供求解器 / 接触稳定性。
    iterations: int = 50
    ls_iterations: int = 20
    solver: str = "Newton"
    # 为僵硬的脚-盒接触提供类似恢复系数的柔软度。
    add_ground_texture: bool = True
    extra_visual: str = ""
    nconmax: int | None = None
    njmax: int | None = None


def _go2_inner_xml() -> str:
    """返回去掉外层 <mujoco> 包装后的 go2.xml 内容。

    ``meshdir`` 被刻意保留为原本的相对路径 ``"assets"``：网格内容由
    :func:`go2_assets` 从内存提供，所以完全不会发生目录查找（见模块 docstring）。
    """
    text = GO2_XML.read_text(encoding="utf-8")
    match = _OUTER.match(text)
    if not match:
        raise ValueError(f"could not parse {GO2_XML}: missing <mujoco> wrapper")
    inner = match.group("body")

    if not _MESHDIR.search(inner):
        # 防御性处理：没有 meshdir 时 MuJoCo 会去 CWD 里找。
        inner, n = _COMPILER.subn(
            lambda m: m.group(0)[:-2] + ' meshdir="assets"/>', inner, count=1
        )
        if n == 0:
            raise ValueError(f"could not parse {GO2_XML}: missing <compiler/>")
    return inner


@functools.lru_cache(maxsize=1)
def go2_assets() -> dict[str, bytes]:
    """按 basename 索引的网格文件内容，供 MuJoCo 的内存 VFS 使用。

    只读一次并缓存：16 个 OBJ 文件合计约 27 MB，每个 worker 进程只付一次
    这份开销。

    键必须是*裸 basename*：MuJoCo 会拒绝含重复文件名的映射
    （"Repeated file name in assets dict"），所以不要另外注册
    ``assets/<name>`` 别名。
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
    """组装出可编译的 MJCF 字符串：Go2 + 地面 / 地形 + 灯光。

    Parameters
    ----------
    terrain_xml:
        作为 ``<worldbody>`` 最后一批子元素插入的原始 XML。用于楼梯、
        heightfield、障碍物以及任何逐回合变化的几何。
    floor_geom:
        可选的地面平面 ``<geom .../>`` 字符串。传 ``None`` 表示不要地板
        （地形自带）。
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
    """位于高度 ``z`` 的一块带贴图的无限平面。

    这块平面带 ``group=TERRAIN_GEOM_GROUP``，这样 :mod:`robotdog.tools.touch`
    里的前向测距能看到地面，却永远看不到机器人。选这个 group 要小心 ——
    MuJoCo 默认隐藏 group 3-5，选错会让整个世界从渲染里消失。
    """
    return (
        f'<geom name="floor" type="plane" pos="0 0 {z:g}" size="{size:g} {size:g} 0.05" '
        f'condim="3" friction="0.9 0.005 0.0001" material="groundplane" '
        f'group="{TERRAIN_GEOM_GROUP}"/>'
    )


# --------------------------------------------------------------------------- #
# 编译后模型元数据
# --------------------------------------------------------------------------- #


@dataclass
class Go2Meta:
    """编译后的 Go2 场景中解析出来的 id / 索引。"""

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

    # 机器人碰撞 geom 的拆分（全部 ``geom_group == 3``）。
    #
    # ``trunk_geom_ids`` 是焊在 ``base`` 上的三个碰撞图元
    # （box + cylinder + sphere）：这里出现接触意味着躯干本身撞上了地形，
    # 也就是真摔。
    #
    # ``leg_geom_ids`` 是其余所有机器人碰撞图元（髋 / 大腿 / 小腿）减去四个
    # 脚球。这里出现接触是小腿 / 膝盖在机器人仍然站得住、还在探索时*蹭*到
    # 立面 —— 必须扣分，而不是立刻判死。
    trunk_geom_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    leg_geom_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    nq: int = 0
    nv: int = 0
    nu: int = 0

def _resolve_id(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx < 0:
        raise KeyError(f"{objtype} named {name!r} not found in compiled model")
    return int(idx)


def extract_meta(model: mujoco.MjModel) -> Go2Meta:
    """从编译后的模型里读回环境需要的每一个 id / 索引。"""
    joint_ids = np.array(
        [_resolve_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES], dtype=int
    )
    actuator_ids = np.array(
        [_resolve_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES], dtype=int
    )
    foot_geom_ids = np.array(
        [_resolve_id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOM_NAMES], dtype=int
    )
    # 脚是球体；geom_xpos 给的是球心，所以算真实离地间隙需要它。
    foot_radius = float(model.geom_size[foot_geom_ids[0], 0])

    base_body_id = _resolve_id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    imu_site_id = _resolve_id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")

    # 自由关节始终占用最前面的 7 个 qpos / 6 个 qvel。
    qpos_joint_adr = np.array([model.jnt_qposadr[j] for j in joint_ids], dtype=int)
    qvel_joint_adr = np.array([model.jnt_dofadr[j] for j in joint_ids], dtype=int)
    dof_joint_adr = qvel_joint_adr

    joint_ranges = np.array([model.jnt_range[j] for j in joint_ids], dtype=float)
    ctrl_ranges = np.array([model.actuator_ctrlrange[a] for a in actuator_ids], dtype=float)

    home_joint_angles = np.asarray(HOME_JOINT_ANGLES, dtype=float)
    home_qpos = np.zeros(model.nq, dtype=float)
    home_qpos[2] = BASE_HEIGHT_HOME
    home_qpos[3] = 1.0  # 四元数的 w 分量
    home_qpos[qpos_joint_adr] = home_joint_angles

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    # ``mj_name2id`` 找不到名字时返回 ``-1``。不把它剔掉的话，下面 terrain_ids
    # 会把 -1 当成「最后一个 geom」并进地形集合，于是某一条腿的碰撞 geom 被静默
    # 当成地面，接触判定与奖励全都错。宁可在这里就炸。
    if floor_id < 0:
        raise RuntimeError(
            "场景里没有名为 'floor' 的 geom；default_floor_geom() 与场景 XML 不一致。"
        )
    stair_ids = [
        g
        for g in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("stair_")
    ]

    # 在相关处把机器人自己的 12 个 body 从质量求和中排除。
    total_mass = float(model.body_mass[1:].sum())

    # --- 把机器人的碰撞 geom 拆成躯干 / 腿 -----------------------------------------------
    # Go2 的 MJCF 把纯视觉网格放在 group 2，把每一个碰撞图元放在 group 3，
    # 所以 ``group == 3`` 恰好等价于「能和地形产生接触」。
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
    """构建并编译一个场景，返回 (model, meta, xml_text)。"""
    if floor_geom is None:
        floor_geom = default_floor_geom()
    xml = build_scene_xml(terrain_xml, opts=opts, floor_geom=floor_geom)

    if write_to is not None:
        write_to.parent.mkdir(parents=True, exist_ok=True)
        write_to.write_text(xml, encoding="utf-8")

    try:
        model = mujoco.MjModel.from_xml_string(xml, assets=go2_assets())
    except Exception as exc:  # 把生成的 XML 暴露出来便于调试
        if write_to is None:
            from robotdog.tools.paths import GENERATED_DIR

            GENERATED_DIR.mkdir(parents=True, exist_ok=True)
            dump = GENERATED_DIR / "failed_scene.xml"
            dump.write_text(xml, encoding="utf-8")
            raise RuntimeError(f"MJCF compile failed; dumped to {dump}") from exc
        raise

    meta = extract_meta(model)
    return model, meta, xml


def home_qpos(meta: Go2Meta, *, x: float = 0.0, y: float = 0.0, z: float | None = None,
              yaw: float = 0.0) -> np.ndarray:
    """home keyframe 的 qpos，可选地平移 / 定向（绕世界 Z 轴的 yaw）。"""
    qpos = meta.home_qpos.copy()
    qpos[0] = x
    qpos[1] = y
    if z is not None:
        qpos[2] = z
    half = yaw / 2.0
    qpos[3:7] = (np.cos(half), 0.0, 0.0, np.sin(half))
    return qpos


# --------------------------------------------------------------------------- #
# 帧约定
# --------------------------------------------------------------------------- #

def quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    """(w,x,y,z) -> 3x3 旋转矩阵。"""
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
    """(w,x,y,z) -> 绕世界 X/Y/Z 轴的 (roll, pitch, yaw)（ZYX 顺序）。"""
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
