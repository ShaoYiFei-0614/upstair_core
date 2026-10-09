# `robotdog/go2_model.py` —— Go2 MJCF 组装与场景编译

## 1. 一句话定位

属于**环境层**：把 vendored 的 Unitree Go2 MJCF（`models/unitree_go2/go2.xml`）读进来，
用字符串手术注入地形 / 灯光 / 材质与 `<option>` 覆盖，编译成一个**自包含的 MJCF**，
再把环境需要的全部 id / 索引 / 限位 / 质量读回成 `Go2Meta`。
它是「模型装配」层，不是「运动控制」层。

## 2. 职责边界

**做：**

* 读取 `GO2_XML`、剥掉外层 `<mujoco>` 包装、在原 `<option/>` 上做覆盖（`_go2_inner_xml` / `_option_xml`）。
* 内存 VFS：把 `GO2_ASSETS_DIR` 下的 16 个 `.obj` 读成 `{basename: bytes}` 交给
  `MjModel.from_xml_string(assets=...)`（`go2_assets`），从而**绕开非 ASCII 路径**。
* 组装可编译的 MJCF 字符串：地面/地形、两盏平行光、`<asset>` 追加地面材质与天空盒、
  `<visual>` 覆盖（`build_scene_xml`）。
* 提供默认地板 geom（`default_floor_geom`）。
* 编译并返回 `(MjModel, Go2Meta, xml_text)`；失败时把 XML 落盘到 `.generated/failed_scene.xml`
  （`compile_scene`）。
* 从编译后的模型解析 id / 地址 / 限位：`extract_meta` + `_resolve_id`
  （joint / actuator / foot geom / base body / imu site / 质量 / qpos·qvel 地址 /
  关节限位 / 控制限位 / 楼梯 geom / 机器人碰撞 geom 的躯干-腿拆分）。
* 提供 home qpos 与姿态换算工具：`home_qpos` / `quat_to_mat` / `quat_to_rpy`。

**不做：**

* 不修改 `models/` 下任何 vendored 文件（只读）。
* 不定义楼梯参数与高度剖面（`robotdog/tools/terrain.py` 的 `StairSpec` / `build_terrain_xml`）。
* 不建 `MjData`、不推进物理、不做控制 —— 那是 `robotdog/env.py`。
* **不做任何接触力读取与射线投射**。仓库里已经没有触觉模块（`robotdog/tools/touch.py`
  已在「盲走」重构中删除），所以本模块也不必为测距仪准备任何 geom 分组约定。
* 不训练、不评估、不渲染、不导出。
* 不使用 `<include>`（见 §3.1）。

## 3. 关键设计决策与理由

### 3.1 用字符串手术 + 自建 scratch 场景，**不用** `<include>`

vendored 的 `models/unitree_go2/scene.xml` 本身就是一个「`<include file="go2.xml"/>` + 地面 +
灯光」的场景，看上去正合用。**被否决**，理由写在模块 docstring 里：

> `<include>` is deliberately not used: MuJoCo resolves relative asset directories against the
> *main* file, and compiling from a string has no main file directory to resolve against.

即：`from_xml_string()` 没有「主文件目录」这个概念，而 `go2.xml` 的 `<compiler meshdir="assets"/>`
是相对路径；include 展开后 meshdir 的解析基准直接丢失。所以改为：

1. `_go2_inner_xml()` 用 `_OUTER` 正则取出 `<mujoco ...> ... </mujoco>` 的**内部文本**，
   丢掉外层包装（不复制、不重写 vendored 文件）。
2. 在内部文本上替换 `<option .../>`、追加 `<asset>` 子元素、追加 `<worldbody>` 子元素。
3. 重新套上 `<mujoco model="go2_stairs">` 头尾，得到一个**自包含字符串**。

这样 vendored 模型永远保持原样，而场景（每回合换楼梯）由 `robotdog/tools/terrain.py` 的
`build_terrain_xml(spec)` 产出并以参数形式注入。副产品是 `.generated/` 下的 scratch 场景：
`compile_scene(write_to=...)` 可以把最终字符串落盘给人看，失败时**无条件**落盘成
`.generated/failed_scene.xml`，因为从 Python 里 `mujoco.MjModel.from_xml_string` 抛出的异常
通常只说「某行某处」，没有 XML 很难定位。

`TERRAIN_GEOM_GROUP = 1` 的问题在这里只涉及一句：地形与地板必须落在 **group 1**
（MuJoCo 默认隐藏 group 3/4/5，而 Unitree 的 MJCF 占用了 2/3），细节归
`docs/robotdog/tools/terrain.md`，见 §3.5。

### 3.2 `meshdir` 保持相对 `"assets"`，网格走内存 VFS

工作区路径含中文（本文档所在仓就是 `D:\项目\RobortDog\upstair_core`）。
MuJoCo 的资源读取器把窄字符（`const char*`）文件名按 **Windows ANSI 代码页**解析，
所以让 `meshdir` 指向中文路径会失败，报错形如：

```
Error opening file 'D:/<mojibake>/models/unitree_go2/assets/base_0.obj'
```

* **否决方案**：自定义 `mujoco.MjVfs` 回调。`MjVfs` 只向 Python 暴露 `close()`，
  拿不到 `fopen`/`fread` 回调。
* **否决方案**：把 `meshdir` 改写成绝对路径、或把资产复制到纯 ASCII 目录。
  前者仍要经过同一套窄字符解码；后者要在首次运行时复制 27 MB 网格。
* **采用的方案**：`meshdir` **保持原样**（相对 `"assets"`），网格内容通过
  `MjModel.from_xml_string(xml, assets=go2_assets())` 由内存提供，
  于是**根本没有目录查找发生**（`_go2_inner_xml` 的 docstring 明说这一点）。
  这也是本项目唯一能用的修法。

`go2_assets()` 用 `@functools.lru_cache(maxsize=1)` 缓存：16 个 OBJ 合计
**28,409,145 B ≈ 27.1 MiB**，每个 worker 进程只付一次成本。

* 键必须是**裸文件名**（`path.name`）。**不能**同时注册 `assets/<name>` 别名 ——
  MuJoCo 会因为映射里出现重复文件名而报 `Repeated file name in assets dict`。
* **否决方案**：不做缓存，每次编译重读 27 MB。8 环境 + 课程每次换楼梯都会重编译，
  这个成本会直接吃掉采样吞吐。

### 3.3 `_MESHDIR` 防御分支：只在 meshdir 真的缺失时才动手

`_go2_inner_xml()` 先用 `_MESHDIR` 检查内部文本是否已有 `meshdir="..."`。若有 —— 直接放行
（vendored 现状就是 `<compiler angle="radian" meshdir="assets" autolimits="true"/>`）。
若没有 —— 用 `_COMPILER.subn(lambda m: m.group(0)[:-2] + ' meshdir="assets"/>', inner, count=1)`
把属性插进 `<compiler .../>` 的 `/>` 之前；连 `<compiler>` 都找不到就抛
`ValueError(f"could not parse {GO2_XML}: missing <compiler/>")`。

理由是**防御性的**：没有 `meshdir` 时 MuJoCo 会去 **CWD** 找网格，那是个与调用者当前目录
相关的、极难复现的失败模式。

### 3.4 `<option>` 覆盖：丢掉原值、只保留 cone/impratio

`_option_xml(opts)` 生成一条**完整替换**用的 `<option .../>`：

```xml
<option cone="elliptic" impratio="100" timestep="..." integrator="..." gravity="..."
        iterations="..." ls_iterations="..." solver="..."/>
```

`cone="elliptic"` 与 `impratio="100"` 是硬编码保留的（vendored 的 `<option cone="elliptic"
impratio="100"/>` 本来是默认值来源，替换后由本函数负责保住它）。
`SceneOptions` 里那些「solver / contact stability」参数（`iterations=50`、`ls_iterations=20`、
`solver="Newton"`）是为了让 **~15 kg 四足站在薄台阶边缘上**时接触求解稳定；默认值
（`iterations=100` 是 MuJoCo 默认，但 `solver` 默认 `Newton` 而 `ls_iterations` 默认更小）
在早期实验里出现过渡抖动，所以这里写死。

`nconmax` / `njmax` 默认为 `None`，此时**不产出 `<size .../>`**；只有显式设置时才追加，
这是为了不改变默认行为（旧 checkpoint 的位对位复现）。

### 3.5 `default_floor_geom` 与 group 1

```python
f'<geom name="floor" type="plane" pos="0 0 {z:g}" size="{size:g} {size:g} 0.05" '
f'condim="3" friction="0.9 0.005 0.0001" material="groundplane" group="{TERRAIN_GEOM_GROUP}"/>'
```

* 地板必须带 `group=TERRAIN_GEOM_GROUP(=1)`，理由是**可见性**：group 1 落在
  `mjv_defaultOption` 的默认可见范围 `[1,1,1,0,0,0]` 内。把地形盖进 group 4 会让楼梯
  在全部离屏渲染里消失、而物理照常爬（`experience.md` §1.2）。
  历史上还有第二个理由（触觉测距仪只启用地形组），它随触觉模块一起被删除。
  `Go2StairsEnv.render()` 另外强制打开全部六组，所以改这个常量也不会再静默隐藏世界。
* `friction="0.9 0.005 0.0001"` 与台阶块一致（`tools/terrain.py` 的 `DEFAULT_STAIR_FRICTION`），
  避免「地板和台阶摩擦不同」这种隐性口径差。

### 3.6 `COLLISION_GEOM_GROUP = 3`：靠 geom group 把机器人拆成躯干 / 腿

`extract_meta()` 里「哪些几何能碰地形」完全由 Go2 MJCF 的约定决定：

* `group="2"`：纯视觉网格（`contype="0" conaffinity="0"`），永不碰撞。
* `group="3"`：**全部碰撞图元** —— 躯干 box/cylinder/sphere、hip/thigh/calf 形状、四个足球。

所以 `geom_group == 3` 精确等于「这个几何能和地形产生接触」。在此基础上再分两类
（`Go2Meta.trunk_geom_ids` / `leg_geom_ids`）：

* `trunk_geom_ids`：`geom_group == 3` 且 `geom_bodyid == base_body_id` —— 焊在 `base` 上的
  躯干图元。这里出现接触 = **真摔**（默认唯一致命的接触）。
* `leg_geom_ids`：其余（hip/thigh/calf）**减去**四个足球。这里出现接触 = 小腿/膝盖**蹭**到立面，
  但机器人还站得住 —— 必须**扣分而不是立刻判死**，否则试错被杀掉
  （`experience.md` §2.4，`robotdog/env.py` 的 `leg_collision` / `terminate_contact`）。

**被否决的方案**：按几何**名字**硬编码哪些是躯干。名字是 vendored XML 的私有约定，
换个模型版本就烂掉；用 group + bodyid 是结构性的、且这个 MJCF 的 group 约定本来就是
「可碰撞 / 仅视觉」的语义。

**trap**：如果哪天 vendored XML 把碰撞图元改到别的 group，`trunk_ids` 与 `leg_ids` 会同时为空，
而 `env._contact_flags()` 对空集是**显式提前返回 `(False, False)`** —— 接触致命性会静默失效，
不会报错。

### 3.7 home 姿态来自**镜像常量**，不是读 `<keyframe>`

`BASE_HEIGHT_HOME = 0.27`、`HOME_JOINT_ANGLES = (0.0, 0.9, -1.8) × 4 条腿` 是硬编码常量，
`extract_meta()` 用它们**构造** `home_qpos`（`qpos[2]=0.27`、`qpos[3]=1.0`（四元数 w）、
`qpos[qpos_joint_adr]=home_joint_angles`），而不是从模型里读 keyframe。

* 理由：`home_qpos` 有明确的形状契约（`(model.nq,)`，自由关节占前 7 个 qpos / 6 个 qvel），
  从 keyframe 读要处理「keyframe 有没有、ctrl 要不要一起读、class 继承」等分支；
  常量足够，而且 `extract_meta` 会**验证**它们与模型一致（编译出来的模型必须接受这些角度）。
* 代价：常量与 vendored XML 之间没有自动一致性检查。当前两者一致
  （`go2.xml` 第 215 行块里 `<key name="home" qpos="0 0 0.27 1 0 0 0 0 0.9 -1.8 …"/>`），
  但改模型不会自动改常量 —— 见 §9。
* `NOMINAL_MASS = 15.206408` 只是**注释性常量**（`extract_meta` 用的是
  `float(model.body_mass[1:].sum())`，跳过 world body 0），用于对照
  `6.921 + 4×(0.678 + 1.152 + 0.241352)`。

### 3.8 «extract_meta» 只解析一次，之后全靠它

`extract_meta()` 把「以后会反复查名字」变成「只查一次 id，之后用数组索引」：
`joint_ids` / `actuator_ids` / `foot_geom_ids` / `qpos_joint_adr` / `qvel_joint_adr` /
`joint_ranges` / `ctrl_ranges` 都是 `np.ndarray`。理由：`mujoco.mj_name2id` 是字符串查找，
50 Hz × 8 环境 × 每控制步多次调用是不可接受的；而 `qpos[self.meta.qpos_joint_adr]` 是纯索引。

`dof_joint_adr = qvel_joint_adr`（`jnt_dofadr` 就是 `qvel` 地址，两个名字保留下来是因为
调用点语义不同）。

### 3.9 `quat_to_mat` / `quat_to_rpy` 放在本模块

这两个换算是**帧约定**的一部分（`env.py` 的 docstring 有一整段帧约定说明），
而 `go2_model.py` 是「模型与坐标系」这一层的边界，所以放这里而不是 `env.py`：

* `quat_to_mat`：`(w,x,y,z) → 3×3` 旋转矩阵，展开式写死，不用 `mujoco.mju_quat2Mat`
  以免再多一层 `mujoco` 调用（这两个函数在观测与奖励的每步路径上）。
* `quat_to_rpy`：`(w,x,y,z) → (roll, pitch, yaw)`，**ZYX 顺序、绕世界 X/Y/Z 轴**；
  `pitch` 用 `arcsin(clip(..., -1, 1))` 保护，避免数值越界产生 `nan`。
* `env.base_rpy` 与 `env._projected_gravity` 都直接依赖这两个函数的输出口径。

## 4. 公开 API

### 4.1 模块级常量与正则

```python
LEGS: tuple[str, str, str, str] = ("FL", "FR", "RL", "RR")
JOINT_SUFFIXES: tuple[str, str, str] = ("hip", "thigh", "calf")

ACTUATOR_NAMES: tuple[str, ...] = tuple(f"{leg}_{s}" for leg in LEGS for s in JOINT_SUFFIXES)
JOINT_NAMES: tuple[str, ...] = tuple(f"{name}_joint" for name in ACTUATOR_NAMES)
FOOT_GEOM_NAMES: tuple[str, ...] = LEGS

BASE_HEIGHT_HOME = 0.27
HOME_JOINT_ANGLES: tuple[float, ...] = tuple(angle for _ in LEGS for angle in (0.0, 0.9, -1.8))

NOMINAL_MASS = 15.206408
DEFAULT_TIMESTEP = 0.002
DEFAULT_CONTROL_DECIMATION = 10          # -> 50 Hz control, 500 Hz physics

VISUAL_GEOM_GROUP = 2
COLLISION_GEOM_GROUP = 3

_OUTER = re.compile(r"\s*<mujoco\b[^>]*>(?P<body>.*)</mujoco>\s*$", re.DOTALL)
_OPTION = re.compile(r"<option\b[^>]*/>")
_COMPILER = re.compile(r"<compiler\b[^>]*/>")
_MESHDIR = re.compile(r'meshdir\s*=\s*"[^"]*"')
```

* `ACTUATOR_NAMES` 的顺序就是 `go2.xml` 声明的执行器顺序，也**就是动作向量的顺序**：
  `FL_hip, FL_thigh, FL_calf, FR_hip, …, RR_hip, RR_thigh, RR_calf`。
* `JOINT_NAMES` = `ACTUATOR_NAMES` 每个名字加 `_joint` 后缀（MJCF 里 joint 与 actuator 不同名）。
* `FOOT_GEOM_NAMES = LEGS`，即足球 geom 就叫 `FL` / `FR` / `RL` / `RR`
  （`assets` 里是 `<geom name="FL" class="foot"/>` 之类）。

### 4.2 `SceneOptions`

```python
@dataclass
class SceneOptions:
    timestep: float = DEFAULT_TIMESTEP                        # 0.002
    integrator: str = "implicitfast"
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    iterations: int = 50
    ls_iterations: int = 20
    solver: str = "Newton"
    add_ground_texture: bool = True
    extra_visual: str = ""
    nconmax: int | None = None
    njmax: int | None = None
```

* `add_ground_texture=True` 时 `build_scene_xml` 追加 `_GROUND_ASSETS`（checker 地面 +
  天空盒 + `stairmat`/`edgemat` 材质）与 `_GROUND_VISUAL`；`False` 时两者都不加。
* `extra_visual` 非空时被插到 `</visual>` **之前**（在 `_GROUND_VISUAL` 内部替换），
  所以它是「补充 visual 子元素」而不是「替换」。
* `nconmax` / `njmax`：只要有一个非 `None`，就追加一条 `<size .../>`
  （只写非 `None` 的那些属性）。

### 4.3 文本装配

```python
def _go2_inner_xml() -> str
```
返回 `go2.xml` 去掉外层 `<mujoco>` 包装的内部文本。`meshdir` **保持原始相对值 `"assets"`**
（网格由 `go2_assets()` 内存提供，不做任何目录查找）。
异常：`ValueError(f"could not parse {GO2_XML}: missing <mujoco> wrapper")`；
若内部文本既无 `meshdir` 又无法插入，`ValueError(f"could not parse {GO2_XML}: missing <compiler/>")`。

```python
@functools.lru_cache(maxsize=1)
def go2_assets() -> dict[str, bytes]
```
`GO2_ASSETS_DIR` 下全部 `*.obj` 的**内容**，键为**裸文件名** `path.name`。只读一次并缓存。
异常：`FileNotFoundError(f"no .obj meshes in {GO2_ASSETS_DIR}; run scripts/download_go2.py")`。
**不要**在返回值里加 `assets/<name>` 别名（MuJoCo 会报 `Repeated file name in assets dict`）。

```python
def _option_xml(opts: SceneOptions) -> str
```
生成替换用的 `<option cone="elliptic" impratio="100" .../>`，可选追加 `<size .../>`。

```python
def build_scene_xml(terrain_xml: str = "", *, opts: SceneOptions | None = None,
                    floor_geom: str | None = None, include_lights: bool = True) -> str
```
组装可编译的 MJCF 字符串：`'<mujoco model="go2_stairs">\n' + inner + visual + "\n</mujoco>\n"`。

* `terrain_xml`：原样插到 `<worldbody>` 的**最后**（楼梯、heightfield、障碍、每回合几何）。
* `floor_geom`：`<geom .../>` 字符串；`None` 表示**不加地板**（由 terrain 自带），
  空字符串 `""` 会走「追加一个空字符串」的分支，等价于不加地板但会创建一个多余的换行。
* `include_lights`：追加两盏 `castshadow="false"` 的灯（`pos="0 0 3"` 平行光 +
  `pos="3 -3 3"` 补光）。
* `opts=None` ⇒ `SceneOptions()`（不是无覆盖，而是用默认覆盖）。

```python
def default_floor_geom(z: float = 0.0, size: float = 60.0) -> str
```
返回一条 `<geom name="floor" type="plane" pos="0 0 {z:g}" size="{size:g} {size:g} 0.05"
condim="3" friction="0.9 0.005 0.0001" material="groundplane" group="1"/>`。
`group=TERRAIN_GEOM_GROUP` 是 load-bearing 的（§3.5）。

### 4.4 `Go2Meta`

```python
@dataclass
class Go2Meta:
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

    joint_ranges: np.ndarray            # (12, 2)
    ctrl_ranges: np.ndarray             # (12, 2)
    home_qpos: np.ndarray               # (model.nq,)  -- 注释写的是 (19,)
    home_joint_angles: np.ndarray       # (12,)
    total_mass: float
    floor_geom_id: int
    stair_geom_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    trunk_geom_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    leg_geom_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    nq: int = 0
    nv: int = 0
    nu: int = 0

    @property
    def num_joints(self) -> int: ...    # len(self.joint_names)
```

* 24 个字段里前 18 个是**必填**（无默认值），后 6 个（`stair_geom_ids`、`trunk_geom_ids`、
  `leg_geom_ids`、`nq`、`nv`、`nu`）有默认值。
* `foot_radius = float(model.geom_size[foot_geom_ids[0], 0])`：脚是**球体**，
  `geom_xpos` 给的是**球心**，所以真实离地间隙必须减这个半径。
* `home_qpos[3] == 1.0` 是四元数的 `w` 分量。
* `total_mass = float(model.body_mass[1:].sum())` —— **跳过 body 0（world）**。
* `dof_joint_adr` 就是 `qvel_joint_adr` 的同一份数组（`jnt_dofadr`）。

```python
def _resolve_id(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int
```
`mj_name2id` 包装：`idx < 0` 时抛
`KeyError(f"{objtype} named {name!r} not found in compiled model")`，否则返回 `int(idx)`。

```python
def extract_meta(model: mujoco.MjModel) -> Go2Meta
```
解析全部 id / 地址 / 限位 / 质量 / geom 分组。异常：`KeyError`（缺 joint / actuator / geom /
body `base` / site `imu` 之一）。

```python
def compile_scene(terrain_xml: str = "", *, opts: SceneOptions | None = None,
                  floor_geom: str | None = None,
                  write_to: Path | None = None) -> tuple[mujoco.MjModel, Go2Meta, str]
```
`floor_geom is None` 时自动用 `default_floor_geom()`；`write_to` 非 `None` 时先把 XML 写盘；
然后 `mujoco.MjModel.from_xml_string(xml, assets=go2_assets())`；最后 `extract_meta(model)`。
异常：

* 编译失败且 `write_to is None` → 把 XML 落到 `GENERATED_DIR / "failed_scene.xml"` 并抛
  `RuntimeError(f"MJCF compile failed; dumped to {dump}")`（异常链保留原错误）；
* 编译失败且 `write_to` 非 `None` → **直接重抛原异常**（XML 已经在调用者指定的位置了）；
* 缺网格 → 由 `go2_assets()` 抛 `FileNotFoundError`。

```python
def home_qpos(meta: Go2Meta, *, x: float = 0.0, y: float = 0.0,
              z: float | None = None, yaw: float = 0.0) -> np.ndarray
```
`meta.home_qpos.copy()`，写 `qpos[0]=x`、`qpos[1]=y`，`z is not None` 时写 `qpos[2]=z`，
并按 `qpos[3:7] = (cos(yaw/2), 0.0, 0.0, sin(yaw/2))` 设置纯 yaw 姿态。
`z=None` 时保留 home 高度 `0.27`。

```python
def quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray
```
`(w,x,y,z) → 3×3` 旋转矩阵（`float64`）。

```python
def quat_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray
```
`(w,x,y,z) → (roll, pitch, yaw)`，ZYX 顺序、绕世界 X/Y/Z 轴，`float64`。

## 5. 内部结构与数据流

### 5.1 `compile_scene()` 主管线

```
compile_scene(terrain_xml, opts, floor_geom, write_to)
  → floor_geom = default_floor_geom()                     （仅当 None）
  → build_scene_xml(terrain_xml, opts=opts, floor_geom=floor_geom)
      → _go2_inner_xml()
          → GO2_XML.read_text(encoding="utf-8")
          → _OUTER.match() → group("body")                （缺包装 → ValueError）
          → _MESHDIR.search()                             （已有 meshdir → 放行）
              否则 _COMPILER.subn(...)                    （找不到 → ValueError）
      → _OPTION.sub(_option_xml(opts), inner, count=1)
      → inner.replace("</asset>", _GROUND_ASSETS + "  </asset>", 1)     （add_ground_texture）
      → worldbody 追加：floor_geom? → lights? → terrain_xml?
          inner.replace("</worldbody>", "\n".join(additions) + "\n  </worldbody>", 1)
      → visual = _GROUND_VISUAL（add_ground_texture）；extra_visual 插到 </visual> 之前
      → '<mujoco model="go2_stairs">' + inner + visual + '</mujoco>'
  → write_to 非 None → mkdir(parents=True) + write_text(xml, encoding="utf-8")
  → mujoco.MjModel.from_xml_string(xml, assets=go2_assets())
      → 失败且 write_to is None → GENERATED_DIR.mkdir() →
        GENERATED_DIR/"failed_scene.xml".write_text(xml) → RuntimeError(...)
  → extract_meta(model)
  → (model, meta, xml)
```

### 5.2 `extract_meta()` 管线

```
extract_meta(model)
  → __resolve_id(mjOBJ_JOINT,    JOINT_NAMES[i])    × 12   → joint_ids
  → _resolve_id(mjOBJ_ACTUATOR, ACTUATOR_NAMES[i])  × 12   → actuator_ids
  → _resolve_id(mjOBJ_GEOM,     FOOT_GEOM_NAMES[i]) ×  4   → foot_geom_ids
  → foot_radius = geom_size[foot_geom_ids[0], 0]
  → _resolve_id(mjOBJ_BODY, "base")                        → base_body_id
  → _resolve_id(mjOBJ_SITE, "imu")                         → imu_site_id
  → qpos_joint_adr = jnt_qposadr[joint_ids]
  → qvel_joint_adr = jnt_dofadr[joint_ids] ; dof_joint_adr = qvel_joint_adr
  → joint_ranges = jnt_range[joint_ids] (12,2)
  → ctrl_ranges  = actuator_ctrlrange[actuator_ids] (12,2)
  → home_joint_angles = asarray(HOME_JOINT_ANGLES)
  → home_qpos = zeros(nq) ; [2]=BASE_HEIGHT_HOME ; [3]=1.0 ; [qpos_joint_adr]=home_joint_angles
  → floor_id  = mj_name2id(mjOBJ_GEOM, "floor")            （**不检查 < 0**）
  → stair_ids = [g for g in range(ngeom) if name(g).startswith("stair_")]
  → total_mass = body_mass[1:].sum()
  → terrain_ids = {floor_id} | set(stair_ids) ; foot_set = set(foot_geom_ids)
    for g in range(ngeom):
        if geom_group[g] != COLLISION_GEOM_GROUP(3): continue
        if g in terrain_ids or g in foot_set:        continue
        if geom_bodyid[g] == base_body_id: trunk_ids.append(g)
        else:                              leg_ids.append(g)
  → Go2Meta(...)
```

### 5.3 两个换算函数

`quat_to_mat(quat_wxyz)`：解包 `w,x,y,z`，返回
`[[1-2(y²+z²), 2(xy-wz), 2(xz+wy)], [2(xy+wz), 1-2(x²+z²), 2(yz-wx)], [2(xz-wy), 2(yz+wx), 1-2(x²+y²)]]`。
`env.base_lin_vel_body` 用它做 `rot.T @ qvel[0:3]`。

`quat_to_rpy(quat_wxyz)`：
`roll = arctan2(2(wx+yz), 1-2(x²+y²))`、
`pitch = arcsin(clip(2(wy-zx), -1, 1))`、
`yaw = arctan2(2(wz+xy), 1-2(y²+z²))`。

### 5.4 被谁调用（数据流下游）

```
Go2StairsEnv._build_model()
  → build_terrain_xml(self.stair)          [tools/terrain]
  → compile_scene(..., opts=SceneOptions(timestep=cfg.timestep), floor_geom=default_floor_geom())
  → self.model, self.meta, self.xml

Go2StairsEnv.reset()
  → home_qpos(self.meta, x=…, y=…, z=…, yaw=…)

Go2StairsEnv.base_rpy / ._projected_gravity()
  → quat_to_rpy(...) / quat_to_mat(...)
```

## 6. 关键常量与默认值

**几何与质量**

| 名称 | 值 |
| --- | --- |
| `LEGS` | `("FL", "FR", "RL", "RR")` |
| `JOINT_SUFFIXES` | `("hip", "thigh", "calf")` |
| `ACTUATOR_NAMES` | `("FL_hip","FL_thigh","FL_calf","FR_hip","FR_thigh","FR_calf","RL_hip","RL_thigh","RL_calf","RR_hip","RR_thigh","RR_calf")` |
| `JOINT_NAMES` | 同上每个 + `"_joint"` |
| `FOOT_GEOM_NAMES` | `("FL", "FR", "RL", "RR")` |
| `BASE_HEIGHT_HOME` | `0.27` m |
| `HOME_JOINT_ANGLES` | `(0.0, 0.9, -1.8)` × 4 = 12 个值 |
| `NOMINAL_MASS` | `15.206408` kg（注释：`6.921 + 4×(0.678 + 1.152 + 0.241352)`） |
| `DEFAULT_TIMESTEP` | `0.002` s |
| `DEFAULT_CONTROL_DECIMATION` | `10`（⇒ 50 Hz 控制 / 500 Hz 物理） |
| `VISUAL_GEOM_GROUP` | `2` |
| `COLLISION_GEOM_GROUP` | `3` |

**`SceneOptions` 默认值**：`timestep=0.002`、`integrator="implicitfast"`、
`gravity=(0.0, 0.0, -9.81)`、`iterations=50`、`ls_iterations=20`、`solver="Newton"`、
`add_ground_texture=True`、`extra_visual=""`、`nconmax=None`、`njmax=None`。

**函数默认参数**：`build_scene_xml(terrain_xml="", opts=None, floor_geom=None, include_lights=True)`、
`default_floor_geom(z=0.0, size=60.0)`、`compile_scene(terrain_xml="", opts=None, floor_geom=None, write_to=None)`、
`home_qpos(meta, x=0.0, y=0.0, z=None, yaw=0.0)`。

**内联 XML 常量**

| 名称 | 内容要点 |
| --- | --- |
| `_GROUND_ASSETS` | `texture groundplane`（checker，`rgb1="0.25 0.32 0.40"`、`rgb2="0.14 0.19 0.25"`、`markrgb="0.85 0.85 0.85"`、300×300）、`material groundplane`（`texrepeat="8 8"`、`reflectance="0.05"`）、`texture skybox`（gradient，512×3072）、`material stairmat`（`rgba="0.62 0.60 0.55 1"`）、`material edgemat`（`rgba="0.95 0.72 0.15 1"`） |
| `_GROUND_VISUAL` | `<visual>`：headlight `diffuse="0.6 0.6 0.6"` / `ambient="0.35 0.35 0.35"` / `specular="0 0 0"`；`<rgba haze="0.15 0.25 0.35 1"/>`；`<global azimuth="-130" elevation="-20" offwidth="1280" offheight="720"/>` |
| 灯光（`include_lights`） | `<light pos="0 0 3" dir="0 0 -1" directional="true" castshadow="false"/>`、`<light pos="3 -3 3" dir="-1 1 -1" diffuse="0.35 0.35 0.35" castshadow="false"/>` |

**网格**：16 个 `.obj`，合计 28,409,145 B ≈ 27.1 MiB；`go2_assets()` 缓存 `maxsize=1`。

**vendored XML 关键行**（供交叉核对，`models/unitree_go2/go2.xml`）：
第 1 行 `<mujoco model="go2">`；第 2 行 `<compiler angle="radian" meshdir="assets" autolimits="true"/>`；
第 4 行 `<option cone="elliptic" impratio="100"/>`；第 39–61 行 `<asset>…</asset>`；
第 63–198 行 `<worldbody>…</worldbody>`；第 64 行 `<body name="base" pos="0 0 0.445" childclass="go2">`；
第 76 行 `<site name="imu" pos="-0.02557 0 0.04232"/>`；第 103/133/163/193 行
`<geom name="FL|FR|RL|RR" class="foot"/>`；第 200 行起 `<actuator>`；
第 215–218 行 `<keyframe><key name="home" qpos="0 0 0.27 1 0 0 0 0 0.9 -1.8 …"/>…</keyframe>`；
无 `<visual>`、无 `<size>`、无 `<sensor>`。同目录的 `scene.xml` 用 `<include file="go2.xml"/>`
并自带 visual/地面 —— **本模块刻意不用它**（§3.1）。

## 7. 与其它模块的关系

```
mujoco / numpy / functools / re / dataclasses / pathlib
        ↑
robotdog.go2_model
   ↑ 上游（本模块 import 它们）
   ├── robotdog.tools.paths   : GO2_ASSETS_DIR, GO2_XML (, GENERATED_DIR 在失败分支里延迟 import)
   └── robotdog.tools.terrain : TERRAIN_GEOM_GROUP
   ↓ 下游（依赖本模块）
   ├── robotdog.env      : SceneOptions, compile_scene, default_floor_geom, home_qpos,
   │                       quat_to_mat, quat_to_rpy
   └── scripts/*         : verifications/step2_go2.py（帧约定验证）、场景检查脚本
```

方向为什么是这样：

* 本模块**只依赖工具层**（`tools/paths` 给路径常量、`tools/terrain` 给地形组常量），
  不依赖 `env` / `gait` / `training` —— 所以它能被 `env` 依赖而不会成环。
* 分层规则要求 `tools/` **不 import** `robotdog.env`，因此 `TERRAIN_GEOM_GROUP` 定义在
  `tools/terrain.py`（工具层），由本模块 import，而不是反过来在 `go2_model` 里定义后
  让 `terrain` 反向依赖。**地形组的唯一真源在 `tools/terrain.py`。**
* `env.py` 依赖本模块与依赖 `tools/terrain.py` 是并行两条边：前者管「模型装好没有」，
  后者管「这一级台阶多高」。`compile_scene` 的信息流是
  `StairSpec → build_terrain_xml → terrain_xml → build_scene_xml → MjModel → Go2Meta → env`。
* `paths.py` 与本模块的耦合是**单向且只在模块加载时取值**：`GO2_XML` / `GO2_ASSETS_DIR` 在
  import 时算好；`GENERATED_DIR` 在失败分支里**延迟 import**，避免把「失败时才会用到的路径」
  放进正常路径的 import 序列。

## 8. 实测与经验教训

本节**只列引用**，结论请读原文：

* `experience.md` §1.1（踏步立面与顶面**共面接缝**把脚夹死 —— 它直接决定
  `tools/terrain.py` 的 `CREST_SEAM_OVERLAP` 与「实心块」几何；本模块负责把这段 XML 与 Go2
  装进同一个 `<worldbody>`）。
* `experience.md` §1.2（地形 geomgroup 被 MuJoCo 默认隐藏 ⇒ `TERRAIN_GEOM_GROUP = 1`；
  `default_floor_geom` 与 `env.render()` 都受这条影响）。
* `experience.md` §1.3（零高度台阶导致 `RuntimeError: MJCF compile failed` ——
  `compile_scene` 落盘 `failed_scene.xml` 的诊断通路就是为这类问题准备的）。
* `experience.md` §5.8（非 ASCII 路径在 C++ 层被按窄字符/ANSI 代码页解码的同类问题；
  `torch.jit.save` 的失败与本文档 §3.2 的 `meshdir` 失败是**同一根因**：
  C++ 侧拿到的是窄字符路径）。这也是本文档反复强调「不要给 `meshdir` 传绝对中文路径」的原因。

## 9. 已知限制与坑

1. **非 ASCII 工作区路径是硬约束**：`meshdir` 必须保持相对 `"assets"`，网格必须走
   `assets=go2_assets()`。任何「直接读磁盘网格」的改法都会重现
   `Error opening file 'D:/<mojibake>/models/unitree_go2/assets/base_0.obj'`。
   `mujoco.MjVfs` 只暴露 `close()`，没有自定义 VFS 回调可用。
2. **`go2_assets()` 的键必须是裸文件名**；加 `assets/<name>` 别名会触发
   `Repeated file name in assets dict`。
3. **`extract_meta()` 里的 `floor_geom_id` 用 `mj_name2id` 且不检查 `< 0`**：
   如果编译时没有地板 geom（例如 `compile_scene(..., floor_geom="")`），
   `meta.floor_geom_id == -1`，而 `env` 侧的 terrain 集合会把 `-1` 当成一个合法 geom id 收进去
   （恰好无害，但这是个隐性契约）。所有其它 id 都走 `_resolve_id()`，缺了就抛 `KeyError`。
4. **`_OUTER` / `_OPTION` / `_COMPILER` 正则只认自闭合标签**：`_OPTION = re.compile(r"<option\b[^>]*/>")`
   匹配的是 `<option .../>`。若 vendored XML 改成 `<option ...> … </option>`（非自闭合），
   `_OPTION.sub(..., count=1)` **静默不替换**，`SceneOptions` 全部失效而**没有任何报错**。
   同理 `inner.replace("</asset>", ...)` / `inner.replace("</worldbody>", ...)` 找不到锚点时
   也是静默跳过。字符串手术的代价就是没有断言。
5. **`<visual>` 会被追加到 Go2 内部内容之后**（`build_scene_xml` 的返回拼接），
   而不是插在文档头部。当前 vendored `go2.xml` **没有** `<visual>` 元素，所以不会产生重复；
   但如果以后 vendored XML 自带 `<visual>`，就会有两个 —— 那时应该改成「替换」而不是「追加」。
6. **`HOME_JOINT_ANGLES` / `BASE_HEIGHT_HOME` 与 vendored XML 没有一致性检查**。
   源码注释写的是「from go2.xml `<keyframe name="home">`」，实际 vendored 文件里是
   `<keyframe><key name="home" qpos="…"/>…</keyframe>`（`<keyframe>` 本身**没有** name 属性），
   而 `extract_meta()` 也**不读** keyframe —— 它用常量构造 `home_qpos`。
   模型换版本时必须手工同步这两个常量，否则 home 姿态静默漂移。
7. **模块 docstring 里的工作区路径是 `D:\项目\RobortDog\upstair`**，而本仓实际是
   `...\upstair_core`（`tools/paths.py` 的 `WORKSPACE = Path(__file__).resolve().parents[2]`）。
   docstring 是历史漂移，路径以 `paths.py` 为准。
8. **`GO2_SCENE_XML`（`models/unitree_go2/scene.xml`）在本模块里没有被使用** ——
   它是 vendored 的完整场景（`<include file="go2.xml"/>` + 地面 + visual），
   只作为「原版怎么写」的参考。训练路径上真正的场景由 `build_scene_xml` 生成。
9. **`trunk_geom_ids` / `leg_geom_ids` 依赖 vendored XML 的 group 约定**：碰撞图元必须在
   `geom_group == 3`，否则两者为空，`env._contact_flags()` 会显式返回 `(False, False)`，
   终止判据静默失效（不会抛错）。
10. **`terrain_ids` 用 `stair_` 前缀识别楼梯 geom**（`name(g).startswith("stair_")`）：
    给楼梯换名字前缀（`StairSpec.name_prefix`）会让 `stair_geom_ids` 变空，
    于是「足-地形接触」只认地板。`StairSpec.name_prefix` 的默认值就是 `"stair"`。
11. **`extract_meta()` 的 `foot_radius` 只取第一只脚的数据**（`foot_geom_ids[0]`）：
    四条腿的球半径若不一致，后三条会被静默用第一条的值。
12. **编译失败落盘的路径是 `.generated/failed_scene.xml`（固定名）**：
    连续两次失败会互相覆盖。`GENERATED_DIR` 也只在失败分支或调用者显式
    `write_to` 时才被创建。
13. **`compile_scene` 只有 `write_to` 为 `None` 时才落盘排错 XML**：调用者传了 `write_to`
    却编译失败时，异常原样抛出（XML 已被写到 `write_to`，但**未必**是最终形态，因为
    `write_to.write_text` 在编译之前）。
14. **`NOMINAL_MASS` 不参与任何计算**，只是注释用的对照常量；真值是
    `meta.total_mass = float(model.body_mass[1:].sum())`。不要把它当断言基准去比较
    —— 域随机化会改 `body_mass`（`env._apply_domain_randomization`），
    但 `Go2Meta.total_mass` 是**编译时**快照，不会跟着变。

## 10. 示例用法

```python
"""最小可运行：装一个 5 cm × 8 级的场景 → 编译 → 读回 id 与姿态。"""
from pathlib import Path

import mujoco
import numpy as np

from robotdog.go2_model import (
    SceneOptions,
    compile_scene,
    default_floor_geom,
    home_qpos,
    quat_to_mat,
    quat_to_rpy,
)
from robotdog.tools.terrain import StairSpec, build_terrain_xml

spec = StairSpec(step_height=0.05, tread=0.50, n_up=8, n_down=0)   # top_landing=1.20
terrain_xml = build_terrain_xml(spec)                             # "" ⇒ 平地为空串

model, meta, xml = compile_scene(
    terrain_xml,
    opts=SceneOptions(timestep=0.002, iterations=50, solver="Newton",
                      nconmax=200, njmax=400),
    floor_geom=default_floor_geom(),
    write_to=Path(".generated") / "demo_scene.xml",   # 人类可读的 scratch 场景
)

print("nq/nv/nu =", meta.nq, meta.nv, meta.nu, "num_joints =", meta.num_joints)
print("actuators =", meta.actuator_names[:3], "...")
print("foot_radius =", meta.foot_radius, "total_mass =", round(meta.total_mass, 6))
print("stair geoms =", meta.stair_geom_ids, "floor =", meta.floor_geom_id)
print("trunk geoms =", meta.trunk_geom_ids, "leg geoms =", meta.leg_geom_ids)
print("joint_ranges[0] =", meta.joint_ranges[0], "ctrl_ranges[0] =", meta.ctrl_ranges[0])

data = mujoco.MjData(model)
data.qpos[:] = home_qpos(meta, x=1.00, y=0.0, z=0.32, yaw=0.10)   # z=None 时用 home 的 0.27
mujoco.mj_forward(model, data)

print("qpos[0:3] =", np.round(data.qpos[0:3], 4))
print("rpy       =", np.round(quat_to_rpy(data.qpos[3:7]), 4))
print("R(q)^T e_x=", np.round(quat_to_mat(data.qpos[3:7]).T @ np.array([1.0, 0.0, 0.0]), 4))
```
