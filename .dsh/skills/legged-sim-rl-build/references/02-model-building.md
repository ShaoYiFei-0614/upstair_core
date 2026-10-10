# 模型进仓与场景组装

> 对应 `SKILL.md` 阶段 1–2（以及阶段 4 的 PD 增益）。素材取自 upstair_core（Unitree Go2 盲走楼梯 · MuJoCo 3.12 + gymnasium 1.3 + SB3 2.9） 的真实源码与实测记录。
> 具体数值只作**取证**用；换机器人时按文中的「换机器人要改什么」逐条替换。

**未验证项（承自素材）**：`scripts/search_trot.py` 已不在仓内，`artifacts/trot_stairs.json` 的搜索过程不可重建；全仓没有 `mj_fullM` 调用 ⇒ `kp`/`kd` 是事后核算，不是自动整定流程。

## 1. MJCF 组装：从 vendored 模型到可仿真场景
### 1.1 素材与路径
```
models/unitree_go2/go2.xml        11,678 B  机器人主 MJCF（只读，永不修改）
models/unitree_go2/scene.xml         816 B  <include file="go2.xml"/> + 地面 —— 本仓刻意不用
models/unitree_go2/assets/*.obj   16 个，合计 28,409,145 B ≈ 27.1 MiB（base/calf/calf_mirror/foot/hip/thigh/thigh_mirror）
models/unitree_go2/go2.png / LICENSE
```
```python
# robotdog/tools/paths.py（9 个常量、0 个函数、38 行；import 无副作用、不建目录）
WORKSPACE: Path = Path(__file__).resolve().parents[2]   # <ws>/robotdog/tools/paths.py 上溯 3 级
MODELS_DIR = WORKSPACE / "models";  GO2_DIR = MODELS_DIR / "unitree_go2"
GO2_XML = GO2_DIR / "go2.xml";      GO2_ASSETS_DIR = GO2_DIR / "assets"
RUNS_DIR = WORKSPACE / "runs"
OUTPUT_DIR = WORKSPACE / "output";  OUTPUT_GIFS_DIR = OUTPUT_DIR / "gifs"   # 唯一交付物目录
GENERATED_DIR = WORKSPACE / ".generated"   # 动态生成的 MJCF（点号前缀=调试产物）
```
vendored `go2.xml` 关键行：L1 `<mujoco model="go2">`；L2 `<compiler angle="radian" meshdir="assets" autolimits="true"/>`；L4 `<option cone="elliptic" impratio="100"/>`；L39–61 `<asset>`；L63–198 `<worldbody>`；L64 `<body name="base" pos="0 0 0.445" childclass="go2">`；L76 `<site name="imu" pos="-0.02557 0 0.04232"/>`；L103/133/163/193 `<geom name="FL|FR|RL|RR" class="foot"/>`；L200+ `<actuator>`；L215–218 `<keyframe><key name="home" qpos="0 0 0.27 1 0 0 0 0 0.9 -1.8 …"/>`。无 `<visual>`、无 `<size>`、无 `<sensor>`。

### 1.2 函数签名（robotdog/go2_model.py，460 行）
```python
@dataclass
class SceneOptions:   # 编译场景时叠加到 Go2 模型上的覆盖项
    timestep: float = 0.002          # DEFAULT_TIMESTEP
    integrator: str = "implicitfast"
    gravity: tuple[float,float,float] = (0.0, 0.0, -9.81)
    iterations: int = 50; ls_iterations: int = 20; solver: str = "Newton"
    add_ground_texture: bool = True; extra_visual: str = ""
    nconmax: int | None = None; njmax: int | None = None

def _go2_inner_xml() -> str                                   # 剥外层 <mujoco>
@functools.lru_cache(maxsize=1)
def go2_assets() -> dict[str, bytes]                          # 裸 basename -> 内容
def _option_xml(opts: SceneOptions) -> str
def build_scene_xml(terrain_xml: str = "", *, opts: SceneOptions | None = None,
                    floor_geom: str | None = None, include_lights: bool = True) -> str
def default_floor_geom(z: float = 0.0, size: float = 60.0) -> str
def compile_scene(terrain_xml: str = "", *, opts: SceneOptions | None = None,
                  floor_geom: str | None = None,
                  write_to: Path | None = None) -> tuple[mujoco.MjModel, Go2Meta, str]
def _resolve_id(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int
def extract_meta(model: mujoco.MjModel) -> Go2Meta
def home_qpos(meta: Go2Meta, *, x=0.0, y=0.0, z: float | None = None, yaw=0.0) -> np.ndarray
def quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray   # (w,x,y,z)->3x3
def quat_to_rpy(quat_wxyz: np.ndarray) -> np.ndarray   # ZYX 绕世界轴 (roll,pitch,yaw)，pitch 有 arcsin clip
```
### 1.3 组装 8 步（字符串手术，不用 `<include>`）
1. `GO2_XML.read_text(encoding="utf-8")`，`_OUTER = re.compile(r"\s*<mujoco\b[^>]*>(?P<body>.*)</mujoco>\s*$", re.DOTALL)` 取 `group("body")`，只保留内部文本；**`meshdir` 保持相对 `"assets"`**。
2. 防御分支：内部无 `meshdir` 时 `_COMPILER.subn(lambda m: m.group(0)[:-2] + ' meshdir="assets"/>', inner, count=1)` 插到 `<compiler .../>` 的 `/>` 前；连 `<compiler>` 都没有 → `ValueError(... missing <compiler/>)`。理由：无 `meshdir` 时 MuJoCo 去 **CWD** 找网格（不可复现失败）。
3. `_OPTION.sub(_option_xml(opts), inner, count=1)` **整条替换** `<option .../>` 为 `<option cone="elliptic" impratio="100" timestep=… integrator=… gravity=… iterations=… ls_iterations=… solver=…/>`；`cone`/`impratio` 硬编码保住原值。`nconmax`/`njmax` 只要一个非 None → 追加 `<size nconmax="…" njmax="…"/>`；都为 None 则不产出（让旧 checkpoint 位对位复现）。
4. `add_ground_texture=True` → `inner.replace("</asset>", _GROUND_ASSETS + "  </asset>", 1)`（checker 地面 + 天空盒 + `stairmat`/`edgemat`）。
5. 一次 `inner.replace("</worldbody>", … , 1)` 按序追加：`floor_geom`? → 两盏 `castshadow="false"` 灯（`include_lights`）→ `terrain_xml`（**最后**，每回合变的几何）。
6. `'<mujoco model="go2_stairs">\n' + inner + visual + "\n</mujoco>\n"`；`visual = _GROUND_VISUAL`（`<visual>` headlight / rgba haze / global azimuth=-130 elevation=-20 1280x720），`extra_visual` 插到 `</visual>` 前。
7. `mujoco.MjModel.from_xml_string(xml, assets=go2_assets())`；`write_to` 非 None 时**先**落盘再编译。
8. 失败处理：`write_to is None` → `GENERATED_DIR.mkdir(parents=True, exist_ok=True)`，dump 到 `GENERATED_DIR / "failed_scene.xml"`（**固定名，连续失败互相覆盖**），抛 `RuntimeError(f"MJCF compile failed; dumped to {dump}") from exc`；`write_to` 非 None → 原样重抛。最后 `extract_meta(model)`，返回 `(model, meta, xml)`。
### 1.4 两条硬约束（换任何机器人/机器都适用）
* **不用 `<include>`**：`from_xml_string()` 没有「主文件目录」概念，而 `<compiler meshdir="assets"/>` 是相对路径 —— include 展开后解析基准直接丢失。故改为「剥壳 + 注入 + 重包」得到一个**自包含字符串**。
* **非 ASCII 工作区路径 → 必须走内存 assets 映射**：MuJoCo 资源读取器按 **Windows ANSI 代码页**解析窄字符文件名，`meshdir` 指向中文路径必失败：`Error opening file 'D:/<mojibake>/models/unitree_go2/assets/base_0.obj'`。`mujoco.MjVfs` 只向 Python 暴露 `close()`，自定义 VFS 回调不可用（已否决）。故 `meshdir` 保留相对值、内容由 `assets={path.name: path.read_bytes()}` 提供，**根本没有任何目录查找**。**键必须是裸 basename**；加 `assets/<name>` 别名会触发 `Repeated file name in assets dict`。`lru_cache(maxsize=1)`：27 MB 每个 worker 进程只付一次。
### 1.5 最小骨架
```python
from robotdog.go2_model import SceneOptions, compile_scene, default_floor_geom, home_qpos
from robotdog.tools.terrain import StairSpec, build_terrain_xml
spec = StairSpec(step_height=0.05, tread=0.50, n_up=8, n_down=0)   # top_landing 默认 1.20
model, meta, xml = compile_scene(
    build_terrain_xml(spec),                                       # 平地返回 ""
    opts=SceneOptions(timestep=0.002, iterations=50, solver="Newton", nconmax=200, njmax=400),
    floor_geom=default_floor_geom(), write_to=Path(".generated") / "demo_scene.xml")
data = mujoco.MjData(model); data.qpos[:] = home_qpos(meta, x=1.00, z=0.32, yaw=0.10)
mujoco.mj_forward(model, data)
```

## 2. 从编译后的模型提取元信息
`Go2Meta` 24 字段：`joint_names / actuator_names / foot_geom_names / joint_ids / actuator_ids / foot_geom_ids / foot_radius / qpos_joint_adr / qvel_joint_adr / dof_joint_adr / base_body_id / imu_site_id / joint_ranges(12,2) / ctrl_ranges(12,2) / home_qpos(nq,) / home_joint_angles(12,) / total_mass / floor_geom_id / stair_geom_ids / trunk_geom_ids / leg_geom_ids / nq / nv / nu`（后 6 个有默认值）。
关节名 / 顺序（= 动作向量顺序，别无二致）：
```python
LEGS = ("FL", "FR", "RL", "RR"); JOINT_SUFFIXES = ("hip", "thigh", "calf")
ACTUATOR_NAMES = tuple(f"{leg}_{s}" for leg in LEGS for s in JOINT_SUFFIXES)
JOINT_NAMES    = tuple(f"{n}_joint" for n in ACTUATOR_NAMES)   # MJCF 里 joint 与 actuator 不同名
FOOT_GEOM_NAMES = LEGS                                          # 足球 geom 就叫 FL/FR/RL/RR
```
顺序 = `FL_hip, FL_thigh, FL_calf, FR_hip, …, RR_hip, RR_thigh, RR_calf`（= `go2.xml` 声明顺序）。实测（本次重新编译确认，与 README §3.1 表格一致）：
```
nq/nv/nu = 19/18/12     total_mass = 15.206408 (= model.body_mass[1:].sum()，跳过 world body 0)
foot_radius = 0.022     (= model.geom_size[foot_geom_ids[0], 0]；脚是球，geom_xpos 是球心)
base_body_id = 1   imu_site_id = 0   floor_geom_id = 0   stair_geom_ids = []
trunk_geom_ids = [6 7 8]     leg_geom_ids = [11 14 17 18 23 26 29 30 35 38 41 42 47 50 53 54]
ctrl_ranges: hip ±23.7 N·m   thigh ±23.7 N·m   calf ±45.43 N·m   (12 维，逐腿重复)
joint_ranges: hip [-1.0472, 1.0472]   calf [-2.7227, -0.83776]   (四条腿同)
              thigh: FL/FR [-1.5708, 3.4907]；RL/RR [-0.5236, 4.5379]
home: BASE_HEIGHT_HOME = 0.27 m；HOME_JOINT_ANGLES = (0.0, 0.9, -1.8) x 4 条腿
      home_qpos = zeros(nq); [2]=0.27; [3]=1.0(四元数 w); [qpos_joint_adr]=home_joint_angles
```
* **home 是硬编码常量，不读 `<keyframe>`**：`extract_meta` 用常量**构造** `home_qpos`，模型换版本必须手工同步这两个常量，否则 home 姿态静默漂移。
* `qpos_joint_adr = jnt_qposadr[joint_ids]`（自由关节恒占最前 7 个 qpos / 6 个 qvel）；`dof_joint_adr = qvel_joint_adr = jnt_dofadr[joint_ids]`。
* `floor_geom_id` 用 `mj_name2id` 且**不检查 < 0**（`floor_geom=""` 时为 -1，被 env 当合法 id 收下）；`stair_geom_ids` 用 `name(g).startswith("stair_")`，改 `StairSpec.name_prefix` 会让它变空。
* `extract_meta` 只查一次名字 → 之后全是纯数组索引（`mj_name2id` 是字符串查找，50 Hz × 8 环境不可接受）。
**几何 group 约定（把机器人拆成躯干/腿的唯一依据）**：`group="2"` = 纯视觉网格（`contype=0 conaffinity=0`），永不碰撞；`group="3"` = **全部碰撞图元**（躯干 box/cylinder/sphere、hip/thigh/calf 形状、四个足球）⇒ `geom_group == 3` 精确等价于「能和地形接触」，`COLLISION_GEOM_GROUP = 3`。拆分：`terrain_ids = {floor} | set(stair_*)`、`foot_set = set(foot_geom_ids)`；对 `geom_group==3` 且不在上述集合者，`geom_bodyid[g] == base_body_id` → `trunk_geom_ids`（接触=**真摔**），否则 → `leg_geom_ids`（接触=小腿蹭立面，**扣分不判死**）。否决「按 geom 名字硬编码躯干」：名字是 vendored XML 的私有约定，换版本就烂。**坑**：vendored XML 若把碰撞图元改到别的 group，两集合同时为空，`env._contact_flags()` 对空集显式返回 `(False, False)` —— 终止判据**静默失效**，不报错。
**取关节空间惯量对角元（mujoco 3.12 签名）**：
```python
data = mujoco.MjData(model); data.qpos[:] = home_qpos(meta); mujoco.mj_forward(model, data)
M = np.zeros((model.nv, model.nv))          # dst 必须 (nv,nv) float64、C-contiguous
mujoco.mj_fullM(model, data, M)             # 3.12: mj_fullM(m: MjModel, d: MjData, dst: NDArray[float64,[m,n]]) -> None
J = np.diag(M)[meta.dof_joint_adr]          # 关节空间惯量对角元（= qvel 地址）
```
实测 `np.diag(M)`（nv=18）：`[15.206408, 15.206408, 15.206408, 0.17725848, 0.49077632, 0.53544362, 0.03357173, 0.03039024, 0.0163263, …（后 12 项按腿重复）]` → 前 3 个是 base 平动（= 总质量）、接着 3 个是 base 转动；`diag[meta.dof_joint_adr]` = hip `0.03357173` / thigh `0.03039024` / calf `0.01632630` kg·m²（四条腿完全相同）。

## 3. PD 增益选型：按惯量配带宽与阻尼比（可迁移公式）
### 3.1 本仓实现
```python
# robotdog/env.py:178-179（EnvConfig）
kp: tuple[float, float, float] = (30.0, 30.0, 45.0)   # 每条腿 hip/thigh/calf
kd: tuple[float, float, float] = (0.6, 0.6, 0.9)
# __init__：self._kp = np.tile(self.cfg.kp, 4); self._kd = np.tile(self.cfg.kd, 4)   → 12 维
# robotdog/env.py:922-929（每个控制步内 decimation=10 次）
for _ in range(self.cfg.decimation):
    q  = self.data.qpos[self.meta.qpos_joint_adr]
    qd = self.data.qvel[self.meta.qvel_joint_adr]
    tau = self._kp * (target - q) - self._kd * qd
    self.data.ctrl[:] = np.clip(tau, self.meta.ctrl_ranges[:, 0], self.meta.ctrl_ranges[:, 1])
    mujoco.mj_step(self.model, self.data)
```
目标角 `target = clip(home + gait_gain*gait.radians(t) + cfg.action_scale*action, meta.joint_ranges[:,0], meta.joint_ranges[:,1])`（`env.py:893-911`）—— clip 的是**关节限位**，不是 ±1。位置伺服由 `ctrlrange` 饱和兜底：增益再大也只饱和，不会输出超规格力矩。
### 3.2 公式化步骤（换任何机器人照做）
1. 编译模型，`mj_forward(model, data)` 到 **home 姿态**（必须用实际标称站姿，惯量随构型变）。
2. `M = zeros((nv,nv)); mujoco.mj_fullM(model, data, M); J = diag(M)[dof_joint_adr]` —— 每个关节一个 `J`。
3. 选目标带宽 `f0`（Hz）→ `w0 = 2*pi*f0`。三条硬约束：`f0 ≫` 步态频率（本仓 2–5.5 Hz，取 4.8/5.0/8.4 Hz）；`f0 ≪` 控制环频率（本仓 `timestep=0.002 × decimation=10 → control_dt=0.02 s = 50 Hz`）；显式积分稳定 `w0 * timestep ≪ 1`（本仓 `52.5 × 0.002 = 0.105`，安全，`integrator="implicitfast"` 再给一层阻尼）。
4. `kp = J * w0**2`。
5. 选阻尼比 `zeta`（0.3–0.5）→ `kd = 2*zeta*sqrt(kp*J) = 2*zeta*J*w0`。**惯量最小的关节给更高的 zeta**（否则触地冲击下振铃）：calf 的 `J` 最小、`w0` 最高，所以 `kd` 从 0.6 提到 0.9。
6. 复核 `1/(zeta*w0)` = 闭环响应时间尺度，应显著小于步态周期、与 `control_dt` 同量级或更慢（本仓 36/101/112 ms vs 183 ms 步态周期、20 ms 控制周期）。
7. 落地后按 `|kp*(target-q)|` 峰值核对是否长期贴 `ctrlrange`；振铃优先加 `kd` 而非 `kp`。
### 3.3 本仓实测核算表（`kp/kd` 是**事后**核算，本仓没做自动整定）
| 关节 | `J` (kg·m²) | `kp` | `kd` | `w0=sqrt(kp/J)` | `f0` | `zeta=kd/(2*sqrt(kp*J))` | `1/(zeta*w0)` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hip ×4 | 0.03357173 | 30.0 | 0.60 | 29.893 rad/s | 4.758 Hz | 0.2989 | 111.9 ms |
| thigh ×4 | 0.03039024 | 30.0 | 0.60 | 31.419 rad/s | 5.001 Hz | 0.3142 | 101.3 ms |
| calf ×4 | 0.01632630 | 45.0 | 0.90 | 52.500 rad/s | 8.356 Hz | 0.5250 | 36.3 ms |
`kd` 比值 1.5 比严格等比缩放 `sqrt(45/30)=1.225` 更保守（刻意留裕量）。

## 4. 关键签名清单（速查）
```
go2_model.py: SceneOptions / _go2_inner_xml / go2_assets / _option_xml / build_scene_xml /
              default_floor_geom / Go2Meta / _resolve_id / extract_meta / compile_scene /
              home_qpos / quat_to_mat / quat_to_rpy
terrain.py:   StairSpec / flat_spec / demo_stairs_spec / build_terrain_xml
gait.py:      TROT_PHASE / CALF_SHAPES / GaitParams / load_gait
paths.py:     9 个 Path 常量（无函数）      console.py: setup_utf8_console() -> None
env.py:       EnvConfig.kp/.kd/.action_scale/.gait_prior/.gait/.gait_gain/.control_dt;
              _gait_action / _target_from_action / step 的 PD+clip 循环
```

## 5. 平台坑（Windows/PowerShell，迁移必带）
* `console.setup_utf8_console()` 必须由脚本在 `main()` 第一行显式调用（`scripts/train.py:294`、`scripts/play.py:601`）；`import robotdog` **不会**触发。它只 `reconfigure(encoding="utf-8", errors="replace")` stdout/stderr、吞 `(ValueError, OSError)`，不改 stdin、不换流对象、不设 `PYTHONIOENCODING`，也不解决控制台代码页（那要 `chcp 65001`）。
* 后台 job 的 cwd 与 `PYTHONPATH`：直接 `python .tmp\x.py` 时 `sys.path[0]` 是 `.tmp`，`import robotdog` 会 `ModuleNotFoundError` —— 需 `$env:PYTHONPATH='D:\项目\RobortDog\upstair_core'`。
* `>` 重定向在 Windows 下可能写出 UTF-16，读日志用 `Out-File -Encoding utf8` 一类写法。
