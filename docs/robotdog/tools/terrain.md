# `robotdog/tools/terrain.py` —— 参数化楼梯地形（解析几何 + MJCF）

## 1. 一句话定位

**工具层**（`robotdog/tools/`）里的地形几何模块：把「一段楼梯」描述成一个 `StairSpec` 数据类，由它派生（1）注入 MJCF 的实心 box 几何、（2）一个**解析高度剖面** `height_at(x)`、（3）与 MuJoCo 自身碰撞一致的向下射线高度查询。它解决的问题是：奖励项、出生点、成功判据、评估指标**必须与几何本身说的是同一件事**，而不是各自去猜地形。

## 2. 职责边界

**做**：

* `StairSpec`：一段楼梯的全部几何参数 + 派生量（crest 高度、各段结束位置）。
* `step_blocks()`：把楼梯展开成实心 box 列表（前→后有序）。
* `to_xml()` / `build_terrain_xml()`：生成注入 `<worldbody>` 的 MJCF `<geom>` 片段。
* `height_at(x)`：**解析**地形高度剖面（阶跃函数）。
* `TerrainRandomization` + `sample_spec()`：逐 episode 的地形随机化采样。
* `flat_spec()` / `demo_stairs_spec()` / `demo_pyramid_spec()`：课程各阶段的便捷构造器。
* `raycast_heights()` / `mujoco_mj_ray()`：用 MuJoCo 自己的碰撞做向下射线高度查询（校验/运动学指标用）。

**不做**：

* **不解析 XML、不注入 XML、不编译模型**。本模块只**返回字符串**；把它拼进场景并 `mujoco.MjModel` 编译的是 `robotdog/go2_model.py`（`compile_scene`）。
* **不 import mujoco（模块顶层）**。只有 `mujoco_mj_ray()` 在函数体内延迟 `import mujoco`。
* **不建 heightfield / hfield**，不生成网格，不做程序化噪声地形。
* **不知道观测、奖励、终止判据**。它只提供 `height_at`，谁用、怎么用由 `robotdog/env.py` 决定。
* **不做碰撞过滤**，不改 `condim`/`solref`/`solimp` 之外的选项（`condim="3"` 是写死的）。
* **不 import `robotdog.env`**，不知道 PPO / 课程 / 训练循环（课程**配置**由 `robotdog/curriculum.py` 用本模块的构造器拼出来，方向是下游依赖本模块）。

## 3. 关键设计决策与理由

### 3.1 每一级都是**落地实心块**，不是薄板

源码 `Design notes` 第一条：`Every step is a **solid block resting on the ground plane** (not a thin slab): half-size (tread/2, width/2, top_z/2) centred at z = top_z/2. Solid blocks cannot be caught on from underneath and keep contact geometry simple.`

理由：薄板的下表面与地面之间会留下一个空隙，脚/小腿可以钻进去，被**从下面**挂住；实心块的侧面从 z=0 一直到顶面，不存在这个空腔。代价是块与块之间大量**静-静重叠** —— 源码明确指出这无害：`Static-static overlap between neighbouring blocks is harmless in MuJoCo -- contacts are only generated for pairs where at least one body is dynamic.`

**被否决的方案**：薄板（省内存、视觉一样）—— 否决，引入「从下方被挂住」的新失败模式，而本任务的核心难点正是脚与立面的接触。**另一个被否决的方案**：用 `heightfield`（天然无缝、单 geom）—— 否决，理由见 §3.2。

### 3.2 保留**锐利 box 边**，不用 heightfield

源码：`Crisp box edges are deliberate: the robot must learn to clear a sharp riser, whereas a heightfield would round the edge over one cell.`

理由：本项目的任务定义就是「抬脚越过**立面**」。heightfield 把台阶边缘在一个网格单元内插值成斜坡，等于**替策略把难点抹平**（多出一个可以借力的斜面），训练出来的策略在真实台阶几何上不成立。box 的边是数学意义上的锐边，`mj_ray` 打在上面得到精确的高度跃变，`height_at` 的阶跃函数才与几何严格一致（见 §3.4）。

**被否决的方案**：heightfield —— 否决，除上述「抹平难点」外，还破坏 `height_at` 与几何的一致性（阶梯函数 vs 插值斜坡），而 §3.4 说明这个一致性是奖励正确性的前提。

### 3.3 【载荷决策 a】`CREST_SEAM_OVERLAP = 1.0`

```python
#: How far (in treads) the flat crest / top landing reaches *back* into the last
#: ascending step.  It must be > 0, and the reason is a genuine MuJoCo trap:
#: ...
CREST_SEAM_OVERLAP = 1.0  # in treads
```

**必须 > 0**，因为顶层平台（或金字塔的 crest）与最后一级踏面**等高**，两块 box 共面。当它们只是**相邻**（crest 正好从 `up_end_x` 开始）时，接触面是一条零高度的缝：

1. 中心落在这条缝上的脚球，对两个竖直面都是**零穿透**，但对两个顶面都还有几十毫米距离；
2. MuJoCo 对「球在盒内」的接触沿**最小穿透轴**求解 —— 于是两个 box 都在**水平方向**推这只脚，方向相反；
3. 两个推力相互抵消，z 方向没有任何东西支撑脚，脚被夹住（pin）：x 被钉死、下沉到踏面下 30~40 mm，且因为接触法向是水平的，`touch.riser_graze` 每一步都被点亮。

实测（8 cm × 8 级，`.tmp/stall_probe.py`）：两只前脚在剩余 **700** 个控制步里冻结在 `x = 5.500 == up_end_x`，即机器人爬完整段楼梯后**停在终点线前 0.114 m** 直到 20 s 上限 —— 这正是高台阶训练长期卡在 20~40 % 的原因（实验记录见 `experience.md` §1.1）。

修法**不是**把 crest 抬高或降低（那会改变几何与解析剖面），而是让 crest **向最后一级踏面回伸一个踏面**：

```python
ov = CREST_SEAM_OVERLAP * self.tread
cx = self.up_end_x + (self.top_landing - ov) / 2.0
size = ((self.top_landing + ov) / 2.0, hy, top / 2.0)
```

于是 crest 的后侧面被**埋进**最后一级台阶内部，共面的那对表面消失，只剩真正的立面。源码强调：`The solid *union* is unchanged, so the analytic profile, the downward raycast and every rendered pixel are identical.` —— 因为两个实心块的并集没变，所以 **`height_at`、`raycast_heights` 与渲染结果全都不需要跟着改**，这是这个修法最关键的优点。

取 `1.0`（一个完整踏面）而不是「一小段 overlap」：整块踏面回伸让重叠体积足够大，避免留下**部分共面**的窄缝（窄缝同样会夹脚）；而且表达式保持整数倍踏面，`cx`/`size` 与 `step_blocks()` 的其它块形式统一，可读性最好。

**被否决的方案**：

* 让 crest 从 `up_end_x - tread` 起算（等价于手工改 `up_end_x`）—— 否决：`up_end_x` 是**成功判据**与解析剖面共用的量（`env.py:424, 431`），改它会同时移动终点线，把一个接触问题变成任务定义问题。
* 给 crest 加一点 z 偏移（例如 `+1e-4`）以消除共面 —— 否决：制造一个肉眼不可见但物理上真实的台阶，脚会被这个伪台阶绊住；而且解析剖面与射线高度会对不上。
* 用 `margin` / `gap` 之类的接触参数（`solref`、`margin`）绕开 —— 否决：那是全局接触参数，会改变**所有**接触（含脚-地面），副作用远超问题本身。
* 干脆不要顶层平台（`top_landing = 0`）—— 否决：那样楼梯顶部就是 0.40 m 悬崖，成功判据永远无法满足（见 §3.5 对 `top_landing` 的解释）。

### 3.4 【载荷决策 b】`TERRAIN_GEOM_GROUP = 1`

```python
#: MuJoCo geom group reserved for terrain (stairs **and** the ground floor).
#: ...
TERRAIN_GEOM_GROUP = 1
```

这个值**双重**载荷：

1. **给测距用的掩码**。`robotdog/tools/touch.py` 的前向测距必须能「向前看」而不打到自己的前腿 —— 而前腿正好扫过被探测的体积。`mj_ray` 接受 **geom group mask**，把所有地形几何（台阶 **和** 地面）放进一个专用组，再用「只开这一组」的掩码投射，就**构造性地**让探针看不见机器人。因为 Unitree 的 MJCF 已经把 `group="2"` 给了纯视觉 geom、`group="3"` 给每一个碰撞 geom（躯干盒、膝圆柱、四个脚球），2 和 3 已被占用；没有显式 group 的 geom 落在 0。
2. **渲染可见性**。MuJoCo 的 `mjv_defaultOption` 出厂是 `geomgroup = [1, 1, 1, 0, 0, 0]` —— **3、4、5 组默认被隐藏**。如果给楼梯打上 group 4，它在**每一个离屏渲染里都不可见，而物理完全不受影响** —— 于是出现「狗在爬空气，HUD 却愉快地报告高度增益」这一幕（真的发生过，见 `experience.md` §1.2）。group 1 落在默认可见范围内，所以楼梯不需要改渲染器就能显示。

`robotdog/env.py` 的 `render()` 另外**强制打开全部六个组**，所以改这个常量不会再悄悄把地形藏起来（源码注释：`so changing this constant cannot silently hide the terrain again`）。

**被否决的方案**：

* group 0（默认）—— 否决：0 是「没写 group 的 geom」的桶，地面 plane 与其它随手加的 geom 会混在一起，测距掩码就无法「只开地形」。
* group 4/5（视觉上更「独立」）—— 否决：正是上面那个默认隐藏的坑。
* 不用 geom group，改用 `bodyexclude` 排除机器人 —— 否决：`mj_ray` 的 `bodyexclude` 只能排除**一个** body，而机器人的四个脚球分属四个 body，挡不住全部。
* 不用 geom group，改用几何上的短射线（保证打不到腿）—— 否决：探针跨度是设计参数（`probe_span = 0.60`），限制射线长度会让「台阶还有多远」这个量直接失去量程。

**与 `raycast_heights` 的不对称（注意）**：`terrain.py` 自己的 `mujoco_mj_ray()` 传的是 `geomgroup=None`（`mj_ray(..., None, 1, -1, geomid)`），即**不过滤组**。这是刻意的不对称：这个函数的用途是「校验几何是否与解析剖面一致」，它要打到**任何**东西（包括万一机器人正好站在采样点上），从而暴露不一致；而 `touch.cast_ray()` 默认传 `TERRAIN_GEOM_MASK`（实测 `[0, 1, 0, 0, 0, 0]`），因为它的任务是「我只关心地形」。

### 3.5 `height_at(x)` 是 x 的**阶跃函数**，而且它必须与几何严格一致

源码 docstring：`The analytic height profile is exposed through :func:`height_at` so that reward terms, spawn placement and evaluation metrics all agree with the geometry.`

实现是纯解析、纯分段：

```python
if x < self.start_x:
    return 0.0
if x < self.up_end_x:
    i = int((x - self.start_x) // self.tread)
    return (i + 1) * self.step_height
if self.n_down == 0:
    return self.crest_height if x < self.up_end_x + self.top_landing else 0.0
if x < self.up_end_x + self.landing:
    return self.crest_height
j = int((x - self.up_end_x - self.landing) // self.tread)
if j >= self.n_down:
    return 0.0
return max(0.0, self.crest_height - (j + 1) * self.step_height)
```

**它是阶跃的**（在 `start_x + i*tread` 处跳变，在 `up_end_x + top_landing` 处从 crest 直接掉到 0），这有两层后果，都必须知道：

1. **奖励侧的 `foot_clearance` 建立在它之上**。`robotdog/env.py:529` 的 `foot_clearance` 属性、`:543` 的 `np.array([self.stair.height_at(float(xi)) for xi in arr])` —— 也就是「每只脚相对它**脚下那个 x** 的地形高」这个量，参照面就是 `height_at`。脚跨过立面的一瞬间参照面跳一整级，抬脚高度（clearance）会**逐帧翻转**。同类问题在 `env.py:932, 944, 959`（支撑面参照 / 终止判据的高度参照）也会出现 —— 这正是 `experience.md` §1.4、§3.3 记录的「幽灵摔倒」与「`base_height` 参照面也是阶跃的」两个坑。本模块**只是把事实提供出去**，改参照口径是环境层的事（`terminate_height_ref="support"`）。
2. **它和几何是同一份真值**。`step_blocks()` 与 `height_at()` 必须永远同步 —— 这也是 §3.3 那个「并集不变」的修法为何重要的原因：改几何而不改剖面（或反之）会让奖励说的高度与实际地形不同。

**被否决的方案**：从 MuJoCo 射线实时查高度（`raycast_heights`）作为奖励参照 —— 否决：(a) 每次调用要做 4 次射线，而 `foot_clearance` 在**每一步**都要算；(b) 射线会被机器人自己的腿挡住（`geomgroup=None`），得到「脚踩在自己小腿上」的高度；(c) 解析式是**精确**的，射线是近似的（浮点 + 面元）。

### 3.6 随机化是**重建 spec**，不是打补丁

`TerrainRandomization` 的 docstring：`Sampling happens at ``reset()``; the spec is rebuilt rather than patched so that the analytic profile stays exact.`

`sample_spec()` 用 `dataclasses.replace()` 从 `base` 派生一个**新** spec：

```python
base = base or StairSpec()
n_up = int(rng.integers(rand.n_up_range[0], rand.n_up_range[1] + 1))
return replace(base, step_height=..., tread=..., width=..., n_up=n_up, n_down=n_up,
               start_x=rand.start_x, landing=rand.landing)
```

理由：`StairSpec` 的所有派生量都是 `@property`（**每次访问重算**），所以只要换掉输入字段，`crest_height` / `up_end_x` / `height_at` 自动一致 —— 不存在「改了 `step_height` 忘了同步 `crest_height` 缓存」的窗口。用 `replace()` 而不是手写构造，保证 `base` 上的其它字段（`friction`、`rgba`、`top_landing`、`center_y`、`name_prefix`）被完整继承。

**采样细节**：`n_up` 用 `rng.integers(lo, hi + 1)`（右开区间 → 含 `hi`），而 `height_range` / `tread_range` / `width_range` 用 `rng.uniform(*range)`（含两端点）。`n_down = n_up` ⇒ **随机地形永远是金字塔**（上楼 + 下楼），不做「只上不下」。

**被否决的方案**：运行时改 `spec.n_up` 等字段（就地打补丁）—— 否决：`StairSpec` 会被上传给 `curriculum.py` / 报告 JSON（`env.py:1198` 把 `crest_height` 打进 info），就地改会让「报告里那份 spec」与实际几何不一致；而且同一个 `StairSpec` 实例可能被多个环境共享。**另一个被否决的方案**：用 `random` 模块 —— 否决，`numpy.random.Generator` 可指定 seed、可传进各子环境，且本仓的复现实验依赖 `np.random.Generator` 的流。

### 3.7 便捷构造器 `flat_spec` / `demo_stairs_spec` / `demo_pyramid_spec` 的分工

* `flat_spec(**kw)`：`n_up`、`n_down` 用 `setdefault` 设成 `0`，其余透传。**平地只用它**。
* `demo_stairs_spec(...)`：显式参数 + `kw.setdefault("top_landing", 1.20 if n_down == 0 else 0.0)`。docstring：`Explicit parameters so callers can override every field.`
* `demo_pyramid_spec(...)`：`n_up` 与 `n_down` 都默认 `8`，直接透传给 `StairSpec`。

`demo_stairs_spec` 里那句 `top_landing` 的 `setdefault` 是**任务定义**层面的决策：只在「只上不下」时给顶层平台；金字塔模式下 crest 后面直接接下降段，`top_landing` 惰性（`step_blocks()` 与 `height_at()` 在 `n_down > 0` 时都不读它）。`setdefault` 而非直接赋值，是为了让调用方仍能显式覆盖。

### 3.8 【必须知道的陷阱】`demo_stairs_spec(step_height=0.0, n_up=8)` 无法编译

`step_blocks()` 里上升段是 `top = (i + 1) * self.step_height`，`size_z = top / 2.0`。当 `step_height = 0.0` 时 **每一级都是零高度 box**：`size="0.250000 1.000000 0.000000"`。零尺寸是非法几何体，MuJoCo 编译直接失败：

```
ValueError: Error: size 2 must be positive in geom
Element name 'stair_up0', id 0, line 2
```

经由 `robotdog/go2_model.py:428` 的包装后，调用方看到的是：

```
RuntimeError: MJCF compile failed; dumped to <...>\.generated\failed_scene.xml
```

（失败场景会被转储到 `GENERATED_DIR`，即 `.generated/failed_scene.xml`，方便对着实际 XML 排查。）

**正确做法**：平地一律用 `flat_spec()` / `flat_spec(n_up=0, n_down=0, ...)` —— 此时 `n_up == n_down == 0`，`build_terrain_xml()` 直接返回 `""`，**根本不生成任何地形 geom**，所以 `step_height` 取什么值都无所谓（`robotdog/curriculum.py:143` 正是 `flat_spec(n_up=0, n_down=0, step_height=0.0, ...)`）。见 `experience.md` §1.3。

`step_blocks()` 里下降段有一道自己的零高度守门 `if top <= 1e-9: continue`，保证「级数多于 `crest_height / step_height`」时不会生成零高度块；但**上升段没有这道守门**（`step_height = 0` 会生成零高度块），所以这个陷阱只对「`step_height = 0` 且 `n_up > 0`」成立。

**被否决的方案**：在 `step_blocks()` 里也跳过零高度上升块 —— 否决：那会让 `n_up = 8, step_height = 0` 静默变成平地，把一个**调用错误**伪装成合法地形（`height_at` 仍会算出 8 级台阶的位置，与几何不一致）。让编译失败是更安全的行为。

### 3.9 `top_landing` 默认 `1.20`（而不是 `0.0`）是**任务可成功性**的前提

字段上方的注释写得很明确：`Must be > 0 for the terrain to be succeedable at all: StairSpec.height_at returns 0 past up_end_x + top_landing, and the success test requires the base to be past up_end_x *and* sitting at crest height. With the old default of 0.0 a bare StairSpec() was a 0.40 m cliff at the crest, so Go2StairsEnv() could never report a success.`

即：成功判据是「base 越过 `up_end_x` **且**脚下地形高 ≈ `crest_height`」（`robotdog/env.py:429-432`）。而 `height_at` 在 `up_end_x + top_landing` 之后返回 0。若 `top_landing = 0.0`，越过 `up_end_x` 的瞬间地形高度就是 0 —— 那里是一个 0.40 m 悬崖，`Go2StairsEnv()` 永远报不出成功。

**被否决的方案**：保持 `0.0` 并在成功判据里放宽「高度必须等于 crest」—— 否决：那会在狗**掉下悬崖**的那一刻宣布成功（它确实越过了 `up_end_x`），把成功判据变成失败判据。

### 3.10 `to_xml()` 的摩擦串用 `:g` 格式化、坐标用 `:.6f`

```python
fr = " ".join(f"{v:g}" for v in self.friction)
...
f'pos="{cx:.6f} {cy:.6f} {cz:.6f}" '
f'size="{hx:.6f} {hy:.6f} {hz:.6f}" '
f'condim="3" friction="{fr}" rgba="{self.rgba}" '
f'group="{TERRAIN_GEOM_GROUP}"/>'
```

* 摩擦用 `:g`：`(0.9, 0.005, 0.0001)` 写成 `"0.9 0.005 0.0001"` —— 去掉尾随零，避免 `0.000100` 这种噪声，也避免把 `0.0001` 写成 `0.00010000000000000000479` 之类的浮点展开。
* 坐标用 `:.6f`：微米级精度足够表达任何台阶尺寸，同时输出**定长**（便于 diff 生成的 XML，`.generated/failed_scene.xml` 就是给人读的）。
* `rgba` 与 `friction` 是**字符串/元组透传**（`rgba` 本身就是 MJCF 语法的字符串），因此调用方可以直接写 `"1 0 0 1"`。
* `indent: str = "    "`（4 空格）暴露为参数，因为它要被插进不同缩进层级的 `worldbody`。

## 4. 公开 API

### 模块级常量

```python
DEFAULT_STAIR_FRICTION: tuple[float, float, float] = (0.9, 0.005, 0.0001)
DEFAULT_STAIR_RGBA = "0.62 0.60 0.55 1"
DEFAULT_EDGE_RGBA = "0.95 0.72 0.15 1"
TERRAIN_GEOM_GROUP = 1
CREST_SEAM_OVERLAP = 1.0  # in treads
```

### `@dataclass class StairSpec`

字段（名称 / 类型 / 默认值 / 语义，逐项对应源码）：

| 字段 | 类型 | 默认值 | 语义（源码注释原文要点） |
| --- | --- | --- | --- |
| `step_height` | `float` | `0.05` | `riser height [m]  (台阶高度)` |
| `tread` | `float` | `0.50` | `tread depth along +x [m]  (台阶宽度)` |
| `width` | `float` | `2.00` | `lateral extent along y [m]` |
| `n_up` | `int` | `8` | `number of ascending risers` |
| `n_down` | `int` | `0` | `ascending only when 0; >0 makes a pyramid` |
| `start_x` | `float` | `1.50` | `world x where the first riser begins` |
| `landing` | `float` | `1.20` | `flat crest length between the up and down flights [m]` |
| `top_landing` | `float` | `1.20` | `Flat top platform when n_down == 0 [m].` 必须 > 0（见 §3.9） |
| `center_y` | `float` | `0.0` | 楼梯在 y 方向的中心（机器人出生点与测距都用它） |
| `friction` | `tuple[float, float, float]` | `DEFAULT_STAIR_FRICTION` = `(0.9, 0.005, 0.0001)` | 直接写进 geom 的 `friction=` |
| `rgba` | `str` | `DEFAULT_STAIR_RGBA` = `"0.62 0.60 0.55 1"` | MJCF 颜色串 |
| `name_prefix` | `str` | `"stair"` | `step_blocks()` 生成的名字前缀（如 `stair_up0`） |

派生属性（全部只读、每次访问重算）：

| 属性 | 返回类型 | 公式 |
| --- | --- | --- |
| `crest_height` | `float` | `n_up * step_height` |
| `up_end_x` | `float` | `start_x + n_up * tread`（`x where the last riser's tread ends (= start of the crest)`） |
| `down_end_x` | `float` | `up_end_x + landing + n_down * tread`（`x where the descending flight returns to ground level`） |
| `span_end_x` | `float` | `down_end_x`（若 `n_down > 0`）；`up_end_x`（若 `n_up == 0`）；否则 `up_end_x + top_landing`（`x where solid staircase geometry ends`） |
| `max_height` | `float` | `crest_height` |

方法：

```python
def step_blocks(self) -> list[tuple[str, tuple[float, float, float],
                                   tuple[float, float, float]]]:
    """Return ``[(name, (cx, cy, cz), (hx, hy, hz)), ...]`` for every block."""


def to_xml(self, indent: str = "    ") -> str:
    """MJCF ``<geom>`` elements for the whole staircase."""


def height_at(self, x: float) -> float:
    """Analytic terrain height [m] at world x (0 outside the staircase)."""
```

* `step_blocks()`：返回 `(名字, 中心坐标三元组, 半尺寸三元组)` 列表，**前→后有序**（源码：`The full staircase is emitted front-to-back so that the raycast and the analytic profile in height_at line up exactly.`）。`hx, hy = tread / 2.0, width / 2.0`；上升块 `i` 的名字是 `f"{name_prefix}_up{i}"`、中心 `(start_x + (i+0.5)*tread, center_y, top/2.0)`、半尺寸 `(tread/2, width/2, top/2.0)`，其中 `top = (i+1)*step_height`。`n_down > 0` 时追加 `"_crest"` 块再追加 `"_down{j}"`；否则在 `top_landing > 0.0 and n_up > 0` 时追加 `"_top"` 块。无异常。**无返回值语义模糊点**：`name` 由 `name_prefix` 决定，重复调用返回相同内容（纯函数）。
* `to_xml(indent="    ")`：返回**用 `\n` 连接**的多个 `<geom .../>` 行，**不带**首尾换行、**不带** `\n` 前缀。每个 geom 固定带 `type="box"`、`condim="3"`、`friction`、`rgba`、`group`。无异常。`indent` 是每行前缀（默认 4 空格）。
* `height_at(x)`：返回该 x 处的解析地形高（米），楼梯之外返回 `0.0`。**阶跃**，无插值。无异常。注意参数是 `float`；传入 numpy 标量也可以（比较与取整都可用）。

### `@dataclass class TerrainRandomization`

| 字段 | 类型 | 默认值 |
| --- | --- | --- |
| `height_range` | `tuple[float, float]` | `(0.03, 0.08)` |
| `tread_range` | `tuple[float, float]` | `(0.30, 0.60)` |
| `width_range` | `tuple[float, float]` | `(1.2, 3.0)` |
| `n_up_range` | `tuple[int, int]` | `(4, 9)` |
| `start_x` | `float` | `1.20` |
| `landing` | `float` | `1.20` |

docstring：`Per-episode stair randomization (used from the domain-randomization step). Sampling happens at reset(); the spec is rebuilt rather than patched so that the analytic profile stays exact.` **注意它是纯数据类，没有方法**（见 §9 关于 `rand.sample_spec(rng)` 的坑）。

### 函数

```python
def sample_spec(rng: np.random.Generator, rand: TerrainRandomization,
                base: StairSpec | None = None) -> StairSpec:
    """Draw a randomized :class:`StairSpec` (pyramid: up then down)."""


def flat_spec(**kw) -> StairSpec: ...


def demo_stairs_spec(
    step_height: float = 0.05,
    tread: float = 0.50,
    n_up: int = 8,
    n_down: int = 0,
    **kw,
) -> StairSpec:
    """Explicit parameters so callers can override every field."""


def demo_pyramid_spec(
    step_height: float = 0.05,
    tread: float = 0.50,
    n_up: int = 8,
    n_down: int = 8,
    **kw,
) -> StairSpec: ...


def build_terrain_xml(spec: StairSpec) -> str:
    """MJCF fragment to inject into ``<worldbody>``."""


def raycast_heights(model, data, spec: StairSpec, xs: np.ndarray,
                    y: float | None = None, z_high: float = 6.0) -> np.ndarray:
    """Downward-raycast terrain height at each x, using MuJoCo's own collision.
    Returns ``nan`` where no geometry was hit."""


def mujoco_mj_ray(model, data, pnt: np.ndarray, geomid: np.ndarray) -> float:
    import mujoco
    return float(
        mujoco.mj_ray(model, data, pnt, np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
    )
```

| 函数 | 参数语义 | 返回 | 异常 |
| --- | --- | --- | --- |
| `sample_spec(rng, rand, base=None)` | `rng`：`np.random.Generator`；`rand`：`TerrainRandomization`；`base`：基准 spec（`None` ⇒ `StairSpec()`） | 新 `StairSpec`（`n_up == n_down`，即金字塔） | 无显式 raise |
| `flat_spec(**kw)` | `n_up`/`n_down` 默认设 `0`，其余字段透传 | `StairSpec` | 透传 `StairSpec` 的 `TypeError`（未知字段） |
| `demo_stairs_spec(step_height=0.05, tread=0.50, n_up=8, n_down=0, **kw)` | 显式四个参数；`kw.setdefault("top_landing", 1.20 if n_down == 0 else 0.0)` | `StairSpec` | 同上 |
| `demo_pyramid_spec(step_height=0.05, tread=0.50, n_up=8, n_down=8, **kw)` | 四个显式参数，透传 `**kw` | `StairSpec` | 同上 |
| `build_terrain_xml(spec)` | `spec`：`StairSpec` | **`""`**（当 `n_up == 0 and n_down == 0`）；否则 `"\n" + spec.to_xml()`（**带一个前导换行**） | 无 |
| `raycast_heights(model, data, spec, xs, y=None, z_high=6.0)` | `xs`：要采样的世界 x 数组；`y`：`None` ⇒ `spec.center_y`；`z_high`：射线起点高度 | `np.ndarray`，长度 `len(xs)`；没打到返回 `np.nan` | 无（`mujoco` 导入错误除外） |
| `mujoco_mj_ray(model, data, pnt, geomid)` | `pnt`：射线原点；`geomid`：长度 1 的 `np.int32` 数组（出参） | `float`（沿 `(0,0,-1)` 的距离） | 函数内 `import mujoco` 失败时 `ImportError` |

注意 `mujoco_mj_ray` 传的是 `geomgroup=None`（不过滤 geom 组）、`flg_static=1`、`bodyexclude=-1`。

## 5. 内部结构与数据流

构造 → 几何（真正生成 MJCF 的链）：

```
StairSpec(step_height, tread, ..., name_prefix)
  → build_terrain_xml(spec)
      → spec.n_up == 0 and spec.n_down == 0 ?  return ""
      → "\n" + spec.to_xml()
          → spec.step_blocks()
              → crest_height / up_end_x（属性，重算）
              → 上升块 ×n_up、crest 块（n_down>0）、下降块 ×n_down（跳过 top<=1e-9）
              → 或 top 平台块（n_down==0 且 top_landing>0.0 且 n_up>0）
          → 对每块 f'<geom name=... pos="{:.6f} ..." size="{:.6f} ..." condim="3"
                            friction="{fr}" rgba="{rgba}" group="{TERRAIN_GEOM_GROUP}"/>'
      → 返回给 robotdog/go2_model.py
  → go2_model.py:334  build_terrain_xml(self.stair) 注入 <worldbody>
  → go2_model.py:403  compile_scene(...)  →  mujoco.MjModel
```

高度剖面的消费者（环境层，全部走 `env.stair.height_at`）：

```
env.py:484   terrain_height            = self.stair.height_at(float(self.base_pos[0]))
env.py:529   foot_clearance (property)  →  env.py:543  [self.stair.height_at(float(xi)) for xi in arr]
env.py:932, 944, 959  支撑面/判定用的每足地形高  [self.stair.height_at(float(x)) for x in self.foot_pos[:, 0]]
```

随机化链（`reset()` 时）：

```
np.random.Generator（由环境持有）
  → sample_spec(rng, rand, base)          # ⚠ 必须这样调；见 §9
      → base = base or StairSpec()
      → n_up = int(rng.integers(lo, hi + 1))
      → replace(base, step_height=uniform(*height_range), tread=uniform(*tread_range),
                width=uniform(*width_range), n_up=n_up, n_down=n_up,
                start_x=rand.start_x, landing=rand.landing)
  → Go2StairsEnv.set_stair(spec)  → env.py:363-372  _build_model() 重新编译（~250 ms）
```

射线校验链：

```
xs = np.linspace(x0, x1, n)
  → raycast_heights(model, data, spec, xs, y=None, z_high=6.0)
      → y = spec.center_y
      → 对每个 x： pnt = [x, y, 6.0]
          → mujoco_mj_ray(model, data, pnt, geomid)
              → import mujoco
              → mujoco.mj_ray(m, d, pnt, [0,0,-1], None, 1, -1, geomid)
          → out[k] = 6.0 - dist  if dist >= 0  else nan
  → 与 [spec.height_at(x) for x in xs] 逐点比对（校验用）
```

源码 docstring 里的布局说明（`Layouts (walking direction is +x)`）：

```
up      : 平地 → n_up 级上升 → 平的顶层平台
pyramid : 平地 → n_up 级上升 → 平的 crest → n_down 级下降 → 平地
          （直接同时提供上楼与下楼，不需要挖坑）
```

## 6. 关键常量与默认值

| 常量 / 默认值 | 真实值 | 位置 |
| --- | --- | --- |
| `DEFAULT_STAIR_FRICTION` | `(0.9, 0.005, 0.0001)` | 模块级 |
| `DEFAULT_STAIR_RGBA` | `"0.62 0.60 0.55 1"` | 模块级 |
| `DEFAULT_EDGE_RGBA` | `"0.95 0.72 0.15 1"` | 模块级（**当前无使用者**） |
| `TERRAIN_GEOM_GROUP` | `1` | 模块级（载荷，见 §3.4） |
| `CREST_SEAM_OVERLAP` | `1.0`（单位：踏面） | 模块级（载荷，见 §3.3） |
| `StairSpec.step_height` | `0.05` | 字段默认 |
| `StairSpec.tread` | `0.50` | 字段默认 |
| `StairSpec.width` | `2.00` | 字段默认 |
| `StairSpec.n_up` | `8` | 字段默认 |
| `StairSpec.n_down` | `0` | 字段默认 |
| `StairSpec.start_x` | `1.50` | 字段默认 |
| `StairSpec.landing` | `1.20` | 字段默认 |
| `StairSpec.top_landing` | `1.20` | 字段默认（必须 > 0，见 §3.9） |
| `StairSpec.center_y` | `0.0` | 字段默认 |
| `StairSpec.friction` | `DEFAULT_STAIR_FRICTION` | 字段默认 |
| `StairSpec.rgba` | `DEFAULT_STAIR_RGBA` | 字段默认 |
| `StairSpec.name_prefix` | `"stair"` | 字段默认 |
| `TerrainRandomization.height_range` | `(0.03, 0.08)` | 字段默认 |
| `TerrainRandomization.tread_range` | `(0.30, 0.60)` | 字段默认 |
| `TerrainRandomization.width_range` | `(1.2, 3.0)` | 字段默认 |
| `TerrainRandomization.n_up_range` | `(4, 9)` | 字段默认 |
| `TerrainRandomization.start_x` | `1.20` | 字段默认 |
| `TerrainRandomization.landing` | `1.20` | 字段默认 |
| `demo_stairs_spec` 默认 | `step_height=0.05, tread=0.50, n_up=8, n_down=0` | 签名 |
| `demo_stairs_spec` 的 `top_landing` | `1.20 if n_down == 0 else 0.0` | `kw.setdefault` |
| `demo_pyramid_spec` 默认 | `step_height=0.05, tread=0.50, n_up=8, n_down=8` | 签名 |
| `to_xml` 的 `indent` 默认 | `"    "`（4 空格） | 签名 |
| `to_xml` 坐标格式 | `:.6f` | 源码 f-string |
| `to_xml` 摩擦格式 | `:g`（逐元素，空格连接） | `fr = " ".join(f"{v:g}" for v in self.friction)` |
| MJCF `condim` | `"3"`（写死） | `to_xml` |
| 下降段零高度守门 | `if top <= 1e-9: continue` | `step_blocks()` |
| `raycast_heights` 的 `z_high` 默认 | `6.0` | 签名 |
| `raycast_heights` 的射线方向 | `np.array([0.0, 0.0, -1.0])` | `mujoco_mj_ray` |
| `mujoco_mj_ray` 的 `flg_static` | `1` | 位置参数 |
| `mujoco_mj_ray` 的 `bodyexclude` | `-1` | 位置参数 |
| `mujoco_mj_ray` 的 `geomgroup` | `None`（**不过滤**） | 位置参数 |
| 默认 `StairSpec()` 的派生量 | `crest_height = 0.4`、`up_end_x = 5.5`、`down_end_x = 6.7`、`span_end_x = 6.7`、`max_height = 0.4`；`len(step_blocks()) == 9`（`stair_up0..up7` + `stair_top`） | 计算值 |
| `demo_pyramid_spec()` 的派生量 | `down_end_x = 10.7`、`span_end_x = 10.7`、`len(step_blocks()) == 16` | 计算值 |
| `height_at` 阶跃实测（默认 spec） | `height_at(5.499)=0.4`、`5.500 → 0.4`、`6.69 → 0.4`、`6.70 → 0.0` | 实测 |

## 7. 与其它模块的关系

**上游**：stdlib `dataclasses`（`dataclass, field, replace` —— **`field` 被导入但未使用**）、`numpy`。**模块顶层不 import mujoco**；只有 `mujoco_mj_ray()` 函数体内 `import mujoco`。

**下游**：

| 位置 | 使用的符号 | 依赖方向为什么是这样 |
| --- | --- | --- |
| `robotdog/go2_model.py:42` | `TERRAIN_GEOM_GROUP` | 生成地面 plane 时也要打同一个 group（`default_floor_geom`），否则测距看不到地面 |
| `robotdog/go2_model.py:334` | `build_terrain_xml(self.stair)` | 拼场景字符串需要地形片段；**编译**在 `go2_model.py` 里做，本模块只产字符串 |
| `robotdog/env.py:43` | `StairSpec, build_terrain_xml` | 环境层持有 spec 并驱动编译 |
| `robotdog/env.py:277/369` | `StairSpec()` 默认值、`set_stair` | 课程切阶段时换 spec 并重编译 |
| `robotdog/env.py:424/431/432` | `up_end_x`, `crest_height` | 出生点与**成功判据** |
| `robotdog/env.py:484/543/932/944/959` | `height_at` | 地形高、`foot_clearance`、支撑面参照（见 §3.5） |
| `robotdog/curriculum.py:27` | `StairSpec, demo_stairs_spec, flat_spec` | 课程各阶段**配置**由构造器拼出（`:143` 平地、`:153/163/173` 三档台阶） |
| `robotdog/evaluate.py:29` | `StairSpec, TerrainRandomization, demo_stairs_spec, flat_spec` | 评估集：训练用档 + 未见过档（`:59-75`）+ 随机档（`:87`，⚠ 见 §9） |
| `robotdog/render.py:32` | `StairSpec` | 渲染 HUD 要显示 `tread` / `crest_height`（`:263`） |
| `scripts/train.py:643` | `stage.spec.crest_height` | 课程进度打印 |
| `scripts/render_checkpoint.py:64, 466, 574`、`scripts/eval.py:147` | `StairSpec, flat_spec` | 脚本层用工具层构造地形 |

**为什么方向是「环境层 ← 工具层」而不是反过来**：本模块只认识「参数 → 几何/高度」，不需要知道谁在用它 —— 它给 `env.py` 提供 `height_at`，但从不 `import env`。这样奖励口径、终止口径、评估口径**共用同一个高度真值**，而三者的**用法**差异留在环境层（例如 `terminate_height_ref="support"` 是 `env.py` 的选择，不是 `height_at` 的选择）。

**与 `robotdog/tools/touch.py` 的关系**：`touch.py` 的 `geomgroup` 掩码之所以能工作，完全依赖本模块把**所有**地形几何（含地面）打进 `TERRAIN_GEOM_GROUP`；`touch.py:53` 直接用 `from robotdog.tools.terrain import TERRAIN_GEOM_GROUP` 构造掩码。这是工具层**内部**唯一一条依赖（`touch → terrain`），也是 `robotdog/tools/__init__.py` 里导入顺序天然正确的原因。

**文档路径说明**：本模块的规范导入路径是 `robotdog.tools.terrain`（文件在 `robotdog/tools/terrain.py`）。`experience.md` §1.1 与 §1.2 写作 `terrain.py`，指的是同一个文件；仓里**没有**顶层 `robotdog/terrain.py`。

## 8. 实测与经验教训

* 共面接缝夹脚（`CREST_SEAM_OVERLAP` 的由来与实际收益），见 `experience.md` §1.1。
* geomgroup 默认隐藏导致「爬空气」，见 `experience.md` §1.2。
* 零高度台阶无法编译，见 `experience.md` §1.3。
* 阶跃的 `height_at` 造成幽灵摔倒，以及 `base_height` 参照面同样阶跃，见 `experience.md` §1.4、§3.3。
* 与台阶高相关的奖励目标写死（抬升目标）以及参照面口径问题，见 `experience.md` §2.3、§6.1。
* 课程各阶段的台阶高度/踏面/级数与毕业阈值，见 `train.md` §1（本文档不重复其结论）。

## 9. 已知限制与坑

* **零高度台阶无法编译**：`demo_stairs_spec(step_height=0.0, n_up=8)` 生成的 XML 含 `size="0.250000 1.000000 0.000000"`，直接编译得到
  `ValueError: Error: size 2 must be positive in geom`
  （并附带 `Element name 'stair_up0', id 0, line 2`）；经 `go2_model.py:428` 包装后调用方看到
  `RuntimeError: MJCF compile failed; dumped to <...>\.generated\failed_scene.xml`。
  平地必须用 `flat_spec()` / `flat_spec(n_up=0, n_down=0, ...)`（`experience.md` §1.3）。
* **`TerrainRandomization` 上没有 `sample_spec` 方法**，它是模块级函数。`robotdog/evaluate.py:87` 写成 `spec = rand.sample_spec(rng)`，实测会抛
  `AttributeError: 'TerrainRandomization' object has no attribute 'sample_spec'`。
  正确写法是 `sample_spec(rng, rand)`。触发路径：`robotdog/evaluate.py:100-101` 只在 `only == "random"` 或 `--n-random > 0` 时传非零数量 → `scripts/eval.py:308` 的 `all_terrains(include_random=..., ...)`；默认 `only="all"` 传 0，`range(0)` 不会执行到这行，所以默认评估路径不会炸。
* **随机化只产生金字塔**：`sample_spec` 里 `n_down = n_up`（源码 docstring `pyramid: up then down`），无法采样「只上不下」的随机楼梯。
* **`height_at` 是阶跃的**：在 `start_x + i*tread` 与 `up_end_x + top_landing` 处跳变。任何把单点 `height_at(base_x)` 当参照面的量都会在跨越立面时**跳一整级**，逐帧翻转（`experience.md` §1.4、§3.3）。
* **`height_at` 与几何必须同步**：`step_blocks()` 用 `CREST_SEAM_OVERLAP` 扩展 crest/top 块，但 `height_at` **不读**这个常量 —— 正确性依赖于「两个实心块的并集不变」。若有人把这个常量改成 > 1（回伸过多、block 越过前一级顶面）或让 crest 高度与最后一级踏面不等，剖面与几何就会分叉。
* **`mujoco_mj_ray` 不过滤 geom 组**（`geomgroup=None`）：机器人自身的腿会挡住射线。用它做「奖励侧高度查询」会得到错误结果；`raycast_heights` 只适合做**校验**。
* **`raycast_heights` / `mujoco_mj_ray` / `demo_pyramid_spec` / `DEFAULT_EDGE_RGBA` / `span_end_x` / `max_height` 在当前仓库里都没有使用者**（只在 `terrain.py` 内部或定义处出现）。改动它们不会影响任何现有行为，但也意味着它们**没有回归覆盖**。
* **`step_blocks()` 的下降段有零高度守门，上升段没有**：`if top <= 1e-9: continue` 只作用于 `_down{j}`；`n_up > 0` 且 `step_height = 0` 时仍会生成非法 block（见上）。
* **`to_xml()` 返回的串不带换行前缀**，而 `build_terrain_xml()` 会加一个 `"\n"`。直接调用 `to_xml()` 拼进场景时若忘记补换行，生成的 MJCF 会把 `<geom>` 与前一个标签挤在同一行（MuJoCo 仍能解析，但转储文件没法读）。
* **`build_terrain_xml()` 对平地返回空串**：调用方不能用它判断「地形是否就绪」，只能判断「是否生成了 geom」。
* **`StairSpec` 是可变 dataclass**（非 `frozen`），可以就地改字段；共享实例被改动会让报告里的 spec 与实际几何不一致（`experience.md`/`env.py:1198` 会把 `crest_height` 打进 info）。随机化刻意用 `replace()` 避免这一点。
* **`width` 必须为正**：`width <= 0` 会让 `hy = width/2 <= 0`，同样触发 `size must be positive` 编译失败。
* **`n_down * step_height > crest_height` 时下降段不会到地面**：多出的级被 `top <= 1e-9` 守门跳过，`height_at` 也在 `j >= n_down` 处返回 0，二者一致 —— 但这意味着「下降段级数」可以被静默截断，不报错。

## 10. 示例用法

```python
import numpy as np

from robotdog.tools.terrain import (
    StairSpec,
    build_terrain_xml,
    demo_stairs_spec,
    flat_spec,
    sample_spec,
    TerrainRandomization,
)

# 1) 默认台阶：8 级 × 5 cm，踏面 50 cm
spec = StairSpec()
print(spec.crest_height, spec.up_end_x, spec.span_end_x)   # 0.4 5.5 6.7
print(len(spec.step_blocks()))                             # 9  (up0..up7 + top)
print(spec.height_at(5.499), spec.height_at(6.70))         # 0.4 0.0

# 2) 生成注入 <worldbody> 的 MJCF 片段
xml = build_terrain_xml(spec)
if xml:
    print(xml.splitlines()[0])
    # <geom name="stair_up0" type="box" pos="1.750000 0.000000 0.025000" size="0.250000 1.000000 0.025000" condim="3" friction="0.9 0.005 0.0001" rgba="0.62 0.60 0.55 1" group="1"/>

# 3) 平地阶段：必须用 flat_spec（不要用 step_height=0.0 的 demo_stairs_spec）
flat = flat_spec(n_up=0, n_down=0, step_height=0.0)
assert build_terrain_xml(flat) == ""        # 不生成任何地形 geom
print(flat.height_at(3.0))                  # 0.0

# 4) 金字塔（同时上楼与下楼）
pyr = demo_stairs_spec(step_height=0.08, tread=0.50, n_up=6, n_down=6)
print(pyr.top_landing, pyr.down_end_x)      # 0.0 8.7

# 5) 随机化：用模块级函数，不要写 rand.sample_spec(rng)
rng = np.random.default_rng(0)
rand = TerrainRandomization()
s = sample_spec(rng, rand)
print(s.n_up, s.n_down, round(s.step_height, 4))   # n_up == n_down（金字塔）
```
