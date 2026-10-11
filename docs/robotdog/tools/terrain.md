# `robotdog/tools/terrain.py` —— 参数化楼梯地形（纯几何）

> 本模块的规范导入路径是 `robotdog.tools.terrain`。`experience.md` §1.1 / §1.2
> 里写作 `terrain.py`，指的是同一个文件；仓里**没有**顶层 `robotdog/terrain.py`。

## 1. 一句话定位

把「一段楼梯」变成 **(a) 一串实心 box 的 MJCF 文本** 和 **(b) 一个解析高度剖面
`height_at(x)`**，让奖励、出生点、成功判据与评估指标**共用同一份几何真值**。

它是整个仓库的**最底层**：不 import 任何 `robotdog.*` 兄弟模块（连 `mujoco` 都不 import）。

## 2. 职责边界

**做**：

* `StairSpec`：一段楼梯的全部几何参数 + 派生量（crest 高度、各段结束位置）。
* `step_blocks()`：把楼梯展开成实心 box 列表（前→后有序）。
* `to_xml()` / `build_terrain_xml()`：生成注入 `<worldbody>` 的 MJCF `<geom>` 片段。
* `height_at(x)`：**解析**地形高度剖面（阶跃函数）。
* `flat_spec()` / `demo_stairs_spec()`：课程各阶段的便捷构造器。

**不做**：

* **不解析 XML、不注入 XML、不编译模型**。本模块只**返回字符串**；把它拼进场景并
  `mujoco.MjModel` 编译的是 `robotdog/go2_model.py`（`compile_scene`）。
* **完全不 import mujoco**（模块顶层和函数体内都不 import）。所以它可以在没有
  MuJoCo 的进程里被 import、被单测。
* **不建 heightfield / hfield**，不生成网格，不做程序化噪声地形。
* **不知道观测、奖励、终止判据**。它只提供 `height_at`，谁用、怎么用由
  `robotdog/env.py` 决定。
* **不做碰撞过滤**，不改 `condim`/`solref`/`solimp` 之外的选项（`condim="3"` 写死）。
* **不 import `robotdog.env`**，不知道 PPO / 课程 / 训练循环（课程**配置**由
  `robotdog/config.py` 用本模块的构造器拼出来，方向是下游依赖本模块）。

## 3. 关键设计决策与理由

### 3.1 每一级都是**落地实心块**，不是薄板

模块 docstring 第一条：每一级是**落在地面上的实心块**，半尺寸
`(tread/2, width/2, top_z/2)`、中心在 `z = top_z/2`。

理由：薄板的下表面与地面之间会留下一个空隙，脚/小腿可以钻进去，被**从下面**挂住；
实心块的侧面从 z=0 一直到顶面，不存在这个空腔。代价是块与块之间大量**静-静重叠**
—— 模块 docstring 明确指出这无害：只有一对几何体中至少有一个是动态体时 MuJoCo 才会
生成接触。

**被否决的方案**：薄板（省内存、视觉一样）—— 否决，引入「从下方被挂住」的新失败模式，
而本任务的核心难点正是脚与立面的接触。**另一个被否决的方案**：用 `heightfield`
（天然无缝、单 geom）—— 否决，理由见 §3.2。

### 3.2 保留**锐利 box 边**，不用 heightfield

模块 docstring：`heightfield` 会把棱边在一个网格单元内磨圆，而本项目的任务定义就是
「抬脚越过**立面**」。

理由：heightfield 把台阶边缘插值成斜坡，等于**替策略把难点抹平**（多出一个可以借力的
斜面），训练出来的策略在真实台阶几何上不成立。box 的边是数学意义上的锐边，
`height_at` 的阶跃函数才与几何严格一致（见 §3.4）。

**被否决的方案**：heightfield —— 否决，除上述「抹平难点」外，还破坏 `height_at` 与
几何的一致性（阶梯函数 vs 插值斜坡），而 §3.4 说明这个一致性是奖励正确性的前提。

### 3.3 【载荷决策 a】`CREST_SEAM_OVERLAP = 1.0`

```python
#: 平的 crest / 顶层平台 *向回*伸进最后一级上升台阶的长度（单位：踏面）。
CREST_SEAM_OVERLAP = 1.0  # 单位：踏面
```

**必须 > 0**，因为顶层平台与最后一级踏面**等高**，两块 box 共面。当它们只是**相邻**
（crest 正好从 `up_end_x` 开始）时，接触面是一条零高度的缝：

1. 中心落在这条缝上的脚球，对两个竖直面都是**零穿透**，但对两个顶面都还有几十毫米距离；
2. MuJoCo 对「球在盒内」的接触沿**最小穿透轴**求解 —— 于是两个 box 都在**水平方向**
   推这只脚，方向相反；
3. 两个推力相互抵消，z 方向没有任何东西支撑脚，脚被夹住（pin）：x 被钉死、下沉到踏面下
   30~40 mm，且因为接触法向是水平的，`touch.riser_graze` 每一步都被点亮。

实测（8 cm × 8 级，`.tmp/stall_probe.py`）：两只前脚在剩余 **700** 个控制步里冻结在
`x = 5.500 == up_end_x`，即机器人爬完整段楼梯后**停在终点线前 0.114 m** 直到 20 s 上限
—— 这正是高台阶训练长期卡在 20~40 % 的原因（实验记录见 `experience.md` §1.1）。

修法**不是**把 crest 抬高或降低（那会改变几何与解析剖面），而是让 crest
**向最后一级踏面回伸一个踏面**：

```python
ov = CREST_SEAM_OVERLAP * self.tread
cx = self.up_end_x + (self.top_landing - ov) / 2.0
size = ((self.top_landing + ov) / 2.0, hy, top / 2.0)
```

于是 crest 的后侧面被**埋进**最后一级台阶内部，共面的那对表面消失，只剩真正的立面。
模块 docstring 强调：实心块的**并集**不变，所以解析剖面与每一个渲染像素都完全一致。

（实现见 `robotdog/tools/terrain.py:147-202` 的 `step_blocks()`。）

取 `1.0`（一个完整踏面）而不是「一小段 overlap」：整块踏面回伸让重叠体积足够大，避免留下
**部分共面**的窄缝（窄缝同样会夹脚）；而且表达式保持整数倍踏面，`cx`/`size` 与
`step_blocks()` 的其它块形式统一，可读性最好。

**被否决的方案**：

* 让 crest 从 `up_end_x - tread` 起算（等价于手工改 `up_end_x`）—— 否决：`up_end_x` 是
  **成功判据**与解析剖面共用的量（`env.py` 的 `_success_reached()`），改它会同时移动
  终点线，把一个接触问题变成任务定义问题。
* 给 crest 加一点 z 偏移（例如 `+1e-4`）以消除共面 —— 否决：制造一个肉眼不可见但物理上
  真实的台阶，脚会被这个伪台阶绊住；而且解析剖面与几何会对不上。
* 用 `margin` / `gap` 之类的接触参数（`solref`、`margin`）绕开 —— 否决：那是全局接触参数，
  会改变**所有**接触（含脚-地面），副作用远超问题本身。

### 3.4 【载荷决策 b】`TERRAIN_GEOM_GROUP = 1`

```python
TERRAIN_GEOM_GROUP = 1
```

把全部地形几何放进一个专用 geom group，前向探针就能「向前看」而看不到机器人自己的前腿：
`mj_ray` 接受一个 *geom group mask*，只开这一组的掩码构造性地让探针看不见机器人。

**为什么偏偏是 1**，这个值承担双重载荷（都写在源码注释里）：

1. Unitree 的 MJCF 把 `group="2"` 分给纯视觉 geom、`group="3"` 分给每一个碰撞 geom
   （躯干盒、膝部圆柱、四个脚球），所以 2 和 3 已被占用；没有显式 group 的 geom 落在 0。
2. MuJoCo 的 `mjv_defaultOption` 出厂就是 `geomgroup = [1, 1, 1, 0, 0, 0]`
   —— **group 3、4、5 默认被隐藏**。于是把楼梯打在 group 4 上，它在每一个离屏渲染里都
   不可见，而物理完全不受影响：画面上是狗在爬空气，HUD 却愉快地报告高度增益。
   （真的发生过；见 `experience.md` §1.2。）group 1 落在默认可见范围内。

`robotdog/env.py` 的 `render()` 还会额外强制打开*全部六个* group，所以再改这个常量也
不会悄悄把地形藏起来。

### 3.5 【必须知道的陷阱】`demo_stairs_spec(step_height=0.0, n_up=8)` 无法编译

`step_blocks()` 里上升段是 `top = (i + 1) * self.step_height`、`size_z = top / 2.0`。
当 `step_height = 0.0` 时**每一级都是零高度 box**：`size="0.250000 1.000000 0.000000"`。
零尺寸是非法几何体，MuJoCo 编译直接失败：

```
ValueError: Error: size 2 must be positive in geom
Element name 'stair_up0', id 0, line 2
```

经 `robotdog/go2_model.py` 的 `compile_scene` 包装后，调用方看到的是：

```
RuntimeError: MJCF compile failed; dumped to <...>\.generated\failed_scene.xml
```

**正确做法**：平地一律用 `flat_spec(...)` —— 此时 `n_up == n_down == 0`，
`build_terrain_xml()` 直接返回 `""`，**根本不生成任何地形 geom**，所以 `step_height`
取什么值都无所谓（`robotdog/config.py` 的 `config_to_stages()` 正是
`flat_spec(n_up=0, n_down=0, step_height=0.0, ...)`）。见 `experience.md` §1.3。

`step_blocks()` 的下降段有一道自己的零高度守门 `if top <= 1e-9: continue`，保证
「级数多于 `crest_height / step_height`」时不会生成零高度块；但**上升段没有这道守门**，
所以这个陷阱只对「`step_height = 0` 且 `n_up > 0`」成立。

**被否决的方案**：在 `step_blocks()` 里也跳过零高度上升块 —— 否决：那会让
`n_up = 8, step_height = 0` 静默变成平地，把一个**调用错误**伪装成合法地形
（`height_at` 仍会算出 8 级台阶的位置，与几何不一致）。让编译失败是更安全的行为。

### 3.6 `top_landing` 默认 `1.20`（而不是 `0.0`）是**任务可成功性**的前提

字段上方的注释写得很明确：`top_landing` 必须 > 0，否则地形根本无法成功 ——
`height_at` 在 `up_end_x + top_landing` 之后返回 0，而成功判据要求 base 越过 `up_end_x`
**并且**处在 crest 高度上。用旧的默认值 `0.0` 时，一个裸的 `StairSpec()` 在 crest 处
就是 0.40 m 悬崖，所以 `Go2StairsEnv()` 永远报告不出成功。

**被否决的方案**：保持 `0.0` 并在成功判据里放宽「高度必须等于 crest」—— 否决：那会在
狗**掉下悬崖**的那一刻宣布成功（它确实越过了 `up_end_x`），把成功判据变成失败判据。

### 3.7 `to_xml()` 的摩擦串用 `:g` 格式化、坐标用 `:.6f`

```python
fr = " ".join(f"{v:g}" for v in self.friction)
f'pos="{cx:.6f} {cy:.6f} {cz:.6f}" '
f'size="{hx:.6f} {hy:.6f} {hz:.6f}" '
f'condim="3" friction="{fr}" rgba="{self.rgba}" group="{TERRAIN_GEOM_GROUP}"/>'
```

* 摩擦用 `:g`：`(0.9, 0.005, 0.0001)` 写成 `"0.9 0.005 0.0001"` —— 去掉尾随零，避免
  `0.000100` 这种噪声，也避免浮点展开（`0.00010000000000000000479`）。
* 坐标用 `:.6f`：微米级精度足够表达任何台阶尺寸，同时输出**定长**（便于 diff 生成的 XML，
  `.generated/failed_scene.xml` 就是给人读的）。
* `rgba` 与 `friction` 是**字符串/元组透传**（`rgba` 本身就是 MJCF 语法串），因此调用方
  可以直接写 `"1 0 0 1"`。
* `indent: str = "    "`（4 空格）暴露为参数，因为它要被插进不同缩进层级的 `worldbody`。

### 3.8 便捷构造器只有两个，且都只是 `StairSpec` 的薄包装

* `flat_spec(**kw)`：`n_up` / `n_down` 用 `setdefault` 设成 `0`，其余透传。**平地只用它**。
* `demo_stairs_spec(...)`：显式参数 + `kw.setdefault("top_landing", 1.20 if n_down == 0 else 0.0)`。

`demo_stairs_spec` 里那句 `top_landing` 的 `setdefault` 是**任务定义**层面的决策：
只在「只上不下」时给顶层平台；金字塔模式下 crest 后面直接接下降段，`top_landing` 惰性
（`step_blocks()` 与 `height_at()` 在 `n_down > 0` 时都不读它）。用 `setdefault` 而非直接
赋值，是为了让调用方仍能显式覆盖。

## 4. 公开 API

### 4.1 模块级常量

```python
DEFAULT_STAIR_FRICTION: tuple[float, float, float] = (0.9, 0.005, 0.0001)   # :27
DEFAULT_STAIR_RGBA = "0.62 0.60 0.55 1"                                     # :28
TERRAIN_GEOM_GROUP = 1                                                     # :48
CREST_SEAM_OVERLAP = 1.0                                                   # :67
```

> 模块级常量**只有这四个**。已删除（不要再引用）：`DEFAULT_EDGE_RGBA`
> （`"0.95 0.72 0.15 1"`，零使用者）、`TerrainRandomization`、`sample_spec()`、
> `demo_pyramid_spec()`、`raycast_heights()`、`mujoco_mj_ray()` —— 全部在死代码清理中
> 移除，理由见 §8。

### 4.2 `@dataclass class StairSpec`

字段（名称 / 类型 / 默认值 / 语义，逐项对应源码 `robotdog/tools/terrain.py:78-94`）：

| 字段 | 类型 | 默认值 | 语义（源码注释要点） |
| --- | --- | --- | --- |
| `step_height` | `float` | `0.05` | 立面高度 [m]（台阶高度） |
| `tread` | `float` | `0.50` | 沿 +x 的踏面深度 [m]（台阶宽度） |
| `width` | `float` | `2.00` | 沿 y 的横向范围 [m] |
| `n_up` | `int` | `8` | 上升立面的级数 |
| `n_down` | `int` | `0` | 为 0 时只上行；>0 则构成金字塔 |
| `start_x` | `float` | `1.50` | 第一道立面开始处的世界坐标 x |
| `landing` | `float` | `1.20` | 上升段与下降段之间平 crest 的长度 [m] |
| `top_landing` | `float` | `1.20` | `n_down == 0` 时的平顶层平台 [m]。必须 > 0（见 §3.6） |
| `center_y` | `float` | `0.0` | 楼梯在 y 方向的中心（出生点与测距都用它） |
| `friction` | `tuple[float, float, float]` | `DEFAULT_STAIR_FRICTION` | 直接写进 geom 的 `friction=` |
| `rgba` | `str` | `DEFAULT_STAIR_RGBA` | MJCF 颜色串 |
| `name_prefix` | `str` | `"stair"` | `step_blocks()` 生成的名字前缀（如 `stair_up0`） |

派生属性（全部只读、每次访问重算 —— 所以**不存在「改了输入忘了同步缓存」的窗口**）：

| 属性 | 返回类型 | 公式 | 源码行 |
| --- | --- | --- | --- |
| `crest_height` | `float` | `n_up * step_height` | `:99-100` |
| `up_end_x` | `float` | `start_x + n_up * tread`（最后一道立面的踏面结束处 = crest 起点） | `:103-105` |
| `down_end_x` | `float` | `up_end_x + landing + n_down * tread`（下降段回到地面高度处的 x） | `:108-110` |

> 已删除的派生属性：`span_end_x`（`x where solid staircase geometry ends`）与
> `max_height`（等于 `crest_height`）。它们零使用者。

方法：

```python
def step_blocks(self) -> list[tuple[str, tuple[float, float, float],
                                   tuple[float, float, float]]]:
    """返回 ``[(name, (cx, cy, cz), (hx, hy, hz)), ...]``，逐块列出。"""


def to_xml(self, indent: str = "    ") -> str:
    """整段楼梯的 MJCF ``<geom>`` 元素。"""


def height_at(self, x: float) -> float:
    """世界坐标 x 处的解析地形高 [m]（楼梯之外为 0）。"""
```

* `step_blocks()`：返回 `(名字, 中心坐标三元组, 半尺寸三元组)` 列表，**前→后有序**
  （源码注释：`The full staircase is emitted front-to-back so that the raycast and the
  analytic profile in height_at line up exactly.`）。`hx, hy = tread / 2.0, width / 2.0`；
  上升块 `i` 的名字是 `f"{name_prefix}_up{i}"`、中心
  `(start_x + (i+0.5)*tread, center_y, top/2.0)`、半尺寸 `(tread/2, width/2, top/2.0)`，
  其中 `top = (i+1)*step_height`。`n_down > 0` 时追加 `"_crest"` 块再追加 `"_down{j}"`；
  否则在 `top_landing > 0.0 and n_up > 0` 时追加 `"_top"` 块。纯函数，无异常。
* `to_xml(indent="    ")`：返回**用 `\n` 连接**的多个 `<geom .../>` 行，**不带**首尾换行。
  每个 geom 固定带 `type="box"`、`condim="3"`、`friction`、`rgba`、`group`。无异常。
* `height_at(x)`：返回该 x 处的解析地形高（米），楼梯之外返回 `0.0`。**阶跃**，无插值。
  参数是 `float`；传 numpy 标量也可以。

### 4.3 便捷构造器

```text
def flat_spec(**kw) -> StairSpec                              # :213  见 §3.8
def demo_stairs_spec(
    step_height: float = 0.05,
    tread: float = 0.50,
    n_up: int = 8,
    n_down: int = 0,
    **kw,
) -> StairSpec                                               # :220  同上

def build_terrain_xml(spec: StairSpec) -> str:               # :238
    """要注入 ``<worldbody>`` 的 MJCF 片段。"""
```

| 函数 | 参数语义 | 返回 | 异常 |
| --- | --- | --- | --- |
| `flat_spec(**kw)` | `n_up`/`n_down` 默认设 `0`，其余字段透传 | `StairSpec` | 透传 `StairSpec` 的 `TypeError`（未知字段） |
| `demo_stairs_spec(step_height=0.05, tread=0.50, n_up=8, n_down=0, **kw)` | 四个显式参数；`kw.setdefault("top_landing", 1.20 if n_down == 0 else 0.0)` | `StairSpec` | 同上 |
| `build_terrain_xml(spec)` | `spec`：`StairSpec` | **`""`**（当 `n_up == 0 and n_down == 0`）；否则 `"\n" + spec.to_xml()`（**带一个前导换行**） | 无 |

## 5. 内部结构与数据流

构造 → 几何（真正生成 MJCF 的链）：

```
StairSpec(step_height, tread, ..., name_prefix)
  → build_terrain_xml(spec)                                  robotdog/tools/terrain.py:283-287
      → spec.n_up == 0 and spec.n_down == 0 ?  return ""
      → "\n" + spec.to_xml()
          → spec.step_blocks()
              → crest_height / up_end_x（属性，重算）
              → 上升块 ×n_up、crest 块（n_down>0）、下降块 ×n_down（跳过 top<=1e-9）
              → 或 top 平台块（n_down==0 且 top_landing>0.0 且 n_up>0）
          → 对每块 f'<geom name=... pos="{:.6f} ..." size="{:.6f} ..." condim="3"
                            friction="{fr}" rgba="{rgba}" group="{TERRAIN_GEOM_GROUP}"/>'
      → 返回给 robotdog/go2_model.py
  → robotdog/go2_model.py  build_terrain_xml(self.stair) 注入 <worldbody>
  → robotdog/go2_model.py  compile_scene(...)  →  mujoco.MjModel
```

高度剖面的消费者（全部在环境层，走 `env.stair.height_at`）：

```
robotdog/env.py        terrain_height_under_base = self.stair.height_at(float(self.base_pos[0]))
robotdog/env.py        foot_clearance (property)  →  _terrain_height() → height_at(每只脚的 x)
robotdog/env.py        _support_height()          = min(每足地形高)
robotdog/env.py        _support_height_mean()     = mean(每足地形高)
robotdog/env.py        成功判据                    = terrain_height_under_base >= stair.crest_height - 1e-9
```

模块 docstring 里的布局说明（前进方向为 +x）：

```
up      : 平地 → n_up 级上升立面 → 平的顶层平台
pyramid : 平地 → n_up 级上升立面 → 平的 crest → n_down 级下降立面 → 地面
          （直接同时提供上楼与下楼，不需要挖坑）
```

## 6. 关键常量与默认值

| 常量 / 默认值 | 真实值 | 位置 |
| --- | --- | --- |
| `DEFAULT_STAIR_FRICTION` | `(0.9, 0.005, 0.0001)` | 模块级 `:27` |
| `DEFAULT_STAIR_RGBA` | `"0.62 0.60 0.55 1"` | 模块级 `:28` |
| `TERRAIN_GEOM_GROUP` | `1` | 模块级 `:48`（载荷，见 §3.4） |
| `CREST_SEAM_OVERLAP` | `1.0`（单位：踏面） | 模块级 `:67`（载荷，见 §3.3） |
| `StairSpec.step_height` | `0.05` | 字段默认 |
| `StairSpec.tread` | `0.50` | 字段默认 |
| `StairSpec.width` | `2.00` | 字段默认 |
| `StairSpec.n_up` | `8` | 字段默认 |
| `StairSpec.n_down` | `0` | 字段默认 |
| `StairSpec.start_x` | `1.50` | 字段默认 |
| `StairSpec.landing` | `1.20` | 字段默认 |
| `StairSpec.top_landing` | `1.20` | 字段默认（必须 > 0，见 §3.6） |
| `StairSpec.center_y` | `0.0` | 字段默认 |
| `StairSpec.friction` | `DEFAULT_STAIR_FRICTION` | 字段默认 |
| `StairSpec.rgba` | `DEFAULT_STAIR_RGBA` | 字段默认 |
| `StairSpec.name_prefix` | `"stair"` | 字段默认 |
| `demo_stairs_spec` 默认 | `step_height=0.05, tread=0.50, n_up=8, n_down=0` | 签名 |
| `demo_stairs_spec` 的 `top_landing` | `1.20 if n_down == 0 else 0.0` | `kw.setdefault` |
| `to_xml` 的 `indent` 默认 | `"    "`（4 空格） | 签名 |
| `to_xml` 坐标格式 | `:.6f` | 源码 f-string |
| `to_xml` 摩擦格式 | `:g`（逐元素，空格连接） | `fr = " ".join(f"{v:g}" for v in self.friction)` |
| MJCF `condim` | `"3"`（写死） | `to_xml` |
| 下降段零高度守门 | `if top <= 1e-9: continue` | `step_blocks()` |
| 默认 `StairSpec()` 的派生量 | `crest_height = 0.4`、`up_end_x = 5.5`、`down_end_x = 6.7`；`len(step_blocks()) == 9`（`stair_up0..up7` + `stair_top`） | 计算值 |
| `height_at` 阶跃实测（默认 spec） | `height_at(5.499)=0.4`、`5.500 → 0.4`、`6.69 → 0.4`、`6.70 → 0.0` | 实测 |
| 文件行数 | 242 行 | 实测 |

## 7. 与其它模块的关系

**上游**：stdlib `dataclasses`（只用 `dataclass` 与 `field`）+ 无第三方依赖。
**完全不 import numpy，也不 import mujoco** —— 这是删除 `raycast_heights()` /
`mujoco_mj_ray()` 之后的结果，本模块现在是纯标准库 + `dataclasses`。

**下游**：

| 位置 | 使用的符号 | 依赖方向为什么是这样 |
| --- | --- | --- |
| `robotdog/go2_model.py:229` | `TERRAIN_GEOM_GROUP` | 生成地面 plane 时也要打同一个 group（`default_floor_geom`） |
| `robotdog/env.py:329` | `StairSpec, build_terrain_xml` | 环境层持有 spec 并驱动编译（`_build_model()` 调 `build_terrain_xml(self.stair)`） |
| `robotdog/env.py` | `height_at` | 地形高、`foot_clearance`、支撑面参照（见 §5） |
| `robotdog/env.py` | `crest_height` | 出生点与**成功判据**（`terrain_height_under_base >= crest_height - 1e-9`） |
| `robotdog/config.py:326` | `demo_stairs_spec, flat_spec` | `config_to_stages()` 按 `config.yaml` 造阶段表 —— **当前唯一的课程构造路径**（见 `docs/robotdog/config.md`） |
| `robotdog/policy.py:315/318` | `StairSpec, demo_stairs_spec, flat_spec` | `stair_from_height()` 由高度造地形，服务 `play.py` |
| `robotdog/training.py:38` | `StairSpec` | 建环境/向量环境时透传 spec |
| `scripts/play.py` | `stair.crest_height` | 评估/渲染的头部信息打印 |

**为什么方向是「环境层 ← 工具层」而不是反过来**：本模块只认识「参数 → 几何/高度」，
不需要知道谁在用它 —— 它给 `env.py` 提供 `height_at`，但从不 `import env`。这样奖励口径、
终止口径、评估口径**共用同一个高度真值**，而三者的**用法**差异留在环境层
（例如 `terminate_height_ref="support"` 是 `env.py` 的选择，不是 `height_at` 的选择）。

**曾经的 `robotdog/tools/touch.py` 依赖**：历史上 `touch.py` 的前向测距靠
`from robotdog.tools.terrain import TERRAIN_GEOM_GROUP` 构造「只对地形生效」的射线掩码，
那是工具层**内部**唯一一条依赖。触觉/测距整套逻辑已随「盲走楼梯」重构删除，现在工具层
内部**没有任何跨模块依赖**，`TERRAIN_GEOM_GROUP` 的唯一消费者是
`robotdog/go2_model.py:39`（给地面 plane 打同一组，保留可以让地形与地面在渲染器里同组可见）。

## 8. 实测与经验教训

* 共面接缝夹脚（`CREST_SEAM_OVERLAP` 的由来与实际收益），见 `experience.md` §1.1。
* geomgroup 默认隐藏导致「爬空气」，见 `experience.md` §1.2。
* 零高度台阶无法编译，见 `experience.md` §1.3。
* 阶跃的 `height_at` 造成幽灵摔倒，以及 `base_height` 参照面同样阶跃，见 `experience.md`
  §1.4、§3.3。
* 与台阶高相关的奖励目标写死（抬升目标）以及参照面口径问题，见 `experience.md` §2.3、§3.3，
  以及能力边界 §6。
* 课程各阶段的台阶高度/踏面/级数与毕业阈值，见 `train.md` §1（本文档不重复其结论）。

**本轮死代码清理删掉的六个符号（与理由）**：

| 删除项 | 理由 |
| --- | --- |
| `DEFAULT_EDGE_RGBA = "0.95 0.72 0.15 1"` | 零使用者（只在定义处出现） |
| `TerrainRandomization` | 逐 episode 地形随机化的数据类，重构后零使用者 |
| `sample_spec(rng, rand, base=None)` | 只服务 `TerrainRandomization` |
| `demo_pyramid_spec(...)` | 只服务已被删除的旧课程/评估代码，`demo_stairs_spec(n_down=n)` 完全等价 |
| `raycast_heights(...)` | MuJoCo 向下射线高度查询，消费者（旧 `evaluate.py`）已删；唯一用途是「校验解析剖面」，而解析剖面本身已有单点验证 |
| `mujoco_mj_ray(...)` | 只被 `raycast_heights` 调用；删掉后本模块**彻底不依赖 mujoco/numpy** |
| `StairSpec.span_end_x` / `StairSpec.max_height` | 分别是 `down_end_x`/`up_end_x` 与 `crest_height` 的派生别名，零使用者 |

顺带修掉的一处**历史必炸缺陷**：已删除的 `robotdog/evaluate.py` 曾把它写成方法调用
`spec = rand.sample_spec(rng)` —— `AttributeError: 'TerrainRandomization' object has no
attribute 'sample_spec'`（正确写法是模块级 `sample_spec(rng, rand)`；该文件已随重构删除）。
这是「模块级函数被当成方法调用」这类缺陷的典型：**语法合法，只有执行到那一行才炸**。

## 9. 已知限制与坑

* **零高度台阶无法编译**：`demo_stairs_spec(step_height=0.0, n_up=8)` 生成的 XML 含
  `size="0.250000 1.000000 0.000000"`，直接编译得到
  `ValueError: Error: size 2 must be positive in geom`
  （并附带 `Element name 'stair_up0', id 0, line 2`）；经 `go2_model.py` 的 `compile_scene`
  包装后调用方看到 `RuntimeError: MJCF compile failed; dumped to
  <...>\.generated\failed_scene.xml`。平地必须用 `flat_spec(...)`（`experience.md` §1.3）。
* **`height_at` 是阶跃的**：在 `start_x + i*tread` 与 `up_end_x + top_landing` 处跳变。
  任何把单点 `height_at(base_x)` 当参照面的量都会在跨越立面时**跳一整级**、逐帧翻转
  （`experience.md` §1.4、§3.3）。
* **`height_at` 与几何必须同步**：`step_blocks()` 用 `CREST_SEAM_OVERLAP` 扩展 crest/top 块，
  但 `height_at` **不读**这个常量 —— 正确性依赖于「两个实心块的并集不变」。若有人把这个
  常量改成 > 1（回伸过多、block 越过前一级顶面）或让 crest 高度与最后一级踏面不等，
  剖面与几何就会分叉。
* **`step_blocks()` 的下降段有零高度守门，上升段没有**：`if top <= 1e-9: continue` 只作用于
  `_down{j}`；`n_up > 0` 且 `step_height = 0` 时仍会生成非法 block（见上）。
* **`to_xml()` 返回的串不带换行前缀**，而 `build_terrain_xml()` 会加一个 `"\n"`。直接调用
  `to_xml()` 拼进场景时若忘记补换行，生成的 MJCF 会把 `<geom>` 与前一个标签挤在同一行
  （MuJoCo 仍能解析，但转储文件没法读）。
* **`build_terrain_xml()` 对平地返回空串**：调用方不能用它判断「地形是否就绪」，只能判断
  「是否生成了 geom」。
* **`StairSpec` 是可变 dataclass**（非 `frozen`），可以就地改字段；共享实例被改动会让报告里
  的 spec 与实际几何不一致（`env.py` 会把 `crest_height` 打进 `info`）。
* **`width` 必须为正**：`width <= 0` 会让 `hy = width/2 <= 0`，同样触发
  `size must be positive` 编译失败。
* **`n_down * step_height > crest_height` 时下降段不会到地面**：多出的级被 `top <= 1e-9`
  守门跳过，`height_at` 也在 `j >= n_down` 处返回 0，二者一致 —— 但这意味着「下降段级数」
  可以被静默截断，不报错。
* **随机化能力已经不存在**：`TerrainRandomization` / `sample_spec` 已删除。要恢复逐 episode
  地形随机化，只能从 HEAD 取回，或自己用 `dataclasses.replace()` 写一遍
  （`StairSpec` 的派生量都是 property，`replace()` 后自动一致）。

## 10. 示例用法

```python
from robotdog.tools.terrain import (
    StairSpec,
    build_terrain_xml,
    demo_stairs_spec,
    flat_spec,
)

# 1) 默认台阶：8 级 × 5 cm，踏面 50 cm
spec = StairSpec()
print(spec.crest_height, spec.up_end_x, spec.down_end_x)   # 0.4 5.5 6.7
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
```
