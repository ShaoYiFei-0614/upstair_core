# 参数化地形（台阶 / 上下楼 / 障碍）

> 对应 `SKILL.md` 阶段 3。素材取自 upstair_core（Unitree Go2 盲走楼梯 · MuJoCo 3.12 + gymnasium 1.3 + SB3 2.9） 的真实源码与实测记录。
> 具体数值只作**取证**用；换机器人时按文中的「换机器人要改什么」逐条替换。

签名速查见 `02-model-building.md` §4；本节只讲地形的几何构造、耦合常量与三个易错细节。

## 1. 地形参数化（robotdog/tools/terrain.py，242 行，**不 import mujoco/numpy**）
模块常量（只有 4 个）：
```python
DEFAULT_STAIR_FRICTION = (0.9, 0.005, 0.0001);  DEFAULT_STAIR_RGBA = "0.62 0.60 0.55 1"
TERRAIN_GEOM_GROUP = 1        # 载荷常量，见 4.4a
CREST_SEAM_OVERLAP = 1.0      # 单位：踏面；载荷常量，见 4.4b
```
`StairSpec` 全字段与 API：
```python
@dataclass
class StairSpec:
    step_height: float = 0.05   # 立面高度 [m]
    tread:       float = 0.50   # 沿 +x 的踏面深度 [m]
    width:       float = 2.00   # 沿 y 横向范围 [m]
    n_up:        int   = 8      # 上升立面级数
    n_down:      int   = 0      # 0=只上行；>0 → 金字塔（上楼+下楼，不用挖坑）
    start_x:     float = 1.50   # 第一道立面的世界 x
    landing:     float = 1.20   # 上升段与下降段之间平 crest 长度 [m]
    top_landing: float = 1.20   # n_down==0 时的平顶层平台 [m]，必须 > 0
    center_y:    float = 0.0
    friction: tuple[float,float,float] = DEFAULT_STAIR_FRICTION
    rgba: str = DEFAULT_STAIR_RGBA
    name_prefix: str = "stair"
    @property crest_height -> float          # n_up * step_height
    @property up_end_x -> float              # start_x + n_up*tread  （= crest 起点）
    @property down_end_x -> float            # up_end_x + landing + n_down*tread
    def step_blocks(self) -> list[tuple[str, tuple[float,float,float], tuple[float,float,float]]]
    def to_xml(self, indent: str = "    ") -> str
    def height_at(self, x: float) -> float
def flat_spec(**kw) -> StairSpec              # kw.setdefault("n_up"/"n_down", 0)
def demo_stairs_spec(step_height=0.05, tread=0.50, n_up=8, n_down=0, **kw) -> StairSpec
    # kw.setdefault("top_landing", 1.20 if n_down == 0 else 0.0)
def build_terrain_xml(spec: StairSpec) -> str  # n_up==0 and n_down==0 → ""（平地）；否则 "\n" + spec.to_xml()
```
派生量全是 property（每次重算），**不存在「改了输入忘同步缓存」的窗口**。默认 `StairSpec()`：`crest_height=0.4`、`up_end_x=5.5`、`down_end_x=6.7`、`len(step_blocks())==9`（`stair_up0..up7` + `stair_top`）；`height_at(5.499)=0.4`、`6.69→0.4`、`6.70→0.0`。
### 1.3 几何 → XML 的生成方式
* 每一级是**落地实心块**（不是薄板）：半尺寸 `(tread/2, width/2, top_z/2)`、中心 `z = top_z/2`，侧面从 z=0 直通顶面 ⇒ 没有「脚/小腿从下方被挂住」的空腔。块与块之间大量**静-静重叠无害**（只有一对几何体至少一方是动态体时才生成接触）。
* 上升块 `i`：`top = (i+1)*step_height`；`cx = start_x + (i+0.5)*tread`；名字 `f"{name_prefix}_up{i}"`。
* 平 crest / 顶层平台块：`ov = CREST_SEAM_OVERLAP * tread`；`cx = up_end_x + (landing - ov)/2`；半 x `= (landing + ov)/2`（`top_landing` 同理）。
* 下降块 `j`：`top = crest_height - (j+1)*step_height`；`cx = up_end_x + landing + (j+0.5)*tread`；`if top <= 1e-9: continue`。
* 刻意保留**锐利 box 边**、不用 heightfield：hfield 会把棱边在一个网格单元内磨圆，等于替策略把「越过立面」这个难点抹平，并让 `height_at` 的阶跃与几何分叉。
* XML 单行格式（坐标 `:.6f` 定长便于 diff，摩擦 `:g` 去尾零）：
```python
f'{indent}<geom name="{name}" type="box" pos="{cx:.6f} {cy:.6f} {cz:.6f}" '
f'size="{hx:.6f} {hy:.6f} {hz:.6f}" condim="3" friction="{fr}" rgba="{self.rgba}" '
f'group="{TERRAIN_GEOM_GROUP}"/>'      # fr = " ".join(f"{v:g}" for v in self.friction) → "0.9 0.005 0.0001"
```
* `height_at(x)` 是**解析阶跃函数**（无插值）：`x<start_x → 0`；`x<up_end_x → (floor((x-start_x)//tread)+1)*step_height`；`n_down==0` → crest 高度直到 `up_end_x + top_landing`，之后 0；否则 `< up_end_x+landing` → crest，再按 `tread` 递减到 0。
* `build_terrain_xml` 的返回值**自带**前缀换行（返回 `"\n" + spec.to_xml()`）—— 手工拼接别重复补。
### 1.4 三个必须知道的细节
**(a) geometry group 选择：`TERRAIN_GEOM_GROUP = 1`（双重载荷）** — ① Unitree MJCF 已占用 `group=2`（纯视觉）与 `group=3`（全部碰撞），无显式 group 落在 0。② MuJoCo `mjv_defaultOption` 出厂 `geomgroup = [1,1,1,0,0,0]` —— **group 3/4/5 默认隐藏**。把楼梯打在 group 4 上，它在每个离屏渲染里都不可见而物理照常 ⇒ 画面上狗在爬空气、HUD 愉快报告高度增益（真的发生过）。地板 `default_floor_geom` 也打同一组，摩擦与台阶一致（`condim="3" friction="0.9 0.005 0.0001"`）。`env.render()` 额外强制打开全部六组，所以再改这个常量也不会静默藏起世界。
**(b) 顶层平台回伸 `CREST_SEAM_OVERLAP = 1.0`（必须 > 0）** — crest/顶层平台与最后一级踏面**等高 ⇒ 两块 box 共面**。若只是**相邻**（crest 恰好从 `up_end_x` 起），贴合端面是一条**零高度缝**：球心落在缝上的脚球对两个竖直面都是**零穿透**，对两个顶面尚有几十毫米距离；MuJoCo 对「球在盒内」沿**最小穿透轴**求解 ⇒ 两块 box 都在**水平方向**反向推脚，两推力抵消、z 向无支撑，脚被**夹死 (pin)**：x 被钉死、下沉到踏面下 30–40 mm，接触法线水平 ⇒ `touch.riser_graze` 每步点亮。实测（8 cm × 8 级，`.tmp/stall_probe.py`）：两只前脚在剩余 **700** 个控制步里冻结在 `x = 5.500 == up_end_x`，即爬完整段楼梯后**停在终点线前 0.114 m** 直到 20 s 上限 —— 这正是高台阶训练长期卡在 20–40 % 的原因。修法：crest 向最后一级踏面**回伸一个完整踏面**，后侧面埋进台阶内部，共面表面消失、只剩真立面；实心块的**并集不变** ⇒ 解析剖面、向下射线、每个渲染像素完全一致。不要改 `up_end_x`（它是成功判据与剖面共用的量，改它等于移动终点线），不要给 crest 加 z 偏移（制造伪台阶），不要动 `margin`/`solref`（全局接触参数）。
**(c) 零高度 box 非法 ⇒ 平地必须用 `flat_spec`** — `demo_stairs_spec(step_height=0.0, n_up=8)` 每级 `top = 0` ⇒ `size="0.250000 1.000000 0.000000"`，MuJoCo 编译直接失败：`ValueError: Error: size 2 must be positive in geom` + `Element name 'stair_up0', id 0, line 2`；经 `compile_scene` 包装后调用方看到 `RuntimeError: MJCF compile failed; dumped to <...>\.generated\failed_scene.xml`。正确做法：平地一律 `flat_spec(...)`（`n_up==n_down==0` ⇒ `build_terrain_xml` 返回 `""`，根本不生成地形 geom，`step_height` 取任何值都无所谓；`config.py` 的 `config_to_stages()` 就是 `flat_spec(n_up=0, n_down=0, step_height=0.0, ...)`）。**上升段没有零高度守门**（只有下降段有 `if top <= 1e-9: continue`）；不要把守门补到上升段 —— 那会把调用错误伪装成合法平地，而 `height_at` 仍算出 8 级台阶。
**(d) 附带两条** — `top_landing` 必须 > 0，否则 crest 处是 `crest_height` 的悬崖，成功判据（base 越过 `up_end_x` **且**处于 crest 高度）永远无法满足；`height_at` 是阶跃的，任何把单点 `height_at(base_x)` 当参照面的量都会在跨立面时**跳一整级**、逐帧翻转（幽灵摔倒）—— 本仓用 `terminate_height_ref="support"`（四足下方地形的最小值）而非 `"base"`。
