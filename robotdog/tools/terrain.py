"""MuJoCo 参数化楼梯地形，由实心 box 几何体拼成。

设计说明
--------
* 每一级都是**落在地面上的实心块**（不是薄板）：半尺寸
  ``(tread/2, width/2, top_z/2)``，中心在 ``z = top_z/2``。实心块不会从下方
  被挂住，接触几何也简单。
* 相邻方块之间的静态-静态重叠在 MuJoCo 里无害 —— 只有一对几何体中至少
  有一个是动态体时才会生成接触。
* 刻意保留 box 的锐利棱边：机器人必须学会越过一道锋利的立面，而
  heightfield 会把棱边在一个网格单元内磨圆。

布局（前进方向为 +x）
---------------------
``up``      : 平地 -> n_up 级上升立面 -> 平的顶层平台。
``pyramid`` : 平地 -> n_up 级上升立面 -> 平的 crest -> n_down 级下降立面 -> 地面。
              这样直接同时提供上楼与下楼，不需要挖坑。

解析高度剖面通过 :meth:`StairSpec.height_at` 暴露出来，使奖励项、出生点摆放
与评估指标都与几何保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_STAIR_FRICTION: tuple[float, float, float] = (0.9, 0.005, 0.0001)
DEFAULT_STAIR_RGBA = "0.62 0.60 0.55 1"
#: 为地形（楼梯**以及**地面）保留的 MuJoCo geom group。
#:
#: 必须落在 MuJoCo 渲染默认可见的组里（默认隐藏 3–5；Go2 的 MJCF 占用 2/3）——
#: 选错组会得到「物理在爬楼梯、画面里狗在爬空气」的世界（experience.md §1.2）。
#: :meth:`robotdog.env.Go2StairsEnv.render` 还会强制打开全部六个组。
TERRAIN_GEOM_GROUP = 1

#: 平的 crest / 顶层平台*向回*伸进最后一级上升台阶的长度（单位：踏面）。
#:
#: 必须 > 0：crest 与最后一级踏面高度相同、共面，仅相邻时贴合端面是一条零高度
#: 的缝，脚球落上去会被两块 box 沿最小穿透轴水平对推、z 向无支撑而夹死
#: （experience.md §1.1）。回伸让共面对消失，实心块并集不变。
CREST_SEAM_OVERLAP = 1.0


@dataclass
class StairSpec:
    """一段楼梯的几何。

    高度遵循用户的课程：先用 5 cm 立面配 50 cm 踏面，课程学习中再依次用
    3-8 cm、8-15 cm 与 15-22 cm 的高度带。
    """

    step_height: float = 0.05  # 立面高度 [m]（台阶高度）
    tread: float = 0.50  # 沿 +x 的踏面深度 [m]（台阶宽度）
    width: float = 2.00  # 沿 y 的横向范围 [m]
    n_up: int = 8  # 上升立面的级数
    n_down: int = 0  # 为 0 时只上行；>0 则构成金字塔
    start_x: float = 1.50  # 第一道立面开始处的世界坐标 x
    landing: float = 1.20  # 上升段与下降段之间平 crest 的长度 [m]
    # n_down == 0 时的平顶层平台 [m]。它必须 > 0，否则地形根本无法成功：
    # ``StairSpec.height_at`` 在 ``up_end_x + top_landing`` 之后返回 0，而成功
    # 判据要求 base 越过 ``up_end_x`` *并且*处在 crest 高度上。用旧的默认值 0.0
    # 时，一个裸的 ``StairSpec()`` 在 crest 处就是 0.40 m 悬崖，所以
    # ``Go2StairsEnv()`` 永远报告不出成功。
    top_landing: float = 1.20
    center_y: float = 0.0
    friction: tuple[float, float, float] = DEFAULT_STAIR_FRICTION
    rgba: str = DEFAULT_STAIR_RGBA
    name_prefix: str = "stair"

    def __post_init__(self) -> None:
        """拒绝会生成非法 MJCF 或算术上无意义的参数（零高度台阶 → MuJoCo
        编译期错、tread ≤ 0 → height_at 除零、n_down > 0 且 landing ≤ 0 →
        下降段与上升段重叠）。

        注意 ``flat_spec()`` 走 ``n_up = n_down = 0``，平地不受这些校验影响。
        """
        if self.n_up > 0 and self.step_height <= 0.0:
            raise ValueError(
                f"step_height 必须 > 0（n_up={self.n_up}）：零高度的台阶会生成"
                f"零高度的 box，MuJoCo 编译期会以 size 2 must be positive 失败。"
                f"只要平地请用 flat_spec()。"
            )
        if self.n_down > 0 and self.step_height <= 0.0:
            raise ValueError(
                f"step_height 必须 > 0（n_down={self.n_down}）：同上。"
            )
        if self.tread <= 0.0:
            raise ValueError(f"tread 必须 > 0，实际 {self.tread}。")
        if self.width <= 0.0:
            raise ValueError(f"width 必须 > 0，实际 {self.width}。")
        if self.n_down > 0 and self.landing <= 0.0:
            raise ValueError(
                f"n_down > 0 时 landing 必须 > 0，实际 {self.landing}："
                f"否则下降段会与上升段重叠。"
            )

    # ---- 派生几何 -------------------------------------------------------------- #

    @property
    def crest_height(self) -> float:
        return self.n_up * self.step_height

    @property
    def up_end_x(self) -> float:
        """最后一道立面的踏面结束处的 x（= crest 起点）。"""
        return self.start_x + self.n_up * self.tread

    @property
    def down_end_x(self) -> float:
        """下降段回到地面高度处的 x。"""
        return self.up_end_x + self.landing + self.n_down * self.tread

    def step_blocks(self) -> list[tuple[str, tuple[float, float, float],
                                       tuple[float, float, float]]]:
        """返回每一个方块的 ``[(name, (cx, cy, cz), (hx, hy, hz)), ...]``。

        整条楼梯按从前到后的顺序给出，使射线投射与 :meth:`StairSpec.height_at`
        中的解析剖面严格对齐。
        """
        blocks: list[tuple[str, tuple[float, float, float],
                           tuple[float, float, float]]] = []
        hx, hy = self.tread / 2.0, self.width / 2.0

        # 上升段：第 i 个方块在 x 上覆盖 [start_x + i*tread, +tread]，
        # 顶面在 z = (i+1) * step_height。
        for i in range(self.n_up):
            top = (i + 1) * self.step_height
            cx = self.start_x + (i + 0.5) * self.tread
            blocks.append(
                (f"{self.name_prefix}_up{i}",
                 (cx, self.center_y, top / 2.0), (hx, hy, top / 2.0))
            )

        # 两段之间的平 crest。``ov`` 向回伸进最后一级上升台阶，避开「共面接缝
        # 夹脚」陷阱（见 CREST_SEAM_OVERLAP 的 docstring）。
        ov = CREST_SEAM_OVERLAP * self.tread
        if self.n_down > 0:
            top = self.crest_height
            cx = self.up_end_x + (self.landing - ov) / 2.0
            blocks.append(
                (f"{self.name_prefix}_crest",
                 (cx, self.center_y, top / 2.0),
                 ((self.landing + ov) / 2.0, hy, top / 2.0))
            )

            # 下降段：第 j 个方块的顶面比前一块低一道立面，
            # 最终与地面齐平。
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
            # 只上行的楼梯：用一块平平台收尾，这样机器人越过最后一道立面后
            # 不会走下 40 cm 悬崖。这里要求 ``n_up > 0``：平地段 crest 高度为 0，
            # 「顶层平台」会退化成一块零高度的方块。
            top = self.crest_height
            cx = self.up_end_x + (self.top_landing - ov) / 2.0
            blocks.append(
                (f"{self.name_prefix}_top",
                 (cx, self.center_y, top / 2.0),
                 ((self.top_landing + ov) / 2.0, hy, top / 2.0))
            )
        return blocks

    def to_xml(self, indent: str = "    ") -> str:
        """整条楼梯的 MJCF ``<geom>`` 元素。

        每个方块都打上 :data:`TERRAIN_GEOM_GROUP`。正是这一点让
        :mod:`robotdog.tools.touch` 里的前向测距能投射一条看不见机器人自己腿的
        射线 —— 机器人的 geom 位于 group 2 和 3。geom group 对物理没有影响，但它
        *确实*决定方块是否被渲染，所以改动前先看那个常量的 docstring。

        ⭐ ``solref``/``solimp``（2026-10-13）：默认软接触（时间常数 20ms）在
        15kg 全身重量经单足压在踏面/立面上时允许 **厘米级持续下陷**（实测
        12~33mm，足端几何上陷进 box 内"无法抬起"—— experience.md §23.15）。
        收紧为 5ms + 高阻抗，穿透降到毫米级。
        """
        out: list[str] = []
        fr = " ".join(f"{v:g}" for v in self.friction)
        for name, (cx, cy, cz), (hx, hy, hz) in self.step_blocks():
            out.append(
                f'{indent}<geom name="{name}" type="box" '
                f'pos="{cx:.6f} {cy:.6f} {cz:.6f}" '
                f'size="{hx:.6f} {hy:.6f} {hz:.6f}" '
                f'condim="3" friction="{fr}" rgba="{self.rgba}" '
                f'group="{TERRAIN_GEOM_GROUP}" '
                f'solref="0.005 1" solimp="0.95 0.99 0.001"/>'
            )
        return "\n".join(out)

    def height_at(self, x: float, y: float = 0.0) -> float:
        """世界坐标 ``(x, y)`` 处的解析地形高度 [m]（楼梯之外为 0）。

        ``y`` 只给有二维高度场的子类用（``robotdog/tools/rough.py``）；普通楼梯
        的剖面只依赖 ``x``，传什么都一样 —— 参数放在这里是为了统一接口。
        """
        if x < self.start_x:
            return 0.0
        if x < self.up_end_x:
            i = int((x - self.start_x) // self.tread)
            return (i + 1) * self.step_height
        if self.n_down == 0:
            # 只上行：可选的顶层平台范围内取 crest 高度，之后是裸地面
            # （几何确实在那里结束）。
            return self.crest_height if x < self.up_end_x + self.top_landing else 0.0
        if x < self.up_end_x + self.landing:
            return self.crest_height
        j = int((x - self.up_end_x - self.landing) // self.tread)
        if j >= self.n_down:
            return 0.0
        return max(0.0, self.crest_height - (j + 1) * self.step_height)


# --------------------------------------------------------------------------- #
# 便捷构造器：与课程使用的几种地形一一对应
# --------------------------------------------------------------------------- #

#: 只有平地 —— 课程的第一级。
def flat_spec(**kw) -> StairSpec:
    """平地。

    三个字段必须一起归零：``n_up = n_down = 0`` 让几何一个方块都不生成；
    ``step_height = 0`` 让观测里的台阶高读数真的是 0（否则会留 ``StairSpec``
    的默认值，观测里凭空多出一个台阶）。
    """
    kw.setdefault("n_up", 0)
    kw.setdefault("n_down", 0)
    kw.setdefault("step_height", 0.0)
    return StairSpec(**kw)


#: 默认 5 cm 立面 / 50 cm 踏面 —— 课程每一级台阶都用这个形状。
def demo_stairs_spec(
    step_height: float = 0.05,
    tread: float = 0.50,
    n_up: int = 8,
    n_down: int = 0,
    **kw,
) -> StairSpec:
    """参数显式列出，便于调用方覆盖每一个字段。"""
    kw.setdefault("top_landing", 1.20 if n_down == 0 else 0.0)
    return StairSpec(
        step_height=step_height,
        tread=tread,
        n_up=n_up,
        n_down=n_down,
        **kw,
    )


def build_terrain_xml(spec: StairSpec) -> str:
    """要注入 ``<worldbody>`` 的 MJCF 片段。"""
    # 判据用「有没有几何块」而不是「有没有台阶」：粗糙地面（n_up=n_down=0）
    # 也走这条路注入方块（robotdog/tools/rough.py）。
    if not spec.step_blocks():
        return ""
    return "\n" + spec.to_xml()
