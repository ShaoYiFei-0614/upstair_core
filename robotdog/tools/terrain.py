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
#: 把全部地形几何放进一个专用 group，前向探针（``robotdog/env.py`` 的楼梯测距）
#: 就能「向前看」而看不到机器人自己的前腿：``mj_ray`` 接受一个 *geom group mask*，
#: 只开这一组的掩码构造性地让探针看不见机器人。
#:
#: **取 Group 1，而且这个值承担双重载荷：**
#:
#: 1. Unitree 的 MJCF 把 ``group="2"`` 分给纯视觉 geom，把 ``group="3"`` 分给
#:    每一个碰撞 geom（躯干盒、膝部圆柱以及四个脚球），所以 2 和 3 已被占用；
#:    没有显式 group 的 geom 落在 group 0。
#: 2. MuJoCo 的 ``mjv_defaultOption`` 出厂就是 ``geomgroup = [1, 1, 1, 0, 0, 0]``
#:    —— **group 3、4、5 默认被隐藏。** 于是把楼梯打在 group 4 上，它在每一个
#:    离屏渲染里都不可见，而物理完全不受影响：画面上是狗在爬空气，HUD 却愉快地
#:    报告高度增益。（真的发生过；这就是这个常量被钉死在 1 的由来。）
#:    group 1 落在默认可见范围内，所以楼梯不用改渲染器就能显示。
#:
#: :meth:`robotdog.env.Go2StairsEnv.render` 还会额外强制打开*全部六个* group，
#: 所以改这个常量不会再悄悄把地形藏起来。
TERRAIN_GEOM_GROUP = 1

#: 平的 crest / 顶层平台 *向回*伸进最后一级上升台阶的长度（单位：踏面）。
#: 它必须 > 0，原因是 MuJoCo 的一个真实陷阱：
#:
#: crest 与最后一级踏面高度完全相同，所以两块 box 共面。当它们只是*相邻*
#: （crest 恰好从 ``up_end_x`` 开始）时，彼此贴合的端面是一条零高度的缝。
#: 球心落在这条缝上的脚球，对两个竖直面都是**零**穿透，但对两个顶面还有
#: 几十毫米距离，而 MuJoCo 沿*最小穿透轴*求解「球在盒内」的接触 —— 于是两块
#: box 都在水平方向推这只脚，方向相反。两个推力相互抵消，z 方向没有任何东西
#: 支撑脚，脚就被夹住：x 被钉死，下沉到踏面以下 30-40 mm，而且因为接触法线
#: 是水平的，``touch.riser_graze`` 每一步都被点亮。在 8 cm x 8 级楼梯上实测
#: （``.tmp/stall_probe.py``）：两只前脚在剩余 700 个控制步里冻结在
#: ``x = 5.500 == up_end_x``，也就是说机器人爬完整段楼梯后，直到 20 s 上限
#: 都停在终点线前 0.114 m —— 这正是高台阶训练长期卡在 20-40 % 的原因。
#:
#: 让 crest 向最后一级台阶内部回伸，会把它的后侧面埋进那一级内部，于是共面的
#: 那对表面消失，只剩真正的立面。实心块的*并集*不变，所以解析剖面、向下射线
#: 投射与每一个渲染像素都完全一致。
CREST_SEAM_OVERLAP = 1.0  # 单位：踏面


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
        """拒绝会生成非法 MJCF 或算术上无意义的参数。

        这些组合过去都不会在这里报错，而是各自以更晚、更难懂的方式失败：

        * ``n_up > 0`` 且 ``step_height <= 0`` —— 生成 ``n_up`` 个**零高度**的
          box，MuJoCo 在编译期抛 ``ValueError: size 2 must be positive in
          geom``（错误信息里只有 geom 名字，看不到是 ``step_height`` 没给）。
        * ``tread <= 0`` —— :meth:`height_at` 的 ``// self.tread`` 直接
          ``ZeroDivisionError: float floor division by zero``。
        * ``n_down > 0`` 且 ``landing <= 0`` —— 下降段与上升段重叠。

        注意 ``flat_spec()`` 走的正是 ``n_up = n_down = 0``（它把两个级数设为
        0 而不是把 ``step_height`` 设为 0），所以平地段不受这些校验影响。
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

        # 两段之间的平 crest。``ov`` 向回伸进最后一级上升台阶，使两块共面的
        # box 相互重叠而不是仅仅相邻 —— 它避开的那种「脚被夹死」的陷阱见 CREST_SEAM_OVERLAP。
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
        """世界坐标 x 处的解析地形高度 [m]（楼梯之外为 0）。"""
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

    三个字段必须一起归零：``n_up = n_down = 0`` 让 :meth:`StairSpec.step_blocks`
    一个方块都不生成；``step_height = 0`` 则让地形观测里的台阶高读数真的是 0
    （否则会留下 ``StairSpec`` 的 5 cm 默认值，观测里凭空多出一个台阶）。

    ``top_landing`` 归零是可选的优化：``n_up = 0`` 时 :meth:`step_blocks` 的
    收尾分支本来就被 :meth:`StairSpec.__post_init__` 之外的 ``n_up > 0`` 判据挡住。
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
    if spec.n_up == 0 and spec.n_down == 0:
        return ""
    return "\n" + spec.to_xml()
