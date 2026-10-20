"""随机粗糙地面（2–5 cm 起伏）：由一组实心方块拼成的离散高度场。

为什么不用 MuJoCo ``hfield``
---------------------------
``hfield`` 需要在 ``data`` 侧写入高程（MJCF 只能引用外部文件），而且会把台阶棱边
在一个网格单元内磨圆；本仓楼梯教学明确要求**锐利棱边**（terrain.py 的设计说明）。
这里用「一格一块实心 box」拼出离散粗糙面：

* 解析高度 ``height_at(x, y)`` 与几何**逐格一致**（同一份 ``heights`` 数组），
  所以出生点、摔倒参照面、足端离地都能按真值算；
* 方块名字以 ``stair_`` 开头 ⇒ ``robotdog/go2_model.py::compile_scene`` 会自动把它们
  并入地形几何集合（``meta.stair_geom_ids``），奖励/接触判定/触觉测量不需要额外接线；
* 高度场在 x 两端做线性渐入（taper），避免出生点或离开粗糙区时一堵 5 cm 垂直墙。

幅度 2–5 cm 的取值、格子 0.12 m、走廊半宽 0.8 m 都是**数据采集用**的默认值，
不是训练课程的一部分；改参数直接传 ``rough_spec(...)``。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotdog.tools.terrain import CREST_SEAM_OVERLAP, StairSpec, TERRAIN_GEOM_GROUP


@dataclass
class RoughSpec(StairSpec):
    """随机粗糙地面：``x`` 方向的走廊 + 逐格随机高度（2–5 cm）。

    继承 :class:`StairSpec` 是为了复用 ``to_xml`` / ``build_terrain_xml`` 以及环境里
    「地形是一个有 ``height_at(x, y)`` 的对象」这一约定；``n_up = n_down = 0``、
    ``step_height = 0`` 让课程/步态/成功判据都走平地分支，只有几何与高度剖面不同。
    """

    amp_lo: float = 0.02            # m，格点高度下界
    amp_hi: float = 0.05            # m，格点高度上界
    cell: float = 0.12              # m，格子边长（x/y 相同）
    rough_x0: float = 2.20          # m，粗糙区起点（之前是平地，供出生/加速）
    rough_x1: float = 5.60          # m，粗糙区终点
    corridor_half_width: float = 0.80  # m，走廊半宽（|y| 之外为平地）
    rough_seed: int = 0
    taper_cells: int = 3            # x 两端渐入的格子数
    smooth_passes: int = 1          # 随机场的邻域平滑次数（0 = 独立噪声）
    name_prefix: str = "stair_rough"

    def __post_init__(self) -> None:
        if not (0.0 < self.amp_lo <= self.amp_hi):
            raise ValueError(
                f"需要 0 < amp_lo <= amp_hi，实际 {self.amp_lo}, {self.amp_hi}")
        if self.cell <= 0.0:
            raise ValueError(f"cell 必须 > 0，实际 {self.cell}")
        if not (self.rough_x0 < self.rough_x1):
            raise ValueError(
                f"需要 rough_x0 < rough_x1，实际 {self.rough_x0}, {self.rough_x1}")
        if int(self.taper_cells) < 1:
            raise ValueError(f"taper_cells 必须 >= 1，实际 {self.taper_cells}")
        if int(self.smooth_passes) < 0:
            raise ValueError(f"smooth_passes 必须 >= 0，实际 {self.smooth_passes}")
        if self.corridor_half_width <= 0.0:
            raise ValueError(f"corridor_half_width 必须 > 0，实际 {self.corridor_half_width}")
        # 让粗糙区长度是 cell 的整数倍：否则最后不满一格的高度与解析剖面
        # 会互相矛盾（几何只铺到最后一整格，height_at 却可能返回半格的高度）。
        n_x = max(2, int(round((self.rough_x1 - self.rough_x0) / self.cell)))
        self.rough_x1 = self.rough_x0 + n_x * self.cell
        n_y = max(2, int(round(2.0 * self.corridor_half_width / self.cell)))
        self.corridor_half_width = 0.5 * n_y * self.cell
        # StairSpec.__post_init__ 会对 n_up/n_down/step_height<=0 报错；本类强制平地。
        self.n_up = 0
        self.n_down = 0
        self.step_height = 0.0
        self.tread = max(float(self.tread), float(self.cell))
        self.start_x = float(self.rough_x0)   # 出生点 = start_x - spawn_back
        super().__post_init__()

    # ---- 派生量 ---------------------------------------------------------- #

    @property
    def has_height_profile(self) -> bool:
        """环境据此决定是否走 ``height_at``（平地可以走零高度的快路径）。"""
        return True

    @property
    def n_x(self) -> int:
        return int(round((self.rough_x1 - self.rough_x0) / self.cell))

    @property
    def n_y(self) -> int:
        return int(round(2.0 * self.corridor_half_width / self.cell))

    def heights(self) -> np.ndarray:
        """(n_x, n_y) 的高程数组，缓存不变（同一 spec 多次调用逐位相同）。"""
        cached = getattr(self, "_heights_cache", None)
        if cached is not None:
            return cached
        rng = np.random.default_rng(int(self.rough_seed))
        h = rng.uniform(self.amp_lo, self.amp_hi, size=(self.n_x, self.n_y))
        for _ in range(int(self.smooth_passes)):
            padded = np.pad(h, 1, mode="edge")
            h = 0.25 * (padded[:-2, 1:-1] + padded[2:, 1:-1]
                        + padded[1:-1, :-2] + padded[1:-1, 2:])
            h = np.clip(h, self.amp_lo, self.amp_hi)
        # x 两端线性 taper：**首/末格高度严格为 0**，taper 步后到全幅 ——
        # 出口处与区域外（高度 0）连续，不会留下 ≤amp/3 的垂直落差。
        taper = max(1, int(self.taper_cells))
        taper = min(taper, max(1, self.n_x // 2))
        ramp = np.arange(taper, dtype=np.float64) / float(taper)   # 0, 1/t, …
        h[:, :taper] *= ramp[None, :]
        h[:, -taper:] *= ramp[::-1][None, :]
        self._heights_cache = h
        return h

    # ---- 几何 / 高度 ----------------------------------------------------- #

    def step_blocks(self):
        """逐格一个实心方块（顶面 = 该格高度，底面落在地面 z=0）。"""
        h = self.heights()
        blocks = []
        hx = hy = self.cell / 2.0
        for i in range(self.n_x):
            cx = self.rough_x0 + (i + 0.5) * self.cell
            for j in range(self.n_y):
                hz = float(h[i, j])
                if hz <= 1e-9:
                    continue
                cy = -self.corridor_half_width + (j + 0.5) * self.cell
                blocks.append(
                    (f"{self.name_prefix}_{i}_{j}", (cx, cy, hz / 2.0),
                     (hx, hy, hz / 2.0)))
        return blocks

    def height_at(self, x: float, y: float = 0.0) -> float:
        """世界坐标 ``(x, y)`` 处的地形高度（与方块顶面逐格一致）。"""
        x = float(x)
        y = float(y)
        if x < self.rough_x0 or x >= self.rough_x1:
            return 0.0
        if abs(y) >= self.corridor_half_width:
            return 0.0
        i = min(self.n_x - 1, int((x - self.rough_x0) // self.cell))
        j = min(self.n_y - 1, int((y + self.corridor_half_width) // self.cell))
        return float(self.heights()[i, j])


def rough_spec(
    amp_lo: float = 0.02,
    amp_hi: float = 0.05,
    seed: int = 0,
    **kw,
) -> RoughSpec:
    """2–5 cm 随机粗糙地面的便捷构造器（``seed`` 决定起伏形态）。"""
    kw.setdefault("name_prefix", "stair_rough")
    return RoughSpec(amp_lo=amp_lo, amp_hi=amp_hi, rough_seed=seed,
                     n_up=0, n_down=0, step_height=0.0, **kw)
