"""足底触觉测量 + 四相位状态机（参考 TactileStep, arXiv:2609.28959）。

为什么需要
----------
本仓过去是**盲走**（观测里不放任何接触/地形真值）。从 2026-10-20 起策略切换为
「逐步增加接触信息以正向评估其作用」，第一步就是把 TactileStep 的测量层与
相位条件策略落地（用户指令）。

论文要点 → 本模块的对应物
------------------------
1. 触觉特征（每足）：[F̄, p_cop, Ā] = 归一化法向合力、压力中心、接触面积。
   真机用压力鞋垫测；仿真里用「MuJoCo 接触点 + 法向力」在一个**虚拟足底 taxel
   网格**上重建压力分布（kNN 高斯扩散，抑制单点接触）。**不查地形真值**：
   与真实鞋垫同口径，只是仿真替身。
2. 四相位状态机：SWING / PRELANDING / LANDING / STANCE，按「是否接触 + 足端
   竖直速度 + 离地间隙」在线推断。三个相位条件策略的数学在 reward.py
   （tactile_prelanding / tactile_impact / tactile_stance），它们读的
   **数据面**全在这里（力/面积/CoP 的历史与相位）。
3. 后续可扩展：dual-critic（稠密/稀疏奖励分组）不在本模块；观测块开关在
   EnvConfig.obs.tactile（16 维 = 4 足 × [F̄, Ā, CoPx, CoPy]）。

为什么接触面积/CoP 由「虚拟 taxel 网格」定义
------------------------------------------
Go2 的足端是球（点接触），物理上没有面积。面积/CoP 是**支撑质量**的量度：
把每个接触点的法向力按高斯核摊到足底矩形网格上，网格里有力的格子数 × 格子面积
即「接触面积」，力的加权中心即 CoP。这样同一套定义对球足、平足、真机鞋垫都成立，
且随鞋垫尺寸/网格密度可配。**默认关闭**（TactileConfig.enabled=False）时
连一次接触遍历都不多花，逐步奖励与观测逐位不变。
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

#: 相位编号（int8，写进 info 也写进奖励门控）。
SWING, PRELANDING, LANDING, STANCE = 0, 1, 2, 3
PHASE_NAMES: tuple[str, str, str, str] = ("swing", "prelanding", "landing", "stance")


@dataclass
class TactileConfig:
    """足底触觉的几何/阈值配置。**默认全部关闭**（enabled=False）。"""

    enabled: bool = False
    # --- 虚拟足底（仿真替身；真机 = 压力鞋垫的物理尺寸） -------------------- #
    sole_length: float = 0.06           # m，足底矩形长边（前后向）
    sole_width: float = 0.04            # m，足底矩形短边（左右向）
    n_rows: int = 4                     # 长边方向的 taxel 数
    n_cols: int = 5                     # 短边方向的 taxel 数
    # --- 力/面积/CoP 的归一化与阈值 ----------------------------------------- #
    force_ref_n: float = 0.0            # N；<=0 ⇒ 用 model 总质量×g 自动算（= 1 体重）
    taxel_force_min: float = 1.0        # N，单 taxel 上超过它才计为「接触中的格子」
    spread_cells: int = 1               # kNN 扩散半径（格）；0 = 只落在最近格
    cop_margin_ref: float = 0.02        # m，CoP 到支撑边界距离的归一化尺度
    # --- 相位机 ------------------------------------------------------------- #
    prelanding_clearance: float = 0.06  # m，离地低于它且还在下降 ⇒ PRELANDING
    prelanding_vz: float = 0.05         # m/s，向下的足端速度门限（绝对值）
    impact_window_steps: int = 5        # LANDING 相位总帧数（含触地首帧；冲击窗）

    def __post_init__(self) -> None:
        if self.sole_length <= 0.0 or self.sole_width <= 0.0:
            raise ValueError("tactile.sole_length / sole_width 必须为正")
        if self.n_rows <= 0 or self.n_cols <= 0:
            raise ValueError("tactile.n_rows / n_cols 必须为正")
        if self.taxel_force_min < 0.0:
            raise ValueError("tactile.taxel_force_min 不能为负")
        if self.impact_window_steps < 1:
            raise ValueError("tactile.impact_window_steps 至少为 1")

    @property
    def n_taxels(self) -> int:
        return int(self.n_rows * self.n_cols)

    @property
    def cell_area(self) -> float:
        return (self.sole_length / self.n_rows) * (self.sole_width / self.n_cols)

    @property
    def sole_area(self) -> float:
        return self.sole_length * self.sole_width

    @property
    def actor_size(self) -> int:
        """actor 可见的触觉特征维数：4 足 × [F̄, Ā, CoPx, CoPy]。"""
        return 16


class TactileState:
    """一帧触觉读数（每足）：力、面积、CoP、相位 + 相位机需要的上一次量。

    所有数组长度 = 足数（本仓 4）。字段语义见 :class:TactileSensor。
    """

    def __init__(self, nfoot: int) -> None:
        self.nfoot = int(nfoot)
        self.force_n = np.zeros(nfoot)          # 法向合力 [N]
        self.force_tan = np.zeros(nfoot)        # 切向合力模长 [N]
        self.force_resultant = np.zeros(nfoot)  # |合力| [N]
        # 三维合力（世界系 / 足体系，每足 (3,)）—— 供状态估计器做标签。
        # 世界系用于校验（站立时 z 分量之和 ≈ +m·g）；足体系才是「触觉测到的
        # 局部三维力」，估计器的 RMSE/MAE 以它为主口径。
        self.force_vec = np.zeros((nfoot, 3))       # 世界系 [N]
        self.force_vec_foot = np.zeros((nfoot, 3))  # 足体系 [N]
        self.force_norm = np.zeros(nfoot)       # F̄ = 法向力 / 体重
        self.resultant_norm = np.zeros(nfoot)
        self.area = np.zeros(nfoot)             # 接触面积 [m²]
        self.area_norm = np.zeros(nfoot)        # Ā = A / 足底面积
        self.cop = np.zeros((nfoot, 2))         # 足体系 CoP [m]
        self.cop_norm = np.zeros((nfoot, 2))    # 除以半尺寸 ⇒ [-1, 1]
        self.margin = np.zeros(nfoot)           # CoP 到支撑边界的归一化裕度
        self.phase = np.zeros(nfoot, dtype=np.int8)
        self.vz = np.zeros(nfoot)               # 足端竖直速度 [m/s]（+上）
        self.delta_force_norm = np.zeros(nfoot)  # ΔF̄（本步 - 上一步）
        self.cop_jump = np.zeros(nfoot)         # 归一化 CoP 跳变
        self.impact_peak = np.zeros(nfoot)      # LANDING 窗内的 F̄ 峰值

    def actor_features(self) -> np.ndarray:
        """(16,) 观测块：逐足 [F̄, Ā, CoPx, CoPy]（顺序与足端索引一致）。"""
        rows = np.stack([self.force_norm, self.area_norm,
                         self.cop_norm[:, 0], self.cop_norm[:, 1]], axis=1)
        return rows.reshape(-1).astype(np.float64)

    def as_info(self) -> dict:
        return {
            "phase": [PHASE_NAMES[int(p)] for p in self.phase],
            "phase_id": [int(p) for p in self.phase],
            "force_n": [float(v) for v in self.force_n],
            "force_tan": [float(v) for v in self.force_tan],
            "force_norm": [float(v) for v in self.force_norm],
            "force_vec": [[float(a) for a in row] for row in self.force_vec],
            "force_vec_foot": [[float(a) for a in row]
                               for row in self.force_vec_foot],
            "area": [float(v) for v in self.area],
            "area_norm": [float(v) for v in self.area_norm],
            "cop": [[float(a), float(b)] for a, b in self.cop],
            "cop_norm": [[float(a), float(b)] for a, b in self.cop_norm],
            "margin": [float(v) for v in self.margin],
            "cop_jump": [float(v) for v in self.cop_jump],
        }


class TactileSensor:
    """把「MuJoCo 接触点 + 法向力」重建为足底压力分布，并推进四相位状态机。

    真机对应物：薄型压力鞋垫（每 taxel 一个压力读数）+ 同几何的 taxel 网格。
    仿真里的输入只有 data.contact（接触位置）与 mj_contactForce（法向力），
    **不读地形高度** —— 因而这套测量可以直接搬到真机（换成鞋垫读数）而不改接口。
    """

    def __init__(self, cfg: TactileConfig, model: mujoco.MjModel,
                 data: mujoco.MjData, meta) -> None:
        self.cfg = cfg
        self.model = model
        self.data = data
        self.nfoot = len(meta.foot_geom_ids)
        self._foot_geoms = [int(g) for g in meta.foot_geom_ids]
        self._foot_bodies = [int(model.geom_bodyid[g]) for g in self._foot_geoms]
        self._terrain = {int(meta.floor_geom_id)} | {
            int(g) for g in meta.stair_geom_ids}
        mass = float(np.sum(model.body_mass))
        self._force_ref = (float(cfg.force_ref_n) if float(cfg.force_ref_n) > 0.0
                           else max(mass * 9.81, 1e-6))
        # 虚拟足底网格（足体系：x 前后、y 左右，位于球心下方 foot_radius 处）。
        us = (np.arange(cfg.n_cols) + 0.5) / cfg.n_cols - 0.5
        vs = (np.arange(cfg.n_rows) + 0.5) / cfg.n_rows - 0.5
        grid = np.stack(np.meshgrid(us * cfg.sole_length, vs * cfg.sole_width,
                                    indexing="xy"), axis=-1).reshape(-1, 2)
        self._grid = grid
        self._cell_w = cfg.sole_length / cfg.n_rows
        self._cell_h = cfg.sole_width / cfg.n_cols
        # kNN 高斯扩散核（行归一化）：一个接触点把力摊到邻近 taxel。
        d = np.linalg.norm(grid[:, None, :] - grid[None, :, :], axis=-1)
        sigma = 0.7 * max(self._cell_w, self._cell_h)
        radius = float(cfg.spread_cells) * max(self._cell_w, self._cell_h) + 1e-9
        w = np.exp(-(d ** 2) / (2.0 * sigma ** 2)) * (d <= radius)
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
        self._weights = w
        self.state = TactileState(self.nfoot)
        self._prev_contact = np.zeros(self.nfoot, dtype=bool)
        self._prev_force_norm = np.zeros(self.nfoot)
        self._prev_cop_norm = np.zeros((self.nfoot, 2))
        self._landing_left = np.zeros(self.nfoot, dtype=np.int64)
        self._res = np.zeros(6, dtype=np.float64)

    # ------------------------------------------------------------------ #
    def reset(self) -> TactileState:
        self.state = TactileState(self.nfoot)
        self._prev_contact[:] = False
        self._prev_force_norm[:] = 0.0
        self._prev_cop_norm[:] = 0.0
        self._landing_left[:] = 0
        return self.state

    def update(self, foot_contact: np.ndarray, foot_pos: np.ndarray,
               foot_clearance: np.ndarray, prev_foot_pos: np.ndarray,
               control_dt: float) -> TactileState:
        cfg = self.cfg
        st = self.state
        fn = np.zeros(self.nfoot)
        ft = np.zeros(self.nfoot)
        vec = np.zeros((self.nfoot, 3))
        taxel_force = np.zeros((self.nfoot, cfg.n_taxels))
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self._terrain and g2 not in self._terrain:
                other, foot_is_g2 = g2, True
            elif g2 in self._terrain and g1 not in self._terrain:
                other, foot_is_g2 = g1, False
            else:
                continue
            try:
                k = self._foot_geoms.index(other)
            except ValueError:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, self._res)
            f_normal = abs(float(self._res[0]))
            fn[k] += f_normal
            ft[k] += float(np.hypot(self._res[1], self._res[2]))
            # 三维力向量：frame 的列是接触轴在世界系的方向 ⇒ frame @ f 是接触力
            # 在世界系的表达；符号约定见下面一段（mj_contactForce 作用于 geom1）。
            frame = np.asarray(c.frame, dtype=np.float64).reshape(3, 3)
            f_world = frame @ np.array([self._res[0], self._res[1],
                                        self._res[2]], dtype=np.float64)
            # MuJoCo 的 mj_contactForce 是**作用在 geom1 上**的力（实测：站立时
            # frame @ f 的 z 分量为 -m·g，翻号后才是地面反力）⇒ 足是 geom1 时
            # 直接取 frame @ f，足是 geom2 时取反。
            vec[k] += f_world if not foot_is_g2 else -f_world
            # 接触点 → 足体系 → 最近的 taxel；法向力按扩散核摊过去。
            body = self._foot_bodies[k]
            rot = self.data.xmat[body].reshape(3, 3)
            local = rot.T @ (np.asarray(c.pos, dtype=np.float64)
                             - self.data.xpos[body])
            idx = int(np.argmin(np.linalg.norm(self._grid - local[:2], axis=1)))
            taxel_force[k] += f_normal * self._weights[idx]

        total_mass_force = self._force_ref
        for k in range(self.nfoot):
            total = float(taxel_force[k].sum())
            st.force_n[k] = fn[k]
            st.force_tan[k] = ft[k]
            st.force_resultant[k] = float(np.hypot(fn[k], ft[k]))
            st.force_norm[k] = fn[k] / total_mass_force
            st.resultant_norm[k] = st.force_resultant[k] / total_mass_force
            st.force_vec[k] = vec[k]
            rot_k = self.data.xmat[self._foot_bodies[k]].reshape(3, 3)
            st.force_vec_foot[k] = rot_k.T @ vec[k]
            if total > 1e-9:
                active = taxel_force[k] >= float(cfg.taxel_force_min)
                st.area[k] = float(np.count_nonzero(active)) * cfg.cell_area
                cop = (self._grid * taxel_force[k][:, None]).sum(axis=0) / total
                st.cop[k] = cop
                st.cop_norm[k] = np.clip(
                    cop / np.array([cfg.sole_length / 2.0, cfg.sole_width / 2.0]),
                    -1.0, 1.0)
                st.margin[k] = float(np.clip(min(
                    1.0 - abs(st.cop_norm[k, 0]), 1.0 - abs(st.cop_norm[k, 1])),
                    0.0, 1.0))
            else:
                st.area[k] = 0.0
                st.cop[k] = 0.0
                st.cop_norm[k] = 0.0
                st.margin[k] = 0.0
            st.area_norm[k] = st.area[k] / cfg.sole_area

        # --- 四相位状态机（TactileStep §Method: 由接触与足端竖直速度推断） ---
        dt = max(float(control_dt), 1e-9)
        vz = (np.asarray(foot_pos[:, 2]) - np.asarray(prev_foot_pos[:, 2])) / dt
        st.vz = np.asarray(vz, dtype=np.float64)
        st.delta_force_norm = st.force_norm - self._prev_force_norm
        st.cop_jump = np.where(
            self._prev_contact & foot_contact,
            np.linalg.norm(st.cop_norm - self._prev_cop_norm, axis=1), 0.0)
        for k in range(self.nfoot):
            contact = bool(foot_contact[k])
            if contact:
                if not self._prev_contact[k]:
                    st.phase[k] = LANDING
                    # 触地首帧也算在窗口里：窗口总帧数 = impact_window_steps
                    # （旧写法置 window 个剩余帧 ⇒ 实际出现 window+1 帧，见
                    # `.tmp/tactile_report.py` 的实测：5 步配置跑出 6 帧 LANDING）
                    self._landing_left[k] = max(0, int(cfg.impact_window_steps) - 1)
                    st.impact_peak[k] = st.force_norm[k]
                elif self._landing_left[k] > 0:
                    st.phase[k] = LANDING
                    self._landing_left[k] -= 1
                    st.impact_peak[k] = max(st.impact_peak[k], st.force_norm[k])
                else:
                    st.phase[k] = STANCE
                    st.impact_peak[k] = 0.0
            else:
                self._landing_left[k] = 0
                st.impact_peak[k] = 0.0
                near = float(foot_clearance[k]) < float(cfg.prelanding_clearance)
                descending = float(vz[k]) < -float(cfg.prelanding_vz)
                st.phase[k] = PRELANDING if (near and descending) else SWING
        self._prev_contact = np.asarray(foot_contact, dtype=bool).copy()
        self._prev_force_norm = st.force_norm.copy()
        self._prev_cop_norm = st.cop_norm.copy()
        return st
