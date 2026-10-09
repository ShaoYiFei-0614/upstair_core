"""Gymnasium 环境：Unitree Go2 盲走上箱式楼梯。

**盲走（blind locomotion）**：本模块中没有任何触觉传感器、没有射线投射，
也没有任何接触力读取。策略只能看到本体感知——关节角/角速度、机身姿态与角速度、
上一步动作——外加一个标量，告诉它被放上去的那条楼梯有多高。它必须仅凭这些
发现迈步的节奏。

* **Action** -- 12 个关节目标角（``FL_hip .. RR_calf``）。
* **Observation** -- ``ObsConfig``：每帧 42 个本体感知量（关节位置/速度、机身
  roll/pitch/yaw、角速度、上一步动作），再加上被打开的那些观测块；最近
  ``history_len`` 帧会被堆叠。任何地方都没有接触或测距传感器数据。
* **Reward** -- ``RewardConfig``：前进速度、姿态、关节力矩、足端抬升、
  机身/腿部碰撞、爬梯进展（外加常规的成形项）。

帧约定：

* ``qpos[0:3]``  机身位置，**世界**系。
* ``qpos[3:7]``  机身姿态四元数 ``(w, x, y, z)``，世界->机身。
* ``qvel[0:3]``  机身线速度，**世界**系。
* ``qvel[3:6]``  机身角速度，**局部（机身）**系 —— 这正是真实 IMU 陀螺报告的
  值，所以直接用于观测。

控制以 ``1 / (timestep * decimation)`` Hz 运行（默认 50 Hz），而物理以 500 Hz 运行。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from robotdog.gait import GaitParams
from robotdog.go2_model import (
    SceneOptions,
    compile_scene,
    default_floor_geom,
    home_qpos,
    quat_to_mat,
    quat_to_rpy,
)
from robotdog.tools.terrain import StairSpec, build_terrain_xml

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


@dataclass
class ObsConfig:
    """观测布局。

    ``base_frame_size`` 始终存在（42 个本体感知量）。下面四个布尔块按声明顺序
    追加；``step_height`` 是盲走策略无法低成本推断的那一项，所以它是一个普通
    数字，而不是要机器人自己去估计的量。``README.md`` §3.2 列出了由此得到的
    维度。
    """

    history_len: int = 10
    base_lin_vel_body: bool = True      # 3：机身系线速度
    base_height: bool = False           # 1：机身相对其正下方地形的高度
    gait_phase: bool = False            # 2：步态时钟的 sin/cos
    step_height: bool = True            # 1：本回合的楼梯高度
    base_frame_size: int = 42

    @property
    def frame_size(self) -> int:
        extra = 0
        extra += 3 if self.base_lin_vel_body else 0
        extra += 1 if self.base_height else 0
        extra += 2 if self.gait_phase else 0
        extra += 1 if self.step_height else 0
        return self.base_frame_size + extra

    @property
    def size(self) -> int:
        return self.frame_size * self.history_len


@dataclass
class RewardConfig:
    """奖励权重。

    大致平衡正项与负项。这些数值所来自的历史调参——每个权重堵住哪种失败——见
    ``experience.md`` §2，公式见 ``README.md`` §3.3。
    """

    # --- 五个要求的项 ----------------------------------------------------- #
    forward_vel: float = 2.5            # [SPEC] 对 command_vx 的高斯跟踪
    forward_vel_sigma: float = 0.25
    orientation: float = -1.2           # [SPEC] roll^2 + pitch^2（平方和）
    torque: float = -1.5e-4             # [SPEC] sum(tau^2)（关节力矩平方和）
    foot_lift: float = 0.9              # [SPEC] 摆动期离地间隙成形
    body_collision: float = -4.0        # [SPEC] 躯干接触地形

    # --- 前进运动 ---------------------------------------------------------- #
    forward_progress: float = 1.5       # 对 vx 线性，所以 vx=0 时一分也拿不到
    forward_progress_max: float = 1.5
    forward_progress_min: float = -0.5

    # --- 摆动足抬升 -------------------------------------------------------- #
    foot_lift_target: float = 0.08      # 目标值的下限，而不是完整的目标值
    foot_lift_step_ratio: float = 1.4   # 取较大者：target = max(target, ratio * step_height)
    foot_lift_gate_speed: float = 0.25  # m/s，三维足端速度门限

    # --- 步态发现 ---------------------------------------------------------- #
    air_time: float = 0.6
    air_time_target: float = 0.30
    air_time_max: float = 0.60

    # --- 辅助成形 ---------------------------------------------------------- #
    base_height: float = -2.0
    joint_vel: float = -2.0e-5
    action_rate: float = -0.02
    joint_limit: float = -0.5
    alive: float = 0.15
    termination: float = -15.0
    success: float = 50.0

    # --- 爬升进展 ---------------------------------------------------------- #
    stair_progress: float = 3.0
    leg_collision: float = -0.25


@dataclass
class EnvConfig:
    """一个环境实例的全部可调项。"""

    timestep: float = 0.002
    decimation: int = 10                    # -> 50 Hz 控制
    episode_length_s: float = 20.0
    action_mode: str = "residual"           # 动作模式："residual" | "absolute"
    action_scale: float = 0.30              # rad，策略权限（仅残差模式）
    command_vx: float = 0.55                # m/s，前向指令
    command_vy: float = 0.0
    command_yaw_rate: float = 0.0

    # --- 开环步态之上的残差 RL --------------------------------------------- #
    # PD 目标变为 home + gait_gain * gait.radians(t) + action_scale * action
    gait_prior: bool = False
    gait: GaitParams = field(default_factory=GaitParams.default_trot)
    gait_gain: float = 1.0
    kp: tuple[float, float, float] = (30.0, 30.0, 45.0)
    kd: tuple[float, float, float] = (0.6, 0.6, 0.9)

    # --- 终止 -------------------------------------------------------------- #
    # 回合结束前躯干必须持续接触地形的连续控制步数。>1 可以对单帧的
    # 撞击消抖。
    terminate_contact_steps: int = 10
    # 「躯干是否摔倒」的高度判据以哪个面为基准：
    #   "base" -> 机身下方的地形（阶梯函数，会产生幽灵摔倒）
    #   "support" -> 四足下方的地形（真正支撑它的那个面）
    terminate_height_ref: str = "support"
    terminate_min_height: float = 0.13
    terminate_max_tilt: float = 1.0         # rad，roll 或 pitch
    terminate_on_success: bool = True

    # --- 重置噪声 ---------------------------------------------------------- #
    reset_z: float = 0.32
    reset_joint_noise: float = 0.06
    reset_xy_noise: float = 0.12
    reset_yaw_noise: float = 0.10

    # --- 成功 -------------------------------------------------------------- #
    flat_success_distance: float = 3.0      # 仅平地阶段
    spawn_back: float = 0.55                # 第一个立面前方的米数

    # --- 观测 / 奖励 ------------------------------------------------------- #
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @property
    def control_dt(self) -> float:
        return self.timestep * self.decimation


# --------------------------------------------------------------------------- #
# 环境
# --------------------------------------------------------------------------- #


class Go2StairsEnv(gym.Env):
    """一只走在参数化箱式楼梯上的 Go2。"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        stair: StairSpec | None = None,
        cfg: EnvConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.stair = stair or StairSpec()
        self._build_model()

        self.num_act = 12
        self.act_low = self.meta.joint_ranges[:, 0].astype(np.float32)
        self.act_high = self.meta.joint_ranges[:, 1].astype(np.float32)
        self.home_angles = self.meta.home_joint_angles.astype(np.float32)

        self.action_space = spaces.Box(-1.0, 1.0, (self.num_act,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (self.cfg.obs.size,), dtype=np.float32
        )

        self._kp = np.tile(self.cfg.kp, 4)
        self._kd = np.tile(self.cfg.kd, 4)
        self._max_steps = int(self.cfg.episode_length_s / self.cfg.control_dt)

        self._hist: deque[np.ndarray] = deque(maxlen=self.cfg.obs.history_len)
        self._prev_action = np.zeros(self.num_act, dtype=np.float64)
        self._last_action = np.zeros(self.num_act, dtype=np.float64)
        self._step_count = 0
        self._start_terrain_height = 0.0
        self._reached_top = False
        # 躯干 / 腿部持续接触地形（见 ``EnvConfig.terminate_contact_steps``）
        # 的连续步数。
        self._trunk_contact_streak = 0
        self._leg_contact_streak = 0
        self._success_x = self._success_target_x()
        self.nominal_pose = float(self.meta.home_qpos[2])  # 0.27 m 站立高度
        self._prev_foot_pos = self.foot_pos.copy()
        self._foot_air_time = np.zeros(len(self.meta.foot_geom_ids))
        self._prev_foot_contact = np.zeros(len(self.meta.foot_geom_ids), dtype=bool)
        # 步态先验时钟（自回合开始起的秒数）。
        self._gait_t = 0.0
        self._gait_phase = 0.0
        self._rng = np.random.default_rng(seed)
        self._renderer: mujoco.Renderer | None = None
        self._render_cam = None
        self._scene_option = None
        self._render_size = None

    # ------------------------------------------------------------------- 构建

    def _build_model(self) -> None:
        opts = SceneOptions(timestep=self.cfg.timestep)
        self.model, self.meta, self.xml = compile_scene(
            build_terrain_xml(self.stair),
            opts=opts,
            floor_geom=default_floor_geom(),
        )
        self.data = mujoco.MjData(self.model)
        # ``mujoco.Renderer`` 绑定到某一个 ``MjModel`` 实例，所以它绝不能比
        # 构建它的那个模型活得更久 —— 否则 ``set_stair()`` 会一直画*上一块*
        # 地形。
        if getattr(self, "_renderer", None) is not None:
            self._renderer.close()
            self._renderer = None
            self._render_size = None
            self._scene_option = None

    def set_stair(self, stair: StairSpec) -> None:
        """替换地形（课程学习会用到）。会重新编译模型。

        调用方之后必须调用 ``reset()``：重新编译会替换底层的
        ``MjModel``/``MjData``，所以旧状态没有意义。
        """
        self.stair = stair
        self._build_model()
        self._success_x = self._success_target_x()
        self._reached_top = False

    # ------------------------------------------------------------------- 成功

    def _success_target_x(self) -> float:
        """本次回合要被算作成功，机身必须越过的 x。

        楼梯：最后一级立面的远边缘（还必须达到顶部高度，见
        ``_success_reached``）。平地：一段固定的前进距离，因为平地阶段没有
        立面要跨过去。
        """
        if self.stair.n_up > 0:
            return float(self.stair.up_end_x)
        return float(self.stair.start_x - self.cfg.spawn_back
                     + self.cfg.flat_success_distance)

    def _success_reached(self) -> bool:
        if self.stair.n_up > 0:
            return bool(
                self.base_pos[0] > self.stair.up_end_x
                and self.terrain_height_under_base >= self.stair.crest_height - 1e-9
            )
        return bool(self.base_pos[0] > self._success_x)

    # ------------------------------------------------------------------- 属性

    @property
    def base_pos(self) -> np.ndarray:
        return self.data.qpos[0:3]

    @property
    def base_rpy(self) -> np.ndarray:
        return quat_to_rpy(self.data.qpos[3:7])

    @property
    def base_quat(self) -> np.ndarray:
        return self.data.qpos[3:7]

    @property
    def base_lin_vel_world(self) -> np.ndarray:
        return self.data.qvel[0:3]

    @property
    def base_lin_vel_body(self) -> np.ndarray:
        """机体系下的线速度，即 ``R(q)^T @ qvel[0:3]``。

        与 :attr:`base_lin_vel_world` 不同，它不会随机器人偏航而改变含义，
        所以「前进」永远是分量 ``[0]``。
        """
        rot = quat_to_mat(self.base_quat)
        return rot.T @ self.data.qvel[0:3]

    @property
    def base_ang_vel_body(self) -> np.ndarray:
        """陀螺读数。qvel[3:6] 已经在机体系下。"""
        return self.data.qvel[3:6]

    @property
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.meta.qpos_joint_adr]

    @property
    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.meta.qvel_joint_adr]

    @property
    def foot_pos(self) -> np.ndarray:
        """(4, 3) 世界系下四个足端球心的位置。"""
        return self.data.geom_xpos[self.meta.foot_geom_ids]

    @property
    def terrain_height_under_base(self) -> float:
        return self.stair.height_at(float(self.base_pos[0]))

    @property
    def yaw(self) -> float:
        return float(self.base_rpy[2])

    # ------------------------------------------------------------------ 辅助

    @property
    def foot_clearance(self) -> np.ndarray:
        """每只足端*表面*与其下方地形之间的间隙（m）。

        ``geom_xpos`` 是足端球的球心，所以在与地形高度比较之前必须减去半径。
        """
        feet = self.foot_pos
        return feet[:, 2] - self.meta.foot_radius - self._terrain_height(feet[:, 0])

    def _terrain_height(self, x: np.ndarray | float) -> np.ndarray:
        """一个或多个世界系 x 坐标处的地形高度。"""
        arr = np.atleast_1d(np.asarray(x, dtype=float))
        if self.stair.n_up == 0 and self.stair.n_down == 0:
            return np.zeros_like(arr)
        return np.array([self.stair.height_at(float(xi)) for xi in arr])

    def _obs_frame(self) -> np.ndarray:
        obs_cfg = self.cfg.obs
        parts = [
            self.joint_pos,
            self.joint_vel,
            self.base_rpy,
            self.base_ang_vel_body,
            self._last_action,
        ]
        if obs_cfg.base_lin_vel_body:
            parts.append(self.base_lin_vel_body)
        if obs_cfg.base_height:
            parts.append(np.array([self.base_pos[2] - self._height_reference()]))
        if obs_cfg.gait_phase:
            theta = 2.0 * np.pi * self.cfg.gait.freq * self._gait_phase
            parts.append(np.array([np.sin(theta), np.cos(theta)]))
        if obs_cfg.step_height:
            # 楼梯信息里唯一直接告诉策略的一项。盲走仍然需要知道自己在*哪一段*
            # 楼梯上；但它不会知道下一级立面在哪里。
            parts.append(np.array([float(self.stair.step_height)]))
        return np.concatenate(parts).astype(np.float64)

    def _observation(self) -> np.ndarray:
        return np.concatenate(list(self._hist)).astype(np.float32)

    # ------------------------------------------------------------------- 奖励

    def _compute_reward(self, action: np.ndarray, trunk_contact: bool,
                        foot_contact: np.ndarray | None = None,
                        leg_contact: bool = False
                        ) -> tuple[float, dict[str, float]]:
        rc = self.cfg.reward
        rpy = self.base_rpy
        lin = self.base_lin_vel_world
        target_height = self.nominal_pose

        terms: dict[str, float] = {}

        # 1) 前进速度奖励 -- 前向速度跟踪 ------------------------------------
        err = lin[0] - self.cfg.command_vx
        terms["forward_vel"] = (
            rc.forward_vel * float(np.exp(-(err * err) / rc.forward_vel_sigma))
        )
        # 线性伴随项：机器人不动时恰好为 0，所以「原地站死」的策略一分也
        # 拿不到。
        terms["forward_progress"] = rc.forward_progress * float(
            np.clip(lin[0], rc.forward_progress_min, rc.forward_progress_max)
        )

        # 2) 姿态稳定奖励 -- 惩罚 roll 与 pitch -----------------------------
        terms["orientation"] = rc.orientation * float(rpy[0] ** 2 + rpy[1] ** 2)

        # 3) 关节力矩惩罚 -- 关节力矩平方和 ---------------------------------
        tau = self.data.actuator_force
        terms["torque"] = rc.torque * float(np.sum(np.square(tau)))

        # 4) 足端抬升奖励 -- 摆动足必须越过地形 -----------------------------
        feet = self.foot_pos
        foot_dt = self.cfg.control_dt
        # 三维足端速度：只看水平分量会奖励快速的贴地平扫，同时砍掉高台阶
        # 需要的慢速抬腿。
        foot_speed = np.linalg.norm(
            (feet - self._prev_foot_pos) / foot_dt, axis=1
        )
        gate = np.clip(foot_speed / max(float(rc.foot_lift_gate_speed), 1e-6),
                       0.0, 1.0)
        # geom_xpos 是球心，所以要减去半径，才能得到足端表面与地形之间的
        # 真实间隙。
        clearance = self.foot_clearance
        # 目标随地形一起变高，但在 5 cm 处它与原来的常量相等，所以已经学会
        # 的简单阶段不会回退。
        lift_target = max(
            rc.foot_lift_target, rc.foot_lift_step_ratio * float(self.stair.step_height)
        )
        lift = np.clip(clearance / lift_target, 0.0, 1.0)
        terms["foot_lift"] = rc.foot_lift * float(np.mean(gate * lift))

        # 5) 机身碰撞惩罚 -- 躯干接触地形 -----------------------------------
        # 逐步行且有界：它随躯干擦地的时长成比例，而不是在终止步给一个
        # 布尔量。
        terms["body_collision"] = rc.body_collision * float(trunk_contact)
        # 胫部 / 膝部蹭到立面。逐步有界，而且关键是它*不*结束回合，所以
        # 机器人可以在高楼梯上继续尝试。
        terms["leg_collision"] = rc.leg_collision * float(leg_contact)

        # 6) 足端腾空时间 -- 步态成形（一次良好摆动后的落地） ---------------
        # ``_foot_air_time`` 保存的是当前步*之前*累积的腾空时间，
        # ``_prev_foot_contact`` 保存的是它之前的接触状态，所以这一项恰好在
        # 足端落地的那一步结算。
        if rc.air_time != 0.0:
            if foot_contact is None:
                foot_contact = self._foot_contact_mask()
            landed = foot_contact & ~self._prev_foot_contact
            air = self._foot_air_time
            credit = np.clip(air - rc.air_time_target, 0.0, None)
            overrun = np.clip(air - rc.air_time_max, 0.0, None)
            terms["air_time"] = rc.air_time * float(
                np.sum((credit - overrun) * landed)
            )
        else:
            terms["air_time"] = 0.0

        # --- 辅助成形 -----------------------------------------------------
        terms["base_height"] = rc.base_height * float(
            (self.base_pos[2] - self.terrain_height_under_base - target_height) ** 2
        )
        terms["joint_vel"] = rc.joint_vel * float(np.sum(np.square(self.joint_vel)))
        terms["action_rate"] = rc.action_rate * float(
            np.sum(np.square(action - self._prev_action))
        )
        lo, hi = self.act_low, self.act_high
        q = self.joint_pos
        over = np.clip(lo - q, 0, None) + np.clip(q - hi, 0, None)
        terms["joint_limit"] = rc.joint_limit * float(np.sum(over))
        terms["alive"] = rc.alive

        # --- 爬升进展 -------------------------------------------------------
        # 机身下方的地形高度相对回合起点的高度，所以在已经到达的台阶上原地
        # 不动拿不到任何新奖励。
        terms["stair_progress"] = rc.stair_progress * float(
            max(0.0, self.terrain_height_under_base - self._start_terrain_height)
        )
        return float(sum(terms.values())), terms

    def _foot_contact_mask(self) -> np.ndarray:
        """逐足布尔量：这个足端 geom 此刻是否接触地形？"""
        foot_ids = [int(g) for g in self.meta.foot_geom_ids]
        terrain = {int(self.meta.floor_geom_id)} | {
            int(g) for g in self.meta.stair_geom_ids
        }
        mask = np.zeros(len(foot_ids), dtype=bool)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for k, fg in enumerate(foot_ids):
                if (g1 == fg and g2 in terrain) or (g2 == fg and g1 in terrain):
                    mask[k] = True
        return mask

    def _advance_foot_air_time(self, foot_contact: np.ndarray) -> None:
        """累加腾空时间；每个控制步调用一次，在奖励计算之后。"""
        self._foot_air_time = np.where(
            foot_contact, 0.0, self._foot_air_time + self.cfg.control_dt
        )
        self._prev_foot_contact = foot_contact

    def _contact_flags(self) -> tuple[bool, bool]:
        """机器人 geom 与地形接触时的 (trunk_contact, leg_contact)。

        ``trunk_contact`` —— 躯干碰撞体之一（焊在 ``base`` body 上的 box /
        cylinder / sphere）正在接触地形。这是真正的摔倒，也是默认唯一致命的
        接触。

        ``leg_contact`` —— 髋 / 大腿 / 小腿的碰撞形状在躯干没接触时接触了
        地形。在楼梯上，这几乎总是爬升途中胫部蹭到立面：它每步扣
        ``leg_collision``，但回合继续运行，这正是保留试错的原因。
        """
        trunk = {int(g) for g in self.meta.trunk_geom_ids}
        legs = {int(g) for g in self.meta.leg_geom_ids}
        if not trunk and not legs:
            return False, False
        terrain = {int(self.meta.floor_geom_id)} | {
            int(g) for g in self.meta.stair_geom_ids
        }
        trunk_hit = False
        leg_hit = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in terrain:
                other = g2
            elif g2 in terrain:
                other = g1
            else:
                continue
            if other in trunk:
                trunk_hit = True
            elif other in legs:
                leg_hit = True
        return trunk_hit, leg_hit

    def _effective_contact_steps(self) -> int:
        """会让回合结束的躯干接触连续控制步数。"""
        return max(1, int(self.cfg.terminate_contact_steps))

    def _contact_fatal(self, trunk_contact: bool, leg_contact: bool) -> bool:
        """躯干接触地形的时间是否已长到致命？

        连续计数在这里推进而不是在 ``step`` 里，这样当测试直接驱动它时，
        「调用本方法一次 == 一个控制步」也成立。胫部蹭立面由
        ``reward.leg_collision`` 定价，永不致命 —— 因它而终止回合会抹掉高
        台阶恰恰需要的那个信号。
        """
        self._trunk_contact_streak = self._trunk_contact_streak + 1 if trunk_contact else 0
        self._leg_contact_streak = self._leg_contact_streak + 1 if leg_contact else 0
        if not trunk_contact:
            return False
        return self._trunk_contact_streak >= self._effective_contact_steps()

    def _support_height(self) -> float:
        """四足下方最低的地形高度 —— 支撑面。

        与 :meth:`terrain_height_under_base` 相对照，后者是躯干 x 位置的
        阶梯函数。
        """
        heights = [self.stair.height_at(float(x)) for x in self.foot_pos[:, 0]]
        return float(min(heights)) if heights else self.terrain_height_under_base

    def _support_height_mean(self) -> float:
        """四足下方地形高度的均值。

        ``support``（最小值）在最后一只脚落地时一帧就跳一整个立面；均值把这个
        跳变摊到四次 ``h/4`` 增量上。它*并不*只是更宽松 —— 跨越过程中均值高于
        最小值，所以会更早触发；它消除的是单帧的不连续。
        """
        heights = [self.stair.height_at(float(x)) for x in self.foot_pos[:, 0]]
        return float(np.mean(heights)) if heights else self.terrain_height_under_base

    def _height_reference(self) -> float:
        """``obs.base_height`` 的高度参照面：机身正下方的地形。"""
        return self.terrain_height_under_base

    def _fall_reference_height(self) -> float:
        """摔倒判据用来衡量躯干的地形高度。

        ``"support"`` 用四足下方最低的地形，那才是真正撑住机器人的那个面；
        *机身*下方的地形是躯干 x 位置的阶梯函数，相邻两帧之间会跳一整个立面，
        从而产生幽灵摔倒。
        """
        mode = str(self.cfg.terminate_height_ref).strip().lower()
        if mode == "support_mean":
            return self._support_height_mean()
        if mode != "support":
            return self.terrain_height_under_base
        return self._support_height()

    def _terminated(self, trunk_contact: bool, leg_contact: bool) -> tuple[bool, bool]:
        rpy = self.base_rpy
        height = self.base_pos[2] - self._fall_reference_height()
        fell = (
            height < self.cfg.terminate_min_height
            or abs(rpy[0]) > self.cfg.terminate_max_tilt
            or abs(rpy[1]) > self.cfg.terminate_max_tilt
        )
        contact_fatal = self._contact_fatal(trunk_contact, leg_contact)
        truncated = self._step_count >= self._max_steps
        return bool(fell or contact_fatal), truncated

    # ------------------------------------------------------------ Gymnasium

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)

        x = self.stair.start_x - self.cfg.spawn_back + self._rng.uniform(
            -self.cfg.reset_xy_noise, self.cfg.reset_xy_noise
        )
        y = self.stair.center_y + self._rng.uniform(
            -self.cfg.reset_xy_noise, self.cfg.reset_xy_noise
        )
        yaw = self._rng.uniform(-self.cfg.reset_yaw_noise, self.cfg.reset_yaw_noise)
        self.data.qpos[:] = home_qpos(self.meta, x=x, y=y,
                                      z=self.terrain_height_under_base + self.cfg.reset_z,
                                      yaw=yaw)
        noise = self._rng.uniform(
            -self.cfg.reset_joint_noise, self.cfg.reset_joint_noise, self.num_act
        )
        self.data.qpos[self.meta.qpos_joint_adr] += noise
        self.data.qvel[:] = self._rng.uniform(-0.05, 0.05, self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        self.nominal_pose = float(self.meta.home_qpos[2])
        self._start_terrain_height = self.terrain_height_under_base
        self._prev_action = np.zeros(self.num_act)
        self._last_action = np.zeros(self.num_act)
        self._step_count = 0
        self._reached_top = False
        self._trunk_contact_streak = 0
        self._leg_contact_streak = 0
        self._prev_foot_pos = self.foot_pos.copy()
        self._foot_air_time = np.zeros(len(self.meta.foot_geom_ids))
        self._prev_foot_contact = self._foot_contact_mask()
        # 步态先验时钟，单位为自回合开始起的秒数。观测里报告的相位就是本步
        # 实际使用的那一个。
        self._gait_t = 0.0
        self._gait_phase = 0.0
        self._hist.clear()
        frame = self._obs_frame()
        for _ in range(self.cfg.obs.history_len):
            self._hist.append(frame.copy())

        return self._observation(), {}

    def _gait_action(self) -> np.ndarray:
        """步态先验在当前相位下的关节角偏移（rad）。

        先验被禁用时恒为零，所以 ``_target_from_action`` 对两种模式（以及对
        ``GaitParams.static()``）都是同一条代码路径。
        """
        if not self.cfg.gait_prior or self.cfg.gait_gain == 0.0:
            return np.zeros(self.num_act, dtype=np.float64)
        return self.cfg.gait_gain * self.cfg.gait.radians(self._gait_phase)

    def _target_from_action(self, action: np.ndarray) -> np.ndarray:
        """把策略动作映射为 PD 关节目标（整条动作流水线）。

        残差 RL 有意把两个幅值分开：

            target = home + gain * gait.radians(t) + action_scale * action

        ``gait.action_scale`` 表示先验动作有多大（它是被搜索的步态的一部分）；
        ``cfg.action_scale`` 表示策略可以偏离它多少。共用一个尺度会让策略的
        探索噪声随先验一起缩放，瞬间就把机器人搞失稳。
        """
        if self.cfg.action_mode == "residual":
            target = (self.home_angles
                      + self._gait_action()
                      + self.cfg.action_scale * action)
        else:
            # "absolute" 把动作直接映射到关节范围上，所以残差先验在那里没有
            # 意义；它被有意忽略。
            lo, hi = self.act_low, self.act_high
            target = lo + (action + 1.0) * 0.5 * (hi - lo)
        return np.clip(target, self.act_low, self.act_high)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)

        # 残差 RL：先验提供一个能用的周期步态，策略提供修正量。观测里报告的
        # 相位就是本步实际使用的那一个，所以转移保持马尔可夫性。
        self._gait_phase = self._gait_t
        target = self._target_from_action(action)
        self._gait_t += self.cfg.control_dt

        for _ in range(self.cfg.decimation):
            q = self.data.qpos[self.meta.qpos_joint_adr]
            qd = self.data.qvel[self.meta.qvel_joint_adr]
            tau = self._kp * (target - q) - self._kd * qd
            self.data.ctrl[:] = np.clip(
                tau, self.meta.ctrl_ranges[:, 0], self.meta.ctrl_ranges[:, 1]
            )
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        trunk_contact, leg_contact = self._contact_flags()
        foot_contact = self._foot_contact_mask()
        terminated, truncated = self._terminated(trunk_contact, leg_contact)

        # 目标终止。到达顶部 / 平地目标会*成功*结束回合，而不是让机器人走出
        # 楼梯远端摔下去。``_reached_top`` 在回合内是粘性的，所以更早某一步
        # 的摔倒仍然算失败。
        if self._success_reached():
            self._reached_top = True
            if self.cfg.terminate_on_success and not terminated:
                terminated = True
                truncated = False

        reward, terms = self._compute_reward(
            action, trunk_contact, foot_contact, leg_contact=leg_contact
        )
        self._advance_foot_air_time(foot_contact)
        terminal = 0.0
        if terminated:
            terminal = (self.cfg.reward.success if self._reached_top
                        else self.cfg.reward.termination)
            reward += terminal
        terms["terminal"] = float(terminal)

        self._prev_action = self._last_action.copy()
        self._last_action = action.copy()
        # 足端 xy 位移必须恰好在一个控制步上测量，所以这个快照每一步都要
        # 推进。
        self._prev_foot_pos = self.foot_pos.copy()
        self._hist.append(self._obs_frame())

        info = {
            "reward_terms": terms,
            "trunk_contact": float(trunk_contact),
            "leg_contact": float(leg_contact),
            "terminate_contact_steps": int(self.cfg.terminate_contact_steps),
            "trunk_contact_streak": int(self._trunk_contact_streak),
            "base_x": float(self.base_pos[0]),
            "base_y": float(self.base_pos[1]),
            "base_z": float(self.base_pos[2]),
            "terrain_height": float(self.terrain_height_under_base),
            "forward_vel": float(self.base_lin_vel_world[0]),
            "roll": float(self.base_rpy[0]),
            "pitch": float(self.base_rpy[1]),
            "crest_height": float(self.stair.crest_height),
            "reach_top_x": float(self.stair.up_end_x),
            "reached_top": float(self._reached_top),
            "is_success": bool(self._reached_top),
            "fallen": bool(terminated) and not self._reached_top,
            "gait_prior": bool(self.cfg.gait_prior),
        }
        return self._observation(), float(reward), terminated, truncated, info

    # ---------------------------------------------------------------- 渲染

    #: :meth:`render` 能识别的相机预设。
    CAMERAS = ("side", "track", "chase", "front", "top")

    def render(self, camera: str = "side", *, width: int = 640, height: int = 480):
        """离屏渲染当前状态；``camera`` 取值见 :attr:`CAMERAS`。

        ``side``（默认）是固定的整段楼梯侧视图，``track`` 是同一视角但注视点
        沿 x 跟随机器人，``chase`` 是沿 +x 的后上方视角，``front`` / ``top``
        用于出静帧。
        """
        if camera not in self.CAMERAS:
            raise ValueError(f"未知相机 {camera!r}, 可选 {self.CAMERAS}")
        size = (int(width), int(height))
        if self._renderer is None or getattr(self, "_render_size", None) != size:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=size[1], width=size[0])
            self._render_size = size
            self._render_cam = mujoco.MjvCamera()
            # ``mjv_defaultOption`` 默认 ``geomgroup = [1, 1, 1, 0, 0, 0]``，
            # 所以 group 3、4、5 里的任何 geom 默认都不可见。地形放在专用
            # group 里（见 ``terrain.TERRAIN_GEOM_GROUP``），而它曾经被塞进隐藏
            # 区间，导致楼梯在每张 GIF 里都消失、物理却还在往上爬。把六个
            # group 全打开，这样渲染器就永远不会悄悄丢掉整个世界。
            self._scene_option = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self._scene_option)
            for g in range(6):
                self._scene_option.geomgroup[g] = 1
        mujoco.mjv_defaultFreeCamera(self.model, self._render_cam)
        base = self.base_pos
        if camera == "side":
            self._render_cam.azimuth = 90.0
            self._render_cam.elevation = -12.0
            self._render_cam.distance = 4.0
            self._render_cam.lookat[:] = (self.stair.start_x + 1.0, 0.0, 0.3)
        elif camera == "track":
            self._render_cam.azimuth = 90.0
            self._render_cam.elevation = -9.0
            self._render_cam.distance = 2.5
            self._render_cam.lookat[:] = (
                float(base[0]) + 0.35, float(base[1]), float(base[2]) + 0.02)
        elif camera == "chase":
            self._render_cam.azimuth = 135.0
            self._render_cam.elevation = -22.0
            self._render_cam.distance = 3.0
            self._render_cam.lookat[:] = (
                float(base[0]) + 0.6, float(base[1]), float(base[2]) + 0.10)
        elif camera == "front":
            self._render_cam.azimuth = 180.0
            self._render_cam.elevation = -10.0
            self._render_cam.distance = 3.0
            self._render_cam.lookat[:] = (
                float(base[0]) + 1.2, float(base[1]), float(base[2]) + 0.10)
        else:  # 顶视
            self._render_cam.azimuth = 90.0
            self._render_cam.elevation = -85.0
            self._render_cam.distance = 4.5
            self._render_cam.lookat[:] = (
                float(base[0]) + 0.8, float(base[1]), float(base[2]))
        self._renderer.update_scene(
            self.data, camera=self._render_cam, scene_option=self._scene_option)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._render_size = None
        self._scene_option = None
