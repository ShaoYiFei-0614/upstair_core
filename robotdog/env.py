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
from robotdog.reward import RewardConfig, compute_reward
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

    ``base_frame_size`` 始终存在（42 个本体感知量）。下面几个块按声明顺序
    追加；``step_height`` 是盲走策略无法低成本推断的那一项，所以它是一个普通
    数字，而不是要机器人自己去估计的量。``README.md`` §3.2 列出了由此得到的
    维度。
    """

    history_len: int = 10
    base_lin_vel_body: bool = True      # 3：机身系线速度
    base_height: bool = False           # 1：机身相对其正下方地形的高度
    gait_phase: bool = False            # 2：步态时钟的 sin/cos
    step_height: bool = True            # 1：本回合的楼梯高度
    command: bool = True                # 2：横向速度指令 + 偏航角速度指令
    base_frame_size: int = 42

    def __post_init__(self) -> None:
        # 这条不是「默认关掉」，而是**禁止打开**。机身相对正下方地形的高度里含
        # 地形真值（原实现是 ``base_pos[2] - _height_reference()``，而
        # ``_height_reference()`` 就是 ``terrain_height_under_base``）。盲走策略
        # 一旦拿到它就不再是盲走。原来这里只有一行注释，结果它被打开过 ——
        # 所以改成硬错误，并连带删掉了 `_obs_frame` 里那条分支与
        # `_height_reference()` 本身。
        if self.base_height:
            raise ValueError(
                "obs.base_height 必须为 false：这一项是『机身相对正下方地形的高度』，"
                "含地形真值，盲走策略不允许拿到（见 agent.md §8.6 的信息来源清单）。"
                "要观察离地高度请用本体感知的替代品，或保持 step_height 这一项。"
            )

    @property
    def frame_size(self) -> int:
        extra = 0
        extra += 3 if self.base_lin_vel_body else 0
        extra += 2 if self.gait_phase else 0
        extra += 1 if self.step_height else 0
        extra += 2 if self.command else 0
        return self.base_frame_size + extra

    @property
    def size(self) -> int:
        return self.frame_size * self.history_len


@dataclass
class CommandConfig:
    """横向 / 旋转指令：**范围**在这里，**当前值**在 ``_now`` 里。

    三个指令是 ``vx``（前向 m/s）、``vy``（横向 m/s）、``wz``（偏航角速度
    rad/s）。这就是「外部控制器」的接口：控制器每个控制步都可以调用
    :meth:`Go2StairsEnv.set_command_now` 改写 ``vy``/``wz``，训练时则由
    ``reset()`` 从 ``[-vy_max, +vy_max]`` / ``[-yaw_max, +yaw_max]`` 里随机采样，
    用随机指令覆盖指令空间。

    存取分离的原因：奖励要读「当前指令」，而它每步都在变；把当前值放在一个
    长度 3 的数组里、范围放在标量字段里，两者就不会互相覆盖。
    ``config.yaml`` 的 ``curriculum.command_vy_max`` / ``command_yaw_max`` 按阶段
    给出范围，见 ``robotdog/config.py`` 的 ``config_to_stages``。

    ``vy_max`` 与 ``yaw_max`` 都是 0 时横向/偏航指令恒为 0，于是整条奖励与观测
    路径退化成旧的「恒速前向」，逐位一致。
    """

    vx: float = 0.55        # m/s，前向指令（定值；不随回合变化）
    vy_max: float = 0.0     # m/s，横向指令的对称上界；0 = 关闭
    yaw_max: float = 0.0    # rad/s，偏航指令的对称上界；0 = 关闭
    _now: list = field(default_factory=lambda: [0.55, 0.0, 0.0], repr=False)

    def __post_init__(self) -> None:
        self._now[0] = float(self.vx)

    @property
    def has_lateral(self) -> bool:
        """横向或旋转指令是否启用。"""
        return bool(self.vy_max or self.yaw_max)

    @property
    def vy(self) -> float:
        """当前横向指令（m/s）。"""
        return float(self._now[1])

    @property
    def wz(self) -> float:
        """当前偏航角速度指令（rad/s）。"""
        return float(self._now[2])

    def _set_now(self, vx: float | None = None, vy: float | None = None,
                 wz: float | None = None) -> None:
        if vx is not None:
            self._now[0] = float(vx)
        if vy is not None:
            self._now[1] = float(vy)
        if wz is not None:
            self._now[2] = float(wz)


@dataclass
class EnvConfig:
    """一个环境实例的全部可调项。"""

    timestep: float = 0.002
    decimation: int = 10                    # -> 50 Hz 控制
    episode_length_s: float = 20.0
    action_scale: float = 0.30              # rad，策略权限

    # --- 指令（外部控制器接口）-------------------------------------------- #
    # 三个量：前向 vx（定值）、横向 vy、偏航角速度 wz。横向与偏航是随机指令，
    # 每回合从 [-max, +max] 里采样，随课程按阶段放宽（``config_to_stages``）。
    # 它们是**盲走允许的信息**：指令由操作者/上游规划器给出，不含地形真值，
    # 与 ``command_vx`` 同源（见 agent.md §8.6 的信息来源清单）。
    command: CommandConfig = field(default_factory=CommandConfig)

    # --- 开环步态之上的残差 RL --------------------------------------------- #
    # PD 目标变为 home + gait_gain * gait.radians(t) + action_scale * action
    gait_prior: bool = False
    gait: GaitParams = field(default_factory=GaitParams.default_trot)
    gait_gain: float = 1.0
    kp: tuple[float, float, float] = (30.0, 30.0, 45.0)
    kd: tuple[float, float, float] = (0.6, 0.6, 0.9)

    # --- 终止 -------------------------------------------------------------- #
    # 高度判据（"摔倒了没有"）以哪个面为基准：
    #   "support" -> 四足下方的地形（真正支撑它的那个面）
    #   "support_mean" -> 四个支撑面取均值（台阶边界处更平滑）
    #   "base" -> 机身下方的地形（阶梯函数，会产生幽灵摔倒）
    terminate_height_ref: str = "support"
    # 摔倒的高度阈值：机身高度（base_z 减参照面）低于它就判摔倒。
    # 取值依据见 experience.md §12：躯干碰撞体最低点在 0.118 m，所以
    # 0.15 留了 3.2 cm 余量，同时比原来的 0.13 更早抓到「趴下去」。
    terminate_min_height: float = 0.15
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
    # 平地阶段是否要求「真的往前走一段」才算成功。**原地学横移/扭转的课程必须
    # 把它关掉**：那种阶段 `command.vx = 0`，要求前进就永远拿不到成功，
    # 进度行的成功率恒为 0、看不出学习信号；关掉后平地阶段改成「整回合不摔即成功」。
    flat_success_needs_progress: bool = True

    # --- 观测 / 奖励 ------------------------------------------------------- #
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    # 逐级奖励覆盖：``config.yaml`` 的 ``curriculum.reward_overrides`` 会按阶段
    # 填这里，``config_to_env`` 与 ``set_reward_overrides()`` 把它叠到 ``reward``
    # 上。默认空 ⇒ 与没有这个机制时逐位一致。
    reward_overrides: dict[str, float] = field(default_factory=dict)

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
        # 本回合机身正下方地形曾达到的最大高度，用于「爬升进展」的一次性
        # 结算（见 ``robotdog/reward.py`` 的 ``stair_progress``）。
        self._peak_terrain_height = 0.0
        self._reached_top = False
        self._leg_contact_streak = 0
        # 「被阻挡 → 试探抬腿」的状态（见 reward.py 的 probe_lift）。
        self._blocked = False
        self._blocked_peak = 0.0
        self._probe_count = 0
        self._probe_best = 0.0
        self._peak_swing_clearance = np.zeros(len(self.meta.foot_geom_ids))

        self._success_x = self._success_target_x()
        self.nominal_pose = float(self.meta.home_qpos[2])  # 0.27 m 站立高度
        # 当前指令 ``[vx, vy, wz]``。**与 ``cfg.command._now`` 是同一个列表**，
        # 所以「外部控制器改写指令」与「奖励读当前指令」看到的是同一份数据，
        # 不存在两份状态不同步的窗口。
        self._command = self.cfg.command._now
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
        """替换地形。会重新编译模型。

        调用方之后必须调用 ``reset()``：重新编译会替换底层的
        ``MjModel``/``MjData``，所以旧状态没有意义。

        **当前没有任何调用点**（本轮清理时删掉了唯一调用者
        ``training.set_terrain()``）：课程的每一级都是新建一个环境
        （``make_vec_env``），不是复用同一个环境换地形。保留它是因为
        「换地形」本身是真需求，而且文档把它当公开 API 记录。
        """
        self.stair = stair
        self._build_model()
        self._success_x = self._success_target_x()
        self._reached_top = False

    # ------------------------------------------------------------------- 指令

    def _sample_command(self) -> list[float]:
        """按当前范围抽一条指令 ``[vx, vy, wz]``。

        ``vx`` 是定值；``vy``/``wz`` 在 ``[-max, +max]`` 上均匀采样。上界为 0
        时该项恒为 0，于是这条路径与「恒速前向」逐位一致。
        """
        cmd = self.cfg.command
        vy = float(self._rng.uniform(-cmd.vy_max, cmd.vy_max)) if cmd.vy_max else 0.0
        wz = (float(self._rng.uniform(-cmd.yaw_max, cmd.yaw_max))
              if cmd.yaw_max else 0.0)
        # vx 取**当前值**而不是字段值：``set_command_now(vx=...)`` 可以按课程阶段
        # 压低它，如果这里读回 ``cmd.vx``，下一次 ``reset()`` 就会把它抹掉。
        return [float(cmd._now[0]), vy, wz]

    def set_commands(self, vy_max: float | None = None,
                     yaw_max: float | None = None) -> None:
        """改写横向/偏航指令的采样上界，并立刻重抽一条当前指令。

        **外部控制器接口**：训练时由 ``scripts/train.py`` 在每个课程阶段开始时
        按 ``curriculum.command_vy_max`` / ``command_yaw_max`` 调用（向量环境下
        走 ``venv.env_method("set_commands", ...)``）；评估时也可以每个控制步
        调一次，用外部规划器给出的值覆盖 —— 这就是模拟人类遥感操控的接法，
        见 ``scripts/teleop.py``。

        ``None`` 表示该项不变。传 0 会关掉那一项（但**观测维数不会因此改变**：
        ``ObsConfig.command`` 与 ``EnvConfig.command`` 的初始范围决定维数，
        训练中途改动维数会让 PPO 的网络与 RolloutBuffer 对不上）。
        """
        cmd = self.cfg.command
        if vy_max is not None:
            cmd.vy_max = float(vy_max)
        if yaw_max is not None:
            cmd.yaw_max = float(yaw_max)
        self._command[:] = self._sample_command()

    def set_flat_success_distance(self, distance: float) -> None:
        """改写平地阶段的成功距离 [m]，并立刻刷新成功判据的阈值。

        只对平地段有意义（楼梯段的成功判据是「跨过最后一级立面并达到顶部
        高度」）。指令课程放宽横移/旋转之后，机器人会为了跟指令而绕路，原来
        3 m 的前进判据会把成功拖成超时，所以平地预热级要能单独调短它。
        """
        self.cfg.flat_success_distance = float(distance)
        self._success_x = self._success_target_x()

    def set_flat_success_needs_progress(self, needs_progress: bool) -> None:
        """平地阶段是否要求真的前进一段才算成功。

        **原地指令课程必须传 False**：那种阶段 ``command.vx = 0``，要求前进就
        永远拿不到成功，训练日志的成功率恒为 0（看不出学习信号）。传 False 后
        成功判据变成「撑满整个回合」，见 :meth:`_success_reached`。
        """
        self.cfg.flat_success_needs_progress = bool(needs_progress)

    def set_command_now(self, vy: float | None = None,
                        yaw_rate: float | None = None,
                        vx: float | None = None) -> list[float]:
        """**立即**写入一条指令，不重新采样（外部控制器的高频入口）。

        与 :meth:`set_commands` 的分工：那个改的是「采样范围」，由课程在阶段
        边界调用；这个改的是「这一刻的指令」，可以由控制器每个控制步调用。
        传 ``None`` 的项保持当前值。返回改写后的 ``[vx, vy, wz]``。

        传了 ``vx`` 时会**同时改写基线** ``cfg.command.vx``：否则 ``reset()``
        重抽指令时会把课程下发的定值前进指令丢掉（这是课程里
        ``curriculum.command_vx`` 生效的路径）。``vy`` / ``wz`` 只改当前值，
        它们每回合本来就该重抽。
        """
        self.cfg.command._set_now(vx=vx, vy=vy, wz=yaw_rate)
        if vx is not None:
            self.cfg.command.vx = float(vx)
        # ⭐ 刷新「已经算好、还没被取走」的那一帧观测。
        #
        # 时序：``step()`` 结束时把 ``_obs_frame()`` 压进 ``_hist``，下一次
        # ``step()`` 再把它取出来返回。外部控制器是在**两次 step 之间**下指令
        # 的（``teleop.py`` / ``cmd_demo.py`` 的循环就是这个顺序），所以如果不
        # 刷新，策略拿到的那一帧观测里带的还是**上一步**的指令 —— 一个控制步的
        # 滞后（50 Hz 下 20 ms）。单步看不出来，但它是「设了指令策略却没反应」
        # 这类误判的来源之一。指令是外生输入，改它不破坏马尔可夫性。
        if self._hist and self._hist[-1].shape == self._obs_frame().shape:
            self._hist[-1] = self._obs_frame()
        return list(self._command)

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
        if not self.cfg.flat_success_needs_progress:
            # 原地指令课程：没有前进指令可跟，成功 = **撑满整个回合**。
            # ⚠ 必须带 ``_step_count`` 条件：不带的话第一步就算成功，
            # ``terminate_on_success`` 会让回合在第 1 步结束、白拿 success 奖励。
            return bool(self._step_count >= self._max_steps - 1)
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

    @property
    def yaw_rate(self) -> float:
        """偏航角速度，**机体系 z 轴分量**（rad/s），是偏航速率指令要跟踪的量。

        为什么用机体系而不是世界系：这一项要拿去和「平稳的偏航指令」比较，而奖励
        信号的噪声底必须远小于要跟踪的量程。世界系 z 角速度
        ``wz + tan(pitch) * (cos(yaw) * wx + sin(yaw) * wy)`` 在数学上同样是偏航
        速率，但四足步态下躯干以几十赫兹小幅摆动，那个 ``tan(pitch)`` 乘积项就能
        贡献约 0.28 rad/s —— 与指令量程（±0.70）同阶，等于把一半权重扔进了策略
        根本控制不到的通道（三种独立方法实测见 ``experience.md`` §16）。

        机体系读数的抖动与 ``qvel[5]`` 同阶（≈0.16），计算更便宜，而且**更贴近
        真机**：机载陀螺直接给的就是机体系角速度，真机不需要先做欧拉角换算。

        只读陀螺与姿态，不引入任何地形信息，所以可以进奖励。
        """
        wx, wy, wz = quat_to_mat(self.base_quat).T @ self.data.qvel[3:6]
        return float(wz)

    @property
    def foot_clearance(self) -> np.ndarray:
        """每只足端*表面*与其下方地形之间的间隙（m）。

        ``geom_xpos`` 是足端球的球心，所以在与地形高度比较之前必须减去半径。

        **这是特权信息，只进奖励。** 它要和 ``stair.height_at()`` 相减，真机上
        拿不到这个数值；策略能学到的只是「按 ``step_height`` 这一个可观测量把脚
        抬够高」这一条相关性。**没有**任何观测项直接等于它。
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
        # 注意：这里**故意没有** base_height 分支。它原本是
        # ``base_pos[2] - _height_reference()``，而 ``_height_reference()`` 就是
        # 机身正下方的地形高度 —— 那会把地形真值塞进观测。``ObsConfig`` 的
        # ``__post_init__`` 现在直接禁止打开该开关，所以这条分支已整体删除。
        if obs_cfg.gait_phase:
            theta = 2.0 * np.pi * self.cfg.gait.freq * self._gait_phase
            parts.append(np.array([np.sin(theta), np.cos(theta)]))
        if obs_cfg.step_height:
            # 楼梯信息里唯一直接告诉策略的一项。盲走仍然需要知道自己在*哪一段*
            # 楼梯上；但它不会知道下一级立面在哪里。
            parts.append(np.array([float(self.stair.step_height)]))
        if obs_cfg.command:
            # 横向 / 偏航指令。这是**允许进入观测**的：指令来自操作者或上游
            # 规划器，不含地形真值（与 vx 同源，见 agent.md §8.6）。
            parts.append(self._command[1:].copy())
        return np.concatenate(parts).astype(np.float64)

    def _observation(self) -> np.ndarray:
        return np.concatenate(list(self._hist)).astype(np.float32)

    # ------------------------------------------------------------------- 奖励

    def _compute_reward(self, action: np.ndarray, trunk_contact: bool,
                        foot_contact: np.ndarray | None = None,
                        leg_contact: bool = False,
                        terminated: bool = False
                        ) -> tuple[float, dict[str, float]]:
        """本步奖励与逐项明细；数学全在 :func:`robotdog.reward.compute_reward`。"""
        base, terms, terminal = compute_reward(
            self, action, trunk_contact, foot_contact,
            leg_contact=leg_contact, terminated=terminated,
        )
        terms["terminal"] = terminal
        return base + terminal, terms

    def _foot_contact_mask(self) -> np.ndarray:
        """逐足布尔量：这个足端 geom 此刻是否接触地形？

        **信息来源：仿真器的接触真值。** 真机上对应的是「用关节力矩残差 / 电流
        尖峰估计足端触地」，这是标准的 sim-to-real 做法，所以它不是特权信息 ——
        但它**只进奖励与腾空时间统计，绝不进观测**（``reward.py`` 的 ``air_time``
        项与 ``clearance`` 的分布都靠它区分摆动相与支撑相）。
        """
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

    def _track_swing_peak(self, foot_contact: np.ndarray) -> None:
        """维护每只脚「本次摆动期内的最大离地间隙」，在奖励计算**之前**调用。

        为什么要单独维护：落地那一帧脚已经贴回踏面，当时的间隙接近 0，所以
        「这次摆动到底抬了多高」只能靠摆动期的峰值来判断，落地帧本身看不出
        来。脚一落地就把峰值清零，为下一次摆动重新计。

        ⚠ **清零必须发生在奖励结算之后，而不是接触的那一帧**。实测踩过这个
        坑：如果一接触就清零，那么落地帧读到的是「上一帧的峰值」，比整个摆动
        期的真实峰值低 0.01~0.03 m，探测项的判据会系统性偏严。做法是「落地那
        一帧保留旧峰值，结算完再清零」。
        """
        clearance = self.foot_clearance
        landed = foot_contact & ~self._prev_foot_contact
        self._peak_swing_clearance = np.where(
            landed, self._peak_swing_clearance,
            np.where(foot_contact, 0.0,
                     np.maximum(self._peak_swing_clearance, clearance)),
        )

    def _update_probe_state(self, foot_contact: np.ndarray) -> None:
        """维护「被阻挡」状态，在物理推进之后、奖励计算**之前**调用。

        判据：**有前进指令，但实测前进速度远低于指令**（比例见
        ``reward.probe_blocked_frac``）。一旦判定为被阻挡就**锁住**，直到机身
        下方的地形刷新了本回合的最高高度（真的上去了）或前进速度恢复为止 ——
        因为机器人撞上立面时会前后晃，不锁住的话这个布尔量会一步一跳。

        ``_blocked_peak`` 是锁住那一刻记录的地形高，用来判断「有没有上去」。
        解除阻挡时顺带把 ``_probe_best`` 清零，让下一次被卡住可以重新累积
        「抬高进展」（否则势函数在整条回合里只增不减，第二次卡住就没有梯度）。
        """
        rc = self.cfg.reward
        cmd_vx = float(self.cfg.command.vx)
        vx = float(self.base_lin_vel_world[0])
        terrain = float(self.terrain_height_under_base)
        if self._blocked:
            recovered = terrain > self._blocked_peak + 1e-9
            if recovered or vx > max(0.5 * cmd_vx, 0.05):
                self._blocked = False
                self._probe_best = 0.0
        if not self._blocked and cmd_vx > 1e-6:
            if vx < rc.probe_blocked_frac * cmd_vx:
                self._blocked = True
                self._blocked_peak = terrain
                self._probe_best = 0.0

    @property
    def probe_target(self) -> float:
        """「试探抬腿」的起算线 [m]（势函数塑形的地板）。

        直接取 ``reward.probe_lift_target``，**不乘台阶高**。原因：实测
        （``.tmp/calib_probe3.py``）在 15 cm 立面之前被阻挡时，默认步态的摆动
        峰值只到 0.09~0.12 m；而 ``action_scale=0.30`` 下搜遍固定动作，最好的
        一种也只到 0.148 m（``.tmp/calib_probe4.py``）。门槛若取
        ``1.6 × 0.15 = 0.24 m``，梯度恒为零、这一项永远不触发。改成只跟「当前
        策略自己抬多高」比的 0.06 m 之后，普通步态的上沿被扣掉，剩下的都是
        真的「比平时抬得更高」。
        """
        return float(self.cfg.reward.probe_lift_target)

    def set_reward_overrides(self, overrides: dict[str, float] | None) -> None:
        """把一组奖励覆盖值叠到 ``cfg.reward`` 上（课程逐级调整奖励的入口）。

        ``scripts/train.py`` 在每一级建完环境之后用
        ``venv.env_method("set_reward_overrides", stage.reward_overrides)`` 调用，
        所以**同一个进程里每一级都能有自己的一套奖励**。传 ``None`` 或 ``{}``
        等价于「不动」。

        为什么不做成「新建 RewardConfig」：那样每一级都要把所有字段抄一遍，
        而实际上大多数级只改一两项；叠加重载让 ``config.yaml`` 里只写要改的
        那几个键就够了。
        """
        for key, value in (overrides or {}).items():
            if not hasattr(self.cfg.reward, key):
                raise ValueError(
                    f"未知的奖励字段：{key}；可用字段见 RewardConfig.__dataclass_fields__"
                )
            setattr(self.cfg.reward, key, value)

    def _contact_flags(self) -> tuple[bool, bool]:
        """机器人 geom 与地形接触时的 (trunk_contact, leg_contact)。

        ``trunk_contact`` —— 躯干碰撞体之一（焊在 ``base`` body 上的 box /
        cylinder / sphere）正在接触地形。这是真正的摔倒，也是默认唯一致命的
        接触。

        ``leg_contact`` —— 髋 / 大腿 / 小腿的碰撞形状在躯干没接触时接触了
        地形。在楼梯上，这几乎总是爬升途中胫部蹭到立面：它每步扣
        ``leg_collision``，但回合继续运行，这正是保留试错的原因。

        **只进奖励与终止判定，不进观测。** 真机上躯干碰地靠 IMU 的冲击尖峰 +
        关节力矩异常即可检测，级别上等同于一阶故障判据。
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

    def _contact_fatal(self, trunk_contact: bool, leg_contact: bool) -> bool:
        """接触地形是否会结束回合 —— 现在**恒为 False**。

        历史上这里对躯干接触计数：连续 ``terminate_contact_steps`` 步就判
        摔倒。实测证明那条规则从未生效过，而且随时可能误杀（见
        ``experience.md`` §12）：躯干碰撞体最低点在 0.118 m，而高度阈值
        ``terminate_min_height`` 比它还高，所以躯干一碰地，高度判据早就触发
        了；反过来，躯干在 ``h ≈ 0.28 m`` 的完全安全姿态下贴着立面蹭可以
        连续到 8 步，只差 2 步就会被判死 —— 那正是「机身上台架天然要蹭上去」
        的情形。

        保留这个方法是为了让 ``_terminated`` 的调用形状不变，也为了留住
        膝盖/胫部接触的连续计数（诊断用，永不致命 —— 因它而终止回合会抹掉
        高台阶恰恰需要的那个信号）。代价由 ``reward.body_collision``（逐步、
        有界）承担，不由终止承担。
        """
        self._leg_contact_streak = self._leg_contact_streak + 1 if leg_contact else 0
        return False

    def _support_height(self) -> float:
        """四足下方最低的地形高度 —— 支撑面。

        与 :meth:`terrain_height_under_base` 相对照，后者是躯干 x 位置的
        阶梯函数。

        **这是特权信息，只服务终止判定。** 它读 ``stair.height_at()``（地形几何
        真值），真机上要靠高程图或触地估计才能得到；因为只用于「判断摔没摔」，
        而 RL 的终止信号本来就是上帝视角，所以没有问题 —— 但**绝不能**把它写进
        ``_obs_frame``。
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

    def _fall_reference_height(self) -> float:
        """摔倒判据用来衡量躯干的地形高度。

        **只用于终止判定，绝不进观测。** 它读 ``stair.height_at()``，即地形几何
        真值；这在 RL 里是允许的（奖励与终止本来就是「上帝视角」），但一旦被写进
        ``_obs_frame`` 就不再是盲走。三条分支的存在是为了选一个**帧间连续**的参照：

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
        self._peak_terrain_height = self.terrain_height_under_base
        self._prev_action = np.zeros(self.num_act)
        self._last_action = np.zeros(self.num_act)
        self._command[:] = self._sample_command()
        self._step_count = 0
        self._reached_top = False
        self._leg_contact_streak = 0
        self._blocked = False
        self._blocked_peak = 0.0
        self._probe_count = 0
        self._probe_best = 0.0
        self._peak_swing_clearance = np.zeros(len(self.meta.foot_geom_ids))
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

        先验被禁用时恒为零，于是动作流水线对 ``gait_prior=False``
        和对 ``GaitParams.static()`` 是同一条代码路径。
        """
        if not self.cfg.gait_prior or self.cfg.gait_gain == 0.0:
            return np.zeros(self.num_act, dtype=np.float64)
        return self.cfg.gait_gain * self.cfg.gait.radians(self._gait_phase)

    def _target_from_action(self, action: np.ndarray) -> np.ndarray:
        """把策略动作映射为 PD 关节目标（整条动作流水线）。

        只有一种映射：**残差 RL**。它有意把两个幅值分开：

            target = home + gait_gain * gait.radians(t) + action_scale * action

        ``gait.action_scale`` 表示先验动作有多大（它是被搜索的步态的一部分）；
        ``cfg.action_scale`` 表示策略可以偏离它多少。共用一个尺度会让策略的
        探索噪声随先验一起缩放，瞬间就把机器人搞失稳。

        （历史上有过 "absolute" 模式：``target = lo + (a+1)/2 * (hi-lo)``。
        它从未被训练或被搜索使用过，只是 ``EnvConfig`` 里一个无人设置的
        分支，已随本次清理删除。）
        """
        target = (self.home_angles
                  + self._gait_action()
                  + self.cfg.action_scale * action)
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
        # 这两步必须在奖励计算之前：``_track_swing_peak`` 提供「本次摆动抬了
        # 多高」，``_update_probe_state`` 提供「现在是不是撞着立面推不动」，
        # 而 ``probe_lift`` 项要同时用到它们。
        self._track_swing_peak(foot_contact)
        self._update_probe_state(foot_contact)
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
            action, trunk_contact, foot_contact,
            leg_contact=leg_contact, terminated=terminated,
        )
        self._advance_foot_air_time(foot_contact)

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
            "leg_contact_streak": int(self._leg_contact_streak),
            "blocked": float(self._blocked),
            "probe_count": int(self._probe_count),
            "base_x": float(self.base_pos[0]),
            "base_y": float(self.base_pos[1]),
            "base_z": float(self.base_pos[2]),
            "terrain_height": float(self.terrain_height_under_base),
            "forward_vel": float(self.base_lin_vel_world[0]),
            "lateral_vel": float(self.base_lin_vel_body[1]),
            "yaw_rate": float(self.yaw_rate),
            "command": [float(v) for v in self._command],
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
