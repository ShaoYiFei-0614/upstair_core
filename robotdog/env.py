"""Gymnasium 环境：Unitree Go2 盲走上箱式楼梯。

**盲走（blind locomotion）**：本模块中没有任何触觉传感器、没有射线投射，
也没有任何接触力读取。策略只能看到本体感知——关节角/角速度、机身姿态与角速度、
上一步动作——外加一个标量，告诉它被放上去的那条楼梯有多高。它必须仅凭这些
发现迈步的节奏。

* **Action** -- 12 个关节目标角（``FL_hip .. RR_calf``）。
* **Observation** -- ``ObsConfig``：每帧 42 个本体感知量（关节位置/速度、机身
  roll/pitch/yaw、角速度、上一步动作），再加上被打开的那些观测块（三轴速度指令、
  楼梯高度等）；最近 ``history_len`` 帧会被堆叠。任何地方都没有接触或测距传感器
  数据。
* **Reward** -- ``RewardConfig``：三轴指令跟随、姿态、关节力矩、足端抬升、
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
from robotdog.reward import (
    RewardConfig,
    compute_reward,
    track_swing_peak,
    update_probe_state,
)
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

    每帧 = 42 维本体感知 + **6 维结构固定块**（机身系线速度 3 + 三轴指令 3：
    指令跟踪的「实测」与「指令」两侧，无开关直接写死）+ 按声明顺序追加的
    可选块；沿时间堆叠 ``history_len`` 帧。``step_height`` 是盲走策略无法
    低成本推断的那一项，所以它是一个普通数字，而不是要机器人自己去估计的量。
    ``doc/env.md`` 有逐块讲解与维度账。
    """

    history_len: int = 10
    base_height: bool = False           # 1：机身相对其正下方地形的高度（构造期禁止）
    gait_phase: bool = False            # 2：步态时钟的 sin/cos
    step_height: bool = True            # 1：本回合的楼梯高度
    vel_filter: bool = False            # 2：机身系横向速度 / 偏航角速度的**低通量**
    base_frame_size: int = 42

    def __post_init__(self) -> None:
        # 含地形真值、破坏盲走 —— 构造即拒绝，而不是默认关掉（该分支与实现已
        # 整体删除；「约定不是保证」，experience.md §11）。
        if self.base_height:
            raise ValueError(
                "obs.base_height 必须为 false：这一项是『机身相对正下方地形的高度』，"
                "含地形真值，盲走策略不允许拿到（见 doc/env.md 的信息来源清单）。"
                "要观察离地高度请用本体感知的替代品，或保持 step_height 这一项。"
            )

    @property
    def frame_size(self) -> int:
        # 结构固定块：机身系线速度 3 + 三轴指令 3（指令跟踪的两测，见类 docstring）。
        extra = 6
        extra += 2 if self.gait_phase else 0
        extra += 1 if self.step_height else 0
        extra += 2 if self.vel_filter else 0
        return self.base_frame_size + extra

    @property
    def size(self) -> int:
        return self.frame_size * self.history_len


@dataclass
class CommandConfig:
    """三轴速度指令 ``[vx, vy, wz]``。

    每个轴都是「范围 + 当前值」两层：范围由指令课程按阶段下发（``robotdog/config.py``
    的 ``config_to_stages``），当前值在 ``reset()`` 时从范围内采样，也可以由外部
    控制器用 :meth:`Go2StairsEnv.set_command_now` 随时改写（``scripts/teleop.py``）。
    ``_now`` 就是当前值，奖励与观测都读它。

    范围的上界与下界取 ``±vy_max`` / ``±yaw_max``；前向是对称轴之外的量（不可能
    往回走），所以用 ``vx_min`` / ``vx_max`` 一对。``vx_max == 0`` 表示「前向不采样」，
    此时 ``vx`` 作为定值基线使用。
    """

    vx: float = 0.55        # m/s，前向定值（vx_max == 0 时生效）
    vy_max: float = 0.0     # m/s，横向指令的对称上界；0 = 关闭
    yaw_max: float = 0.0    # rad/s，偏航指令的对称上界；0 = 关闭
    vx_min: float = 0.0     # m/s，前向指令的下界（含）
    vx_max: float = 0.0     # m/s，前向指令的上界（含）；0 = 不采样，用 vx
    _now: list = field(default_factory=lambda: [0.55, 0.0, 0.0], repr=False)

    def __post_init__(self) -> None:
        self._now[0] = float(self.vx)
        # ``vx_max == 0`` 是「前向不采样、用 ``vx`` 定值」，此时 ``vx_min`` 无意义。
        # 只要真的采样，区间就必须非空 —— 否则会在 ``reset()`` 里抽出 ``lo > hi``。
        if self.vx_max and self.vx_min > self.vx_max:
            raise ValueError(
                f"前向指令范围为空：vx_min={self.vx_min} > vx_max={self.vx_max}"
            )

    @property
    def has_lateral(self) -> bool:
        """横向或旋转指令是否启用。"""
        return bool(self.vy_max or self.yaw_max)

    @property
    def has_forward(self) -> bool:
        """前向指令是否按范围采样。"""
        return bool(self.vx_max or self.vx_min)

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

    # --- 指令（外部控制器接口）：盲走允许的信息（操作者/上游规划器给出，不含
    # 地形真值）。横向与偏航每回合从 [-max, +max] 采样，量程随课程放宽。
    command: CommandConfig = field(default_factory=CommandConfig)
    # 横向速度 / 偏航角速度的低通时间常数 [s]（压掉步态周期、跟上指令变化）。
    vel_filter_tau: float = 0.2

    # --- 开环步态之上的残差 RL --------------------------------------------- #
    # PD 目标 = home + gait_gain * gait.radians(t) + action_scale * action
    gait_prior: bool = False
    gait: GaitParams = field(default_factory=GaitParams.default_trot)
    gait_gain: float = 1.0
    gait_amp_scale: float = 1.0         # 全局幅值缩放（只缩 amp 保留步频，缩后过 normalize()）
    gait_speed_track: bool = True       # 速度参数化：先验幅值按当前指令缩放（doc/gait.md）
    kp: tuple[float, float, float] = (30.0, 30.0, 45.0)
    kd: tuple[float, float, float] = (0.6, 0.6, 0.9)

    # --- 终止 -------------------------------------------------------------- #
    # 高度判据（"摔倒了没有"）以哪个面为基准：
    #   "support" -> 四足下方的地形（真正支撑它的那个面）
    #   "support_mean" -> 四个支撑面取均值（台阶边界处更平滑）
    #   "base" -> 机身下方的地形（阶梯函数，会产生幽灵摔倒）
    terminate_height_ref: str = "support"
    # 摔倒的高度阈值：机身高度（base_z 减参照面）低于它就判摔倒。
    terminate_min_height: float = 0.15
    terminate_max_tilt: float = 1.0         # rad，roll 或 pitch
    terminate_on_success: bool = True

    # --- 重置噪声 ---------------------------------------------------------- #
    reset_z: float = 0.32
    reset_joint_noise: float = 0.06
    reset_xy_noise: float = 0.12
    reset_yaw_noise: float = 0.10

    # --- 成功 -------------------------------------------------------------- #
    spawn_back: float = 0.55                # 第一个立面前方的米数
    # 平地阶段的成功判据是**指令跟随**：脱靶步占比与「实测量真的跟着指令变」的
    # 相关系数双门槛（常数输出能把平均误差压到最小，骗不过相关系数）。容差按
    # 「真在跟的策略实测误差的 1.4~1.6 倍」标定，依据见 experience.md §18。
    flat_track_tol_vy: float = 0.30         # m/s
    flat_track_tol_wz: float = 0.70         # rad/s
    flat_track_tol_vx: float = 0.35         # m/s（比横移松：步态的周期速度波动主要在前后向）
    flat_track_miss_ratio: float = 0.30     # 上限含指令阶跃过渡段的天然脱靶份额
    flat_track_min_corr: float = 0.30       # 只对指令本身在变的轴判定；作用是排除零响应
    # 回合内每隔这么多**控制步**重抽一次指令（0 或负数 = 整回合定值）。定值指令
    # 下「输出常数」是最优解；回放/遥操作用 set_command_now(hold=True) 关掉。
    command_hold_steps: int = 100

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
        self._last_action = np.zeros(self.num_act, dtype=np.float64)
        self._step_count = 0
        # 本回合四足支撑面曾达到的最大高度（由 reward.py 的 stair_progress 结算）。
        self._peak_terrain_height = 0.0
        self._reached_top = False
        self._leg_contact_streak = 0
        # 「被阻挡 → 试探抬腿」的状态（见 reward.py 的 probe_lift）。
        self._blocked = False
        self._blocked_peak = 0.0
        self._probe_count = 0
        self._probe_best = np.zeros(len(self.meta.foot_geom_ids))
        self._peak_swing_clearance = np.zeros(len(self.meta.foot_geom_ids))

        # 低通滤波后的横向速度 / 偏航角速度：一阶 IIR，``alpha = dt / tau``，
        # 初值取 reset 时的瞬时值（避免开头几十毫秒从 0 慢慢爬）。
        tau = max(1e-6, float(self.cfg.vel_filter_tau))
        self._vel_alpha = min(1.0, self.cfg.control_dt / tau)
        self.filtered_lateral_vel = 0.0
        self.filtered_yaw_rate = 0.0
        # 平地指令跟随判据的脱靶步计数，每回合清零（见 ``_success_reached``）。
        self._tracking_miss_count = 0
        # 跟踪统计量（每回合清零；供 _tracking_quality 算相关系数，见其 docstring）。
        self._track_n = 0
        self._track_cmd_sum = np.zeros(3)
        self._track_cmd_sq = np.zeros(3)
        self._track_cross = np.zeros(3)
        self._cmd_age = 0
        self._track_act_sum = np.zeros(3)
        self._track_act_sq = np.zeros(3)

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

        # observation_space 在构造时冻结，而真实观测宽度由 obs 开关现算 ——
        # 构造之后再动开关会拖到 PPO 内部才爆，所以就地现造一帧核对一次。
        frame_size = int(self._obs_frame().shape[0])
        real_size = frame_size * int(self.cfg.obs.history_len)
        frozen = int(self.observation_space.shape[0])
        if real_size != frozen:
            raise ValueError(
                f"观测维度不自洽：observation_space 是 {frozen} 维，但按当前 obs 开关"
                f"算出来是每帧 {frame_size} 维 × history_len="
                f"{self.cfg.obs.history_len} = {real_size} 维。"
                f"请用 robotdog.config.config_to_env() 构造配置，构造之后不要再改 obs 的开关。"
            )


    # ------------------------------------------------------------------- 构建

    def _build_model(self) -> None:
        """编译场景；模型在环境生命周期内不变（每一级课程都是新建环境）。

        ``mujoco.Renderer`` 绑定到具体的 ``MjModel``、不能比模型活得久 ——
        「只在 ``__init__`` 编译一次」由结构保证这一不变量。
        """
        opts = SceneOptions(timestep=self.cfg.timestep)
        self.model, self.meta, self.xml = compile_scene(
            build_terrain_xml(self.stair),
            opts=opts,
            floor_geom=default_floor_geom(),
        )
        self.data = mujoco.MjData(self.model)

    # ------------------------------------------------------------------- 指令

    def _sample_command(self) -> list[float]:
        """按当前范围抽一条指令 ``[vx, vy, wz]``。上界为 0 的轴恒为 0。"""
        cmd = self.cfg.command
        vx = (float(self._rng.uniform(cmd.vx_min, cmd.vx_max))
              if cmd.vx_max else float(cmd.vx))
        vy = float(self._rng.uniform(-cmd.vy_max, cmd.vy_max)) if cmd.vy_max else 0.0
        wz = (float(self._rng.uniform(-cmd.yaw_max, cmd.yaw_max))
              if cmd.yaw_max else 0.0)
        return [vx, vy, wz]

    def set_commands(self, vy_max: float | None = None,
                     yaw_max: float | None = None,
                     vx_min: float | None = None,
                     vx_max: float | None = None) -> None:
        """改写指令的采样范围，并立刻重抽一条当前指令。

        **外部控制器接口**：训练时由 ``scripts/train.py`` 在每个课程阶段开始时
        按 ``curriculum.command_*`` 调用；评估时也可以每个控制步调一次，用外部
        规划器给出的值覆盖 —— 这就是模拟人类遥感操控的接法，见
        ``scripts/teleop.py``。

        ``None`` 表示该项不变。传 0 会关掉那一项（但**观测维数不会因此改变**：
        ``ObsConfig.command`` 与 ``EnvConfig.command`` 的初始范围决定维数，
        训练中途改动维数会让 PPO 的网络与 RolloutBuffer 对不上）。
        """
        cmd = self.cfg.command
        if vy_max is not None:
            cmd.vy_max = float(vy_max)
        if yaw_max is not None:
            cmd.yaw_max = float(yaw_max)
        if vx_min is not None:
            cmd.vx_min = float(vx_min)
        if vx_max is not None:
            cmd.vx_max = float(vx_max)
        if cmd.vx_max < cmd.vx_min:
            raise ValueError(
                f"前向指令范围为空：vx_min={cmd.vx_min} > vx_max={cmd.vx_max}"
            )
        self._command[:] = self._sample_command()

    def set_command_now(self, vy: float | None = None,
                        yaw_rate: float | None = None,
                        vx: float | None = None,
                        *, hold: bool = False) -> list[float]:
        """**立即**写入一条指令，不重新采样（外部控制器的高频入口）。

        与 :meth:`set_commands` 的分工：那个改的是「采样范围」，由课程在阶段
        边界调用；这个改的是「这一刻的指令」，可以由控制器每个控制步调用。
        传 ``None`` 的项保持当前值。返回改写后的 ``[vx, vy, wz]``。

        传了 ``vx`` 时会**同时改写基线** ``cfg.command.vx``，即
        ``vx_max == 0`` 阶段的重置回退值；``vx_max > 0`` 的阶段每回合会重抽
        覆盖它。``vy`` / ``wz`` 只改当前值，它们每回合本来就该重抽。

        ``hold=True`` 表示「这条指令要一直生效，别被回合内的周期重抽覆盖」
        —— 回放、遥操作与 GIF 都用它；默认 ``False`` 则只写到下一次重抽为止。
        """
        if hold:
            # 把年龄顶到不可能到达的值：``step()`` 里的重抽判据就永远不成立。
            self._cmd_age = 10 ** 9
        else:
            self._cmd_age = 0
        self.cfg.command._set_now(vx=vx, vy=vy, wz=yaw_rate)
        if vx is not None:
            self.cfg.command.vx = float(vx)
        # 刷新「已经算好、还没被取走」的那一帧观测：控制器是在两次 ``step()``
        # 之间下指令的，不刷新的话策略拿到的那一帧还带着上一步的指令（50 Hz
        # 下 20 ms 的滞后）。指令是外生输入，改它不破坏马尔可夫性。
        if self._hist:
            frame = self._obs_frame()
            if self._hist[-1].shape == frame.shape:
                self._hist[-1] = frame
        return list(self._command)

    # ------------------------------------------------------------------- 成功

    def _success_target_x(self) -> float:
        """本次回合要被算作成功，机身必须越过的 x（**只对楼梯段有意义**）。

        平地段的成功判据是指令跟随，不涉及位置，所以返回 ``+inf`` —— 平地上
        没有「走过去就算赢」的终点。
        """
        if self.stair.n_up > 0:
            return float(self.stair.up_end_x)
        return float("inf")

    def _tracking_miss(self) -> bool:
        """本控制步的三轴速度是否脱靶（实测与指令之差超过容差）。

        横向/偏航只读**滤波后**的量（瞬时值被步态摆动主导，等于在判步态相位）；
        前向用机身系瞬时速度（本来平滑）。
        """
        cmd = self.cfg.command
        err_vx = abs(float(self.base_lin_vel_body[0]) - float(self._command[0]))
        err_vy = abs(self.filtered_lateral_vel - float(self._command[1]))
        err_wz = abs(self.filtered_yaw_rate - float(self._command[2]))
        return bool(err_vy > self.cfg.flat_track_tol_vy
                    or err_wz > self.cfg.flat_track_tol_wz
                    or err_vx > self.cfg.flat_track_tol_vx)

    def _record_tracking_stats(self) -> None:
        """累加三轴「指令 vs 实测」的一阶/二阶矩（只在 ``_step_count > 0`` 时调：
        第 0 步的速度还是 reset 后的零向量，混进去会压低相关系数）。"""
        act = np.array([float(self.base_lin_vel_body[0]),
                        float(self.filtered_lateral_vel),
                        float(self.filtered_yaw_rate)])
        cmd = np.asarray(self._command, dtype=np.float64)
        self._track_n += 1
        self._track_cmd_sum += cmd
        self._track_cmd_sq += cmd * cmd
        self._track_act_sum += act
        self._track_act_sq += act * act
        self._track_cross += cmd * act

    def _tracking_quality(self) -> tuple[int, float]:
        """返回 ``(参与判定的轴数, 最差轴的相关系数)``。

        平均绝对误差在「输出常数」时取最小，判跟随必须用相关系数（对常数输出
        恒为 0）。取**最差**轴（只跟对一轴不算按指令走）；只判指令本身在变的轴
        （std > 0.05，定值基线轴上没有「跟随」可言）；没有可判轴时返回 0.0 ——
        「没有证据」不能当「通过」。
        """
        n = max(1, self._track_n)
        cmd_mean = self._track_cmd_sum / n
        act_mean = self._track_act_sum / n
        cmd_var = np.maximum(self._track_cmd_sq / n - cmd_mean * cmd_mean, 0.0)
        act_var = np.maximum(self._track_act_sq / n - act_mean * act_mean, 0.0)
        cov = self._track_cross / n - cmd_mean * act_mean
        axes = 0
        worst = 1.0
        for i in range(3):
            if float(np.sqrt(cmd_var[i])) <= 0.05:
                continue
            axes += 1
            denom = float(np.sqrt(cmd_var[i] * act_var[i]))
            corr = float(cov[i] / denom) if denom > 1e-9 else 0.0
            worst = min(worst, corr)
        if axes == 0:
            return 0, 0.0
        return axes, worst

    def _success_reached(self) -> bool:
        if self.stair.n_up > 0:
            return bool(
                self.base_pos[0] > self.stair.up_end_x
                and self.terrain_height_under_base >= self.stair.crest_height - 1e-9
            )
        # 平地：撑满整个回合 **且** 全程按指令走。两个条件缺一不可 —— 只看
        # 时长会奖励「原地不动」，只看随动会奖励「活到一半就摔」。
        if self._step_count < self._max_steps - 1:
            return False
        miss = self._tracking_miss_count / max(1, self._step_count)
        if miss > self.cfg.flat_track_miss_ratio:
            return False
        # 第二个条件：实测量必须真的**跟着指令变**，不能靠输出常数满足容差。
        axes, corr = self._tracking_quality()
        return bool(axes == 0 or corr >= self.cfg.flat_track_min_corr)

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
        """偏航角速度，**机体系 z 轴分量**（rad/s）—— 偏航指令要跟踪的量。

        不用世界系公式（``wz + tan(pitch)·(cos yaw·wx + sin yaw·wy)``）：它被与
        指令量程同阶的步态振荡污染，等于把一半权重扔进策略控制不到的通道
        （experience.md §16）。机体系读数便宜、抖动小、贴近真机陀螺口径；
        不引入地形信息，可进奖励。
        """
        wx, wy, wz = quat_to_mat(self.base_quat).T @ self.data.qvel[3:6]
        return float(wz)

    @property
    def foot_clearance(self) -> np.ndarray:
        """每只足端*表面*与其下方地形之间的间隙（m）。

        ``geom_xpos`` 是足端球的球心，比较前先减去半径。
        **特权信息，只进奖励**（foot_lift、probe_lift 的摆动峰值）：策略只能
        学到「按 ``step_height`` 抬够高」这条相关性，没有任何观测项直接等于它。
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
        # 结构固定块：机身系线速度（指令跟踪的「实测」侧，无开关）。
        parts.append(self.base_lin_vel_body)
        # 这里故意没有 base_height 分支：它含地形真值，ObsConfig 构造期已禁止。
        if obs_cfg.gait_phase:
            theta = 2.0 * np.pi * self.cfg.gait.freq * self._gait_phase
            parts.append(np.array([np.sin(theta), np.cos(theta)]))
        if obs_cfg.step_height:
            # 楼梯信息里唯一直接告诉策略的一项。盲走仍然需要知道自己在*哪一段*
            # 楼梯上；但它不会知道下一级立面在哪里。
            parts.append(np.array([float(self.stair.step_height)]))
        # 结构固定块：三轴速度指令（「指令」侧；来自操作者/上游规划器，不含地形真值）。
        parts.append(np.asarray(self._command, dtype=np.float64).copy())
        if obs_cfg.vel_filter:
            # 滤波后的横向速度 / 偏航角速度。奖励的 command_follow 读同一批量 ——
            # 只改奖励不改观测会把滤波器状态藏起来、破坏马尔可夫性，必须成对出现。
            parts.append(np.array([self.filtered_lateral_vel, self.filtered_yaw_rate]))
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

    def _contact_scan(self) -> tuple[bool, bool, np.ndarray]:
        """一次遍历 ``data.ncon`` 同时判定三类「与地形的接触」。

        返回 ``(trunk_contact, leg_contact, foot_contact_mask)``。原先这是两个
        方法（``_contact_flags`` / ``_foot_contact_mask``）各扫一遍接触列表，
        而它们判定的是同一批接触对，合并成一次扫描。

        ``trunk_contact`` —— 躯干碰撞体之一（焊在 ``base`` body 上的 box /
        cylinder / sphere）正在接触地形。这是真正的摔倒，也是默认唯一致命的
        接触。

        ``leg_contact`` —— 髋 / 大腿 / 小腿的碰撞形状在躯干没接触时接触了
        地形。在楼梯上，这几乎总是爬升途中胫部蹭到立面：它每步扣
        ``leg_collision``，但回合继续运行，这正是保留试错的原因。

        ``foot_contact_mask`` —— 逐足布尔量：这个足端 geom 此刻是否接触地形。
        **信息来源：仿真器的接触真值。** 真机上对应的是「用关节力矩残差 / 电流
        尖峰估计足端触地」，这是标准的 sim-to-real 做法，所以它不是特权信息 ——
        但它**只进奖励与腾空时间统计，绝不进观测**（``reward.py`` 的 ``air_time``
        项与 ``clearance`` 的分布都靠它区分摆动相与支撑相）。
        """
        foot_index = {int(g): k for k, g in enumerate(self.meta.foot_geom_ids)}
        terrain = {int(self.meta.floor_geom_id)} | {
            int(g) for g in self.meta.stair_geom_ids
        }
        trunk = {int(g) for g in self.meta.trunk_geom_ids}
        legs = {int(g) for g in self.meta.leg_geom_ids}
        trunk_hit = False
        leg_hit = False
        mask = np.zeros(len(foot_index), dtype=bool)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in terrain and g2 not in terrain:
                other = g2
            elif g2 in terrain and g1 not in terrain:
                other = g1
            else:
                continue
            if other in trunk:
                trunk_hit = True
            elif other in legs:
                leg_hit = True
            k = foot_index.get(other)
            if k is not None:
                mask[k] = True
        return trunk_hit, leg_hit, mask

    def _advance_foot_air_time(self, foot_contact: np.ndarray) -> None:
        """累加腾空时间；每个控制步调用一次，在奖励计算之后。"""
        self._foot_air_time = np.where(
            foot_contact, 0.0, self._foot_air_time + self.cfg.control_dt
        )
        self._prev_foot_contact = foot_contact

    def set_reward_overrides(self, overrides: dict[str, float] | None) -> None:
        """把一组奖励覆盖值叠到 ``cfg.reward`` 上（课程逐级调整奖励的入口）。

        就地改、只覆盖传入的键。调用方必须保证：每级子环境持有**自己的 cfg
        深拷贝**，且**先叠基线再叠本级键**（否则上一级覆盖过、本级没覆盖的键
        会残留上一级的值）—— ``scripts/train.py`` 的 ``make_env`` 都做了。
        """
        for key, value in (overrides or {}).items():
            if not hasattr(self.cfg.reward, key):
                raise ValueError(
                    f"未知的奖励字段：{key}；可用字段见 RewardConfig.__dataclass_fields__"
                )
            setattr(self.cfg.reward, key, value)

    def _support_height(self) -> float:
        """四足下方最低的地形高度 —— 真正撑住机器人的支撑面。

        **特权信息**（读 ``stair.height_at()`` 真值）：只服务终止判定与
        stair_progress，绝不能写进 ``_obs_frame``（盲走契约，见 doc/env.md）。
        """
        heights = self._terrain_height(self.foot_pos[:, 0])
        return float(np.min(heights)) if heights.size else self.terrain_height_under_base

    def _support_height_mean(self) -> float:
        """四足下方地形高度的均值：把「最后一只脚落地时一帧跳一整个立面」的
        单帧不连续摊平（并非更宽松 —— 跨越过程中均值高于最小值，会更早触发）。"""
        heights = self._terrain_height(self.foot_pos[:, 0])
        return float(np.mean(heights)) if heights.size else self.terrain_height_under_base

    def _fall_reference_height(self) -> float:
        """摔倒判据的地形参照面（**特权信息**，绝不进观测）。

        ``"support"``（默认）= 四足下方最低地形，那才是真正撑住机器人的面；
        机身下方地形是躯干 x 的阶梯函数，相邻帧跳一整个立面会产生幽灵摔倒
        （experience.md §1.4）。``"support_mean"`` 为可选的平滑口径。
        """
        mode = str(self.cfg.terminate_height_ref).strip().lower()
        if mode == "support_mean":
            return self._support_height_mean()
        if mode != "support":
            raise ValueError(
                f"未知的 terminate_height_ref {self.cfg.terminate_height_ref!r}，"
                f"可选 'support'（四足下方最低地形，默认）或 'support_mean'。"
            )
        return self._support_height()

    def _terminated(self, leg_contact: bool) -> tuple[bool, bool]:
        """摔倒（高度 + 倾角）与超时判定。

        接触类信号**不**终止回合：躯干接触的代价由 ``reward.body_collision``
        （逐步、有界）承担；「连续接触 N 步判摔」的历史规则从未生效且会误杀
        上台架时的正常蹭立面（experience.md §12.4）。膝/胫接触只推进诊断计数。
        """
        self._leg_contact_streak = self._leg_contact_streak + 1 if leg_contact else 0
        rpy = self.base_rpy
        height = self.base_pos[2] - self._fall_reference_height()
        fell = (
            height < self.cfg.terminate_min_height
            or abs(rpy[0]) > self.cfg.terminate_max_tilt
            or abs(rpy[1]) > self.cfg.terminate_max_tilt
        )
        truncated = self._step_count >= self._max_steps
        return bool(fell), truncated

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
        self._peak_terrain_height = self._support_height()   # 足端支撑面，不是躯干 x（见 §2.7）
        self._last_action = np.zeros(self.num_act)
        self._command[:] = self._sample_command()
        self._step_count = 0
        self._tracking_miss_count = 0
        self._track_n = 0
        self._cmd_age = 0
        self._track_cmd_sum[:] = 0.0
        self._track_cmd_sq[:] = 0.0
        self._track_cross[:] = 0.0
        self._track_act_sum[:] = 0.0
        self._track_act_sq[:] = 0.0
        self._reached_top = False
        self._leg_contact_streak = 0
        self._blocked = False
        self._blocked_peak = 0.0
        self._probe_count = 0
        self._probe_best = np.zeros(len(self.meta.foot_geom_ids))
        self._peak_swing_clearance = np.zeros(len(self.meta.foot_geom_ids))
        self._prev_foot_pos = self.foot_pos.copy()
        self._foot_air_time = np.zeros(len(self.meta.foot_geom_ids))
        self._prev_foot_contact = self._contact_scan()[2]
        # 步态先验时钟，单位为自回合开始起的秒数。观测里报告的相位就是本步
        # 实际使用的那一个。
        self._gait_t = 0.0
        self._gait_phase = 0.0
        self._hist.clear()
        # 滤波器用「回合第一步的瞬时值」做初值：从 0 开始爬会白白浪费开头一段，
        # 而 reset 时机器人是站在原地的，瞬时值本身就是合理的初值。
        self.filtered_lateral_vel = float(self.base_lin_vel_body[1])
        self.filtered_yaw_rate = float(self.yaw_rate)
        frame = self._obs_frame()
        for _ in range(self.cfg.obs.history_len):
            self._hist.append(frame.copy())

        return self._observation(), {}

    def _gait_scale(self) -> float:
        """当前指令下先验应有的幅值缩放（速度参数化）。

        关掉 ``gait_speed_track`` 时恒为 1.0，于是动作流水线退化回「固定先验」
        那条老路径（`config_to_env` 里的全局 ``gait_amp_scale`` 仍然生效）。
        """
        if not self.cfg.gait_speed_track:
            return 1.0
        return float(GaitParams.scale_for_speed(float(self._command[0])))

    def _gait_action(self) -> np.ndarray:
        """步态先验在当前相位下的关节角偏移（rad）。

        先验被禁用时恒为零，于是动作流水线对 ``gait_prior=False``
        和对 ``GaitParams.static()`` 是同一条代码路径。
        """
        if not self.cfg.gait_prior or self.cfg.gait_gain == 0.0:
            return np.zeros(self.num_act, dtype=np.float64)
        gait = self.cfg.gait
        scale = self._gait_scale()
        if scale != 1.0:
            gait = gait.with_dynamic_scale(scale)
        return self.cfg.gait_gain * gait.radians(self._gait_phase)

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

        # 回合内周期性重抽指令：定值指令下「输出常数」是最优解，重抽之后
        # 「跟得上变化」才有意义（见 command_hold_steps 与 experience.md §18）。
        self._cmd_age += 1
        if self._cmd_age >= self.cfg.command_hold_steps:
            self._cmd_age = 0
            self._command[:] = self._sample_command()

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
        trunk_contact, leg_contact, foot_contact = self._contact_scan()
        # 奖励结算之前推进势函数状态（状态机数学在 reward.py，见 doc/reward.md）。
        track_swing_peak(self, foot_contact)
        update_probe_state(self)
        terminated, truncated = self._terminated(leg_contact)

        # 低通滤波必须在奖励之前推进：``command_follow`` 读的就是这两个量，
        # 观测里报的也是它们（见 ``ObsConfig.vel_filter``）。
        alpha = self._vel_alpha
        self.filtered_lateral_vel += alpha * (
            float(self.base_lin_vel_body[1]) - self.filtered_lateral_vel)
        self.filtered_yaw_rate += alpha * (float(self.yaw_rate) - self.filtered_yaw_rate)

        # 脱靶步与跟踪统计：楼梯段不用它判成功，但 info 要报同一口径；
        # 统计量晚一拍不影响整回合的相关系数。
        if self._step_count > 0:
            self._record_tracking_stats()
        if self._tracking_miss():
            self._tracking_miss_count += 1

        # 目标终止。到达顶部会*成功*结束回合，而不是让机器人走出楼梯远端摔下去。
        # ``_reached_top`` 在回合内是粘性的，所以更早某一步的摔倒仍然算失败。
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
            "tracking_miss_ratio": float(
                self._tracking_miss_count / max(1, self._step_count)
            ),
            "tracking_corr": float(self._tracking_quality()[1]),
            "gait_prior": bool(self.cfg.gait_prior),
        }
        return self._observation(), float(reward), terminated, truncated, info

    # ---------------------------------------------------------------- 渲染

    #: :meth:`render` 能识别的相机预设。
    CAMERAS = ("side", "track", "chase", "front", "top")

    #: 各相机的 ``(azimuth, elevation, distance)``。注视点在 :meth:`render` 里
    #: 按相机名取：``side`` 固定在楼梯入口，其余跟随机身。
    _CAMERA_VIEW = {
        "side": (90.0, -12.0, 4.0),
        "track": (90.0, -9.0, 2.5),
        "chase": (135.0, -22.0, 3.0),
        "front": (180.0, -10.0, 3.0),
        "top": (90.0, -85.0, 4.5),
    }

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
            # 打开全部 6 个 geom group：MuJoCo 默认隐藏 3–5，地形曾因此整段消失
            # 而物理照常（experience.md §1.2）。
            self._scene_option = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self._scene_option)
            for g in range(6):
                self._scene_option.geomgroup[g] = 1
        mujoco.mjv_defaultFreeCamera(self.model, self._render_cam)
        base = self.base_pos
        azimuth, elevation, distance = self._CAMERA_VIEW[camera]
        self._render_cam.azimuth = azimuth
        self._render_cam.elevation = elevation
        self._render_cam.distance = distance
        # ``side`` 固定看整段楼梯入口，其余注视点跟随机身。
        self._render_cam.lookat[:] = {
            "side":  (self.stair.start_x + 1.0, 0.0, 0.3),
            "track": (float(base[0]) + 0.35, float(base[1]), float(base[2]) + 0.02),
            "chase": (float(base[0]) + 0.6, float(base[1]), float(base[2]) + 0.10),
            "front": (float(base[0]) + 1.2, float(base[1]), float(base[2]) + 0.10),
            "top":   (float(base[0]) + 0.8, float(base[1]), float(base[2])),
        }[camera]
        self._renderer.update_scene(
            self.data, camera=self._render_cam, scene_option=self._scene_option)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._render_size = None
        self._scene_option = None
