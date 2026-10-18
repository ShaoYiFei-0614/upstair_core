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
from dataclasses import dataclass, field, replace

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
    # 2：步态指令条件（freq [Hz] + swing_scale）—— Walk These Ways 式：步态参数是
    # **观测条件**而不是动作自由度（P2 的教训：参数放 action 两次全败，
    # experience.md §23.27/§23.28）。旧权重 json 缺此键，重建时按 False 还原 53 维。
    gait_cmd: bool = False
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
        extra += 2 if self.gait_cmd else 0
        return self.base_frame_size + extra

    @property
    def size(self) -> int:
        return self.frame_size * self.history_len


@dataclass
class CommandConfig:
    """三轴速度指令 ``[vx, vy, wz]`` —— **vx/vy 在世界系给定**，wz 为绕竖直轴
    的角速度（两系同值）。

    每个轴都是「范围 + 当前值」两层：范围由指令课程按阶段下发（``robotdog/config.py``
    的 ``config_to_stages``），当前值在 ``reset()`` 时从范围内采样，也可以由外部
    控制器用 :meth:`Go2StairsEnv.set_command_now` 随时改写（``scripts/teleop.py``）。
    ``_now`` 就是当前值（**世界系**），奖励与观测读的都是它的**机体系投影**
    （:meth:`Go2StairsEnv.command_body`）。

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
    action_scale: float = 0.30              # rad，策略权限（residual 模式的残差上限）
    # 动作模式："residual" = 12 维关节残差 + 固定先验（当前唯一在用，sub9_v5
    # 同款）；"gait_params" = P2 参数化生成器（楼梯两次全败已归档，
    # experience.md §23.24/25；字段保留供实验）。
    action_mode: str = "residual"
    gait_freq_flat: float = 1.67            # gait_params：平地步频 [Hz]
    gait_freq_stair: float = 1.25           # gait_params：楼梯步频 [Hz]
    stair_stride_boost: float = 1.4         # gait_params：楼梯步幅放大倍率
    param_amp_max: float = 0.8              # gait_params：幅值参数上限（归一化单位）
    param_off_max: float = 0.4              # gait_params：偏置参数上限
    gait_param_nominal: tuple[float, float, float, float] = (0.25, 0.50, 0.12, 0.0)
    param_hip_amp_max: float = 0.5
    param_hip_off_max: float = 0.4

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
    # ⭐ 分档先验频率（2026-10-14）：residual 非条件化下，楼梯级先验频率 =
    # stair_band 中点（1.25，自然降频诉求）、平地 = 先验 json 频率（1.67），
    # 见 ``_gait_freq_eff``。obs.gait_cmd=true 时 freq/swing 还作为**观测条件**
    # 进 obs（Walk-These-Ways 式）——⚠ 实测条件化把 RL 推向贴地（§23.29），
    # 当前关闭。
    gait_freq_flat_band: tuple[float, float] = (1.50, 1.85)    # 平地 freq 采样范围 [Hz]
    gait_freq_stair_band: tuple[float, float] = (1.10, 1.40)   # 楼梯 freq 采样范围 [Hz]
    #: 楼梯级**固定频率** [Hz]（>0 生效；0 = 按台阶高插值）。
    # 用户要求：楼梯段 1.25 Hz（experience.md §23.32）。
    gait_freq_stair_fixed: float = 0.0
    #: 楼梯级**专用先验**（2026-10-15，爬行步态）。None = 平楼梯共用 ``gait``。
    # 设了它时楼梯级的先验轨迹与频率都取自它（频率字段优先于上面的固定值）——
    # 平地段保持 trot 自然步态、楼梯段用爬行（4 足错相 + 小占空比 = 静态稳定）。
    gait_stair: GaitParams | None = None
    gait_swing_flat_band: tuple[float, float] = (0.90, 1.10)   # 平地 swing 缩放范围
    gait_swing_stair_band: tuple[float, float] = (1.00, 1.35)  # 楼梯 swing 缩放范围
    # 先验状态门控（P1）：受阻/站立时先验平滑衰减。⚠ 受阻门控实测有害默认关
    # （§23.20：先验在受阻态的节奏正是连续登踏的动力来源）；站立门控保留
    # （课程内不触发，服务于回放/遥操作的「站着别踏步」）。
    prior_gate_on_block: bool = False
    prior_gate_on_stand: bool = True
    prior_gate_tau: float = 0.3         # 门控平滑时间常数 [s]
    prior_gate_floor: float = 0.0       # 受阻/站立时的门控目标值
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
    # 平地阶段的成功判据是**指令跟随**：脱靶步占比与相关系数双门槛（常数输出
    # 骗不过相关系数）。容差 ≈ 2×「真在跟的策略」的预期误差（不放在典型误差
    # 恰好落地的值上，避免骑边界）。
    flat_track_tol_vy: float = 0.12         # m/s（vy_max ≤ 0.15，留 ~2.5 倍过渡余量）
    flat_track_tol_wz: float = 0.40         # rad/s（yaw_max ≤ 0.6；不跟者满量程 0.6 出局）
    flat_track_tol_vx: float = 0.35         # m/s（vx 带 [0.2,0.6] 未变，维持原值）
    flat_track_miss_ratio: float = 0.30     # 上限含指令阶跃过渡段的天然脱靶份额
    flat_track_min_corr: float = 0.30       # 只对指令本身在变的轴判定；作用是排除零响应
    # 回合内每隔这么多**控制步**重抽一次指令（0 或负数 = 整回合定值）。定值指令
    # 下「输出常数」是最优解；回放/遥操作用 set_command_now(hold=True) 关掉。
    command_hold_steps: int = 100
    # 指令缓变扰动（OU 过程）：训练时模拟真实操作手的轻微无规律输入 —— 每步
    # ``w ← w·exp(−dt/τ) + std·√dt·N(0,1)``，稳态幅度 ≈ std·√(τ/2)。叠加在采样
    # 指令之上；hold=True（回放/遥操作）时完全不生效；0 = 关闭。
    command_wander_std: float = 0.0
    command_wander_tau: float = 1.5
    # ⭐ 楼梯段**闭环外部引导**（2026-10-13，用户指令）：楼梯级（n_up > 0）的
    # vy/wz 指令不再来自课程采样，而是由外部控制器每步按状态闭环生成 ——
    # vy 按机身偏离台阶中线（center_y）的距离反号拉回，wz 按朝向误差对准
    # 课程方向（+x）。这是「人在遥感操控」的机制化（teleop.py 的
    # StairController 搬进环境）：跟随引导指令 = 保持中线与航向，狗走偏时
    # 指令会主动纠正它。vx 仍由课程采样（前进速度独立）。
    # 平地级不受影响（指令跟随任务照旧用采样范围）。
    stair_guidance: bool = True
    guidance_vy_gain: float = 1.2      # vy [m/s] 每米横向偏差
    guidance_vy_limit: float = 0.15    # vy 引导限幅
    guidance_yaw_gain: float = 1.5     # wz [rad/s] 每弧度朝向误差
    guidance_yaw_limit: float = 0.60   # wz 引导限幅

    # --- 观测 / 奖励 ------------------------------------------------------- #
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @property
    def control_dt(self) -> float:
        return self.timestep * self.decimation

    @property
    def needs_foot_force(self) -> bool:
        """本步要不要读足端接触力（2026-10-17 新增）。

        **默认 False ⇒ 训练热路径上一次接触遍历都不多花**。开启条件只有两个：
        遥测开关，或者「接触奖励总开关开着且至少一项权重非 0」。力只进奖励与
        `info`，**绝不进观测**（加观测维会让现有全部权重无法 warm-start）。
        """
        rc = self.reward
        if rc.contact_telemetry:
            return True
        return bool(rc.contact_rewards
                    and (rc.foot_scuff or rc.foot_impact
                         or rc.load_share or rc.com_margin))


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
        if self.cfg.action_mode not in ("residual", "gait_params"):
            raise ValueError(f"未知的 action_mode {self.cfg.action_mode!r}，"
                             f"可选 'residual' 或 'gait_params'")
        # 动作维度随模式变：residual=12（逐关节残差）；gait_params=12
        # （前/后对各 6 个参数：amp_th, amp_ca, off_th, off_ca, hip_amp, hip_off）
        self.num_action = 12 if self.cfg.action_mode == "residual" else 12
        self.act_low = self.meta.joint_ranges[:, 0].astype(np.float32)
        self.act_high = self.meta.joint_ranges[:, 1].astype(np.float32)
        self.home_angles = self.meta.home_joint_angles.astype(np.float32)

        self.action_space = spaces.Box(-1.0, 1.0, (self.num_action,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (self.cfg.obs.size,), dtype=np.float32
        )

        self._kp = np.tile(self.cfg.kp, 4)
        self._kd = np.tile(self.cfg.kd, 4)
        self._max_steps = int(self.cfg.episode_length_s / self.cfg.control_dt)

        self._hist: deque[np.ndarray] = deque(maxlen=self.cfg.obs.history_len)
        self._last_action = np.zeros(self.num_action, dtype=np.float64)
        # 上一步**下发的关节偏移**（12 维）：residual 模式即 action；gait_params 模式
        # 是由 8 个参数生成的轨迹。观测里报的是它（「我上一步命令了什么」）。
        self._last_traj = np.zeros(self.num_act, dtype=np.float64)
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
        # 受阻态驻留势的 Φ（见 reward.py 的 foot_hold）
        self._hold_phi = 0.0

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
        self._cmd_hold = np.zeros(3)      # 重抽的「驻留」指令（缓变扰动围绕它走）
        self._cmd_wander = np.zeros(3)    # 指令缓变扰动（OU 过程状态）
        self._prev_foot_pos = self.foot_pos.copy()
        self._foot_air_time = np.zeros(len(self.meta.foot_geom_ids))
        self._prev_foot_contact = np.zeros(len(self.meta.foot_geom_ids), dtype=bool)
        # 足端接触力（2026-10-17）。**只在 cfg.needs_foot_force 时才有值** ——
        # 默认全 0，关掉时一个字节都不会被读。三个数组的分工见
        # ``foot_normal_force`` / ``foot_impact_force`` 的 docstring。
        nfoot = len(self.meta.foot_geom_ids)
        self._foot_fn = np.zeros(nfoot)        # 本控制步**末**的法向力 [N]
        self._foot_ft = np.zeros(nfoot)        # 同刻的切向力模长 [N]
        self._foot_fn_peak = np.zeros(nfoot)   # 本控制步内（逐物理子步）的法向力峰值
        # 步态先验时钟（自回合开始起的秒数）+ gait_params 模式的步态时钟（周期数）。
        self._gait_t = 0.0
        self._gait_phase = 0.0
        self._gait_clock = 0.0
        self._gait_phase_cycles = 0.0
        self._prior_gate = 1.0            # 先验状态门控（受阻/站立时向 floor 衰减）
        self._vx_ema = 0.0                # 前向速度 EMA（门控的受阻判据）
        self._gate_slow_streak = 0        # 「平滑后速度接近零」的连续步数（去抖）
        self._gate_has_moved = False      # 本回合是否已动过（冷启动自锁的护栏）
        # 步态指令条件（freq/swing，见 EnvConfig.gait_freq_*_band）：reset 采样、
        # 回合内固定；obs.gait_cmd 开启时报给策略，策略伺服它（Walk These Ways 式）。
        self._gait_freq_cmd = float(self.cfg.gait.freq)
        self._gait_swing_cmd = 1.0
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
        """按当前范围抽一条**世界系**指令 ``[vx, vy, wz]``。上界为 0 的轴恒为 0。"""
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
        —— 回放、遥操作与视频录制都用它；默认 ``False`` 则只写到下一次重抽为止。
        ``vx`` / ``vy`` 为**世界系**分量（机体系投影由 :meth:`command_body`
        每步现算）。
        """
        if hold:
            # ⭐ 负值 = 永不满足重抽判据(``_cmd_age >= hold_steps``)。旧写法
            # ``10**9`` 恒满足 ``>=100`` → hold 的钉指令完全失效、回放时指令
            # 每步随机重抽(2026-10-15 实锤:同权重审查 3/3 vs play 0/10,
            # experience.md §23.30)。
            self._cmd_age = -(10 ** 9)
        else:
            self._cmd_age = 0
        self.cfg.command._set_now(vx=vx, vy=vy, wz=yaw_rate)
        if vx is not None:
            self.cfg.command.vx = float(vx)
        # 同步驻留值：缓变扰动/重抽以写入值为基准（hold=True 时扰动不生效）。
        self._cmd_hold[:] = [float(v) for v in self._command]

    def _sample_gait_cmd(self) -> None:
        """按地形采一条步态指令 ``(freq, swing)``,回合内固定。

        平地档中心 1.67 Hz、楼梯档中心 1.25 Hz（trot_natural167 标定）；
        swing 是先验幅值的条件缩放（楼梯档偏大 = 更从容的大步幅）。范围见
        ``EnvConfig.gait_freq_*_band`` / ``gait_swing_*_band``。
        """
        stair = self.stair.n_up > 0
        fb = (self.cfg.gait_freq_stair_band if stair
              else self.cfg.gait_freq_flat_band)
        sb = (self.cfg.gait_swing_stair_band if stair
              else self.cfg.gait_swing_flat_band)
        self._gait_freq_cmd = float(self._rng.uniform(*fb))
        self._gait_swing_cmd = float(self._rng.uniform(*sb))

    def set_gait_cmd_now(self, freq: float | None = None,
                         swing: float | None = None) -> tuple[float, float]:
        """**立即**写入步态指令（部署/play 的实时步态切换入口）。

        ``None`` 表示该项保持当前值；返回写后的 ``(freq, swing)``。swing 缩放
        先验幅值、不动常量偏置（与 ``GaitParams.with_dynamic_scale`` 同语义）。
        训练中 freq/swing 每回合由 :meth:`_sample_gait_cmd` 重采，这里的写入
        会被下一次 reset 覆盖 —— 要锁定请配合 ``set_command_now(hold=True)``
        的用法自行在回放循环里每步调用。
        """
        if freq is not None:
            self._gait_freq_cmd = float(freq)
        if swing is not None:
            self._gait_swing_cmd = float(swing)
        return self._gait_freq_cmd, self._gait_swing_cmd
        # 刷新「已经算好、还没被取走」的那一帧观测：控制器是在两次 ``step()``
        # 之间下指令的，不刷新的话策略拿到的那一帧还带着上一步的指令（50 Hz
        # 下 20 ms 的滞后）。指令是外生输入，改它不破坏马尔可夫性。
        if self._hist:
            frame = self._obs_frame()
            if self._hist[-1].shape == frame.shape:
                self._hist[-1] = frame
        return list(self._command)

    def _guidance_command(self) -> tuple[float, float]:
        """楼梯段的**闭环外部引导** ``(vy, wz)`` —— 「人在遥感操控」的机制化。

        vy 按机身偏离台阶中线（``center_y``）的距离**反号**拉回；wz 按朝向误差
        对准课程方向（+x，即 wrap(−yaw)）。增益与限幅见
        ``EnvConfig.guidance_*``。只服务楼梯级（平地级的指令跟随任务照旧用
        课程采样范围）。
        """
        e_y = float(self.base_pos[1]) - float(self.stair.center_y)
        e_yaw = float(np.arctan2(np.sin(-self.base_rpy[2]), np.cos(-self.base_rpy[2])))
        vy = float(np.clip(-self.cfg.guidance_vy_gain * e_y,
                           -self.cfg.guidance_vy_limit, self.cfg.guidance_vy_limit))
        wz = float(np.clip(self.cfg.guidance_yaw_gain * e_yaw,
                           -self.cfg.guidance_yaw_limit, self.cfg.guidance_yaw_limit))
        return vy, wz

    def command_body(self) -> np.ndarray:
        """世界系指令的**机体系投影** ``[vx_b, vy_b, wz]``。

        指令由上层在世界系给定（vx 沿课程方向 +x，vy 为世界横向）；策略的执行器
        只产生机体系速度，所以奖励的比较、观测的呈现、脱靶判据用的都是这个投影。
        转换只绕竖直轴（yaw）：`[vx_b, vy_b] = R(yaw)ᵀ·[vx, vy]`，wz 两系同值。

        为什么统一到机体系：世界系指令 + 机体系投影后，「掉头背对指令方向走开」
        会立即表现为机体系误差（指令说「你机体系后退」），无需再按地形切换参照系
        —— §21.14 的地形分支由此统一（experience.md §23.9）。
        """
        yaw = float(self.base_rpy[2])
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c * self._command[0] + s * self._command[1],
                         -s * self._command[0] + c * self._command[1],
                         float(self._command[2])])

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

        指令用机体系投影（``command_body``）；横向/偏航实测只读**滤波后**的量
        （瞬时值被步态摆动主导，等于在判步态相位）；前向用机身系瞬时速度。
        """
        cb = self.command_body()
        err_vx = abs(float(self.base_lin_vel_body[0]) - float(cb[0]))
        err_vy = abs(self.filtered_lateral_vel - float(cb[1]))
        err_wz = abs(self.filtered_yaw_rate - float(cb[2]))
        return bool(err_vy > self.cfg.flat_track_tol_vy
                    or err_wz > self.cfg.flat_track_tol_wz
                    or err_vx > self.cfg.flat_track_tol_vx)

    def _record_tracking_stats(self) -> None:
        """累加三轴「指令 vs 实测」的一阶/二阶矩（只在 ``_step_count > 0`` 时调：
        第 0 步的速度还是 reset 后的零向量，混进去会压低相关系数）。"""
        act = np.array([float(self.base_lin_vel_body[0]),
                        float(self.filtered_lateral_vel),
                        float(self.filtered_yaw_rate)])
        cmd = self.command_body()
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
            self._last_traj,
        ]
        # 结构固定块：机身系线速度（指令跟踪的「实测」侧，无开关）。
        parts.append(self.base_lin_vel_body)
        # 这里故意没有 base_height 分支：它含地形真值，ObsConfig 构造期已禁止。
        if obs_cfg.gait_phase:
            # 报「产生器实际在用的那个相角」（residual=先验相位；gait_params=生成器
            # 时钟）—— 观测与动作流水线必须同一个相位，否则破坏马尔可夫性。
            theta = self._gait_theta()
            parts.append(np.array([np.sin(theta), np.cos(theta)]))
        if obs_cfg.step_height:
            # 楼梯信息里唯一直接告诉策略的一项。盲走仍然需要知道自己在*哪一段*
            # 楼梯上；但它不会知道下一级立面在哪里。
            parts.append(np.array([float(self.stair.step_height)]))
        # 结构固定块：三轴速度指令（「指令」侧；**机体系投影**，来自操作者/上游
        # 规划器的世界系指令经 :meth:`command_body` 转换，不含地形真值）。
        parts.append(self.command_body())
        if obs_cfg.vel_filter:
            # 滤波后的横向速度 / 偏航角速度。奖励的 command_follow 读同一批量 ——
            # 只改奖励不改观测会把滤波器状态藏起来、破坏马尔可夫性，必须成对出现。
            parts.append(np.array([self.filtered_lateral_vel, self.filtered_yaw_rate]))
        if obs_cfg.gait_cmd:
            # 步态指令条件（freq [Hz] + swing 缩放）：策略要伺服的两个步态参数，
            # 与 _gait_action 生成先验用的是同一组值（马尔可夫性）。
            parts.append(np.array([float(self._gait_freq_cmd),
                                   float(self._gait_swing_cmd)]))
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

    def _scan_foot_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """扫一遍接触列表，返回 ``(法向力 [N], 切向力模长 [N])``，各 4 维。

        与 :meth:`_contact_scan` 扫的是同一批接触对，但**故意不合并**：力只在
        ``cfg.needs_foot_force`` 为真时才需要（默认关），合并进去等于让每一次
        训练步都白付一趟。真要让力默认开启，再合并也不迟。

        力取自 ``mujoco.mj_contactForce`` 的 6 维接触力，``res[0]`` 是法向分量、
        ``res[1:3]`` 是切向。一只脚可能同时有多个接触点（球-面多点），逐点求和；
        法向取绝对值只为防符号约定变化，正常解算下它就是压缩力。
        """
        foot_index = {int(g): k for k, g in enumerate(self.meta.foot_geom_ids)}
        terrain = {int(self.meta.floor_geom_id)} | {
            int(g) for g in self.meta.stair_geom_ids
        }
        nfoot = len(self.meta.foot_geom_ids)
        fn = np.zeros(nfoot)
        ft = np.zeros(nfoot)
        res = np.zeros(6, dtype=np.float64)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in terrain and g2 not in terrain:
                other = g2
            elif g2 in terrain and g1 not in terrain:
                other = g1
            else:
                continue
            k = foot_index.get(other)
            if k is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, res)
            fn[k] += abs(float(res[0]))
            ft[k] += float(np.hypot(res[1], res[2]))
        return fn, ft

    @property
    def foot_normal_force(self) -> np.ndarray:
        """(4,) 每只足端在本控制步**末**受到的**法向**接触力 [N]（无接触为 0）。

        **策略不可见的特权量**（不进观测）：`foot_scuff` 的滑擦判据、`load_share`
        的负载占比、`com_margin` 的力加权支撑中心都读它。真机上对应足端力传感器
        或关节力矩残差估计，所以它不是仿真专属信息，只是暂不进观测。
        """
        return self._foot_fn

    @property
    def foot_tangential_force(self) -> np.ndarray:
        """(4,) 与 :attr:`foot_normal_force` 同刻的**切向**力模长 [N]。"""
        return self._foot_ft

    @property
    def foot_impact_force(self) -> np.ndarray:
        """(4,) 本控制步内**逐物理子步**取到的法向力峰值 [N]。

        为什么不直接用步末读数：控制步是 20 ms 而物理步是 2 ms，落地冲击在头
        两个物理子步里就被柔顺接触吸收掉了，50 Hz 的步末读数**看不到真正的尖峰**。
        """
        return self._foot_fn_peak

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
        self._last_action = np.zeros(self.num_action)
        self._last_traj = np.zeros(self.num_act)
        self._cmd_hold[:] = self._sample_command()
        self._command[:] = list(self._cmd_hold)
        self._sample_gait_cmd()
        self._cmd_wander[:] = 0.0
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
        self._hold_phi = 0.0
        self._prev_foot_pos = self.foot_pos.copy()
        self._foot_air_time = np.zeros(len(self.meta.foot_geom_ids))
        self._prev_foot_contact = self._contact_scan()[2]
        self._foot_fn[:] = 0.0
        self._foot_ft[:] = 0.0
        self._foot_fn_peak[:] = 0.0
        # 步态先验时钟，单位为自回合开始起的秒数。观测里报告的相位就是本步
        # 实际使用的那一个。
        self._gait_t = 0.0
        self._gait_phase = 0.0
        self._gait_clock = 0.0
        self._gait_phase_cycles = 0.0
        self._prior_gate = 1.0
        self._vx_ema = 0.0
        self._gate_slow_streak = 0
        self._gate_has_moved = False
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

    def _active_gait(self) -> GaitParams:
        """当前地形的先验：楼梯级且配了 ``gait_stair`` 时用它，否则用 ``gait``。"""
        if self.stair.n_up > 0 and self.cfg.gait_stair is not None:
            return self.cfg.gait_stair
        return self.cfg.gait

    def _gait_action(self) -> np.ndarray:
        """步态先验在当前相位下的关节角偏移（rad）。

        先验被禁用时恒为零，于是动作流水线对 ``gait_prior=False``
        和对 ``GaitParams.static()`` 是同一条代码路径。``_prior_gate`` 是
        状态门控（受阻/站立时向 floor 衰减，见 ``EnvConfig.prior_gate_*``）。
        """
        if not self.cfg.gait_prior or self.cfg.gait_gain == 0.0:
            return np.zeros(self.num_act, dtype=np.float64)
        gait = self._active_gait()
        if self.cfg.obs.gait_cmd:
            # 条件化（Walk These Ways 式）：先验按**采样的 gait 指令**生成 ——
            # freq 换成指令值、幅值乘 swing（只缩幅值不动偏置，_scale_dyn 语义）。
            # freq 回合内恒定，秒制相位 2π·f·t 连续；副本每步现造（开销可忽略）。
            # 非条件化走旧路径，与 sub9_v5 逐位等价。
            scale = self._gait_scale() * float(self._gait_swing_cmd)
            gait = replace(gait, freq=float(self._gait_freq_cmd),
                           _scale_dyn=float(scale))
            return (self.cfg.gait_gain * self._prior_gate
                    * gait.radians(self._gait_theta()))
        scale = self._gait_scale()
        if scale != 1.0:
            gait = gait.with_dynamic_scale(scale)
        f_eff = self._gait_freq_eff()
        if f_eff != gait.freq:
            # 非条件化分档：楼梯级先验频率降到楼梯档中点（与 _gait_theta
            # 同源，保持马尔可夫性）；平地/频率一致时零开销。
            gait = replace(gait, freq=f_eff)
        return (self.cfg.gait_gain * self._prior_gate
                * gait.radians(self._gait_phase))

    def _gait_freq(self) -> float:
        """gait_params 模式的步频 [Hz]：楼梯更低（更从容的登踏），平地 1.67。"""
        return (float(self.cfg.gait_freq_stair) if self.stair.n_up > 0
                else float(self.cfg.gait_freq_flat))

    def _gait_freq_eff(self) -> float:
        """奖励与先验生成**共用**的当前实际步态频率 [Hz]（门控/air_time 等
        成形项以它做频率归一 —— 用错基准会让等价性悄悄破掉，§23.29）。

        与 :meth:`_gait_freq` 的分工：那个是 gait_params 时钟的地形常量。
        取值：gait_params → 地形常量；residual 条件化 → 采样指令；
        residual 非条件化 → **分档**（楼梯级 = stair_band 中点、平地 = 先验
        json 频率）。"""
        if self.cfg.action_mode == "gait_params":
            return self._gait_freq()
        if self.cfg.obs.gait_cmd:
            return float(self._gait_freq_cmd)
        if self.stair.n_up > 0 and self.cfg.gait_stair is not None:
            return float(self.cfg.gait_stair.freq)   # 楼梯专用先验自带频率
        if self.stair.n_up > 0:
            # 楼梯级频率：``gait_freq_stair_fixed > 0`` 时钉死该值（用户要求的
            # 「楼梯段固定 1.25 Hz」）；否则按台阶高线性插值 先验json 频率→1.0。
            if float(self.cfg.gait_freq_stair_fixed) > 0.0:
                return float(self.cfg.gait_freq_stair_fixed)
            h = float(self.stair.step_height)
            f_flat = float(self.cfg.gait.freq)
            return f_flat + (1.0 - f_flat) * min(1.0, h / 0.15)
        return float(self.cfg.gait.freq)

    def _gait_theta(self) -> float:
        """当前步态相角 [rad]。

        产生器用的与观测报的**必须是同一个**（马尔可夫性）：residual 模式取固定
        先验的 ``2π·f·t``；gait_params 模式取生成器时钟的 ``2π·周期数``。
        """
        if self.cfg.action_mode == "gait_params":
            return 2.0 * np.pi * self._gait_phase_cycles
        if self.cfg.obs.gait_cmd:
            # 条件化：residual 的相位跟随**采样的步态频率**（回合内恒定 → 秒制
            # 相位连续；与 _gait_action 用同一个值，保持马尔可夫性）。
            return 2.0 * np.pi * self._gait_freq_cmd * self._gait_phase
        # 非条件化：频率与 _gait_action 的先验生成同源（分档后楼梯级为
        # 楼梯档中点）。
        return 2.0 * np.pi * self._gait_freq_eff() * self._gait_phase

    def _gait_traj_from_params(self, action: np.ndarray) -> np.ndarray:
        """P2：8 维步态参数 → 12 维关节角偏移（已乘先验 ``action_scale``）。

        ``action = [前对(amp_th, amp_ca, off_th, off_ca, hip_amp, hip_off), 后对(同六项)]``：
        每项以 ``gait_param_nominal``（标称步态）为**零点**做双侧线性映射 ——
        ``a < 0`` 压向 0（幅值 0 = 该腿对驻留）、``a > 0`` 放到 ``param_*_max``
        （楼梯上幅值上限再乘 ``stair_stride_boost``）。髋的标称是 0（artifact
        形态，髋不摆），幅值单侧、偏置双侧 —— **偏航/横移的作动权限在髋上**，
        没有这四项 P2 在结构上就删掉了转向能力（§23.25）。相位用固定的 trot
        对角反相（``gait.leg_phase``），波形/相位偏置沿用先验结构。

        为什么按**前/后对**而不是对角对：对角对把前腿与后腿锁在一起，无法表达
        受阻态需要的「前脚驻留 + 后腿抬升」（前对幅值→0、后对幅值拉大）——
        这是 P2 相对固定先验的核心增益（§23.17）。

        为什么零点是**标称步态**而不是全零：全零 = ``off_th=0`` = 原地踏步
        （§23.6 标定结论；§23.24 实测从零学不动）。以会走的步态为零点，
        策略只需学「怎么改」——与 residual 模式同构，但改的是**结构化的参数**。
        """
        g = self.cfg.gait
        nom = self.cfg.gait_param_nominal          # (amp_th, amp_ca, off_th, off_ca)
        amp_max = float(self.cfg.param_amp_max)
        if self.stair.n_up > 0:
            amp_max *= float(self.cfg.stair_stride_boost)
        off_max = float(self.cfg.param_off_max)
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        def amp_map(ai: float, nominal: float) -> float:
            return nominal * (1.0 + ai) if ai < 0 else nominal + ai * (amp_max - nominal)

        def off_map(ai: float, nominal: float) -> float:
            return nominal + ai * (off_max - nominal if ai > 0 else off_max + nominal)

        amp_th = np.array([amp_map(a[0], nom[0]), amp_map(a[6], nom[0])])   # 前对/后对
        amp_ca = np.array([amp_map(a[1], nom[1]), amp_map(a[7], nom[1])])
        off_th = np.array([off_map(a[2], nom[2]), off_map(a[8], nom[2])])
        off_ca = np.array([off_map(a[3], nom[3]), off_map(a[9], nom[3])])
        # 髋：标称 0（artifact 形态，髋不摆）。幅值用**有符号**形式（负号 = 反相
        # 摆动，语义等价、量程不浪费）；偏置双侧 ±hip_off_max。
        hip_amp = np.array([self.cfg.param_hip_amp_max * a[4],
                            self.cfg.param_hip_amp_max * a[10]])
        hip_off = np.array([self.cfg.param_hip_off_max * a[5],
                            self.cfg.param_hip_off_max * a[11]])
        theta = self._gait_theta()
        traj = np.zeros((4, 3), dtype=np.float64)       # 每条腿 hip/thigh/calf
        for leg in range(4):
            pair = 0 if leg < 2 else 1                  # FL/FR = 前对；RL/RR = 后对
            p = theta + float(g.leg_phase[leg])
            traj[leg, 0] = hip_off[pair] + hip_amp[pair] * np.sin(p)   # 髋摆（横移/转向权限）
            traj[leg, 1] = off_th[pair] + amp_th[pair] * np.sin(p + g.ph_th)
            traj[leg, 2] = off_ca[pair] + amp_ca[pair] * g._calf_wave(p + g.ph_ca)
        return float(g.action_scale) * traj.ravel()

    def _target_from_action(self, action: np.ndarray) -> np.ndarray:
        """把策略动作映射为 PD 关节目标（整条动作流水线）。

        **residual 模式**（12 维）：固定先验 + 策略残差，两个幅值刻意分开 ——

            target = home + gait_gain * gait.radians(t) + action_scale * action

        ``gait.action_scale`` 表示先验动作有多大；``cfg.action_scale`` 表示策略
        可以偏离它多少。共用一个尺度会让探索噪声随先验一起缩放，瞬间失稳。

        **gait_params 模式**（8 维，P2）：动作**就是**步态参数，经 sin 结构 +
        trot 相位生成轨迹 —— ``target = home + traj(params, θ)``。结构提供协调性
        （随机探索也落在协调步态上），参数的前/后分离提供受阻态的表达力。

        （历史上有过 "absolute" 模式，从未被使用，已随清理删除。）
        """
        if self.cfg.action_mode == "gait_params":
            traj = self._gait_traj_from_params(action)
            self._last_traj = traj.copy()               # 供观测的「上一步下发偏移」
            return np.clip(self.home_angles + traj, self.act_low, self.act_high)
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
            self._cmd_hold[:] = self._sample_command()
        # 指令缓变扰动（OU 过程）：模拟真实操作手的轻微无规律输入。只在未钉住
        # （训练流）时生效；回放/遥操作 hold=True 完全不受影响。
        if self._cmd_age < self.cfg.command_hold_steps and self.cfg.command_wander_std > 0:
            dt = self.cfg.control_dt
            self._cmd_wander *= float(np.exp(-dt / self.cfg.command_wander_tau))
            self._cmd_wander += (self.cfg.command_wander_std * np.sqrt(dt)
                                 * self._rng.standard_normal(3))
            np.clip(self._cmd_wander, -0.08, 0.08, out=self._cmd_wander)
            self._command[:] = (self._cmd_hold + self._cmd_wander).tolist()
        # 楼梯段闭环外部引导：vy/wz 由引导控制器按状态生成（拉回中线 + 对准
        # 航向），vx 保留课程采样；缓变扰动仍叠加在引导值上（真实操作手也有
        # 手抖）。平地级不受影响。
        if self.stair.n_up > 0 and self.cfg.stair_guidance:
            vy_g, wz_g = self._guidance_command()
            self._command[1] = float(np.clip(vy_g + self._cmd_wander[1], -0.2, 0.2))
            self._command[2] = float(np.clip(wz_g + self._cmd_wander[2], -0.8, 0.8))

        # 先验状态门控（P1）：受阻 = 「有前进指令但**平滑后**的前向速度持续
        # 接近零」——不能直接用 ``_blocked``（按瞬时速度判定，步态摆动下每秒
        # 闪烁多次）。三个防误触发参数都是实测标定的（experience.md §23.18）：
        # EMA τ=0.8s 压掉整个步态周期（2 Hz 下的摆动残余 ±0.03）、阈值
        # 0.15×cmd 与正常行走速度（≈0.5×cmd）留足余量、40 步（0.8s）去抖让
        # 出生静止与行进中的短暂减速都不触发；否则门控会把平地正常行走也
        # 关掉先验，而先验一关速度归零 → 自锁死。
        if self.cfg.prior_gate_on_block:
            a_v = min(1.0, self.cfg.control_dt / 0.8)
            self._vx_ema += a_v * (float(self.base_lin_vel_body[0]) - self._vx_ema)
            need = 0.15 * max(abs(float(self._command[0])), 0.05)
            self._gate_slow_streak = (self._gate_slow_streak + 1
                                      if self._vx_ema < need else 0)
            # 「先动过」标志（冷启动自锁的教训，experience.md §23.18）：门控只
            # 衰减在「先行走、后被卡住」的情形（用户原意：爬楼途中被卡住时先验
            # 别乱动）；从没动过的策略必须先靠先验起步，否则一衰减就永久冻住。
            if self._vx_ema > 0.5 * max(abs(float(self._command[0])), 0.05):
                self._gate_has_moved = True
        stand = (self.cfg.prior_gate_on_stand
                 and abs(float(self._command[0])) < 1e-3)
        # 受阻是**楼梯现象**且要过**出生宽限**与**先动过**两道门（平地滞留不是
        # 受阻；出生加速期的 EMA 尚未爬到行走速度）。
        blocked_gate = (self.cfg.prior_gate_on_block
                        and self.stair.n_up > 0
                        and self._step_count > 75
                        and self._gate_has_moved
                        and self._gate_slow_streak >= 40)
        goal = self.cfg.prior_gate_floor if (stand or blocked_gate) else 1.0
        alpha = min(1.0, self.cfg.control_dt
                    / max(1e-6, self.cfg.prior_gate_tau))
        self._prior_gate += alpha * (goal - self._prior_gate)

        # 残差 RL：先验提供一个能用的周期步态，策略提供修正量。观测里报告的
        # 相位就是本步实际使用的那一个，所以转移保持马尔可夫性。
        self._gait_phase = self._gait_t
        self._gait_phase_cycles = self._gait_clock
        target = self._target_from_action(action)
        self._gait_t += self.cfg.control_dt
        self._gait_clock += self.cfg.control_dt * self._gait_freq()

        need_force = self.cfg.needs_foot_force
        if need_force:
            self._foot_fn_peak[:] = 0.0
        for _ in range(self.cfg.decimation):
            q = self.data.qpos[self.meta.qpos_joint_adr]
            qd = self.data.qvel[self.meta.qvel_joint_adr]
            tau = self._kp * (target - q) - self._kd * qd
            self.data.ctrl[:] = np.clip(
                tau, self.meta.ctrl_ranges[:, 0], self.meta.ctrl_ranges[:, 1]
            )
            mujoco.mj_step(self.model, self.data)
            if need_force:
                # 逐物理子步取力：峰值只在子步分辨率下看得见（见 foot_impact_force）。
                fn_sub, ft_sub = self._scan_foot_forces()
                np.maximum(self._foot_fn_peak, fn_sub, out=self._foot_fn_peak)
                self._foot_fn[:] = fn_sub
                self._foot_ft[:] = ft_sub

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
        # gait_params 模式下 _last_traj 已在 _target_from_action 里设为生成的轨迹；
        # residual 模式下它就是本步 action（下发的关节偏移）。
        if self.cfg.action_mode == "residual":
            self._last_traj = action.copy()
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
            "steps_mounted": int(round(
                self._peak_terrain_height / max(self.stair.step_height, 1e-9))),
            "gait_prior": bool(self.cfg.gait_prior),
        }
        # 足端力遥测（2026-10-17）：只记录，不改奖励。标定 R1~R4 的阈值靠它 ——
        # 没有分布就设阈值 = 拍脑袋（见 doc/contact_reward_design.md §2.2）。
        if self.cfg.needs_foot_force and self.cfg.reward.contact_telemetry:
            info["foot_force"] = {
                "fn": [float(v) for v in self._foot_fn],
                "ft": [float(v) for v in self._foot_ft],
                "peak": [float(v) for v in self._foot_fn_peak],
                "contact": foot_contact.astype(np.int8).tolist(),
            }
        return self._observation(), float(reward), terminated, truncated, info

    # ---------------------------------------------------------------- 渲染

    #: :meth:`render` 能识别的相机预设。
    CAMERAS = ("side", "track", "chase", "front", "top", "side_high")

    #: 各相机的 ``(azimuth, elevation, distance)``。注视点在 :meth:`render` 里
    #: 按相机名取：``side`` 固定在楼梯入口，其余跟随机身。
    _CAMERA_VIEW = {
        "side": (90.0, -12.0, 4.0),
        "track": (90.0, -9.0, 2.5),
        "chase": (135.0, -22.0, 3.0),
        "front": (180.0, -10.0, 3.0),
        "top": (90.0, -85.0, 4.5),
        # 审查用:侧上方 30° 俯视——立面与踏面同现(side 的 -12° 俯角低于
        # 台阶立面可见临界 atan(立面/踏面)≈13.5°,立面被踏面遮挡,§23.30)
        "side_high": (90.0, -30.0, 5.0),
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
            "side_high": (self.stair.start_x + 1.2, 0.0, 0.5),
            # 视频用：沿前进方向平移跟随 + z 跟随机身高度(y 钉楼梯中线)——
            # z 钉死会在爬高后把台阶压扁成「一块平地」(立面全被踏面遮挡,
            # 2026-10-15 用户发现);机身 z 比足端平稳,不引入步态抖动。
            "track": (float(base[0]) + 0.35, 0.0, float(base[2])),
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
