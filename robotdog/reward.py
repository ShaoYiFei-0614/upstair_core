"""奖励函数：本仓库所有奖励数学的唯一来源（设计说明见 ``doc/reward.md``）。

本模块由三部分构成：

* :class:`RewardConfig` —— 全部权重与阈值（``config.yaml`` 的 ``reward`` 段）。
* :func:`compute_reward` —— 奖励数学本体：输入是环境状态，输出是**逐步奖励**、
  **逐项明细**与**终止奖励**，不做任何物理推进。它**不是纯函数** —— 几个势函数
  项要靠环境上的记忆字段（``_peak_terrain_height`` / ``_probe_best`` /
  ``_probe_count``）做到「总量有界、不可刷分」，所以会写这几个字段，**不能为了
  对比而重复调用**。
* :func:`track_swing_peak` / :func:`update_probe_state` —— ``env.step`` 在奖励
  结算**之前**调用的两个势函数状态机（摆动峰值记录、「被阻挡」判定）。

``terms`` 里每一项都已经乘过权重，``sum(terms.values())`` 就是未经终止奖励的逐步
奖励（``env`` 侧把终止奖励并进同一张表），所以逐项拆解可以安全地相加对账。

调用点只有一个：``robotdog/env.py`` 的 ``Go2StairsEnv.step``。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotdog.tactile import LANDING, PRELANDING, STANCE

# --------------------------------------------------------------------------- #
# 权重
# --------------------------------------------------------------------------- #

@dataclass
class RewardConfig:
    """奖励权重。设计依据与「每个权重堵住哪种失败」见 ``doc/reward.md``
    与 ``experience.md``；这里只保留语义。"""

    # --- 指令跟踪 ---------------------------------------------------------- #
    forward_vel: float = 10.0           # 对 vx 指令的绝对误差窗口跟踪
    forward_overspeed: float = 1.0      # 超速侧额外惩罚斜率（× forward_vel 量级）
    overspeed_range: float = 0.45       # m/s，超速惩罚的斜率量程
    forward_track_window: float = 0.25  # m/s，绝对误差窗口（必须低于最窄指令带宽度）
    track_tail: float = 1.0             # 死区外线性尾项系数（越界仍有回头梯度）
    overspeed_cap: float = 2.0          # 超速惩罚倍率上界（× forward_vel）
    orientation: float = -1.2           # roll^2 + pitch^2（平方和）
    torque: float = -1.5e-4             # sum(tau^2)（关节力矩平方和）
    foot_lift: float = 0.9              # 摆动期离地间隙成形
    body_collision: float = -1.2        # 躯干接触地形（逐步、有界，不终止回合）

    # --- 横向 / 旋转指令跟踪 ------------------------------------------------ #
    lateral_vel: float = 6.0            # 对 vy 指令的线性死区跟踪（恒在，不门控）
    # 横向窗口下限 [m/s]：window = 0.5·max(vy_max, lateral_range_floor)。
    # 与偏航通道对称去门控（vy_max=0 的级别也要有「不许横漂」的价钱）；
    # 门控级别里漂移自由生长的教训见 experience.md §23.10。
    lateral_range_floor: float = 0.15
    yaw_rate: float = 4.0               # 对 wz 指令的线性死区跟踪（恒在，不门控）
    yaw_range_floor: float = 0.30       # yaw_max=0 时的量程下限 [rad/s]，兼防除零
    # 零偏航指令的基线惩罚放大系数：cmd.wz == 0 时误差按增益放大，否则
    # 「命令我不许转」没有任何价钱（experience.md §23.8）。
    yaw_zero_gain: float = 1.5

    # --- 摆动足抬升 ---------------------------------------------------------- #
    foot_lift_target: float = 0.08      # 目标值的下限，而不是完整的目标值
    foot_lift_step_ratio: float = 1.4   # 有效目标 = max(target, ratio × 台阶高)
    foot_lift_gate_speed: float = 0.25  # m/s，三维足端速度门限

    # --- 被阻挡时的试探抬腿（势函数塑形，见 doc/reward.md） ------------------ #
    probe_lift: float = 1.0             # 抬高进展的权重
    probe_lift_target: float = 0.06     # m，起算线（跟策略当前能力比，不跟台阶高走）
    probe_blocked_frac: float = 0.30    # 实测前进速度 < 该比例 × 指令速度 ⇒ 被阻挡
    # 低足门控开关（默认关，2026-10-14 消融，experience.md §23.27）：关 =
    # 全部脚结算（sub9_v5 行为）；开启必须过 stair3 250k 对照关。
    probe_lift_low_foot_only: bool = False
    # 受阻态驻留势函数（见 doc/reward.md）：Φ = 台面脚数 × 台阶高，按 ΔΦ
    # 结算。⚠ 默认 0（2026-10-14 消融，experience.md §23.27）：与 stair3
    # 探索期的「受阻→抬脚再爬」转移对打，双臂 A/B 楼梯全灭的条件之一；
    # 重引入必须过 stair3 250k 对照关。
    foot_hold: float = 0.0
    hold_elev_frac: float = 0.3         # 判定「在台面上」的高度阈值（× 台阶高）
    # 俯仰姿态惩罚：只罚抬头侧。⚠ 默认 0（2026-10-14 实测否决，experience.md
    # §23.27）：死区覆盖不住健康姿态分布（平地 −7°±5.7°），flat_4 76%→18%；
    # 该形式在平地无解，平地抬头改用课程顺序解。字段留作楼梯级实验位。
    pitch_posture: float = 0.0
    pitch_tol_flat: float = 0.14        # rad，平地死区
    pitch_tol_per_m: float = 1.2        # rad/m，随台阶高放宽
    pitch_cap: float = 0.4              # rad，尾项上限（惩罚有界）

    # --- 步态发现 ------------------------------------------------------------ #
    air_time: float = 0.6
    air_time_target: float = 0.30
    air_time_max: float = 0.60

    # --- 接触 / 足端力奖励（2026-10-17 新增，设计见 doc/contact_reward_design.md）- #
    # ⚠ **默认全关**：字段默认值让「导入即等于改动前」——`contact_rewards=False`
    # 时下面这些键连 `terms` 都不进，逐步奖励逐位不变（`.tmp/regress_reward.py`）。
    # 打开哪一项就把哪一项的权重给非 0；`set_reward_overrides` 按 `hasattr` 就地
    # 覆盖，所以权重字段不用改任何课程/训练代码就能逐级下发。
    contact_rewards: bool = False       # ★ 总开关（关 = 整组不结算、不进 terms）
    contact_telemetry: bool = False     # 只把足端力写进 info，不改奖励（标定时用）

    foot_scuff: float = 0.0             # R1 滑擦惩罚（负权重）
    scuff_v_ref: float = 0.15           # 判「在切向滑动」的足端速度门 [m/s]
    scuff_mu_soft: float = 0.35         # 切向/法向比死区（低于它不算滑擦）
    scuff_mu_hard: float = 0.80         # 该比值达到它时罚满

    foot_impact: float = 0.0            # R2 落地冲击惩罚（负权重）
    # ⚠ 阈值默认值 = **标定值**（2026-10-17，§2.2 ③）：3.0×体重 = 落在落地尖峰
    #   分布的 p50 与 p90 之间，只罚真砸下去的。**别把 tol 留成 0** —— 代码里
    #   没有「0 = 用标定值」的兜底，tol=0/scale=1 会让每一项都瞬间罚满，
    #   等于每步恒扣满权重（踩过这个坑就会白烧一整轮训练）。
    impact_force_tol: float = 450.0     # N ≈ 3.0 W（W = 149.2 N）
    impact_force_scale: float = 450.0   # N，罚满所需的超出量（≈ 6 W 罚满）

    load_share: float = 0.0             # R3 负载分担（负权重）
    # ⚠ 默认 = 标定值（§2.2 ④）：平地实测双足帧 p50 = 0.65 ⇒ cap 取 0.70（≈p75）
    #   才只罚明确偏载；原设 0.55 会罚掉 79~84% 的双足帧。
    load_share_cap: float = 0.70        # 单足法向力占比上限
    # ⚠ 原设 3 已被推翻：平地三足着地只占 **6.2%**（§2.2 ①），门控等于常年关闭。
    #   平地是对角双足步态（双足帧 30.3%），所以下限取 2。
    load_share_min_feet: int = 2        # 着地足数少于它就不结算（否则奖励站桩）
    load_share_flat_only: bool = True   # 楼梯上跨级站立天然不均，默认只罚平地

    com_margin: float = 0.0             # R4 质心支撑裕度（负权重）
    # ⚠ 默认 = 标定值（§2.2 ⑤）：实测 p50 = 0.16 m、p90 = 0.25 m ⇒ 原设 0.06
    #   让 83% 的帧直接罚满，既无区分度又过大。tol 取 ≈p50，scale 让 0.30 m 罚满。
    com_margin_tol: float = 0.15        # m，机身投影到力加权支撑中心的允许偏差
    com_margin_scale: float = 0.15      # m，罚满所需的超出量（tol+scale = 0.30 m ≈ p97）
    com_margin_flat_only: bool = True   # 楼梯的支撑多边形是跨级的，默认只罚平地

    # --- 足底触觉相位条件策略（TactileStep，2026-10-20） --------------------- #
    # ⚠ **默认全 0**：三个策略都不结算、键不进 terms，逐步奖励逐位不变。
    #   打开前必须 env.tactile.enabled=True（Go2StairsEnv.__init__ 会校验）。
    #   数据面（F̄/Ā/CoP/四相位）在 robotdog/tactile.py，设计见 doc/tactile.md。
    tactile_prelanding: float = 0.0     # ① 落地前**向下**的足端速度惩罚（负权重）
    prelanding_vz_ref: float = 0.15     # m/s，归一化参考（罚满 = 该向下速度）
    tactile_impact: float = 0.0         # ② 落地冲击惩罚（负权重）
    impact_force_ref: float = 3.0       # × 体重，LANDING 窗内平均法向力的参考量级
    impact_delta_ref: float = 1.5       # × 体重/步，力变化的参考量级
    impact_peak_ref: float = 6.0        # × 体重，LANDING 窗内峰值的参考量级
    tactile_stance: float = 0.0         # ③ 支撑质量（面积 + CoP 裕度，正权重）
    stance_area_target: float = 0.60    # Ā 目标：达到它面积项封顶（避免追求满掌）
    stance_cop_jump_ref: float = 0.05   # 归一化 CoP 每步跳变的参考尺度

    # --- L1-stance 修正（2026-10-21） ---------------------------------------- #
    # 背景：只加 stance 正奖（Level1 的 l1_st）收敛到「蹲稳不动」—— stance 项
    # 按帧发奖、不要求前进，且零指令通道（vy_max=0 / wz=0）的跟踪项每步白给
    # +lateral_vel+yaw_rate。三项默认**中性/关闭**，逐步奖励逐位不变。
    # 设计说明见 doc/introduce_tactile.md §9。
    follow_zero_baseline: bool = False  # True：零指令通道按「完美跟踪=0 分」计
    # ⚠ 「零指令」必须用**死区**判定，不能对 ``cmd_body`` 做 ``== 0.0``：
    #   cmd_body 含 command_wander_std≈0.03 的 OU 扰动与 stair_guidance 注入的
    #   vy/wz（env.py 每步覆盖），楼梯级几乎恒非 0（reviewer/D 审计 2026-10-21）。
    #   死区取「比扰动大一档、比真实指令小一档」：横向 0.05 m/s、偏航 0.10 rad/s。
    zero_cmd_eps_vy: float = 0.05       # m/s，|横向指令| < 该值视为「零指令通道」
    zero_cmd_eps_wz: float = 0.10       # rad/s，|偏航指令| < 该值视为「零指令通道」
    stance_min_vx: float = 0.0          # m/s；>0 时 stance 奖励 × clip(vx/min_vx,0,1)
    tactile_slip: float = 0.0           # 支撑相足端水平滑移惩罚（负权重）
    slip_ref: float = 0.005             # m/步，滑移罚满参考（实测支撑相≈2–3 mm/步）

    # --- 辅助成形 ------------------------------------------------------------ #
    base_height: float = -2.0
    joint_vel: float = -2.0e-5
    action_rate: float = -0.02
    joint_limit: float = -0.5
    alive: float = 0.15
    termination: float = -15.0          # 摔倒一次性罚（config.yaml 用远大于此的值）
    # 摔倒罚的**进度分级**（experience.md §19.2/§23.7）：终止罚按「本回合已
    # 爬升高度占比」线性减免 —— 高处的尝试已被 stair_progress 逐级支付，
    # 全额罚会把「高处的尝试」罚得比「脚下空耗」还重。平地/0 高度时 frac=0。
    termination_progress_relief: float = 0.5
    success: float = 50.0

    # --- 爬升进展 ------------------------------------------------------------ #
    stair_progress: float = 3.0         # 只在刷新本回合最高支撑面时一次性付
    leg_collision: float = -0.25


# --------------------------------------------------------------------------- #
# 计算
# --------------------------------------------------------------------------- #


def track_shape(x: float, tail: float) -> float:
    """指令跟踪的共同形状：``x = |误差| / 量程`` 的单调递减函数。

    ``x <= 1`` 是线性死区 ``1 - x``；``x > 1`` 换成线性尾项 ``-tail * (x - 1)``。
    契约三条：在 ``x = 1`` 处连续（两侧都是 0）、全程单调递减（尾项必须是**线性**
    而非二次 —— 二次减项在 ``x > 1`` 后先增后减，等于把误差推向更大）、惩罚有
    下界（调用处 ``max(-1.0, ·)`` 兜底）。反面案例见 ``experience.md`` §13。
    """
    if x <= 1.0:
        return 1.0 - x
    return -tail * (x - 1.0)


def compute_reward(env, action: np.ndarray, trunk_contact: bool,
                   foot_contact: np.ndarray | None = None,
                   leg_contact: bool = False,
                   terminated: bool = False
                   ) -> tuple[float, dict[str, float], float]:
    """逐步奖励 + 逐项明细 + 终止奖励。

    **不是纯函数**：读 ``env`` 的状态，也推进势函数记忆字段（见模块 docstring），
    同样的 ``(env 状态, action)`` 在不同时刻可能给出不同结果，**不能为了对比而
    重复调用**。

    :param env: ``Go2StairsEnv``。
    :param action: 本控制步的策略动作（12 维）。
    :param trunk_contact: 躯干是否接触地形。
    :param foot_contact: 逐足接触掩码；``None`` 时现算（只在 ``air_time`` 非零时用到）。
    :param leg_contact: 胫部/膝部是否蹭到立面。
    :param terminated: 本步是否回合终止（成功或摔倒）。
    :returns: ``(本步奖励, 逐项明细, 终止奖励)``。终止奖励单列，因为它取决于
        「是成功登顶还是摔倒」，那是 ``step`` 的判定，不属于逐项成形。
    各项的设计依据见 ``doc/reward.md`` 与 ``experience.md``。
    """
    rc = env.cfg.reward
    cmd = env.cfg.command
    rpy = env.base_rpy
    # 指令在世界系给定，统一投影到机体系再比较（``env.command_body()``）：执行器
    # 只产生机体系速度，而世界系指令在机身转向后含义不变 —— 「掉头走开」会立即
    # 表现为机体系误差，§21.14 的地形参照分支由此统一（experience.md §23.9）。
    # 「指令跟踪读滤波量还是瞬时量」由观测开关决定 —— 奖励与观测必须指同一个数。
    v_fwd = float(env.base_lin_vel_body[0])
    cmd_body = env.command_body()
    target_height = env.nominal_pose
    vel_filter = bool(env.cfg.obs.vel_filter)

    terms: dict[str, float] = {}

    # 1) 前进速度跟踪：绝对误差窗口上的线性死区 + 超速侧额外惩罚（有倍率上界），
    #    两处都有界。目标取**本步真实采样的指令** ``env._command[0]``，不是
    #    ``cmd.vx`` —— 那只是采样范围为空时的回退定值，课程下发的是范围。
    err = v_fwd - float(cmd_body[0])
    track = max(-1.0, track_shape(abs(err) / float(rc.forward_track_window),
                                  float(rc.track_tail)))
    if err > 0.0:
        over = min(
            float(rc.forward_overspeed) * err / float(rc.overspeed_range),
            float(rc.overspeed_cap),
        )
        track -= over
    terms["forward_vel"] = rc.forward_vel * float(track)

    # 1b) 横向 / 偏航指令跟踪：形状同前进项，窗口 = 0.5 × max(量程, 下限)。
    # 两通道都**恒在**（横向与偏航对称去门控 —— 门控级别里漂移自由生长的
    # 教训见 experience.md §23.10）。两项**求和**而不是取 max：取 max 时
    # 误差小的那项会盖掉另一项的梯度。
    # 读**滤波后**的量（与观测的 vel_filter 块同一个数，保持马尔可夫性）。
    follow = 0.0
    tail = float(rc.track_tail)
    zero_base = bool(rc.follow_zero_baseline)
    window_vy = 0.5 * max(float(cmd.vy_max), float(rc.lateral_range_floor))
    if cmd.vy_max or rc.lateral_vel:
        vy = (env.filtered_lateral_vel if vel_filter else env.base_lin_vel_body[1])
        xy = abs(vy - float(cmd_body[1])) / window_vy
        track_vy = max(-1.0, track_shape(xy, tail))
        if zero_base and abs(float(cmd_body[1])) < float(rc.zero_cmd_eps_vy):
            # 零横向指令：完美跟踪记 0 分，偏差记负分（而不是每步白给 +lateral_vel）
            track_vy -= 1.0
        follow += rc.lateral_vel * track_vy
    # 偏航项**恒在**（不受 ``yaw_max`` 门控），量程取 ``max(yaw_max, yaw_range_floor)``。
    # 门控的代价实测过：``yaw_max == 0`` 的级别里没有任何一项读 wz，策略学会
    # 以 −0.608 rad/s **恒定自转**（`.tmp/zz_cmd_response.py` 扫 wz ∈ [−0.6, +0.6]
    # 七档，实测斜率 −0.0021、范围 −0.608~−0.606，即输出与指令逐位无关），
    # 而且这个转法在后续级别里冻住 —— 加进偏航指令后它仍在转。底噪 ``floor``
    # 保证「不转」在零指令级别也有价钱，同时避免 ``yaw_max == 0`` 时除零；
    # 加惩罚必须**有界**（下界 −1，即最差 −``yaw_rate``），理由同前进项；
    # ⚠ ``follow_zero_baseline=True`` 且该通道指令在死区内时再 −1 ⇒ 下界 −2。
    wz = env.filtered_yaw_rate if vel_filter else env.yaw_rate
    yaw_err = abs(wz - float(cmd_body[2]))
    # ``wz = 0`` 的级别补一个零指令基线惩罚：否则「命令我不许转」没有任何价钱
    # （放大系数见 ``RewardConfig.yaw_zero_gain``）。
    # 注意这里是「**级别量程**判零」（`command.yaw_max` 由课程下发，整级恒 0），
    # 与下面 `zero_cmd_eps_*` 的「**采样指令**死区」是两件事，不要合并。
    if cmd.wz == 0.0:
        yaw_err = rc.yaw_zero_gain * (yaw_err + abs(wz))
    yaw_range = 0.5 * max(float(cmd.yaw_max), float(rc.yaw_range_floor))
    track_yaw = max(-1.0, track_shape(yaw_err / yaw_range, tail))
    if zero_base and abs(float(cmd_body[2])) < float(rc.zero_cmd_eps_wz):
        track_yaw -= 1.0
    follow += rc.yaw_rate * track_yaw
    terms["command_follow"] = follow

    # 2) 姿态稳定奖励 -- 惩罚 roll 与 pitch -----------------------------
    terms["orientation"] = rc.orientation * float(rpy[0] ** 2 + rpy[1] ** 2)
    # 2b) 俯仰姿态：只罚**抬头**侧，死区随台阶高放宽（爬楼需要抬头，平地不需要）。
    pitch_tol = (rc.pitch_tol_flat
                 + rc.pitch_tol_per_m * float(env.stair.step_height))
    nose_up = max(0.0, -float(rpy[1]) - pitch_tol)
    terms["pitch_posture"] = rc.pitch_posture * min(nose_up, rc.pitch_cap)

    # 3) 关节力矩惩罚 -- 关节力矩平方和 ---------------------------------
    tau = env.data.actuator_force
    terms["torque"] = rc.torque * float(np.sum(np.square(tau)))

    # 4) 足端抬升奖励 -- 摆动足必须越过地形 -----------------------------
    feet = env.foot_pos
    foot_dt = env.cfg.control_dt
    # 三维足端速度：只看水平分量会奖励快速的贴地平扫，同时砍掉高台阶
    # 需要的慢速抬腿。
    foot_speed = np.linalg.norm(
        (feet - env._prev_foot_pos) / foot_dt, axis=1
    )
    # 门控阈值：平地随频率归一（低频慢摆同样开门，wtw7 高抬证据）、楼梯固定
    # 0.25（归一在楼梯让立面贴地蹭速达标 → 假梯度、stair3 崩；地形二分见
    # experience.md §23.29）。
    if env.stair.n_up > 0:
        gate_speed_eff = float(rc.foot_lift_gate_speed)
    else:
        gate_speed_eff = (float(rc.foot_lift_gate_speed)
                          * float(env._gait_freq_eff()) / 1.67)
    gate = np.clip(foot_speed / max(gate_speed_eff, 1e-6),
                   0.0, 1.0)
    # geom_xpos 是球心，所以要减去半径，才能得到足端表面与地形之间的
    # 真实间隙。
    clearance = env.foot_clearance
    # 目标随地形一起变高，但在 5 cm 处它与原来的常量相等，所以已经学会
    # 的简单阶段不会回退。
    lift_target = max(
        rc.foot_lift_target, rc.foot_lift_step_ratio * float(env.stair.step_height)
    )
    # 守卫：``foot_lift_target`` 与台阶高同时为 0 时目标为 0，除法会产生 nan
    # 并污染整步奖励（``np.clip(nan, 0, 1)`` 仍是 nan）。
    lift = np.clip(clearance / max(lift_target, 1e-6), 0.0, 1.0)
    terms["foot_lift"] = rc.foot_lift * float(np.mean(gate * lift))

    # 5) 机身碰撞惩罚 -- 躯干接触地形 -----------------------------------
    # 逐步行且有界：它随躯干擦地的时长成比例，而不是在终止步给一个
    # 布尔量。
    terms["body_collision"] = rc.body_collision * float(trunk_contact)
    # 胫部 / 膝部蹭到立面。逐步有界，而且关键是它*不*结束回合，所以
    # 机器人可以在高楼梯上继续尝试。
    terms["leg_collision"] = rc.leg_collision * float(leg_contact)

    # 6) 足端腾空时间 -- 步态成形（一次良好摆动后的落地） ---------------
    # ``env._foot_air_time`` 保存的是当前步*之前*累积的腾空时间，
    # ``env._prev_foot_contact`` 保存的是它之前的接触状态，所以这一项恰好在
    # 足端落地的那一步结算。
    if rc.air_time != 0.0:
        if foot_contact is None:
            foot_contact = env._contact_scan()[2]
        landed = foot_contact & ~env._prev_foot_contact
        air = env._foot_air_time
        # target 随频率缩放——**只在平地**（与门控二分对称，experience.md
        # §23.29）：楼梯级保持 0.30（sub9_v5 口径；wtw9/10 楼梯崩的未排除
        # 变量，按「楼梯保持 sub9_v5 口径」原则一并二分）。平地缩放让慢摆的
        # 高抬跨步重新拿得到滞空收益。
        if env.stair.n_up > 0:
            target = float(rc.air_time_target)
        else:
            target = rc.air_time_target * float(env._gait_freq_eff()) / 2.0
        credit = np.clip(air - target, 0.0, None)
        overrun = np.clip(air - rc.air_time_max, 0.0, None)
        terms["air_time"] = rc.air_time * float(
            np.sum((credit - overrun) * landed)
        )
    else:
        terms["air_time"] = 0.0

    # --- 辅助成形 -----------------------------------------------------
    terms["base_height"] = rc.base_height * float(
        (env.base_pos[2] - env.terrain_height_under_base - target_height) ** 2
    )
    terms["joint_vel"] = rc.joint_vel * float(np.sum(np.square(env.joint_vel)))
    # 相邻步动作差 ``‖a_t − a_{t-1}‖``：奖励结算发生在动作历史更新之前，
    # 所以 ``_last_action`` 此刻就是上一步的动作。
    terms["action_rate"] = rc.action_rate * float(
        np.sum(np.square(action - env._last_action))
    )
    lo, hi = env.act_low, env.act_high
    q = env.joint_pos
    over = np.clip(lo - q, 0, None) + np.clip(q - hi, 0, None)
    terms["joint_limit"] = rc.joint_limit * float(np.sum(over))
    terms["alive"] = rc.alive

    # --- 爬升进展：一次性结算，只在支撑面刷新本回合最高时付增量（站桩拿不到
    # 新的钱，同一级往返也不付两次）。参照面是**四足支撑面**而不是躯干 x ——
    # 「总量有界」不等于「不可刷分」，参照量必须是任务真正想度量的事件
    # （experience.md §21.13）。
    climbed = max(0.0, env._support_height() - env._peak_terrain_height)
    if climbed > 0.0:
        env._peak_terrain_height = env._support_height()
    terms["stair_progress"] = rc.stair_progress * float(climbed)

    # 8) 被阻挡时的试探抬腿：势函数塑形 —— 逐足结算「摆动峰值比该足历史最好
    # （地板取起算线 probe_lift_target）又高出多少」。总量有界、不可刷分、
    # 每一步抬得更高都有梯度；门控条件与「被阻挡」的判定见 ``update_probe_state``，
    # 为什么起算线不跟台阶高走见 experience.md §15。
    if rc.probe_lift != 0.0 and env._blocked:
        peaks = env._peak_swing_clearance
        # 只让**低面脚**（还没上到台面的那些，即需要爬升的腿）的峰值参与结算 ——
        # 台面上的脚不该再抬高（它们该驻留，见 foot_hold），旧写法把前脚刨立面的
        # 峰值也算进「抬高进展」，等于奖励了错的腿（§23.26）。
        terr = env._terrain_height(env.foot_pos[:, 0], env.foot_pos[:, 1])
        elev = terr > (env._support_height()
                       + rc.hold_elev_frac * float(env.stair.step_height))
        # 低足门控默认关（消融证据 §23.27）；开启必须过 stair3 250k 对照关。
        if rc.probe_lift_low_foot_only and elev.any():
            peaks = np.where(elev, 0.0, peaks)
        floor = np.maximum(env._probe_best, float(rc.probe_lift_target))
        gain = float(np.sum(np.clip(peaks - floor, 0.0, None)))
        if gain > 0.0:
            env._probe_best = np.maximum(env._probe_best, peaks)
            env._probe_count += 1
        terms["probe_lift"] = rc.probe_lift * gain
    else:
        terms["probe_lift"] = 0.0

    # 9) 受阻态驻留（势函数，见 RewardConfig.foot_hold）：受阻时，「踩在台面上且
    #    正触地」的脚数 × 台阶高 构成势 Φ，按 ΔΦ 结算。把台上的脚抬走要付代价，
    #    踩住/多踩一脚给分。非受阻期只同步 Φ（不结算），避免进出受阻时的假跳变。
    if rc.foot_hold != 0.0:
        if foot_contact is None:
            foot_contact = env._contact_scan()[2]
        terr_h = env._terrain_height(env.foot_pos[:, 0], env.foot_pos[:, 1])
        elev_h = terr_h > (env._support_height()
                           + rc.hold_elev_frac * float(env.stair.step_height))
        phi = (float(np.sum(foot_contact & elev_h))
               * float(env.stair.step_height))
        terms["foot_hold"] = (rc.foot_hold * (phi - env._hold_phi)
                              if env._blocked else 0.0)
        env._hold_phi = phi
    else:
        terms["foot_hold"] = 0.0

    # 10) 接触 / 足端力（2026-10-17 新增；设计见 doc/contact_reward_design.md）。
    #     **总开关关着就整段跳过**：这些键连 terms 都不进，逐步奖励逐位不变
    #     （回归见 `.tmp/regress_reward.py`）。四项共用的量只在这里取一次。
    #     力是**策略看不见的**（不进观测）：加观测维会让现有全部权重无法
    #     warm-start，等于把 3→15cm 整条链作废。
    if rc.contact_rewards and (rc.foot_scuff or rc.foot_impact
                               or rc.load_share or rc.com_margin):
        fn = env.foot_normal_force            # (4,) 本控制步末的法向力 [N]
        ft = env.foot_tangential_force        # (4,) 同刻的切向力模长 [N]
        if foot_contact is None:
            foot_contact = env._contact_scan()[2]

        # R1 滑擦：**接触中还在切向滑动**才算滑擦。速度门是关键 —— 支撑相里
        # 切向力高是正常的（蹬地推进），但那时切向速度 ≈ 0。这一项正是
        # `foot_lift` 的三维速度门控分不开的「抬高 vs 前扫」：抬高的脚没有力，
        # 前扫的脚有力。
        if rc.foot_scuff != 0.0:
            ratio = ft / np.maximum(fn, 1e-6)
            span = max(float(rc.scuff_mu_hard) - float(rc.scuff_mu_soft), 1e-6)
            x = np.clip((ratio - float(rc.scuff_mu_soft)) / span, 0.0, 1.0)
            sliding = foot_contact & (foot_speed > float(rc.scuff_v_ref))
            terms["foot_scuff"] = rc.foot_scuff * float(np.mean(x * sliding))
        else:
            terms["foot_scuff"] = 0.0

        # R2 冲击：只在**落地帧**结算，用的是本控制步内逐物理子步取到的法向力
        # 峰值（50 Hz 控制步末的读数已经被柔顺接触吸收了，看不到真正的尖峰）。
        if rc.foot_impact != 0.0:
            landed = foot_contact & ~env._prev_foot_contact
            over = np.clip((env.foot_impact_force - float(rc.impact_force_tol))
                           / max(float(rc.impact_force_scale), 1e-6), 0.0, 1.0)
            terms["foot_impact"] = rc.foot_impact * float(np.sum(over * landed))
        else:
            terms["foot_impact"] = 0.0

        # R3 负载分担：罚「一只脚扛了太多」。**必须门控着地足数**，否则它会奖励
        # 「保持四足着地」= 站桩。默认只罚平地：楼梯上四足跨级站立天然一头受力
        # 大，在楼梯上罚不均等于在罚正确行为。
        if rc.load_share != 0.0:
            n_down = int(np.count_nonzero(foot_contact))
            if (n_down >= int(rc.load_share_min_feet)
                    and not (rc.load_share_flat_only and env.stair.n_up > 0)):
                f = np.where(foot_contact, fn, 0.0)
                total = float(np.sum(f))
                if total > 1e-6:
                    cap = float(rc.load_share_cap)
                    worst = float(np.max(f)) / total
                    terms["load_share"] = rc.load_share * float(
                        np.clip((worst - cap) / max(1.0 - cap, 1e-6), 0.0, 1.0))
                else:
                    terms["load_share"] = 0.0
            else:
                terms["load_share"] = 0.0
        else:
            terms["load_share"] = 0.0

        # R4 质心支撑裕度：机身水平投影到**按力加权**的支撑中心的距离。按力加权
        # 而不是几何中心 —— 「三足着地、一足虚搭」时几何中心会给出错误的支撑点，
        # 而那正是要把「虚搭」暴露出来的场景。支撑多边形用中心近似（平地够用；
        # 楼梯的支撑多边形是跨级的，所以默认只在平地结算）。
        if rc.com_margin != 0.0:
            if rc.com_margin_flat_only and env.stair.n_up > 0:
                terms["com_margin"] = 0.0
            else:
                f = np.where(foot_contact, fn, 0.0)
                total = float(np.sum(f))
                if total > 1e-6:
                    support = (f[:, None] * feet[:, :2]).sum(axis=0) / total
                    dist = float(np.linalg.norm(env.base_pos[:2] - support))
                    terms["com_margin"] = rc.com_margin * float(
                        np.clip((dist - float(rc.com_margin_tol))
                                / max(float(rc.com_margin_scale), 1e-6), 0.0, 1.0))
                else:
                    terms["com_margin"] = 0.0
        else:
            terms["com_margin"] = 0.0

    # 11) 足底触觉相位条件策略（TactileStep，2026-10-20）。
    #     **四个权重都为 0 时整段跳过** —— 逐步奖励逐位不变（eq_check 0 差异）。
    #     相位机与力/面积/CoP 的测量都在 robotdog/tactile.py；这里只做结算。
    if (rc.tactile_prelanding or rc.tactile_impact or rc.tactile_stance
            or rc.tactile_slip):
        tac = env.tactile
        pre = tac.phase == PRELANDING
        landing = tac.phase == LANDING
        stance = tac.phase == STANCE
        if rc.tactile_prelanding:
            # ① PreLanding compliance：只看**向下**的足端速度（向上不罚）。
            down = np.clip(-tac.vz / max(float(rc.prelanding_vz_ref), 1e-6),
                           0.0, 1.0)
            terms["tactile_prelanding"] = float(rc.tactile_prelanding) * float(
                np.mean(down ** 2 * pre))
        else:
            terms["tactile_prelanding"] = 0.0
        if rc.tactile_impact:
            # ② Landing impact：平均法向力 + 力变化 + 窗内峰值，各自按参考量级
            #    归一化后相加（1.0 ≈ 恰好到参考量级，2.0 封顶）。
            f_mean = np.clip(tac.force_norm / max(float(rc.impact_force_ref), 1e-6),
                             0.0, 2.0)
            f_delta = np.clip(
                np.abs(tac.delta_force_norm)
                / max(float(rc.impact_delta_ref), 1e-6), 0.0, 2.0)
            f_peak = np.clip(tac.impact_peak / max(float(rc.impact_peak_ref), 1e-6),
                             0.0, 2.0)
            terms["tactile_impact"] = float(rc.tactile_impact) * float(
                np.mean((f_mean + f_delta + f_peak) * landing))
        else:
            terms["tactile_impact"] = 0.0
        if rc.tactile_stance:
            # ③ Stance support：奖励接触面积（到目标封顶）与 CoP 裕度，罚 CoP 跳变。
            area = np.clip(
                tac.area_norm / max(float(rc.stance_area_target), 1e-6), 0.0, 1.0)
            jump = np.clip(
                tac.cop_jump / max(float(rc.stance_cop_jump_ref), 1e-6), 0.0, 1.0)
            stance_term = float(np.mean((area + tac.margin - jump) * stance))
            # 「真的在前进」才付支撑分：否则原地蹲稳也能拿满（l1_st 的教训）
            min_vx = float(rc.stance_min_vx)
            if min_vx > 0.0:
                vx_fwd = max(0.0, float(env.base_lin_vel_body[0]))
                stance_term *= min(1.0, vx_fwd / min_vx)
            terms["tactile_stance"] = float(rc.tactile_stance) * stance_term
        else:
            terms["tactile_stance"] = 0.0
        if rc.tactile_slip:
            # ④ 支撑相足端水平滑移（本次控制步的足端位移，仅支撑足结算）
            prev = getattr(env, "_prev_foot_pos", None)
            if prev is None:
                terms["tactile_slip"] = 0.0
            else:
                disp = np.linalg.norm(
                    np.asarray(env.foot_pos)[:, :2]
                    - np.asarray(prev)[:, :2], axis=1)
                slip_n = np.clip(disp / max(float(rc.slip_ref), 1e-6), 0.0, 1.0)
                terms["tactile_slip"] = float(rc.tactile_slip) * float(
                    np.mean(slip_n * stance))
        else:
            terms["tactile_slip"] = 0.0

    # --- 终止 -----------------------------------------------------------
    # 到达顶部 / 平地目标会*成功*结束回合，而不是让机器人走出楼梯远端摔下去。
    # ``env._reached_top`` 在回合内是粘性的，所以更早某一步的摔倒仍然算失败。
    # 摔倒罚按爬升进度分级（见 ``RewardConfig.termination_progress_relief``）：
    # 高处的摔倒比脚下摔倒罚得轻 —— 进展已经由 stair_progress 逐级支付。
    terminal = 0.0
    if terminated:
        if env._reached_top:
            terminal = rc.success
        else:
            crest = float(env.stair.crest_height)
            frac = min(1.0, env._peak_terrain_height / crest) if crest > 1e-9 else 0.0
            terminal = rc.termination * (1.0 - rc.termination_progress_relief * frac)

    return float(sum(terms.values())), terms, float(terminal)


# --------------------------------------------------------------------------- #
# 势函数状态机（env.step 在奖励结算**之前**调用）
# --------------------------------------------------------------------------- #

def track_swing_peak(env, foot_contact: np.ndarray) -> None:
    """维护每只脚「本次摆动期内的最大离地间隙」（``env._peak_swing_clearance``）。

    为什么要单独维护：落地那一帧脚已经贴回踏面，当时的间隙接近 0，所以
    「这次摆动到底抬了多高」只能靠摆动期的峰值来判断，落地帧本身看不出
    来。脚一落地就把峰值清零，为下一次摆动重新计。

    ⚠ **清零必须发生在奖励结算之后，而不是接触的那一帧**。实测踩过这个
    坑：如果一接触就清零，那么落地帧读到的是「上一帧的峰值」，比整个摆动
    期的真实峰值低 0.01~0.03 m，探测项的判据会系统性偏严。做法是「落地那
    一帧保留旧峰值，结算完再清零」。
    """
    clearance = env.foot_clearance
    landed = foot_contact & ~env._prev_foot_contact
    env._peak_swing_clearance = np.where(
        landed, env._peak_swing_clearance,
        np.where(foot_contact, 0.0,
                 np.maximum(env._peak_swing_clearance, clearance)),
    )


def update_probe_state(env) -> None:
    """维护「被阻挡」状态（``env._blocked``），在物理推进之后、奖励结算之前调用。

    判据：**有前进指令，但实测前进速度远低于指令**（比例见
    ``reward.probe_blocked_frac``）。一旦判定为被阻挡就**锁住**，直到机身
    下方的地形刷新了本回合的最高高度（真的上去了）或前进速度恢复为止 ——
    因为机器人撞上立面时会前后晃，不锁住的话这个布尔量会一步一跳。

    ``_blocked_peak`` 是锁住那一刻记录的地形高，用来判断「有没有上去」。
    解除阻挡时顺带把逐足的 ``_probe_best`` 清零，让下一次被卡住可以重新累积
    「抬高进展」（否则势函数在整条回合里只增不减，第二次卡住就没有梯度）。
    """
    rc = env.cfg.reward
    cmd_body = env.command_body()
    cmd_vx = float(cmd_body[0])              # 机体系投影（见 command_body 的 docstring）
    # 机身系：这是「往前走没走」的判据，世界系分量在机器人偏航时会换号。
    vx = float(env.base_lin_vel_body[0])
    terrain = float(env.terrain_height_under_base)
    if env._blocked:
        recovered = terrain > env._blocked_peak + 1e-9
        if recovered or vx > max(0.5 * cmd_vx, 0.05):
            env._blocked = False
            env._probe_best[:] = 0.0
    if not env._blocked and cmd_vx > 1e-6:
        if vx < rc.probe_blocked_frac * cmd_vx:
            env._blocked = True
            env._blocked_peak = terrain
            env._probe_best[:] = 0.0
