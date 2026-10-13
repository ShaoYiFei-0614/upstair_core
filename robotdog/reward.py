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
    lateral_vel: float = 6.0            # 对 vy 指令的线性死区跟踪（vy_max 非零才参与）
    yaw_rate: float = 4.0               # 对 wz 指令的线性死区跟踪（恒在，不门控）
    yaw_range_floor: float = 0.30       # yaw_max=0 时的量程下限 [rad/s]，兼防除零
    # 零偏航指令的基线惩罚放大系数：cmd.wz == 0 时误差按 ``yaw_zero_gain ×
    # (|err| + |wz|)`` 放大，否则「命令我不许转」没有任何价钱。实测（2 Hz 步态、
    # 滤波 wz 摆幅 ≈0.1 rad/s）×1.5 使该通道在零指令级恒饱和于 −yaw_rate/步 ——
    # 是「跟住不许转」的强信号，但也是逐项账面里最大的恒定负项（experience.md §23.8）。
    yaw_zero_gain: float = 1.5

    # --- 摆动足抬升 ---------------------------------------------------------- #
    foot_lift_target: float = 0.08      # 目标值的下限，而不是完整的目标值
    foot_lift_step_ratio: float = 1.4   # 有效目标 = max(target, ratio × 台阶高)
    foot_lift_gate_speed: float = 0.25  # m/s，三维足端速度门限

    # --- 被阻挡时的试探抬腿（势函数塑形，见 doc/reward.md） ------------------ #
    probe_lift: float = 1.0             # 抬高进展的权重
    probe_lift_target: float = 0.06     # m，起算线（跟策略当前能力比，不跟台阶高走）
    probe_blocked_frac: float = 0.30    # 实测前进速度 < 该比例 × 指令速度 ⇒ 被阻挡

    # --- 步态发现 ------------------------------------------------------------ #
    air_time: float = 0.6
    air_time_target: float = 0.30
    air_time_max: float = 0.60

    # --- 辅助成形 ------------------------------------------------------------ #
    base_height: float = -2.0
    joint_vel: float = -2.0e-5
    action_rate: float = -0.02
    joint_limit: float = -0.5
    alive: float = 0.15
    termination: float = -15.0          # 摔倒一次性罚（config.yaml 用远大于此的值）
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

    # 1b) 横向 / 偏航指令跟踪：形状同前进项，窗口取 0.5 × 量程（零输出不再白拿分）。
    # 横向按量程门控（量程为 0 = 该级不练这个方向）；偏航**恒在**（量程下限
    # ``yaw_range_floor``）—— 门控分轴判断，见 experience.md §13.3。两项**求和**
    # 而不是取 max：取 max 时误差小的那项会盖掉另一项的梯度。
    # 读**滤波后**的量（与观测的 vel_filter 块同一个数，保持马尔可夫性）。
    follow = 0.0
    tail = float(rc.track_tail)
    if cmd.vy_max:
        vy = (env.filtered_lateral_vel if vel_filter else env.base_lin_vel_body[1])
        xy = abs(vy - float(cmd_body[1])) / (0.5 * float(cmd.vy_max))
        follow += rc.lateral_vel * max(-1.0, track_shape(xy, tail))
    # 偏航项**恒在**（不受 ``yaw_max`` 门控），量程取 ``max(yaw_max, yaw_range_floor)``。
    # 门控的代价实测过：``yaw_max == 0`` 的级别里没有任何一项读 wz，策略学会
    # 以 −0.608 rad/s **恒定自转**（`.tmp/zz_cmd_response.py` 扫 wz ∈ [−0.6, +0.6]
    # 七档，实测斜率 −0.0021、范围 −0.608~−0.606，即输出与指令逐位无关），
    # 而且这个转法在后续级别里冻住 —— 加进偏航指令后它仍在转。底噪 ``floor``
    # 保证「不转」在零指令级别也有价钱，同时避免 ``yaw_max == 0`` 时除零；
    # 加惩罚必须**有界**（下界 −1，即最差 −``yaw_rate``），理由同前进项。
    wz = env.filtered_yaw_rate if vel_filter else env.yaw_rate
    yaw_err = abs(wz - float(cmd_body[2]))
    # ``wz = 0`` 的级别补一个零指令基线惩罚：否则「命令我不许转」没有任何价钱
    # （放大系数见 ``RewardConfig.yaw_zero_gain``）。
    if cmd.wz == 0.0:
        yaw_err = rc.yaw_zero_gain * (yaw_err + abs(wz))
    yaw_range = 0.5 * max(float(cmd.yaw_max), float(rc.yaw_range_floor))
    follow += rc.yaw_rate * max(-1.0, track_shape(yaw_err / yaw_range, tail))
    terms["command_follow"] = follow

    # 2) 姿态稳定奖励 -- 惩罚 roll 与 pitch -----------------------------
    terms["orientation"] = rc.orientation * float(rpy[0] ** 2 + rpy[1] ** 2)

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
    gate = np.clip(foot_speed / max(float(rc.foot_lift_gate_speed), 1e-6),
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
        credit = np.clip(air - rc.air_time_target, 0.0, None)
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
        floor = np.maximum(env._probe_best, float(rc.probe_lift_target))
        gain = float(np.sum(np.clip(peaks - floor, 0.0, None)))
        if gain > 0.0:
            env._probe_best = np.maximum(env._probe_best, peaks)
            env._probe_count += 1
        terms["probe_lift"] = rc.probe_lift * gain
    else:
        terms["probe_lift"] = 0.0

    # --- 终止 -----------------------------------------------------------
    # 到达顶部 / 平地目标会*成功*结束回合，而不是让机器人走出楼梯远端摔下去。
    # ``env._reached_top`` 在回合内是粘性的，所以更早某一步的摔倒仍然算失败。
    terminal = 0.0
    if terminated:
        terminal = rc.success if env._reached_top else rc.termination

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
