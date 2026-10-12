"""奖励函数：本仓库所有奖励数学的唯一来源。

**为什么单独一个文件。** 奖励是这套系统里唯一「写错了不会报错、只会训不出来」的
部分：前向速度、姿态、力矩、足端抬升、碰撞、爬升进展各项互相拉扯，任何一个权重
或公式的改动都会在几小时后才以「策略站着不动」的形式暴露。把它从
``robotdog/env.py`` 里整块搬出来，是为了能在一个屏幕里读完、单独改、单独做数值
实验，而不必在 800 行的环境文件里翻找。

本模块由三部分构成：

* :class:`RewardConfig` —— 全部权重与阈值（``config.yaml`` 的 ``reward`` 段）。
* :func:`compute_reward` —— 奖励数学本体：输入是环境状态，输出是**逐步奖励**、
  **逐项明细**与**终止奖励**，不做任何物理推进。它**不是纯函数** —— 几个势函数
  项要靠环境上的记忆字段（``_peak_terrain_height`` / ``_probe_best`` /
  ``_probe_count``）做到「总量有界、不可刷分」，所以会写这几个字段。
* :func:`track_swing_peak` / :func:`update_probe_state` —— ``env.step`` 在奖励
  结算**之前**调用的两个势函数状态机（摆动峰值记录、「被阻挡」判定）。它们与
  :func:`compute_reward` 读写同一批记忆字段，所以与奖励数学同居一处，而不是
  散在环境文件里。

关于逐项明细：:func:`compute_reward` 返回的 ``terms`` 里，每一项**都已经乘过它的
权重**，所以 ``sum(terms.values())`` 就是未经终止奖励的总奖励。想看「这个回合是谁
把分扣光的」直接按值排序即可。

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
    """奖励权重。

    大致平衡正项与负项。这些数值所来自的历史调参——每个权重堵住哪种失败——见
    ``experience.md`` §2，公式见 ``README.md`` §3.3。
    """

    # --- 五个要求的项 ----------------------------------------------------- #
    # 前进项的形状是「**绝对**误差窗口」：量程是一个常数，不随指令大小变。
    # 超速侧另加 ``forward_overspeed`` 的惩罚。
    forward_vel: float = 10.0           # [SPEC] 对 vx 指令的绝对误差窗口跟踪
    forward_overspeed: float = 1.0      # [SPEC] 超速侧惩罚斜率（× forward_vel 量级）
    # 超速侧惩罚的**斜率量程** [m/s]：超速每 1 m/s 扣 ``forward_vel /
    # overspeed_range`` 分。取 0.45 表示「超速 0.45 m/s 就把这一项的满分扣光」，
    # 而实测小跑的过冲正好在这一量级 —— 既留得住梯度，又不会让超速 0.05 m/s
    # 就吃掉整项。
    overspeed_range: float = 0.45
    # 前进跟踪项的**绝对**误差窗口 [m/s]：``|v - cmd|`` 超过它就把这一项扣光。
    #
    # 为什么必须是**绝对**量程：早先这里取 ``max(|cmd.vx|, forward_range_floor)``，
    # 于是分值变成「相对误差」—— 指令区间一窄，站着不动就几乎满分。实测
    # 无先验从零训练时 ``cmd.vx = 0.25``、策略 ``v = 0`` 仍拿 ``+1.10/步``，
    # 配上 ``command_follow`` 的 ``+3.76/步`` 与 ``alive +0.15``，站桩合计
    # ``+4.93/步``；三臂各 200k~300k 步全都收敛到同一个常数输出
    # （``|action|`` 均值 0.051、``vx = −0.0003``）。改成绝对窗口后，
    # 「离指令远」不再因为指令本身小而被赦免：``v = 0`` 对 ``cmd = 0.25``
    # 的绝对误差就是 0.25，窗口 0.15 下这一项为负。
    #
    # 取值要**低于**最窄的指令带宽度，否则带内所有速度都拿满分、没有区分度。
    forward_track_window: float = 0.25
    # 线性死区**外侧**的线性尾项系数。死区形状 ``max(0, 1 - |err|/range)`` 在
    # ``|err| > range`` 之后恒为 0、梯度也恒为 0，于是「已经偏出量程」的状态
    # 完全没有回头路 —— 实测偏航误差 0.44 rad/s 而 ``yaw_max`` 只有 0.15 时，
    # 这一项整段归零，策略 250k 步都没把自转收回来。补一项
    # ``-track_tail * clamp(|err|/range - 1, 0, None)`` 让越界之后仍有梯度。
    # 符号必须是**减**（越界越远越差），反例与依据见 :func:`track_shape`。
    track_tail: float = 1.0
    # 超速惩罚的**倍率上界**（``forward_vel`` 的倍数，不是 m/s）。超速惩罚随误差
    # 线性增长且原本没有上界，平地上「先验 2.05 m/s 对指令 0.45」会把它推到 −2.7 倍，
    # 单项吃满 −27/步、单回合 −65000，价值函数学不动。取 2.0 ⇒ 最差 −2.0×forward_vel。
    overspeed_cap: float = 2.0
    orientation: float = -1.2           # [SPEC] roll^2 + pitch^2（平方和）
    torque: float = -1.5e-4             # [SPEC] sum(tau^2)（关节力矩平方和）
    foot_lift: float = 0.9              # [SPEC] 摆动期离地间隙成形
    body_collision: float = -1.2        # [SPEC] 躯干接触地形

    # --- 横向 / 旋转指令跟踪 ----------------------------------------------- #
    # 权重是「完美跟踪每步得多少分」。横向项只在 ``cmd.vy_max`` 非零时才参与
    # （见 ``compute_reward``）；偏航项**恒在**，量程取
    # ``max(cmd.yaw_max, yaw_range_floor)``。
    lateral_vel: float = 6.0            # 对 vy 指令的线性死区跟踪
    yaw_rate: float = 4.0               # 对 wz 指令的线性死区跟踪
    # ``yaw_max`` 为 0 时，偏航跟踪项用来归一化误差的**量程下限** [rad/s]，
    # 与 ``forward_range_floor`` 同一个用意：偏航项不门控，零指令级别也要有
    # 「不转」的价钱，同时避免除零。取 0.30 与 ``command_vx`` 量程下限同量级。
    yaw_range_floor: float = 0.30

    # --- 摆动足抬升 -------------------------------------------------------- #
    foot_lift_target: float = 0.08      # 目标值的下限，而不是完整的目标值
    foot_lift_step_ratio: float = 1.4   # 取较大者：target = max(target, ratio * step_height)
    foot_lift_gate_speed: float = 0.25  # m/s，三维足端速度门限

    # --- 被阻挡时的试探抬腿 ------------------------------------------------ #
    # 机器人在够不着的立面之前被阻挡时，唯一能改变局面的行为是「把脚抬得
    # 比平时更高再试」。这一项是**势函数塑形**：只结算「被阻挡期间达到过的
    # 最高摆动峰值又提高了多少」，所以总量有界、不可刷分，但对「抬得更高」
    # 每一步都有梯度。
    #
    # 起算线不能取 ``ratio × 台阶高``：实测在 15 cm 台阶前被阻挡时，默认步态的
    # 摆动峰值只到 0.09~0.12 m，而 ``action_scale=0.30`` 下能搜到的最好固定动作
    # 也只到 0.148 m。起算线一旦取 1.6×15cm = 0.24，梯度恒为零、这一项永远不触发。
    # 所以起算线只跟「当前策略自己能抬多高」比 —— 见 ``RewardConfig.probe_lift_target``。
    probe_lift: float = 1.0             # 抬高进展的权重
    probe_lift_target: float = 0.06     # m，进展的起算线（普通步态峰值带的上沿）
    probe_blocked_frac: float = 0.30    # 实测前进速度 < 该比例 × 指令速度 ⇒ 被阻挡

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


# --------------------------------------------------------------------------- #
# 计算
# --------------------------------------------------------------------------- #


def track_shape(x: float, tail: float) -> float:
    """指令跟踪的共同形状：``x = |误差| / 量程`` 的单调递减函数。

    ``x <= 1`` 是线性死区 ``1 - x``；``x > 1`` 换成更陡的线性尾项
    ``-tail * (x - 1)``。三条性质都是必需的：

    * **在 ``x = 1`` 处连续**（两侧都是 0），所以「刚好贴住容差」与「差一点点」
      的分值一样；
    * **全程单调递减**。早期的尾项是 ``- tail * (x - 1)**2``（减号），于是
      ``x > 1`` 之后惩罚先增后减，在 ``x = 1 + 1/(2*tail)`` 取到**极小值** ——
      那不是「惩罚不够」，而是奖励在主动把误差**推得更大**。实测缩幅值先验的
      平地上策略稳定在 ``wz = -0.764 rad/s`` 对指令 0（``x = 1.53``），正好越过
      那个极小点，成功率被 ``tol_wz = 0.5`` 一票否决；
    * **尾项要够陡**。二次型在刚越界时几乎不罚（``x = 1.53`` 只值 ``-0.28``），
      压不住已经养成的漂移。线性尾项在同一处值 ``-0.80``（``tail = 1.5``）。
      偏离超过 ``1 + 1/tail`` 之后由调用处的 ``max(-1.0, ...)`` 兜住，所以惩罚
      有下界、不会像早期的无界超速项那样把回报压到 −65000。
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

    这个函数**不是纯函数**：它读 ``env`` 的状态，也会推进三个「势函数」用的
    记忆字段 —— ``env._peak_terrain_height``（爬升进展的一次性结算）、
    ``env._probe_best``（逐足的摆动峰值最好成绩）、``env._probe_count``。
    也就是说同样的 ``(env 状态, action)`` 在不同时刻可能给出不同结果，**不能
    为了对比而重复调用**。

    :param env: ``Go2StairsEnv``。读位置/速度/接触/腾空时间/地形，也写上面那三个字段。
    :param action: 本控制步的策略动作（12 维）。
    :param trunk_contact: 躯干是否接触地形。
    :param foot_contact: 逐足接触掩码；``None`` 时现算（只在 ``air_time`` 非零时用到）。
    :param leg_contact: 胫部/膝部是否蹭到立面。
    :param terminated: 本步是否回合终止（成功或摔倒）。
    :returns: ``(本步奖励, 逐项明细, 终止奖励)``。终止奖励单列，因为它取决于
        「是成功登顶还是摔倒」，那是 ``step`` 的判定，不属于逐项成形。

    各项的物理含义与「它堵住哪种失败」见 ``README.md`` §3.3 与
    ``experience.md`` §2。
    """
    rc = env.cfg.reward
    cmd = env.cfg.command
    rpy = env.base_rpy
    # 「前进」取**机身系** x 分量，不取世界系。世界系 x 是本位姿导数，机器人自转时
    # 它会在步态周期内换号；实测巡航策略世界系 x 均值 +0.54 m/s 而机身系 +0.72、
    # 偏航漂到 −47°，奖励于是默认了「一边转一边走」也是一种跟踪。执行器只产
    # 机身系的推力，所以「前进」只能是机身系量；偏航对齐由 ``yaw_rate`` 项负责。
    v_fwd = _forward_speed(env)
    target_height = env.nominal_pose
    # 指令跟踪读滤波量还是瞬时量，由观测开关决定（两者必须一致，否则策略看到的
    # 和它被评判的不是同一个数；见下面 1b 的说明）。
    vel_filter = bool(env.cfg.obs.vel_filter)

    terms: dict[str, float] = {}

    # 1) 前进速度跟踪 ------------------------------------------------------
    # 形状是**绝对**误差窗口上的线性死区：``1 - |v - cmd| / forward_track_window``，
    # 外侧接 ``track_tail`` 的线性尾项让越界之后仍有回头梯度。另外**超速侧单独
    # 加一段惩罚**：``err > 0`` 时减 ``forward_overspeed * err / overspeed_range``，
    # 否则「冲到 1.9 m/s」与「刚好 0.45 m/s」拿同样的分（实测指令量程 0~0.90
    # 扫一遍，策略输出恒为 1.27 m/s，完全开环）。
    #
    # 量程是常数 ``rc.forward_track_window``，**不随指令大小变**。历史教训：
    # 早先取 ``max(|cmd.vx|, forward_range_floor)``，把分值变成「相对误差」，
    # 指令带一窄站桩就几乎满分 —— 无先验三臂全部收敛到常数输出，逐项对账见
    # :attr:`RewardConfig.forward_track_window`。
    #
    # 两处都必须**有界**：死区外没有任何下界时（``−（超速量/0.45）`` 可以到 −7 以上），
    # 平地上「先验 2.05 m/s 对指令 0.45」的固定偏差会把逐步奖励压到 −27/步、
    # 回报 −65000，价值函数彻底学不动（实测 explained_variance −10~0，策略梯度
    # 损失只有 ±0.015）。现在 shape 下界到 −1（即 ``forward_vel`` 最差 −10），
    # 超速倍率上界到 ``rc.overspeed_cap``，于是最差一项是 −15/步，与其它惩罚同量级。
    # 目标取**这一刻真正采样的指令** ``env._command[0]``，不是 ``cmd.vx``：
    # ``cmd.vx`` 是「采样范围为 0 时的回退定值」，只在 ``set_command_now`` 里跟
    # 着改；课程下发的是 ``vx_min/vx_max`` 范围，于是 ``cmd.vx`` 会一直停在
    # ``config.yaml`` 的出厂值上。实测代价：课程 vx∈[0.05,0.20] 而 ``cmd.vx=0.55``
    # 时，``|err|`` 恒 ≥0.35，``track_shape`` 一步就撞到 −1 下界，
    # ``forward_vel`` 每一步都吃满 −10；而观测与成功判据用的是真实采样值 ——
    # 奖励和判据指着两个不同的目标，策略自然学不动。
    err = float(v_fwd) - float(env._command[0])
    track = max(-1.0, track_shape(abs(err) / float(rc.forward_track_window),
                                  float(rc.track_tail)))
    if err > 0.0:
        over = min(
            float(rc.forward_overspeed) * err / float(rc.overspeed_range),
            float(rc.overspeed_cap),
        )
        track -= over
    terms["forward_vel"] = rc.forward_vel * float(track)

    # 1b) 横向 / 偏航指令跟踪 ---------------------------------------------
    # 形状同前进项；量程取指令上界**再乘 0.5**（2026-10-12）—— 只取上界时
    # 什么都不做的机器人白拿 ``+0.5/轴``（实测 +5.4/步，把机器人钉在墙脚），
    # 减半后 **≈ −0.083/轴**（尾项被 ``max(-1.0, ·)`` 截断）。证明 P1、见 §21.18.5。
    #
    # 横向项被**门控**：``vy_max == 0`` 时不参与。不门控的代价已经量过 ——
    # 「什么都没做」每步也能白拿约 1.0 分（5 cm 上实测 125 步拿了 +139.7），
    # 等于又造一个「原地拖延」的正项。
    #
    # 两个方向**求和**而不是取 ``max``：取 max 时误差小的那一个会把另一个的
    # 梯度整个盖掉，偏航因此永远学不会（实测 r=−0.085、斜率 −0.006）。
    #
    # 读**滤波后**的量：瞬时 ``yaw_rate`` 在小跑步态下 ``std`` 可达 0.37 rad/s，
    # 判据会变成「在判步态相位」而不是「机器人在往哪走」。滤波时间常数
    # ``cfg.vel_filter_tau`` 压掉一个步态周期，同一个量也进观测以保持马尔可夫性。
    follow = 0.0
    tail = float(rc.track_tail)
    if cmd.vy_max:
        vy = (env.filtered_lateral_vel if vel_filter else env.base_lin_vel_body[1])
        xy = abs(vy - cmd.vy) / (0.5 * float(cmd.vy_max))
        follow += rc.lateral_vel * max(-1.0, track_shape(xy, tail))
    # 偏航项**恒在**（不受 ``yaw_max`` 门控），量程取 ``max(yaw_max, yaw_range_floor)``。
    # 门控的代价实测过：``yaw_max == 0`` 的级别里没有任何一项读 wz，策略学会
    # 以 −0.608 rad/s **恒定自转**（`.tmp/zz_cmd_response.py` 扫 wz ∈ [−0.6, +0.6]
    # 七档，实测斜率 −0.0021、范围 −0.608~−0.606，即输出与指令逐位无关），
    # 而且这个转法在后续级别里冻住 —— 加进偏航指令后它仍在转。底噪 ``floor``
    # 保证「不转」在零指令级别也有价钱，同时避免 ``yaw_max == 0`` 时除零；
    # 加惩罚必须**有界**（下界 −1，即最差 −``yaw_rate``），理由同前进项。
    wz = env.filtered_yaw_rate if vel_filter else env.yaw_rate
    yaw_err = abs(wz - cmd.wz)
    # ``wz=0`` 时再补一个零指令基线惩罚（系数 1.5 的理由同前进项：实测三轴指令
    # 的平阶级上策略偏航漂到 −47°~−88°，旧系数 0.5 只值 −0.7/步，挡不住）。
    if cmd.wz == 0.0:
        yaw_err = 1.5 * (yaw_err + abs(wz))
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
    # 相邻步动作差 ``‖a_t − a_{t-1}‖``。奖励结算发生在 ``env.step`` 更新动作
    # 历史之前，所以 ``_last_action`` 此刻存的就是上一步的动作 ``a_{t-1}``。
    # （2026-10-12 修复：原先读 ``_prev_action``，它的更新时序滞后一步，让这一项
    # 实际算成了「与上上步动作的差」。）
    terms["action_rate"] = rc.action_rate * float(
        np.sum(np.square(action - env._last_action))
    )
    lo, hi = env.act_low, env.act_high
    q = env.joint_pos
    over = np.clip(lo - q, 0, None) + np.clip(q - hi, 0, None)
    terms["joint_limit"] = rc.joint_limit * float(np.sum(over))
    terms["alive"] = rc.alive

    # --- 爬升进展 -------------------------------------------------------
    # 一次性结算：只有当机身下方的地形**刷新了本回合的最高高度**时才付，
    # 付的是这次新增的高度。
    #
    # 原来的写法是 ``stair_progress * max(0, 下方地形高 - 回合起点高)``，它
    # 每步都付 —— 只要机身正下方是高台阶就一直拿钱，于是「爬到某一级然后
    # 站着不动」每一步都白拿 +0.083/步（15 cm），这是「原地拖延」局部最优
    # 的第二根支柱（见 experience.md §12）。改成按最大值结算后，累计总量
    # 仍是 ``stair_progress × 爬升高度``，与走多少步无关 —— 但站着不动再也
    # 拿不到新的钱。⚠ 2026-10-12：参照面由**躯干 x** 改成**四足支撑面**（见 experience.md §2.7）。
    climbed = max(0.0, env._support_height() - env._peak_terrain_height)
    if climbed > 0.0:
        env._peak_terrain_height = env._support_height()
    terms["stair_progress"] = rc.stair_progress * float(climbed)

    # 8) 被阻挡时的试探抬腿 ------------------------------------------------
    # 门控（全部满足才可能给分）：
    #   ① 本回合已经被判定为「被阻挡」（``env._blocked``）—— 由
    #      ``update_probe_state()`` 在物理推进之后维护，判据是「有前进指令
    #      但实测前进速度远低于指令」，而且一旦锁住就持续到机身下方的地形
    #      刷新了新高度（真的上去了）才解除；
    #   ② 本步的摆动峰值越过起算线 ``rc.probe_lift_target``。实测默认先验的峰值
    #      带是 0.06~0.12 m（``.tmp/calib_probe3.py``），而 ``action_scale=0.30``
    #      下搜到的最好固定动作能到 0.148 m（``.tmp/calib_probe4.py``）——
    #      所以起算线取 0.06 m 这个「普通步态的上沿」，留出可优化的空间。
    #
    # ⭐ 这是**势函数塑形**（potential-based shaping），不是逐步固定分：
    # ``env._probe_best`` 逐足记住本回合「被阻挡期间达到过的最高峰值」，每一步只
    # 结算「比该足自己的历史最好还高出多少」。因此
    #   * 总量有界：最多 ``probe_lift × Σ(可达最高峰值 − 起算线)``；
    #   * 不可刷分：重复做同样的高度拿不到任何分；
    #   * 有梯度：每一次抬得更高都被记录，即使那一脚最终没踩上去。
    # 逐足而不是全体取 max：四足同一步同时抬高时，标量地板会把只抬了一只脚的
    # 那些足也算成「进展」，下一步那只脚单独再抬就再也拿不到分（实测两足同到
    # 0.10 记 0.08，随后单独到 0.08 记 0.0）。
    # 为什么不用「落地那一刻峰值 > 门槛就给固定分」：实测那条路要么门槛太低
    # （等于奖励普通步态），要么太高（一次都不触发）。见 experience.md §15。
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
    cmd_vx = float(env._command[0])          # 采样指令，不是 cfg.command.vx（见 env.md §3.13）
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


# --------------------------------------------------------------------------- #
# 俯仰补偿（只服务前进项）
# --------------------------------------------------------------------------- #

def _forward_speed(env) -> float:
    """前进项该跟踪的速度 —— **按地形分两个参照系**。

    平地段：**机身系**前进速度。那里「前方」就是鼻子，机器人可以自由转向，转向本身
    是指令的一部分。

    楼梯级：**课程方向**（世界系 +x）的水平速度。楼梯是按 ``start_x`` 沿 +x 铺开的
    **一条课程**，所以 ``vx`` 只能是「沿课程前进的速度」。

    ⚠ 为什么楼梯级必须换参照系（2026-10-12，见 ``experience.md`` §21.14）：跟踪
    机身系速度时「前方」是鼻子指哪算哪，而出生点背后是**无障碍平地** —— 训练出来的
    策略学会**掉头 173° 背对楼梯、以指令带内的速度走开**：世界系 vx −0.22 而机身系
    +0.34，``forward_vel`` 照付 **+7480**，一整回合净赚 +10667，而爬完整条 15 cm
    楼梯只值 ``success 50 + stair_progress 1200×1.20 ≈ +1490``。奖励在指反方向。

    离线预验（同一批真实轨迹上两种参照的回合累计）：

        真爬上去（3 / 6 cm 成功）       +2301 / +2731  →  +2363 / +2655（只差 2%）
        掉头走开（stair15 / 12 训练后） +7480 / +6248  →  **−7222 / −5561**
        卡在立面前（stair9）            −5150          →  −5490

    即**只打漏洞、不打正解**。

    历史上这里还有一层「俯仰投影补偿」（把机身系速度换算到水平面内沿机头的方向）。
    课程方向是**世界系**量、本来就不含俯仰投影误差，所以那层在楼梯级被完全取代；
    留在平地上只有约 1% 的效应，却会破坏「平地逐位不变」这条可测断言，故一并删除。
    （它当时的推导还犯过一个符号错误 —— 只除以 ``cos``、漏掉 ``− v_world[2]·e_x[2]``，
    实测比原始读数更差，记录在 ``experience.md`` §21.10。）
    """
    if env.stair.n_up > 0:
        return float(env.base_lin_vel_world[0])
    return float(env.base_lin_vel_body[0])
