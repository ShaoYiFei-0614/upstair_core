"""奖励函数：本仓库所有奖励数学的唯一来源。

**为什么单独一个文件。** 奖励是这套系统里唯一「写错了不会报错、只会训不出来」的
部分：前向速度、姿态、力矩、足端抬升、碰撞、爬升进展各项互相拉扯，任何一个权重
或公式的改动都会在几小时后才以「策略站着不动」的形式暴露。把它从
``robotdog/env.py`` 里整块搬出来，是为了能在一个屏幕里读完、单独改、单独做数值
实验，而不必在 800 行的环境文件里翻找。

本模块由两半构成：

* :class:`RewardConfig` —— 全部权重与阈值（``config.yaml`` 的 ``reward`` 段）。
* :func:`compute_reward` —— 奖励数学本体：输入是环境状态，输出是**逐步奖励**、
  **逐项明细**与**终止奖励**，不做任何物理推进。它**不是纯函数** —— 几个势函数
  项要靠环境上的记忆字段（``_peak_terrain_height`` / ``_probe_best`` /
  ``_probe_count``）做到「总量有界、不可刷分」，所以会写这几个字段。

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
    # 形状是「按量程归一化的线性死区」，与 ``lateral_vel`` / ``yaw_rate``
    # 一致；量程取 ``max(|cmd.vx|, forward_range_floor)``，超速侧另加
    # ``forward_overspeed`` 的惩罚。
    forward_vel: float = 10.0           # [SPEC] 对 vx 指令的线性死区跟踪
    forward_overspeed: float = 1.0      # [SPEC] 超速侧惩罚斜率（× forward_vel 量级）
    # 超速侧惩罚的**斜率量程** [m/s]：超速每 1 m/s 扣 ``forward_vel /
    # overspeed_range`` 分。取 0.45 表示「超速 0.45 m/s 就把这一项的满分扣光」，
    # 而实测小跑的过冲正好在这一量级 —— 既留得住梯度，又不会让超速 0.05 m/s
    # 就吃掉整项。
    overspeed_range: float = 0.45
    # ``vx`` 指令为 0 时，前进跟踪项用来归一化误差的**量程下限** [m/s]。
    # ``vx`` 是定值基线、没有「采样上界」可借（横向/偏航有 ``vy_max``/``yaw_max``），
    # 所以要用一个下限把分母撑开。取 0.30 的含义：vx=0 时允许 ±0.30 m/s 的余量
    # 拿满分，超过 0.30 m/s 归零 —— 与实测「小跑步态自带 0.24 m/s 前进偏置」
    # 同量级，足以把「别前进」这句话变成有价钱的要求。
    forward_range_floor: float = 0.30
    orientation: float = -1.2           # [SPEC] roll^2 + pitch^2（平方和）
    torque: float = -1.5e-4             # [SPEC] sum(tau^2)（关节力矩平方和）
    foot_lift: float = 0.9              # [SPEC] 摆动期离地间隙成形
    body_collision: float = -1.2        # [SPEC] 躯干接触地形

    # --- 横向 / 旋转指令跟踪 ----------------------------------------------- #
    # 只在 ``EnvConfig.command`` 的范围非零时才可能非零（见 ``compute_reward`` 里的
    # ``cmd.vy_max`` / ``cmd.yaw_max`` 门），所以「只往前走」的阶段与旧行为逐位一致。
    # 线性死区、按指令量程归一化，所以这两个权重是「完美跟踪每步得多少分」。
    lateral_vel: float = 6.0            # 对 vy 指令的线性死区跟踪
    yaw_rate: float = 4.0               # 对 wz 指令的线性死区跟踪

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
    # 所以起算线只跟「当前策略自己能抬多高」比 —— 见 ``env.probe_target``。
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
    lin = env.base_lin_vel_world
    target_height = env.nominal_pose
    # 指令跟踪读滤波量还是瞬时量，由观测开关决定（两者必须一致，否则策略看到的
    # 和它被评判的不是同一个数；见下面 1b 的说明）。
    vel_filter = bool(env.cfg.obs.vel_filter)

    terms: dict[str, float] = {}

    # 1) 前进速度跟踪 ------------------------------------------------------
    # 形状与横向/偏航一致（按量程归一化的线性死区），但**超速侧单独加一段惩罚**：
    # 线性死区在 ``|err| > range`` 之后恒为 0，梯度也跟着消失，于是「冲到 1.9 m/s」
    # 与「刚好 0.45 m/s」拿同样的分（实测指令量程 0~0.90 扫一遍，策略输出恒为
    # 1.27 m/s，完全开环）。超速是主动选择，必须一路有价钱。
    #
    # 量程取 ``max(|cmd.vx|, rc.forward_range_floor)`` 把 ``vx=0`` 时的分母撑开，
    # 否则「别前进」这个指令没有价钱。超速惩罚的斜率量程是 ``rc.overspeed_range``。
    err = float(lin[0]) - float(cmd.vx)
    fwd_range = max(abs(float(cmd.vx)), float(rc.forward_range_floor))
    track = max(0.0, 1.0 - abs(err) / fwd_range)
    if err > 0.0:
        track -= float(rc.forward_overspeed) * err / float(rc.overspeed_range)
    terms["forward_vel"] = rc.forward_vel * float(track)

    # 1b) 横向 / 偏航指令跟踪 ---------------------------------------------
    # 与前进项同一种形状（相对误差的线性死区），量程取指令自己的上界，所以
    # 它是「相对误差」：完美跟踪拿满分，原地不动降到 ``1 - |cmd|/range``，且
    # 大指令罚得最重 —— 没有「躲在 0 附近」这条路。
    #
    # 两项被**门控**：对应量程为 0 时该项不参与。不门控的代价已经量过 ——
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
    if cmd.vy_max:
        vy = (env.filtered_lateral_vel if vel_filter else env.base_lin_vel_body[1])
        follow += rc.lateral_vel * float(
            max(0.0, 1.0 - abs(vy - cmd.vy) / float(cmd.vy_max))
        )
    if cmd.yaw_max:
        wz = env.filtered_yaw_rate if vel_filter else env.yaw_rate
        yaw_err = abs(wz - cmd.wz)
        # ``wz=0`` 时再补一个零指令基线惩罚：只练横向的级别（``yaw_max=0``）
        # 本来没有任何一项读 wz，策略能以 −0.10~−0.20 rad/s 稳定自转
        # （实测 wz=−0.200）。补上之后，「让我不转」也有价钱。
        if cmd.wz == 0.0:
            yaw_err = 0.5 * (yaw_err + abs(wz))
        follow += rc.yaw_rate * float(
            max(0.0, 1.0 - yaw_err / float(cmd.yaw_max))
        )
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
            foot_contact = env._foot_contact_mask()
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
    terms["action_rate"] = rc.action_rate * float(
        np.sum(np.square(action - env._prev_action))
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
    # 拿不到新的钱。
    climbed = max(0.0, env.terrain_height_under_base - env._peak_terrain_height)
    if climbed > 0.0:
        env._peak_terrain_height = env.terrain_height_under_base
    terms["stair_progress"] = rc.stair_progress * float(climbed)

    # 8) 被阻挡时的试探抬腿 ------------------------------------------------
    # 门控（全部满足才可能给分）：
    #   ① 本回合已经被判定为「被阻挡」（``env._blocked``）—— 由
    #      ``_update_probe_state()`` 在物理推进之后维护，判据是「有前进指令
    #      但实测前进速度远低于指令」，而且一旦锁住就持续到机身下方的地形
    #      刷新了新高度（真的上去了）才解除；
    #   ② 本步的摆动峰值越过起算线 ``env.probe_target``。实测默认先验的峰值
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
        floor = np.maximum(env._probe_best, env.probe_target)
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
