"""奖励函数：本仓库所有奖励数学的唯一来源。

**为什么单独一个文件。** 奖励是这套系统里唯一「写错了不会报错、只会训不出来」的
部分：前向速度、姿态、力矩、足端抬升、碰撞、爬升进展各项互相拉扯，任何一个权重
或公式的改动都会在几小时后才以「策略站着不动」的形式暴露。把它从
``robotdog/env.py`` 里整块搬出来，是为了能在一个屏幕里读完、单独改、单独做数值
实验，而不必在 800 行的环境文件里翻找。

本模块由两半构成：

* :class:`RewardConfig` —— 全部权重与阈值（``config.yaml`` 的 ``reward`` 段）。
* :func:`compute_reward` —— 纯函数，输入是环境状态，输出是**逐步奖励**、**逐项
  明细**与**终止奖励**，不做任何物理推进。

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
    forward_vel: float = 2.5            # [SPEC] 对 vx 指令的高斯跟踪
    forward_vel_sigma: float = 0.25     # m/s，高斯的标准差（**平方后作分母**）
    orientation: float = -1.2           # [SPEC] roll^2 + pitch^2（平方和）
    torque: float = -1.5e-4             # [SPEC] sum(tau^2)（关节力矩平方和）
    foot_lift: float = 0.9              # [SPEC] 摆动期离地间隙成形
    body_collision: float = -1.2        # [SPEC] 躯干接触地形

    # --- 横向 / 旋转指令跟踪（m08247 新增）--------------------------------- #
    # 只在 ``EnvConfig.command`` 的范围非零时才可能非零（见 compute_reward 里的
    # ``cmd.on`` 门），所以「只往前走」的阶段与旧行为逐位一致。
    lateral_vel: float = 4.0            # 对 vy 指令的高斯跟踪
    lateral_vel_sigma: float = 0.25     # m/s
    yaw_rate: float = 3.0               # 对 wz 指令的高斯跟踪
    yaw_rate_sigma: float = 0.25        # rad/s

    # --- 摆动足抬升 -------------------------------------------------------- #
    foot_lift_target: float = 0.08      # 目标值的下限，而不是完整的目标值
    foot_lift_step_ratio: float = 1.4   # 取较大者：target = max(target, ratio * step_height)
    foot_lift_gate_speed: float = 0.25  # m/s，三维足端速度门限

    # --- 被阻挡时的试探抬腿（m08247 后续新增）------------------------------ #
    # 机器人在够不着的立面之前被阻挡时，唯一能改变局面的行为是「把脚抬得
    # 比平时更高再试」。这一项是**势函数塑形**：只结算「被阻挡期间达到过的
    # 最高摆动峰值又提高了多少」，所以总量有界、不可刷分，但对「抬得更高」
    # 每一步都有梯度。
    #
    # 起算线为什么不用 ``ratio × 台阶高``：实测（``.tmp/calib_probe3.py``）在
    # 15 cm 台阶前被阻挡时，默认步态的摆动峰值只到 0.09~0.12 m，而
    # ``action_scale=0.30`` 下能搜到的最好固定动作也只到 0.148 m
    # （``.tmp/calib_probe4.py``）。起算线一旦取 1.6×15cm = 0.24，梯度恒为零，
    # 这一项永远不触发。所以起算线只跟「当前策略自己能抬多高」比 —— 见
    # ``env.probe_target``。
    probe_lift: float = 1.0             # 抬高进展的权重
    probe_lift_target: float = 0.06     # m，进展的起算线（普通步态峰值带的上沿）
    probe_lift_ratio: float = 1.0       # 记录用：曾按 ratio × 台阶高 取起算线，已弃用
    probe_lift_max: int = 12            # 记录用：势函数本身有界，不再按次数截断
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

    :param env: ``Go2StairsEnv``。只**读**状态（位置/速度/接触/腾空时间/地形），
        不推进物理，也不改任何字段。
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
    rpy = env.base_rpy
    lin = env.base_lin_vel_world
    target_height = env.nominal_pose

    terms: dict[str, float] = {}

    # 1) 前进速度奖励 -- 前向速度跟踪 ------------------------------------
    # ``sigma`` 是**标准差**，所以平方后作分母：vx=0 时 err=0.55 m/s，
    # 指数是 exp(-(0.55/0.25)^2) = exp(-4.84) ≈ 0.0079，奖励约 +0.02/步，
    # 而不是原来写成 exp(-err^2/sigma) 时的 +0.72/步。历史那个「没平方」的
    # 写法等于把 0.55 m/s 的偏差只按 sqrt(0.25)=0.5 m/s 折算，「站着不动」
    # 因此白拿 +0.86/步，是「原地拖延」局部最优的第一根支柱
    # （见 experience.md §12）。
    err = lin[0] - env.cfg.command.vx
    terms["forward_vel"] = (
        rc.forward_vel * float(np.exp(-(err * err) / (rc.forward_vel_sigma ** 2)))
    )

    # 1b) 横向 / 偏航指令跟踪 ---------------------------------------------
    # 指令来自外部控制器（或 reset 时的随机采样），是盲走允许的信息。
    # 机体系横向速度直接取 ``base_lin_vel_body[1]``：它不会随偏航改变含义，
    # 所以「横向」永远是这一个分量。
    #
    # ⭐ 这一项被门控住：**两个指令范围都为 0 时它恒为 0**。不门控的代价已经
    # 量过 —— 高斯在 err 上有 0.25 的宽度，而机器人直走时 ``vy``/``yaw_rate``
    # 的偏差恰好落在 sigma 附近，于是「什么都没做」每步也能白拿约 1.0 分
    # （5 cm 上实测 125 步拿了 +139.7）。那就等于又造了一个「原地拖延」的正项。
    #
    # ⭐ 两个方向为什么是**求和**而不是取 ``max``（m10558 起）：取 max 时，
    # 哪个方向的误差小就只保留那一个 —— 横向跟得还行时偏航项的梯度被整个丢掉，
    # 策略于是永远学不会偏航。用一条实测确认了后果：course3 的 flat_3 检查点
    # 横向上还有微弱相关性（r=+0.661、斜率 +0.028），偏航则完全为 0
    # （r=−0.085、斜率 −0.006）。求和让两个方向各自都有梯度。
    cmd = env.cfg.command
    follow = 0.0
    if cmd.vy_max:
        vy_err = env.base_lin_vel_body[1] - cmd.vy
        follow += rc.lateral_vel * float(
            np.exp(-(vy_err * vy_err) / (rc.lateral_vel_sigma ** 2))
        )
    if cmd.yaw_max:
        yaw_err = env.yaw_rate - cmd.wz
        follow += rc.yaw_rate * float(
            np.exp(-(yaw_err * yaw_err) / (rc.yaw_rate_sigma ** 2))
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
    # ``env._probe_best`` 记住本回合「被阻挡期间达到过的最高峰值」，每一步只
    # 结算「比历史最好还高出多少」。因此
    #   * 总量有界：最多 ``probe_lift × (可达最高峰值 − 起算线)``；
    #   * 不可刷分：重复做同样的高度拿不到任何分；
    #   * 有梯度：每一次抬得更高都被记录，即使那一脚最终没踩上去。
    # 为什么不用「落地那一刻峰值 > 门槛就给固定分」：实测那条路要么门槛太低
    # （等于奖励普通步态），要么太高（一次都不触发）。见 experience.md §15。
    if rc.probe_lift != 0.0 and env._blocked:
        peaks = env._peak_swing_clearance
        floor = max(env._probe_best, env.probe_target)
        gain = float(np.sum(np.clip(peaks - floor, 0.0, None)))
        if gain > 0.0:
            env._probe_best = float(np.max(peaks))
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
