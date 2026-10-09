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
    forward_vel: float = 2.5            # [SPEC] 对 command_vx 的高斯跟踪
    forward_vel_sigma: float = 0.25     # m/s，高斯的标准差（**平方后作分母**）
    orientation: float = -1.2           # [SPEC] roll^2 + pitch^2（平方和）
    torque: float = -1.5e-4             # [SPEC] sum(tau^2)（关节力矩平方和）
    foot_lift: float = 0.9              # [SPEC] 摆动期离地间隙成形
    body_collision: float = -4.0        # [SPEC] 躯干接触地形

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
    err = lin[0] - env.cfg.command_vx
    terms["forward_vel"] = (
        rc.forward_vel * float(np.exp(-(err * err) / (rc.forward_vel_sigma ** 2)))
    )

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

    # --- 终止 -----------------------------------------------------------
    # 到达顶部 / 平地目标会*成功*结束回合，而不是让机器人走出楼梯远端摔下去。
    # ``env._reached_top`` 在回合内是粘性的，所以更早某一步的摔倒仍然算失败。
    terminal = 0.0
    if terminated:
        terminal = rc.success if env._reached_top else rc.termination

    return float(sum(terms.values())), terms, float(terminal)
