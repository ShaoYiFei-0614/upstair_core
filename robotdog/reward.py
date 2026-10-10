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

#: ``vx`` 指令为 0 时，前进跟踪项用来归一化误差的**量程下限** [m/s]。
#: ``vx`` 是定值基线、没有「采样上界」可借（横向/偏航有 ``vy_max``/``yaw_max``），
#: 所以要用一个下限把分母撑开。取 0.30 的含义：vx=0 时允许 ±0.30 m/s 的余量
#: 拿满分，超过 0.30 m/s 归零 —— 与实测「小跑步态自带 0.24 m/s 前进偏置」
#: 同量级，足以把「别前进」这句话变成有价钱的要求。
FORWARD_RANGE_FLOOR = 0.30


@dataclass
class RewardConfig:
    """奖励权重。

    大致平衡正项与负项。这些数值所来自的历史调参——每个权重堵住哪种失败——见
    ``experience.md`` §2，公式见 ``README.md`` §3.3。
    """

    # --- 五个要求的项 ----------------------------------------------------- #
    # ⭐ 形状是「按量程归一化的线性死区」（m11166 起），与 ``lateral_vel`` /
    # ``yaw_rate`` 一致；量程取 ``max(|cmd.vx|, FORWARD_RANGE_FLOOR)``。
    # ``forward_vel_sigma`` 因此不再参与计算（保留只为兼容旧配置）。
    forward_vel: float = 10.0           # [SPEC] 对 vx 指令的线性死区跟踪
    forward_vel_sigma: float = 0.25     # m/s（历史字段，不再参与计算）
    orientation: float = -1.2           # [SPEC] roll^2 + pitch^2（平方和）
    torque: float = -1.5e-4             # [SPEC] sum(tau^2)（关节力矩平方和）
    foot_lift: float = 0.9              # [SPEC] 摆动期离地间隙成形
    body_collision: float = -1.2        # [SPEC] 躯干接触地形

    # --- 横向 / 旋转指令跟踪（m08247 新增）--------------------------------- #
    # 只在 ``EnvConfig.command`` 的范围非零时才可能非零（见 compute_reward 里的
    # ``cmd.vy_max`` / ``cmd.yaw_max`` 门），所以「只往前走」的阶段与旧行为
    # 逐位一致。
    #
    # ⭐ 形状是**线性死区**、量纲是「按指令量程归一化的相对误差」，所以下面两个
    # 权重是「完美跟踪每步得多少分」，``*_sigma`` 不再参与计算（保留只为兼容
    # 旧配置与文档）。为什么是这个形状、以及为什么权重必须这么大量过三次 ——
    # 见 ``compute_reward`` 里 ``command_follow`` 那一段的 (a)(b)(c)。
    lateral_vel: float = 6.0            # 对 vy 指令的线性死区跟踪
    lateral_vel_sigma: float = 0.25     # m/s（历史字段，不再参与计算）
    yaw_rate: float = 4.0               # 对 wz 指令的线性死区跟踪
    yaw_rate_sigma: float = 0.25        # rad/s（历史字段，不再参与计算）

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
    cmd = env.cfg.command
    rpy = env.base_rpy
    lin = env.base_lin_vel_world
    target_height = env.nominal_pose
    # 指令跟踪读滤波量还是瞬时量，由观测开关决定（两者必须一致，否则策略看到的
    # 和它被评判的不是同一个数；见下面 1b 的说明）。
    vel_filter = bool(env.cfg.obs.vel_filter)

    terms: dict[str, float] = {}

    # 1) 前进速度跟踪 ------------------------------------------------------
    # ⭐⭐ **一致性**：这一项以前是**高斯核且只拿 ``cfg.command.vx`` 当目标**，
    # 而 ``vx`` 是**定值基线**、不随回合重抽（``_sample_command`` 返回
    # ``[cmd._now[0], vy, wz]``）。课程里 ``command_vx = 0.0`` 的那几级因此
    # 出现了一个致命的自相矛盾：**指令说「别前进」，但没有任何一项惩罚前进**。
    # 实测代价（`.tmp/eval_cmd_range.py`，cmdtrack4 第 1 级，vx 指令 0）：
    # 策略以 0.24 m/s 一路向前、13 步就走完 0.5 m；而横向指令 ±0.10 下
    # 实测只有 ±0.02 m/s。也就是说 45 分的横向跟踪收益，被「顺着先验往前冲」
    # 白拿的 93 分盖过去了 —— 策略当然选择不跟指令。
    #
    # 改成与横向/偏航**同一种形状**（按量程归一化的线性死区），三个方向就都
    # 是「误差越小分越高、且不动拿不到高分」：
    #   * 完美跟踪 1.0；原地不动 = 1 − |cmd|/range；
    #   * vx 指令为 0 时，原地不动拿满分 1.0，**往前冲直接掉分** ——
    #     「别前进」这句话第一次有了价钱。
    #
    # 量程取 ``max(|cmd.vx|, FORWARD_RANGE_FLOOR)``：``vx`` 是定值、没有
    # 「采样上界」可借（横向/偏航有 ``vy_max``/``yaw_max``），所以要用一个
    # 下限把「vx=0」时的量程撑开，否则分母为 0。
    #
    # ``cmd`` 在上面的取值区已经取出（``cmd = env.cfg.command``）。
    err = lin[0] - cmd.vx
    fwd_range = max(abs(float(cmd.vx)), FORWARD_RANGE_FLOOR)
    terms["forward_vel"] = rc.forward_vel * float(
        max(0.0, 1.0 - abs(err) / fwd_range)
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
    follow = 0.0
    # ⭐ 形状从「高斯核」一路改成「按指令量程归一化的线性死区」，中间试过二次型
    # （m10954 起）。三段证据缺一不可：
    #
    # (a) **高斯核不需要真的跟就能拿高分**。高斯只要求 ``|v - cmd| < 2 sigma``，
    #     不要求 v 跟 cmd 相关。实测（`.tmp/diag_follow_stepwise.py`，cmdtrack2
    #     第 1 级检查点）：指令 vy 是 ±0.4 上的均匀分布（std 0.21），策略的滤波
    #     vy 均值 +0.017、std 0.070 —— **几乎不随指令变化**（跨指令斜率 −0.028），
    #     但 ``command_follow`` 照样 +0.756/步。它优化的是**分布的巧合**，
    #     不是相关性；σ 从 0.25 收到 0.08 反而让这种巧合更值钱。
    # (b) **二次型梯度太小**。换成 ``1 - (err/range)^2`` 后，同一个检查点仍然是
    #     「站着不动」：`scripts/cmd_demo.py --phase inplace` 实测横向斜率 +0.005、
    #     相关系数 +0.007，而原地不动在数学上就该拿 2/3 满分（实测单步 2.66 /
    #     满分 6.0），均回报 2659 ≈ 2.66×1000 —— 与「一动不动」逐位吻合。
    #     算一下梯度就知道为什么：``dReward/dv = -2W(v-cmd)/range^2``，W=4、
    #     range=0.4 时是 13.75·(cmd−v)；而 ``|a|`` 变化一个单位大约改 0.435 m/s
    #     （`.tmp/action_authority.py` 的 scale=0.30 行），所以**每单位动作只有
    #     约 1.35 分**，比 PPO 的探索噪声大不了多少，策略于是停在原地。
    # (c) **线性形状把梯度提到与误差无关的常数**：``-W/range``。W=6、range=0.4
    #     时是每 m/s 15 分 = **每单位动作约 6.5 分**（二次型的 5 倍），而且
    #     误差很大时也不塌缩成 0（二次型在 ``|err| > range`` 后梯度为 0）。
    #     代价是多一个折点 —— 这不重要，PPO 只吃梯度方向。
    #
    # 量程取**指令自己的上界**，所以它是「相对误差」而不是「绝对误差」：
    # 完美跟踪对 ±0.4 的均匀指令平均 1.0，原地不动降到 0.5（=−|cmd|/range 的
    # 期望），且**大指令罚得最重** —— 没有「躲在 0 附近」这条路。
    #
    # ``lateral_vel_sigma`` / ``yaw_rate_sigma`` 因此**不再参与计算**，只作为
    # 历史字段保留在 ``RewardConfig`` 里（配置文件与文档仍会写它们）。
    cmd = env.cfg.command
    follow = 0.0
    # ⭐ 读**滤波后**的量（``ObsConfig.vel_filter`` 打开时）。
    #
    # 起因是一条实测（`.tmp/diag_cmd_jitter.py`）：瞬时 ``yaw_rate`` 在小跑步态下
    # ``mean = -0.027`` 而 ``std = 0.370 rad/s`` —— 机器人其实**没有转**，但瞬时
    # 值每秒扫过指令区间几十次，``command_follow`` 照样给 +0.81/步（整回合
    # 净位移 Δy 只有 +0.05 m）。也就是说判据当时读的是「步态摆动」，不是
    # 「机器人在往哪走」。滤波时间常数 ``cfg.vel_filter_tau``（默认 0.2 s）压掉
    # 一个步态周期，同时把同一个量放进观测以保持马尔可夫性。
    #
    # 这也是「奖励项上线前要量噪声底」（experience.md §16）的第三次翻版：
    # 前两次是可达性（§15）与抖动的导数（§16），这一次是**被测量本身不合格** ——
    # 物理上正确的瞬时量不一定是判据该读的量。
    if cmd.vy_max:
        vy = (env.filtered_lateral_vel if vel_filter else env.base_lin_vel_body[1])
        follow += rc.lateral_vel * float(
            max(0.0, 1.0 - abs(vy - cmd.vy) / float(cmd.vy_max))
        )
    if cmd.yaw_max:
        wz = env.filtered_yaw_rate if vel_filter else env.yaw_rate
        yaw_err = abs(wz - cmd.wz)
        # ⭐ 除了「跟指令」这一段，再补一个**零指令基线惩罚**（只在 ``wz=0``
        # 时生效）。原因是实测：cmdtrack4 第 1 级只练横向（``yaw_max=0``），
        # 策略却能以 −0.10~−0.20 rad/s 稳定自转（`cmd_demo --phase inplace`
        # 的「站住」段实测 wz=−0.200），因为那一级**没有任何一项读 wz**。
        # 带上这个基线后，「指令让我不转」和「指令让我转」都有价钱。
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
