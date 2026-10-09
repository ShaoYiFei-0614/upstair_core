"""Gymnasium environment: Unitree Go2 walking up and down box stairs.

Implements exactly the interface the user specified:

* **Action** -- 12 joint target angles (``FL_hip .. RR_calf``).
* **Observation** -- joint positions, joint velocities, base roll/pitch/yaw,
  angular velocity, previous action; the last ``history_len`` frames are
  stacked.  **No contact-sensor data is used.**
* **Reward** -- forward velocity, posture stability, joint torque, foot lift,
  body collision (plus a few standard shaping terms, all weight-configurable).

Frame conventions (verified empirically in ``scripts/verifications/step2_go2.py``):

* ``qpos[0:3]``  base position, **world** frame.
* ``qpos[3:7]``  base orientation quaternion ``(w, x, y, z)``, world->body.
* ``qvel[0:3]``  base linear velocity, **world** frame.
* ``qvel[3:6]``  base angular velocity, **LOCAL (body)** frame -- this is the
  value a real IMU gyro reports, so it is used directly for the observation.

Control runs at ``1 / (timestep * decimation)`` Hz (default 50 Hz) while physics
runs at 500 Hz.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from robotdog.gait import GaitParams
from robotdog.go2_model import (
    SceneOptions,
    compile_scene,
    default_floor_geom,
    home_qpos,
    quat_to_mat,
    quat_to_rpy,
)
from robotdog.terrain import StairSpec, build_terrain_xml
from robotdog.touch import (
    TouchConfig,
    TouchReading,
    TouchSensor,
    empty_reading,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class ObsConfig:
    """Observation layout. Defaults match the user's specification exactly."""

    history_len: int = 10  # 10~20 stacked frames
    # Extra quantities that are useful but were not requested. Off by default
    # so that the observation matches the spec; enable for harder terrains.
    base_lin_vel: bool = False
    # Body-frame linear velocity (3 values).  ``base_lin_vel`` above is the
    # *world* frame velocity, which is a poor feature: it changes meaning as the
    # robot yaws, so a policy cannot learn "move forward" from it.  The body
    # frame version is what legged-RL stacks normally feed.
    #
    # Measured motivation (course12, 630-dim obs): the reward already contains
    # ``lateral_vel``, ``lateral_vel_track``, ``yaw_rate`` and ``heading`` --
    # four terms that are pure functions of vy and wz -- but with this block off
    # the policy could not observe vy or wz at all.  It was being graded on
    # quantities it could not see, and had to infer them from joint velocities
    # and a 10-frame history.  That is the main reason the "walk straight"
    # shaping had so little authority.
    base_lin_vel_body: bool = False
    base_height: bool = False
    # 高度参照改用**四足足下最低地形**而不是机身 x 处的单点高度。
    #
    # 默认 False = 历史行为。`.tmp/signal_audit.py` 实测（README §11.22）：
    # `base_height` 的原口径 `self.base_pos[2] - self.terrain_height_under_base`
    # 里那一项是**阶跃函数**（terrain.py:222-238），所以躯干 x 越过立面时
    # 这一维会在**单帧内跳 0.11 m**，而且躯干在立面附近来回震荡时会**每一步
    # 反复翻转**（实测样本 `base_x=1.503 Δref=+0.1100` 紧接
    # `base_x=1.491 Δref=-0.1100`）。打开后改用 support 口径，无跳变。
    base_height_support: bool = False
    foot_height: bool = False
    # Gait-clock block (2 values: sin/cos of the gait cycle).  Only meaningful
    # together with ``EnvConfig.gait_prior``: a residual policy on top of a
    # periodic prior cannot be optimal if it cannot tell where in the cycle it
    # is.  Off by default so that the plain observation matches the spec.
    gait_phase: bool = False
    # --- step 11: tactile + rangefinder blocks ----------------------------- #
    # 9 values: 4 weight-normalised foot normal forces, 4 slip ratios (tan/
    # normal) and 1 "some foot is shoving against a riser" flag.  This is the
    # contact information the original spec deliberately excluded; it is what
    # lets the policy feel *how hard* each foot is pressing instead of only
    # whether it touches.
    touch: bool = False
    # 3 values: how tall the obstacle ahead is, how close it is, and whether the
    # forward rangefinder hit anything at all.  This is the "检测到楼梯" half.
    stair_probe: bool = False
    # 追加 2 维（probe_slope, probe_far_step）。这两个量**本来就被算出来了**
    # （`robotdog/touch.py:409-410`），只是没进观测；默认 False = 历史 660 维。
    stair_probe_full: bool = False
    # 2 values: lateral offset from the **stair centre line** and heading error
    # relative to the **stair axis** (yaw = 0).  Without them the policy has no
    # way to observe that it is drifting sideways.
    #
    # These used to be measured against the episode's own spawn pose, which
    # quietly turned the "walk straight" reward into "keep the random heading you
    # were thrown in with": the reset samples yaw uniformly in
    # +/-reset_yaw_noise (0.10 rad = +/-5.7 deg) and y in +/-reset_xy_noise, so a
    # policy that perfectly holds its spawn pose still walks up to 0.16 m off the
    # centre line at 5.7 deg.  Measured on flat, 30 episodes: |y|max = 0.071 m
    # mean / 0.159 m worst, |yaw - yaw0| = 2.97 deg mean / 7.28 deg worst, but
    # "last-20%-yaw minus first-20%-yaw" = -0.53 deg, i.e. the robot was not
    # curving at all -- it was faithfully holding its spawn heading.
    drift: bool = False
    # 4 values: how far the two FRONT (and then the two rear) feet are from the
    # height they must reach to clear the step in front, normalised by that
    # target.  ``0`` = the foot has already reached the required height this
    # swing, ``1`` = this swing's peak is zero.
    #
    # Motivation (measured on the reference course22 policy, `.tmp/probe_kin.py`
    # + `.tmp/analyze_kin.py`, 8.0 vs 11.0 cm, 3 episodes each): the swing-foot
    # peak clearance is ~0.091-0.093 m at 8 cm but only ~0.079-0.085 m at 11 cm
    # -- i.e. the prior/policy lifts the foot to a nearly CONSTANT height and
    # even lifts it *lower* on the taller stair.  At 8 cm 101/135 swings clear
    # the step; at 11 cm only 1-8/236 do, so the foot simply rams the riser
    # (11 cm episodes spent 54% of their steps grazing a riser, and their
    # timeout episodes never got past max_x = 1.88 m).  The policy cannot fix
    # this because nothing in the 660-dim observation says *how much more* lift
    # is required: ``foot_height`` is measured against the terrain directly
    # under the foot, which reads ~0 on both sides of a riser, and the 3-dim
    # ``stair_probe`` only describes the step ahead of the *base*, not the
    # deficit of each individual foot.  With this block the deficit is explicit
    # and per-foot.
    #
    # Measured separation with the lookahead reference (`.tmp/foot_trace.py`,
    # reference course22 policy, 1 episode each, target = 1.0 * step height):
    #   8.0 cm: swing peak p50 = 0.103 m vs goal 0.080 -> deficit 0 (satisfied)
    #   11.0 cm: swing peak p50 = 0.084 m vs goal 0.110 -> deficit 0.24
    # so this 4-dim block is the signal that actually tells the policy the 11 cm
    # stair is out of reach, which the 660-dim observation never did.
    foot_lift_gap: bool = False
    # ``foot_lift_deficit``（4 维，默认 False）：每只脚**此刻**还要再抬高多少才能越
    # 过它将踩上去的那一级，按 ``required_lift()`` 归一化后 clip 到 [0, 1]。
    #
    # 与上面 ``foot_lift_gap`` 的分工是「回顾 / 当下」：
    #   foot_lift_gap      = 这只脚**上一轮摆动**离目标还差多少（落地清零，只在摆动
    #                        结束后才有非零值）。它携带的是「我刚才那次尝试的结果」。
    #   foot_lift_deficit  = 这只脚**此刻**离目标还差多少（瞬时值）。它携带的是
    #                        「现在抬腿还来不来得及」。
    # 单独给任何一个都不够：只有 gap，策略要等下一次落地才知道失败原因；只有
    # deficit，策略在支撑相看到的是一个恒定的、与自己刚做的动作无关的数。
    #
    # 用 ``advance_clearance()`` 而不是 :attr:`foot_clearance`，理由见
    # ``RewardConfig.foot_placement_reach``：后者的参照面在立面处是阶跃的，11 cm
    # 上给出的绝对读数反而比 8 cm 还大，作为观测完全没有区分度。
    foot_lift_deficit: bool = False
    # Per-frame size: q(12) + qd(12) + rpy(3) + omega(3) + prev_action(12)
    base_frame_size: int = 42

    @property
    def frame_size(self) -> int:
        extra = 0
        extra += 3 if self.base_lin_vel else 0
        extra += 3 if self.base_lin_vel_body else 0
        extra += 1 if self.base_height else 0
        extra += 4 if self.foot_height else 0
        extra += 2 if self.gait_phase else 0
        extra += 9 if self.touch else 0
        extra += 3 if self.stair_probe else 0
        extra += 2 if (self.stair_probe and self.stair_probe_full) else 0
        extra += 2 if self.drift else 0
        extra += 4 if self.foot_lift_gap else 0
        extra += 4 if self.foot_lift_deficit else 0
        return self.base_frame_size + extra

    @property
    def size(self) -> int:
        return self.frame_size * self.history_len


@dataclass
class RewardConfig:
    """Reward weights. The five terms the user asked for are marked [SPEC]."""

    # 1. 前进速度奖励 -- [SPEC] track a forward velocity command
    forward_vel: float = 2.5
    forward_vel_sigma: float = 0.22
    # Linear "any forward motion pays" term.  The Gaussian above still pays
    # 0.22*forward_vel for standing perfectly still, which together with the
    # survival bonus created a local optimum where the policy simply parked
    # itself (measured: 120k flat steps -> 0.00 m/s, 0.02 m travelled, while a
    # zero-action stance scored MORE than the learned policy).  This term is
    # exactly zero at vx=0 and grows linearly, so the gradient never
    # disappears, and it is clipped so the policy cannot gain by diving.
    forward_progress: float = 1.0
    forward_progress_max: float = 1.5  # clip vx at this value in the linear term
    forward_progress_min: float = -0.5  # clip backwards motion
    # 2. 姿态稳定奖励 -- [SPEC] keep the torso level
    orientation: float = -1.2
    # 3. 关节力矩惩罚 -- [SPEC]
    torque: float = -2.0e-4
    # 4. 足端抬升奖励 -- [SPEC] lift the swing foot clear of the terrain
    foot_lift: float = 0.6
    # ``foot_lift_target`` 是「抬起多少算满分」的**地板**，不是全部。
    #
    # ⛔ 原来这里只有一个固定的 0.08 m, 于是 ``lift = clip(clearance/0.08, 0, 1)``
    # 在 8 cm 处**完全饱和**: 再抬高一毫米, ``foot_lift`` 与 ``stair_probe_lift``
    # 都一分不多给。实测的墙正好在这里 —— course13 卡 7.3cm(4%), course14 卡
    # 7.8cm(8%), 而 6.4cm 能到 80%。纯开环先验更惨: 4cm 40%, 5cm 10%, >=6cm
    # 任何 gain (1.0/1.3/1.6/2.0) 都是 0% (``.tmp/prior_probe.py``)。
    #
    # 所以真正的抬升目标必须跟着**地形**走: 要踏上 h 米的台阶, 足端至少要比
    # 脚下那一级高出 h。``foot_lift_step_ratio`` 就是那个比例系数, 有效的目标
    # 高度 = ``max(foot_lift_target, ratio * 台阶高)``。
    # ratio=1.4 时: 5cm 台阶 -> max(8, 7)=8cm (与旧行为**完全一致**, 所以已经
    # 学会的 easy 阶段不会被改坏); 8cm -> 11.2cm; 12cm -> 16.8cm。
    foot_lift_target: float = 0.08
    foot_lift_step_ratio: float = 1.4
    # ``foot_lift_target_cap``：有效抬升目标的上限 (m)，0 = 不设限。
    #
    # ratio=1.4 时 12 cm 台阶要求 16.8 cm 的足端离地，已经接近这条腿的几何
    # 极限；再往上（19 cm -> 26.6 cm）就是**不可能满足**的目标，梯度会变成
    # 常数推力而不是「再抬一点」。设 0.13~0.16 可以把目标压回可达区。
    foot_lift_target_cap: float = 0.0
    # ``foot_lift_goal_ratio``：**摆峰门槛**（``foot_lift_clear`` / ``obs.foot_lift_gap``）
    # 用的目标 = ``max(foot_lift_target, goal_ratio * 台阶高)``。
    #
    # 为什么与连续项的 ``foot_lift_step_ratio = 1.4`` 分开: 两者量的不是同一件事。
    #   连续项 ``foot_lift`` 用的是**逐步瞬时**离地间隙，量的是「这只脚此刻比脚下
    #   地面高多少」，1.4 倍是针对「必须抬过头顶才能真正跨过去」的老口径标定的。
    #   摆峰门槛用 ``advance_clearance()``（脚前方探路），它问的是「这只脚能不能
    #   越过前方那一级」，物理上**恰好 1.0 倍台阶高**就已经跨过去了。
    # 实测（参考 course22 权重，8/11 cm 各 1 回合，`.tmp/foot_trace.py`）以
    # ``foot_lift_goal_ratio = 1.0`` 计:
    #   8.0 cm: 摆峰 p50 = 0.103 m，目标 0.080 -> 达标;
    #   11.0 cm: 摆峰 p50 = 0.084 m，目标 0.110 -> **差 0.026 m（24%）**。
    # 用旧口径（正下方地形 + 1.4 倍）这个区分是**反的**：8 cm p50 0.125 vs
    # 目标 0.112（比 1.11），11 cm p50 0.103 vs 目标 0.154（比 0.67）—— 比值虽然
    # 也区分得开，但绝对读数在 11 cm 上反而**更大**，与「11 cm 抬不起来」的事实
    # 相反，奖励里那 0.6 分的连续项因此在 11 cm 上给得比 8 cm 还多。
    foot_lift_goal_ratio: float = 1.0
    # ``foot_lift_support_ref``：抬升奖励/亏欠观测的**参照面**口径。
    #
    #   False (历史) : ``self.foot_clearance`` = 足尖减**该脚正下方**的地形高。
    #                  问题是那个参照面是阶跃函数：脚在立面**正前方**时它读到的是
    #                  下一级的高度，脚一旦越过立面上方，同一个离地高度会瞬间
    #                  **少算一个台阶高**（`terrain.py:222-238 height_at`）。于是
    #                  「已经跨过台阶的脚」和「还贴在地面上的脚」可能给出同样的
    #                  clearance 数值 —— 奖励分不清这两者。
    #   True         : 改用 :meth:`advance_clearance` 的**落脚面**口径，见那里的
    #                  注释。名字保留是为了兼容已经写下的配置。
    foot_lift_support_ref: bool = False
    # ``foot_placement_reach``：判断「这只脚接下来要踩到哪一级」时，沿机身 x 方向
    # 向前看到的距离 (m)，以足半径为基数再加这一段。
    #
    # 这是这次改动里最关键的一条。原来的 ``foot_clearance`` 问的是「这只脚比它
    # **正下方**的地面高多少」，而台阶处真正决定成败的问题是「这只脚够不够高，
    # 能不能越过**它即将踩上去的**那一级」。后者才是「11 cm 是墙」的判据：在
    # 立面正前方时脚的正下方还是下一级，所以原口径会给出一个与任务无关的读数。
    #
    # 参照面 = ``max(脚当前位置, 前方 reach/2, 前方 reach`` 三处的踏面高 ``)``，
    # 只用**位置**决定，与脚当前高度无关，所以同一个抬腿动作的读数不会在跨过立面
    # 的那一帧发生跳变。取 0.10 m（加足半径约 0.13 m）：足端每控制步前进约
    # 0.004 m，0.10 m 相当于提前约 25 步看到下一级，足够让策略在抬腿阶段就收到
    # 「还不够高」的信号；再远就会开始看到**下一级之后**的台阶，给出这一轮摆动
    # 无论如何都达不到的目标（那是下面 ``foot_lift_goal_ratio`` 要处理的事）。
    foot_placement_reach: float = 0.10
    # ``foot_lift_clear``：**摆动峰值**抬升奖励（默认 0.0 = 关闭），与 ``foot_lift``
    # 的逐步连续项并列，不替代它。
    #
    # 为什么需要连续项之外再要一个峰值项: 11 cm 上 `foot_lift` 的实测满分线是
    # 1.4*11 = 15.4 cm，而参考策略的摆动峰值只有 ~8 cm，于是它拿到的
    # ``clip(8/15.4, 0, 1)`` 是 0.52 —— 已经**过半**。逐步连续项因此一直在奖励
    # 「再多抬一点点」，却从不告诉策略「现在这样是**不够**的」。峰值项在**落地那一
    # 步**一次性结算整个摆动过程的最高点，而且可以直接用二值门槛
    # （``foot_lift_clear_binary``），于是「抬到 8 cm」和「抬到 11 cm」的差别从
    # 0.52 -> 0.71 的渐变变成 0 -> 1 的阶跃。
    foot_lift_clear: float = 0.0
    # True = 峰值项用 0/1 门槛（峰值 >= 目标才付钱），False = 用 ``clip(peak/target)``
    # 的连续比例。两者都只在落地那一步结算。
    foot_lift_clear_binary: bool = False
    # 摆动门控的阈值 (m/s, 三维足端速度)。原来只看水平速度、阈值 0.35 m/s:
    # 高台阶需要的是**慢慢抬腿**, 水平速度很小, 于是门控把 credit 掐掉,
    # 等于鼓励「快速平扫」而不是「抬高跨越」。加上竖直分量并降低阈值。
    foot_lift_gate_speed: float = 0.25
    # Feet-air-time shaping: pay a foot when it LANDS after a swing of roughly
    # ``air_time_target`` seconds, and claw the payment back if the swing drags
    # on.  This is the single strongest gait-discovery signal in the usual
    # legged-robot reward sets: without it a policy can satisfy every other term
    # by shuffling or by standing still, because nothing ever asks two feet to
    # be in the air and two on the ground at the same time.
    air_time: float = 0.6
    air_time_target: float = 0.30  # s of flight that earns full credit
    air_time_max: float = 0.60  # s beyond which the credit is fully revoked
    # 5. 机身碰撞惩罚 -- [SPEC] penalise the torso touching the terrain
    body_collision: float = -4.0
    # 第二十四步: 高台阶上把躯干蹭立面的**惩罚**缩小 (终止判据侧由
    # ``EnvConfig.trunk_contact_adaptive`` 控制)。系数乘在 ``body_collision`` 上:
    # ``1.0`` = 历史行为 (默认, 整段逐位相同); ``0.25`` = 只罚四分之一。
    # 只在 ``EnvConfig.trunk_contact_adaptive=True`` 且
    # ``stair.step_height >= EnvConfig.trunk_contact_hi_step_h`` 时生效。
    body_collision_hi_scale: float = 1.0
    # --- supporting shaping ------------------------------------------------ #
    lateral_vel: float = -0.8  # stay on the centre line
    yaw_rate: float = -0.15  # do not spin
    base_height: float = -3.0  # keep the nominal stance height
    joint_vel: float = -2.0e-5  # smooth motion
    action_rate: float = -0.02  # penalise jerky targets
    joint_limit: float = -0.5  # soft joint-limit avoidance
    # Survival bonus.  Kept small on purpose: the 120k-step flat calibration
    # showed that alive=0.4 was 49% of a standing-still episode's return, which
    # is what made "do nothing" so attractive.
    alive: float = 0.12
    termination: float = -2.0  # fell / collided
    # Terminal bonus for actually reaching the goal (the staircase crest, or the
    # flat-stage target distance).  Recorded in ``info["reward_terms"]["terminal"]``
    # so ``sum(reward_terms) == reward`` keeps holding on terminal steps too.
    success: float = 20.0
    # Task progress: extra reward for advancing up the staircase
    stair_progress: float = 2.5
    # 参照面口径 + 承重门控（默认 False/0 = 历史行为，保证老 checkpoint 可复现）。
    #
    # 历史行为的问题（`.tmp/signal_audit.py` 实测，README §11.22）：这一项引用的是
    # **机身 x 处的单点阶跃高度**，所以躯干一越过第 k 级立面，之后**每一步**都白拿
    # `stair_progress * k * h`（11 cm 上第 8 级 = +2.2/步，是 ``alive`` 的 18 倍，
    # 也超过 ``forward_vel`` 的满值 2.5）。梯度于是指向「把躯干怼过 x 线」，而不是
    # 「把脚放到上一级」—— 11 cm 上正好就是「躯干过线、脚没过线，然后摔」。
    #
    # ``stair_progress_foot_ref=True`` 改用**四足足下最低地形高度**（与
    # ``_fall_reference_height()`` 的 "support" 口径同一个量），奖励「脚真的站在
    # 更高一级上」；再叠加 ``stair_progress_min_loaded_feet``（至少 N 只脚的法向力
    # 超过 ``stair_progress_force_eps``）把「躯干冲线」这条旁路彻底堵掉。
    stair_progress_foot_ref: bool = False
    stair_progress_min_loaded_feet: int = 0
    stair_progress_force_eps: float = 5.0  # N
    # **真实爬升奖励**（默认 0.0 = 关闭）：四足足下最低地形高度相对上一步的**增量**，
    # 除以台阶高后 clip 到 [0, 1]。历史奖励里没有任何一项奖励「脚真的上了一级」：
    # ``foot_lift`` 奖励离地间隙，``stair_probe_lift`` 奖励看到台阶，``stair_progress``
    # 奖励躯干过线。这一项补上缺的那块 —— 且因为是**增量**，只在真正跨级那一步付钱。
    foothold: float = 0.0

    # ------------------------------------------------------------------ #
    # 触觉 / 碰撞 (step 11)
    # ------------------------------------------------------------------ #
    # A foot shoving sideways against a **vertical riser** (contact normal has a
    # small world-z component).  This is the literal "撞到台阶了" event and the
    # clearest possible signal that the foot should have been lifted.
    riser_graze: float = -0.60
    # 蹭立面的门控模式（默认 "legacy" = 历史行为）。
    #   "legacy" : 任意一只脚蹭到立面就罚。`signal_audit` 实测该事件在 10/10.5/11 cm
    #              上分别是 5.7% / 13.1% / **31.9%** 的控制步，平均罚
    #              -0.035 / -0.079 / **-0.192** 每步 —— 11 cm 上这一项已经超过
    #              ``foot_lift`` 的全部实得（+0.129/步）。而物理上要爬上 11 cm 的
    #              立面，脚/小腿**本来就必须先蹭到它**，奖励因此与任务方向打架。
    #   "lifting": 只有当**所有**蹭立面的脚都没有明显向上速度（真卡住）时才罚；
    #              正在抬起跨越立面的脚免罚。
    riser_graze_mode: str = "legacy"
    riser_graze_vz_eps: float = 0.10  # m/s，判定「脚在往上走」的垂直速度阈值
    # ``riser_graze_probe_gate``：只在**前方测距确实看到台阶**时才罚蹭立面
    # （默认 False = 历史行为，不看测距）。
    #
    # 为什么需要: 蹭立面的检测只看接触法向的世界 z 分量，在**平地**上小腿/髋部
    # 与地面的边角接触也会被算成「蹭立面」。平地段（课程第一档，也是整个课程里
    # 唯一一开始就能拿满分的档）因此一直拿到一个与任务无关的常数惩罚，而它恰恰
    # 是策略最早期、最脆弱的阶段。用前方测距命中做门控后，平地上的假阳性被清掉，
    # 真台阶上的惩罚不变（测距在台阶前 0.62 m 内必然命中）。
    riser_graze_probe_gate: bool = False
    # Hard landing: quadratic excess of the touchdown normal force over
    # ``touch_impact_target``.  Soft landings are what keep the trot from
    # hammering the knees on the way down a staircase.
    touch_impact: float = -0.50
    touch_impact_target: float = 95.0  # N, roughly 2.5x a single foot's share
    # ``touch_impact_step_scale``: 让上面那个目标力随**台阶高度**放大。
    #
    # 为什么需要: 落地的法向力天然随落差增长, 而 ``touch_impact_target`` 是常数
    # 95 N。台阶从 5 cm 涨到 11 cm 后, 同样的好落地也会越过 95 N, 于是这个平方
    # 惩罚在高台阶上从「提醒别砸膝盖」变成「罚所有必要的动作」——
    # `.tmp/signal_audit.py` 实测 11.0 cm 上它已经吃到 -0.187/步。
    #
    # 口径: ``ref = touch_impact_target * (1 + scale * (step_height/0.05 - 1))``。
    # 5 cm 台阶下因子恒为 1.0, 所以**只要 scale=0 就与历史行为逐位相同**;
    # scale=0.5 时 11 cm 的目标力是 95 * 1.6 = 152 N。
    touch_impact_step_scale: float = 0.0
    # Dense slip penalty (tangential/normal), applied only to feet that are
    # actually carrying weight -- gauging on force avoids punishing a foot that
    # is legitimately swinging through the air.
    touch_slip: float = -0.35
    touch_slip_safe: float = 0.55
    # Cap on the *normalised* excess that both squared tactile penalties are
    # allowed to see, per foot per step.  Contact forces are unbounded in
    # MuJoCo when a foot is jammed against a riser edge, so without this a
    # single step produced a penalty of O(10^5) and poisoned the value function
    # for an entire course stage.
    touch_excess_clip: float = 5.0
    # The positive half of the same lesson: pay the robot when the rangefinder
    # sees a step coming **and** the front feet are already lifted for it.
    # This is 检测到楼梯就抬腿, expressed as a reward.
    #
    # ``stair_probe_lift_use_advance``（默认 False = 历史行为）把这里的 ``lift``
    # 从「正下方地形参照」换成 ``advance_clearance()`` 的**前方探路**口径，与
    # ``foot_lift_clear`` 一致。需要换的理由与 ``foot_lift_lookahead`` 相同：正
    # 下方口径在脚跨过立面时会把同一个离地高度**少算一个台阶高**，于是这一项在
    # 「看到台阶就抬腿」这个最需要它的时刻反而给不出分。
    stair_probe_lift: float = 1.20
    stair_probe_lift_use_advance: bool = False
    # Per-step penalty when a *leg* geom (hip / thigh / calf shape, never the
    # torso, never a foot sphere) is in contact with the terrain.
    #
    # This replaces "any leg contact instantly ends the episode with -2.0".
    # Measured consequence of the old rule (course12, 30 episodes per terrain):
    # at 10 cm x 8 the policy averaged 62.6 steps and never got past x = 1.42 m,
    # while the first riser is at x = 1.50 m -- it was killed by a shin brushing
    # the riser *before its feet had reached a single tread*, so it had nothing
    # to learn from and correctly converged to "do not even try".  A bounded
    # per-step penalty keeps the "do not scrape the stairs" gradient without
    # removing the trial-and-error the height curriculum needs.
    leg_collision: float = -0.25

    # ------------------------------------------------------------------ #
    # 走直线 (step 11)
    # ------------------------------------------------------------------ #
    # Lateral offset from the spawn line.  Penalising the *position* error, not
    # only the velocity, is what actually fixes the drift: ``lateral_vel`` alone
    # can sit near zero on average while y still walks away over a 20 s episode.
    #
    # Measured motivation (see .tmp/straight_probe.py): with the old weight of
    # -2.0 the whole straight-line term contributed only ~0.65% of a 200-point
    # episode return, and a 100k-step policy was indistinguishable from the
    # unshaped baseline.  -20.0 puts it at roughly half the return at |y| = 0.2 m
    # while ``straight_line_max`` keeps the worst case bounded -- the same lesson
    # the tactile penalties taught (an unbounded penalty destroys the value
    # function; there is no reward term that is safe to leave uncapped).
    straight_line: float = -20.0
    straight_line_max: float = 1.00  # m, |y| beyond this is already "off the stairs"
    # Heading error, so the robot cannot "fix" its lateral error by yawing away
    # and walking diagonally.
    heading: float = -8.0
    heading_max: float = 0.60  # rad (~34 deg)
    # Dense command-tracking term exp(-(vy - command_vy)^2 / sigma^2); unlike a
    # pure penalty it never collapses to "do nothing" because it pays a bonus.
    lateral_vel_track: float = 0.5
    lateral_vel_sigma: float = 0.15


@dataclass
class EnvConfig:
    """Everything tunable about one environment instance."""

    timestep: float = 0.002
    decimation: int = 10  # -> 50 Hz control
    episode_length_s: float = 12.0
    action_mode: str = "residual"  # "residual" (offset from home pose) | "absolute"
    action_scale: float = 0.30  # rad, only for "residual"
    command_vx: float = 0.55  # m/s forward command
    # Lateral + yaw command, both zero: the task is "walk straight up the
    # staircase".  Exposed as commands rather than hard-coded zeros so that the
    # same code can later steer around obstacles.
    command_vy: float = 0.0  # m/s sideways command
    command_yaw_rate: float = 0.0  # rad/s yaw-rate command
    # --- residual RL over an open-loop gait (see robotdog/gait.py) ---------- #
    # Off by default.  When on, the PD target becomes
    #     home + gait_gain * gait.radians(t) + action_scale * action
    # so the policy learns a correction on top of a working trot instead of
    # having to discover locomotion from nothing.  Measured on flat: 120k and
    # 150k from-scratch steps -> 0% success and a policy worse than doing
    # nothing; the searched trot alone walks 0.550 m/s with 100% survival.
    gait_prior: bool = False
    gait: GaitParams = field(default_factory=GaitParams.default_trot)
    gait_gain: float = 1.0  # <1 leaves the policy more authority over the prior
    gait_phase_randomize: bool = False  # random start phase per episode
    kp: tuple[float, float, float] = (30.0, 30.0, 45.0)
    kd: tuple[float, float, float] = (0.6, 0.6, 0.9)
    # Which terrain contacts end the episode.
    #   "trunk" (default) -> only the TORSO touching the terrain is fatal; that
    #       is a real fall.  Leg geoms scraping a riser are penalised through
    #       ``RewardConfig.leg_collision`` but the episode keeps running.
    #   "any"             -> legacy: any non-foot robot geom touching terrain is
    #       fatal.  Kept so old runs stay reproducible.
    #   "none"            -> contacts never terminate; only falls do.
    # Measured reason for the default (course12, 30 episodes per terrain, goal +
    # termination census, see .tmp/diag_v2.py):
    #     5 cm x 8  63.3% success, terminations {success 19, leg/body 7, fall 3}
    #     6 cm x 8   6.7% success, terminations {leg/body 21, fall 7, success 2}
    #    10 cm x 8   0.0% success, terminations {leg/body 30}, max x 1.42 m
    #    15 cm x 8   0.0% success, terminations {leg/body 30}, max x 1.38 m
    # The first riser is at x = 1.50 m, so on tall stairs the robot was being
    # killed by its own shin before it ever reached a tread.
    terminate_contact: str = "trunk"
    terminate_body_contact: bool = True  # legacy switch, honoured for "any"
    # How many *consecutive* control steps the selected contact class must
    # persist before it counts as fatal.  ``1`` == the historical behaviour:
    # the very first step with a trunk contact ends the episode.
    #
    # Measured reason for this knob (.tmp/stop_probe.py, 20 deterministic
    # episodes x 9.0/9.5/10.0 cm x 8, SEED0=21000, course19 checkpoints):
    #   medium4 @ 10.0 cm   trunk mode: 15% success / 160 steps
    #                       none  mode: 15% success / 287 steps
    #   medium5 @  9.0 cm   trunk mode: 35% success / 246 steps
    #                       none  mode: 35% success / 271 steps
    #   medium5 @  9.5 cm   trunk mode: 30% success / 208 steps
    #                       none  mode: 30% success / 276 steps
    #   medium5 @ 10.0 cm   trunk mode: 20% success / 194 steps
    #                       none  mode: 20% success / 334 steps
    # On the very first step carrying a trunk contact the *geometric* fall
    # criterion was already satisfied in 0 of 14 / 12 / 7 / 14 episodes: base
    # height ~0.26 m (standing is ~0.27 m), |pitch| 12-16 deg, so the robot was
    # standing upright with its chest brushing the first riser, not falling.
    # Killing that instantly shortens the episode by ~35% and removes exactly
    # the "chest against the riser, still hunting for a foothold" window the
    # policy needs to learn from.  Setting this to 10 (0.2 s) lets a graze pass
    # while a real collapse is still caught inside the window by
    # ``terminate_min_height`` / ``terminate_max_tilt``.
    terminate_contact_steps: int = 1
    # --- 第二十四步: 按台阶高度自适应的接触去抖 ---------------------------
    #
    # 动机 (README §11.23 / §11.24): ``terminate_contact_steps`` 扫出来的结论不是
    # 「越大越好」也不是「越小越好」，而是**随台阶高度换向**：
    #
    #   - 8~10.5 cm 上, 躯干蹭立面是**真卡住**的前兆, 去抖 10 (0.2 s) 最好：
    #     它剪掉正在失败的回合, 而不是把「胸口贴立面找落脚点」那 0.2 s 也杀掉。
    #   - 11.0 cm 上反过来, 去抖 1 (立刻判死) 明显更好：40 回合确定性评估
    #     10.0% -> 27.5%, 配对 8:1, p=0.020 (course40, 唯一显著为正的一档)。
    #
    # 机理: 阈值不是「严格 vs 宽松」, 而是一个**课程信号**。小台阶上给 0.2 s
    # 让策略学会「蹭住也能救回来」; 大台阶上这种「救回来」的姿态本身会毁掉
    # 下一步的落脚机会, 长去抖等于奖励硬撑。既然每种高度要的信号相反, 就让
    # 阈值跟着 ``stair.step_height`` 走。
    #
    # 默认 ``False`` ⇒ 整段与历史行为**逐位相同** (course22 复现不受影响)。
    contact_debounce_adaptive: bool = False
    # 台阶高 ≥ 这个值 (m) 时改用 ``contact_debounce_hi_steps``。
    contact_debounce_hi_step_h: float = 0.105
    # 高台阶上的去抖步数 (控制步, 50 Hz)。
    contact_debounce_hi_steps: int = 1
    # --- 第二十四步: 躯干蹭立面的判据条件化 -------------------------------
    #
    # 只对 ``terminate_contact = "trunk"`` 生效: 台阶高 ≥
    # ``trunk_contact_hi_step_h`` 时, 躯干接触**不再终止回合** (仍然照常扣
    # ``RewardConfig.body_collision`` 的分, 可以用
    # ``RewardConfig.body_collision_hi_scale`` 缩放)。
    #
    # 动机: 11 cm 上唯一可行的那条路是「蹭着立面上」, 而现在的惩罚结构让它的
    # 期望收益为负 —— 12 cm / 19 cm 上实测 0% 摔倒、100% 超时, 策略的理性回答
    # 是**不试**。把「蹭」从「致命」降级为「贵」, 是让尝试重新有正期望的最小改动。
    #
    # 默认 ``False`` ⇒ 整段与历史行为逐位相同。
    trunk_contact_adaptive: bool = False
    trunk_contact_hi_step_h: float = 0.105
    # Which ground surface the "trunk has fallen" height test measures against.
    #
    #   "base"    -> ``stair.height_at(base_x)``, i.e. the step the *base* is
    #                horizontally over.  Historical behaviour.
    #   "support" -> the lowest terrain under the four feet, i.e. the surface
    #                actually carrying the robot.
    #   "support_mean" -> the **mean** terrain height under the four feet.
    #                Same value as "support" whenever all four feet share a
    #                tread, but it ramps in four ``h/4`` steps while the feet
    #                cross a riser one by one instead of jumping by the full
    #                riser height the instant the last foot lands.  See the
    #                docstring of ``_support_height_mean``.
    #   "support_load" -> normal-force weighted terrain height under the feet,
    #                i.e. the tread actually carrying the robot (second step,
    #                README §11.24 改动①).  Equal to "support" when all four
    #                feet share a tread, continuous during a transition, and it
    #                also removes a **missed** fall that "support" has: with the
    #                front feet already up and the hind feet collapsing off the
    #                lower tread, ``min`` reads the lower tread and overstates
    #                the trunk clearance by a full riser height.
    #
    # ``height_at`` is a **step function**: the instant ``base_x`` crosses a
    # riser the reference jumps up by the full riser height, while the trunk is
    # still physically at the previous step's level.  ``height`` therefore drops
    # by ``step_height`` inside a single control step (0.02 s), so on tall stairs
    # the test fires on a robot that is standing perfectly well -- and the
    # ceiling it imposes is ``step_height < (trunk clearance while moving) -
    # terminate_min_height``.  Measured (.tmp/step2_probe.py, course21 final,
    # 20 deterministic episodes): see the ``terminate_height_ref`` ablation in
    # README step 19.  ``"base"`` stays the default so that every existing
    # checkpoint and regression stays reproducible.
    terminate_height_ref: str = "base"
    terminate_min_height: float = 0.13
    terminate_max_tilt: float = 1.0  # rad, roll or pitch
    reset_z: float = 0.32
    reset_joint_noise: float = 0.06
    reset_xy_noise: float = 0.12
    reset_yaw_noise: float = 0.10
    # Success criterion used by the curriculum (step 6).
    #   stairs -> base_x passes the last riser AND the base sits at crest height
    #   flat   -> base_x advances this many metres from the spawn point
    flat_success_distance: float = 3.0
    # How far in front of the first riser the robot spawns.  The old hard-coded
    # 0.55 m gave the policy only ~0.6 s (3 gait cycles) to recover from the
    # noisy reset before the first step, which is not enough for a shaky trot.
    spawn_back: float = 0.55
    # End the episode the moment the goal is reached (see ``_success_reached``).
    # Measured reason this exists: the stairs trot climbs all 8 x 5 cm steps and
    # reaches the 0.40 m crest, then runs off the far edge of the 1.2 m top
    # landing at x~=6.7 and falls at x=7.26.  With the old
    # ``is_success = reached_top and not terminated`` that fall erased the
    # success, so a policy that had genuinely climbed the staircase scored 0%.
    terminate_on_success: bool = True
    # Domain randomization (step 8). Disabled until explicitly turned on.
    # The ranges are fields (not literals inside ``_apply_domain_randomization``)
    # so the documentation can state them, ``--cfg`` can calibrate them, and the
    # regression can pin them.
    #
    # The defaults below are **measured**, not guessed: `.tmp/dr_calibrate.py`
    # swept each knob against a frozen ``course22`` policy (90% on 10.0 cm) and
    # recorded what each severity costs.  The original literals were
    # friction (0.5, 1.5) / mass (0.9, 1.15) / latency 0..3 steps / sigma 0.02,
    # and two of those exceeded what this controller can absorb at all:
    #
    #   knob                     measured success (frozen policy, 10.0 cm)
    #   friction 0.8~1.2 / 0.7~1.3 / 0.6~1.4 / 0.5~1.5   80 / 75 / 65 / 60 %
    #   mass     0.97~1.03 / 0.95~1.05 / 0.92~1.08 / 0.9~1.15  100/95/100/65 %
    #   latency  0 / 1 / 2 / 3 control steps              90 / 55 / 35 / 30 %
    #   imu sigma 0.0 / 0.01 / 0.02 / 0.04                85 / 90 / 85 / 85 %
    #
    # Latency is the binding constraint -- one 20 ms control step already costs
    # 35 points, because the gait period is only 183 ms (5.48 Hz) and the PD loop
    # was tuned for zero delay.  Mass up to +-8% and IMU noise are essentially
    # free.  See also the ``randomize_scale`` ramp, which is what makes even the
    # calibrated band learnable.
    randomize_friction: bool = False
    randomize_mass: bool = False
    randomize_latency: bool = False
    randomize_imu: bool = False
    #: Per-geom sliding-friction scale range (uniform, drawn independently).
    randomize_friction_range: tuple[float, float] = (0.7, 1.3)
    #: Uniform scale range applied to every body's mass **and** inertia.
    randomize_mass_range: tuple[float, float] = (0.92, 1.08)
    #: Actuator delay is drawn from ``0 .. randomize_latency_max`` control steps
    #: (50 Hz, so one step is 20 ms and the maximum is 20 ms).
    randomize_latency_max: int = 1
    #: Standard deviation of the per-episode IMU attitude bias, radians.
    randomize_imu_sigma: float = 0.02
    #: Severity dial for the four ranges above: 0.0 = nominal model (no
    #: randomization at all), 1.0 = the configured range in full.  Every range is
    #: interpolated towards its nominal point, so ``scale`` never changes *what*
    #: is randomized, only how far.  The curriculum ramps it 0 -> 1 across a
    #: stage, because the un-ramped full range is not learnable with this
    #: controller (see ``set_domain_randomization``).
    randomize_scale: float = 1.0
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    # Geometry/thresholds of the tactile + rangefinder suite (step 11).
    touch: TouchConfig = field(default_factory=TouchConfig)

    @property
    def control_dt(self) -> float:
        return self.timestep * self.decimation


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


class Go2StairsEnv(gym.Env):
    """A Go2 walking on a parametric box staircase."""

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
        # 上一步的「四足足下最低地形」——``foothold`` 奖励的差分基准。
        self._prev_support_height = 0.0
        self._reached_top = False
        # Consecutive trunk / leg terrain-contact streaks (see
        # ``EnvConfig.terminate_contact_steps``).
        self._trunk_contact_streak = 0
        self._leg_contact_streak = 0
        # Pose the episode started from.  "Walk straight" is measured against
        # these, not against the staircase centreline, because reset itself adds
        # lateral/yaw jitter that the robot must not be blamed for.
        self._y0 = 0.0
        self._yaw0 = 0.0
        self._success_x = self._success_target_x()
        self.nominal_pose = float(self.meta.home_qpos[2])  # 0.27 m stance height
        self._prev_foot_pos = self.foot_pos.copy()
        self._foot_air_time = np.zeros(len(self.meta.foot_geom_ids))
        self._prev_foot_contact = np.zeros(len(self.meta.foot_geom_ids), dtype=bool)
        # 摆动相内的**峰值**「相对支撑面的离地间隙」，落地那一步用于
        # ``foot_lift_clear`` 结算，也是 ``obs.foot_lift_gap`` 的来源之一。
        self._peak_swing_clearance = np.zeros(len(self.meta.foot_geom_ids))
        self._peak_tread = np.zeros(len(self.meta.foot_geom_ids))
        # Gait-prior clock (seconds since episode start).
        self._gait_t = 0.0
        self._gait_phase = 0.0
        self._pending_ctrl: deque[np.ndarray] = deque()
        self._rng = np.random.default_rng(seed)
        self._renderer: mujoco.Renderer | None = None
        self._render_cam = None
        self._scene_option = None
        self._render_size = None
        # Per-episode domain-randomization state
        self._imu_bias = np.zeros(3)
        self._latency_steps = 0

    # ------------------------------------------------------------------ build

    def _build_model(self) -> None:
        opts = SceneOptions(timestep=self.cfg.timestep)
        self.model, self.meta, self.xml = compile_scene(
            build_terrain_xml(self.stair),
            opts=opts,
            floor_geom=default_floor_geom(),
        )
        self.data = mujoco.MjData(self.model)
        # Tactile + rangefinder reader (step 11).  Rebuilt here because it holds
        # references to both the model and the data.  The ray fan is the costly
        # half, so it is only enabled when something actually reads it: the
        # ``stair_probe`` observation block, or the ``stair_probe_lift`` reward.
        self.touch_sensor = TouchSensor(
            self.model, self.data, self.meta, self.cfg.touch,
            probe_enabled=bool(self.cfg.obs.stair_probe
                               or self.cfg.reward.stair_probe_lift != 0.0),
        )
        self._touch: TouchReading = empty_reading()
        # Nominal values, so per-episode randomization always scales from the
        # pristine model instead of compounding across resets.
        self._nominal_body_mass = self.model.body_mass.copy()
        self._nominal_body_inertia = self.model.body_inertia.copy()
        self._nominal_geom_friction = self.model.geom_friction.copy()
        # ``mujoco.Renderer`` binds to one ``MjModel`` instance, so it must never
        # outlive the model it was built from -- otherwise ``set_stair()`` would
        # keep drawing the *previous* terrain.
        if getattr(self, "_renderer", None) is not None:
            self._renderer.close()
            self._renderer = None
            self._render_size = None
            self._scene_option = None

    def set_stair(self, stair: StairSpec) -> None:
        """Swap the terrain (used by the curriculum). Recompiles the model.

        The caller must call ``reset()`` afterwards: recompiling replaces the
        underlying ``MjModel``/``MjData``, so the old state is meaningless.
        """
        self.stair = stair
        self._build_model()
        self._success_x = self._success_target_x()
        self._reached_top = False

    def set_gait_prior(self, enable: bool, gait: GaitParams | None = None) -> None:
        """Turn the residual-gait prior on/off at runtime (used by the CLI).

        Changing the prior mid-run is safe because it only affects how the PD
        target is built; it does not change the observation size, so it can be
        toggled without resetting ``observation_space``.
        """
        self.cfg.gait_prior = bool(enable)
        if gait is not None:
            self.cfg.gait = gait
        self._gait_t = 0.0
        self._gait_phase = 0.0

    def set_domain_randomization(self, scale: float | None = None,
                                **flags: bool) -> None:
        """Step 8: toggle friction/mass/latency/IMU randomization at runtime.

        The curriculum turns this on once the robot can already walk, so the
        flat stage is learned on a deterministic model.

        ``scale`` is the **severity** dial: 0.0 reproduces the nominal model,
        1.0 uses the configured range in full, and values in between interpolate
        every range towards its nominal point.  The curriculum ramps it 0 -> 1
        because the full range applied from the first step of a stage is not
        learnable here: measured on a frozen policy, latency alone removes 60
        points of success at 10.0 cm (``.tmp/dr_calibrate.py``).
        """
        valid = {
            "randomize_friction",
            "randomize_mass",
            "randomize_latency",
            "randomize_imu",
        }
        unknown = set(flags) - valid
        if unknown:
            raise AttributeError(
                f"unknown domain-randomization flag(s) {sorted(unknown)}; "
                f"valid: {sorted(valid)}"
            )
        for key, value in flags.items():
            setattr(self.cfg, key, bool(value))
        if scale is not None:
            self.cfg.randomize_scale = float(np.clip(scale, 0.0, 1.0))

    # ---------------------------------------------------------------- success

    def _success_target_x(self) -> float:
        """x the base must pass for this episode to count as a success.

        Stairs: the far edge of the last riser (crest height must also be
        reached, see ``_success_reached``). Flat: a fixed forward distance,
        because a flat stage has no riser to clear.
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
        return bool(self.base_pos[0] > self._success_x)

    # ------------------------------------------------------------- properties

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
        """Linear velocity in the body frame, i.e. ``R(q)^T @ qvel[0:3]``.

        Unlike :attr:`base_lin_vel_world` this does not change meaning as the
        robot yaws, so "forward" is always ``[0]``.
        """
        rot = quat_to_mat(self.base_quat)
        return rot.T @ self.data.qvel[0:3]

    @property
    def base_ang_vel_body(self) -> np.ndarray:
        """Gyro reading. qvel[3:6] is already in the body frame."""
        return self.data.qvel[3:6]

    @property
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.meta.qpos_joint_adr]

    @property
    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.meta.qvel_joint_adr]

    @property
    def foot_pos(self) -> np.ndarray:
        """(4, 3) world positions of the foot sphere centres."""
        return self.data.geom_xpos[self.meta.foot_geom_ids]

    @property
    def terrain_height_under_base(self) -> float:
        return self.stair.height_at(float(self.base_pos[0]))

    @property
    def yaw(self) -> float:
        return float(self.base_rpy[2])

    @property
    def lateral_offset(self) -> float:
        """Sideways displacement from the **stair centre line** (m).

        Measured against ``stair.center_y``, not against the episode's spawn y.
        The old spawn-relative version made the reward "keep whatever lateral
        jitter the reset gave you" -- the reset samples y uniformly in
        ``+/-reset_xy_noise``, so a policy that held its spawn line perfectly
        still ended up off-centre.  The centre line is the only reference that
        means "walking straight up this staircase", and the reset jitter is at
        most ``reset_xy_noise = 0.12 m`` inside a 2.0 m wide staircase, so
        there is nothing unfair about asking the robot to converge on it.
        """
        return float(self.base_pos[1] - self.stair.center_y)

    @property
    def spawn_lateral_offset(self) -> float:
        """Sideways displacement from the line the episode started on (m).

        Kept for diagnostics: comparing this against :attr:`lateral_offset`
        separates "the reset threw me off" from "I drifted during the episode".
        """
        return float(self.base_pos[1] - self._y0)

    @property
    def heading_error(self) -> float:
        """Yaw relative to the **stair axis** (world +x), wrapped to (-pi, pi]."""
        d = float(self.base_rpy[2])
        return float((d + np.pi) % (2.0 * np.pi) - np.pi)

    @property
    def spawn_heading_error(self) -> float:
        """Yaw relative to the episode's spawn heading, wrapped to (-pi, pi]."""
        d = float(self.base_rpy[2]) - self._yaw0
        return float((d + np.pi) % (2.0 * np.pi) - np.pi)

    # ---------------------------------------------------------------- helper

    @property
    def foot_clearance(self) -> np.ndarray:
        """Gap between each foot *surface* and the terrain below it (m).

        ``geom_xpos`` is the centre of the foot sphere, so the radius must be
        subtracted before comparing against the terrain height.
        """
        feet = self.foot_pos
        return feet[:, 2] - self.meta.foot_radius - self._terrain_height(feet[:, 0])

    def _terrain_height(self, x: np.ndarray | float) -> np.ndarray:
        """Terrain height under one or more world x coordinates."""
        arr = np.atleast_1d(np.asarray(x, dtype=float))
        if self.stair.n_up == 0 and self.stair.n_down == 0:
            return np.zeros_like(arr)
        return np.array([self.stair.height_at(float(xi)) for xi in arr])

    def _projected_gravity(self) -> np.ndarray:
        """Gravity direction in the body frame (classic locomotion feature)."""
        w, x, y, z = self.base_quat
        # R^T @ (0,0,-1)
        return np.array(
            [
                2 * (x * z - w * y),
                2 * (y * z + w * x),
                1 - 2 * (x * x + y * y),
            ]
        )

    def _obs_frame(self) -> np.ndarray:
        obs_cfg = self.cfg.obs
        ang_vel = self.base_ang_vel_body
        if self.cfg.randomize_imu:
            ang_vel = ang_vel + self._imu_bias  # 八、IMU 噪声
        parts = [
            self.joint_pos,
            self.joint_vel,
            self.base_rpy,
            ang_vel,
            self._last_action,
        ]
        if obs_cfg.base_lin_vel:
            parts.append(self.base_lin_vel_world)
        if obs_cfg.base_lin_vel_body:
            parts.append(self.base_lin_vel_body)
        if obs_cfg.base_height:
            parts.append(np.array([self.base_pos[2] - self._height_reference()]))
        if obs_cfg.foot_height:
            parts.append(self.foot_clearance)
        if obs_cfg.gait_phase:
            theta = 2.0 * np.pi * self.cfg.gait.freq * self._gait_phase
            parts.append(np.array([np.sin(theta), np.cos(theta)]))
        # --- step 11: touch / rangefinder / drift ---------------------------
        if obs_cfg.touch:
            parts.append(self._touch.foot_obs(self.cfg.touch))
        if obs_cfg.stair_probe:
            parts.append(self._touch.probe_obs_full(self.cfg.touch)
                         if obs_cfg.stair_probe_full
                         else self._touch.probe_obs(self.cfg.touch))
        if obs_cfg.drift:
            parts.append(np.array([self.lateral_offset, self.heading_error]))
        if obs_cfg.foot_lift_gap:
            parts.append(self.foot_lift_gap())
        if obs_cfg.foot_lift_deficit:
            parts.append(self.foot_lift_deficit())
        return np.concatenate(parts).astype(np.float64)

    def _observation(self) -> np.ndarray:
        return np.concatenate(list(self._hist)).astype(np.float32)

    # ----------------------------------------------------------------- reward

    def _compute_reward(self, action: np.ndarray, trunk_contact: bool,
                        foot_contact: np.ndarray | None = None,
                        leg_contact: bool = False
                        ) -> tuple[float, dict[str, float]]:
        rc = self.cfg.reward
        rpy = self.base_rpy
        lin = self.base_lin_vel_world
        target_height = self.nominal_pose

        terms: dict[str, float] = {}

        # 1) 前进速度奖励 -- forward velocity tracking -----------------------
        err = lin[0] - self.cfg.command_vx
        terms["forward_vel"] = (
            rc.forward_vel * float(np.exp(-(err * err) / rc.forward_vel_sigma))
        )
        # Linear companion term: exactly 0 when the robot is not moving, so the
        # "stand perfectly still" policy cannot collect it at all.
        terms["forward_progress"] = rc.forward_progress * float(
            np.clip(lin[0], rc.forward_progress_min, rc.forward_progress_max)
        )

        # 2) 姿态稳定奖励 -- penalise roll and pitch ------------------------
        terms["orientation"] = rc.orientation * float(rpy[0] ** 2 + rpy[1] ** 2)

        # 3) 关节力矩惩罚 -- joint torque -----------------------------------
        tau = self.data.actuator_force
        terms["torque"] = rc.torque * float(np.sum(np.square(tau)))

        # 4) 足端抬升奖励 -- swing feet must clear the terrain --------------
        feet = self.foot_pos
        foot_dt = self.cfg.control_dt
        # **三维**足端速度。只看水平分量会奖励「贴着地面快速平扫」而掐掉
        # 「慢慢抬高跨越」—— 高台阶需要的恰恰是后者 (见
        # RewardConfig.foot_lift_gate_speed 的注释)。
        foot_speed = np.linalg.norm(
            (feet - self._prev_foot_pos) / foot_dt, axis=1
        )
        gate = np.clip(foot_speed / max(float(rc.foot_lift_gate_speed), 1e-6),
                       0.0, 1.0)
        # geom_xpos is the SPHERE CENTRE, so subtract the radius to get the
        # real gap between the foot surface and the terrain.
        #
        # ``foot_lift_support_ref`` 时参照面换成 ``advance_clearance()``（见
        # RewardConfig 注释：原参照面是阶跃函数，脚跨过立面时会凭空少算一个台阶高，
        # 于是 11 cm 上的绝对读数反而比 8 cm 大）。
        clearance = (
            self.advance_clearance() if bool(rc.foot_lift_support_ref)
            else self.foot_clearance
        )
        # 目标随地形长大: 要踏上 h 米高的台阶, 足端至少要高出脚下那一级 h 米。
        # 5 cm 台阶下 max(0.08, 1.4*0.05)=0.08 与旧行为**逐位相同**, 所以已经
        # 学会的 easy 阶段不会因为这次改动而退化; 8/12 cm 才会真正收到更高的目标。
        lift_target = self.required_lift()
        lift = np.clip(clearance / lift_target, 0.0, 1.0)
        terms["foot_lift"] = rc.foot_lift * float(np.mean(gate * lift))
        # 摆动峰值结算项（默认关闭）。峰值用 ``_peak_swing_clearance``：它是本摆动
        # 到**这一步为止**的最大离地间隙（见 ``_advance_swing_clearance``），
        # ``landed_lc`` 为真的这一步就是落地那一步，结算之后
        # ``_advance_foot_air_time`` 会把峰值清零，所以这一项每个摆动周期恰好结算
        # 一次。用整段摆动的峰值而不是落地瞬间的瞬时值，是因为落地那一步足端已经
        # 贴地、瞬时值恒为零。注意峰值与判定用的是同一个参照面口径（都在
        # :meth:`_advance_swing_clearance` 里算），落地这一步不需要再补测。
        if rc.foot_lift_clear != 0.0:
            if foot_contact is None:
                foot_contact = self._foot_contact_mask()
            landed_lc = foot_contact & ~self._prev_foot_contact
            if landed_lc.any():
                ratio = self._peak_swing_clearance[landed_lc] / self.lift_goal()
                credit = (
                    (ratio >= 1.0).astype(np.float64) if bool(rc.foot_lift_clear_binary)
                    else np.clip(ratio, 0.0, 1.0)
                )
                terms["foot_lift_clear"] = rc.foot_lift_clear * float(np.sum(credit))
            else:
                terms["foot_lift_clear"] = 0.0
            # the swing is settled -> a foot that just landed starts fresh
            np.copyto(
                self._peak_swing_clearance,
                np.where(foot_contact, 0.0, self._peak_swing_clearance),
            )
        else:
            terms["foot_lift_clear"] = 0.0

        # 5) 机身碰撞惩罚 -- torso touching the terrain ---------------------
        # Per-step, bounded: the penalty scales with how long the torso is
        # scraping rather than with a single boolean at the terminal step.
        #
        # 第二十四步: 台阶高 >= trunk_contact_hi_step_h 时按
        # ``body_collision_hi_scale`` 缩放 (默认 1.0 ⇒ 与历史行为逐位相同)。
        # 见 RewardConfig.body_collision_hi_scale。
        collision_scale = 1.0
        if bool(self.cfg.trunk_contact_adaptive) and float(self.stair.step_height) >= float(
            self.cfg.trunk_contact_hi_step_h
        ):
            collision_scale = float(rc.body_collision_hi_scale)
        terms["body_collision"] = rc.body_collision * collision_scale * float(trunk_contact)
        # Shin / knee scraping a riser.  Bounded per step, and crucially it does
        # NOT end the episode, so the robot can keep trying on tall stairs.
        terms["leg_collision"] = rc.leg_collision * float(leg_contact)

        # 6) 足端腾空时间 -- gait shaping (landing after a good swing) ------
        # ``_foot_air_time`` holds the flight time accumulated BEFORE the current
        # step, and ``_prev_foot_contact`` the contact state before it, so this
        # pays out exactly on the step where a foot touches down.
        if rc.air_time != 0.0:
            if foot_contact is None:
                foot_contact = self._foot_contact_mask()
            landed = foot_contact & ~self._prev_foot_contact
            air = self._foot_air_time
            credit = np.clip(air - rc.air_time_target, 0.0, None)
            overrun = np.clip(air - rc.air_time_max, 0.0, None)
            terms["air_time"] = rc.air_time * float(
                np.sum((credit - overrun) * landed)
            )
        else:
            terms["air_time"] = 0.0

        # --- supporting shaping -------------------------------------------
        terms["lateral_vel"] = rc.lateral_vel * float(lin[1] ** 2)
        yaw_err = float(self.base_ang_vel_body[2]) - self.cfg.command_yaw_rate
        terms["yaw_rate"] = rc.yaw_rate * float(yaw_err ** 2)
        terms["base_height"] = rc.base_height * float(
            # 这里**故意**继续用单点参照面 (`terrain_height_under_base`), 而不是
            # `_height_reference()`。原因: `obs.base_height_support` 要当**观测**
            # 的单变量开关 —— 实测把它同时接到这一项上 (即 course34 的第一版) 会让
            # 均回报从 +554 崩到 -674/回合, 观测与奖励两个变量就分不开了。
            # 奖励侧要用足端参照面时, 用 `stair_progress_foot_ref` (那一项才是
            # signal_audit 指出的问题)。见 README §11.22。
            (self.base_pos[2] - self.terrain_height_under_base - target_height) ** 2
        )
        terms["joint_vel"] = rc.joint_vel * float(np.sum(np.square(self.joint_vel)))
        terms["action_rate"] = rc.action_rate * float(
            np.sum(np.square(action - self._prev_action))
        )
        lo, hi = self.act_low, self.act_high
        q = self.joint_pos
        over = np.clip(lo - q, 0, None) + np.clip(q - hi, 0, None)
        terms["joint_limit"] = rc.joint_limit * float(np.sum(over))
        terms["alive"] = rc.alive

        # --- 爬升进展（口径可选，默认历史行为） ---------------------------
        #
        # ``stair_progress`` 的参照面可以是「机身 x 处的单点高度」（历史）或
        # 「四足足下最低地形」（``stair_progress_foot_ref``）。后者不会被躯干冲线
        # 骗到，见 RewardConfig.stair_progress 的注释与 README §11.22。
        if bool(rc.stair_progress_foot_ref):
            progress_ref = self._support_height()
            if int(rc.stair_progress_min_loaded_feet) > 0:
                enough = self._loaded_foot_count(rc.stair_progress_force_eps) \
                    >= int(rc.stair_progress_min_loaded_feet)
                if not enough:
                    progress_ref = self._start_terrain_height
        else:
            progress_ref = self.terrain_height_under_base
        terms["stair_progress"] = rc.stair_progress * float(
            max(0.0, progress_ref - self._start_terrain_height)
        )

        # ``foothold``：四足足下最低地形相对**上一步**的真实增量（只在跨级那一步付钱）。
        foot_h = self._support_height()
        if rc.foothold != 0.0:
            step_h = max(float(self.stair.step_height), 1e-6)
            gain = (foot_h - self._prev_support_height) / step_h
            terms["foothold"] = rc.foothold * float(np.clip(gain, 0.0, 1.0))
        else:
            terms["foothold"] = 0.0
        self._prev_support_height = foot_h

        # --- step 11: 触觉 / 碰撞 ------------------------------------------
        touch = self._touch
        # ``riser_graze_probe_gate``：只在测距真的命中前方台阶时才认定「蹭立面」。
        # 平地段的假阳性（髋/小腿与地面边角接触也会让接触法向的世界 z 分量偏小）
        # 因此被清掉；真台阶上 ``trigger`` 在台阶前 0.62 m 就为真，惩罚不变。
        probe_gate_ok = True
        if bool(rc.riser_graze_probe_gate):
            probe_gate_ok = bool(touch.trigger(self.cfg.touch))
        # ``riser_graze_mode = "lifting"``：只惩罚**真卡住**（所有蹭立面的脚都没有
        # 向上速度）的步；正在抬脚跨越立面的脚免罚。见 RewardConfig.riser_graze_mode。
        if str(rc.riser_graze_mode).strip().lower() == "lifting":
            grazing = np.asarray(touch.riser_graze, dtype=bool)
            if grazing.any():
                feet_now = self.foot_pos
                vz = (feet_now[:, 2] - self._prev_foot_pos[:, 2]) / max(
                    float(self.cfg.control_dt), 1e-9)
                climbing = bool(np.any(vz[grazing] > float(rc.riser_graze_vz_eps)))
                stuck = not climbing
            else:
                stuck = False
            terms["riser_graze"] = rc.riser_graze * float(stuck and probe_gate_ok)
        else:
            terms["riser_graze"] = rc.riser_graze * float(
                touch.riser_graze.any() and probe_gate_ok
            )

        # Both tactile penalties below are *squared* functions of a physics
        # ratio, and MuJoCo contact forces spike without bound when a foot is
        # jammed against a riser edge or lands hard.  Without a cap a single
        # pathological step reached O(10^5) and blew up the value function for a
        # whole stage (observed: mean episode return -586285 on `easy2` of
        # run course11).  Clipping the *normalised excess* keeps the penalty a
        # strong shaping signal while bounding the worst case per step per foot
        # to ``touch_excess_clip ** 2``.
        excess_clip = max(float(rc.touch_excess_clip), 0.0)

        if foot_contact is None:
            foot_contact = self._foot_contact_mask()
        landed = foot_contact & ~self._prev_foot_contact
        if landed.any() and rc.touch_impact != 0.0:
            # Normalised excess over the target so the penalty is O(1) per
            # landing rather than ~10^4, which is what a raw newtons-squared
            # term would be next to the other shaping terms.
            #
            # ``touch_impact_step_scale``: 目标力按台阶高度放大 (5 cm 上恒为
            # 原值, 默认 0 时整段与历史行为逐位相同)。见 RewardConfig。
            ref = max(rc.touch_impact_target, 1e-6)
            if float(rc.touch_impact_step_scale) != 0.0:
                h_ratio = float(self.stair.step_height) / 0.05 - 1.0
                ref = max(ref * (1.0 + float(rc.touch_impact_step_scale) * h_ratio),
                          1e-6)
            over = np.clip((touch.foot_normal_force[landed] - ref) / ref, 0.0, excess_clip)
            terms["touch_impact"] = rc.touch_impact * float(np.sum(over ** 2))
        else:
            terms["touch_impact"] = 0.0

        # Slip is only meaningful on a foot that is carrying weight; a swinging
        # foot has ~0 normal force, so its ratio is pure noise.
        if rc.touch_slip != 0.0:
            loaded = touch.foot_normal_force > self.cfg.touch.slip_loaded_force
            excess = np.clip(touch.foot_slip - rc.touch_slip_safe, 0.0, excess_clip)
            terms["touch_slip"] = rc.touch_slip * float(np.sum(excess ** 2 * loaded))
        else:
            terms["touch_slip"] = 0.0

        # 检测到楼梯就抬腿: pay only when the rangefinder has a step in front of
        # it AND the two front feet are actually lifted for it.  ``lift`` above
        # is the per-foot normalised clearance, ordered FL, FR, RL, RR.
        if rc.stair_probe_lift != 0.0:
            probe_lift = (
                np.clip(self.advance_clearance() / lift_target, 0.0, 1.0)
                if bool(rc.stair_probe_lift_use_advance) else lift
            )
            terms["stair_probe_lift"] = rc.stair_probe_lift * float(
                touch.trigger(self.cfg.touch) * np.mean(probe_lift[:2])
            )
        else:
            terms["stair_probe_lift"] = 0.0

        # --- step 11: 走直线 -----------------------------------------------
        # Both are quadratics in an error that is *not* inherently bounded (a
        # falling robot can yaw through most of a turn while sliding sideways),
        # so clip the error before squaring.  Squaring without a cap is what let
        # a single tactile step reach -5.5e7; every penalty here is bounded.
        y_lim = max(float(rc.straight_line_max), 0.0)
        h_lim = max(float(rc.heading_max), 0.0)
        y_err = float(np.clip(self.lateral_offset, -y_lim, y_lim))
        h_err = float(np.clip(self.heading_error, -h_lim, h_lim))
        terms["straight_line"] = rc.straight_line * y_err * y_err
        terms["heading"] = rc.heading * h_err * h_err
        err_vy = lin[1] - self.cfg.command_vy
        terms["lateral_vel_track"] = rc.lateral_vel_track * float(
            np.exp(-(err_vy * err_vy) / (rc.lateral_vel_sigma ** 2))
        )
        return float(sum(terms.values())), terms

    def _foot_contact_mask(self) -> np.ndarray:
        """Per-foot boolean: is this foot geom touching the terrain right now?"""
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
        """Accumulate flight time; called once per control step, after reward."""
        # 先记下本步的峰值（含落地这一步的最高点），再清零落地脚的摆动状态。
        self._advance_swing_clearance(foot_contact)
        self._foot_air_time = np.where(
            foot_contact, 0.0, self._foot_air_time + self.cfg.control_dt
        )
        self._prev_foot_contact = foot_contact

    def _contact_flags(self) -> tuple[bool, bool]:
        """(trunk_contact, leg_contact) for robot-geom-vs-terrain contacts.

        ``trunk_contact`` -- one of the torso collision primitives (the box /
        cylinder / sphere welded to the ``base`` body) is touching the terrain.
        That is a genuine fall, and the only contact that is fatal by default.

        ``leg_contact`` -- a hip / thigh / calf collision shape is touching the
        terrain while the torso is not.  On a staircase this is almost always a
        shin brushing a riser mid-climb: it earns ``leg_collision`` per step but
        the episode keeps running, which is what preserves trial and error.
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

    def _body_contact(self) -> bool:
        """True if a non-foot robot geom (torso *or* leg) touches the terrain."""
        trunk, leg = self._contact_flags()
        return bool(trunk or leg)

    def _effective_contact_steps(self) -> int:
        """本步实际生效的去抖阈值 (控制步)。

        ``contact_debounce_adaptive=False`` 时恒等于
        ``terminate_contact_steps`` (历史行为)。否则在
        ``stair.step_height >= contact_debounce_hi_step_h`` 上切换到
        ``contact_debounce_hi_steps``。见 ``EnvConfig`` 里那段动机说明。
        """
        steps = max(1, int(self.cfg.terminate_contact_steps))
        if bool(self.cfg.contact_debounce_adaptive):
            hi_h = float(self.cfg.contact_debounce_hi_step_h)
            if float(self.stair.step_height) >= hi_h:
                steps = max(1, int(self.cfg.contact_debounce_hi_steps))
        return steps

    def _trunk_contact_relaxed(self) -> bool:
        """高台阶上躯干接触是否**不再致命** (仍照常扣分)。

        需要 ``trunk_contact_adaptive=True``、``terminate_contact="trunk"`` 且
        ``stair.step_height >= trunk_contact_hi_step_h``。
        """
        if not bool(self.cfg.trunk_contact_adaptive):
            return False
        if str(self.cfg.terminate_contact).strip().lower() != "trunk":
            return False
        return float(self.stair.step_height) >= float(self.cfg.trunk_contact_hi_step_h)

    def _contact_fatal(self, trunk_contact: bool, leg_contact: bool) -> bool:
        """Is the contact seen *this* step fatal, under the streak threshold?

        ``terminate_contact_steps`` is the number of **consecutive** control
        steps the selected contact class must persist before it kills the
        episode.  ``1`` reproduces the historical immediate termination.

        The streak counters are advanced here rather than in ``step`` so that
        "one call to ``_terminated`` == one control step" also holds for the
        regression tests, which drive ``_terminated`` directly.
        """
        self._trunk_contact_streak = self._trunk_contact_streak + 1 if trunk_contact else 0
        self._leg_contact_streak = self._leg_contact_streak + 1 if leg_contact else 0
        mode = str(self.cfg.terminate_contact).strip().lower()
        if mode == "none" or not self.cfg.terminate_body_contact:
            return False
        if self._trunk_contact_relaxed():
            # 高台阶: 躯干蹭立面降级为「贵」而不是「致命」(见 EnvConfig)。
            if mode == "trunk":
                return False
            live = bool(leg_contact)
            streak = self._leg_contact_streak
        elif mode == "any":
            live = bool(trunk_contact or leg_contact)
            streak = max(self._trunk_contact_streak, self._leg_contact_streak)
        else:  # "trunk": a shin scrape is penalised, never fatal
            live = bool(trunk_contact)
            streak = self._trunk_contact_streak
        if not live:
            return False
        return streak >= self._effective_contact_steps()

    def _support_height(self) -> float:
        """**四足足下最低地形高度** —— 真正托住机器人的那个面。

        与 :meth:`terrain_height_under_base`（机身 x 处的单点阶跃高度）相对。
        台阶交界处两者的差正好是一个台阶高，实测样本见 README §11.22。
        """
        heights = [self.stair.height_at(float(x)) for x in self.foot_pos[:, 0]]
        return float(min(heights)) if heights else self.terrain_height_under_base

    def _foot_tread_heights(self) -> np.ndarray:
        """每只脚正下方的地形高度 (m, 4 维) —— ``foot_clearance`` 的参照面。"""
        return np.array(
            [self.stair.height_at(float(x)) for x in self.foot_pos[:, 0]],
            dtype=np.float64,
        )

    def _step_surface_ahead(self) -> np.ndarray:
        """每只脚**下次落脚时将会踩到**的那个踏面高度 (m, 4 维)。

        台阶是阶梯状的, 所以「等一下要走过去的那块地」就是**从脚当前位置向前看,
        脚和它的落脚区间会经历的最高踏面**。原 :attr:`foot_clearance` 用的是
        「正下方」, 它在立面处是**阶跃**的: 脚还站在第 k 级、脚前缘已经在第 k+1
        级上方时, 读数按哪一级算完全取决于脚本身有没有跨过 ``start_x + k*tread``
        这条线, 于是同一个抬腿动作在跨线的一瞬间凭空少算/多算一个台阶高。

        这里只用**位置**决定参照面, 与脚当前高度无关, 所以同一个动作的读数连续。
        """
        rc = self.cfg.reward
        x = self.foot_pos[:, 0]
        reach = float(self.meta.foot_radius) + float(rc.foot_placement_reach)
        pts = x[:, None] + np.array(
            [[0.0, 0.5 * reach, reach]], dtype=np.float64
        )
        heights = np.array(
            [self.stair.height_at(float(v)) for v in pts.ravel()],
            dtype=np.float64,
        )
        return heights.reshape(pts.shape).max(axis=1)

    def advance_clearance(self) -> np.ndarray:
        """每只脚相对**它将踩上去的那个踏面**的离地间隙 (m, 4 维)。

        参照面是 :meth:`_step_surface_ahead` —— 脚当前位置与落脚区间内最高的踏面。
        与原 :attr:`foot_clearance` 的区别就是参照面:

        * **脚在立面正前方**时, 原口径读到的是脚下那一级 (与任务无关), 本口径读到
          的是**要越过去的那一级** —— 这正是成败所在。
        * **脚跨过立面之后**原口径会凭空跳一个台阶高, 本口径不会。

        读数为负表示**还没抬够**: 例如站在平地、前方 11 cm 台阶、脚底抬到 8 cm 时
        读数是 ``-0.03``。这比原来那个「11 cm 上比 8 cm 还大」的读数直接得多。
        """
        return self.foot_clearance + (
            self._foot_tread_heights() - self._step_surface_ahead()
        )

    def required_lift(self) -> float:
        """**连续项** ``foot_lift`` 的满分线 (m)。

        ``max(foot_lift_target, foot_lift_step_ratio * step_height)``，再按
        ``foot_lift_target_cap``（>0 时）截断。用的是**逐步瞬时**离地间隙，所以
        与 ``_compute_reward`` 里 ``lift = clip(clearance / required_lift())``
        的口径严格一致。
        """
        rc = self.cfg.reward
        target = max(
            float(rc.foot_lift_target),
            float(rc.foot_lift_step_ratio) * float(self.stair.step_height),
        )
        cap = float(rc.foot_lift_target_cap)
        if cap > 0.0:
            target = min(target, cap)
        return float(max(target, 1e-6))

    def lift_goal(self) -> float:
        """**摆动峰值门槛**（``foot_lift_clear`` / ``obs.foot_lift_gap``）的目标 (m)。

        ``max(foot_lift_target, foot_lift_goal_ratio * step_height)``，同样受
        ``foot_lift_target_cap`` 约束。物理含义是「这只脚在这次摆动里至少要抬到
        比前方那一级高出多少」。
        """
        rc = self.cfg.reward
        target = max(
            float(rc.foot_lift_target),
            float(rc.foot_lift_goal_ratio) * float(self.stair.step_height),
        )
        cap = float(rc.foot_lift_target_cap)
        if cap > 0.0:
            target = min(target, cap)
        return float(max(target, 1e-6))

    def foot_lift_gap(self) -> np.ndarray:
        """每只脚**上一轮摆动**距离「越过前方台阶」还差多少，归一化后 clip 到 [0, 1]。

        用**摆动峰值**（``_peak_swing_clearance``，落地清零）而不是瞬时值：
        瞬时值在整个支撑相恒为 0，对策略毫无信息量；峰值则在这只脚刚刚抬过的高度
        上给出「够/不够」的明确读数，并且天然携带「这只脚这一轮尝试的结果」。

        0 = 这只脚这一轮已经抬到足够高，1 = 一点都没抬起来（峰值 ≈ 0）。
        """
        goal = self.lift_goal()
        peak = getattr(self, "_peak_swing_clearance", None)
        if peak is None or peak.shape != (len(self.meta.foot_geom_ids),):
            peak = np.zeros(len(self.meta.foot_geom_ids), dtype=np.float64)
        return np.clip((goal - peak) / goal, 0.0, 1.0)

    def foot_lift_deficit(self) -> np.ndarray:
        """每只脚**此刻**还要再抬高多少才能越过它将踩上去的那一级，归一化到 [0, 1]。

        用 ``advance_clearance()``（落脚面口径）除以 ``lift_goal()`` —— 与
        :meth:`foot_lift_gap` 用同一个门槛，于是「达到 goal」在观测里明确对应
        deficit 归零，而不会逼着策略去追 ``required_lift()`` 那个更高、实际处在腿长
        几何极限上的连续项满分线。已经足够高时为 0，完全没有抬起时为 1。

        这是「现在就抬腿」这个动作唯一能直接看到的反馈量 —— 660 维的原始观测里
        ``foot_height`` 在立面两侧读数都接近 0，分辨不出「站在台阶前面、脚还贴地」
        和「已经站在上一级踏面上」。
        """
        goal = self.lift_goal()
        return np.clip(1.0 - self.advance_clearance() / goal, 0.0, 1.0)

    def _advance_swing_clearance(self, foot_contact: np.ndarray) -> None:
        """Track how much the foot cleared *the surface it actually met*.

        Two running peaks are kept for every foot, both reset on landing:

        * ``_peak_tread`` -- the highest terrain directly under the foot since
          this swing began.  It answers "which surface did this swing have to
          clear?".  It is a **local** measurement, so it is immune to the
          step-function error of :attr:`foot_clearance` -- that one reads the
          tread under the foot's *kernel* and therefore drops by a whole step
          the instant the foot crosses a riser, which made the old reading
          *larger* on an 8 cm stair than on an 11 cm one.
        * ``_peak_swing_clearance`` -- the largest value of
          ``(peak foot height) - (peak tread)`` reached so far this swing,
          i.e. the best clearance actually achieved against that surface.

        Using the *peak* rather than the instantaneous value matters: on the
        landing step the foot is already on the ground, so the instantaneous
        clearance is exactly zero and a reward built on it would never pay.

        ⚠ 时序：``_peak_swing_clearance`` 在**落地这一步也不再清零**，要等
        :meth:`_compute_reward` 把 ``foot_lift_clear`` 结算完，才由 :meth:`reset`
        或下一次摆动重新开始。否则落地那一步奖励读到的峰值已经归零，这一项永远
        等于 0（这是实现里踩过的一个真坑）。
        """
        peak = getattr(self, "_peak_swing_clearance", None)
        tread = getattr(self, "_peak_tread", None)
        shape = foot_contact.shape
        if peak is None or peak.shape != shape:
            peak = np.zeros(shape, dtype=np.float64)
            self._peak_swing_clearance = peak
        if tread is None or tread.shape != shape:
            tread = np.zeros(shape, dtype=np.float64)
            self._peak_tread = tread
        under = self._foot_tread_heights()
        radius = float(self.meta.foot_radius)
        foot_top = self.foot_pos[:, 2] - radius
        # fold in this step's measurement against the surface met so far
        np.copyto(peak, np.maximum(peak, foot_top - tread))
        np.copyto(tread, np.maximum(tread, under))
        # a foot that just landed starts a fresh swing from this tread
        np.copyto(tread, np.where(foot_contact, under, tread))

    def _support_height_mean(self) -> float:
        """**四足足下地形高度的算术均值** —— 把 ``min`` 的那一次整跳拆成四步。

        ``min`` 带一个**整台阶高的量化**: 四只脚一只一只跨上第 k+1 级时,
        ``min`` 在**最后一只脚**落地的那一帧从 ``h_k`` 整跳到 ``h_k +
        step_height``。判摔是 ``base_z - ref < terminate_min_height``, 所以这一
        帧的读数会**瞬间掉一个台阶高**, 11 cm 上就是 0.11 m —— 而阈值本身才
        0.13 m。均值把这一跳摊成 ``h/4`` 的四小步。

        ⚠ 必须说清楚它**不是**单纯变宽松: 跨级过程中 ``mean`` 比 ``min`` **高**
        (更悲观), 所以它在过渡期**更早**判摔; 它去掉的是**那一帧的突变**, 不是
        把判据放松。物理上哪一端更对取决于「躯干下面真正是哪个面」, 这正是
        :meth:`_support_height_load` 要回答的问题。
        """
        heights = [self.stair.height_at(float(x)) for x in self.foot_pos[:, 0]]
        return float(np.mean(heights)) if heights else self.terrain_height_under_base

    def _support_height_load(self) -> float:
        """**按法向力加权的足下地形高度** —— 「真正托住机器人的那个面」。

        ``min`` 与 ``mean`` 都只看**几何**（脚在哪一级），不看**谁在承重**。
        按载荷加权则是直接量出物理定义: 承重的脚所在的那一级就是托住机器人的
        面。过渡期后腿还在承重 ⇒ 参照面停在下一级 (与 ``min`` 一致, 不冤枉);
        载荷转移的过程中参照面**连续**爬升 (没有 ``h`` 的突跳); 四只脚都在同一
        级上时与 ``min`` **恒等** (所以真摔倒的判据完全不变)。

        与 ``min`` 相比它修掉的是一类**漏判**: 前脚已经上了第 k+1 级、后脚正在
        从第 k 级塌下去时, ``min`` 读到的是下面那一级, 于是实测 ``height`` 被
        **高估整整一个台阶高**, 判摔迟到 —— 训练期这种「测量比现实宽松」的信号
        正是策略学会一个复现不出来的动作的典型来源 (README §11.24 改动①)。

        四只脚全在空中 (法向力全 0) 时退回 ``min``, 与历史行为一致。
        """
        heights = np.array(
            [self.stair.height_at(float(x)) for x in self.foot_pos[:, 0]],
            dtype=np.float64,
        )
        if heights.size == 0:
            return self.terrain_height_under_base
        weights = np.clip(
            np.asarray(self._touch.foot_normal_force, dtype=np.float64), 0.0, None
        )
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 1e-9:
            return float(heights.min())
        return float((heights * weights).sum() / total)

    def _height_reference(self) -> float:
        """**观测**用的高度参照面（只给 ``obs.base_height`` 用）。

        ``obs.base_height_support = True`` 时用无跳变的 support 口径，
        否则保持历史行为（机身 x 单点）。

        ⚠ 不要把它接到奖励上：``reward.base_height`` 那一项继续用单点参照面，
        因为 ``obs.base_height_support`` 必须是一个**只动观测**的单变量开关
        （实测把它同时接到奖励上会让均回报从 +554 崩到 -674/回合，两个变量就
        分不开了）。奖励侧要用足端参照面请用 ``reward.stair_progress_foot_ref``。
        """
        if bool(self.cfg.obs.base_height_support):
            return self._support_height()
        return self.terrain_height_under_base

    def _loaded_foot_count(self, eps: float) -> int:
        """法向力超过 ``eps`` 的脚数 —— 判断「真的站在地上」而不是「飘着」。"""
        forces = np.asarray(self._touch.foot_normal_force, dtype=np.float64)
        return int(np.count_nonzero(forces > float(eps)))

    def _fall_reference_height(self) -> float:
        """Terrain height the fall test measures the trunk against.

        ``terminate_height_ref = "support"`` returns the lowest terrain under the
        four feet.  That is the surface actually holding the robot up, so it has
        no riser discontinuity: while the front feet are already on the next
        tread and the hind feet are still on the current one, the reference stays
        on the current (lower) tread, which is exactly the level the trunk is
        standing above during the manoeuvre.

        ``"support_mean"`` returns the **mean** terrain height under the four
        feet instead (see :meth:`_support_height_mean`): identical to
        ``"support"`` whenever all four feet share a tread, but it ramps in
        ``step_height / 4`` increments while the feet cross a riser one at a
        time, removing the single full-riser jump that costs the robot
        ``step_height + terminate_min_height`` of trunk clearance for one frame.

        It also stays honest about real falls: once the robot has collapsed, all
        four feet are on whichever tread it collapsed onto, so the reference
        becomes that tread and a low trunk is caught immediately.
        """
        mode = str(self.cfg.terminate_height_ref).strip().lower()
        if mode == "support_mean":
            return self._support_height_mean()
        if mode == "support_load":
            return self._support_height_load()
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

    # ------------------------------------------------------------ gymnasium

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._apply_domain_randomization()
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
        # Reference pose for the "walk straight" shaping terms.
        self._y0 = float(self.base_pos[1])
        self._yaw0 = float(self.base_rpy[2])
        self._prev_action = np.zeros(self.num_act)
        self._last_action = np.zeros(self.num_act)
        self._step_count = 0
        self._reached_top = False
        self._trunk_contact_streak = 0
        self._leg_contact_streak = 0
        self._pending_ctrl.clear()
        self._prev_foot_pos = self.foot_pos.copy()
        # ``foothold`` 的差分基准必须在 reset 后归位，否则跨回合的第一个控制步
        # 会看到上一回合结尾的高度，凭空拿到一次「爬升」。
        self._prev_support_height = self._support_height()
        self._foot_air_time = np.zeros(len(self.meta.foot_geom_ids))
        self._prev_foot_contact = self._foot_contact_mask()
        self._peak_swing_clearance = np.zeros(len(self.meta.foot_geom_ids))
        self._peak_tread = np.zeros(len(self.meta.foot_geom_ids))
        # Prime the tactile/rangefinder reading so the very first observation
        # frame carries real forces instead of the neutral placeholder.
        self._touch = self.touch_sensor.read(
            base_pos=self.base_pos,
            yaw=self.yaw,
            terrain_height=self.terrain_height_under_base,
        )
        if self.cfg.gait_prior and self.cfg.gait_phase_randomize and self.cfg.gait.freq > 0:
            self._gait_t = float(self._rng.uniform(0.0, self.cfg.gait.period))
        else:
            self._gait_t = 0.0
        self._gait_phase = self._gait_t
        self._hist.clear()
        frame = self._obs_frame()
        for _ in range(self.cfg.obs.history_len):
            self._hist.append(frame.copy())

        return self._observation(), {}

    def _gait_action(self) -> np.ndarray:
        """The gait prior's joint-angle offset (rad) at the current phase.

        Zeros whenever the prior is disabled, so ``_target_from_action`` is a
        single code path for both modes (and for ``GaitParams.static()``).
        """
        if not self.cfg.gait_prior or self.cfg.gait_gain == 0.0:
            return np.zeros(self.num_act, dtype=np.float64)
        return self.cfg.gait_gain * self.cfg.gait.radians(self._gait_phase)

    def _target_from_action(self, action: np.ndarray) -> np.ndarray:
        """Map a policy action to PD joint targets (the whole action pipeline).

        Residual RL keeps the two amplitudes separate on purpose:

            target = home + gain * gait.radians(t) + action_scale * action

        ``gait.action_scale`` says how large the prior's motion is (it is part
        of the searched gait); ``cfg.action_scale`` says how much the policy may
        deviate from it.  Sharing one scale made the policy's exploration noise
        scale with the prior, which destabilised the robot instantly.
        """
        if self.cfg.action_mode == "residual":
            target = (self.home_angles
                      + self._gait_action()
                      + self.cfg.action_scale * action)
        else:
            # "absolute" maps the action straight onto the joint range, so a
            # residual prior has no meaning there; it is deliberately ignored.
            lo, hi = self.act_low, self.act_high
            target = lo + (action + 1.0) * 0.5 * (hi - lo)
        return np.clip(target, self.act_low, self.act_high)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)

        # Residual RL: the prior supplies a working periodic gait, the policy
        # supplies the correction.  The phase reported in the observation is
        # the one this step actually used, so the transition stays Markov.
        self._gait_phase = self._gait_t
        target = self._target_from_action(action)
        self._gait_t += self.cfg.control_dt

        # Actuator latency is applied **once per control step** (20 ms), not once
        # per physics substep (2 ms).  Buffering inside the decimation loop made
        # ``latency_steps <= 3`` mean at most 6 ms of delay, which is far below
        # any real actuator delay and therefore useless as domain randomization.
        target = self._apply_latency(target)

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
        body_contact = bool(trunk_contact or leg_contact)
        foot_contact = self._foot_contact_mask()
        # Tactile + rangefinder reading for this control step.  Must be taken
        # from the post-step contact state, and before the reward is computed.
        self._touch = self.touch_sensor.read(
            base_pos=self.base_pos,
            yaw=self.yaw,
            terrain_height=self.terrain_height_under_base,
        )
        terminated, truncated = self._terminated(trunk_contact, leg_contact)

        # Goal termination.  Reaching the crest / the flat target ends the
        # episode *successfully* instead of letting the robot walk off the far
        # edge of the staircase and fall.  ``_reached_top`` is sticky inside the
        # episode, so a fall on an earlier step still means failure.
        if self._success_reached():
            self._reached_top = True
            if self.cfg.terminate_on_success and not terminated:
                terminated = True
                truncated = False

        reward, terms = self._compute_reward(
            action, trunk_contact, foot_contact, leg_contact=leg_contact
        )
        self._advance_foot_air_time(foot_contact)
        terminal = 0.0
        if terminated:
            terminal = (self.cfg.reward.success if self._reached_top
                        else self.cfg.reward.termination)
            reward += terminal
        terms["terminal"] = float(terminal)

        self._prev_action = self._last_action.copy()
        self._last_action = action.copy()
        # Foot xy displacement must be measured over exactly one control step,
        # so this snapshot has to advance every step.
        self._prev_foot_pos = self.foot_pos.copy()
        self._hist.append(self._obs_frame())

        info = {
            "reward_terms": terms,
            "body_contact": float(body_contact),
            "trunk_contact": float(trunk_contact),
            "leg_contact": float(leg_contact),
            "terminate_contact": str(self.cfg.terminate_contact),
            "terminate_contact_steps": int(self.cfg.terminate_contact_steps),
            "terminate_contact_steps_effective": int(self._effective_contact_steps()),
            "trunk_contact_relaxed": bool(self._trunk_contact_relaxed()),
            "trunk_contact_streak": int(self._trunk_contact_streak),
            "leg_contact_streak": int(self._leg_contact_streak),
            "base_x": float(self.base_pos[0]),
            "base_y": float(self.base_pos[1]),
            "base_z": float(self.base_pos[2]),
            "terrain_height": float(self.terrain_height_under_base),
            "forward_vel": float(self.base_lin_vel_world[0]),
            "roll": float(self.base_rpy[0]),
            "pitch": float(self.base_rpy[1]),
            "crest_height": float(self.stair.crest_height),
            "reached_top": float(self._reached_top),
            "is_success": bool(self._reached_top),
            "fallen": bool(terminated) and not self._reached_top,
            "gait_prior": bool(self.cfg.gait_prior),
            # --- step 11 diagnostics ---------------------------------------
            "lateral_offset": float(self.lateral_offset),
            "spawn_lateral_offset": float(self.spawn_lateral_offset),
            "heading_error": float(self.heading_error),
            "spawn_heading_error": float(self.spawn_heading_error),
            "probe_hit": float(self._touch.probe_hit),
            "probe_step": float(self._touch.probe_step),
            "probe_range": float(self._touch.probe_range),
            "probe_trigger": float(self._touch.trigger(self.cfg.touch)),
            "riser_graze": bool(self._touch.riser_graze.any()),
            "total_normal_force": float(self._touch.total_normal_force),
            "max_slip": float(np.max(self._touch.foot_slip))
            if self._touch.foot_slip.size
            else 0.0,
        }
        return self._observation(), float(reward), terminated, truncated, info

    # ------------------------------------------------------- randomization

    def _apply_domain_randomization(self) -> None:
        """Step 8: friction / mass / latency / IMU bias. Off unless enabled.

        ``cfg.randomize_scale`` interpolates every range towards its nominal
        point, so the same configuration can be ramped from "exactly the model
        we trained on" to "the full range" without touching which knobs are on.
        """
        c = self.cfg
        # Always restore the pristine model first so scaling cannot compound.
        self.model.body_mass[:] = self._nominal_body_mass
        self.model.body_inertia[:] = self._nominal_body_inertia
        self.model.geom_friction[:] = self._nominal_geom_friction
        self._imu_bias[:] = 0.0
        self._latency_steps = 0

        # Interpolation weight.  A scale of 1 returns the configured range
        # untouched, so a run that leaves ``randomize_scale`` alone reproduces
        # the historical behaviour bit-for-bit.
        w = float(np.clip(c.randomize_scale, 0.0, 1.0))

        def toward_nominal(rng: tuple[float, float]) -> tuple[float, float]:
            """Shrink a multiplicative scale range towards 1.0 by ``w``."""
            lo, hi = float(rng[0]), float(rng[1])
            return (1.0 + w * (lo - 1.0), 1.0 + w * (hi - 1.0))

        if c.randomize_friction:
            lo, hi = toward_nominal(c.randomize_friction_range)
            scale = self._rng.uniform(lo, hi, size=self.model.ngeom)
            self.model.geom_friction[:, 0] = np.clip(
                self._nominal_geom_friction[:, 0] * scale, 0.05, 3.0
            )
        if c.randomize_mass:
            lo, hi = toward_nominal(c.randomize_mass_range)
            scale = self._rng.uniform(lo, hi)
            self.model.body_mass[:] = self._nominal_body_mass * scale
            self.model.body_inertia[:] = self._nominal_body_inertia * scale
        if c.randomize_latency:
            max_steps = int(round(w * c.randomize_latency_max))
            self._latency_steps = (int(self._rng.integers(0, max_steps + 1))
                                   if max_steps > 0 else 0)
        if c.randomize_imu:
            self._imu_bias = self._rng.normal(
                0.0, w * c.randomize_imu_sigma, 3
            )
        self._pending_ctrl.clear()

    def _apply_latency(self, ctrl: np.ndarray) -> np.ndarray:
        """Delay ``ctrl`` by ``self._latency_steps`` **control steps**.

        Call this exactly once per :meth:`step` (see the comment there); it must
        not be called per physics substep or the effective delay shrinks by the
        decimation factor.

        ``ctrl`` here is the **PD joint-angle target**, not a torque, so while
        the buffer is still filling it must be primed with the *first* command
        and never with zeros: ``np.zeros_like(ctrl)`` would mean "drive all
        twelve joints to 0 rad", i.e. straighten the legs, which is not a delay
        of anything.

        Honest note on how much that mattered.  A controlled A/B
        (``.tmp/latency_bug_cost.py``, which monkeypatches the old
        ``return np.zeros_like(ctrl)`` back in while keeping everything else
        bit-identical) measured **0.0 percentage points** of difference at
        10.0 cm for ``latency_steps`` 1, 2 and 3 (20 deterministic episodes
        each, ``course22``).  The zero target only ever lands in the first
        ``latency_steps`` control steps *after a reset*, while the robot is
        still standing, so it is a harmless 20-60 ms transient and the fix is
        about correctness rather than speed.  The reason ``--randomize`` with
        the *old* defaults destroyed a trained policy (90% -> 5% at 10.0 cm)
        is genuine delay and friction bandwidth, not this bug; see
        ``EnvConfig.randomize_latency_max`` for the measured severity sweep.
        """
        if not self.cfg.randomize_latency or self._latency_steps == 0:
            return ctrl
        self._pending_ctrl.append(ctrl.copy())
        if len(self._pending_ctrl) > self._latency_steps:
            return self._pending_ctrl.popleft()
        return self._pending_ctrl[0]

    # -------------------------------------------------------------- render

    #: Camera presets understood by :meth:`render`.
    CAMERAS = ("side", "track", "chase", "front", "top")

    def render(self, camera: str = "side", *, width: int = 640, height: int = 480):
        """Offscreen render of the current state.

        ``camera`` is one of :attr:`CAMERAS`:

        ``side``
            The original fixed profile view of the whole staircase (default,
            so existing callers are unaffected).
        ``track``
            Same profile view, but the look-at point follows the robot in x --
            use this for GIFs of a 20 s climb, otherwise the dog walks out of
            frame.
        ``chase``
            Behind and above the robot, looking along +x (the climbing
            direction); best for showing foot placement.
        ``front`` / ``top``
            Straight-on and bird's-eye views, handy for stills.
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
            # ``mjv_defaultOption`` ships ``geomgroup = [1, 1, 1, 0, 0, 0]``, so
            # any geom in group 3, 4 or 5 is invisible by default.  The terrain
            # lives in a dedicated group (see ``terrain.TERRAIN_GEOM_GROUP``) and
            # used to be stamped into the hidden range, which made the staircase
            # vanish from every GIF while the physics kept climbing it.  Turn all
            # six groups on so the renderer can never silently drop the world.
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
        else:  # top
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
