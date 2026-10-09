"""Gymnasium environment: Unitree Go2 walking blind up box stairs.

**Blind locomotion.**  There is no touch sensor, no ray casting and no contact
force reading anywhere in this module.  The policy sees only proprioception
joint angles/velocities, base attitude and rate, the previous action) plus a
single scalar telling it how tall the staircase it was placed on is.  It has to
discover the stepping rhythm from that alone.

* **Action** -- 12 joint target angles (``FL_hip .. RR_calf``).
* **Observation** -- ``ObsConfig``: 42 proprioceptive values per frame (joint
  positions/velocities, base roll/pitch/yaw, angular velocity, previous action)
  plus the blocks that are switched on; the last ``history_len`` frames are
  stacked.  No contact or range-sensor data exists anywhere.
* **Reward** -- ``RewardConfig``: forward speed, posture, joint torque, foot
  lift, torso/leg collision, staircase progress (plus standard shaping).

Frame conventions:

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
from robotdog.tools.terrain import StairSpec, build_terrain_xml

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class ObsConfig:
    """Observation layout.

    ``base_frame_size`` is always present (42 proprioceptive values).  The four
    boolean blocks below are appended in declaration order; ``step_height`` is the
    one the blind policy cannot infer cheaply, so it is a plain number rather
    than something the robot has to estimate.  ``README.md`` §3.2 lists the
    resulting dimensions.
    """

    history_len: int = 10
    base_lin_vel_body: bool = True      # 3: body-frame velocity
    base_height: bool = False           # 1: height above the terrain under the base
    gait_phase: bool = False            # 2: sin/cos gait clock
    step_height: bool = True            # 1: the staircase height of this episode
    base_frame_size: int = 42

    @property
    def frame_size(self) -> int:
        extra = 0
        extra += 3 if self.base_lin_vel_body else 0
        extra += 1 if self.base_height else 0
        extra += 2 if self.gait_phase else 0
        extra += 1 if self.step_height else 0
        return self.base_frame_size + extra

    @property
    def size(self) -> int:
        return self.frame_size * self.history_len


@dataclass
class RewardConfig:
    """Reward weights.

    Roughly balancing positive and negative terms.  Historical tuning that these
    values come from -- which failure each weight closes -- is in
    ``experience.md`` §2, formulas in ``README.md`` §3.3.
    """

    # --- the five requested terms ----------------------------------------- #
    forward_vel: float = 2.5            # [SPEC] Gaussian tracking of command_vx
    forward_vel_sigma: float = 0.25
    orientation: float = -1.2           # [SPEC] roll^2 + pitch^2
    torque: float = -1.5e-4             # [SPEC] sum(tau^2)
    foot_lift: float = 0.9              # [SPEC] swing-clearance shaping
    body_collision: float = -4.0        # [SPEC] torso touching the terrain

    # --- forward motion ---------------------------------------------------- #
    forward_progress: float = 1.5       # linear in vx, so vx=0 earns nothing
    forward_progress_max: float = 1.5
    forward_progress_min: float = -0.5

    # --- swing-foot lift --------------------------------------------------- #
    foot_lift_target: float = 0.08      # floor of the target, not the whole target
    foot_lift_step_ratio: float = 1.4   # target = max(target, ratio * step_height)
    foot_lift_gate_speed: float = 0.25  # m/s 3-D foot speed gate

    # --- gait discovery ---------------------------------------------------- #
    air_time: float = 0.6
    air_time_target: float = 0.30
    air_time_max: float = 0.60

    # --- supporting shaping ------------------------------------------------ #
    base_height: float = -2.0
    joint_vel: float = -2.0e-5
    action_rate: float = -0.02
    joint_limit: float = -0.5
    alive: float = 0.15
    termination: float = -15.0
    success: float = 50.0

    # --- staircase progress ------------------------------------------------ #
    stair_progress: float = 3.0
    leg_collision: float = -0.25


@dataclass
class EnvConfig:
    """Everything tunable about one environment instance."""

    timestep: float = 0.002
    decimation: int = 10                    # -> 50 Hz control
    episode_length_s: float = 20.0
    action_mode: str = "residual"           # "residual" | "absolute"
    action_scale: float = 0.30              # rad, policy authority (residual only)
    command_vx: float = 0.55                # m/s forward command
    command_vy: float = 0.0
    command_yaw_rate: float = 0.0

    # --- residual RL over an open-loop gait -------------------------------- #
    # PD target becomes home + gait_gain * gait.radians(t) + action_scale * action
    gait_prior: bool = False
    gait: GaitParams = field(default_factory=GaitParams.default_trot)
    gait_gain: float = 1.0
    kp: tuple[float, float, float] = (30.0, 30.0, 45.0)
    kd: tuple[float, float, float] = (0.6, 0.6, 0.9)

    # --- termination ------------------------------------------------------- #
    # Number of consecutive control steps the trunk must be in contact with the
    # terrain before the episode ends.  >1 debounces single-frame impacts.
    terminate_contact_steps: int = 10
    # Which surface the "trunk has fallen" height test measures against:
    #   "base" -> terrain under the base (step function, gives ghost falls)
    #   "support" -> terrain under the four feet (what actually supports it)
    terminate_height_ref: str = "support"
    terminate_min_height: float = 0.13
    terminate_max_tilt: float = 1.0         # rad, roll or pitch
    terminate_on_success: bool = True

    # --- reset noise ------------------------------------------------------- #
    reset_z: float = 0.32
    reset_joint_noise: float = 0.06
    reset_xy_noise: float = 0.12
    reset_yaw_noise: float = 0.10

    # --- success ----------------------------------------------------------- #
    flat_success_distance: float = 3.0      # flat stage only
    spawn_back: float = 0.55                # m in front of the first riser

    # --- observation / reward ---------------------------------------------- #
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

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
        self._reached_top = False
        # Consecutive trunk / leg terrain-contact streaks (see
        # ``EnvConfig.terminate_contact_steps``).
        self._trunk_contact_streak = 0
        self._leg_contact_streak = 0
        self._success_x = self._success_target_x()
        self.nominal_pose = float(self.meta.home_qpos[2])  # 0.27 m stance height
        self._prev_foot_pos = self.foot_pos.copy()
        self._foot_air_time = np.zeros(len(self.meta.foot_geom_ids))
        self._prev_foot_contact = np.zeros(len(self.meta.foot_geom_ids), dtype=bool)
        # Gait-prior clock (seconds since episode start).
        self._gait_t = 0.0
        self._gait_phase = 0.0
        self._rng = np.random.default_rng(seed)
        self._renderer: mujoco.Renderer | None = None
        self._render_cam = None
        self._scene_option = None
        self._render_size = None

    # ------------------------------------------------------------------ build

    def _build_model(self) -> None:
        opts = SceneOptions(timestep=self.cfg.timestep)
        self.model, self.meta, self.xml = compile_scene(
            build_terrain_xml(self.stair),
            opts=opts,
            floor_geom=default_floor_geom(),
        )
        self.data = mujoco.MjData(self.model)
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
        parts = [
            self.joint_pos,
            self.joint_vel,
            self.base_rpy,
            self.base_ang_vel_body,
            self._last_action,
        ]
        if obs_cfg.base_lin_vel_body:
            parts.append(self.base_lin_vel_body)
        if obs_cfg.base_height:
            parts.append(np.array([self.base_pos[2] - self._height_reference()]))
        if obs_cfg.gait_phase:
            theta = 2.0 * np.pi * self.cfg.gait.freq * self._gait_phase
            parts.append(np.array([np.sin(theta), np.cos(theta)]))
        if obs_cfg.step_height:
            # The one piece of the staircase the policy is told outright.  Blind
            # locomotion still needs to know *which* staircase it is on; it does
            # not get to know where the next riser is.
            parts.append(np.array([float(self.stair.step_height)]))
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
        # 3-D foot speed: the horizontal component alone would reward a fast
        # flat sweep and cut off the slow lift a tall step needs.
        foot_speed = np.linalg.norm(
            (feet - self._prev_foot_pos) / foot_dt, axis=1
        )
        gate = np.clip(foot_speed / max(float(rc.foot_lift_gate_speed), 1e-6),
                       0.0, 1.0)
        # geom_xpos is the SPHERE CENTRE, so subtract the radius to get the
        # real gap between the foot surface and the terrain.
        clearance = self.foot_clearance
        # The target grows with the terrain, but at 5 cm it equals the old
        # constant, so an already-learned easy stage does not regress.
        lift_target = max(
            rc.foot_lift_target, rc.foot_lift_step_ratio * float(self.stair.step_height)
        )
        lift = np.clip(clearance / lift_target, 0.0, 1.0)
        terms["foot_lift"] = rc.foot_lift * float(np.mean(gate * lift))

        # 5) 机身碰撞惩罚 -- torso touching the terrain ---------------------
        # Per-step and bounded: it scales with how long the torso scrapes
        # rather than being a single boolean at the terminal step.
        terms["body_collision"] = rc.body_collision * float(trunk_contact)
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
        terms["base_height"] = rc.base_height * float(
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

        # --- 爬升进展 -------------------------------------------------------
        # Terrain height under the base relative to where the episode started, so
        # standing still on a step already reached earns nothing new.
        terms["stair_progress"] = rc.stair_progress * float(
            max(0.0, self.terrain_height_under_base - self._start_terrain_height)
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

    def _effective_contact_steps(self) -> int:
        """Consecutive control steps of trunk contact that end the episode."""
        return max(1, int(self.cfg.terminate_contact_steps))

    def _contact_fatal(self, trunk_contact: bool, leg_contact: bool) -> bool:
        """Has the torso been in contact with the terrain long enough to be fatal?

        The streak counter is advanced here rather than in ``step`` so that "one
        call to this method == one control step" also holds when a test drives it
        directly.  A shin scrape is priced by ``reward.leg_collision``, never
        fatal -- killing the episode on it removes the very signal a tall step
        needs.
        """
        self._trunk_contact_streak = self._trunk_contact_streak + 1 if trunk_contact else 0
        self._leg_contact_streak = self._leg_contact_streak + 1 if leg_contact else 0
        if not trunk_contact:
            return False
        return self._trunk_contact_streak >= self._effective_contact_steps()

    def _support_height(self) -> float:
        """Lowest terrain height under the four feet -- the supporting surface.

        Contrast with :meth:`terrain_height_under_base`, which is a step
        function of the trunk's x position.
        """
        heights = [self.stair.height_at(float(x)) for x in self.foot_pos[:, 0]]
        return float(min(heights)) if heights else self.terrain_height_under_base

    def _support_height_mean(self) -> float:
        """Mean terrain height under the four feet.

        ``support`` (the minimum) jumps by a full riser in one frame when the
        last foot lands; the mean spreads that over four ``h/4`` increments.
        It is *not* simply more permissive -- during a crossing the mean is
        higher than the minimum, so it fires earlier; what it removes is the
        single-frame discontinuity.
        """
        heights = [self.stair.height_at(float(x)) for x in self.foot_pos[:, 0]]
        return float(np.mean(heights)) if heights else self.terrain_height_under_base

    def _height_reference(self) -> float:
        """Height reference for ``obs.base_height``: the terrain under the base."""
        return self.terrain_height_under_base

    def _fall_reference_height(self) -> float:
        """Terrain height the fall test measures the trunk against.

        ``"support"`` uses the lowest terrain under the four feet, which is what
        is actually holding the robot up; the terrain under the *base* is a step
        function of the trunk's x position and flips by a whole riser between two
        adjacent frames, producing ghost falls.
        """
        mode = str(self.cfg.terminate_height_ref).strip().lower()
        if mode == "support_mean":
            return self._support_height_mean()
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
        self._prev_action = np.zeros(self.num_act)
        self._last_action = np.zeros(self.num_act)
        self._step_count = 0
        self._reached_top = False
        self._trunk_contact_streak = 0
        self._leg_contact_streak = 0
        self._prev_foot_pos = self.foot_pos.copy()
        self._foot_air_time = np.zeros(len(self.meta.foot_geom_ids))
        self._prev_foot_contact = self._foot_contact_mask()
        # Gait-prior clock, in seconds since the start of the episode.  The phase
        # reported in the observation is the one the step actually used.
        self._gait_t = 0.0
        self._gait_phase = 0.0
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
        foot_contact = self._foot_contact_mask()
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
            "trunk_contact": float(trunk_contact),
            "leg_contact": float(leg_contact),
            "terminate_contact_steps": int(self.cfg.terminate_contact_steps),
            "trunk_contact_streak": int(self._trunk_contact_streak),
            "base_x": float(self.base_pos[0]),
            "base_y": float(self.base_pos[1]),
            "base_z": float(self.base_pos[2]),
            "terrain_height": float(self.terrain_height_under_base),
            "forward_vel": float(self.base_lin_vel_world[0]),
            "roll": float(self.base_rpy[0]),
            "pitch": float(self.base_rpy[1]),
            "crest_height": float(self.stair.crest_height),
            "reach_top_x": float(self.stair.up_end_x),
            "reached_top": float(self._reached_top),
            "is_success": bool(self._reached_top),
            "fallen": bool(terminated) and not self._reached_top,
            "gait_prior": bool(self.cfg.gait_prior),
        }
        return self._observation(), float(reward), terminated, truncated, info

    # -------------------------------------------------------------- render

    #: Camera presets understood by :meth:`render`.
    CAMERAS = ("side", "track", "chase", "front", "top")

    def render(self, camera: str = "side", *, width: int = 640, height: int = 480):
        """Offscreen render of the current state; ``camera`` is in :attr:`CAMERAS`.

        ``side`` (default) is the fixed whole-staircase profile view, ``track``
        the same view with the look-at point following the robot in x, ``chase``
        from behind/above along +x, and ``front`` / ``top`` for stills.
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
