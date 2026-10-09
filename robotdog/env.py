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
from robotdog.tools.terrain import StairSpec, build_terrain_xml
from robotdog.tools.touch import (
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
    """Observation layout.

    ``base_frame_size`` is always present; the boolean blocks below are appended
    in the order they are declared and each one adds a fixed number of values.
    Every block is **off by default**, so the plain observation matches the
    original specification (420 values).

    The measured motivation for each block -- what broke without it -- is in
    ``experience.md`` §3. ``README.md`` §3.2 lists the dimensions.
    """

    history_len: int = 10
    base_lin_vel: bool = False          # world-frame velocity -- poor feature
    base_lin_vel_body: bool = False     # body-frame velocity (preferred)
    base_height: bool = False
    base_height_support: bool = False   # use foot support plane as the reference
    foot_height: bool = False
    gait_phase: bool = False            # sin/cos gait clock
    touch: bool = False                 # 9: 4 forces + 4 slips + 1 riser flag
    stair_probe: bool = False           # 3: obstacle height / distance / hit
    stair_probe_full: bool = False      # +2: probe_slope, probe_far_step
    drift: bool = False                 # 2: lateral offset, heading error
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
        return self.base_frame_size + extra

    @property
    def size(self) -> int:
        return self.frame_size * self.history_len


@dataclass
class RewardConfig:
    """Reward weights.

    The five terms the user asked for are marked ``[SPEC]``.  Every remaining
    term is a documented fix for a specific failure; ``experience.md`` §2 says
    which failure each one closes, and ``README.md`` §3.3 lists the formulas.
    """

    # --- the five requested terms ----------------------------------------- #
    forward_vel: float = 2.5            # [SPEC] Gaussian tracking of command_vx
    forward_vel_sigma: float = 0.22
    orientation: float = -1.2           # [SPEC] roll^2 + pitch^2
    torque: float = -2.0e-4             # [SPEC] sum(tau^2)
    foot_lift: float = 0.6              # [SPEC] swing-clearance shaping
    body_collision: float = -4.0        # [SPEC] torso touching the terrain

    # --- forward motion ---------------------------------------------------- #
    # Linear counterpart to the Gaussian: exactly zero at vx=0, so the gradient
    # never vanishes.  Without it the Gaussian's floor makes standing still pay.
    forward_progress: float = 1.0
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
    lateral_vel: float = -0.8
    yaw_rate: float = -0.15
    base_height: float = -3.0
    joint_vel: float = -2.0e-5
    action_rate: float = -0.02
    joint_limit: float = -0.5
    alive: float = 0.12
    termination: float = -2.0
    success: float = 20.0

    # --- staircase progress ------------------------------------------------ #
    stair_progress: float = 2.5
    stair_progress_foot_ref: bool = False
    stair_progress_min_loaded_feet: int = 0
    stair_progress_force_eps: float = 5.0
    foothold: float = 0.0

    # --- torso collision --------------------------------------------------- #
    body_collision_hi_scale: float = 1.0

    # --- tactile / collision ----------------------------------------------- #
    riser_graze: float = -0.60
    riser_graze_mode: str = "legacy"    # "legacy" | "lifting"
    riser_graze_vz_eps: float = 0.10
    touch_impact: float = -0.50
    touch_impact_target: float = 95.0
    touch_impact_step_scale: float = 0.0
    touch_slip: float = -0.35
    touch_slip_safe: float = 0.55
    touch_excess_clip: float = 5.0      # keeps one jammed step from poisoning V(s)
    stair_probe_lift: float = 1.20
    leg_collision: float = -0.25

    # --- walking straight -------------------------------------------------- #
    straight_line: float = -20.0
    straight_line_max: float = 1.00
    heading: float = -8.0
    heading_max: float = 0.60
    lateral_vel_track: float = 0.5
    lateral_vel_sigma: float = 0.15


@dataclass
class EnvConfig:
    """Everything tunable about one environment instance.

    Defaults are the historical behaviour: every switch added later is ``False``
    or ``0`` so that an old checkpoint stays bit-for-bit reproducible.
    ``experience.md`` §1/§4 explains why each default is where it is.
    """

    timestep: float = 0.002
    decimation: int = 10                    # -> 50 Hz control
    episode_length_s: float = 12.0
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
    gait_phase_randomize: bool = False
    kp: tuple[float, float, float] = (30.0, 30.0, 45.0)
    kd: tuple[float, float, float] = (0.6, 0.6, 0.9)

    # --- termination ------------------------------------------------------- #
    #   "trunk" -> only the torso is fatal; a shin scrape is priced, not fatal
    #   "any"   -> legacy: any non-foot robot geom is fatal
    #   "none"  -> contacts never terminate
    terminate_contact: str = "trunk"
    terminate_body_contact: bool = True
    # Consecutive control steps the selected contact class must persist.
    terminate_contact_steps: int = 1
    contact_debounce_adaptive: bool = False
    contact_debounce_hi_step_h: float = 0.105
    contact_debounce_hi_steps: int = 1
    trunk_contact_adaptive: bool = False
    trunk_contact_hi_step_h: float = 0.105
    # Which surface the "trunk has fallen" height test measures against:
    #   "base" -> terrain under the base (step function -- historical behaviour)
    #   "support" / "support_mean" / "support_load" -> under the four feet
    terminate_height_ref: str = "base"
    terminate_min_height: float = 0.13
    terminate_max_tilt: float = 1.0         # rad, roll or pitch
    terminate_on_success: bool = True

    # --- reset noise ------------------------------------------------------- #
    reset_z: float = 0.32
    reset_joint_noise: float = 0.06
    reset_xy_noise: float = 0.12
    reset_yaw_noise: float = 0.10

    # --- success ----------------------------------------------------------- #
    flat_success_distance: float = 3.0      # flat stages only
    spawn_back: float = 0.55                # m in front of the first riser

    # --- domain randomization (all off by default) ------------------------- #
    randomize_friction: bool = False
    randomize_mass: bool = False
    randomize_latency: bool = False
    randomize_imu: bool = False
    randomize_friction_range: tuple[float, float] = (0.7, 1.3)
    randomize_mass_range: tuple[float, float] = (0.92, 1.08)
    randomize_latency_max: int = 1          # control steps
    randomize_imu_sigma: float = 0.02       # rad
    # Severity dial: 0 = nominal model, 1 = the full configured range.  The
    # curriculum ramps it because the un-ramped range is not learnable.
    randomize_scale: float = 1.0

    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
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
        """Toggle friction/mass/latency/IMU randomization at runtime.

        ``scale`` is the severity dial: 0.0 reproduces the nominal model, 1.0
        uses the configured range in full, and values in between interpolate
        every range towards its nominal point.  The curriculum ramps it 0 -> 1
        because applying the full range from the first step of a stage is not
        learnable here with this controller.
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
            # Deliberately keeps the single-point reference rather than
            # ``_height_reference()``: ``obs.base_height_support`` has to stay a
            # single-variable switch on the observation.  For a foot-based
            # reward reference use ``stair_progress_foot_ref``.
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
        # The reference is either the single-point height under the base
        # (historical) or the lowest terrain under the four feet.
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
            terms["riser_graze"] = rc.riser_graze * float(stuck)
        else:
            terms["riser_graze"] = rc.riser_graze * float(
                touch.riser_graze.any()
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
            terms["stair_probe_lift"] = rc.stair_probe_lift * float(
                touch.trigger(self.cfg.touch) * np.mean(lift[:2])
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

    def _support_height_load(self) -> float:
        """Normal-force weighted terrain height under the feet.

        Answers "which tread is actually carrying the robot", rather than
        "which treads are the feet over".  Identical to ``support`` when all
        four feet share a tread; continuous while the load transfers; and it
        closes a *missed* fall that ``support`` has (front feet up, hind feet
        collapsing off the lower tread, where ``min`` overstates the trunk
        clearance by a full riser height).  Falls back to ``min`` when no foot
        carries load.
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
        """Height reference for ``obs.base_height`` only.

        Do **not** wire this into the reward: ``obs.base_height_support`` has to
        stay a single-variable switch on the observation (measured: also wiring
        it into the reward collapsed the mean return from +554 to -674 per
        episode).  For a foot-based reward reference use
        ``reward.stair_progress_foot_ref``.
        """
        if bool(self.cfg.obs.base_height_support):
            return self._support_height()
        return self.terrain_height_under_base

    def _loaded_foot_count(self, eps: float) -> int:
        """Number of feet whose normal force exceeds ``eps``."""
        forces = np.asarray(self._touch.foot_normal_force, dtype=np.float64)
        return int(np.count_nonzero(forces > float(eps)))

    def _fall_reference_height(self) -> float:
        """Terrain height the fall test measures the trunk against.

        See :meth:`_support_height_mean` / :meth:`_support_height_load` for the
        variants; ``"base"`` (the default) is the historical step function.
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

        Call once per :meth:`step`, never per physics substep, or the effective
        delay shrinks by the decimation factor.

        ``ctrl`` is the PD joint-angle target, so while the buffer fills it must
        be primed with the first command and never with zeros -- an all-zero
        target means "straighten all twelve joints", which is a delay of
        nothing.  Measured cost of that bug: 0.0 percentage points at 10.0 cm
        (see ``experience.md`` §5).
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
