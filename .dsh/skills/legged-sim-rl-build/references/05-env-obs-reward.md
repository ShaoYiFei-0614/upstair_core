# 环境类 · 观测契约 · 奖励数学 · 动作管线 · 配置层

> 对应 `SKILL.md` 阶段 6–7。素材取自 upstair_core（Unitree Go2 盲走楼梯 · MuJoCo 3.12 + gymnasium 1.3 + SB3 2.9） 的真实源码与实测记录。
> 具体数值只作**取证**用；换机器人时按文中的「换机器人要改什么」逐条替换。

奖励项的形状与权重以**源码 + `config.yaml`** 为准：本仓的奖励形状经历过「高斯核 → 线性死区」的改写，若与 `README.md`/`docs/robotdog/reward.md` 的旧描述冲突，以后者为准会读错（详见 `07-pitfalls.md` B4/B5、`experience.md` §18–§19）。

## 1. 环境类结构 `Go2StairsEnv`

`class Go2StairsEnv(gym.Env)`（`robotdog/env.py`，1066 行，单文件含环境+奖励转发+终止+渲染）

签名（逐字）：
```python
metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
CAMERAS = ("side", "track", "chase", "front", "top")
def __init__(self, stair: StairSpec | None = None, cfg: EnvConfig | None = None, *, seed: int | None = None)
def reset(self, *, seed: int | None = None, options=None)
def step(self, action: np.ndarray)
def render(self, camera: str = "side", *, width: int = 640, height: int = 480)
def close(self) -> None
def set_stair(self, stair: StairSpec) -> None
def set_commands(self, vy_max: float | None = None, yaw_max: float | None = None) -> None
def set_command_now(self, vy: float | None = None, yaw_rate: float | None = None, vx: float | None = None) -> list[float]
def set_flat_success_distance(self, distance) -> None
def set_flat_success_needs_progress(self, flag: bool) -> None
def set_reward_overrides(self, overrides: dict) -> None
def _obs_frame(self) -> np.ndarray                 # float64, 长 obs.frame_size
def _observation(self) -> np.ndarray               # float32, 长 obs.size
def _compute_reward(self, action, trunk_contact, foot_contact=None, leg_contact=False, terminated=False) -> tuple[float, dict]
def _terminated(self, trunk_contact: bool, leg_contact: bool) -> tuple[bool, bool]
def _success_reached(self) -> bool
def _contact_flags(self) -> tuple[bool, bool]
def _contact_fatal(self, trunk_contact, leg_contact) -> bool    # 恒 False
def _support_height(self) -> float ; def _support_height_mean(self) -> float
def _fall_reference_height(self) -> float
def _gait_action(self) -> np.ndarray ; def _target_from_action(self, action) -> np.ndarray
def _foot_contact_mask(self) -> np.ndarray ; def _advance_foot_air_time(self, foot_contact) -> None
def _track_swing_peak(self, foot_contact) -> None ; def _update_probe_state(self, foot_contact) -> None
def _sample_command(self) -> list[float] ; def _success_target_x(self) -> float ; def _terrain_height(self, x)
```

构造成员：`num_act=12`；`act_low/act_high = meta.joint_ranges[:,0/1]`；`home_angles = meta.home_joint_angles`；
`action_space = Box(-1,1,(12,),float32)`；`observation_space = Box(-inf,inf,(cfg.obs.size,),float32)`；
`_kp = np.tile(cfg.kp,4)`、`_kd = np.tile(cfg.kd,4)`；`_max_steps = int(episode_length_s/control_dt)`=1000；
`_hist = deque(maxlen=history_len)`；`_prev_action/_last_action = zeros(12)`；`_gait_t/_gait_phase = 0.0`；
`_rng = np.random.default_rng(seed)`；`nominal_pose = float(meta.home_qpos[2])` = **0.27**；
`_command = cfg.command._now`（**同一个 list 对象**）；`_renderer/_scene_option/_render_size = None`。
`_build_model()`：`compile_scene(build_terrain_xml(stair), SceneOptions(timestep=cfg.timestep), floor_geom=default_floor_geom())`
→ `model/meta/xml` + `MjData`；重建前必须先 `close()` 旧 renderer（`Renderer` 绑定 MjModel）。
只读属性：`base_pos=qpos[0:3]`、`base_rpy=quat_to_rpy(qpos[3:7])`、`base_quat=qpos[3:7](w,x,y,z)`、
`base_lin_vel_world=qvel[0:3]`、`base_lin_vel_body=quat_to_mat(base_quat).T @ qvel[0:3]`、
`base_ang_vel_body=qvel[3:6]`（**陀螺读数，机体系，勿做变换**）、`joint_pos/joint_vel`、
`foot_pos=geom_xpos[foot_geom_ids] (4,3)`（**球心**）、`terrain_height_under_base=stair.height_at(base_x)`、
`yaw=base_rpy[2]`、`yaw_rate=(quat_to_mat(base_quat).T @ qvel[3:6])[2]`（机体系）、
`foot_clearance = feet[:,2] - meta.foot_radius - terrain_height(feet[:,0])`（特权量，只进奖励）、`probe_target`。

### 1.1 `step()` 完整流水线（严格有序）
```python
action = np.asarray(action, float64).clip(-1, 1)              # ①
self._gait_phase = self._gait_t ; target = self._target_from_action(action) ; self._gait_t += control_dt   # ②③④
for _ in range(decimation):                                    # ⑤ 500 Hz ×10
    tau = self._kp * (target - q) - self._kd * qd
    self.data.ctrl[:] = np.clip(tau, meta.ctrl_ranges[:,0], meta.ctrl_ranges[:,1]) ; mujoco.mj_step(model, data)
self._step_count += 1                                          # ⑥
trunk_contact, leg_contact = self._contact_flags()             # ⑦
foot_contact = self._foot_contact_mask()                       # ⑧
self._track_swing_peak(foot_contact) ; self._update_probe_state(foot_contact)   # ⑨ 必须在奖励前
terminated, truncated = self._terminated(trunk_contact, leg_contact)           # ⑩
self.filtered_lateral_vel += alpha * (lin_body[1] - self.filtered_lateral_vel) # ⑪ 低通，奖励前
self.filtered_yaw_rate    += alpha * (self.yaw_rate - self.filtered_yaw_rate)  #    alpha = min(1, control_dt/vel_filter_tau)
if self._success_reached(): self._reached_top = True ; if terminate_on_success and not terminated: terminated, truncated = True, False   # ⑫
reward, terms = self._compute_reward(action, trunk_contact, foot_contact, leg_contact, terminated)   # ⑬
self._advance_foot_air_time(foot_contact)                      # ⑭ 必须在奖励后（air_time 读"本步之前"的累积）
self._prev_action = self._last_action.copy() ; self._last_action = action.copy()   # ⑮
self._prev_foot_pos = self.foot_pos.copy()                     # ⑯ 脚的位移必须恰好跨一个控制步
self._hist.append(self._obs_frame())                           # ⑰
return self._observation(), float(reward), terminated, truncated, info           # ⑱
```
`info` 键（当前代码，14 键）：`reward_terms, trunk_contact, leg_contact, leg_contact_streak, blocked, probe_count,
base_x, base_y, base_z, terrain_height, forward_vel, lateral_vel, yaw_rate, command, roll, pitch, crest_height,
reach_top_x, reached_top, is_success, fallen, gait_prior`（`is_success = bool(_reached_top)`，
`fallen = bool(terminated) and not _reached_top`；`reward_terms` = 13 项 + `terminal`）。

### 1.2 `reset()`（有序）
`super().reset(seed)` → `mj_resetData` → `x = stair.start_x - spawn_back + U(±reset_xy_noise)`、
`y = stair.center_y + U(±)`、`yaw = U(±reset_yaw_noise)` → `home_qpos(meta, x, y, terrain_height_under_base + reset_z, yaw)`
→ 关节角 `+= U(±reset_joint_noise,12)` → `qvel = U(-0.05,0.05,nv)` → `mj_forward` →
重置全部记账字段（`_command[:] = _sample_command()`、`_prev_foot_contact/_prev_foot_pos`、
`_gait_t/_gait_phase = 0`、`filtered_*` 用第一步瞬时值、`_peak_terrain_height = _start_terrain_height`、`_reached_top=False`）
→ `_hist.clear()`，把**同一帧复制 `history_len` 次**（补零会造出分布外样本）→ `return self._observation(), {}`。

### 1.3 其余关键成员
- `_obs_frame()`：`[joint_pos(12), joint_vel(12), base_rpy(3), base_ang_vel_body(3), _last_action(12)] = 42`，
  再按**字段声明顺序**追加 `base_lin_vel_body(3)` / `gait_phase(2, theta=2π·freq·_gait_phase→[sin,cos])` /
  `step_height(1)` / `command(self._command[1:].copy(), 2)` / `vel_filter(2)`；`base_height` 分支**已整体删除**。
- `_terminated(tc, lc)`：`height = base_pos[2] - _fall_reference_height()`；
  `fell = height < terminate_min_height or abs(rpy[0]) > terminate_max_tilt or abs(rpy[1]) > terminate_max_tilt`；
  `truncated = _step_count >= _max_steps`；返回 `(fell or _contact_fatal(...), truncated)`。
- `_contact_fatal()` **恒返回 False**（只维护 `_leg_contact_streak` 供诊断）。
- `_success_reached()`：楼梯 → `base_pos[0] > stair.up_end_x and terrain_height_under_base >= crest_height - 1e-9`；
  平地且 `not needs_progress` → `_step_count >= _max_steps - 1`（少了这条第 1 步即"成功"）；否则 `base_pos[0] > _success_x`。
- `render()`：校验 `camera ∈ CAMERAS` 否则 `ValueError(f"未知相机 {camera!r}, 可选 {self.CAMERAS}")`；
  **必须 `for g in range(6): scene_option.geomgroup[g] = 1`** —— `mjv_defaultOption` 隐藏 group 3/4/5，
  机器人碰撞体在 group 3（曾把楼梯藏进隐藏组，物理照爬、GIF 爬空气）；
  preset（azimuth/elev/distance/lookat）：`side 90/-12/4/(start_x+1,0,0.3)`、`track 90/-9/2.5/(base+0.35)`、
  `chase 135/-22/3/(base+0.6,+0.10)`、`front 180/-10/3/(base+1.2)`、`top 90/-85/4.5/(base+0.8)`。

---

## 2. 观测契约

| 块 | 维 | 默认 | config.yaml | 内容 | 真机出处 |
|---|---|---|---|---|---|
| 基础帧 | 42 | 恒在 | — | joint_pos12 + joint_vel12 + base_rpy3 + base_ang_vel_body3 + last_action12 | 关节编码器 / IMU 姿态 / 陀螺 / 上一步动作 |
| `base_lin_vel_body` | 3 | True | true | `R(q)ᵀ qvel[0:3]` | 状态估计器（真机需估） |
| `base_height` | 1 | False | false | **禁止打开** | — |
| `gait_phase` | 2 | False | false | `[sin θ, cos θ]`, θ=2π·freq·t | 内部步态时钟 |
| `step_height` | 1 | True | true | `float(stair.step_height)` | **任务参数，随任务下发**（唯一非传感器项） |
| `command` | 2 | True | （恒真） | `[vy, wz]`；`vx` 是定值不进观测 | 上层遥控 |
| `vel_filter` | 2 | False | false | `[filtered_lateral_vel, filtered_yaw_rate]` | — |

`ObsConfig`（`robotdog/env.py`）：`history_len=10, base_lin_vel_body=True, base_height=False, gait_phase=False,
step_height=True, command=True, vel_filter=False, base_frame_size=42`；
`frame_size = 42 + 3*lin_vel_body + 2*gait_phase + 1*step_height + 2*command + 2*vel_filter`；`size = frame_size * history_len`。

**硬性禁止项**：`__post_init__` 里 `if self.base_height: raise ValueError("obs.base_height 必须为 false：…")
—— 该项实现是 `base_pos[2] - terrain_height_under_base`，含**地形几何真值**，盲走策略拿到就不是盲走。
`_obs_frame()` 里对应分支与 `_height_reference()` 均已删除；`policy.py:_obs_config_for_dim` 的 16 组枚举里
**故意不含 base_height**，所以反推永远造不出开它的配置。模块 docstring 逐字约束：
"There is no contact sensor, no ray casting, and no contact force reading anywhere in this module."

**维数恒定约束**：PPO 的 ActorCritic 网络与 `RolloutBuffer` 维度在 `learn()` 开始后**不可变**，
所以课程只能改指令的**范围**（`vy_max/yaw_max`）不能开关这 2 维；`ObsConfig.command` 恒真。
本仓 `env.obs.history_len = 1` ⇒ **48 维观测**（dataclass 默认 10 ⇒ 480 维，`EnvConfig()` 默认 460/480）。
合法 frame_size = 42~50（command 恒真 ⇒ 实际 47~50）。

---

## 3. 奖励机制（`robotdog/reward.py`，375 行，**独立纯函数模块**）

```python
def compute_reward(env, action: np.ndarray, trunk_contact: bool, foot_contact: np.ndarray | None = None,
                   leg_contact: bool = False, terminated: bool = False
                   ) -> tuple[float, dict[str, float], float]:   # (sum(terms), terms, terminal)
```
**只读** `env`（不写任何字段）；`terms` 每项**已乘权重**。`env._compute_reward` 只做三行转发：
```python
base, terms, terminal = compute_reward(self, action, trunk_contact, foot_contact, leg_contact=leg_contact, terminated=terminated)
terms["terminal"] = terminal
return base + terminal, terms
```
**账目恒等式**：`step()` 返回的 `reward == sum(info["reward_terms"].values())`（含 terminal，14 键）；
`reward == 逐步奖励 + 终止奖励`；`reward.py` 返回的 `terms` 是 13 键**不含** terminal（不是同一个 dict）。

`RewardConfig` 全项（`默认值` / 口径公式；★=一次性结算，◆=势函数塑形，⛔=必须门控）：

| 项 | 默认 | 公式 |
|---|---|---|
| forward_vel | 10.0 | `W * max(0, 1 - |lin[0]-cmd.vx| / max(abs(cmd.vx), 0.30))` 线性死区 |
| forward_vel_sigma | 0.25 | 历史字段，不再参与计算 |
| orientation | -1.2 | `W * (rpy[0]² + rpy[1]²)` |
| torque | -1.5e-4 | `W * sum(data.actuator_force²)` |
| foot_lift | 0.9 | `W * mean(gate*lift)`；`gate=clip(‖(feet-_prev_foot_pos)/dt‖/0.25,0,1)`（**三维**速度）；`lift=clip(clearance/max(0.08, 1.4·h),0,1)` |
| body_collision | -1.2 | `W * float(trunk_contact)` 逐步有界、**不终止** |
| leg_collision | -0.25 | `W * float(leg_contact)` 逐步、**永不致命** |
| lateral_vel (+sigma 0.25) | 6.0 | ⛔ `cmd.vy_max>0` 时并入 `W*max(0,1-|vy-cmd.vy|/cmd.vy_max)` |
| yaw_rate (+sigma 0.25) | 4.0 | ⛔ `cmd.yaw_max>0` 时并入 `W*max(0,1-yaw_err/cmd.yaw_max)`；`wz==0` 时 `yaw_err=0.5*(yaw_err+|wz|)` 基线惩罚 |
| command_follow | （键名） | 上两项**求和**，两者都 0 时恒 `0.0`（逐位精确） |
| air_time | 0.6 | ★ 仅落地帧：`W*sum((clip(air-0.30,0)-clip(air-0.60,0))*landed)`，`landed=fc & ~_prev_fc` |
| base_height | -2.0 | `W * (base_pos[2] - terrain_height_under_base - 0.27)²`（**T** 单点参照面，与观测解耦） |
| joint_vel | -2.0e-5 | `W * sum(joint_vel²)` |
| action_rate | -0.02 | `W * sum((action - _prev_action)²)` |
| joint_limit | -0.5 | `W * sum(over(lo-q) + over(q-hi))`（软限位 `act_low/act_high`，**非** ctrl_ranges） |
| alive | 0.15 | 常数（与状态无关） |
| stair_progress | 3.0 | ★ `W * max(0, terrain_height_under_base - _peak_terrain_height)`，>0 时前移峰值 |
| probe_lift | 0.0（基线；课程逐级打开） | ◆ 门控 `_blocked`：`W*sum(clip(_peak_swing_clearance - max(_probe_best, probe_lift_target),0,None))`，>0 时前移 `_probe_best` |
| probe_lift_target / ratio / max | 0.06 / 1.0 / 12 | 起算线 / 保留字段 / 保留字段 |
| probe_blocked_frac | 0.30 | `vx < 0.30*cmd_vx` ⇒ `_blocked=True` |
| termination / success | -15.0 / 50.0 | terminal：`success if _reached_top else termination`，仅 `terminated=True` 时非 0 |

**参数 vs 键**：30 字段中 13 个是参数不是 key（sigma ×3、foot_lift_target/step_ratio/gate_speed、
probe_lift_ratio/max、air_time_target/max、termination/success）。
**设计要点**：① 抽成纯函数（奖励是唯一"写错不报错、几小时后才发现"的部分；抽离后做过等价性验证
`.tmp/verify_reward_refactor.py`：4,503 步 `max|Δreward|=0`、`max|Δobs|=0`）；
② 明细已乘权重 ⇒ `sum(terms)` 就是本步奖励；③ 终止奖励单列（`terms` 不含 terminal）。
**必须门控的经验**：不门控时高斯在直走偏差处每步白拿约 1.0 分（实测 125 步 `lateral_vel=+139.73`）；
横向/偏航两项由 `max` 改**求和**（取 max 会丢掉偏航梯度）；
形状从高斯→二次型→**线性死区**（高斯只要求 |v-cmd|<2σ 不要求相关；二次型每单位动作仅约 1.35 分，
策略停在原地；线性梯度 `-W/range` 为常数且大误差不塌缩）。

---

## 4. 动作管线（唯一映射，无 absolute 分支）

```python
target = np.clip(home_angles + self._gait_action() + cfg.action_scale * action, self.act_low, self.act_high)
# _gait_action(): zeros(12) if (not gait_prior or gait_gain == 0.0) else gait_gain * gait.radians(self._gait_phase)
# 力矩：tau = _kp*(target - q) - _kd*qd ; data.ctrl[:] = clip(tau, meta.ctrl_ranges[:,0], meta.ctrl_ranges[:,1]) ; mj_step
```
- **两个尺度必须分开**：`cfg.action_scale=0.30` rad 是**策略权限**（残差），`gait.action_scale=0.90` 是**先验幅度**；
  共用会让探索噪声与先验同乘一个系数，实测 2 s 摔倒。
- `action` 先被 `clip(-1,1)`，**再**按 `action_scale` 缩放；最终 clip 到 `act_low/act_high`（关节软限位）而**不是** ±1。
- `gait_prior=false` 时 `_gait_action()` 返回 `zeros(12)`，与 `GaitParams.static()` 是同一条代码路径。
- 先验**只能在构造时定**（`set_gait_prior()` 已删）：换先验就改 `cfg.gait/gait_gain` 后重建环境。
- `kp=(30,30,45)`、`kd=(0.6,0.6,0.9)`（hip,thigh,calf），`np.tile(...,4)` ⇒ 12 维；物理 500 Hz / 控制 50 Hz。

---

## 5. 配置层（`robotdog/config.py`，411 行）

```python
DEFAULT_CONFIG_PATH = paths.WORKSPACE / "config.yaml"
DEFAULT_STAGE_TIMESTEPS = [100_000] + [250_000] * (len(STAGE_HEIGHTS_CM) - 1) ; DEFAULT_LR = 1.5e-4
def load_config(path: str | Path | None = None) -> dict[str, Any]
def config_to_env(raw: Mapping) -> EnvConfig
def config_to_stages(raw) -> list[StageSpec]
def config_to_ppo(raw) ; def config_to_run(raw) ; def config_to_play(raw) ; def n_envs(raw) ; def device(raw)
```
- `load_config`：文件不存在 `raise SystemExit(f"找不到配置文件：{target}")`；YAML 错 → `SystemExit`；顶层非映射 → `SystemExit`。
- 缺省值只写一遍：先 `EnvConfig()` 再只对 YAML 出现的键 setattr（dataclass 是唯一默认值来源）。
- **未知键报错的实现**：
```python
valid = set(getattr(target, "__dataclass_fields__", {}))
unknown = sorted(k for k in values if k not in valid)
if unknown: raise SystemExit(f"config.yaml 的 {where}: 无法识别的键 {unknown}；可用键 = {sorted(valid)}")
```
  奖励段另有 `_validate_reward_keys`（以 `RewardConfig.__dataclass_fields__` 为唯一真值）。
  理由：手写 YAML 里 `n_envs` 写成 `num_envs` 语法上合法，静默忽略＝"你以为调了其实没调"。
- `config_to_env`：pop 掉 `obs`/`command`/`n_envs`/`gait_file` 单独处理；`env.command` 写入后调
  `cfg.command.__post_init__()` 同步 vx（少了它 `_sample_command()` 抽到过期 vx）；
  `if cfg.gait_prior: cfg.gait = _load_gait(gait_file)`（相对**仓根**解析）。
- `config_to_stages`：`tread`/`n_up` 默认 `DEFAULT_TREAD=0.50`/`DEFAULT_N_UP=8`；`heights_cm` 必须非空、非负、**升序**；
  `_timesteps`（标量或列表；长度不等按 `_spread` 线性插值保持总量，余数补给最难一级）；
  `_floats`（插值）、`_bools`（**不插值**，长度必须等于级数或 1）；`_reward_override_table`（标量=每级，列表须等长）；
  名去重 `_unique_name(base, seen)` → `flat` / `flat_2` / `flat_3`；`h<=0` → `flat_spec(n_up=0,n_down=0,...)`，
  否则 `demo_stairs_spec(step_height=h/100, tread, n_up)`。
- **`config.yaml` 六段各管什么**：`run`（name/seed/save_every/log_every）、
  `curriculum`（heights_cm/tread/n_up/timesteps/learning_rate/command_vy_max/command_yaw_max/command_vx/
  flat_success_distance/reward_overrides）、`env`（n_envs/episode_length_s/action_scale/command/terminate_*/gait_prior/
  gait_file/obs.*）、`reward`（RewardConfig 全权重）、`ppo`（net_arch/n_steps/batch_size/n_epochs/lr/gamma/gae_lambda/
  clip_range/ent_coef/vf_coef/max_grad_norm/target_kl/log_std_init/norm_reward/device）、
  `play`（`episodes_gif` / `episodes_eval` / `seed` / `gif_max_frames` / `stages` / `batch`；相机与 GIF 编码参数是 `scripts/play.py` 的模块常量，不在 yaml 里）。

---

## 6. 未能确认 / 存疑点

1. 起草本文件时源清单未含 `agent.md`、`experience.md`、`train.md`、`README.md`；**已补齐**：
   观测信息来源清单见 `06-training-eval.md` 与本文件 §2，经验类结论见 `07-pitfalls.md`，
   文档纪律见 `00-workflow.md` / `08-verification.md`。
2. `info` 字典 = **22 键**（`robotdog/env.py:970-993`，逐键：`reward_terms`、`trunk_contact`、
   `leg_contact`、`leg_contact_streak`、`blocked`、`probe_count`、`base_x/y/z`、`terrain_height`、
   `forward_vel`、`lateral_vel`、`yaw_rate`、`command`、`roll`、`pitch`、`crest_height`、
   `reach_top_x`、`reached_top`、`is_success`、`fallen`、`gait_prior`）；其中 `reward_terms` 是 **14 键**
   （`reward.py:375` 返回 13 键，`env.py:595` 注入 `terminal`），故恒等式
   `reward == sum(info["reward_terms"].values())` 成立（`env.py:596`）。
   过期描述可删：`docs/robotdog/env.md` §9 的「18 键」、§4.4 的 17 键、`README.md` 的「20 项」
   （后者还漏了 `blocked` / `probe_count`）。
3. `RewardConfig` 的文档默认值确实过期（`forward_vel` 文档 2.5 / 代码与 config 10.0，
   `lateral_vel` 文档 1.5 / 代码与 config 6.0，`yaw_rate` 1.0 / 4.0），
   过期处：`README.md:201`、`README.md:236-244`、`docs/robotdog/reward.md:115-124`、
   `docs/robotdog/reward.md:164-170`、`docs/robotdog/reward.md:246-249`、`docs/robotdog/reward.md:295-296`；
   形状也已从高斯核改为**按量程归一化的线性死区**、`command_follow` 从 `max` 改为**求和**
   （`robotdog/reward.py:49-71`、`:174`、`:220`）。**以代码 + `config.yaml` 为准**。
4. `probe_lift_ratio` / `probe_lift_max` 在代码中标记为保留字段，未确认是否真的完全无读者。
5. 未运行任何脚本验证（无训练/评估实测数字），本文件全部为静态代码+文档蒸馏。
