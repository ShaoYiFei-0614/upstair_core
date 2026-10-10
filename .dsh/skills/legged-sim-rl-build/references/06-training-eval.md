# 课程驱动 · PPO 组装 · 产物三件套 · 评估与 GIF

> 对应 `SKILL.md` 阶段 8–9。素材取自 upstair_core（Unitree Go2 盲走楼梯 · MuJoCo 3.12 + gymnasium 1.3 + SB3 2.9） 的真实源码与实测记录。
> 具体数值只作**取证**用；换机器人时按文中的「换机器人要改什么」逐条替换。

参考实现的默认预算：六级合计 1,350,000 步（平地 100k + 5×250k），每级学习率递减 1.5e-4→6e-5。

## 1. 课程驱动

`robotdog/curriculum.py`（332 行，**纯数据 + 报表**，不 import SB3/env/training）：
```python
STAGE_HEIGHTS_CM = (0.0, 0.0, 0.0, 3.0, 6.0, 9.0) ; DEFAULT_TREAD = 0.50 ; DEFAULT_N_UP = 8
@dataclass StageSpec: name, step_height, spec: StairSpec, timesteps, learning_rate, label="", description="",
    command_vy_max=0.0, command_yaw_max=0.0, command_vx=0.0, flat_success_distance=0.0,
    flat_success_needs_progress=True, reward_overrides: dict[str,float]={}   # property is_flat = (n_up==0 and n_down==0)
@dataclass EpisodeStats: window=50, n_total=0 ; add(*, success, fallen, ret, length) ; add_info(info)->bool ; add_infos(infos)->int
@dataclass StageResult: name, label, step_height, timesteps, mean_return, success_rate, fall_rate, wall_seconds,
    model_path, mean_length, episodes, obs_size, extras={}
def format_curriculum(stages, results=None) -> str
def write_summary(path, stages, results=None, meta=None) -> Path   # meta 平铺到顶层，indent=2/ensure_ascii=False
```
课程**串行**：每级一种台阶几何、级内无随机化；训满预算存权重，下一级 `PPO.set_parameters` warm-start。
`scripts/train.py` 阶段循环（`Trainer.make_env` → `train_stage`，见 §7）；`warm: str | None = None`，
逐级 `venv = trainer.make_env(stage, warm_start=warm)`、`try: train_stage(...) finally: venv.close()`、
`warm = str(rundir.checkpoints / f"{stage.name}_final.zip")`。
**产物三件套**（末级 `shutil.copy2`）：`model_final.zip` + `model_final_vecnormalize.pkl` + `model_final_env.json`；
另有 `checkpoints/<阶段>_final{.zip,_vecnormalize.pkl,_env.json}`、`tensorboard/`、`summary.json`。
`summary.json` = `{"env_config","obs_config","reward_config","final_model","wall_seconds","stages","results","total_timesteps"}`。
入口保护：`if rundir.root.exists(): raise SystemExit("输出目录已存在：…")`（**RunDir 构造时不 mkdir**，否则该保护恒真）。
本级默认课程表（`config.yaml`，总预算 1,350,000 步）：heights `[0,0,0,3,6,9]` cm、
timesteps `[100k,250k,250k,250k,250k,250k]`、lr `[1.5e-4,1.5e-4,1.2e-4,1.0e-4,8e-5,6e-5]`、
command_vx `[0.45,0.45,0.45,0,0,0]`、vy_max `[0,0.15,0.35,0.30,0.35,0.40]`、yaw_max `[0,0.15,0.50,0.35,0.55,0.70]`、
flat_success_distance `[2.0,2.0,2.5,0,0,0]`、reward_overrides `{foot_lift:[0.9,0.9,0.9,1.0,1.2,1.4],
foot_lift_step_ratio:[1.4,1.4,1.5,1.6,1.8,2.0], probe_lift:[0,0,0.4,0.6,0.9,1.2]}`。
前三级故意全留平地（指令跟踪与爬楼是两种能力，混学跟踪会被爬楼压力淹没）。

---

## 2. 训练层（`robotdog/training.py`，323 行）

```python
DEFAULT_NET_ARCH = (512, 256)
def make_env_fn(stair, cfg=None, seed=0) -> Callable[[], Go2StairsEnv]
def make_vec_env(stair, n_envs=8, *, cfg=None, seed=0, normalize=True, monitor=True, norm_reward=True) -> VecEnv
def build_ppo(venv, *, seed=0, n_steps=256, batch_size=512, n_epochs=5, learning_rate=1.5e-4, gamma=0.99,
              gae_lambda=0.95, clip_range=0.2, ent_coef=0.002, vf_coef=0.5, max_grad_norm=1.0, target_kl=0.03,
              log_std_init=-2.0, net_arch=DEFAULT_NET_ARCH, tensorboard_log=None, device="cpu", verbose=1) -> PPO
def load_policy(model_path, *, cfg=None, stair=None, device="cpu", norm_reward=False) -> tuple[PPO, ObsNorm | None]
class RunDir: __init__(self, name, root=None)  # 不 mkdir；ensure() 建 root/checkpoints/tensorboard/artifacts
    save_model(self, model, base, venv=None) -> tuple[Path, Path|None]   # base 必须**不带后缀**
def save_vecnormalize(model, base) ; def restore_vecnormalize(venv, path)
```
- 链路顺序：`DummyVecEnv(8)` → **`VecMonitor`（里层）** → `VecNormalize(training=True, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0)`
  —— VecMonitor 必须在里层，否则 `info["episode"]["r"]` 是归一化后回报。
- `DummyVecEnv` 是本沙箱**唯一**可行并行：禁 Windows 命名管道，`SubprocVecEnv` 抛 `PermissionError: [WinError 5]`；
  `mj_step` 不释放 GIL，线程池无收益；实测 250~400 向量步/秒。并发 ≥3 个训练进程会 `_ArrayMemoryError`。
- `PPO("MlpPolicy", venv, ..., policy_kwargs=dict(net_arch=dict(pi=list(net_arch), vf=list(net_arch)),
  activation_fn=nn.Tanh, log_std_init=log_std_init))`；压掉 `.*mini-batch size.*` UserWarning。
- `load_policy` 三步（缺一不可）：`arch = policy_arch(model_path) or DEFAULT_NET_ARCH` → 建 1 环境 DummyVecEnv →
  `PPO` + `model.set_parameters(str(model_path))`；统计量来自同目录 `<stem>_vecnormalize.pkl`，
  **同时装到两处**：返回值 `ObsNorm`（给驱动裸环境的 play）+ `restore_vecnormalize(venv, cand)`（给走 `model.get_env()` 的调用方）。
  **为什么必须走它**：`model.set_parameters()` 只还原网络权重；不装统计量，策略拿到 mean=0/var=1 的输入
  而训练时是真统计 ⇒ 确定性评估 100%→0%，甚至 `model.get_env()` 与裸 `Go2StairsEnv` 结论相反
  （3 cm 上前者 0% 摔倒、后者 100% 成功）。网络结构必须用训练时那一套，否则
  `RuntimeError: ... size mismatch for mlp_extractor.policy_net.0.weight`。
- 关键超参理由：`log_std_init=-2.0`（归一化动作 std≈0.135；默认 0.0 即 std=1.0 等于纯噪声；过小退化成纯开环步态）；
  `norm_obs=True` 硬性（关节速度 ±20 vs 关节角 ±3，差三个数量级，critic 拟合不动）；
  `norm_reward` 默认 True（曾因课程地形切换奖励尺度漂移而关，结果回合回报近 -6000、explained_variance≈0.01；
  唯一例外：warm-start 未归一化权重时关掉）；`n_steps 256 × n_envs 8 = 2048`/次更新、`batch_size 512`（4 个 minibatch）；
  `target_kl 0.03`（触发 `Early stopping at step k/N due to reaching max kl` 正常）；
  诊断量 `mean|a|` 健康区间 0.3~0.45（参考 0.42；<0.1 = 退化成纯先验）。
- 换地形是**整环境重建**（每级新建整套链 + `venv.close()`）；`Go2StairsEnv.set_stair()` 零调用点。

---

## 3. 评估 / 演示

`scripts/play.py`（665 行）：`--mode gif`（**按课程阶段逐个**连跑 `play.episodes_gif` 回合，逐帧渲染 + HUD → `runs/<运行名>/gifs/<批次>/<阶段>.gif`，平地级与台阶级都渲染）/ `--mode eval`（`play.episodes_eval`，统计成功率/摔倒率/超时率/平均前进）。
**CLI 只有 `--run` / `--mode` / `--config` / `--stage` / `--batch` / `--gifs-dir`**（没有 `--episodes` / `--seed` / `--gif` 这类开关，渲染参数全在 `config.yaml` 的 `play` 段）。
`resolve_target(run)` 接受运行名/目录/`.zip`；`_resolve_norm` 优先 `<stem>_vecnormalize.pkl` 再 `with_suffix(".pkl")`。
`Runner.predict(obs)`：`x = obs if norm is None else norm.normalize(obs)` → `model.predict(x, deterministic=True)`。
`open_runner` 保留 `model.env`（向量环境必须保活）。`FRAME_STRIDE=2`（50 Hz 每 2 步一帧 → 25 fps）。
**地形来源**：优先 `play.stage_height_cm`；否则 `step_height_from_model(model_path)`（读权重自带
`<stem>_env.json` 的 `report.step_height`）并打印"地形来自权重自带记录"；否则 `stair_from_height(None)`
（配置默认 3 cm）。理由：每个 checkpoint 的地形都不一样，只靠最后一级会让中间 checkpoint 全用错地形。
`run_episode` 里**必须在 `reset()` 之后**再 `env.set_command_now(vx=..., vy=..., yaw_rate=...)` 并重取
`obs = env._observation()` —— reset 会重抽指令，不钉死则每次回放指令都不同。指令由调用方逐阶段传入：
`robotdog.config.command_for_stage(stage)`（取该级量程的 0.8 倍，量程为 0 的方向就是 0）。
HUD 的第 3 行打这条指令与回合末的滤波实测值 —— 没有它，看 GIF 的人分不清「它站着不动」和
「指令本来就是 0」。`play.stages` 决定渲染哪些阶段（`auto` = 优先读运行目录 `summary.json` 记的
训练阶段，没有才退回「有 `<阶段>_final.zip` 的那几级」，再退回当前课程全部级）；`--stage` 只渲染一个。
产物落 `runs/<run>/gifs/<批次>/<阶段>/<阶段>.gif`；相机/分辨率/帧率/缩放是 `scripts/play.py` 的
模块常量（`CAMERA_GIF` / `FRAME_WIDTH` / `FRAME_HEIGHT` / `GIF_FPS` / `GIF_SCALE`），不在 yaml 里。
GIF 调色板：先用 `np.linspace` 采 ≤16 帧拼 mosaic → `quantize(256, MEDIANCUT)` 得**全片共用** palette →
逐帧 `quantize(palette=...)`（逐帧各自 quantize 会让同一只狗换颜色）；`save_all=True, duration=max(20,1000/fps), loop=0, disposal=2`。
评估口径提醒："评估至少跑 40 回合，10 回合能给出 ±15 个百分点的错觉"。

`scripts/teleop.py`（288 行）—— 外部控制器接口：
- `env.set_command_now(vy=, yaw_rate=, vx=)`：**每个控制步**改写当前指令，控制器高频入口（本脚本每步调一次）；
  传 `vx` 时同时写 `cfg.command.vx`（否则 reset 重抽会丢掉课程下发的定值），并刷新 `_hist[-1]` 消除 20 ms 滞后。
- `env.set_commands(vy_max=, yaw_max=)`：改**采样范围**，课程在阶段边界调（`venv.env_method`）。
- 控制器可读地形与位姿（遥感操控，比策略多信息**不违反盲走** —— 盲走只约束策略观测）。
  `StairController`：`x < start_x - align_dist` 直走；`x > up_end_x` 收住站稳；
  否则 `desired_yaw = -0.5*arctan2(y-center_y, 1.0)`、`vx *= clip(cos(yaw_err),0,1)`、
  `vy = clip(-0.9*(y-center_y), ±vy_limit)`、`wz = clip(1.6*yaw_err, ±yaw_limit)`。
  `HoldController`：钉死一条 `[vx,vy,wz]` 跑满回合，量"跟踪指定指令"的能力。
  CLI：`--run --config --episodes --seed --height --flat --gif --hold VX VY WZ`。
  权重指令范围全 0 时临时放开到 `vy±0.25 / wz±0.35` 并明确提示"策略未学过横移"。

---

## 4. 可迁移清单

**通用骨架（换机器人/换任务**不改**）**：独立奖励纯函数 + `terms` 账目恒等式；`config.yaml` 单一超参入口 +
dataclass 唯一默认值来源 + 未知键报错；`_unique_name` 阶段名去重；`EpisodeStats` 滑动窗口报表；
`RunDir` 三件套 + `summary.json` + 每 checkpoint 一份 `_env.json`；`ObsNorm` 拷贝（`del raw`）；
`policy_arch()` 读权重自带结构；VecMonitor 里层 + VecNormalize 外层 + 成对加载统计量；
`write_summary(meta 平铺)`；GIF 共用调色板与 HUD；`setup_utf8_console()` 显式调用。

**换机器人必改**：`robotdog/go2_model.py`（MJCF 组装、`Go2Meta` 提取：`joint_ranges/home_joint_angles/
qpos_joint_adr/qvel_joint_adr/foot_geom_ids/foot_radius/ctrl_ranges/trunk_geom_ids/leg_geom_ids/gravity/friction`）；
`EnvConfig.kp/kd`（12 维 → 新关节数）；`num_act`、`action_scale`；`robotdog/gait.py` 的 `GaitParams`（腿数与相位）；
基础帧 42 = 12+12+3+3+12 的构成；`nominal_pose`（0.27）；`terminate_min_height`（须 > 躯干碰撞体最低点）。
**换任务必改**：`robotdog/tools/terrain.py`（`StairSpec/heights/flat_spec/demo_stairs_spec/TERRAIN_GEOM_GROUP`）；
`_success_reached()` 与 `_success_target_x()`；`curriculum.heights_cm` 与 `STAGE_HEIGHTS_CM`；
`step_height` 观测块的语义；`probe_*` / `foot_lift_step_ratio` 等与地形强相关的权重。
**必改常量**：`OBS base_frame_size=42`、`history_len`、`tread=0.50`（改动会让成功率掉一个量级）、
`decimation=10`、`episode_length_s=20.0`、`DEFAULT_NET_ARCH=(512,256)`、`clip_obs=10.0`、leg_phase 四相。

---
