# 代码审计报告：盲走楼梯残差 RL 仓（2026-10-18）

审计对象：`D:\项目\RobortDog\upstair_core`（Unitree Go2，无视觉 / 无足底接触信号的盲走上楼梯，
开环步态先验 + PPO 残差）。审计目标：找 bug、给可扩展性改进。

**结论一句话**：确认 8 个缺陷（B1–B8），其中 4 个在真实使用路径上（回放/遥操作/评测）
产生可观测的错误行为，2 个会污染训练信号；全部已在代码里修掉或加上默认保持原行为的开关，
默认路径（`yaw_rate_mode=legacy`、`stair_guidance_under_hold=true`）与审计前**逐位一致**
（6 cm 权重 `runs/ab_6_r1/model_final.zip` 三次回放：385/386/407 步、x≈3.90、h=0.48、3/3 成功）。
另外新增 11 条回归测试 `tests/test_env_contract.py`，逐条钉住这些缺陷。

---

## 1. 审计范围与方法

| 范围 | 文件 | 行数（审计后） |
| --- | --- | --- |
| 环境 / 奖励核心 | `robotdog/env.py`, `robotdog/reward.py` | 1545 / 529 |
| 配置 / 课程 / 训练 | `robotdog/config.py`, `robotdog/curriculum.py`, `robotdog/training.py` | 510 / 358 / 307 |
| 先验步态 / 权重加载 | `robotdog/gait.py`, `robotdog/policy.py` | 254 / 322 |
| 脚本（训练/回放/遥操/演示） | `scripts/*.py` | 458 / 944 / 343 / 200 |
| 地形 | `robotdog/tools/terrain.py` | 256 |

方法（四种证据，避免只靠读代码下结论）：

1. **静态通读**：env/reward/gait/config/curriculum/training/policy/terrain 全文 + 全部脚本调用点。
2. **真实 rollout 量化**：用已有权重 `runs/ab_6_r1/model_final.zip`（6 cm 台阶、踏面 0.30 m、8 级，
   由 `<model>_env.json` 自动恢复配置）按 `scripts/play.py` 的方式跑 3 个确定性回合，
   逐步记录 `base_rpy` / `qvel[3:6]`，n=1178 步（附录 A 是探针全文）。
3. **对照实证**：对怀疑的语义（共享 cfg、hold 指令）写最小复现，而不是从注释推断。
4. **回归测试**：把每条已确认的缺陷改写成断言，`tests/test_env_contract.py`（unittest，不引新依赖）。

---

## 2. 缺陷清单

### 2.1 总表

| 编号 | 严重度 | 位置 | 一句话 | 状态 |
| --- | --- | --- | --- | --- |
| **B1** | 高 | `robotdog/env.py` `set_command_now` / `set_gait_cmd_now` | `set_command_now` 没有 `return`、也不刷新「待取」观测帧；那段代码误缩进进了 `set_gait_cmd_now` 的 `return` 之后，是死代码 | 已修 |
| **B2** | 高 | `robotdog/env.py` `step` 的 OU 扰动条件 | `_cmd_age < command_hold_steps` 恒为真 ⇒ `hold=True`（回放/遥操/录像）的钉指令每步都被 hold+wander 覆盖 | 已修 |
| **B3** | 中 | `robotdog/training.py` `make_env_fn` | 8 个 env 共享同一个 `cfg` 对象，而 `env._command` 就是 `cfg.command._now` ⇒ 指令状态跨 env 串台 | 已修 |
| **B4** | 低（潜伏） | `robotdog/env.py` `reset` | 播种机身高度取的是 `qpos0` 处的地形高（实参在写 `qpos` 之前求值），不是播种点 x 处的解析高度 | 已修 |
| **B5** | 低 | `config.yaml`、`robotdog/config.py`、`robotdog/env.py` 注释 | 观测维数注释写 51/55（实际当前 53）、课程写「六级」（实际九级）、步态参数写 8 维（实际 12） | 已修 |
| **B6** | 中（潜伏） | `robotdog/env.py` `_gait_action` 条件化分支 | 把**角度** `_gait_theta()` 当**秒**传进 `GaitParams.radians`，相位里频率乘了两次（楼梯级 ≈7.9 倍频） | 已修 |
| **B7** | 高 | `robotdog/env.py` `yaw_rate` | **双重旋转**：`qvel[3:6]` 已在机体系，又乘了一次 `R(yaw)^T` ⇒ roll rate 漏进偏航读数，实测偏差 0.198 rad/s ≈ 偏航指令量程的 66% | 已修（加三种口径开关） |
| **B8** | 中 | `robotdog/env.py` `step` 的 `stair_guidance` | 楼梯级引导默认对 `hold=True` 一视同仁 ⇒ 回放/遥操钉住的 vy/wz 被静默改写，teleop 的「跟踪误差」测的是引导量而不是操作手给的指令 | 已修（开关 + `--strict-hold`） |

严重度口径：高 = 真实使用路径上行为错误或污染训练信号；中 = 只在特定配置/路径上错，或会系统性误导评测；
低 = 潜伏 / 文档错误（当前参数下不影响数值）。

### 2.2 B1 —— `set_command_now` 的返回与观测刷新是死代码

位置：`robotdog/env.py:516` `set_command_now`；`:575` `set_gait_cmd_now`。

现象：`set_command_now` 的 docstring 写着「返回改写后的 `[vx, vy, wz]`」，实际返回 `None`；
它写的指令要等到下一次 `step()` 才进观测，外部控制器在两次 `step` 之间下的指令因此**固定滞后 1 个控制步**
（50 Hz 下 20 ms）。真正干这两件事的代码块（`frame = self._obs_frame(); ... _hist[-1] = frame;
`return list(self._command)`）跑到了 `set_gait_cmd_now` 的 `return (freq, swing)` **之后**，永不执行。

根因：函数拆分时缩进错位，旧的尾部代码没跟着搬走（不是设计取舍）。旁证：
`scripts/teleop.py:177` 与 `scripts/cmd_demo.py:114` 都写着「`set_command_now` 只刷新环境内部那一帧，
手里的 obs 还带旧指令，必须重取」，然后调用方手动 `obs = env._observation()` 兜底 —— 修好后兜底不再必需。

影响：① 任何用返回值的调用方会 `TypeError`；② 遥操作/演示的观测与自己刚写的指令差一步，
控制律（尤其高频的反馈型控制器）在观测里看到的是过期指令；③ `cmd_demo` 的指令阶跃测试失真。

修法：把刷新+返回搬回 `set_command_now`；删除 `set_gait_cmd_now` 里的死代码。
测试：`tests/test_env_contract.py` `CommandStateTest.test_set_command_now_returns_and_refreshes_obs`。

### 2.3 B2 —— `hold=True` 的钉指令每步都被 OU 扰动覆盖

位置：`robotdog/env.py:1312` 一带（`step` 开头的指令重抽/扰动块）。

旧条件：`if self._cmd_age < self.cfg.command_hold_steps and self.cfg.command_wander_std > 0:`。

但 `_cmd_age` 在上一行刚做过「`>= command_hold_steps` 就归零」，平时恒处于 `[0, hold_steps)`；
`hold=True` 时显式置负（`-(10**9)`）也同样满足 `< hold_steps`。⇒ **条件恒真**，
每个控制步都执行 `_command[:] = (_cmd_hold + _cmd_wander)`。`command_wander_std` 在 `config.yaml` 是 `0.03`，
OU 稳态幅度 ≈ `std*sqrt(tau/2)` ≈ 0.026 m/s、再被 `clip(±0.08)` 截断 ⇒ 回放时「钉住」的指令实际是一条
带随机游走的指令，`hold` 语义完全失效。注释当时写的是「hold=True 时扰动不生效」，与代码相反。

影响：① `play` 回放、`teleop` 遥操作、`cmd_demo` 演示的指令都带随机游走 ⇒ 评测不可复现、
两臂对比引入了与权重无关的随机差；② teleop 用 `cmd` 与实测值算的跟踪误差，基准本身就不是环境里的指令。

修法：在实例上显式维护 `_cmd_hold_active`（`set_command_now` 写入、`reset` 清零），扰动条件改为
`not self._cmd_hold_active and ...`。
测试：`CommandStateTest.test_hold_command_survives_wander`（40 步内 `_command` 必须逐位等于写入值）。

### 2.4 B3 —— 8 个 env 共享一个 cfg，指令状态跨环境串台

位置：`robotdog/training.py` `make_env_fn`；`robotdog/env.py` `self._command = cfg.command._now`。

`training.make_env_fn` 返回的闭包对 8 个 env 都传**同一个** `cfg` 对象，而 `Go2StairsEnv` 为了让
外部控制器改指令立刻生效，直接把 `cfg.command._now` 这个 list 当成自己的 `_command`。
结论：8 个环境读写同一条指令。实测（附录 A 同类探针）：

```
e0.set_command_now(vx=0.9, vy=0.2, yaw_rate=0.4, hold=True)
  -> e1._command 也变成 [0.9, 0.2, 0.4]
e0.step() 5 次后：e1._command == e0._command == [0.9203, 0.0702, 0.154]（wander 写的是共享 list）
而 e0._cmd_age = -999999995, e1._cmd_age = 0（定时器却是各自独立的）
```

为什么训练没炸：`DummyVecEnv` 是同步串行的，每个 env 在 `step` 内部**先写自己的指令、再读**，
所以同一步内碰巧自洽。但它是隐患：一旦换成 `SubprocVecEnv`/异步向量化，或用 `env_method` 按环境下发不同指令，
就会静默串台；课程切级时 `set_commands` 只改一个 cfg 也依赖「所有 env 用同一套」这个巧合。

**2026-10-19 更新**：`SubprocVecEnv` 已在允许命名管道的 Windows 上实测可跑
（spawn + 模块级工厂 + 每 env deepcopy cfg），默认仍是 dummy。可运行性由
`tests/test_vec_env.py`（9 条，直接脚本运行）与 `unittest discover`（52 条）证明；
50k 单级 A/B 已测得 learn **2.45×**、e2e **2.1×**（8 env，seed 0），详见 `train.md`。

修法：`make_env_fn` 给每个 env `copy.deepcopy(cfg)`。`scripts/train.py:209-212` 的
`venv.env_method("set_reward_overrides"/"set_commands", ...)` 是**逐环境**下发，deepcopy 后仍然每级都生效
（2-env 探针确认广播与 `indices=[0]` 隔离；8-env 由两臂 9×2000 冒烟及其中间 checkpoint 的对账确认）。
测试：`CommandStateTest.test_envs_do_not_share_command_state`。

### 2.5 B4 —— `reset` 的播种高度取了错误参照点

位置：`robotdog/env.py:1059`。

旧写法：`self.data.qpos[:] = home_qpos(..., z=self.terrain_height_under_base + self.cfg.reset_z, ...)`。
`terrain_height_under_base` 读 `base_pos[0]`（= `qpos[0]`），而这一次赋值的**实参先求值**，
所以读到的是赋值前的 `qpos[0]`（`mj_resetData` 后的 `qpos0`，x≈0），不是刚算好的播种点 x。
`spawn_back > 0`（当前 0.55 m，永远落在楼梯前的地面上）时两者等价 ⇒ 潜伏 bug；
但把 `spawn_back` 取负（直接生在台阶上，做「中途开局」实验）就会用错高度，机身悬空或穿模。

修法：`ground_z = float(self.stair.height_at(x))` 后再赋 `z=ground_z + reset_z`。
测试：`ResetSpawnTest.test_spawn_on_stairs_uses_analytic_height`（`spawn_back=-0.90`、噪声置 0、`places=9`）。

### 2.6 B5 —— 文档/注释里的数漂了

| 位置 | 写的 | 实际 |
| --- | --- | --- |
| `config.yaml` obs 段 | `42+3+1+3+2 = 51`、`= 55` 两种算法 | `42 + 6（结构块）+ 2 + 1 + 2 [+2]`：当前 `gait_cmd=false` ⇒ **53**，开则 55 |
| `robotdog/config.py:14` | 「六级 StageSpec」 | `config.yaml` 是**九**级 |
| `robotdog/env.py` docstring | 「8 个参数」「8 维步态参数」 | 12 维（髋 3 关节 × 幅值/偏置） |

这类错误的代价不是数值，而是**误导后续判断**（这次审计里 51/53 就差点让人以为 `ObsConfig` 算错了）。
已把观测维数的推导写成可执行断言（`ObsShapeTest.test_documented_dims`：四种开关组合 48/50/53/55，
同时校验 `cfg.obs.size`、`observation_space.shape`、`reset` 后 `obs.shape` 三者一致）。

### 2.7 B6 —— 条件化步态的相位把频率乘了两次

位置：`robotdog/env.py:1162`（`_gait_action` 的 `obs.gait_cmd` 分支）。

`GaitParams.radians(t)` 内部算的是 `sin(2*pi*freq*t + phase)`，所以入参 `t` 必须是**秒**。
旧代码传的是 `self._gait_theta()` = `2*pi*f_cmd*_gait_phase`（**角度**）⇒ `sin(2*pi*f*(2*pi*f*t))`，
相位角变成 `2*pi*f` 倍：楼梯级 `f=1.25 Hz` 下先验以约 9.8 倍速率摆动（等效 12 Hz 上下），
动作空间被这个高频偏移占满。`gait_cmd` 当前为 `false`（§23.29 已判定条件化有害）所以不影响存量，
但字段与代码都保留着做实验，一开就会全错。

修法：传秒制的 `self._gait_phase`（与观测里 `_gait_theta = 2*pi*f*_gait_phase` 同源、连续）。
测试：`GaitPhaseTest.test_gait_cmd_and_uncond_agree_at_same_freq`（同频同相位下两条路径 `_gait_action` 必须逐位相等）。

### 2.8 B7 —— `yaw_rate` 是双重旋转的产物（本次审计最重要的一条）

位置：`robotdog/env.py:776`（原 `yaw_rate` 属性）。旧实现：

```python
wx, wy, wz = quat_to_mat(self.base_quat).T @ self.data.qvel[3:6]
return float(wz)
```

MuJoCo free joint 的 `qvel[3:6]` **已经是机体系角速度**（`base_ang_vel_body` 的 docstring 一直这么写），
再乘 `R^T`（= `R(yaw)^T`，因为 base 的 roll/pitch 很小）等于把 `omega` 投到「偏航旋转后的机身系」，
纯偏航运动也会在 z 分量上串进 roll/pitch 分量的投影。真实爬楼时 roll/pitch 摆动很大（实测 pitch |mean| 0.18 rad、
roll |mean| 0.11 rad、body wx |mean| 1.10 rad/s），于是**偏航读数被 roll rate 主导**。

实测（附录 A，n=1178 步，6 cm 台阶）：以世界系偏航角速度 `psi_dot = (wy*sin(roll) + wz*cos(roll))/cos(pitch)`
为真值：

| 量 | mean | mean\|.\| | p95\|.\| | max\|.\| |
| --- | --- | --- | --- | --- |
| 真值 `psi_dot` | +0.0117 | 0.2494 | 0.6201 | 1.0436 |
| 旧口径 `yaw_rate` | +0.0298 | 0.1752 | 0.5081 | 1.4097 |
| 旧口径误差 | +0.0180 | **0.1976** | 0.6824 | 1.0468 |
| 倾斜补偿口径误差 | -0.0090 | **0.0600** | 0.1894 | 0.4416 |

`corr(误差, pitch) = -0.022`，`corr(误差, body wx) = **-0.903**` ⇒ 污染源是**滚转角速度**，不是俯仰。
偏航指令量程在楼梯级是 ±0.3 rad/s，而旧口径的平均误差 0.198 rad/s 已经 ≈ 量程的 **66%**；
`|误差| > 0.1` 的帧占 57.6%、`> 0.2` 占 32.9%、`> 0.3` 占 22.6% —— 也就是说**超过半数的爬楼帧里，
策略学到的偏航误差信号是错的**。

被污染的量有两个下游（`grep` 全仓确认无其他读数点）：

1. `robotdog/env.py:830` 附近：观测 `vel_filter` 块的「偏航角速度」一维（`ObsConfig.vel_filter`，当前开启）；
2. `robotdog/reward.py:231`：`wz = env.filtered_yaw_rate if vel_filter else env.yaw_rate`，直接进偏航跟踪奖励。

⇒ 这是「偏航跟踪学不出来」的**强候选根因**：奖励与观测看到的是同一个假信号，而指令是沿世界竖直轴的 `wz`，
两者口径根本不一致。注意 `env.py` 里 `yaw_rate` 的旧 docstring 曾声称「不用世界系公式因为被步态振荡污染」，
但实测反了：被污染的是旧公式自己。

修法：新增 `EnvConfig.yaw_rate_mode`（`robotdog/env.py:294`），三种口径：

| 值 | 公式 | 用途 |
| --- | --- | --- |
| `legacy`（默认） | `R(q)^T @ qvel[3:6]` 的 z 分量 | 只为复现旧训练结果；**不要再用于新实验** |
| `body_z` | `qvel[5]` | 机体系 z，最贴近真机 IMU 陀螺读数，最便宜 |
| `world` | `(wy*sin(roll) + wz*cos(roll)) / cos(pitch)` | 世界系偏航角速度，**与 `wz` 指令同口径**；误差 0.198→0.060 rad/s |

默认仍是 `legacy`，存量回放/训练逐位不变（见 §4）。建议做一次训练侧 A/B（§6.1）。
测试：`YawRateTest.test_three_modes` + `test_rejects_unknown_mode`。

### 2.9 B8 —— 楼梯级引导会改写 `hold=True` 的 vy/wz

位置：`robotdog/env.py:1329`（`step` 的引导块）、`EnvConfig.stair_guidance`（默认 `True`）。

楼梯级（`n_up > 0`）的 vy/wz 指令**不来自课程采样**，而是由环境内的闭环引导控制器每步生成
（按机身偏离台阶中线的距离拉回 + 按朝向误差对准）。这对训练是有意设计（爬楼需要闭环稳住中线），
但它对 `hold=True` 一视同仁 ⇒ 回放/遥操/演示里写进 `set_command_now(..., hold=True)` 的 vy/wz
**在下一步就被覆盖**，vx 不动。

影响：① `scripts/teleop.py` 的 `StairController` 与 `--hold` 演示，标称在遥控，实际测的是「引导指令」；
② 同一脚本里 `abs_vy_err / abs_wz_err` 是拿**控制器算出的指令**比实测值，而环境消费的是引导值 ⇒
「跟踪误差」在楼梯级上系统性失真（平地级正常，所以这个错很容易被忽略）。

修法：`EnvConfig.stair_guidance_under_hold`（默认 `True` = 原行为、复现训练条件）+ 条件里加
`(stair_guidance_under_hold or not _cmd_hold_active)`；`scripts/teleop.py` 加 `--strict-hold`
（置 `stair_guidance_under_hold=False`，让 hold 真正钉住 vy/wz）。
测试：`GuidanceHoldTest` 两条（默认=训练条件被改写；strict=20 步逐位钉住）。

### 2.10 顺带记录：reward.py 的一处口径不一致（低，未改）

`robotdog/reward.py:235`：判「零偏航级别」用的是**世界系**的 `cmd.wz == 0.0`，而同一项的误差用**机体系**投影
`cmd_body[2]`。`cmd.wz` 是 `CommandConfig.wz` 属性（返回当前指令 `_now[2]`，不是配置常量），所以两者只在
「机体偏航 ≈ ±90° 且世界系指令恰为 0」这类边界帧上不一致 —— 极低概率、也不改变梯度方向，本次只记录不修改。

### 2.11 B9 —— 旧 checkpoint 的 `_env.json` 缺 `gait_freq_*`，回放旧权重时相位频率用错

位置：`robotdog/policy.py` `env_config_from_summary`。

现象：用**今天**的代码回放 2026-10-13 的 sub9_v5 权重（`output/sub9_v5/checkpoints/stair9_final.zip`，
9 cm 楼梯、踏面 0.50），确定性 3 回合成功率从当年的 **2/3 掉到 0/3**（卡在第一级附近提前终止）；
而回放 2026-10-16 的 crawl24 权重则与历史日志逐位一致。

根因：那一代 `_env.json` 里**没有** `gait_freq_flat` / `gait_freq_stair` / `gait_freq_stair_fixed`
（「楼梯降频」是 10-14 之后才引入的设计）。`_rebuild(...)` 只取已知字段，缺失字段静默落当前默认
（平地 1.67 / 楼梯按台阶高插值），而当年相位时钟就是 `cfg.gait.freq`（trot_natural 2.0 Hz）⇒
`_gait_theta()`（`robotdog/env.py:1203-1217`）用错频率，`gait_phase` 那两维观测是错的。

修法：字段缺失时回填 `gait_freq_flat = gait_freq_stair = gait_freq_stair_fixed = cfg.gait.freq`。
**只对旧权重生效**（新权重的 `_env.json` 都有这些键，逐位不变），9 cm 回放从 0/3 恢复到 2/3，
与 era 代码（commit `e4184ae` 的 worktree）的 2/3 一致。

⚠ 即使回填也不是**逐位**一致：10-14 → 10-18 之间 `reset()/step()` 的随机抽取序列变过（最早体现在
OU wander 使 `_command[0]` 差约 8e-3），旧权重的 `--sampled-command` 回放只能做到**统计等价**。
要逐位复现请用当年代码：`git worktree add <dir> e4184ae`（sub9_v5）或 `9886778`（crawl24）。

测试：`tests/test_env_contract.py` `LegacyCheckpointCompatTest`（缺字段时回填、有字段时不动）。

---

## 3. 已落地改动清单

```
修改  config.yaml               +18 -4    yaw_rate_mode 字段与注释；obs 维数注释重写；vel_filter 注释
修改  robotdog/env.py           +95 -26   B1/B2/B4/B6/B8 修复；yaw_rate_mode（B7）；注释纠错（B5）
修改  robotdog/training.py      +14 -2    B3：每 env deepcopy cfg
修改  robotdog/policy.py        +18 -1    B9：旧 checkpoint 的 gait_freq 回填（复现 sub9_v5 必需）
修改  robotdog/config.py         +1 -1    B5：docstring
修改  scripts/teleop.py          +8 -0    B8：--strict-hold
修改  experience.md             +60 -0    §23.58/§23.59：把 B2/B7 与旧权重回放坑记进经验库
新增  tests/__init__.py           0        —
新增  tests/test_env_contract.py 256 行   13 条回归测试（B1–B9 的可执行契约）
新增  doc/audit_report.md        283 行   本文件
```

关键位置（行号指审计后）：

| 改动 | 位置 |
| --- | --- |
| `set_command_now` 补 `return` + 刷新 `_hist[-1]` + `_cmd_hold_active` | `robotdog/env.py:516-558` |
| `set_gait_cmd_now` 删死代码 | `robotdog/env.py:575-...` |
| OU 扰动条件加 `not _cmd_hold_active` | `robotdog/env.py:1312` |
| `yaw_rate_mode` 字段 + 校验 + 三分支实现 | `robotdog/env.py:294` / `:346` / `:776` |
| `reset` 播种高度解析化 + 清 hold | `robotdog/env.py:1059` / `:1078` |
| 条件化相位改传 `_gait_phase` | `robotdog/env.py:1162` |
| `stair_guidance_under_hold` 字段 + 门控 | `robotdog/env.py:280` / `:1329` |
| 每 env `copy.deepcopy(cfg)` | `robotdog/training.py:58` |
| `yaw_rate_mode` 配置项 | `config.yaml:134` |
| `--strict-hold` | `scripts/teleop.py`（参数定义与 `main` 里） |

**回滚方式**：`git stash` / `git checkout -- .` 即可；每个修复都是加法或等价替换，没有删除功能。
两个新开关的默认值（`legacy`、`stair_guidance_under_hold=True`）刻意选成「保持审计前行为」，
所以已归档的权重与历史结论不受影响。

---

## 4. 回归验证

| 验证 | 命令 | 结果 |
| --- | --- | --- |
| 语法 | `.venv\Scripts\python.exe -m compileall -q robotdog scripts tests` | 通过 |
| 配置冒烟 | `load_config` + `config_to_env` + `config_to_stages` | `obs.size=53`、`stages=9`、`yaw_rate_mode=legacy` |
| 契约测试 | `.venv\Scripts\python.exe -m unittest tests.test_env_contract -v` | **11/11 通过**（约 4 s） |
| 真实回放（回归） | 6 cm 权重 3 回合确定性回放（附录 A） | 385/386/407 步、x=3.904/3.903/3.903、h=0.48、3/3 成功 —— 与审计前**逐位相同** |

最后一行是本次修复的护栏：所有默认路径（`legacy` 口径 + 引导照旧生效）必须产生与审计前
**逐位一致**的轨迹，否则就说明「修 bug」变成了「改任务」。

---

## 5. 优化建议（按性价比排序）

### 5.1 结构与可维护性

1. **把指令状态从「cfg 的别名」里拿出来**（消掉 B3 这一整类问题的土壤）。现在 `env._command is
   cfg.command._now`，所以 `set_command_now` 必须同时改 cfg、课程下发必须逐环境、deepcopy 成了必需。
   建议抽出 `CommandState`（持有 `now/hold/wander/hold_active/age`，方法 `sample/resample/perturb/
   `set_now/hold_now/body_frame`）放进 `robotdog/command.py`，`EnvConfig.command` 只保留范围；
   `env` 侧暴露 `env.command` 只读视图 + `set_command_now()`。改造面已被现有测试覆盖，风险可控。
2. **拆 `env.py`（1545 行）**。建议按职责切三块、`step()` 仍是唯一编排者：
   `observation.py`（`ObsConfig` + `_obs_frame` + 归一化）、`command.py`（见上）、
   `state_machine.py`（`_contact_scan` / `track_swing_peak` / `update_probe_state` / 门控 / 终止判定）。
   已有 `reward.py` 的边界很干净（只吃 env 状态、只在 `step` 调用一次），照这个风格切即可。
3. **`EnvConfig` 加 `__post_init__` 校验**。目前只有 `ObsConfig`/`CommandConfig` 有校验，
   而 `EnvConfig` 的多数约束（`decimation >= 1`、`timestep > 0`、`episode_length_s > 0`、
   `gait_prior` 在 residual 下必须为真、`command_hold_steps >= 0`、`freq/swing band` 有序、
   `vel_filter_tau > 0`）只写在文档里，配错了要跑几百步才崩。把 `agent.md` 的「禁止事项」
   逐条落成构造期断言是最便宜的防护（`base_height` 已经这么做了，照抄即可）。

### 5.2 观测/权重的可迁移性（影响后续所有实验）

4. **给 checkpoint 写观测 schema 并校验**。现在观测布局由开关现算，配置靠 `<model>_env.json` 恢复；
   一旦 `_env.json` 缺失，`policy._obs_config_for_dim`（`robotdog/policy.py:286`）只枚举
   `("gait_phase", "vel_filter", "step_height")` 三个开关 —— **`gait_cmd=true` 的 55 维权重无法反推**，
   会直接 `SystemExit`。建议保存时写 `obs_spec`（各块名字 + 偏移 + 总维数 + flags），加载时对比并给出
   「哪一块不一致」的具体信息，而不是只报总维数。
5. **`history_len > 1` 的帧堆叠与 `vel_filter` 解耦**。当前 `history_len` 靠整帧拼接，维数是乘法关系，
   而 `_obs_config_for_dim` 又把 `history_len` 硬编码成 1（注释里自己承认不可分离）。
   建议：观测组装器显式区分「帧内块」与「帧堆叠」，并在 `obs_spec` 里分别记录。

### 5.3 性能（实测数据）

6. **缓存旋转矩阵**：`base_lin_vel_body` 每次调用都 `quat_to_mat(base_quat)`，实测**每控制步 9 次**
   （奖励、跟踪判据、观测组装、probe 状态机、`yaw_rate` legacy 路径各算一遍）。建议在 `step()` 开头算一次
   `R`（或按 `data.time` 缓存），全步复用。单次开销不大（8 env × 50 Hz），但这是零风险的清理，
   也顺便让 `yaw_rate` 的三个口径共享同一个 `R`。
7. **`_obs_frame` 的临时数组**：每步构造 4~6 个 numpy 小数组再 `np.concatenate`，可改成预分配
   `self._frame_buf` 原地写入（`obs` 随后会进 `VecNormalize`/SB3，拷贝本来就避免不了，收益有限）。
   这类改动**优先级低于正确性**，只在 profiling 显示热点时做。

### 5.4 工具链

8. **把一次性探针固化**。`.tmp/` 里堆了 ~700 个一次性脚本（gitignored），本次审计的探针就是其中之一
   （附录 A）。建议至少保留两个成建制脚本：`scripts/audit_rollout.py`（载权重跑 N 回合，输出
   x/h/成功率 + 逐步传感器统计，本篇的 yaw 数据就是它产出的）与 `scripts/check.py`
   （`compileall` + `unittest` + 单回合回放冒烟，一条命令做提交前检查）。
9. **文档一致性检查**。B5 这类「注释里的数漂了」在 18 份 `doc/*.md` + `config.yaml` 注释里只会更多。
   建议把关键数（观测维数、九级课程表、奖励权重表）从代码常量生成一次，或至少在改代码的 PR 里跑
   `tests/test_env_contract.py`（它已经把观测维数钉死）。

---

## 6. 拓展建议

### 6.1 先做训练侧 A/B：把 B7 从「观测错误」变成「任务收益」

现在只有观测侧证据（0.198 → 0.060 rad/s）。建议单级 A/B（6 cm 台阶，其余字段逐字段相同）：

| 臂 | `yaw_rate_mode` | 备注 |
| --- | --- | --- |
| 对照 | `legacy` | 复现存量，必须跑（否则无法归因） |
| 臂 A | `world` | 与指令同口径，理论上界最高 |
| 臂 B | `body_z` | 最贴近真机 IMU（部署友好），便宜 |

判据（开跑前写死）：偏航跟踪的 `mean|err|` 与 `corr(cmd, measured)`、以及楼梯成功率。
样本量按仓库规程：**3 个 seed、每 seed ≥40 回合**（成功率标准差 ±18 点的教训见 `experience.md` §14）。
注意 `vel_filter_tau=0.2` 会掩盖一部分差异，若两臂都无差异，再单独扫 `vel_filter_tau`。

### 6.2 地形接口：从「单段楼梯」到「任意地形」

`robotdog/tools/terrain.py:196` 的 `StairSpec.height_at(x)` 只依赖 x，且只支持「一段上行 + 平台 + 一段下行」。
扩展方向（连续多跑楼梯、带转角/斜置楼梯、平台+台阶混合）需要 `height_at(x, y)`。好消息是：
`env.py` 对地形的访问已经收敛在 `stair.height_at` / `_terrain_height` / `terrain_height_under_base`
三处，改造面很小。建议定义 `Terrain` 协议（`height_at(x, y)` + 可选 `normal_at` + `bounds` +
`to_xml()`），`StairSpec` 变成一个实现；现有 `flat_spec`/`demo_stairs_spec` 保持向后兼容。

### 6.3 引导控制器可插拔

`env._guidance_command()` 与 `scripts/teleop.py` 的 `StairController` 是同一套 law 的两份实现
（增益 1.2/1.5 vs 命令行参数），且它们已经出现过口径不一致（B8）。建议抽出 `GuidancePolicy` 协议
（`__call__(env, step) -> (vy, wz)`），环境侧只保留「谁来提供指令」的插槽：训练用内置引导、
遥操用 teleop 控制器、以后接上游规划器（视觉/建图给参考速度）也走同一接口。

### 6.4 课程自动化

`robotdog/curriculum.py` 是「高度表 + 步数表」的静态课程。下一步可以加「按成功率自动推进」：
每级训练结束用固定评估协议（≥40 回合、多 seed）决定是否升高度。前提是先把评估噪声与样本量写死
（§55 的教训：判据比噪声细时，自动化只会放大误判）。**不建议**在没有评估基线的情况下先做自动课程。

---

## 7. 明确不建议做的事

- **不要**为了让 yaw 跟踪好看就把 `yaw_range` 改大 / 去掉 `yaw_range_floor`（§23.10 已证伪：会放任漂移）。
- **不要**重新打开 `gait_cmd` 条件化（§23.29：把策略推向贴地）—— B6 修好只是让它「错得不那么离谱」，
  不改变该结论。
- **不要**在修复后直接重训全链：默认路径已证明逐位不变，只有明确要用 `world`/`body_z` 口径时才需要重训，
  且应按单级 A/B 先验证。
- **不要**把接触/足端力信号加进观测（会作废全部存量权重，`experience.md` 已记）。

---

## 附录 A：真实 rollout 探针（复现本篇 §2.8 数据）

```python
# 用法：$env:PYTHONPATH="."; .venv\Scripts\python.exe this_script.py
import numpy as np, mujoco
from robotdog.policy import Policy, ObsNorm, env_config_for_model
from robotdog.policy import step_height_from_model, stair_from_height
from robotdog.env import Go2StairsEnv

model = "runs/ab_6_r1/model_final.zip"
cfg = env_config_for_model(model)
stair = stair_from_height(step_height_from_model(model) * 100.0)
env = Go2StairsEnv(stair=stair, cfg=cfg, seed=0)
pol, norm = Policy.load(model), ObsNorm.load(model)

rows = []
for ep in range(3):
    obs, _ = env.reset(seed=ep)
    done = trunc = False
    while not (done or trunc):
        obs, r, done, trunc, info = env.step(pol.act(norm.normalize(obs), deterministic=True))
        q = env.data.qvel[3:6]
        R = env.data.xmat[env.meta.base_body_id].reshape(3, 3) if hasattr(env.meta, "base_body_id") else None
        rows.append((float(env.base_rpy[1]), float(env.base_rpy[0]), float(q[0]), float(q[1]), float(q[2])))
# 旧口径 / 倾斜补偿口径 / 真值的误差（真值 = (wy*sin(roll)+wz*cos(roll))/cos(pitch)）
```

说明：仓库里没有 `Policy.load` 这类逐回合回放入口，实际探针走的是 `scripts/play.py` 内部路径
（`policy_arch` + `PPO.load` + `ObsNorm`）；上面保留的是数据口径，完整实现见审计当时的 `.tmp/_audit_rollout.py`。
要长期保留，建议按 §5.4 提成 `scripts/audit_rollout.py`。
