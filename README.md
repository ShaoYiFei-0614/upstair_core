# Go2 盲走楼梯 · 最小强化学习仓

在一段 **8 级、踏面 50 cm** 的楼梯上训练 Unitree Go2 上下楼梯。

两个刻意的设定：

* **盲走（blind）**。整套代码里**没有任何触觉传感器、射线测距或接触力读取**。
  机器人只能看本体感觉（关节角/角速度、机身姿态与角速度、上一帧动作、机身线速度）
  加上一个**台阶高度标量**，靠自己撞出来怎么上台阶。
* **残差 RL**。动作不是从零学的关节目标角，而是「trot 开环先验 + PPO 残差」：

  ```
  target = home + gait_gain * gait.radians(t) + action_scale * action
  ```

  策略只学「怎么改」先验，不学怎么抬腿。

课程固定八级：**平地（直走）→ 平地（小指令）→ 平地（全量程指令）→ 3 cm → 6 cm →
9 cm → 12 cm → 15 cm**，每级在上一级的权重上继续练。前三级故意留在平地，因为
「跟着指令横移/转向」与「爬台阶」互不干扰；混在一起学时跟踪会被爬楼的压力淹没。
入口脚本有三个：
`scripts/train.py`（训练）、`scripts/play.py`（评估 / 渲染 GIF）、`scripts/teleop.py`
（外部控制器演示横向 / 偏航指令跟踪）。

本文件只讲**环境配置**与**具体代码实现**。配套四份：

* [`agent.md`](agent.md) —— **改代码之前先读这份**：仓库地图、最短验证流程、本机硬约束
  （编码/编码重定向/并发上限）、以及已经实测确认、不要重新论证的领域事实。
* [`experience.md`](experience.md) —— 踩过的坑、错过的判据、以及为什么默认值是现在这个值。
* [`train.md`](train.md) —— 训练过程、实测复现结论、能力边界。
* [`docs/`](docs/README.md) —— `robotdog/` 下**每个 `.py`** 的设计文档（目录一一对应）。

---

## 1. 环境

* **Python 3.10+**（实测 3.13.0）
* **CPU 就够**。8 个环境串行（`DummyVecEnv`，单进程）实测约 **320~400 向量步/秒**
* Windows 10/11 实测；代码里没有平台相关分支

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt ^
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple ^
    --extra-index-url https://download.pytorch.org/whl/cpu
```

`torch` 的 `+cpu` 后缀来自 PyTorch 官方 CPU 源，所以命令里**必须**带
`--extra-index-url https://download.pytorch.org/whl/cpu`（清华镜像是主源）。
`requirements.txt` 的版本与下表一致。

| 包 | 版本 | 必需 |
| --- | --- | --- |
| mujoco | 3.12.0 | ✔ |
| gymnasium | 1.3.0 | ✔ |
| stable-baselines3 | 2.9.0 | ✔ |
| torch | 2.14.1+cpu | ✔ |
| numpy | 2.2.4 | ✔ |
| tensorboard | 2.21.0 | ✔ |
| pyyaml | 6.0.3 | ✔（读 `config.yaml`） |
| pillow | 10.4.0 | 只有 `play.py --mode gif` 需要 |

### 1.1 两个环境侧的坑（装依赖时会撞上）

* **`python -m venv .venv` 可能直接失败**（`ensurepip` 返回非零 ⇒ `No module named pip`）。
  退路：`python -m venv .venv --without-pip`，再用 `get-pip.py` 补 pip。
* **沙箱/受限环境下 curl 与 `Invoke-WebRequest` 的 HTTPS 可能被 TLS 层挡住**
  （`SEC_E_NO_CREDENTIALS` / `接收时发生错误`，http_code 000），但 **Python 的
  `urllib` 正常**。要下载东西时用 Python，不要用 curl。
* 约束模式下 pip 需要写系统临时目录，可能被拒绝（`PermissionError: [Errno 13]`）。
* **读中文：PowerShell 里先设 `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8``。
  不设它 `Get-Content` 会把 UTF-8 显示成 GBK 乱码，**文件本身没有问题**。
* **不要内联 `python -c "..."`**：`pwsh` 会吃掉反引号与花括号。写 `.tmp/*.py` 再跑。
* **长训练必须在受管后台任务里跑并设 `PYTHONUNBUFFERED=1`**：`Start-Process` 起的长任务
  不跨工具调用存活。**并发上限是 2 臂**（4 臂 × 8 env 会 OOM）。

  以上全部细节见 [`agent.md`](agent.md) §5。

---

## 2. 目录

```
upstair_core/
  README.md  agent.md  experience.md  train.md   四份根文档
  config.yaml                            ★ 全部超参（脚本不再从命令行输入）
  requirements.txt
  artifacts/
    trot_stairs.json                     trot 开环先验（唯一不可重建的 artifact）
  models/unitree_go2/                    vendored Go2 MJCF + 网格 + scene.xml
  robotdog/                              库代码（分层见 docs/README.md）
    __init__.py      包入口（只重导出工具层，导入时不修控制台）
    env.py           ★ Go2StairsEnv：观测 / 奖励 / 终止 / 成功判据
    gait.py          开环 trot 先验的参数化与 (de)serialize
    go2_model.py     Go2 MJCF 组装 + 场景编译 + 元信息提取
    policy.py        训练产物的加载：配置 / 观测维度 / 网络结构 / 归一化统计
    config.py        读 config.yaml → EnvConfig + 课程阶段表
    training.py      PPO 组装、向量环境、运行目录、权重加载
    reward.py        ★ 全部奖励数学（RewardConfig + 纯函数 compute_reward）
    curriculum.py    课程数据类（StageSpec/EpisodeStats/StageResult）与汇总写出
    tools/           工具层（不 import env）
      console.py     UTF-8 控制台
      paths.py       路径常量
      terrain.py     参数化楼梯几何（StairSpec / XML / 高度查询）
  scripts/
    train.py         训练入口（自带阶段循环，逐级调 Trainer.train_stage）
    play.py          评估入口（gif / eval 两种模式）
    teleop.py        外部控制器演示（模拟人遥感操控，调 set_command_now）
    download_go2.py  重新下载 vendored Go2 模型
  docs/              每个 .py 的设计文档
  runs/<名字>/       训练产物
  output/gifs/       play.py 渲染出来的 GIF
```

---

## 3. 代码实现

### 3.1 时间与动作管线

| 量 | 值 | 说明 |
| --- | --- | --- |
| 物理步长 `timestep` | 0.002 s | MuJoCo 积分步（500 Hz） |
| 抽控 `decimation` | 10 | 每个控制步做 10 次 `mj_step` |
| 控制频率 | **50 Hz** | `control_dt = timestep * decimation = 0.02 s` |
| `episode_length_s` | 20.0 s | 1000 个控制步 |
| 动作维度 | 12 | 每条腿 `hip / thigh / calf`，顺序 `FL, FR, RL, RR` |
| `action_scale` | 0.30 rad | 残差的权限；先验走 `GaitParams.action_scale`，**两者独立** |
| 增益 `kp` / 阻尼 `kd` | `[30, 30, 45]` / `[0.6, 0.6, 0.9]` | 关节 PD，力矩 `tau = kp*(target-q) - kd*qd` 后按 `ctrlrange` 截断；选型依据见下 |

目标角（`robotdog/env.py` 的 `_target_from_action()`）：

```python
target = home + gait_gain * gait.radians(phase) + action_scale * action
target = clip(target, act_low, act_high)          # 不 clip 到 ±1
```

历史上还有一个 `action_mode: "absolute"` 分支（`target = lo + (a+1)*0.5*(hi-lo)`，
故意忽略先验）。它从未被任何一次训练或被参数搜索用过，已随清理删除 —— 现在只有
上面这一条路径。

先验的**幅值**还要再乘一个 `env.gait_amp_scale`（默认 0.5，只缩 `amp_th`/`amp_ca`、
保留高步频）。`GaitParams.default_trot()` 是先验自身 **2.05 m/s** 的平地小跑，而本任务
的指令区间是 **[0.2, 0.6] m/s** —— 不缩振幅，残差就得常驻 −0.5 才能压到 0.45，这是一个
巨大的系统性偏置，实测「奖励惩罚压它」压了两轮 250k 步都没压出跟踪（策略输出均值 20k 步
只挪 0.05）。只改这一个旋钮的三臂对照（各 200k 步、平地、`vx∈[0.2,0.6]`）：

| 幅值缩放 | \|vx 误差\| 均值 | 逐步回报 | 最负项 |
|---|---|---|---|
| 1.0（原样） | 0.597 | −0.005 | `forward_vel=-7.97` |
| 0.45 | 0.267 | +0.030 | `orientation=-0.105` |
| 0.55 | **0.178** | **+0.071** | `torque=-0.120` |

缩到 0.45~0.55 后误差降 **3.4 倍**，奖励账也从「超速惩罚主导」回到「姿态/力矩主导」。
取 **0.5** 兼顾抬腿高度（幅值 0.45 时 `base_z=0.240` 仍能正常行走，0.35 就会塌到 0.079）
与速度量程。

#### `kp` / `kd` 是怎么定的

`kp` / `kd` 是**关节 PD** 的增益（`tau = kp*(target-q) - kd*qd`），每条腿三个（hip / thigh / calf）
再 `tile(..., 4)` 成 12 维（`robotdog/env.py:293` 的 `self._kp` 与 `robotdog/env.py:294` 的 `self._kd`；
增益本身的定义在 `robotdog/env.py:194-195` 的 `kp` / `kd` 元组）。它们不是拍脑袋的，而是按**实测关节惯量**
配到目标带宽与阻尼比上的。用 `mujoco.mj_fullM(model, data, M)`（mujoco 3.12 的签名是
`mj_fullM(model, data, dst)`）在 `home_qpos` 姿态下取关节空间惯量的对角元 `J`：

| 关节 | `J` (kg·m²) | `kp` | `kd` | `w0 = sqrt(kp/J)` | `f0` | 阻尼比 `zeta = kd/(2*sqrt(kp*J))` | `1/(zeta*w0)` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hip (×4) | 0.03357 | 30.0 | 0.60 | 29.89 rad/s | 4.76 Hz | 0.299 | 112 ms |
| thigh (×4) | 0.03039 | 30.0 | 0.60 | 31.42 rad/s | 5.00 Hz | 0.314 | 101 ms |
| calf (×4) | 0.01633 | 45.0 | 0.90 | 52.50 rad/s | 8.36 Hz | 0.525 | 36 ms |

读法：

* **带宽**：三条腿 4.8 / 5.0 / 8.4 Hz，都远高于本仓 50 Hz 控制环与 2–5.5 Hz 的步态频率，
  所以 PD 不会成为步态的限制；同时低于 500 Hz 物理步长下显式积分的失稳边界。
* **阻尼比**：0.30 / 0.31 / 0.53。calf 惯量最小、带宽最高，所以**刻意给了更高的阻尼比**
  （`kd` 从 0.6 提到 0.9），否则踝关节在触地冲击下会振铃。比值 1.5 比严格等比缩放
  `sqrt(45/30) = 1.225` 更保守一点，留了裕量。
* **力矩上限**：实测 `ctrl_ranges` 是 hip / thigh `±23.70 N·m`、calf `±45.43 N·m`；
  `step()` 每步 `clip(tau, ctrl_ranges[:,0], ctrl_ranges[:,1])`，所以增益再大也只会饱和，
  不会输出超规格力矩。
* **历史**：这套值来自 vendored `go2` MJCF 的作动器风格（位置伺服 + 阻尼），
  本仓没有做自动整定 —— 上面这张表就是事后核算，用来确认它落在合理区间，而不是选型过程。

### 3.2 观测 `ObsConfig`

**每帧 42 维固定块 + 按开关追加的块**，再沿时间堆叠 `history_len` 帧：

| 块 | 维度 | 默认 | 内容 | 真机出处 |
| --- | --- | --- | --- | --- |
| 基础块 | 12 + 12 + 3 + 3 + 12 = **42** | 恒开 | 关节角、关节角速度、机身 roll/pitch/yaw、机身角速度（本地系）、上一帧动作 | 关节编码器 + IMU + 自己上一步的指令 |
| `base_lin_vel_body` | 3 | **开** | 机身系线速度 | 本体感觉估计（腿式里程计） |
| `vel_filter` | 2 | **开** | `filtered_lateral_vel` / `filtered_yaw_rate`：横向速度与偏航角速度的低通量（时间常数 `vel_filter_tau`） | 本体感觉估计（对同一路信号滤波，真机可算） |
| `base_height` | 1 | **禁止打开** | 机身相对其正下方地形的高度 —— **含地形真值**，改成 `true` 会直接抛 `ValueError` | 无（真机拿不到） |
| `gait_phase` | 2 | 关 | 步态时钟的 `sin/cos`（残差策略需要知道自己处在周期哪一点） | 内部时钟，无需传感器 |
| `step_height` | 1 | **开** | 本回合的台阶高度（m）—— 唯一直接告诉策略「在哪条楼梯上」的量 | 任务参数（可随任务下发） |
| `command` | 3 | **开** | 当前指令的前向 `vx`、横向 `vy` 与偏航角速度 `wz` | 外部控制器的输入 |

默认 `frame_size = 42 + 3 + 1 + 3 + 2 = 51`。**`config.yaml` 现在把 `history_len` 设为 `1`**
⇒ 观测就是单帧 **51 维**；`PPO` 与 `policy.py` 的维数全部跟着它走。
（`history_len = 10` 得 510 维；实测单帧在课程内不差、
在 3 cm / 6 cm 上还明显更好，见 `train.md` §3.6。）

**本仓是盲走的**：51 维里没有任何触觉、射线或接触力读数，每一项都能在真机上拿到。
`base_height` 曾经是「默认关掉的开关」，但它算的是 `base_pos[2] − 机身正下方地形高度`，
等于把地形真值塞进观测，所以现在被 `ObsConfig.__post_init__` **硬性禁止**，
`_obs_frame()` 里那条分支与 `_height_reference()` 本身都已删除。
「哪些量是真值、但只进奖励/终止」的完整清单见 `agent.md` §8.6。

帧约定（与 MuJoCo 一致）：`qpos[0:3]` 世界位置、`qpos[3:7]` 四元数 `(w,x,y,z)`、
`qvel[0:3]` 世界线速度、**`qvel[3:6]` 是机身本地角速度**（IMU 陀螺直接就是这个）。

**刻意不给**的量：足端离地高度、机身高度（默认）、任何测距/接触读数。
`step_height` 是标量，它告诉策略「这条楼梯多高」，**但不告诉它下一个立面在哪**。

### 3.3 奖励 `RewardConfig`（公式在 `robotdog/reward.py`）

| 项 | 默认 | 口径 |
| --- | --- | --- |
| `forward_vel` | **+10.0** | 机身系 `vx` 对指令的线性死区：`x = |vx - cmd.vx| / max(|cmd.vx|, 0.30)`、`shape = max(0,1-x) - track_tail*max(0,x-1)`，超速侧（`vx > cmd.vx`）再减 `forward_overspeed * err / overspeed_range`（有 `overspeed_cap` 上界）。量程就是指令本身，所以「指令说别前进」时 `vx = 0` 得 0 分（而不是旧高斯核那种「站着也拿 +0.86/步」）。**读机身系不读世界系**：世界系 x 是位姿导数，机器人自转时会在步态周期内换号 |
| `command_follow` | 见下 | 横向 / 偏航指令的跟踪项，**只在对应指令范围非零时才计**（见下方说明） |
| `orientation` | -1.2 | `roll² + pitch²` |
| `torque` | -1.5e-4 | `Σ tau²` |
| `foot_lift` | +0.9 | 三维足端速度门控（阈值 `foot_lift_gate_speed = 0.25 m/s`）× `clip(离地/目标, 0, 1)`，目标 `= max(0.08, 1.4 × 台阶高)` |
| `probe_lift` | +1.0 | **只在被阻挡时**结算的「抬高进展」（势函数塑形）：`Σ clip(摆动峰值 - 历史最好, 0, ∞)`，起算线 `probe_lift_target = 0.06 m`；总量有界、不可刷分 |
| `air_time` | +0.6 | 落地那一步逐足结算：`(clip(滞空 - 0.30, 0, ∞) - clip(滞空 - 0.60, 0, ∞))`，再对当步落地的足求和 |
| `base_height` | -2.0 | 机身高度相对**躯干正下方地形**的平方偏差（**只用于奖励**，与观测开关无关；观测侧的 `base_height` 已被禁止） |
| `joint_vel` | -2.0e-5 | `Σ qd²` |
| `action_rate` | -0.02 | `Σ (a - a_prev)²` |
| `joint_limit` | -0.5 | 关节越限程度 |
| `leg_collision` | -0.25 | 小腿/大腿碰地，逐帧 |
| `body_collision` | -1.2 | 躯干碰地，逐帧（同一次清理里从 -4.0 降到 -1.2；GIF 复盘显示 -4.0 把「贴着立面蹭」罚得比站桩还狠） |
| `alive` | +0.15 | 每步存活（**故意小**：0.4 时占「站桩」回合回报的 49%） |
| `stair_progress` | +3.0 | **一次性结算**：只在刷新本回合最高台阶时付 `max(0, 新最高 - 旧最高)`（站桩拿不到第二次） |
| `success` | **+50.0** | 一次性，到达顶端平台 |
| `termination` | **-15.0** | 一次性，摔倒/终止 |

两个关键点：

* **所有奖励数学都在 `robotdog/reward.py`**（`compute_reward(env, ...)`；它会写环境的
  三个记忆字段，见该模块 docstring），`robotdog/env.py:714-725` 的 `_compute_reward()`
  只留一个转发方法。为了便于研究与改动，单看那一个文件就够。
* 奖励账目恒等：`_compute_reward()` 返回的 `reward` 恒等于 `sum(reward_terms.values())`
  （`terms["terminal"]` 记 `+success` 或 `-termination`），所以 tensorboard 里的逐项
  拆解可以安全地相加。
* **成功/终止的权重被刻意放大**（`success=+50.0` / `termination=-15.0`，都在
  `robotdog/reward.py:101-102`）。这是实测能让 PPO 残差
  幅度增长的最强杠杆。

#### 指令跟踪项 `command_follow`

横向速度与偏航角速度各有自己的线性死区跟踪项，但**合并成一项**输出（`reward_terms`
里只有 `command_follow` 一行，这样「只往前走」阶段的项集合与加指令之前逐位一致）：

```
follow = 0
if vy_max > 0:
    x = abs(vy_actual - vy_cmd) / vy_max
    follow += 6.0 * (max(0.0, 1.0 - x) - 1.5 * max(0.0, x - 1.0))
yaw_err = abs(wz_actual - wz_cmd)                 # 偏航项恒在，不门控
if wz_cmd == 0.0:
    yaw_err = 1.5 * (yaw_err + abs(wz_actual))    # 零指令基线惩罚
x = yaw_err / max(yaw_max, 0.30)                  # 量程下限 = reward.yaw_range_floor
follow += 4.0 * (max(0.0, 1.0 - x) - 1.5 * max(0.0, x - 1.0))
```

**横向项门控 `if vy_max > 0` 是必须的，不是优化**：不加门控时，机器人直走时
`vy_actual ≈ 0`，误差只有指令量程的零头，等于**每步白拿接近满分**
—— 实测 5 cm 上 125 步拿到 `lateral_vel=+139.73`，又是一条「原地拖延」正项。
加门控后该阶段恒为 `+0.000e+00`（逐位精确零）。权重在 `robotdog/reward.py`：
`lateral_vel = 6.0`、`yaw_rate = 4.0`；`lateral_vel_sigma` / `yaw_rate_sigma`（各 0.25）
是改线性死区之前的高斯宽度；这两个字段与 `forward_vel_sigma` 一起**已经全部删除**（`RewardConfig` 里不再有它们）。
另外 `wz_cmd == 0` 时会给 `yaw_err` 补一个基线惩罚（`yaw_err = 1.5*(yaw_err + |wz|)`）—— 否则「命令我不许转」
没有任何价钱；系数从 0.5 提到 1.5，是因为实测三轴平阶级上策略偏航漂到 −47°~−88°，而 0.5 只值 −0.7/步。

⚠ **偏航项反过来必须不门控**。`yaw_max == 0` 的级别里没有任何一项读 `wz`，策略学会以
−0.608 rad/s **恒定自转**：扫 `wz ∈ [−0.6, +0.6]` 七档，实测 vx 恒 +0.365、wz 恒 −0.608，
斜率 −0.0021 —— 输出与偏航指令逐位无关，而且这个转法在后续级别里冻住。现在偏航项恒在，
量程取 `max(yaw_max, reward.yaw_range_floor)`（默认 0.30 rad/s，与 `forward_range_floor` 同一个用意：
零指令级别也要有「不转」的价钱，同时避免除零）。横向仍门控 —— 它的作动权限小得多
（缩幅值先验下 `hip` 模式 ±0.19 rad/s，横向只有 ±0.10 m/s），代价不对称。

**线性死区外侧的线性尾项**（`reward.track_tail`）是三项跟踪惩罚共用的形状：
`x = |err| / range`、`shape = max(0, 1-x) - track_tail * max(0, x-1)`。死区形状在
`|err| > range` 之后恒为 0、梯度也恒为 0，「已经偏出量程」的状态就没有回头路 ——
实测偏航误差 0.44 rad/s 而 `yaw_max` 只有 0.15 时整项归零，策略 250k 步都没把自转收回来；
线性尾项让越界之后仍有梯度，又不改变死区内的形状。

⚠ 尾项一度写成 `- track_tail * max(0, x-1)**2`（二次）。那是**符号错误**：二次减项在
`x > 1` 之后先增后减，在 `x = 1 + 1/(2·track_tail)` 处取到极小值，等于奖励在主动把误差
**推向更大**。实测缩幅值先验的平地上策略稳定在 `wz = −0.764 rad/s` 对指令 0（`x = 1.53`），
正好越过那个极小点，平地成功判据（`tol_wz = 0.5`）被一票否决 —— 整个平阶级成功率 0%
就是这么来的。形状函数现在抽成 `robotdog/reward.py` 的 `track_shape()`，并有一条
扫 `x ∈ [0, 5]`、四种 `track_tail` 的单调性自检。

注意 `yaw_rate` 读的是**机体系**偏航角速度 `(Rᵀ qvel[3:6])[2]`，**不是** `qvel[5]`，
**也不是**世界系公式 `wz + tan(pitch)·(cos(yaw)·wx + sin(yaw)·wy)`。世界系公式在数学上
也是偏航速率，但它被与指令量程（±0.70）同阶的**步态振荡**主导（实测偏差均值 0.28 vs
机体系的 0.13 rad/s），拿去当奖励会让策略只凑直流分量、实际几乎不转。见 `experience.md` §16。

### 3.4 终止与成功

| 开关 | 默认 | 语义 |
| --- | --- | --- |
| `terminate_height_ref` | `"support"` | 摔倒的高度参照面：`"support"` = 四足足下地形最低点；另有 `"base"`（躯干正下方）与 `"support_mean"` |
| `terminate_min_height` | 0.15 m | 低于参照面这么多就判摔（躯干碰撞体最低点在 0.118 m，故留 3.2 cm 余量） |
| `terminate_max_tilt` | 1.0 rad | `|roll|` 或 `|pitch|` 超过就判摔 |
| `terminate_on_success` | True | 到达顶端即终止回合 |

成功判据：躯干 x 越过 `stair.up_end_x` **且** 机身高度 ≥ 顶端高度。达成后
`_reached_top` 是 sticky 的，回合以 `is_success=True` 结束。

`info["reward_terms"]` 之外的 `info` 字段：`trunk_contact`、`leg_contact`、
`leg_contact_streak`、`base_x/y/z`、`terrain_height`、
`forward_vel`、`lateral_vel`、`yaw_rate`、`command`、`roll`、`pitch`、`crest_height`、
`reach_top_x`、`reached_top`、`is_success`、`fallen`、`gait_prior`（共 20 项）。

> **躯干接触本身不再终止回合。** 原来有一条「躯干连续接触 `terminate_contact_steps=10`
> 个控制步（0.2 s）才算摔倒」的去抖规则，已删除：实测躯干在 `h ≈ 0.28 m` 的完全
> 安全姿态下贴立面蹭可以连续到 8 步，阈值 10 只差 2 步，是纯风险零收益，而且那条
> 规则在所有驱动实验里**从未生效过**。躯干接触的代价改由 `body_collision`（逐步、
> 有界）承担；终止只由高度、倾角与超时决定。

### 3.5 指令与外部控制器

`vx` 是定值 0.55 m/s，横向速度 `vy` 与偏航角速度 `wz` 由**外部控制器**给出，
用于模拟人类遥操作。指令有「范围」和「当前值」两套状态，分别对应两个接口：

| 方法 | 谁调 | 作用 |
| --- | --- | --- |
| `set_commands(vy_max=None, yaw_max=None)` | 课程在阶段边界调 | 改**采样范围**（每个回合重置时在 `[−vy_max, +vy_max]` 内重抽） |
| `set_command_now(vx=None, vy=None, yaw_rate=None)` | 控制器每个控制步调 | 改**当前值**，立即生效 |

**观测维数在整个课程里必须恒定**（PPO 的网络与 `RolloutBuffer` 都不可变），
所以 `ObsConfig.command` 恒为 `True`，课程只能改指令**范围**的宽窄、不能开关那 3 维。
课程表在 `config.yaml` 的 `curriculum` 段（**前四级全是平地**：第 1 级只练偏航、
第 2 级只练横向、第 3 级合并、第 4 级加大并恢复超速惩罚；四级都稳了之后才上台阶）：

```yaml
heights_cm:      [0.0, 0.0, 0.0, 0.0, 3.0,  6.0,  9.0,  12.0, 15.0]
command_vx:      [0.2, 0.6]            # 每级都在 [0.2, 0.6] 内采样
command_vy_max:  [0.0, 0.30, 0.30, 0.30, 0.30, 0.35, 0.40, 0.45, 0.50]
command_yaw_max: [0.30, 0.25, 0.45, 0.55, 0.45, 0.55, 0.70, 0.80, 0.90]
curriculum.reward_overrides:
  forward_overspeed: [0.0, 0.0, 0.0, 0.3, 0.6, 1.0, 1.0, 1.0, 1.0]
  yaw_rate:          [8.0, 6.0, 6.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
```

第 1 级**偏航专练**（`command_vy_max = 0`、`yaw_max = 0.30`、超速惩罚关掉、
`yaw_rate` 权重提到 8.0）：偏航的作动权限在 hip 上（缩幅值先验实测 `hip` 模式
`u = ±0.6` 给 ∓0.16~0.19 rad/s），要它学出来必须先让「转向」在账面上压过「跑得快」；
第 2 级才把横向放开，避免两条新轴同时竞争梯度。
同高度重复出现时阶段名自动变成 `flat` / `flat_2` / `flat_3`，checkpoint 不会互相覆盖。

`scripts/teleop.py` 是控制器演示：自动控制器**可以**读地形与机身位置
（遥操作本来就有这些信息；「盲走」只限制**策略的观测**），
接口是 `StairController`（沿楼梯中线对齐）与 `HoldController`（恒定输出一条指令，
用于回放固定指令看跟踪效果）。

```bat
.venv\Scripts\python.exe scripts\teleop.py --run walk1 --height 9 --episodes 3 --gif
.venv\Scripts\python.exe scripts\teleop.py --run walk1 --flat --hold 0.40 0.35 0.00 --gif
```

**`vy` 用的是机体系横向速度 `base_lin_vel_body[1]`**（不随偏航改变含义），
`wz` 用的是**机体系**偏航角速度 `yaw_rate`（= `(Rᵀ qvel[3:6])[2]`，不是 `qvel[5]`，
也不是世界系公式 —— 后者被与指令量程同阶的步态振荡主导，见 `experience.md` §16）。

### 3.6 地形

`robotdog/tools/terrain.py` 的 `StairSpec` 描述一段楼梯：`step_height`、`tread`、
`width`、`n_up`、`n_down`、`start_x`、`landing`、`top_landing`、`center_y`。
课程里固定 `tread = 0.50`、`n_up = 8`、`width = 2.0`、`start_x = 1.50`、
`top_landing = 1.20`。

三个必须知道的细节：

* **踏面不能改**。实测 0.40 / 0.60 会让 34% 掉到 10%，0.65 直接 0%。
* 地形几何用 `group = 1`。Go2 的 MJCF 占了 group 2/3，而 MuJoCo 默认隐藏
  3/4/5 —— 用别的组会看到「狗在爬空气」。
* 顶层平台向最后一级**回伸一个踏面**（`CREST_SEAM_OVERLAP`）：两个共面 box 的接缝
  会在 z 向没有支撑，把脚夹死。

---

## 4. 用法

### 4.1 训练

```bat
.venv\Scripts\python.exe scripts\train.py
.venv\Scripts\python.exe scripts\train.py --run-name walk1
```

按 `config.yaml` 的 `curriculum` 段串行训练（默认平地 → 平地 → 平地 → 3 → 6 → 9 → 12 → 15 cm），
每级在上一级的权重上 warm-start。默认预算合计 **1,850,000 步**（平地 100k + 7 × 250k），
8 环境单进程约 320~400 步/s。

**脚本只认两个开关**，其余全部来自 `config.yaml`：

| 开关 | 默认 | 说明 |
| --- | --- | --- |
| `--run-name` | `config.yaml` 的 `run.name` | 产物目录 `runs/<名字>/`；目录已存在会拒绝启动 |
| `--config` | 仓根 `config.yaml` | 换一份配置文件（例如做 A/B） |

产物（`runs/<名字>/`）：

```
checkpoints/<阶段>_final.zip                每级结束的权重
checkpoints/<阶段>_final_vecnormalize.pkl   与它成对的观测归一化统计
checkpoints/<阶段>_final_env.json           该权重训练时的地形（台阶高）与完整配置
checkpoints/<阶段>/<阶段>_<步数>_steps.zip  周期检查点
model_final.zip / _vecnormalize.pkl / _env.json   最后一级的交付权重（三件套）
summary.json                                课程定义 + 每级结果 + 完整 env/obs/reward 配置
gifs/<批次>/<阶段>/<阶段>.gif                  play --mode gif 的产物（见 §4.3）
tensorboard/
```

⚠ **观测归一化必须跟权重成对使用**：`model_final.zip` 要配
`model_final_vecnormalize.pkl`。`play.py` 会自动找同名的那个文件，找不到时会打印警告
（缺了它确定性评估会从 100% 掉到 0%）。

### 4.2 评估：跑 100 个回合看成功率

```bat
.venv\Scripts\python.exe scripts\play.py --run walk1 --mode eval
```

`--run` 可以给运行名（查 `runs/<名字>/`）、目录或直接的 `.zip` 路径。
回合数、种子、地形与设备都在 `config.yaml` 的 `play` 段（默认 100 回合）。

**地形默认取权重自带的记录**：`<模型>_env.json` 里写了这个 checkpoint 训练时的台阶高，
所以 `checkpoints/stair9_final.zip` 会自动在 9 cm 上评估。想强制换成别的楼梯，把
`play.stage_height_cm` 取消注释（`0` = 平地）。（渲染参数不在 yaml 里：
相机 / 分辨率 / 帧率 / 缩放固定为 `scripts/play.py` 的模块常量。）

结束时打印成功率、摔倒率、超时率、平均前进/爬升/速度/步数。

⚠ **至少跑 40 个回合**。10 个回合能给出 ±15 个百分点的错觉。

### 4.3 评估：渲染 GIF（按课程阶段逐个渲染）

```bat
.venv\Scripts\python.exe scripts\play.py --run walk1 --mode gif --batch batch1
```

**判据以画面姿态为主**，所以 `--mode gif` 会把**课程里的每个阶段**各渲染成一个
GIF（平地级 + 台阶级），产物落：

```
runs/<运行名>/gifs/<批次>/<阶段>/<阶段>.gif
```

* **`--batch` 是给「不同批次的训练」分家的**：同一个运行名重训一次就得到一套新
  GIF，平铺在一个目录里根本分不清哪张是哪次训练的。不传时从 `config.yaml` 的
  `play.batch` 取；`play.batch` 也为空就直接落
  `runs/<运行名>/gifs/<阶段>/<阶段>.gif`。
* **HUD 第 3 行是本回合钉住的速度指令**（`vx` / `vy` / `wz`）加回合末的滤波实测值。
  没有这一行，看画面的人分不清「它站着不动」和「指令本来就是 0」。
* 每个阶段用**该级量程内**的一条指令（`robotdog.config.command_for_stage`，量程的 0.8 倍）；
  量程为 0 的方向就是 0。渲染哪些阶段由 `play.stages` 决定：`auto`（缺省）优先
  读运行目录 `summary.json` 里**训练时真正跑过的阶段**（改过课程再看老运行也不会
  漏渲染），没有它才退回「当前课程里能对上 `checkpoints/<阶段>_final.zip` 的那几级」；
  `curriculum` 强制渲染当前课程全部级；也可以直接给阶段名或高度列表。
  `--stage <名字或高度>` 只渲染一个。
* `--gifs-dir` 直接指定输出根目录（覆盖 `runs/<运行名>/gifs`，批次目录仍会拼上）。

每个 GIF 是**连续 10 回合**，每 2 个控制步取一帧（25 fps，左上角有 HUD）。
**渲染参数是固定的模块常量**（`scripts/play.py` 的 `CAMERA_GIF = "track"`、
`FRAME_WIDTH/HEIGHT = 960/540`、`GIF_FPS = 25`、`GIF_SCALE = 2`）—— 相机、分辨率、
帧率、缩放都是「看画面」的事，不是训练行为，留成 yaml 旋钮只会让人以为改它有意义。
`play` 段只留真正有语义的开关：回合数、seed、`gif_max_frames`、`stages`、`batch`。

GIF 用 Pillow 写，**全片共用一套调色板**（先取 16 帧拼缩略 mosaic 做中位切分量化，
再逐帧套用），否则回合之间的色块会闪。

### 4.4 所有超参在 `config.yaml`

`config.yaml` 分六段，注释里写了每个键的含义与取值理由：

| 段 | 管什么 |
| --- | --- |
| `run` | 运行名、seed、检查点间隔、日志间隔 |
| `curriculum` | 台阶高度梯度、踏面、级数、每级步数与学习率 |
| `env` | 向量环境数、回合长度、动作尺度、速度指令、终止阈值、观测开关 |
| `reward` | 全部奖励权重 |
| `ppo` | 网络结构、rollout 粒度、优化器与 PPO 超参、奖励归一化、设备 |
| `play` | 回合数、相机、分辨率、fps、缩放、可选的强制台阶高 |

读配置的是 `robotdog/config.py`：没写的键保持代码默认值，**写错的键会直接报错**
并列出可用键（打错一个字母立刻看得见，而不是被静默忽略）。

---

## 5. 已知限制

* **没有触觉/测距**。这是刻意的设计目标，不是缺失：高台阶上机器人在几何上「看不见」
  立面，只能靠撞上之后的机身姿态反馈调整。
* **`EnvConfig` 没有 `stair` 字段**。地形是构造参数：
  `Go2StairsEnv(stair=..., cfg=..., seed=...)`（`seed` 是仅关键字参数）；换地形用
  `env.set_stair(stair)`。
* 地形由**权重自带的 `<模型>_env.json`** 决定；老权重（在本轮配置化之前训的）没有这个
  文件，会退回 `stair_from_height` 的默认 3 cm，需要手动在 `config.yaml` 里写
  `play.stage_height_cm`。相机/分辨率/帧率/缩放则**没有** yaml 开关，见 §4.3。
* **本仓的课程是固定八级**，没有自动晋级/回退、没有域随机化、没有高度带混合。
* **单次训练的成功率不可当结论用**。同一份 `config.yaml` 只换 `run.seed`，3 cm 单级
  的 250,000 步训练在 **36%~72%** 之间摆动（三次运行的实测，见 `experience.md` §14）。
  比两个改动之前，必须在同一 seed 集合上各跑至少 3 次；判据以 GIF 里的姿态为主。
* **能力带约 3~6 cm**（本仓默认 48 维观测、从零训练）。历史笔记里约 10 cm 的能力带
  来自论文式 660 维配置 + 专门调过的超参，本仓的 `config.yaml` 达不到。
* **9 cm 以上还翻不过去**。GIF 复核显示策略会去试、但跨不上 9 cm 以上的立面；
  `body_collision` 已经从 `-4.0` 降到 `-1.2`，再降会让「蹭立面」变成无代价行为。

---

## 6. 代码规模与清理

### 6.1 现在的模块行数

| 文件 | 行数 | 说明 |
| --- | --- | --- |
| `robotdog/env.py` | **1227** | 环境 + 观测 + 指令 + 终止 + 渲染（奖励数学已移出） |
| `robotdog/go2_model.py` | 467 | MJCF 组装 / 场景编译 / 元信息 |
| `robotdog/config.py` | **534** | 读 `config.yaml`（含指令课程 / 奖励课程 / 播放阶段） |
| `robotdog/policy.py` | 339 | 训练产物读取（配置 / 维度 / 结构 / 归一化） |
| `robotdog/curriculum.py` | **339** | 课程数据类 + 汇总写出 |
| `robotdog/training.py` | 323 | 向量环境 / PPO / 运行目录 / 权重加载 |
| `robotdog/reward.py` | **379** | **全部奖励数学**（`RewardConfig` + `compute_reward`） |
| `robotdog/tools/terrain.py` | 287 | 楼梯几何 |
| `robotdog/gait.py` | 191 | 开环 trot 先验 |
| `robotdog/tools/paths.py` | **38** | 路径常量（**没有任何函数**） |
| `robotdog/tools/console.py` | 29 | UTF-8 控制台 |
| `robotdog/__init__.py` / `tools/__init__.py` | 19 / 18 | 包入口 |
| `scripts/play.py` | **856** | 评估 / GIF（按课程阶段逐个渲染 + 速度指令 HUD） |
| `scripts/train.py` | **411** | 训练入口（CLI 只剩 `--run-name` / `--config`） |
| `scripts/teleop.py` | **338** | 外部控制器演示（遥控操控 + 指令课程消费方） |
| `scripts/download_go2.py` | 152 | 拉取 Go2 模型资源 |
| `scripts/cmd_demo.py` | 200 | 指令跟踪的量化评估（`--phase inplace\|walk`） |

### 6.2 清理掉了哪些（本轮）

判据是**「有没有任何读者」**，用 token 流做全仓引用计数。真删的只有 6 个符号：

`load_ppo()`、`mean_abs_action()`、`set_terrain()`（都在 `robotdog/training.py`）、
`GO2_SCENE_XML`、`ensure_dir()`、`ensure_output_dirs()`（都在 `robotdog/tools/paths.py`），
外加三个失去使用者的 import（`training.py` 的 `numpy`、`policy.py` 的 `Sequence`、
`download_go2.py` 的 `sys`）。

**看起来能删但必须留的 7 个**分两类：框架回调（`scripts/train.py` 里两个 `_on_step`，
SB3 `BaseCallback` 钩子）、文档化的公开 API（`GaitParams.static()` / `targets()`、
`policy.Policy`、`StairSpec.down_end_x`、`Go2StairsEnv.yaw`、`Go2StairsEnv.set_stair()`）。
细节与判据（含「零读者 **且** 不是文档化契约才算死」这条规则）见 `experience.md` §9。

> **`GaitParams.static()` 是文档里「怎么关掉步态先验」的唯一正确写法**，所以哪怕零调用者也留着——
> 删掉它，文档就在教一个不存在的函数。
>
> **`Go2StairsEnv.set_stair()`** 的处境相同但更微妙：它**曾经**有调用者
> （`training.set_terrain()`），而 `set_terrain` 是零读者所以被删了，于是 `set_stair`
> 变成零调用点。但「换地形」本身是真需求（当前课程每级是 `make_vec_env` 新建环境，
> 不是换地形），所以保留。

### 6.3 `env.py` 为什么还有 1227 行

它不是「奖励算在里面」——奖励数学在 `robotdog/reward.py`（379 行）。`env.py` 的构成是：

| 类别 | 行数 | 占比 |
| --- | --- | --- |
| 代码 | 677 | 55.2% |
| docstring | 234 | 19.1% |
| 注释 | 153 | 12.5% |
| 空行 | 163 | 13.3% |

（分类口径：**注释 > 字符串 > 代码**，空行单列一类，四类相加恰好等于总行数。）

其中 `class Go2StairsEnv`（`robotdog/env.py:266-1227`）一个类占 **962 行**（代码 562，docstring 238），最大的成员是
`step()`（103 行）、`__init__()`（97 行）、`reset()`（62 行）、`render()`（59 行）、
`_obs_frame()`（37 行）与 `_contact_flags()`（37 行）。类里共 46 个成员、合计 880 行。

长的原因有三条，都是**有意的**：

1. **一个 Gymnasium 环境的接口面天生就宽**。`step` / `reset` / `render` / `close` +
   观测 / 终止 / 成功 / 接触 / 高度参照 / 落脚空气时间 / 模型与场景构建 / 楼梯测距，
   加起来 **46 个方法 + 3 条类级语句**（其中 14 个方法是 `@property` 只读量）。这不是
   「一个类干了很多事」，而是「Gym 环境本来就要求这么多钩子」。
2. **`render()` 的 59 行是相机与 HUD**，与物理仿真没有关系。它可以搬到 `scripts/play.py`
   或独立的渲染模块，但那样它就得通过 30 多个 `env.<私有属性>` 读状态——**耦合面反而更大**，
   所以留在环境里、和状态同生共死。
3. **盲走设计把状态量摊在了很多小方法里**（`_contact_flags` / `_support_height_mean` /
   `_fall_reference_height` / `_obs_frame` …）。这是刻意的：每个方法只回答一个问题，
   且各自有 docstring 说明「为什么不是更简单的写法」（例如 `_support_height_mean` 用均值
   而不是 `min`，因为 `min` 会在最后一只脚落地时一帧跳一整个立面）。

**如果一定要更短**，唯一的无损做法是把 `render()` 拆出去（约 59 行）；剩下的
`step` / `reset` / `_obs_frame` / `_compute_reward` 转发 / 接触与终止判据属于环境本体，
拆开会把「谁能改物理状态」这条边界搅浑。当前的取舍是：**奖励数学已经拆干净，
渲染暂不拆。**

`Go2StairsEnv` 的 41 个成员逐个是干什么的、各自被谁调用，见
[`docs/robotdog/env.md`](docs/robotdog/env.md) §4 与 §5。

### 6.4 改完代码怎么验证

`.tmp/` 下六支脚本是回归门（都在 1 秒内跑完，且**不依赖 `runs/` 里的任何权重**）：

```bat
.venv\Scripts\python.exe -m compileall -q robotdog scripts
.venv\Scripts\python.exe .tmp\check_lines.py     :: 文档里的 文件.py:行号 是否越界/指向空行
.venv\Scripts\python.exe .tmp\ref_truth.py       :: 每个行号指向的真实内容（抓「漂移但未越界」）
.venv\Scripts\python.exe .tmp\check_doc_code.py  :: markdown 的 ```python 块能否 ast.parse
.venv\Scripts\python.exe .tmp\check_xrefs.py     :: 文档之间的相互引用是否存在
.venv\Scripts\python.exe .tmp\audit_docs.py      :: 10 节结构 / 交叉引用 / 索引完整性
```

`check_lines.py` **只报症状不报病因**——两个盲点：①发现不了「文档描述了一个不存在的
API」；②发现不了「行号整体漂移但没越界」（源文件里加一段 docstring 就会造成这种情况，
本轮就踩到了：给 `robotdog/env.py` 的某个方法 docstring 加长 5 行，全仓 14 处
`robotdog/env.py` 行号引用**全部指到了别的函数**，而它们每一行都还是合法的非空行）。
`ref_truth.py` 就是为第 ② 个盲点写的：它**现算**每条引用指向的那一行是什么、指到
`def`/`class` 就报警，所以不需要任何手工维护的期望值。即便有它，改了公开 API 之后
仍要人工读一遍对应文档；完整的验证流程（含训练/播放冒烟与奖励改动的等价性证明）
见 [`agent.md`](agent.md) §7。

