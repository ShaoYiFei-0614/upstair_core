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

课程固定六级：**平地 → 3 cm → 6 cm → 9 cm → 12 cm → 15 cm**，每级在上一级的权重上
继续练。入口脚本只有两个：`scripts/train.py`（训练）与 `scripts/play.py`（评估 /
渲染 GIF）。

本文件只讲**环境配置**与**具体代码实现**。配套两份：

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

---

## 2. 目录

```
upstair_core/
  README.md  experience.md  train.md     三份根文档
  config.yaml                            ★ 全部超参（脚本不再从命令行输入）
  requirements.txt
  artifacts/
    trot_stairs.json                     trot 开环先验（唯一不可重建的 artifact）
  models/unitree_go2/                    vendored Go2 MJCF + 网格 + scene.xml
  robotdog/                              库代码（分层见 docs/README.md）
    __init__.py      包入口，导入即修 UTF-8 控制台
    env.py           ★ Go2StairsEnv：观测 / 奖励 / 终止 / 成功判据
    gait.py          开环 trot 先验的参数化与 (de)serialize
    go2_model.py     Go2 MJCF 组装 + 场景编译 + 元信息提取
    policy.py        训练产物的加载：配置 / 观测维度 / 网络结构 / 归一化统计
    config.py        读 config.yaml → EnvConfig + 课程阶段表
    training.py      PPO 组装、向量环境、运行目录
    curriculum.py    六级课程与阶段回调
    tools/           工具层（不 import env）
      console.py     UTF-8 控制台
      paths.py       路径常量
      png.py         纯 zlib 的 PNG 读写（不依赖 Pillow；当前无调用者，见下）
      terrain.py     参数化楼梯几何（StairSpec / XML / 高度查询）
  scripts/
    train.py         训练入口（调 curriculum）
    play.py          评估入口（gif / eval 两种模式）
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
| `action_mode` | `"residual"` | 先验 + 残差 |
| `action_scale` | 0.30 rad | 残差的权限；先验走 `GaitParams.action_scale`，**两者独立** |
| 增益 `kp` / 阻尼 `kd` | `[30, 30, 45]` / `[0.6, 0.6, 0.9]` | 关节 PD，力矩 `tau = kp*(target-q) - kd*qd` 后按 `ctrlrange` 截断 |

`residual` 模式的目标角：

```python
target = home + gait_gain * gait.radians(phase) + action_scale * action
target = clip(target, act_low, act_high)          # 不 clip 到 ±1
```

`absolute` 模式忽略先验：`target = lo + (a + 1) * 0.5 * (hi - lo)`。

### 3.2 观测 `ObsConfig`

**每帧 42 维固定块 + 按开关追加的块**，再沿时间堆叠 `history_len = 10` 帧：

| 块 | 维度 | 默认 | 内容 |
| --- | --- | --- | --- |
| 基础块 | 12 + 12 + 3 + 3 + 12 = **42** | 恒开 | 关节角、关节角速度、机身 roll/pitch/yaw、机身角速度（本地系）、上一帧动作 |
| `base_lin_vel_body` | 3 | **开** | 机身系线速度 |
| `base_height` | 1 | 关 | 机身相对其正下方地形的高度 |
| `gait_phase` | 2 | 关 | 步态时钟的 `sin/cos`（残差策略需要知道自己处在周期哪一点） |
| `step_height` | 1 | **开** | 本回合的台阶高度（m）—— 唯一直接告诉策略「在哪条楼梯上」的量 |

默认 `frame_size = 42 + 3 + 1 = 46` ⇒ **观测 460 维**。开 `gait_phase` 得 480 维。

帧约定（与 MuJoCo 一致）：`qpos[0:3]` 世界位置、`qpos[3:7]` 四元数 `(w,x,y,z)`、
`qvel[0:3]` 世界线速度、**`qvel[3:6]` 是机身本地角速度**（IMU 陀螺直接就是这个）。

**刻意不给**的量：足端离地高度、机身高度（默认）、任何测距/接触读数。
`step_height` 是标量，它告诉策略「这条楼梯多高」，**但不告诉它下一个立面在哪**。

### 3.3 奖励 `RewardConfig`

| 项 | 默认 | 口径 |
| --- | --- | --- |
| `forward_vel` | +2.5 | `exp(-((vx - command_vx)/0.25)^2)`，`command_vx = 0.55 m/s` |
| `forward_progress` | +1.5 | `clip(vx, -0.5, 1.5)` 线性项 —— vx=0 时恰为 0，堵住「站着不动」的局部最优 |
| `orientation` | -1.2 | `roll² + pitch²` |
| `torque` | -1.5e-4 | `Σ tau²` |
| `foot_lift` | +0.9 | 三维足端速度门控 × `clip(离地/目标, 0, 1)`，目标 `= max(0.08, 1.4 × 台阶高)` |
| `air_time` | +0.6 | 落地那一步结算：`clip(滞空 - 0.30, 0, ...)`，超过 0.60 s 反罚 |
| `base_height` | -2.0 | 机身高度偏离其正下方地形（**只用于奖励**，与观测开关无关） |
| `joint_vel` | -2.0e-5 | `Σ qd²` |
| `action_rate` | -0.02 | `Σ (a - a_prev)²` |
| `joint_limit` | -0.5 | 关节越限程度 |
| `leg_collision` | -0.25 | 小腿/大腿碰地，逐帧 |
| `body_collision` | -4.0 | 躯干碰地，逐帧 |
| `alive` | +0.15 | 每步存活（**故意小**：0.4 时占「站桩」回合回报的 49%） |
| `stair_progress` | +3.0 | `max(0, 参考面高度 - 起始高度)`，参考面取躯干正下方地形 |
| `success` | **+50.0** | 一次性，到达顶端平台 |
| `termination` | **-15.0** | 一次性，摔倒/终止 |

两个关键点：

* 奖励账目恒等：`_compute_reward()` 返回的 `reward` 恒等于 `sum(reward_terms.values())`
  （`terms["terminal"]` 记 `+success` 或 `-termination`），所以 tensorboard 里的逐项
  拆解可以安全地相加。
* **成功/终止的权重被刻意放大**（50 / -15，原值 20 / -2）。这是实测能让 PPO 残差
  幅度增长的最强杠杆。

### 3.4 终止与成功

| 开关 | 默认 | 语义 |
| --- | --- | --- |
| `terminate_contact_steps` | 10 | 躯干持续接触 10 个控制步（0.2 s）才算摔倒 —— 去抖，避免单帧擦碰就判死 |
| `terminate_height_ref` | `"support"` | 摔倒的高度参照面：`"support"` = 四足足下地形最低点；另有 `"base"`（躯干正下方）与 `"support_mean"` |
| `terminate_min_height` | 0.13 m | 低于参照面这么多就判摔 |
| `terminate_max_tilt` | 1.0 rad | `|roll|` 或 `|pitch|` 超过就判摔 |
| `terminate_on_success` | True | 到达顶端即终止回合 |

成功判据：躯干 x 越过 `stair.up_end_x` **且** 机身高度 ≥ 顶端高度。达成后
`_reached_top` 是 sticky 的，回合以 `is_success=True` 结束。

`info["reward_terms"]` 之外的 `info` 字段：`trunk_contact`、`leg_contact`、
`terminate_contact_steps`、`trunk_contact_streak`、`base_x/y/z`、`terrain_height`、
`forward_vel`、`roll`、`pitch`、`crest_height`、`reach_top_x`、`reached_top`、
`is_success`、`fallen`、`gait_prior`。

### 3.5 地形

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

按 `config.yaml` 的 `curriculum` 段串行训练（默认平地 → 3 → 6 → 9 → 12 → 15 cm），
每级在上一级的权重上 warm-start。默认预算合计 **1,350,000 步**（平地 100k + 5 × 250k），
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
`play.stage_height_cm` 取消注释（`0` = 平地）。

结束时打印成功率、摔倒率、超时率、平均前进/爬升/速度/步数。

⚠ **至少跑 40 个回合**。10 个回合能给出 ±15 个百分点的错觉。

### 4.3 评估：渲染 GIF

```bat
.venv\Scripts\python.exe scripts\play.py --run walk1 --mode gif
```

在 `output/gifs/<模型名>.gif` 出一个**连续 10 回合**的 GIF（默认每 2 个控制步取一帧 =
25 fps，左上角有 HUD）。相机固定为 `track`；分辨率、fps、缩放、回合数都在
`config.yaml` 的 `play` 段。

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
* **`EnvConfig` 没有 `stair` 字段**。地形只能通过
  `Go2StairsEnv(stair=..., cfg=..., seed=...)` 传入；`set_stair()` 用于换地形。
* 地形由**权重自带的 `<模型>_env.json`** 决定；老权重（在本轮配置化之前训的）没有这个
  文件，会退回 `stair_from_height` 的默认 3 cm，需要手动在 `config.yaml` 里写
  `play.stage_height_cm`。
* 本仓的课程是**固定六级**，没有自动晋级/回退、没有域随机化、没有高度带混合。
* **`robotdog/tools/png.py` 目前没有调用者**。它的三个历史消费者（`render.py` /
  `render_checkpoint.py` / `eval.py`）都已删除，GIF 与拼图现在由 `scripts/play.py`
  直接用 Pillow 完成。它仍被 `robotdog/tools/__init__.py` 导入，所以
  `import robotdog` 会连带加载 `numpy`。保留是为了避免在没有需求时反复改动包入口；
  它是全仓最明确的候选删除项（见 `docs/robotdog/tools/png.md` §9）。
