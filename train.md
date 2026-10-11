# train.md — 训练过程与实测结论

本文件记录**训练是怎么组织的**、**一次训练长什么样**、以及**实测到的结论与能力边界**。
代码实现见 `README.md`，踩过的坑见 `experience.md`。

---

## 1. 训练是怎么组织的

### 1.1 固定八级课程

`config.yaml` 的 `curriculum` 段（经 `robotdog/config.py:config_to_stages()` 展开）
**当前是八级**：`[0, 0, 0, 3, 6, 9, 12, 15] cm`。**前三级都是平地**：指令跟踪与爬楼是
两种能力，混在一起学时跟踪会被爬楼的压力淹没（实测平地单级 250k 步也学不会，会横移的
策略基本不转向）。

| 阶段 | 台阶 | 踏面 | 级数 | 预算 | 学习率 | 横向 / 偏航上界 | 前向指令区间 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `flat` | 平地（`n_up=0`） | 0.50 | 0 | 100,000 | 1.5e-4 | 0 / 0 | `[0.2, 0.6]` |
| `flat_2` | 平地 | 0.50 | 0 | 250,000 | 1.5e-4 | ±0.15 / ±0.15 | `[0.2, 0.6]` |
| `flat_3` | 平地 | 0.50 | 0 | 250,000 | 1.2e-4 | ±0.35 / ±0.50 | `[0.2, 0.6]` |
| `stair3` | 3 cm | 0.50 | 8 | 250,000 | 1.0e-4 | ±0.30 / ±0.35 | `[0.2, 0.6]` |
| `stair6` | 6 cm | 0.50 | 8 | 250,000 | 8e-5 | ±0.35 / ±0.55 | `[0.2, 0.6]` |
| `stair9` | 9 cm | 0.50 | 8 | 250,000 | 6e-5 | ±0.40 / ±0.70 | `[0.2, 0.6]` |
| `stair12` | 12 cm | 0.50 | 8 | 250,000 | 5e-5 | ±0.45 / ±0.80 | `[0.2, 0.6]` |
| `stair15` | 15 cm | 0.50 | 8 | 250,000 | 4e-5 | ±0.50 / ±0.90 | `[0.2, 0.6]` |

**默认总预算 1,850,000 步**。台阶越高、学习率越低（越往上越接近微调，不希望破坏已学的
地形）。同一高度出现多次时阶段名会自动变成 `flat` / `flat_2` / `flat_3`（代码里的
`_unique_name()`），否则 checkpoint 与 tensorboard 目录会互相覆盖。

**前向指令也是被跟踪的量**：每一回合在 `command_vx` 区间里均匀采样（默认 `[0.2, 0.6]`
m/s，所有级别共用），超出指令的部分由 `reward.forward_overspeed` 额外扣分。
平地的成功判据是**指令跟随质量**（脱靶步数比例 ≤ `env.flat_track_miss_ratio`），
不再要求走满某个固定距离。

```bat
.venv\Scripts\python.exe scripts\train.py --run-name walk1
```

脚本只认 `--run-name` 与 `--config`（默认仓根 `config.yaml`），其余全部来自配置。

启动时会先把整份课程表打出来，形如：

```
[配置] config.yaml
[运行] ...\runs\walk1  (seed=0, n_envs=8)
[观测] 每帧 51 维 × history 1 = 51 维
[动作] 残差模式, action_scale=0.3 rad, 步态先验=开
[指令] 基线 vx=0.45 m/s；实际三轴范围由指令课程按级下发（观测里的那 3 维恒在）
[步态] 5.48 Hz, ±0.265 thigh / ±0.986 calf (norm. units), shape=sin, scale=1.00 rad, period=183 ms
[奖励] forward_vel=10.0 foot_lift=0.9 success=50.0 termination=-15.0
[PPO] 网络=[512, 256] n_steps=256 batch=512 epochs=5 target_kl=0.03 log_std_init=-2.0 norm_reward=True

阶段       标签             台阶    踏面   级数      顶端         预算      学习率
flat     平地           0.0c   50c    0   0.00m    100,000  0.00015
flat_2   平地           0.0c   50c    0   0.00m    250,000  0.00015
flat_3   平地           0.0c   50c    0   0.00m    250,000  0.00012
stair3   3cm 台阶       3.0c   50c    8   0.24m    250,000   0.0001
stair6   6cm 台阶       6.0c   50c    8   0.48m    250,000    8e-05
stair9   9cm 台阶       9.0c   50c    8   0.72m    250,000    6e-05
stair12  12cm 台阶     12.0c   50c    8   0.96m    250,000    5e-05
stair15  15cm 台阶     15.0c   50c    8   1.20m    250,000    4e-05
合计 8 级, 总预算 1,850,000 步
```

> `flat_3` / `stair3` 起每级的课程行末尾会多出 `指令 vy±0.35 wz±0.5`、
> `vx=[0.2,0.6]`、`奖励覆盖 foot_lift=… probe_lift=…` 之类的后缀 ——
> 分别来自 `config.yaml` 的 `curriculum.command_vy_max` / `command_yaw_max` /
> `command_vx` / `reward_overrides`。

每一级开头与结尾各有一行：

```
[课程] >>> 第 4/8 级 'stair3' (3cm 台阶) stair3   3cm 台阶     台阶=  3.0cm 踏面=  50cm 级数=8  顶端= 0.24m 预算=  250,000 lr=0.0001 指令 vy±0.30 wz±0.35 vx=[0.2,0.6] 奖励覆盖 foot_lift=1.0 foot_lift_step_ratio=1.6 probe_lift=0.6
    [warm-start] 从 flat_3_final.zip 继承权重
    [      2,048 步] 回合=    8 成功率= 100.0% 摔倒率=   0.0% 均回报=  332.33 均步数= 112.0     ← ProgressCallback
    训练完成 250,000 步  412 步/秒
[课程] <<< 'stair3' 成功率=100.0% 摔倒率=0.0% 均回报=332.3 用时=606.8s
```

全部跑完后重打一遍课程表（这次带成绩列），并列出产物路径：

```
flat     平地           0.0c   50c    0   0.00m    100,000  0.00015 100.0%   0.0%    332.87   242.7s flat_final.zip
flat_2   平地           0.0c   50c    0   0.00m    250,000  0.00015 ...
...
合计 8 级, 总预算 1,850,000 步

产物: runs/walk1
  模型: runs/walk1/model_final.zip
  汇总: runs/walk1/summary.json

下一步:
  .venv\Scripts\python.exe scripts\play.py --run walk1 --mode eval
  .venv\Scripts\python.exe scripts\play.py --run walk1 --mode gif
```

> 上面带 `[...]` 的进度行与实际步数/耗时是**示意值**；能逐字复现的是冒烟跑（§3.1）
> 里那份每级 2,000 步、总预算 12,000 步的输出。

### 1.2 阶段之间怎么衔接

```
scripts/train.py
  config_to_stages(raw) → [StageSpec × 6]
  for stage in stages:
      venv   = make_vec_env(stage.spec, n_envs, cfg, seed)   # 地形换成这一级
      model  = build_ppo(venv, learning_rate=stage.learning_rate, ...)
      if 上一级完成: model.set_parameters(上一级 checkpoints/<name>_final.zip)   # warm-start
      model.learn(total_timesteps=stage.timesteps,
                  callback=[ProgressCallback, CheckpointCallback])
      RunDir.save_model(model, f"checkpoints/{stage.name}_final", venv)
```

* **逐级 warm-start**：每一级在上一级的权重上继续练（`PPO.set_parameters`）。
* **换地形是整环境重建**，不是在原地调 `set_stair`。主循环每一级都 `trainer.make_env(stage)`
  新建整套 `DummyVecEnv → VecMonitor → VecNormalize`，上一级的链在 `finally: venv.close()`
  里整体丢掉（地形烘焙在编译后的 MJCF 里，重建才换得干净）。
  * 连带后果：**`VecNormalize` 的观测统计每级都从零重新攒**，跨级继承的只有权重。
    所以换级后的前若干步，策略会拿到一套均值/方差还在漂移的观测。
  * `set_terrain(venv, stair)` 已随死代码清理删除；`Go2StairsEnv.set_stair()` 还在但
    零调用点（改地形要自己写 `venv.env_method("set_stair", stair)` + `venv.reset()`，
    代价是每子环境重编译 MJCF 约 250 ms × `n_envs`）。细节见 `experience.md` §9.1
    与 `docs/robotdog/training.md` §3.5。
* PPO 的更新粒度 = `n_steps 256 × n_envs 8 = 2048` 步。**每一级至少有 48 次以上更新**，
  这是刻意的：早期版本的课程毕业判据没有最少步数，导致每档只跑 0~1 次更新、策略输出
  恒等于 0（见 `experience.md` §4.1）。

### 1.3 回调

* `ProgressCallback`：**只在有新回合结束、且距上次打印 ≥ `run.log_every` 次更新时**打印
  一行 `EpisodeStats.format_line()`。早期版本按 `n_calls % log_interval` 无条件打印，
  日志刷屏且看不到回合统计。
* `CheckpointCallback`：每 `run.save_every` 步存
  `checkpoints/<阶段>/<阶段>_<步数>_steps.zip` + 同名 `_vecnormalize.pkl`。

### 1.4 向量环境

`DummyVecEnv → VecMonitor → VecNormalize(training=True, norm_obs=True, norm_reward=True, clip_obs=10.0)`。

* **必须用 `DummyVecEnv`**：本环境禁用 Windows 命名管道，`SubprocVecEnv` 不可用。
* `VecMonitor` 在 `VecNormalize` **内层**，这样 `info["episode"]["r"]` 是**未归一化**的
  真实回报 —— 日志里的均回报才有物理意义。
* `norm_reward` 默认 **True**（实测有帮助，见 `experience.md` §4.2）。

---

## 2. 一次训练长什么样

完整输出样例已放在 §1.1（含课程表、逐级进度行与收尾汇总）。这一节只讲**产物**。

### 产物

```
runs/<名字>/
  checkpoints/<阶段>_final.zip                  每级结束的权重
  checkpoints/<阶段>_final_vecnormalize.pkl     与它成对的观测统计
  checkpoints/<阶段>_final_env.json             该权重训练时的地形（台阶高）+ 完整配置
  checkpoints/<阶段>/<阶段>_<步数>_steps.zip     周期检查点（每 run.save_every 步）
  checkpoints/<阶段>/<阶段>_<步数>_steps_vecnormalize.pkl
  model_final.zip                              最后一级的交付权重（三件套之一）
  model_final_vecnormalize.pkl                 与它成对的观测/奖励归一化统计
  model_final_env.json                         地形记录（play 默认按它选楼梯）
  summary.json                                 课程定义 + 每级结果 + 全部配置
  gifs/<批次>/<阶段>/<阶段>.gif                  play --mode gif 的产物（见 §2.1）
  tensorboard/
```

`<阶段>_final_env.json` 是**每个 checkpoint 各一份**的，因为每一级的地形都不一样：
只有最后一级是 15 cm，`checkpoints/stair9_final.zip` 是 9 cm。
`summary.json` 的键：`stages`（八级定义）、`results`（每级实测）、`env_config`、
`obs_config`、`reward_config`、`total_timesteps`、`wall_seconds`、`final_model`。
（`reward_config` 是 `robotdog/reward.py` 的 `RewardConfig` 全部字段的完整快照，当前 28 个。）
**`play.py` 靠 env.json 与它恢复配置**，所以搬动权重时不要漏掉这两类文件。

### 2.1 用 GIF 判成果（而不是只看日志里的成功率）

**判据以画面姿态为主**：日志里的成功率含动作噪声，`mean_return` 更是可以靠
「站着不动」刷出来（见 §3.4）。一次 `--mode gif` 会把**课程里的每个阶段**各渲染成
一个 GIF，平地级与台阶级都在内 —— 平地那几级是判断「跟不跟横向/偏航指令」的
唯一画面证据：

```bat
.venv\Scripts\python.exe scripts\play.py --run <名字> --mode gif --batch <批次>
```

* 产物落 `runs/<名字>/gifs/<批次>/<阶段>/<阶段>.gif`。**批次目录是必须的**：同一个
  运行名重训一次就会得到一套新 GIF，平铺在一个目录里分不清是哪次训练的；阶段再分
  一层子目录，则是因为一次就出十来张（地形 × 指令），铺平了同样认不出谁是谁。
* **HUD 的第 3 行是本回合钉住的速度指令**（`vx` / `vy` / `wz`）以及回合末的
  滤波实测值。没有这一行，看画面的人分不清「它站着不动」和「指令本来就是 0」。
  平地段 HUD 第 2 行打「地形 平地（无台阶）」，因为 `台阶 0.0cm × 0 级` 是噪声。
* 每个阶段用**该级量程内**的一条指令（`robotdog.config.command_for_stage`，取量程的
  0.8 倍）；量程为 0 的方向就是 0，所以 `flat` 那一级是纯直走。
* `play.stages` 决定渲染哪些阶段。`auto`（缺省）**优先读运行目录的
  `summary.json`** —— 里面记着训练时真正跑过哪几级、每级多高，所以「改了课程再回看
  老运行」不会静默漏掉几级（实测：cmdtrack6 是按 6 级全平地的临时课程训的，若只按
  当前课程名去对 checkpoint，就只剩 3 级能被渲染）。没有 `summary.json` 时才退回
  「当前课程里能对上 `checkpoints/<阶段>_final.zip` 的那几级」，再退到全部级。
  ⚠ 每级的**指令量程**只能按名字从**当前** `config.yaml` 查：训练量程没有存进权重
  （`env.json` 与 `summary.json` 里都只有基线 `vy_max=0`），脚本对查不到的级会打印
  `⚠ <阶段>: 无（指令取 0）`。
* 相机 / 分辨率 / 帧率 / 缩放是 `scripts/play.py` 的**模块常量**（`CAMERA_GIF` /
  `FRAME_WIDTH` / `FRAME_HEIGHT` / `GIF_FPS` / `GIF_SCALE`），不在 yaml 里。
  `play.gif_max_frames`（默认 300）限制单回合写入帧数：一帧 960×540 的 uint8 约
  1.5 MB，回合可能拖过 500 步，不限就会在渲染中途 `MemoryError`（真的发生过）。

读 GIF 的关键画面：脚跟**落在踏面上**还是踢在立面上；跨上第一级后是**站着**还是
**趴着蹭**；下坡/落地时有没有把机身摔出去。这三件事日志都看不出来。

---

## 3. 实测结论（重要）

### 3.1 冒烟验证：流水线是通的

用一份每级 2,000 步的配置（`.tmp/smoke_config.yaml`，`curriculum.timesteps: 2000`、
`env.n_envs: 2`、`ppo.net_arch: [64, 64]`）跑完六级，约 35 秒：

| 阶段 | 步数 | 采样成功率 | 摔倒率 | 均回报 |
| --- | --- | --- | --- | --- |
| flat | 2,000 | 100.0% | 0.0% | 330.87 |
| stair3 | 2,000 | 50.0% | 40.0% | 525.34 |
| stair6 | 2,000 | 0.0% | 80.0% | 236.25 |
| stair9 | 2,000 | 0.0% | 0.0% | 101.56 |
| stair12 | 2,000 | 0.0% | 0.0% | 35.75 |
| stair15 | 2,000 | 0.0% | 0.0% | -39.04 |

⇒ 流水线、回调、存档、`summary.json`、`play.py` 两种模式全部验证通过。
**这些数字只证明管道通，不证明训练有效** —— 每级只有约 1~8 次 PPO 更新。

注意 `stair3` 的采样成功率 **50%** 是**探索**口径（每步动作加 `log_std ≈ 0.135` 的噪声），
它会大幅低估确定性策略。

### 3.2 残差幅度是判断「有没有学起来」的关键量

纯 trot 开环先验（零残差）的能力：**平地 100% 走完 3.0 m**，但楼梯上几乎全摔
（5 cm 8 回合 7 摔、8 cm 6 摔）。所以「策略 ≈ 先验」在楼梯上等价于「没学会」。

判断标准是 `mean|a|`（确定性策略的动作绝对值均值）：

| 策略 | `mean|a|` | `p95|a|` | `|a| < 0.05` 占比 | 结论 |
| --- | --- | --- | --- | --- |
| 纯先验（零残差） | 0 | — | 100% | 上不了楼梯 |
| 参考 checkpoint | **0.42** | 1.00 | 7% | 能上 8~10 cm |
| 从零训练 300k 步 | 0.02~0.10 | 0.1~0.4 | 40~89% | **退化成纯先验** |

参考策略的 `mean|a| ≈ 0.42` 是**结构常数**，不随台阶高变化（2 cm 时 0.421，
11 cm 时 0.412），而且 `p95` 恒为 1.00（长期饱和）。**要进入这个工作区间，起步要百万步
量级**。

### 3.3 从零训练能达到什么程度（本仓实测，旧 660 维观测面）

一次全课程 run（1,175,000 步、63.6 分钟、8 环境）：

| 阶段 | 步数 | 成功率 | 摔倒率 | 均回报 |
| --- | --- | --- | --- | --- |
| 平地 | 125,000 | **100.0%** | 0.0% | +347.9 |
| 5 cm | 300,000 | **0.0%** | 20.0% | -2529.7 |
| 11 cm | 350,000 | **0.0%** | 40.0% | -2482.7 |
| 11 cm（续） | 400,000 | **0.0%** | 66.0% | -1348.7 |

* critic **学起来了**：`train/explained_variance = 0.946`（对比从零单档训练只有 0.01）。
* 策略确实在「努力」：`mean|a| ≈ 0.25~0.29`（参考的 60%），能走能爬 2~3 cm
  （到 `x = 4.72 / 3.96`），5~6 cm 上受限（`x ≈ 2.4~2.6`），11 cm 上几乎原地
  （`mean_vx = +0.044`）。
* **产出策略在标准地形上仍是 0% 成功**（5 cm 和 8 cm 上 100% 摔倒）。

⇒ 对抗性评估下「训练报告变好」不等于「策略变好」：同一批权重里，**训练日志的摔倒率
减半**（easy 阶段 58% → 20~26%），但拿去做标准地形测试仍是 0%。

### 3.4 什么杠杆有效、什么无效

| 杠杆 | 实测效果 |
| --- | --- |
| `norm_reward=True` | **有效但温和**：3 cm 成功率 0.474 → 0.539，`mean\|a\|` 0.070 → 0.080 |
| `success=200 / termination=-20`（放大成功/终止） | **最强**：`mean\|a\|` 增长速度约为「仅归一化」的 2 倍 |
| `action_scale=0.60` | 只降摔倒率，**不增加前进** |
| 降低侧向惩罚（`straight_line`/`heading` 调小） | 只降回报，**不提高前进** |
| 把侧向惩罚全部清零 + 前进奖励翻倍 | **仍然无效**：`mean\|a\|` 卡在 0.025 |
| 增大 `action_scale=0.80` | **无效**：策略不会因此放大输出 |
| 从参考 checkpoint warm-start | 5/8 cm 保持 100%，**11 cm 不改善**（0%，摔倒率反而从 0.38 涨到 0.75） |

⇒ 结论：**奖励尺度（而不是奖励权重）是把策略压回「几乎不动」的主因**，
`norm_reward` 恢复可学的 critic 目标后残差才开始增长；但即使如此，从零训练需要
**百万步量级**才能接近参考的工作区间。

### 3.5 高台阶的困难：不是「抬不起来」

* 纯先验在平地上的摆动峰值是 **0.112 m**（把步频降到 60% 可达 0.166，加大幅值可达 0.229），
  而 **11 cm 台阶只需要 0.11 m** ⇒ 几何上完全够。
* 参考策略在 11 cm 上实测峰值**只有约 0.09 m**，而且几乎不随台阶高缩放。
* 开环调参扫描（18 组 `gain × amp × freq`）在 9/11/13 cm 上**无一成功**。
* 卡在立面前时每足离地只有 0.00~0.03 m，法向力均值 12~70 N（峰值 200~320 N）。

⇒ 缺的不是抬腿高度，而是**摆动相里向前的落脚**。唯一有证据支持的下一步是显式落脚相
IK（触地时记抬脚点 → 求目标踏面 → 摆动相插值到落点 → 解算髋/膝角）。
**本仓没有实现这一步。**

### 3.6 观测历史长度与步态先验：各跑满 1,350,000 步的对照

> 这三个臂是在**旧六级课程**（1,350,000 步）上跑的，成绩记录保持当时口径不变。

三个臂都从零开始、跑完整当时的六级课程（8 环境，约 2,500~2,800 s/臂），
用同一个评测器测。**注意：这三个模型都只到 3~6 cm，远低于 §3.3 引用的历史能力带
（约 10 cm）** —— 那一带来自论文式 660 维配置 + 专门调过的超参，本仓默认
`config.yaml` 从零训练达不到，这一点必须诚实标出。

课程内成绩（成功率 / 摔倒率）：

| 阶段 | `ab_base`（先验开，10 帧） | `ab_hist1`（先验开，**1 帧**） | `ab_gaitoff`（**先验关**，10 帧） |
| --- | --- | --- | --- |
| flat | 100.0% / 0.0% | 100.0% / 0.0% | 0.0% / 0.0% |
| stair3 | 62.0% / 38.0% | **92.0% / 8.0%** | 0.0% / 0.0% |
| stair6 | 0.0% / 90.0% | **22.0% / 78.0%** | 0.0% / 0.0% |
| stair9 | 0.0% / 44.0% | 0.0% / 66.0% | 0.0% / 0.0% |
| stair12 | 0.0% / 16.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| stair15 | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

课程外（同一评测器，10 回合/档，`|a|` 是平均残差幅度）：

| 高度 | `ab_base` | `ab_hist1` | `ab_gaitoff` |
| --- | --- | --- | --- |
| 3.0cm | 50% / 0% adv 4.00m | **100% / 0%** adv 4.51m | 0% / 0% adv 0.04m |
| 5.0cm | 0% / 80% adv 1.64m | **100% / 0%** adv 4.54m | 0% / 0% adv 0.04m |
| 8.0cm | 0% / 100% adv 1.16m | 0% / 80% adv 1.04m | 0% / 0% adv 0.04m |
| 11.0cm | 0% / 0% adv 0.47m | 0% / 0% adv 0.49m | 0% / 0% adv 0.03m |
| `\|a\|` @3cm | 0.184 | 0.212 | **0.090** |

两条结论：

1. **单帧观测量可行，而且不亏。** `history_len: 10 → 1`（460 维 → 46 维，只剩 10%）
   在课程内**每一级都不差、3 cm/6 cm 还明显更好**（stair3 62%→92%，
   stair6 0%→22%），课程外 3 cm 和 5 cm 都是 100% vs 基准的 50%/0%。
   能力带两者都在 5~6 cm 之间。所谓「10 帧不能动」没有实测支撑 ——
   本任务的帧率（50 Hz）下，相邻帧几乎相同，10 帧里绝大部分信息是冗余的。
2. **步态先验是必需的，不是可选项。** `gait_prior: false` 当时那六级全 0、每回合跑满
   1000 步、`|a|≈0.09~0.12`，即**站着不动直到超时、一次都不摔**。原因是
   `action_scale = 0.30 rad` 是按残差设计的：先验提供步态，网络只补偏差。
   去掉先验后网络要同时从零学步态与平衡，在 ±0.30 rad 里做不到。真想去掉，
   必须先把 `action_scale` 放大到接近 gait 自身的 1.0 rad —— 那是另一次实验。

### 3.7 `body_collision` 降权 + 指令课程的正式训练（`runs/course1`）

本轮把 `body_collision` 从 `-4.0` 降到 `-1.2`、加入横向/偏航指令（观测 46 → 48 维）
与指令课程后，用默认 `config.yaml` 跑满 1,350,000 步（8 环境，wall 2514 s）。

| 阶段 | 成功率 | 摔倒率 | 超时率 | 均步数 | 均回报 |
| --- | --- | --- | --- | --- | --- |
| flat | 100.0% | 0.0% | 0.0% | 114.5 | +98.2 |
| stair3 | 44.0% | 56.0% | 0.0% | 379.7 | +900.7 |
| stair6 | 14.0% | 86.0% | 0.0% | 396.3 | +932.2 |
| stair9 | 0.0% | 90.0% | 10.0% | 479.0 | +665.2 |
| stair12 | 0.0% | 10.0% | 90.0% | 946.1 | +777.9 |
| stair15 | 0.0% | 16.0% | 84.0% | 939.2 | +877.5 |

课程外评测（`play.py --mode eval`，15 cm，100 回合）：成功率 0.0%、摔倒 7.0%、
超时 93.0%、平均前进 0.41 m、平均速度 0.02 m/s ⇒ **在 15 cm 上它选择几乎不动**。

**⚠ 这一节的数字不能单种子对比 —— 3 cm 单级的跑间方差极大。** 用同一份
`config.yaml`、只保留 3 cm 一级、跑 250,000 步，只换 `run.seed`：

| 运行 | seed | 3 cm 成功率 | 摔倒率 | 均步数 |
| --- | --- | --- | --- | --- |
| `course1` 的 stair3 级 | 0 | 44.0% | 56.0% | 379.7 |
| `runs/seed3cm_11` | 11 | **72.0%** | 28.0% | 376.3 |
| `runs/seed3cm_12` | 12 | **36.0%** | 64.0% | 243.3 |

⇒ 同一配置的 3 cm 成功率在 **36%~72%** 之间摆动（单级约 1,100 个回合、
250,000 步、8 环境）。所以：

1. **单次训练的 3 cm/6 cm 成功率差异（例如「80% → 44%」）不做结论。**
   要比较两个改动，必须在同一 seed 集合上跑多次，或至少只比较同一次运行内的
   前后级趋势。
2. 判据应以**视觉复核**为主（`play.py --mode gif` 出 GIF 后逐帧看姿态）：
   本轮 GIF 显示 3 cm 真的爬上去了（站在第 3~4 级踏面上、四腿交替）、
   6 cm 跨上第一级后躯干贴踏面翻倒、9/12/15 cm 基本停在第一级立面前。
   这与上一轮（`body_collision=-4.0`）的「撞两下就放弃」不同：**它现在会试，
   但仍然翻不过 9 cm 以上的立面**。
3. 想降低这个方差，只能加长每级预算或加宽环境多样性；`config.yaml` 里
   `run.seed` 是唯一的随机源，`curriculum.timesteps` 的现实下限约 250,000。

**逐帧看图的方法（本轮用的就是它）**：

```powershell
.venv\Scripts\python.exe scripts\play.py --run course1 --mode gif        # 出一档 GIF
.venv\Scripts\python.exe .tmp\gif_batch.py course1                       # 3/6/9/12/15 cm 各一档
.venv\Scripts\python.exe .tmp\gif_frames.py output\gifs\course1_9cm.gif --outdir .tmp\frames\9cm --n 6
```

`gif_batch.py` 把 `play.stage_height_cm` 在内存里改掉再写临时 yaml 调用 `play.py`，
产物落在 `output/gifs/course1_<高度>cm.gif`；`gif_frames.py` 按位置均匀抽 N 帧存 PNG。
**HUD 上的数字与画面不完全自洽（渲染节奏/回合切换导致），判据以画面姿态为准。**

---

## 4. 训练时的核对清单

> 改代码之前先读仓根的 [`agent.md`](agent.md)：那里有完整的验证门（5 支 `.tmp/` 校验脚本 +
> 2 个冒烟）与「改完之后必须跑什么」。

1. **观测维数**与 `summary.json` 的 `obs_config` 对得上（本仓 `history_len=1` ⇒ **48**；
   改回 10 帧是 480，开 `gait_phase` 再加 20，多一条指令就是本轮加的那 2 维）。
2. **`*_vecnormalize.pkl` 与 `*.zip` 成对**。缺了它确定性评估会从 100% 掉到 0%。
3. **`mean|a|` 是否在增长**。停在 0.02~0.10 就是退化成纯先验，继续跑也不会好。
4. **`train/explained_variance`**。接近 0 或负数说明 critic 没学到东西，先查奖励尺度。
5. **采样成功率 ≠ 确定性成功率**。日志里的成功率含动作噪声，交付前必须用
   `play.py --mode eval` 复测。
6. **至少 40 个回合**（`play.py` 默认 100）。
7. 看产品权重在**没训练过的台阶高**上的表现（泛化）比看训练台阶更有信息量。
8. **并发上限是 2 臂**（本机 15.7 GB）。4 臂 × 8 env 会报
   `engine error: Could not allocate memory`，那是内存不足，**不是地形几何问题**。
9. **判成果要落到画面上**（§2.1）：先 `--mode gif --batch <批次>` 逐阶段渲染，再看
   `runs/<名字>/gifs/<批次>/` 里的 GIF。日志里成功率涨、画面上趴着蹭，就是没学会。
10. **单次训练的成功率差在 ±18 个百分点以内不做结论**：同配置同步数、只换
    `run.seed`，3 cm 成功率实测在 36%~72% 之间（`experience.md` §14）。

> `scripts/train.py` 写完日志后，`post_process` 之类的旧入口**已删除**；渲染是
> 独立的一步（`scripts/play.py`），不要在训练脚本里夹带渲染。

