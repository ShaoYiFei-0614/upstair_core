# experience.md — 关键错误与修正

这份文件收纳**踩过的坑、错过的判据、以及为什么默认值是现在这个值**。
每条都尽量给出「现象 → 根因 → 修法 → 实测证据」。纯配方与环境配置见
[`README.md`](README.md)，训练过程与复现结论见 [`train.md`](train.md)。

标记约定：**[实测]** = 本仓当场跑出来的数字；**[原项目]** = 从完整仓
`D:\项目\RobortDog\upstair` 的历史记录（README/COURSES.md/summary.json）里读到的。

---

## 1. 环境与物理侧

### 1.1 踏步立面与顶面共面接缝把脚夹死

* **现象**：8 cm × 8 级只有 1/10 成功；9.0 cm 是 0%。
* **根因**：顶层平台与最后一级踏面共面时，两个 box 沿最小穿透轴反向推脚，
  z 向没有支撑 ⇒ 脚被夹在接缝里。实测 8 cm×8 上两只前脚冻结在
  `x = 5.500 == up_end_x`，持续 700 步，离终点只差 0.114 m。
* **修法**：`robotdog/terrain.py` 的 `CREST_SEAM_OVERLAP = 1.0`（单位：踏面），
  顶层平台向最后一级回伸一个踏面。
* **效果**：修后 8 cm 从 1/10 → 8/10；9.0 cm 从 0% → 65%。**[原项目]**

### 1.2 地形 geomgroup 被 MuJoCo 默认隐藏

* **现象**：狗在「爬空气」，HUD 却报出高度。
* **根因**：Unitree 的 MJCF 占用了 group 2/3，而 `mjv_defaultOption` 默认隐藏
  group 3/4/5。地形若落在这些组里，**物理存在、渲染不可见**。
* **修法**：`TERRAIN_GEOM_GROUP = 1`，且 `env.render()` 内部强制打开 6 个
  geomgroup（否则 GIF 里楼梯会消失）。

### 1.3 零高度台阶无法编译

* **现象**：`demo_stairs_spec(step_height=0.0, n_up=8)` →
  `RuntimeError: MJCF compile failed`。
* **根因**：零高度 box 是非法几何体。
* **修法**：平地一律用 `flat_spec(n_up=0, n_down=0)`，不要用零高度的
  `demo_stairs_spec`。

### 1.4 摔倒判据用了阶跃函数 ⇒ 幽灵摔倒

* **现象**：爬高台阶时无故判死。
* **根因**：老判据的高度参照是「机身 x 处的**单点**地形高」
  （`terrain.py` 的 `height_at`，阶跃函数）。机身跨过立面的一瞬间参考面跳整整
  一个台阶高：实测样本 `base_x=1.503 → Δref=+0.1100` 紧接着
  `base_x=1.491 → Δref=-0.1100`，逐帧翻转。
* **修法**：`terminate_height_ref="support"`（改用四足足下最低地形高）。

---

## 2. 奖励设计侧

### 2.1 惩罚项无上限 ⇒ 单步毁掉整个 value function

* **现象**：一个课程阶段的 value function 被单步污染，之后再也学不动。
* **根因**：脚顶在立面棱角上时 MuJoCo 的接触力无界，平方惩罚出现过
  `O(10^5)` 的单步值。
* **修法**：`touch_excess_clip = 5.0`，把两个平方触觉惩罚**归一化后的超量**
  逐步截断。教训推广：**没有哪一项奖励是可以安全地不设上限的**。

### 2.2 高斯前进奖励在 vx=0 处不为零 ⇒ 学会站桩

* **现象**：12 万平地步之后策略速度 0.00 m/s、总共走了 0.02 m。
* **根因**：`exp(-err²/σ²)` 在静止时仍付 `0.22 × forward_vel`，加上存活奖励
  构成了「站着不动」的局部最优 —— 零动作站桩的得分**高于**学出来的策略。
* **修法**：新增线性项 `forward_progress`（`vx=0` 时恰为 0，梯度不消失），
  并把 `alive` 压到 0.12（实测 `alive=0.4` 时它占站桩回合回报的 49%）。

### 2.3 抬升目标写死 0.08 m ⇒ 在 8 cm 处完全饱和

* **现象**：course13 卡 7.3 cm（4%）、course14 卡 7.8 cm（8%），而 6.4 cm 能到 80%。
* **根因**：`lift = clip(clearance/0.08, 0, 1)` 在 8 cm 已经饱和，再抬一毫米
  `foot_lift` 与 `stair_probe_lift` 都不多给一分。纯开环先验更极端：
  4 cm 40%、5 cm 10%、≥6 cm **任何 gain（1.0/1.3/1.6/2.0）都是 0%**。
* **修法**：目标跟着地形走，`有效目标 = max(foot_lift_target, 1.4 × 台阶高)`。
  `ratio=1.4` 时 5 cm → `max(8, 7) = 8 cm`，与旧行为**逐位相同**，所以已经学会的
  easy 阶段不会被改坏。
* **衍生坑**：`ratio=1.4` 外推到 19 cm 要求 26.6 cm 离地，已超出腿的几何极限，
  梯度退化成常数推力而不是「再抬一点」。⇒ `foot_lift_target_cap` 可以把目标压回
  可达区。

### 2.4 「腿碰地就立刻判死」把试错也一起杀掉

* **现象**：course12 在 10 cm×8 上平均只活 62.6 步，从未越过 `x = 1.42 m`，
  而第一级立面在 `x = 1.50 m`。
* **根因**：小腿蹭一下立面就终局，策略在**一只脚都没踏上任何一级**之前就死了，
  没有任何可学的信息，于是正确地收敛到「干脆别试」。
* **修法**：改成有界的逐步惩罚 `leg_collision = -0.25`，保留「别蹭楼梯」的梯度，
  但不再掐掉课程需要的高度试错。

### 2.5 蹭立面惩罚与任务方向打架

* **现象** 在 10 / 10.5 / 11 cm 上，`riser_graze` 分别占 5.7% / 13.1% / **31.9%**
  的控制步，平均罚 -0.035 / -0.079 / **-0.192** 每步 —— 11 cm 上这一项**超过**
  `foot_lift` 的全部实得（+0.129/步）。
* **根因**：物理上要爬上 11 cm 立面，脚/小腿**本来就必须先蹭到它**。
  奖励与任务方向相反。**[原项目 signal_audit]**
* **修法**：`riser_graze_mode="lifting"` —— 只有**所有**蹭立面的脚都没有明显向上
  速度（真卡住）时才罚；正在抬起跨越立面的脚免罚。另有
  `riser_graze_probe_gate`：只在**前方测距确实看到台阶**时才罚。
* **为什么需要 probe_gate**：蹭立面的检测只看接触法向的世界 z 分量，**平地**上
  小腿/髋与地面的边角接触也会被算成蹭立面。平地段恰恰是整个课程里唯一一开始
  就能拿满分的档，却一直吃着一个与任务无关的常数惩罚。

### 2.6 落地力阈值是常数 ⇒ 高台阶上从提醒变成乱罚

* **现象**：11.0 cm 上 `touch_impact` 吃到 -0.187/步。
* **根因**：落地法向力天然随落差增长，而 `touch_impact_target` 是常数 95 N。
  同样的好落地，台阶从 5 cm 涨到 11 cm 就越过阈值。
* **修法**：`touch_impact_step_scale`，口径
  `ref = 95 × (1 + scale × (step_height/0.05 - 1))`。`scale=0` 时 5 cm 台阶下
  因子恒为 1.0 ⇒ **与历史行为逐位相同**；`scale=0.5` 时 11 cm 的目标力是 152 N。

### 2.7 `stair_progress` 奖励「躯干冲线」而不是「脚上了一级」

* **现象**：11 cm 上「躯干过线、脚没过线，然后摔」。
* **根因**：该奖励引用**机身 x 处的单点阶跃高度**，躯干一越过第 k 级立面，
  之后**每一步**都白拿 `stair_progress × k × h`。11 cm 第 8 级 = +2.2/步，
  是 `alive` 的 18 倍，也超过 `forward_vel` 的满值 2.5。
* **修法**：`stair_progress_foot_ref=True` 改用**四足足下最低地形高度**（与
  `terminate_height_ref="support"` 同一个量），再叠加
  `stair_progress_min_loaded_feet`（至少 N 只脚法向力超过 5 N）把「躯干冲线」
  这条旁路堵死。

### 2.8 走直线项权重太小 ⇒ 形同不存在

* **现象**：`straight_line=-2.0` 时整项只占 200 分回合回报的约 0.65%，
  10 万步策略与无 shaping 基线无法区分。
* **根因**：只罚速度不够。`lateral_vel` 可以在均值上接近 0，而 20 s 回合里
  横向位置 y 仍然一路走远。
* **修法**：`straight_line=-20.0`（`|y|=0.2 m` 时约占回报一半），并用
  `straight_line_max=1.00` 把最坏情况限住。
* **反噬（必须记住）**：这个权重在 3.9M 步的全课程里被前序阶段吸收掉了，
  但**从零训练**时会成为主导项 —— 见 §3.1。

---

## 3. 观测设计侧

### 3.1 策略被打分，却看不到被打分的量

* **现象**：「走直线」这条 shaping 怎么调都没有威力。
* **根因**：reward 里 `lateral_vel`、`lateral_vel_track`、`yaw_rate`、`heading`
  四项都是纯 vy/wz 的函数，但 660 维观测里**根本没有 vy 和 wz**。策略只能靠
  关节速度和 10 帧历史去反推。
* **修法**：`obs.base_lin_vel_body`（机身系线速度，3 维）。世界系版本
  `base_lin_vel` 是差特征：yaw 一转它的含义就变了。

### 3.2 漂移量按「出生位姿」量 ⇒ 奖励保持随机朝向

* **现象**：策略完美保持出生位姿，却被判成在漂移/没走直。
* **根因**：`lateral_offset`/`heading_error` 原口径是对**本回合出生位姿**取差。
  而 reset 会把 yaw 均匀撒在 ±0.10 rad、y 撒在 ±0.12 m ⇒ 一个原地不动的策略
  也会「横向偏 0.16 m、朝向偏 5.7°」。
* **实测**：平地 30 回合，`|y|max` 均值 0.071 / 最坏 0.159 m，
  `|yaw - yaw0|` 均值 2.97° / 最坏 7.28°；但「末 20% 平均 yaw 减去首 20% 平均
  yaw」= **-0.53°**，即机器人**根本没有在弯**，它只是在忠实地保持出生朝向。
* **修法**：改按**楼梯中线** `stair.center_y` 与**楼梯轴**（yaw=0）量。

### 3.3 `base_height` 的参照面也是阶跃的

* 同 §1.4 的机制，`base_height = base_z - terrain_height_under_base` 里那一项
  会在躯干跨过立面时**单帧跳 0.11 m**，并在躯干于立面附近来回震荡时**每一步
  反复翻转**。
* **修法**：`obs.base_height_support` 改用 support 口径，无跳变。
* **⚠ 只能动观测，不能同时接到奖励上**：把它同时接到奖励会让均回报从 +554
  崩到 **-674/回合**。所以 `obs.base_height_support` 被刻意做成**只动观测的
  单变量开关**。

### 3.4 残差策略不知道自己处在步态周期的哪一点

* **根因**：叠加在周期先验之上的残差策略，若无法判断相位，就不可能是最优的。
* **修法**：`obs.gait_phase`（sin/cos 两维）。

---

## 4. 课程与训练侧

### 4.1 ⭐ 课程「毕业」判据把 PPO 的梯度更新饿死（策略输出 ≈ 0）

* **现象**：策略成功率和纯先验**一模一样**。course7 的 easy4_final
  （3.5 cm × 5 级）：policy 75.0% vs zero 75.0% vs random 25.0%，
  `mean|a| = 0.0109`、`|a| < 0.05` 占 **99.9%**。
* **根因**：毕业条件只看 `n_total >= min_episodes and success_rate >= target`，
  **没有「最少步数」概念**。有开环先验时，未训练策略第一轮就达标，`learn()`
  立刻返回 False 进入下一档。PPO 一次更新的粒度是 `n_steps=256 × n_envs=8
  = 2048` 步，而 course7 各档实际步数是 easy1 1824（**0 次更新**）/
  easy2 2272（1 次）/ easy3 2752（1 次）/ easy4 3456（1 次）。
  `log_std_init=-2.5` ⇒ 确定性均值 ≈ 0 ⇒ 策略输出 ≈ 0，完全解释了
  「策略 ≡ 纯先验」。
* **修法**：`StageSpec.min_timesteps`（毕业条件变成三者同时成立）；
  `scripts/train.py` 新增 `--min-stage-frac F`（默认 0.5）；tensorboard 新增
  `curriculum/stage_steps`。
* **正面证据**：course7 的 easy5（4.0 cm × 6 级）是唯一没立即毕业、拿到约 17 次
  更新的档位，成功率 40~52% **高于**纯先验 25~37.5%。course8 的 easy5_final
  （4.6 cm × 7）：policy 45.0% vs zero 20.0% ⇒ **高出先验 25 个百分点**。
* **附带纠正**：course6 的「先验 62% ≈ 策略 62%」也是步数太少，**不是**收敛到
  「别碰先验」的局部最优。

### 4.2 VecNormalize 的奖励归一化：温和的稳定器，不是解药

* **背景**：`training.py` 默认 `norm_reward=False`，因为课程在阶段间切换地形，
  在 5 cm 上学到的奖励尺度到了 11 cm 会漂移。
* **风险**：横向惩罚主导每步奖励（`straight_line` 实测占 72.7%，而
  `forward_vel` 只有 +0.356），挣扎中的策略回报接近 -6000，critic 拟合不动。
* **修法**：`make_vec_env(..., norm_reward=...)` + `--norm-reward`（默认关）。
* **实测（3 cm 台阶，8 环境，150k 步，双方各用自己的归一化器）**：

  | arm | 成功率 | `mean|a|` | 11 cm |
  | --- | --- | --- | --- |
  | off | 0.474 | 0.070 → 0.063 | 无成功 |
  | on  | 0.539 | 0.080 → 0.083 | 无成功 |

* **结论**：**温和的稳定器，不是解药。** 两侧在 11 cm 上都没有成功。
* **⚠ 撤回记录**：本节曾有一个「三倍残差增长、11 cm 摔倒率 0.88 → 0.12」的版本，
  那些数字来自一个后来被发现**会报错回合**的探针（见 §5.1），已全部撤回。

---

## 5. 测量与工具侧（最贵的几个坑）

### 5.1 探针报错回合：`probe()` 与独立评测不一致

* **现象**：同一模型、同一 seed、同一配置，`mini_train.probe()` 报
  `succ=0.00 adv=0.23`，而独立脚本报 `succ=True adv=4.636`（真值 4.636）。
* **已排除**：配置差异（逐字段 diff `total field diffs: 0`）、归一化
  （打印 `raw mean=-0.029 std=0.515 → z mean=-0.162 max=2.48`，与参考
  `obs_rms` 一致）、VecNormalize 复制（`mean[:3]` 与源一致）、模块加载路径
  （`inspect.getsourcefile` 正确）。把同一循环**逐行内联**进脚本就正确，
  紧接着调用真 `probe()` 仍错。**根因未定位。**
* **后果**：所有经旧 `probe()` 得出的数字**全部作废**。
* **修法（重要）**：不要试图修那个函数，而是**收敿成唯一共享实现**——
  新增 `rollout_once(model, norm, height_cm, seed, new_obs, extra)`，
  `probe()` 只做聚合。设计原则：**探针与所有独立评测必须调用同一个 rollout，
  这样 harness bug 不可能让两方不一致。**

### 5.2 估值脚本的单位 bug：cm 当成了 m

* **现象**：`adv` 老是「卡在 0.5 m」。
* **根因**：`evaluate()` 里 `demo_stairs_spec(step_height=height, ...)`，而 CLI
  传的是 **cm**（5/8/11）⇒ 实际在测 500/800/1100 cm 的台阶（顶端 40~88 m）。
* **修法**：`step_height=height / 100.0`。
* **教训**：所有 「adv 卡住」的读数在修这条之前都是垃圾。

### 5.3 打开观测归一化却没跟上权重 ⇒ 策略退化成随机

* **现象**：5 cm 上从 100% 掉到 0%。
* **根因**：策略是在**归一化观测**上训练的，喂原始观测等于随机输入。
* **修法**：`z = clip((x - mean)/sqrt(var + eps), ±clip)`，
  `eps=1e-8`、`clip=10.0`。

### 5.4 `pickle.loads(vecnormalize.pkl)` 的两个坑

* 返回的是 **VecNormalize 对象**，不是 dict。`data["obs_rms"]` 会
  `TypeError: 'VecNormalize' object is not subscriptable`。
  `model.predict(vec_obj, ...)` 会 `TypeError: 'VecNormalize' object is not callable`。
* 任何多余属性访问都会触发 `RecursionError: maximum recursion depth exceeded`
  （`BaseVecEnv.__getattr__` → `getattr_depth_check` 递归）。
  **修法**：立刻把 `obs_rms.mean/var`、`epsilon`、`clip_obs` 拷进一个普通命名
  空间，然后 `del raw`，此后只用那个命名空间。

### 5.5 并行 run 数上限 ≈ 3

* **现象**：6 个 run（各 8 环境）之后，一批 job 全部瞬间 `exit code 1` 且无
  traceback；前台重跑同一命令完全正常。
* **根因**：内存耗尽。崩出来的 traceback 是
  `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.29 MiB for an
  array with shape (512, 660) and data type float32`，位置在
  `stable_baselines3\common\buffers.py:514` 的 `self.observations[batch_inds]`。
* **教训**：并行 run 必须 ≤3，否则失败的不是学习而是内存。

### 5.6 后台 run 的 stderr 会丢失

* 用 `| Tee-Object` 只存 stdout，traceback 行号会指向**正在被编辑的文件**，
  极具误导性。
* **修法**：用 `*> file` 重定向，两个流都留住。

### 5.7 10 回合的评估能给出 ±15 pt 的错觉

* 原项目踩过：10 回合看是 90%，40 回合只有 65%。
* **修法**：验收统一用 `tools/ab_pair.py`，**40 回合 × 6 档、所有策略共用同一批
  种子**，这样「同一回合基线成、变体败」可以直接配对计数。

### 5.8 其它小坑

* `--resume` **静默忽略** `--net-arch` / `--learning-rate` / `--target-kl` /
  `--n-epochs`（只在从零分支生效）。要改只能在从零那一次写进去。
* **不要改踏面宽度**：`--tread` 保持 0.50 m。换成 0.40 / 0.60 会让 34% 掉到 10%，
  0.65 m 直接 0%。
* `summary.json` 的 `reproduce` 字段**分两代**：course31 之后原样回显真实命令行，
  可信；course22 只有 5 个参数，是有损的。
* `model_best.zip` **不是总存在**：只有「毕业过至少一个阶段」的 run 才会写它。
  阶段内最优用 `artifacts/best_<stage>.zip`。
* Windows 的 `>` 重定向会写成 UTF-16，事后读会判成二进制。用
  `Out-File -Encoding utf8`。
* 沙箱禁止命名管道 ⇒ 只能用 `DummyVecEnv`（单进程），不能用 `SubprocVecEnv`。
* `model.policy(tensor)` 的返回值**不是二元组**（`ValueError: too many values to
  unpack`），要用 `model.predict(z, deterministic=True)`。
* 沙箱下的 `venv` 创建：`python -m venv .venv` 可能因 `ensurepip` 失败而
  `No module named pip`；退路是 `--without-pip` + `get-pip.py`。
* 沙箱下 PowerShell/curl 的 HTTPS 可能被 TLS 层阻断
  （`schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS`），
  但 **Python 的 urllib/ssl 正常** ⇒ 需要下载时用 Python。

---

## 6. 能力边界：11 cm 的墙（本仓实测结论）

这一节是本仓**当场做出来的**结论，用来纠正「墙是奖励问题」的直觉。

> ⚠ **代码状态**：本节记录的一整批实验件（观测块 `foot_lift_gap` /
> `foot_lift_deficit`，奖励开关 `foot_lift_clear` / `foot_lift_support_ref` /
> `foot_placement_reach` / `foot_lift_target_cap` / `foot_lift_goal_ratio` /
> `riser_graze_probe_gate` / `stair_probe_lift_use_advance`，以及落脚面口径
> `advance_clearance()` / `_step_surface_ahead()`）**已在最终精简中全部删除**，
> 因为实测没有收益（见 §6.5）。它们**不在当前代码里**；这一节保留的是测量结论，
> 不是可调选项。要恢复代码请看 `.tmp/knob_backup/`。

### 6.1 参照面口径错了 ⇒ 抬升信号在 11 cm 上是反的

* 旧 `foot_clearance` = 足尖减**正下方**地形高。台阶处这个参照面是阶跃的：
  脚在立面**正前方**时读到的是下一级，脚一越到立面上方，同一个离地高度会
  **凭空少算一个台阶高**。
* 于是旧口径在 11 cm 上给出的**绝对读数比 8 cm 还大**，与「11 cm 抬不起来」
  完全相反 —— 奖励里那个 0.6 分的连续项在 11 cm 上反而给得比 8 cm 多。
* **修法**：落脚面口径 `advance_clearance()`，参照面 =
  `max(脚当前位置, 前方 reach/2, 前方 reach)` 三处的踏面高，
  `reach = 足半径 + foot_placement_reach(0.10)`。**只用位置决定，与脚当前高度
  无关**，所以读数不会在跨过立面的那一帧跳变。
* **修后判别力（参考 course22 权重）**：

  | h (cm) | 8.0 | 9.0 | 10.0 | 10.5 | 11.0 |
  | --- | --- | --- | --- | --- | --- |
  | 新口径摆峰达标率 | 0.987 | 0.986 | 0.942 | 0.674 | 0.203 |
  | 实测成功率 | 100% | 100% | 90% | 60% | 10% |

  **同形。** 旧口径恒为 0.000。

### 6.2 「落地清零」把摆峰项变成恒 0

* `_advance_swing_clearance` 原本在**落地那一步就把峰值清零**，而
  `_compute_reward` 读的是清零后的值 ⇒ `foot_lift_clear` 恒等于 0。
* **修法**：落地这一步**不清零**，只把本步测量并进峰值；清零动作移到
  `_compute_reward` 结算完 `foot_lift_clear` **之后**。

### 6.3 「抬不起来」是错的 —— 先验的几何能力足够

* 纯先验（零残差）在**平地**上的摆动峰值实测：
  * 默认频率：**0.1124 m**
  * 频率 ×0.6：**0.1658 m**
  * gain 1.4 + 幅度 1.6 + 频率 ×0.6：**0.21 m**
* 而**参考策略**在 11 cm 上只抬到约 0.09 m ⇒ **先验有这个几何能力，策略没有用它。**
* 早先「摆动抬升恒定 ≈9 cm 且不随台阶高缩放」的说法由此被推翻。

### 6.4 但开环调参也翻不过去

* 扫了 18 组（gain 1.0/1.4/1.8 × 幅度 1.0/1.3/1.6 × 频率 ×1.0/0.6），
  **零残差下 9/11/13 cm 上没有一次成功**，最强的组只前进 1.5 m。
* **结论**：缺的不是峰值高度，而是**摆动相里的向前落脚** —— 脚抬得够高，却没能在
  落地前放到上一级踏面上。
* **正确的下一步**：显式落脚相 IK（触地记抬脚点 → 用 `_step_surface_ahead()` 求目标
  踏面 → 摆动相把足端插值到该踏面上的落点 → 解算髋/膝角），而不是继续调奖励权重。

### 6.5 「新奖励项」的一次诚实失败

* 用参考权重 warm-start，8 cm 微调，150k 步，三条对照：

  | 对照 | 8 cm 保持 | 11 cm |
  | --- | --- | --- |
  | 参考奖励 | 133k 步仍 100% | 0%（adv 1.44 → 0.88） |
  | 新奖励项 | 106k 88% → **133k 62% → 160k 25%** | 0%（ret 恶化到 -3267） |
  | 新奖励 + lr 3e-5 | 160k 75% | 12%（adv 2.56，最好） |

* 前 26624 步三条对照**逐位相同**，起点干净。之后新奖励在默认学习率下
  **把已经学会的 8 cm 弄丢了**。
* **根因**：`foot_lift_clear` 按**摆动峰值**给分，而峰值高不等于成功迈上去 ——
  脚可以抬得很高却落在立面前，照样拿满分。**它可能在奖励一个与任务无关的量**，
  并且与参考奖励同时生效时把稳定的策略推离了工作点。
* **处理**：没有去调权重让它好看。这条与 §6.4 指向同一件事。
