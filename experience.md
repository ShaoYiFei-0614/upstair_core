# experience.md — 关键错误与修正

本文件记录**踩过的坑**：现象 → 根因 → 修法。默认值之所以是现在这个值，理由基本都在这里。
只写结论和机制，不写调试过程。代码实现见 `README.md`，训练结论见 `train.md`。

---

## 1. 环境与物理侧

### 1.1 踏步立面与顶面共面接缝把脚夹死

**现象**：8 cm × 8 级楼梯只有 1/10 回合能上，9.0 cm 是 0%。

**根因**：脚球心正好落在两级踏面的共享平面上时，两个 box 沿最小穿透轴反向推它，
z 向却没有任何支撑 ⇒ 卡死。实测两只前脚在 `x = 5.500 == up_end_x` 冻结 700 个控制步，
离终点只差 0.114 m。

**修法**：`robotdog/tools/terrain.py` 的 `CREST_SEAM_OVERLAP = 1.0` —— 顶层平台向最后
一级**回伸一个踏面**，让接缝不再落在立面棱上。修前 8 cm 只有 1/10，修后 8/10；
9.0 cm 0% → 65%。

### 1.2 地形 geomgroup 被 MuJoCo 默认隐藏

**现象**：狗在爬空气，HUD 却报高度。

**根因**：`mjv_defaultOption` 默认隐藏 geomgroup 3/4/5，而 Go2 的 MJCF 占用了
group 2/3。

**修法**：地形一律用 `TERRAIN_GEOM_GROUP = 1`；`render()` 里强制打开全部 6 个组。

### 1.3 零高度台阶无法编译

**现象**：`demo_stairs_spec(step_height=0.0, n_up=8)` 编译失败。

**根因**：生成的 box 是 `size="0.250000 1.000000 0.000000"`，MuJoCo 抛
`ValueError: Error: size 2 must be positive in geom`（`robotdog/go2_model.py` 把它包装成
`RuntimeError: MJCF compile failed` 并 dump 到 `.generated/failed_scene.xml`）。

**修法**：平地走 `flat_spec(n_up=0, n_down=0, step_height=0.0)`（`n_up=0` 时不生成
任何台阶 box），**不要**用「8 级零高度台阶」表示平地。

### 1.4 摔倒判据用了阶跃函数 ⇒ 幽灵摔倒

**现象**：机器人站得稳稳的却突然判摔。

**根因**：`terrain_height_under_base` 是**阶跃函数** —— 实测 `base_x=1.503` 时
`Δref=+0.1100`，紧接 `base_x=1.491` 时 `Δref=-0.1100`，逐帧正负翻转。

**修法**：`terminate_height_ref = "support"`，参照面改成**四足足下地形的最低点**；
另有 `"support_mean"`（四足足下均值）与 `"base"`（旧口径）。

---

## 2. 奖励设计侧

### 2.1 惩罚项没有上限 ⇒ 单步毁掉整个 value function

**现象**：某一步的回报是 O(10^5) 量级，训练直接崩。

**根因**：`touch_excess_clip` 之类的裁剪缺失时，单个控制步的惩罚可以任意大。

**修法**：所有惩罚项都要有上限；`clip` 的常数要按「物理量纲」定，不要按「希望它多痛」定。

### 2.2 高斯前进奖励在 vx = 0 处不为零 ⇒ 学会站桩

**现象**：平地训 120k 步，前进速度 0.00 m/s，策略学会站着不动。

**根因**：`exp(-((vx - 0.55)/σ)²)` 在 `vx = 0` 时仍有约 0.24 的值，而站桩能白拿
`alive` 与「无碰撞/无侧移」的全部分数。

**修法**：两条一起上 ——
* `forward_progress` 线性项 `clip(vx, -0.5, 1.5)`，**vx = 0 时恰为 0**；
* `alive` 从 0.4 降到 0.12（0.4 时它占「站桩」回合回报的 49%）。

### 2.3 抬升目标写死 0.08 m ⇒ 在 8 cm 处完全饱和

**现象**：course13 卡在 7.3 cm（4%）、course14 卡在 7.8 cm（8%），而 6.4 cm 能到 80%。

**根因**：`foot_lift_target = 0.08` 是常数，8 cm 以上所有台阶得到的抬升分数完全一样
（都饱和），策略收不到「还要更高」的信号。

**修法**：`foot_lift_step_ratio = 1.4` —— 有效目标 `= max(0.08, 1.4 × 台阶高)`。
5 cm 时等于 0.08，与旧行为逐位相同。

### 2.4 「腿碰地就立刻判死」把试错也一起杀掉

**现象**：course12 在 10 cm 上平均 62.6 个控制步就结束，从未越过 `x = 1.42`
（第一级立面在 1.50）。

**根因**：小腿擦到台阶就终止回合，策略连「撞一次、学一次」的机会都没有。

**修法**：腿接触改成**逐帧小惩罚** `leg_collision = -0.25`，不终止；只有躯干接触才是
致命的（且要连续 `terminate_contact_steps` 步）。

### 2.5 蹭立面惩罚与任务方向打架

**现象**：脚在立面上蹭既被罚「蹭」，又被要求「往前」。

**根因**：`riser_graze` 只看脚的法向力方向，分不清「顶着立面推进」和「贴着立面抬高」。

**修法**：`riser_graze_mode = "lifting"` —— 只有**所有蹭立面的脚都没有向上速度**
（`vz <= riser_graze_vz_eps = 0.10 m/s`）时才罚。**注意**：该整项已随「盲走」重构删除。

### 2.6 落地力阈值是常数 ⇒ 高台阶上从提醒变成乱罚

**根因**：`touch_impact_target = 95 N` 是常数，高台阶上正常落地的冲击也超阈值。

**修法**：`touch_impact_step_scale` 让参考力随台阶高放大。
**注意**：触觉相关项已随「盲走」重构全部删除。

### 2.7 `stair_progress` 奖励「躯干冲线」而不是「脚上了一级」

**根因**：参考面取躯干正下方地形，躯干一到立面上方就拿到整级进度分，
于是策略学会「把躯干往前扑」而不是「把脚踩上去」。

**修法**：`stair_progress_foot_ref = True` 改用足端参照面，并加
`stair_progress_min_loaded_feet` 要求至少 N 只脚受力。**注意**：这两项已随重构删除，
现在 `stair_progress` 回到躯干口径（见 `README.md` §3.3）。

### 2.8 走直线项权重太小 ⇒ 形同不存在

**根因**：`straight_line = -2.0` 时它只占 200 分回报的 0.65%，策略根本不理它。

**修法**：提到 -20.0。**但要注意反噬**：它和 `heading_max` 的裁剪合起来会把横向漂移
变成一个**截断后饱和的巨大惩罚**，实测能占全部惩罚量的 99.9%，把策略压回「几乎不动」。
本仓已把整组侧向奖励**删除**，改由 `forward_progress` 与 `base_height` 承担方向约束。

---

## 3. 观测设计侧

### 3.1 策略被打分，却看不到被打分的量

**现象**：奖励里有四个纯 `vy`/`wz` 的函数项（横向速度、航向误差等），策略却上不去。

**根因**：观测里**没有** `vy`/`wz`，策略无法知道自己正被扣分。更糟的是老判据把
「横向漂移」按**出生位姿**量，等于让策略奖励自己保持随机朝向。

**修法**：先把被扣分的量加进观测；漂移改成按**楼梯中线**量。**注意**：整组侧向奖励
已随重构删除，现在观测里也相应不再有漂移量。

### 3.2 漂移量按「出生位姿」量 ⇒ 奖励保持随机朝向

**修法**：按楼梯中线 `stair.center_y` 与楼梯轴量，而不是按出生时的 `y0/yaw0`。
平地上这个量能被验证成 0：末 20% − 首 20% 的航向差**只有 -0.53°**。

### 3.3 `base_height` 的参照面也是阶跃的

**根因**：与 §1.4 同源 —— `terrain_height_under_base` 在立面处逐帧翻转。

**修法**：观测侧加 `base_height_support` 开关，改用足下参照面。
**注意**：**不要把同一开关接到奖励上** —— 实测均回报会从 +554 崩到 -674/回合。
观测与奖励必须能用独立的单变量开关分别验证。

### 3.4 残差策略不知道自己处在步态周期的哪一点

**根因**：`gait.radians(t)` 是周期函数，策略要在它之上做残差，却看不到相位。

**修法**：`gait_phase` 观测块给 `sin/cos` 两个值。

---

## 4. 课程与训练侧

### 4.1 ⭐ 课程「毕业」判据把 PPO 的梯度更新饿死（策略输出 ≈ 0）

**现象**：策略成功率和「零动作纯先验」一模一样。

`scripts/diagnose.py policy` 同地形同 seed 三条基线（course7，3.5 cm × 5 级，20 回合）：

| 基线 | 成功率 | `mean|a|` | `|a| < 0.05` 占比 |
| --- | --- | --- | --- |
| 策略 | 75.0% | **0.0109** | **99.9%** |
| 零动作（纯先验） | **75.0%** | 0 | — |
| 随机 `U(-1,1)` | 25.0% | 0.4998 | — |

**根因**：毕业条件只有「回合数够 + 成功率够」，**没有最少步数**。有开环先验时未训练
策略第一轮就达标，`learn()` 立刻返回，每档只拿到 0~1 次梯度更新
（PPO 更新粒度 = `n_steps 256 × n_envs 8 = 2048` 步；course7 各档实际 1824 / 2272 /
2752 / 3456 步 = 0 / 1 / 1 / 1 次更新）。而 `log_std_init = -2.5` ⇒ 确定性均值 ≈ 0
⇒ 策略输出 ≈ 0，完全解释了「策略 ≡ 纯先验」。

**修法**：毕业条件加第三项 `min_timesteps`，由 `--min-stage-frac`（默认 0.5）算成
`预算 × F`。第一条正面证据是 course8 的 easy5（4.6 cm × 7）：策略 45.0% vs 纯先验
20.0% vs 随机 5.0%，`mean|a| = 0.0893`。

**注意**：本仓重构后**没有** `min_timesteps` 毕业门（六级全跑满固定预算），这个坑的
教训保留在这里。

### 4.2 VecNormalize 的奖励归一化：温和的稳定器，不是解药

同配置 3.0 cm 台阶、8 环境、150k 步的诚实 A/B：

| arm | 回合 | 平均回报 | 成功率 | `explained_variance` | 5 cm `mean\|a\|` |
| --- | --- | --- | --- | --- | --- |
| `norm_reward=False` | 822 | -567.8 | 0.474 | 0.2163 | 0.0698 |
| `norm_reward=True` | 790 | -546.9 | 0.539 | -0.2363 | 0.0803 |

⇒ 3 cm 成功率 0.474 → 0.539，残差幅度 0.070 → 0.080。**两侧在 11 cm 上都没有成功。**
所以本仓把 `norm_reward` 默认设为 **True**（有帮助），但不要指望它解决高台阶。

> ⚠ 早期草稿里曾引用过一组「0.025→0.033 / 0.88→0.12」的数字，那来自一个后来被发现
> **报错回合**的探针（见 §5.1），已撤回。

### 4.3 课程切换不能在 rollout 中间做

**根因**：换地形要重编译 MJCF（约 250 ms/子环境），在 rollout 中间换会让当次观测与
状态不一致。

**修法**：`Curriculum._on_step` 返回 `False` 只结束 `learn()`；真正切阶段由
`scripts/train.py` 在返回之后做，并且逐级 warm-start（`model.set_parameters(上一级权重)`）。

---

## 5. 测量与工具侧（最贵的几个坑）

### 5.1 探针报错回合：`probe()` 与独立评测不一致

**现象**：同一个模型、同一 seed，`probe()` 报 `succ=0.00 adv=0.23`，独立脚本报
`succ=True adv=4.636`。

**排查**：配置逐字段 diff 为 0、归一化打印正确、`inspect.getsourcefile` 指向正确文件、
同进程内两条路径结果不同 —— 根因**始终没有定位**。

**修法（比找根因更重要）**：把探针的内层循环替换成**唯一的共享 rollout 函数**，
`probe()` 只做聚合。设计原则：**探针与所有独立评测必须调用同一个 rollout**，
这样 harness 的 bug 就不可能让两边不一致。

**代价**：这次事故让此前一大批 `probe()` 数字全部作废。

### 5.2 估值脚本的单位 bug：cm 当成了 m

**现象**：所有高度的前进距离都「卡在 0.5 m」。

**根因**：CLI 传的是 cm（5/8/11），脚本里直接
`demo_stairs_spec(step_height=height)` ⇒ 实际在测 **500 / 800 / 1100 cm** 的台阶
（顶端 40~88 m）。

**修法**：`step_height=height / 100.0`。**单位换算只做一次，并且写在边界上。**

### 5.3 打开观测归一化却没跟上权重 ⇒ 策略退化成随机

**现象**：5 cm 上成功率从 100% 掉到 0%。

**根因**：策略是在**归一化观测**上训练的，直接喂原始观测等于喂垃圾。

**修法**：`*_vecnormalize.pkl` 必须与 `*.zip` 成对使用。`play.py` 自动找同名 pkl，
找不到会打印警告。

### 5.4 `pickle.loads(vecnormalize.pkl)` 的两个坑

* 得到的是 **`VecNormalize` 对象**，不是 dict ⇒ `data["obs_rms"]` 报
  `TypeError: 'VecNormalize' object is not subscriptable`。
* 该对象**多访问一个不存在的属性就 `RecursionError`**
  （`BaseVecEnv.__getattr__` → `getattr_depth_check` → `_get_all_attributes` 递归）。
* 它**不可调用** ⇒ `model.predict(vec_normalize_obj, ...)` 报
  `TypeError: 'VecNormalize' object is not callable`。

**修法**：立刻把 `vec.obs_rms.mean/var`、`vec.epsilon`、`vec.clip_obs` 拷进一个普通
命名空间对象，然后 `del vec`，此后只用那份拷贝（`robotdog/policy.py` 的 `ObsNorm`
就是这么做的）。

### 5.5 并行 run 数上限 ≈ 3

**现象**：6 个 run（各 8 环境）并行时集体瞬间退出（exit code 1，无 traceback），
前台重跑同一个命令却完全正常。

**根因**：内存耗尽。traceback 是
`numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.29 MiB for an array with shape (512, 660) and data type float32`，
发生在 `stable_baselines3/common/buffers.py:514`。

**修法**：并行 run 压到 ≤ 3；并且**用 `*> file` 而不是 `| Tee-Object` 启动后台 run**
（后者只存 stdout，stderr 的 traceback 会丢，行的行号还会指向正在被编辑的文件，极具误导性）。

### 5.6 后台 run 的 stderr 会丢失

`Tee-Object` 只捕获 stdout；且 stderr 的行号会错乱指向**正在被编辑**的文件。
一律 `*> .tmp/<name>.log`。

### 5.7 10 回合的评估能给出 ±15 pt 的错觉

**修法**：评估至少 40 个回合。`play.py --mode eval` 默认 100。

### 5.8 其它小坑

* `python -m venv` 可能因 `ensurepip` 失败而建不出 pip；用 `--without-pip` + `get-pip.py`。
* 受限环境里 curl / `Invoke-WebRequest` 的 HTTPS 可能被 TLS 挡住，但 Python 的
  `urllib` 正常 —— 下载用 Python。
* `pwsh` 的 `>` 重定向会写成 **UTF-16**（首字节 `0xff`），`ast.parse` 会报
  `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0`。
  要 UTF-8 就用 `cmd /c "git show ... > file"` 或 Python 的 `Path.write_text`。
* `pwsh` **不支持 heredoc**：`python - <<'PY'` 报 `Missing file specification after
  redirection operator`。要把 Python 落成 `.py` 文件再跑。
* `model.policy(tensor)` 的返回值**不是二元组**（`ValueError: too many values to unpack`），
  要取动作就用 `model.predict(x, deterministic=True)`。
* `EnvConfig` 里**没有** `stair` 字段，`dataclasses.replace(cfg, stair=...)` 必炸。
* 编辑代码块时 `edit` 的 `old_string` 只覆盖一部分 ⇒ 收尾的 `"""` 会残留成语法错误或
  脏文本。**改完必须 `py_compile` 并读回确认。**

---

## 6. 重构（2026-10）：为什么现在是「盲走六级」

### 6.1 这次重构删掉了什么

* **触觉 / 测距 / 接触力整套逻辑**（原 `robotdog/tools/touch.py` 与 `env.py` 里 91 处
  引用）：目标是**盲走楼梯**，机器人只靠本体感觉 + 一个台阶高度标量。
* **全部侧向奖励**（`lateral_vel`、`yaw_rate`、`straight_line`、`heading`、
  `lateral_vel_track`）：历史实测它们能占**全部惩罚量的 99.9%**，把策略压回「几乎不动」
  —— 逐项审计里 `straight_line` 一项就是 -12.67/步（72.7%），而唯一的前进奖励
  `forward_vel` 只有 +0.36/步。机器人明明能爬 2/3 段楼梯，却因为横向漂移 0.8~1.3 m
  被罚到 -17.4/步、-6622/回合。
* **域随机化**、**触觉相关的观测块**（`touch`、`stair_probe`、`drift`、`foot_height`）、
  **逐级晋级/回退的自动课程**（改成固定六级）、`render.py` 与 `evaluate.py`
  （GIF 内联进 `scripts/play.py`，评估就是 `play.py --mode eval`）。

### 6.2 一个真实的坑：subagent 重写核心模块

派后台 subagent 从零重写 `robotdog/training.py` 时，它交出的 784 行版本**在运行期直接炸**：

* `make_env_fn` 里写 `replace(cfg, stair=stair)` —— `EnvConfig` 没有 `stair` 字段；
* `Monitor(Go2StairsEnv(env_cfg, seed=seed))` —— 位置参数错（第一个位置参数是 `stair`）；
* `@dataclass class RunDir` 让 `RunDir('smoke')` 报
  `TypeError: RunDir.__init__() missing 4 required positional arguments`；
* 还有一大批没人调用的死代码。

**教训**：派 subagent 写核心模块时，不能只看它自报的验证输出，必须**逐条运行你自己
脚本里的调用路径**。它自己的四条验证里有一条根本没过。最后 `training.py` 由我重写。

同时还有一个并发写作的教训：subagent 在我编辑同一批文件时报告「有别的 writer 在抢」——
那是**我本人**。多 writer 场景下要么串行、要么明确停掉一个，不要让它自己去"修复"。

### 6.3 三个必须记住的重构原则

1. **能删就删**。用户明确允许破坏现有功能时，「保留实验性旋钮」的代价远大于收益：
   这些旋钮默认关闭、实测**没有收益**（新奖励项 `foot_lift_clear` 在默认学习率下把已经
   学会的 8 cm 从 100% 掉到 25%），却要每个读者都去理解它们。
2. **删之前先备份**。删掉的代码留在 `.tmp/knob_backup/`（含一份 `git diff` 补丁）。
3. **删之后要证明「行为不变」**。做法：用 `importlib` 以私有模块名加载改前的文件，
   在同一串固定动作下逐项比较 reward / obs / 终止标志 / `reward_terms`。
   实测 12 组全部 `max|Δreward| = 0.000e+00`、`max|Δobs| = 0.000e+00`。

---

## 7. 配置化：把超参从命令行搬进 `config.yaml` 之后踩到的坑

用户要求「不要让过多参数可调，相机和 GIF 渲染参数完全可以固定；PPO 等关键超参用
`config.yaml` 配置，不要用 CLI 输入」。改完之后的四个坑：

### 7.1 ⭐ 重建模型必须用**训练时那一套网络结构**

`load_policy()` 原本按默认 `DEFAULT_NET_ARCH = (512, 256)` 建网再
`set_parameters(checkpoint)`。只要训练时用的不是默认结构（比如配置里写
`ppo.net_arch: [64, 64]`），加载时必炸：

```
RuntimeError: Error(s) in loading state_dict for ActorCriticPolicy:
    size mismatch for mlp_extractor.policy_net.0.weight: copying a param with shape
    torch.Size([64, 460]) from checkpoint, the shape in current model is
    torch.Size([512, 460]).
    size mismatch for action_net.weight: copying a param with shape torch.Size([12, 64])
    from checkpoint, the shape in current model is torch.Size([12, 256]).
```

修法：新增 `robotdog/policy.py:policy_arch(model_path)`，从 SB3 `.zip` 的 `data` 条目里读
`policy_kwargs.net_arch`（**是普通 JSON** `{"pi": [...], "vf": [...]}`，不需要解
`":serialized:"` 的 base64 pickle），`load_policy` 用它建网：
`arch = policy_arch(model_path) or DEFAULT_NET_ARCH`。

这是「配置化」放大出来的问题：结构以前写死在两个地方，恰好一致；现在配置能改它，
两处就必须真的对齐。

### 7.2 `RunDir.__init__` 里 eager `mkdir` 会让「目录已存在」的保护永远为真

`train.py` 想拦住「重复用同一个 `--run-name` 覆盖上次结果」：

```python
rundir = RunDir(name)
if rundir.root.exists():
    raise SystemExit(f"输出目录已存在：{rundir.root}")
```

但 `RunDir.__init__` 自己就把 `root/checkpoints/tensorboard/artifacts` 建好了 ⇒
**第一次运行就报「输出目录已存在」**，且退出时留下四个空目录（第二次运行照样报）。
修法：`__init__` 只算路径不建目录，新增 `ensure()` 供调用方在检查之后显式创建；
`save_model` / `write_summary` / `CheckpointCallback` 自己 `mkdir(parents=True)`。

### 7.3 推理时只建 1 个环境 ⇒ SB3 的 mini-batch 警告

`batch_size=512` 与 `n_steps × n_envs` 不整除时 SB3 在 `PPO.__init__` 里 warn：

```
UserWarning: You have specified a mini-batch size of 512, but because the
RolloutBuffer is of size `n_steps * n_envs = 256`, after every 0 untruncated
mini-batches, there will be a truncated mini-batch of size 256
```

训练时是 8 环境（2048，整除）；**推理时 `load_policy` 只建 1 个环境**（256，不整除），
于是每次 `play.py` 都刷一条。对只做 `predict` 的场景这条警告没有意义，在 `build_ppo`
里用 `warnings.catch_warnings()` + `filterwarnings("ignore", message=".*mini-batch size.*")`
压掉。

### 7.4 地形默认值必须**每个 checkpoint 各记一份**

`play.py` 原本在「没显式给 `--stage-height-cm`」时用 3 cm 默认楼梯。但训练产物里每条
`checkpoints/<阶段>_final.zip` 是在**不同台阶**上训出来的（只有最后一级是 15 cm），
用错地形评估等于在测一个它没学过的高度。

修法：训练时每个阶段写一份 `<模型>_env.json`（`policy.save_env_json`），里面
`report.step_height` 记下该权重的地形；`play.py` 用
`step_height_from_model()` 读它，并打印「地形来自权重自带记录」。找不到时（老权重）
才退回默认值 —— 这一点写进了 `README.md` §5 的已知限制。

---

## 8. 已知但**故意不改**的注释

* `robotdog/go2_model.py` 的注释说 `HOME_JOINT_ANGLES` 来自 `go2.xml` 的
  `<keyframe name="home">`。vendored XML 实际是
  `<keyframe><key name="home" qpos="..."/></keyframe>`（`<keyframe>` 没有 name 属性），
  且 `extract_meta()` **从不读 keyframe** —— `home_qpos` 纯靠硬编码常量，无运行期校验。
  **改成读 keyframe 会改变 `home_qpos` 的取值 ⇒ 改变 reset 与所有已训权重的前提**，
  所以只记录不动。
* `extract_meta()` 的 `floor_geom_id` 走 `mj_name2id` 且**不检查 `< 0`**（其余 id 都走
  `_resolve_id` 抛 `KeyError`）；`_OPTION` 正则只认自闭合 `<option .../>`，
  改成非自闭合会**静默失效**。
