# experience.md — 关键错误与修正

本文件记录**踩过的坑**：现象 → 根因 → 修法。默认值之所以是现在这个值，理由基本都在这里。
只写结论和机制，不写调试过程。当前代码结构见 `README.md`，各模块设计见 `doc/`；
文中提到的 `docs/`、`train.md` 与部分 `robotdog/` 符号是**当时的**结构与名称
（本文件是历史记录，与现状冲突时以当前代码为准，见 `agent.md` §1）。

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

> **后续（见 §12）**：`forward_progress` 那一条**实为错修**。真正的病根是
> `forward_vel` 的高斯分母漏了平方（`forward_vel_sigma` **当时**是标准差，却**被**直接当分母用），
> 等于按 `sqrt(0.25)=0.5 m/s` 折算，`vx = 0` 时仍白拿 `exp(-1.21) ≈ 0.30` ⇒ **+0.72/步**
> （实测 +0.86/步）。把分母改成 `σ²` 后只剩 +0.02/步，而线性伴项在 0.55 m/s 附近
> 反而给出 +1.1/步的超线性加成，于是连同 `forward_progress` /
> `forward_progress_max` / `forward_progress_min` 三个字段一并删除。本节的记录保留
> 为历史，但**不要再按它去调线性项**。

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

> **后续（见 §12）**：那半句「躯干接触致命 + 连续 10 步去抖」**也已删除**。实测那条
> 规则从未生效过（所有驱动实验的最长连续躯干接触只有 8 步），而且随时可能误杀：
> 躯干在 `h ≈ 0.28 m` 的完全安全姿态下贴立面蹭就能连续到 8 步，距离阈值只差 2 步。
> 现在 `_contact_fatal()` 恒返回 `False`，躯干接触的代价只由 `body_collision`
> （逐步、有界）承担；终止只剩高度、倾角与超时三条判据。

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
本仓已把整组侧向奖励**删除**，改由 `forward_vel` 与 `base_height` 承担方向约束
（`forward_progress` 那一版见 §2.2 的引用块，已在 §12 里连同字段一起删除）。

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

> **注意（架构已变）**：`Curriculum` 类与 `StageTrainer` 协议已在死代码清理中删除 ——
> `scripts/train.py` 从来就是自带 `for` 循环直接调 `Trainer.train_stage`，那个驱动器是孤儿。
> 现在的「不中途切地形」是靠 `Trainer.train_stage` 每级只 `learn()` 一次固定预算来保证的，
> 不再是回调返回 `False` 的机制。见 `docs/robotdog/curriculum.md` §3.1。

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

### 5.9 ⭐ 同一个权重，两套结论：`VecNormalize` 统计量没装回模型

**症状**：对同一个 `runs/ab_hist1/model_final.zip`，两条评测路径在 3 cm 上给出相反
答案 ——

| 路径 | 3 cm 结果 |
|---|---|
| `scripts/play.py --mode eval`（自己驱动裸 `Go2StairsEnv` + `ObsNorm`） | 成功率 100%、摔倒 0%、前进 4.54 m |
| 自写脚本走 `model.get_env()` → `VecEnv.step()` | `succ=0.00 fall=1.00 adv=3.17m steps=347.8` |

一度怀疑是地形构造、seed、`info` 键名或探针本身有 bug（历史上已发生过一次探针报错
回合）。**真因在 `robotdog/training.py: load_policy()`**：它读 `*_vecnormalize.pkl`
只为了**返回**一个 `ObsNorm` 给调用方，却**没有把统计量装回模型自带的那个
`VecNormalize`**。`model.set_parameters()` 只还原网络权重，`build_ppo` 建出来的
`VecNormalize` 仍是初始的 `mean=0 / var=1`。于是：

* 裸 env 路径：`ObsNorm` 手动归一化 ⇒ 策略看到真统计；
* `model.get_env()` 路径：包装器还是 `mean=0 / var=1` ⇒ **策略看到未归一化的观测**。

**定位手法（可复用）**：并排打印两条路径**重置后的前 6 维原始观测与归一化观测**。
分叉一眼可见 ——

```
raw obs0 A  [-0.0006  0.0068 -0.0087 ...]   ← 空
raw obs0 B  [-0.058   0.9376 -1.7505 ...]   ← 真实关节角
norm obs0 A [-0.0006  0.0068 -0.0087 ...]   ← 归一化等于没做（mean=0 / var=1）
norm obs0 B [ 0.2344 -3.8073  0.4135 ...]   ← 真统计
```

**修法**：`load_policy` 在 `set_parameters` 之后调一次
`restore_vecnormalize(venv, cand)`（`robotdog/training.py` 里这个函数早就写好、也
在 `__all__` 里，**但此前零调用者**）。修完两条路径一致。

**顺带踩到的第二个坑**：`.tmp/path_cmp.py` 里 `make_env_fn(stair, None, seed=seed)`
传了 `cfg=None` ⇒ 建出默认 `history_len=10`（460 维）的环境，而权重是
`history_len=1`（46 维），报 `ValueError: operands could not be broadcast together
with shapes (1,460) (46,)`。**跑权重必须用它自带的 `EnvConfig`**
（`env_config_for_model(model_path)`）；另外 `make_env_fn` 返回的是**零参工厂**，
必须再调一次：`make_env_fn(...)()`。

**教训**：评测器与训练器必须共用同一个「建环境 + 归一化」实现。任何**手搓**的
评测路径都要先和仓库自带脚本对一次数，否则 `|a|`、成功率、摔倒率全都不可信。

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

---

## 9. 死代码清理：哪些能删、哪些看起来能删但必须留

本轮清理用的判据是**「这个符号有没有任何读者」**，用 `.tmp/dead3.py` 按 token 流做全仓引用计数（排除定义行）。它一次报出 13 个「零引用」候选，但**真正该删的只有 6 个**——剩下 7 个每一次都落进下面三类**假死**里。

### 9.1 三类假死（扫描器永远看不出来）

| 类型 | 例子 | 为什么不能删 |
| --- | --- | --- |
| **字符串派发** | *（本仓目前没有实例）* | `venv.env_method("某个方法", ...)` 把名字藏在字符串里，AST 看不见。这是最容易误删的一类，改 `DummyVecEnv` 相关代码时留意 |
| **框架回调** | `ProgressCallback._on_step`（`scripts/train.py:96`）、`CheckpointCallback._on_step`（`scripts/train.py:130`） | SB3 的 `BaseCallback` 按固定名字回调，从不被本仓代码显式调用 |
| **文档化的公开 API** | `GaitParams.static`（`robotdog/gait.py:264`）、`GaitParams.targets`（`robotdog/gait.py:139`）、`policy.Policy`（`robotdog/policy.py:21`）、`StairSpec.down_end_x`（`robotdog/tools/terrain.py:108`）、`Go2StairsEnv.yaw`（`robotdog/env.py:641-642`）、`Go2StairsEnv.set_stair`（`robotdog/env.py:401-414`） | 设计文档把它们的签名与理由逐条写下来了，是给人用的接口面。删掉等于同时删两处文档，而它们各只有 1–5 行 |

**留下的判据**：`GaitParams.static()` 是文档里「怎么关掉先验」的**唯一正确写法**（`docs/robotdog/gait.md:434` 的示例直接 `off = GaitParams.static()`），所以哪怕零调用者也要留——删了文档就在教一个不存在的函数。

**`set_stair()` 的处境更微妙，值得单独记**：它**曾经**有一处真实调用者——`robotdog/training.py` 的 `set_terrain(venv, stair)`（函数体就是 `venv.env_method("set_stair", stair)` + `venv.reset()`）。而 `set_terrain` 自己是零读者，于是被删；删掉它之后 `set_stair` 才变成零调用点。**当前课程根本不用换地形**：`scripts/train.py` 每一级都调 `trainer.make_env(stage)` 新建整个向量环境。所以这条链的真相是「**一个零读者的包装器是另一个符号的唯一调用者**」——删之前必须**看两跳**，否则会把「被间接调用」误判成「被间接需要」。

> **修正一条曾经的错误结论**：本轮清理的早期，`experience.md` 与 `README.md` 都把 `set_stair` 归类为「字符串派发，调用点在 `robotdog/training.py:99` 的 `venv.env_method("set_stair", stair)`」。**那是错的**——全仓 grep `env_method` 与 `set_stair` 只剩定义行与注释，`training.py:99` 是 `build_ppo` 的 `seed: int = 0,` 形参。两个符号当时**都已经删了**。教训：**「字符串派发」这个类别要靠 grep 证明，不能靠印象归类**。

### 9.2 真正删掉的 6 个

| 符号 | 位置 | 为什么它真的死了 |
| --- | --- | --- |
| `load_ppo(path, venv, *, device)` | `robotdog/training.py` | `load_policy` 是唯一实际入口；`load_ppo` 是 `PPO.load()` 的薄包装，没有任何调用者 |
| `mean_abs_action(actions)` | `robotdog/training.py` | 三行的诊断量（`float(np.mean(np.abs(a)))`），训练循环从不统计它；只有临时脚本会算 |
| `set_terrain(venv, stair)` | `robotdog/training.py` | 函数体就是 `env_method("set_stair")` + `reset()` 两行，没有任何调用者。**删它之前看过两跳**：它自己是零读者，但它是 `set_stair` 的唯一调用者，所以删完 `set_stair` 也变成了零调用点（那个保留了，见 §9.1） |
| `GO2_SCENE_XML` | `robotdog/tools/paths.py` | 本仓从不用官方 `scene.xml`（场景由 `compile_scene()` 现场拼 XML）；`scene.xml` 只在 `scripts/download_go2.py:43` 的 `WANT_FILES` 里以**字符串字面量**出现 |
| `ensure_dir(path)` | `robotdog/tools/paths.py` | 零调用者，且自带两个坑（传 `str` 抛 `AttributeError`、对文件路径会建出同名目录） |
| `ensure_output_dirs()` | （已删除） | 零调用者；`scripts/play.py:248` 的 `write_gif` 自己 `path.parent.mkdir(...)` |

顺带删掉三个**失去全部使用者**的 import：`robotdog/training.py` 的 `numpy`（`mean_abs_action` 是唯一使用者）、`robotdog/policy.py` 的 `Sequence`、`scripts/download_go2.py` 的 `sys`（末尾写的是 `raise SystemExit(main())` 而不是 `sys.exit`）。

### 9.3 教训：符号存废是**人工核对项**，行号检查永远抓不到

删掉 `mean_abs_action` 之后，`robotdog/training.py` 从 350 行变成 **323 行**，于是文档里 6 处指向该文件第 39 行（`RUNS_DIR` 与 `StairSpec` 的 import）与一处 `:295-297`（`RunDir.summary`）的引用全部指到了空行或错误位置。

`.tmp/check_lines.py` 抓到了**全部 7 处**——但它抓到的是**症状**（「这一行是空行」），不是**病因**。真正的原因是「源码少了一行，所有下游行号整体上移」。这意味着：

> **每次删函数、加函数，都必须重跑 `check_lines.py`。** 它报的每一条「是空行」都在说「上游源码行数变了」。而它**永远**不会报「这个函数不存在」——`ensure_dir` 在源码里被删掉之后，文档里那 20 多处提及它依然「行号合法」（因为文档写的是散文，没有行号）。

### 9.4 顺带发现：LF/CRLF 在两种工具之间来回翻转

本仓约定 **CRLF**（`core.autocrlf=true`）。用 `write` 工具整篇重写文件时，输出是 **LF**，于是：
* `git status` 会警告 `LF will be replaced by CRLF the next time Git touches it`；
* 更隐蔽的是**行尾不一致本身**——同一个文件里既有 LF 又有 CRLF 时，某些编辑器会显示成「整篇都改了」。

实测有 11 个文件处于裸 LF 状态（`robotdog/env.py`、`robotdog/reward.py`、`scripts/train.py`，以及 8 份文档）。修法是 `.tmp/fix_eol.py`：把 `\r\n` 先归一成 `\n` 再统一展开成 `\r\n`，**只处理含裸 LF 的文本文件**。跑完复检 `still bare-LF: none`，且 `git diff --shortstat HEAD` 没有因为行尾而产生额外的行数变化。

**教训**：整篇 `write` 重写文件之后，要么用 `edit` 做增量修改（保留原行尾），要么立刻跑一次行尾归一化。

---

## 10. 行号漂移：`check_lines.py` 的第二个盲点

### 10.1 症状

给 `robotdog/env.py` 的 `set_stair()` 补了 5 行 docstring（说明「当前零调用点」），
`robotdog/env.py` 从 696 行长到 **701 行**。于是**它后面所有定义的物理行号整体下移 5 行**：

| 定义 | 改前 | 改后 |
| --- | --- | --- |
| `_compute_reward` | 344 | **349** |
| `_target_from_action` | 539 | **544** |
| `_gait_action` | 529 | **534** |
| `_contact_flags` | 379 | **384** |
| `_contact_fatal` | 418 | **423** |
| `_fall_reference_height` | 455 | **460** |
| `base_ang_vel_body` | 274 | **279** |
| `step` | 559 | **564** |

`check_lines.py` 只报了两条（说是「`robotdog/env.py` 的这一行是空行」）——因为它检查的是
**范围的两个端点**，而旧写法那两个范围的末尾恰好还落在下一个函数的 docstring 里。
其余十几处引用（`README.md`、`docs/robotdog/reward.md`、`docs/robotdog/gait.md` …里指向
`_compute_reward` / `_target_from_action` 的那些）**行号完全合法、指向非空行**，
只是指向了**别的函数**（旧数字 344 现在落到了 `_observation` 的 `def` 行上）。

### 10.2 病因

> **行号是「相对坐标」，只要上游插/删一行，下游全部失效。**
> `check_lines.py` 能抓的是「越界」和「空行」——那是**症状**；「漂移」没有任何症状，
> 因为漂移后的行号依然指向一个合法、非空的 Python 行。

### 10.3 修法（两个层次）

1. **引用写法**：文档引用代码改成 `` `robotdog/env.py` 的 `compute_reward()` `` 这种
   **文件 + 符号名**形式，行号只在确实要特指某一行实现时才写。符号名不会漂移。
2. **新增检查器 `.tmp/ref_truth.py`**：把全仓 `.md` 里每条 `文件.py:行号` 引用解析出来，
   打印它**实际指向的那一行内容**。判据三条：
   * 越界 → `OOR`；
   * 空行 → `BAD`；
   * **指向 `def`/`class` 行但没有预先登记** → `BAD`（因为裸数字指到定义行，几乎总是
     漂移后随手写下的数字——想指定义就该写符号名）。
   同时维护一张 `EXPECT` 白名单（「第 N 行应该含 `def xxx`」），合法引用就不会误报。
   本轮跑完：`refs checked: 210   suspicious: 0`。

**教训**：新增一支脚本的时候，要先问「它**不能**抓什么」。`check_lines.py` 从 CRLF 时代
就一直在报「是空行」，让人误以为文档行号是有守护的——其实守护的只是「越界」。

---

## 11. 观测的信息来源：`base_height` 从「默认关掉」改成「禁止打开」

### 11.1 症状

`obs.base_height` 一直是 `false`，但它是靠**默认值和一行注释**守住的。实地核对时发现：

```python
# 旧实现（robotdog/env.py 的 _obs_frame）
if obs_cfg.base_height:
    parts.append(np.array([self.base_pos[2] - self._height_reference()]))

def _height_reference(self) -> float:
    return self.terrain_height_under_base      # = stair.height_at(base_x)
```

也就是说，只要在 `config.yaml` 里把 `base_height` 改成 `true`，**机身正下方的地形几何
真值就进了观测**。本仓是盲走（blind locomotion）环境，这不只是「多一个特征」——
它把任务从「凭本体感觉猜台阶」变成了「直接读出地形高度」，训练出来的策略在真机上
没有对应的传感器。

### 11.2 真因：约定不是保证

「默认关掉 + 注释说明」是一种**约定**，它只对读注释的人有效。这个开关历史上被打开过
（`experience.md` §3.3 记录过一次「把它同时接进观测和奖励」的事故），说明约定失效的
代价是可以真实发生的。凡是「打开了就会破坏前提」的开关，都应该在构造时就拒绝。

### 11.3 修法

1. `ObsConfig.__post_init__` 在 `base_height=True` 时**直接抛 `ValueError`**，错误信息
   说明它为什么含真值、以及正确的替代做法（保持 `step_height`）。
2. 删除 `_obs_frame()` 里那条分支，并删除 `_height_reference()` 本身 —— 让
   「往观测里塞地形真值」**在结构上不可能**，而不只是「被禁止」。
3. `robotdog/policy.py` 的 `_obs_config_for_dim()`（按观测维度反推开关组合）把
   `base_height` 从枚举里去掉：任何会打开它的组合都不可构造。副作用是旧维数
   474 反推不出来（返回 `None` 而不是崩），这是刻意的——那些归档 checkpoint 本来
   就与本仓观测维数（当时 46，加指令后 48）不兼容。
4. `config.yaml` 的注释从「观测开关」升级为「**必须 false，且不能改成 true**」。

### 11.4 顺带做的一次全面审计（结论要保留）

把 `Go2StairsEnv` 里每个「看起来可能引入额外信息」的判断都过了一遍：

| 量 | 真值来源 | 进观测吗 | 结论 |
| --- | --- | --- | --- |
| `_contact_flags()` | MuJoCo 接触对（躯干/腿 × 地形） | 否，只进奖励与终止 | 可接受 |
| `_foot_contact_mask()` | MuJoCo 接触对（足端 × 地形） | 否，只进 `air_time` 奖励与腾空统计 | 可接受（真机有标准等价估计：力矩残差 / 电流尖峰 / IMU 冲击） |
| `_support_height()` / `_support_height_mean()` | `stair.height_at()` 地形几何 | 否，只进摔倒判据 | **特权信息**，但终止信号本来就是上帝视角，允许 |
| `foot_clearance` | 足端位置 − 地形高 | 否，只进 `foot_lift` 奖励 | **特权信息**，策略只能学到「按 `step_height` 抬够高」的相关性 |
| `base_height`（观测） | `terrain_height_under_base` | **是** | ⭐ 破坏盲走，已删除并禁止 |

结论：**只有 `base_height` 一项真正破坏了盲走前提**，其余四项都是「真值只进奖励/终止」
的标准做法。观测逐项都能落到真机传感器上（编码器 / IMU / 上一步动作 / 本体感觉 /
上层指令），唯一非传感器项是标量任务参数 `step_height`。（这段做审计时本仓是 46 维；
`experience.md` §13 之后加了 2 维指令块，现在是 48 维。）

### 11.5 教训

* **「默认关掉」不等于「不会开」。** 破坏性开关要用 `__post_init__` 报错封死，并且把
  实现路径一并删掉——留着代码就是留着被重新接上的可能。
* 审计「一个方法是否引入了额外信息」时，要区分两件事：**它读了什么**（可能是真值）与
  **它去了哪里**（观测 / 奖励 / 终止）。本仓允许读真值做奖励与终止，只禁止进观测。
* 这类审计是一次性的判断题，**不会**被任何自动检查器抓到（就像 §10 的行号漂移）。
  所以结论要写进文档，而不是只留在当次对话里。



## 12. 终止条件与奖励账：删掉「躯干接触致命性」，并修掉一个漏了平方的高斯

### 12.1 症状：站着不动比走完楼梯更赚

三个 A/B run（`runs/ab_base` / `ab_hist1` / `ab_gaitoff`）跑满默认六级课程后，成绩出现
一个反直觉的形态：`stair12` / `stair15` 上「摔倒率 0%、超时率 100%」，而
**平均回报 688~822**，比「平地成功走完 3 m」（回报 358）高一倍以上。

零动作分档结算（400 控制步上限）确认了这件事：

| 场景 | 步数 | 累计回报 | 每步 | 主要来源 |
| --- | --- | --- | --- | --- |
| `0 cm` 零动作 | 113 | **+334.5** | +2.960 | `forward_progress +204.6`、`forward_vel +61.5`、`terminal +50` |
| `8 cm` 零动作 | 400 | +90.6 | +0.226 | `forward_vel +356.7`、`body_collision −236.0` |
| `15 cm` 零动作 | 400 | **−245.2** | −0.613 | `body_collision −548.0`、`forward_vel +341.9` |

⇒ 高台阶上「原地站到超时」比「试着爬」更安全，且回报仍然可观。

### 12.2 真因：两个机制一起把「站着」变成局部最优

1. **`forward_vel` 的高斯分母漏了平方。** `RewardConfig.forward_vel_sigma = 0.25` 是**标准差**，
   但代码写的是 `exp(-(vx - command_vx) ** 2 / rc.forward_vel_sigma)`，等于按
   `sqrt(0.25) = 0.5 m/s` 折算。于是 `vx = 0` 时仍有
   `exp(-(0.55)² / 0.25) = exp(-1.21) ≈ 0.30` ⇒ **+0.72/步**（实测 +0.86/步）。
2. **`stair_progress` 是「每步都付」的。** 它读 `terrain_height_under_base - _start_terrain_height`，
   只要机身正下方是高台阶就一直拿钱，站着不动照样拿。

### 12.3 修法（三项，一次改完）

1. `forward_vel` 的分母改成 `rc.forward_vel_sigma ** 2`：`vx = 0` 时从 +0.72/步 降到 **+0.02/步**。
2. **删除 `forward_progress` / `forward_progress_max` / `forward_progress_min` 三个字段**及
   对应的线性项。σ 修正后高斯项在 `vx = 0` 处已经足够小；而线性项在 0.55 m/s 附近会给出
   **+1.1/步**的超线性加成，反而扭曲速度跟踪。`RewardConfig` 字段**当时从 24 减到 21**。
   （§2.2 把这条线性项当成过「站着不动」的修法，实为错修，已在 §2.2 加引用块更正。）
3. `stair_progress` 改成**一次性结算**：

   ```python
   climbed = max(0.0, env.terrain_height_under_base - env._peak_terrain_height)
   if climbed > 0.0:
       env._peak_terrain_height = env.terrain_height_under_base
   terms["stair_progress"] = rc.stair_progress * float(climbed)
   ```

   累计总量不变（仍是 `stair_progress × 爬升高度`），但站桩拿不到新钱，同一级往返也不会付两次。
   为此在 `__init__` 与 `reset()` 里各加一个 `_peak_terrain_height`。

### 12.4 顺带删掉「躯干接触致命性」：它从未生效过，而且随时可能误杀

用户提出的问题是：「躯干持续接触 10 个控制步（0.2 s）才算摔倒 —— 去抖，避免单帧擦碰
就判死是否合理？机身上台架天然要蹭上去。」

**测一（几何）**：躯干碰撞体 3 个，最低点在 `0.118 m`（机身下探 `0.2020 m`），而
`terminate_min_height` 原来是 `0.13` —— 比它还高 `0.012 m`。也就是说，躯干一碰地，
**高度判据早就触发了**（差 0.012 m ≈ 6 个物理步），这条接触规则在直立姿态下纯属冗余。

**测二（驱动实验，5/8/11/15 cm × forward/forward_fast/tilt/random × seed 0/1/2，共 48 回合）**：

* 躯干接触的**最长连续步数全场只有 8**，从未触发过 10 步的致命判据；
* 躯干接触恰恰发生在「高度完全安全」的时候 —— 「首次安全躯干」与「首次躯干」逐条相同
  （130/130、51/51、39/39 …），状态是 `h ≈ 0.24~0.28 m`、倾角 0.05 rad 贴着立面蹭，
  腿也在蹭。这正是「机身上台架天然要蹭上去」；
* 15 cm 上躯干连续接触最长到 4~5 步（random 档 8 步），**距离阈值只剩 2 步**。

**测三（反事实）**：把「安全」的高度阈值从 0.13 放宽到 0.10 与 0.08，最长「安全躯干接触」
连续步数**仍然是 8**（same case：15 cm / random / seed 3 / 第 59 步，`h = 0.277`、
倾角 `0.067`）。抬高安全余量也救不了：**「躯干接触」这件事本身与「摔倒」无关。**

**结论与改动**：

* 删掉 `EnvConfig.terminate_contact_steps` 与 `_effective_contact_steps()`，删掉
  `_trunk_contact_streak`；`_contact_fatal()` 重写为**恒返回 `False`**，只维护
  `_leg_contact_streak`（诊断用，永不致命）。保留方法是为了让 `_terminated()` 的调用形状不变。
* `info` 里 `terminate_contact_steps` 与 `trunk_contact_streak` 换成 `leg_contact_streak`（键数 18 → 17）。
* `terminate_min_height` **0.13 → 0.15**，换来 3.2 cm 余量（躯干最低点 0.118 m）。
* 躯干接触的代价继续由 `body_collision = -4.0/步`（逐步、有界）承担，**不由终止承担**。

### 12.5 修复后实测

```
    0cm 静止:  113步 累计    +95.9 每步 +0.849
    0cm 前进:  400步 累计   +272.9 每步 +0.682
   15cm 静止:  400步 累计   -573.9 每步 -1.435   ← 从 +0.226/步 翻转过来
   15cm 前进:  400步 累计   -178.6 每步 -0.446
```

15 cm 站桩从 **+0.226/步 降到 −1.435/步**（进步 −1.66/步），而前进 −0.446/步，**站桩不再优于前进**。

诚实记录两点：

* `0 cm 静止` 仍是 +0.849/步，因为 trot 先验在零动作下仍会推着机器人前进 3 m 并触发
  `terminal = +50` —— 那是**先验在走路**，不是「站桩」，不算设计缺陷。
* `body_collision` 在 15 cm 上仍达 **−1.37/步**（躯干蹭立面不可避免），所以高台阶可能
  仍是局部最优。这留作下一轮观察 —— 下一轮（§13）把它从 `-4.0` 降到了 `-1.2`。

### 12.6 教训

* **奖励项里出现「与前进无关却每步都付」的正项时，一定要逐项算它静止时的值。**
  `forward_vel` 在 `vx = 0` 处 +0.86/步 这个数，只有真的把 `reward_terms` 打出来才看得见。
* **标准差当分母用时要检查有没有平方。** 名字叫 `sigma` 的字段写成 `/ sigma` 是静默的
  3.8 倍放大（`exp(-4.84)` vs `exp(-1.21)`），没有任何报错。
* **阈值型终止判据要先量「真实分布的最长连续步数」再定阈值。** 8 与 10 之间只差 2 步，
  而这条规则从未生效过——它既不提供保护，又随时可能误杀。
* **用户的直觉值得直接测量。** 「机身上台架天然要蹭上去」一句话就指向了正确的诊断方向：
  先量接触时的姿态，再决定这条规则是否合理。

## 13. 指令化：`body_collision` 降权、横向/偏航指令与一个新造的「白拿正项」

### 13.1 症状：GIF 里它「试两下就放弃」

改完 §12 之后重训了一次全课程（`runs/rewardfix`，1,350,000 步），成绩是：

| 阶段 | 改后 `rewardfix` | 改前 `ab_base` |
|---|---|---|
| flat | 100% / 摔 0% | 100% / 0% |
| stair3 | **80% / 摔 20%** | 62% / 38% |
| stair6 | **10% / 摔 90%** | 0% / 90% |
| stair9 | 0% / 摔 86% / 超时 14% | 0% / 摔 44% / 超时 56% |
| stair12 | 0% / 摔 38% / 超时 62% | 0% / 摔 16% / 超时 84% |
| stair15 | 0% / **摔 100%** | 0% / **超时 100%** |

**站桩被杀掉了**（GIF 里没有任何高度出现「站着不动等超时」），但只有 3 cm 真正变好。
按用户要求逐档出 GIF 复盘（`output/gifs/rewardfix_{3,6,9,12,15}cm.gif`）后看到：

* **3 cm**：真的在爬 —— 前腿上一级、后腿下一级、重心前压，四腿交替踩踏面。
* **6 cm**：跨上第一级（前进 +1.86 m）后躯干越贴越低，进入「趴着用肚子蹭」的姿态然后翻倒。
* **12 cm**：前腿搭在立面上、后腿撑地，**速度只有 0.03 m/s**，三个回合两个超时。
* **15 cm**：三回合形态一致 —— 走到立面脚下（前进 +0.70/+0.77/+0.80 m）→ 前腿搭上去、
  后腿蹬 → 上不去 → 侧前方翻倒。

### 13.2 真因：`body_collision = -4.0` 让「蹭立面」比站桩还亏

`body_collision` 是逐步量，权重 `-4.0`。15 cm 上一项就能吃掉 **−548**（≈ −1.37/步），
占全部负值一半以上；6 cm 那帧「趴在踏面上蹭」也是同一项在罚。于是策略学到的最优解是
**试两下就放弃**，而不是继续顶。这与 §12 修掉的「站桩被奖励」是同一个病的两面：
只要某个每步都付的项与「继续尝试」负相关，策略就会停止尝试。

修法：`-4.0 → -1.2`。躯干接触**本来就不再终止回合**（§12.4），所以这项只是「不舒服」的
提醒，不该重到压过前进动力。

### 13.3 ⭐ 一次真实的自伤：不加门控的指令跟踪项又造了一个「白拿正项」

用户要求加入**横向移动与旋转指令**（外部控制器模拟人遥感操控）+ 指令课程。第一版
`compute_reward` 里直接加了两个键：

```python
terms["lateral_vel"] = rc.lateral_vel * exp(-(vy_err ** 2) / rc.lateral_vel_sigma ** 2)
terms["yaw_rate"]    = rc.yaw_rate    * exp(-(yaw_err ** 2) / rc.yaw_rate_sigma ** 2)
```

**实测（5 cm，125 步）**：`lateral_vel = +139.73`、`yaw_rate = +28.03`。
期望值本该是 `1.5 × 125 = 187.5` 与 `1.0 × 125 = 125` —— 也就是说，机器人**只是在直走**，
横向/偏航误差恰好落在 σ 附近，于是它凭空得到约 **+1.0/步**。这与 §12 修掉的
`forward_vel` 在 `vx = 0` 处的 +0.86/步是**完全同一类 bug**，我又犯了一次。

**修法：门控。** 只在指令范围真的放开时才计分，并把两项合并成一个键（这样「只往前走」
阶段的键集合与加指令之前逐位一致）：

```python
cmd = env.cfg.command
follow = 0.0
if cmd.vy_max:
    vy_err = env.base_lin_vel_body[1] - cmd.vy      # 机体系横向
    follow = max(follow, rc.lateral_vel * exp(-(vy_err ** 2) / rc.lateral_vel_sigma ** 2))
if cmd.yaw_max:
    yaw_err = env.yaw_rate - cmd.wz
    follow = max(follow, rc.yaw_rate * exp(-(yaw_err ** 2) / rc.yaw_rate_sigma ** 2))
terms["command_follow"] = follow
```

修后实测 `command_follow = +0.000e+00`（逐位精确 0）。

> **后续**：这两项最终**又改成了线性死区**（连同 `command_follow` 合并成一个键），高斯宽度
> `lateral_vel_sigma` / `yaw_rate_sigma` 也随之从 `RewardConfig` **删除**——`robotdog/reward.py`
> 里现在只有 `lateral_vel = 6.0` / `yaw_rate = 4.0` 两个权重。当前形式见 `README.md` §3.3。
> 上面的代码块是**当时**的样子，不要再照它写。

> **更后续（门控只对了横向，偏航是反的）**：门控把偏航项在 `yaw_max == 0` 的级别里整个关掉了，
> 而这个级别**没有任何别的项读 `wz`** —— 策略因此学会以 −0.608 rad/s **恒定自转**并把它冻住。
> 缩幅值先验的训练里，第 1 级（`yaw_max = 0`）跑完的检查点在 `wz ∈ [−0.6, +0.6]` 七档上实测
> `vx` 恒 +0.365、`wz` 恒 −0.608，**斜率 −0.0021**，也就是输出与偏航指令逐位无关；加上偏航指令
> 的后一级也解不开（偏航误差均值 0.555 rad/s vs 容差 0.50，动作权限实测是够的：`hip` 模式
> `u = ±0.6` 给 ∓0.16~0.19 rad/s）。现在的修法是**偏航项恒在**、量程取
> `max(cmd.yaw_max, reward.yaw_range_floor)`，横向仍门控。教训：**「门控」要按轴判断**，
> 一个「正确的门控」可能刚好把另一条轴的唯一梯度来源关掉；判据是「这个量在范围为零时还有没有
> 别的项读它」，不是「范围为零就该关掉」。见 `README.md` §3.3 与 `agent.md` §8.5。

### 13.4 第二个坑：`yaw_rate` 不是 `qvel[5]`

`data.qvel[3:6]` 是**机体系**角速度，不是世界系欧拉角速率。直接取 `qvel[5]` 当偏航角速度
是错的（`base_rpy` 是 `mj_objectState` 给出的世界系 rpy）。用「5000 组随机姿态 + 从旋转矩阵
`dR = [ω]× R` 反解真实偏航角速度」逐一检验，三条候选公式**全部失配**：

* `-sin(p)·wx + cos(p)cos(r)·wy + cos(p)sin(r)·wz` —— 500/500 失配
* `wz + (sin p / cos p)·wy` —— 500/500 失配
* `sin(r)/cos(p)·wy + cos(r)/cos(p)·wz` —— 也不是（只在个别点碰对）

**正确写法**（5000/5000 零失配）：

```python
wx, wy, wz = quat_to_mat(self.base_quat).T @ self.data.qvel[3:6]   # 先转世界系
roll, pitch, yaw = base_rpy
yawdot = wz + tan(pitch) * (cos(yaw) * wx + sin(yaw) * wy)
```

### 13.5 第三个坑：`env_config_from_summary` 必须重建嵌套 dataclass

`_env.json` / `summary.json` 往返之后 `env_config["command"]` 变成普通 `dict`。直接塞给
`EnvConfig` 会让 `env._command` 变成 dict，任何 `set_command_now` 调用当场炸：

```
AttributeError: 'dict' object has no attribute '_now'
```

修法：`robotdog/policy.py` 里 `if isinstance(env_kw.get("command"), dict):
env_kw["command"] = _rebuild(CommandConfig, env_kw["command"])`。**任何嵌套 dataclass 字段
过一遍 JSON 都需要这一步。**

### 13.6 两项改动互不影响的证明（用户明确要求）

1. **动力学完全没动**：固定动作序列在 12 个用例（平地/5/8/11 cm × seed 0/1/7）上
   **步数逐位一致**（127/138/122/125/140/210/180/300/300/124/128/300）。
2. **奖励差完全由 `body_collision` 解释**：`expected(-4.0) = sum(r) + n_contact × 2.8`，
   5 cm `125 步 body=-1.20 sum=+99.60 expected=+96.80`、8 cm `180 步 body=-14.40
   sum=+95.55 expected=+61.95`、11 cm `124 步 body=-3.60 sum=-7.26 expected=-15.66`。

### 13.7 教训

* **同一个错误可以犯两次。** §12 刚修掉「与前进无关却每步都付的正项」，§13 又手写了一个。
  加任何新的逐步奖励项之前，**先算它在「机器人什么都不做、只直走」时的值**。
* **可选的奖励项必须由配置门控。** `if cmd.vy_max:` 这一行不是优化，是正确性：
  它让「没开这个能力」与「加这个能力之前」在数值上完全等价。
* **别用印象给零调用者归类。** `set_stair` 一度被我写成「字符串派发」类型，实际全仓
  `env_method` 零命中 —— 真相是它唯一的调用者 `set_terrain()` 被删了。判据要靠 grep。
* **指标名不等于物理量。** `qvel[5]` 不是偏航角速度；`sigma` 没平方就是另一个标准差。
  这类错误不会报错，只会让训练悄悄变差。

## 14. 结论的统计强度：3 cm 单级的跑间方差高达 36%~72%

**症状**：`body_collision` 降到 `-1.2` 并加上指令（观测 46 → 48 维）之后，正式训练
`runs/course1` 的 stair3 只拿到 **44.0%** 成功率，而上一轮 `runs/rewardfix`
（`body_collision=-1.2`、无指令、46 维）在 3 cm 上是 **80.0%**。第一反应是「加指令
把 3 cm 能力弄丢了」。

**测法**：同一份 `config.yaml`、只保留 3 cm 一级、250,000 步、8 环境，只换 `run.seed`：

| 运行 | seed | 3 cm 成功率 | 摔倒率 | 均步数 | 回合数 |
| --- | --- | --- | --- | --- | --- |
| `runs/course1` 的 stair3 级 | 0 | 44.0% | 56.0% | 379.7 | 1,188 |
| `runs/seed3cm_11` | 11 | **72.0%** | 28.0% | 376.3 | 1,096 |
| `runs/seed3cm_12` | 12 | **36.0%** | 64.0% | 243.3 | 1,202 |

**结论**：**36%~72% 的摆动全部来自 `run.seed`。** 三个运行都跑满 250,000 步、
都是约 1,100 个回合，所以这不是采样噪声（回合数的标准误只有几个百分点），
而是「不同种子收敛到不同的局部最优」。

**教训**：

1. **本仓单次训练的成功率差异在 ±18 个百分点以内不做结论。** 说「改动 A 比改动 B
   好」之前，必须在**同一 seed 集合**上至少各跑 3 次。
   历史笔记里的单次对照（§13.1 的 `rewardfix` vs `ab_base` 表、`train.md` §3.6 的
   三臂表）都应这样读：**趋势可信，具体百分点不可信**。
2. **视觉复核是不可省的裁判。** 数字摆动这么大时，GIF 里的姿态（真的踩上踏面、
   还是趴在踏面上蹭、还是停在立面前不动）比任何单个成功率都更能说明问题。
3. **想压低方差只有两条路**：加长每级预算，或增加环境多样性。`config.yaml` 里
   `run.seed` 是唯一的随机源，`curriculum.timesteps` 的现实下限约 250,000。
4. **同一个 3 cm 单级只花约 7.5 分钟**，所以「多 seed 复现」这步在本仓是廉价的，
   没有理由跳过。

## 15. 「被阻挡时试探抬腿」的奖励：三次尝试与一次门槛标定

**目标**：机器人撞上够不着的立面之后，唯一能改变局面的行为是「把脚抬得比平时更高
再试」。要给这个行为一个奖励。

**尝试一 —— 检测链条根本没通。** 第一版把摆动峰值写成
`np.where(foot_contact, 0.0, np.maximum(...))`，**一接触地形就把峰值清零**；而结算
发生在落地那一帧（`landed = foot_contact & ~_prev_foot_contact`），读到的正是刚被
清零的旧值，比整个摆动期的真实峰值低 0.01~0.03 m。结果 500 步里 `probe_count = 0`、
`probe_lift = +0.000`，即使把门槛降到 0.02 m 也不触发。

> **修法**：清零必须发生在结算**之后** —— 落地那一帧保留旧峰值。
>
> ```python
> landed = foot_contact & ~self._prev_foot_contact
> self._peak_swing_clearance = np.where(
>     landed, self._peak_swing_clearance,
>     np.where(foot_contact, 0.0,
>              np.maximum(self._peak_swing_clearance, clearance)))
> ```

定位手段是临时的逐帧打印（`PROBE_DEBUG` 环境变量门控），打完就删。教训：**「有梯度
但没人走」和「根本没梯度」从回报上看不出区别**，必须把中间量打出来。

**尝试二 —— 门槛标定。** 用 `.tmp/calib_probe3.py` 量「被阻挡帧里的摆动峰值」：

| 地形 | 门槛（原设计 = `1.6 × 台阶高`） | 被阻挡帧峰值 均值 / p99 / max |
| --- | --- | --- |
| 9 cm（seed 0） | 0.144 | 0.0444 / 0.1698 / 0.1698 |
| 9 cm（seed 1） | 0.144 | 0.0317 / 0.1248 / 0.1280 |
| 12 cm | 0.192 | 0.0271 / 0.1054 / 0.1124 |
| 15 cm（3 个 seed） | 0.240 | 0.0254~0.0261 / 0.0959~0.1079 / 0.1086~0.1169 |

再用 `.tmp/calib_probe4.py` 搜「有没有任何固定动作能把脚抬到门槛之上」：
`action_scale=0.30` 下 7 种解析动作的最好值是 **0.1479 m**（`thigh +1 calf +1`）；
放大到 `action_scale=1.00` 时才有 `hip +1 thigh +1 calf +1` 的 0.2576 m 与 100 组随机
动作里的 0.5845 m。**结论：在 `action_scale=0.30` 下，15 cm 的 0.24 m 门槛不可达，
梯度恒为零。** 门槛只能跟「当前策略自己能抬多高」比，于是改成
`probe_lift_target = 0.06`（普通步态峰值带 0.06~0.12 的下沿）。

**尝试三 —— 改成势函数塑形。** 不再「落地帧越线给固定分」，而是「被阻挡期间的最高
摆动峰值又提高了多少就付多少」：

```python
if rc.probe_lift != 0.0 and env._blocked:
    peaks = env._peak_swing_clearance
    floor = max(env._probe_best, env.probe_target)      # probe_target = 0.06
    gain = float(np.sum(np.clip(peaks - floor, 0.0, None)))
    if gain > 0.0:
        env._probe_best = float(np.max(peaks))
        env._probe_count += 1
    terms["probe_lift"] = rc.probe_lift * gain
else:
    terms["probe_lift"] = 0.0
```

性质三条：**总量有界**（≤ `probe_lift ×（可达最高峰值 − 起算线）`）、**不可刷分**
（重复同样高度拿不到第二次）、**有梯度**（抬得更高就被记录，即使那一脚最终没踩上
去）。`_update_probe_state()` 在解除阻挡时把 `_probe_best` 清零，让下一次被卡住能重新
累积梯度。

**实测量级**（`.tmp/probe_gain.py`，`probe_lift = 1.0`，400 步回合）：

| 地形 | 零动作 | `thigh +1` | 40 组随机里最好 |
| --- | --- | --- | --- |
| 9 cm | 0.392 | 0.943 | 1.189（峰值 0.2793） |
| 15 cm | 0.141 | 0.073 | 0.909（峰值 0.2078） |

**教训**：
1. **奖励项要先做「可达性标定」再写权重。** 一个永远拿不到的奖励与一个权重为 0 的
   奖励，对训练是同一件事，但对调试者是完全不同的错觉。
2. **门槛不要跟「任务参数」走，要跟「当前策略的实际能力」走。** `1.6 × 台阶高` 在
   3~9 cm 上看着合理，到 15 cm 就变成了不可能条件。
3. **势函数塑形比「事件越线给分」安全**：它天然有界、天然不可刷分，不需要额外设计
   防刷机制。
4. 实测结论要如实说：**在 15 cm 上 `probe_lift` 的量级（≈0.9/回合）远不足以补上
   `body_collision` 的量级（≈−548/回合）**，它只是把「抬得更高」变成一个可优化方向，
   不是解药。

## 16. 偏航指令跟踪为什么学不出来：奖励读的量本身在抖

**症状**：指令课程训出来的策略**基本不转向**。`runs/course2` 的 `flat_final.zip`
在平地上被要求以 `wz = 0.60 rad/s` 转 3 秒（≈1.8 rad），实际只转了 0.31 rad，
而偏航通道的「平均绝对误差」却高达 0.735 rad/s。

**诊断（`.tmp/check_yaw_rate.py` / `check_yaw_rate2.py` / `check_yaw_rate3.py`）**：
平地、默认先验、零动作、seed 0、`control_dt = 0.02`，用三种互相独立的方法求
「逐帧姿态的真实偏航角速度」：

| 方法 | 与 `env.yaw_rate` 的偏差均值 | 与 `qvel[5]` 的偏差均值 |
|---|---|---|
| 逐帧有限差分 | 0.165 | 0.083 |
| ±2 步中心差分 | 0.256 | 0.157 |
| 四元数相对旋转反解（精确） | **0.283** | 0.156 |

即：真值 −0.0654、`env.yaw_rate` −0.0289、`qvel[5]` −0.0473 rad/s，
**偏差幅度 0.28 rad/s 与我们要跟踪的指令量程（±0.70）处在同一量级**。
偏差随 |roll|/|pitch| 单调增大（|倾角| ≥ 0.15 的帧上 0.615 vs qvel5 的 0.114）。

**根因（不是 bug，是「读错了量」）**：`env.yaw_rate` 实现的是**世界系 z 轴角速度**
（`wz + tan(pitch)·(cos(yaw)·wx + sin(yaw)·wy)`），而它被拿去和逐帧的 `yaw` 欧拉角
比较；两者的时间导数只在姿态为零时相等。四足步态下躯干以几十赫兹小幅摆动，
roll/pitch 的乘积项就能贡献 0.3 rad/s —— 正好把「平稳的偏航指令」淹没。

**为什么这对训练是致命的**：策略无法控制那个振荡分量，于是 `command_follow` 的偏航
通道里有一大半方差是它拿不到的；相对地，「干脆不转、把前进项拿满」反而是更稳的
局部最优。实测两回合的前进速度 1.30 / 1.34 m/s（指令 0.55，超调 2.4 倍）也印证了
这一点：策略把全部预算压在了前进上。

**建议的修法（尚未实施，需要用户确认，因为会改变奖励语义）**：把偏航读数换成
**机体系 z 角速度**

```python
wz_body = float((quat_to_mat(self.base_quat).T @ self.data.qvel[3:6])[2])
```

它同样振荡更小、计算更便宜，而且**更贴近真机**（机载陀螺直接给的就是机体系角速度，
真机不需要先做坐标系换算）。若保留世界系口径，则至少要意识到奖励里混入了
不可控的振荡项。

**教训**：
1. **「物理上正确的公式」不等于「适合当奖励信号的公式」。** `yaw` 欧拉角的导数与
   世界系 z 角速度在数学上都是「偏航速率」，但对一个高频摆动的四足躯干，前者抖动
   大一个量级。
2. **奖励项上线前要量它的噪声底。** 这一次噪声底（0.28）与信号量程（0.70）同阶，
   等价于把一半权重扔进了策略控制不到的通道。这和 §15 的「可达性标定」是同一类
   错误的两面：一个是奖励**拿不到**，一个是奖励**读不准**。

## 17. 三级平地预热把 3 cm 能力从 40% 打到 0%，以及「指令根本跟不上」的两层根因

**症状（m08247 的「做完后训练」，`runs/course3`，1,350,000 步、51 分钟）**：逐级
末 50 回合滚动成功率是 `flat 100% / flat_2 100% / flat_3 100% / stair3 0%（摔倒 86%）
/ stair6 8%（摔倒 92%）/ stair9 0%（摔倒 58%）`。而旧课程 `runs/course1`、`runs/course2`
（都是 1 级平地 + 无指令楼梯）在同一台机器上、用同一条确定性评估器测出来是
`3 cm 40% / 6 cm 0% / 9 cm 0%`。**3 cm 从 40% 掉到 0%。**

**先用证据排除掉三个「假嫌疑人」**：
1. 不是评估口径不同 —— course3 训练日志里 stair3 阶段自己统计的也是 0%。
2. 不是 `VecNormalize` 没装回去 —— 用与训练同源的 8 环境 + `VecMonitor` +
   `VecNormalize`（装回 `stair3_final_vecnormalize.pkl`）重测，3 cm 仍只有 11.1%。
3. 不是地形取错 —— 但这里踩到一个**真陷阱**：`load_policy(model_path)` 不给 `stair=`
   时会建 `demo_stairs_spec()` 的默认 **5 cm** 楼梯，而不是该 checkpoint 记录的 3 cm。
   任何独立评估都必须显式 `env.set_stair(demo_stairs_spec(step_height=h/100, tread=0.5, n_up=8))`
   （`.tmp/eval_run.py` 已经默认这样做了）。

**三臂消融（把「平地级数」与「楼梯段是否带指令」解耦）**：

| 臂 | 平地 | 楼梯段指令 | 3 cm 成功率 | 3 cm 摔倒率 | 3 cm 前进 |
|---|---|---|---|---|---|
| A `runs/courseA_nocmd` | 1 级 100k | 无 | 0.0% | 30% | 0.098 m/s |
| B `runs/courseB_staircmd` | 3 级 300k | vy±0.30 wz±0.35 | 0.0% | 0% | 0.175 m/s |
| C `runs/courseC_nocmd` | 3 级 300k | 无 | 0.0% | 0% | 0.137 m/s |
| 对照 course1/course2 | 1 级 100k | 无 | **40%** | 60% | 0.914 m/s |

B 与 C 只差楼梯段指令、差异为 0；A 与 C 只差平地级数。**结论：回归由「三级平地预热」
造成，与楼梯段的横向/偏航指令无关。** 更细的观察：A 的 3 cm 是「摔倒 30% + 超时 70%」，
B/C 是「100% 超时、几乎不摔、慢慢挪 0.14~0.18 m/s」，course1 是「0.914 m/s 冲上去、
60% 摔但 40% 成功」。**平地预热学出来的是一种更慢更稳的步态，它上不了台。**

**指令跟踪的实测（`runs/course3/checkpoints/flat_3_final.zip`，平地，40 回合）**：
横向 vy 指令与实测的**相关系数 +0.661，但回归斜率只有 +0.028**（指令 ±0.36，
实测只有 −0.03~−0.07 的基线漂移）；偏航 wz 的**相关系数 −0.085、斜率 −0.006**，
即完全没跟。前进 vx 恒定指令 0.45，实测均值 0.285（偏低 37%），而 `teleop --hold 0.45`
下实测 1.26 m/s（**超调 2.8 倍**）—— 训练时指令是每回合随机抽的，策略学到的是
「不管指令、按自己节奏走」。

**第二层根因（零策略基线，`.tmp/prior_cmd_response.py`）**：把残差动作固定为 0，
只让 trot 先验走，实测 `vx=+1.547 vy=+0.004 wz=-0.058`，**对任何 vy/wz 指令都完全不变**；
手工只拧髋关节侧摆（`a[0::3]=±1.0`）也只换来 `vy=-0.03~+0.04`、`wz=-0.19~+0.03`。
⇒ trot 先验是**纯前进步态**，横移与偏航必须由策略主动推翻它；而
`action_scale=0.30` 的残差权限很小，第一层「奖励里跟指令有关的项只有
`lateral_vel=1.5` / `yaw_rate=1.0` 两个高斯（取 `max` 合并成一项），而前进项是 2.5」，
使得「不跟指令、拿满前进分」是更稳的局部最优。

**已采取的修复**：新建 `.tmp/courses/course4.yaml`（**一级平地 + 3/6/9/12/15 cm**，
平地 `vx=0.55`、`flat_success_distance=3.0`，楼梯段 `vx` 基线 0.55 加 vy/wz 指令，
奖励覆盖 `foot_lift [0.9,1.0,1.2,1.4,1.6,1.8]`、`probe_lift [0,0.6,0.9,1.2,1.5,1.8]`），
回到 course1 的课程形状、保留全部新改动，正在训练 `runs/course4`。

**教训**：
1. **课程级数是能力的一部分，不是中性的「热身」。** 三级平地预热让策略在
   「平地 + 指令」上多花了 600k 步，代价是它把步态调成了平地最优解。
2. **横向/偏航指令必须在观测里（已经在了）还不够，还必须有足够大的奖励权重**
   —— 动作权限不是瓶颈（见 §18），奖励结构才是。零策略基线（先验单独走）
   是判断「这个能力是策略学来的还是本来就有的」的最便宜实验。
3. **旧结论要用同一把尺子重测。** 「course1 3 cm 44%」本来是训练期滚动统计，
   换成确定性评估后是 40%，两者接近才让回归变成可比的结论。
4. **course4 的结果**（一级平地 + 保留全部新改动，1,350,000 步、2,512 s）：
   确定性评估 `0 cm 100% / 3 cm 10% / 6 cm 0% / 9 cm 0% / 12 cm 0% / 15 cm 0%`，
   仍低于 course1 的 3 cm 40%。把楼梯段指令置零后 3 cm 反而降到 0%（100% 超时），
   说明**新指令本身也在拖累爬楼**。⇒ 指令跟踪与爬楼必须分开解决，先解决前者。

## 18. 指令跟踪优先：先量「动作权限」，再改「奖励结构」

**目标（用户 m10558：「指令跟踪优先，优先解决这个问题」）**：先把平地指令跟踪
做到「指哪走哪」，再回去处理爬楼。

**第一步就是排除动作权限，而不是先调权重**（`.tmp/action_authority.py`，平地、
随机常量动作 200 组 × 120 步，看平均 vy / wz 能到多少）：

| action_scale | 最好 \|平均 vy\| | 最好 \|平均 wz\| | 指令量程 |
|---|---|---|---|
| 0.30 | **0.435 m/s** | **1.824 rad/s** | vy ±0.40 / wz ±0.70 |
| 0.60 | 0.493 m/s | 3.036 rad/s | 同上 |
| 0.80 | 0.665 m/s | 2.945 rad/s | 同上 |
| 1.00 | 0.512 m/s | 2.558 rad/s | 同上 |

解析动作的对照更能说明问题：`hip 前+1 后−1`（对角髋反向）在 `scale=0.30` 下
就给出 `vy=+0.333`、`wz=-0.808`。⇒ **`action_scale=0.30` 的权限已经超过指令量程，
「学不会跟踪」不是权限不够。** 这条否证很重要：它把问题从「环境改不了」推回
「奖励与课程设计」，也避免了盲目把 action_scale 放大（那会让先验的零点偏移，
先验是按 0.30 标定的，`.tmp/prior_cmd_response.py` 已确认零残差时先验只直走）。

**第二步：改奖励结构（`robotdog/reward.py`）**

1. `lateral_vel` 1.5 → **4.0**、`yaw_rate` 1.0 → **3.0**；
2. `command_follow` 从 `max(横向项, 偏航项)` 改成**求和**。取 max 时哪个方向误差
   小就只保留那一个 —— 横向跟得还行时偏航项的梯度被整个丢掉，策略于是永远学不会
   偏航（这正是 §17 里「横向 r=+0.661 而有微弱响应、偏航 r=−0.085 完全无响应」的
   直接原因）。求和让两个方向各自都有梯度。
3. 两项仍然被 `if cmd.vy_max:` / `if cmd.yaw_max:` 门控住：范围为零的方向不参与，
   「只学直走」的阶段奖励项集合与旧版逐位一致。

**第三步：建「指令跟踪专项」课程 `.tmp/courses/cmdtrack.yaml`（900,000 步、4 级全平地）**

| 级 | 指令 | vx | 奖励覆盖 | 预算 |
|---|---|---|---|---|
| `flat` | vy±0.40 | 0.0 | `forward_vel=0` | 200k |
| `flat_2` | wz±0.70 | 0.0 | `forward_vel=0` | 200k |
| `flat_3` | vy±0.40 | 0.45 | 正常 | 250k |
| `flat_4` | vy±0.40 wz±0.70 | 0.45 | 正常 | 250k |

关键设计：**前两级把 `forward_vel` 覆盖成 0 且 `vx=0`** —— 原地学横移/扭转，
此时「只顾往前走」拿不到前进分，策略只能靠 `command_follow` 得分；而且横移与
偏航被**分开**教，避免一上来就双通道。`action_scale` 保持 0.30 不动（第一步已证明够用）。

**第 0 级为什么用 `flat_success_distance=1.0`**：原地阶段没有前进指令，若沿用
3.0 m 的判据，`is_success` 永远不会触发，进度行的成功率恒为 0，看不出学习信号。


## 19. 速度指令为什么学不出来：一个把「早摔倒」变成最优解的奖励

**背景（m14795 / m17857）**：用户要求「vx 也做指令跟随、指令落在 0.2~0.6、冲刺额外
扣分」，并进一步要求「基本从 0 开始、保留一个较弱的先验做探索种子、先从较窄的
速度带学会走」。围绕这两条做了若干条单级平地实验，**全部失败**，这一节记录为什么。

### 19.1 先修掉的两个真缺陷

**（a）前进奖励读的是陈旧常数。** `compute_reward` 原来取
`err = lin[0] - cfg.command.vx`，而 `cfg.command.vx` 只是「指令范围为 0 时的回退
定值」；课程下发的是 `set_commands(vx_min=..., vx_max=...)`，**它不改那个定值**。
于是奖励的目标永远停在 `config.yaml` 的出厂值（0.45/0.55），而观测与成功判据用的
是真实采样值。课程 `vx∈[0.05,0.20]` 时 `|err| ≥ 0.35`，`track_shape` 一步撞到 −1
下界，`forward_vel` **每步吃满 −10**。改成读瞬时指令 `env._command[0]` 后，同一条件
的零残差回合回报 **−8429.7 → +1571.5**、`forward_vel` **−9.995/步 → +0.016/步**。

**（b）前进项的量程按指令大小归一化。** 原来 `max(|cmd.vx|, forward_range_floor)`：
同一个绝对误差在不同指令档位得到不同分数，且指令越小时量程越假。改成**绝对误差
窗口** `forward_track_window`（默认 0.25 m/s）。

### 19.2 ⭐ 真凶：`termination = -15.0` 让「尽快摔倒」的回报高于「走」

`robotdog/reward.py` 的 `termination` 只在摔倒时付一次 −15。而平地回合是
`episode_length_s=20.0 / control_dt=0.02` = **1000 步**，前进项最差 −10/步 ⇒

| 做法 | 每步 | 回合总回报 |
|---|---|---|
| 完美跟指令走满 1000 步 | +10.00 | **+5000**（+`success`） |
| 未受扰的先验（0.79 m/s，指令带 0.3~0.62）走满 | −9.59 | **−16050** |
| 第 75 步摔倒 | — | **−5829**（−15 时代只 −844） |

**「摔倒」把负流掐断，所以当策略追不上指令时，早摔是严格最优的。** 训练日志
逐位印证：

```
[   8,000 步] 摔倒率=  0.0%  均回报= -9074.72  均步数=1000.0   ← 未受扰的先验
[  66,552 步] 摔倒率=100.0%  均回报=  -682.61  均步数=  87.1   ← 学会了尽早摔倒
[ 178,024 步] 摔倒率= 88.0%  均回报= -1850.70  均步数= 259.5
```

**修法**：`config.yaml` 的 `reward.termination` **−15.0 → −5000.0**。它必须显著
大于「单步奖励动态范围 × 回合步数」，否则这个激励永远存在。改完「早摔倒」不再是
更优解（上表第 4 列）。

### 19.3 被控对象：速度是**可静态标定**的（这一点是好消息）

先验自身（`gait_amp_scale=0.50`、`amp_th=0.13251`、`amp_ca=0.49318`）零残差走
**0.7934 m/s**。给四条大腿的 thigh 关节加一个**常值**偏置，稳态速度几乎是一一对应
的，而且长回合（2900 步）**稳定**：

| 大腿偏置 (rad) | 稳态 vx | 对照指令 |
|---|---|---|
| −0.50 | **+0.3195** | 0.30（\|err\| 0.0195） |
| −0.40 | **+0.3909** | 0.40（\|err\| 0.0091） |
| −0.30 | **+0.4819** | 0.50（\|err\| 0.0181） |
| −0.20 | **+0.5813** | 0.60（\|err\| 0.0187） |
| −0.10 | **+0.6864** | 0.70（\|err\| 0.0136） |
| 0.00 | **+0.7934** | 0.79 |

⇒ **0.32~0.79 m/s 全程稳定、精度约 ±0.02 m/s。** 速度跟踪在物理上不难。

### 19.4 真正挡住策略的是**动作限幅**，不是奖励也不是可学性

要产生上面那些偏置，动作值 = 偏置 / `action_scale`。出厂 `action_scale = 0.30`：

| 目标速度 | 需要的大腿偏置 | 需要的动作值（限幅 ±1.0） |
|---|---|---|
| 0.60 | −0.20 | −0.667 ✅ |
| 0.50 | −0.30 | −1.000 ⚠ 刚好触限 |
| 0.40 | −0.40 | **−1.333 ❌** |
| 0.30 | −0.50 | **−1.667 ❌** |

⇒ **指令带 [0.30, 0.62] 里只有最高的那一小段够得着；「刹车」方向早就撞上限幅。**
策略每次试图跟低指令都会撞死在限幅上，随之失稳摔倒。「刹车残差 = 偏置/尺度」
在 §18 被漏掉了：那一节量的是**横向/偏航权限**（结论「0.30 够用」对那两轴仍然
成立），但**前进刹车**是另一回事，它需要接近满量程的负偏置。

### 19.5 一条更根本的观察：本仓的先验是**开环关节轨迹**，速度不可调

先验是一个固定周期的关节轨迹，它把机器人推到一个由机械结构决定的自然速度
（全幅值 2.03 m/s、本仓 0.5 缩放下 0.79 m/s）。策略只能叠一个小残差。因此
「速度指令」在这套设定里**不是控制量**，而是「要不要跟一个与自然速度冲突的目标」
的取舍。若要让速度真正可指令化，应当把先验参数化（按指令缩放 stride 幅值或步频），
或在动作空间里给一个显式的「步态幅值缩放」通道 —— 而不是继续用 12 维关节残差
去硬凑。

### 19.6 两个探针口径错误（都会给出相反结论）

1. **`episode_length_s / timestep` 是错的**，真值是 `episode_length_s / control_dt`
   （`robotdog/env.py` 的 `_max_steps`）。0.002 而非 0.02 ⇒ 把 1000 步算成 10000 步，
   于是「早摔倒到底划不划算」的临界点差 10 倍。
2. **`gait_amp_scale` 必须改在原始 config dict 上**。`config_to_env` 返回之后
   `cfg.gait` 已经缩放过一次；在返回值上再设 `cfg.gait_amp_scale` 是空操作，而手动
   再缩 `cfg.gait` 会叠加成两次缩放（实测把 0.30 变成 0.0054 m/s 的原地蹭）。同一
   个 `0.80` 在「相对 / 绝对」两种口径下能差 2 倍。

## 20. 速度参数化：先验的步幅跟着指令缩放之后，速度才第一次成为控制量

§19 的结论是「本仓的先验是开环关节轨迹，速度不是控制量」。这一节记录把先验
**参数化**（让它的步幅由 `vx` 指令决定）的完整标定、实现与第一组对照，以及它
**仍然做不到**的那一段。

### 20.1 先验「幅值缩放 → 稳态速度」的标定：只有两支稳定

口径：零残差、平地、`episode_length_s=100`（消除超时混淆）、`settle=300` +
`meas=1200` 步、seed 3，逐回合区分「稳定 / 第 N 步摔倒 / 超时」。缩放乘在
`amp_th` / `amp_ca` 上，然后过一遍 `normalize()`。

| 缩放 `s` | `amp_th` | `amp_ca` | 稳态 `vx` | 结局 |
| --- | --- | --- | --- | --- |
| 0.28 | 0.07421 | 0.27618 | +0.0366 | 稳定（**原地小步支**） |
| 0.34 | 0.09011 | 0.33536 | — | 第 321 步摔倒 |
| 0.40 | 0.10601 | 0.39455 | — | 起始摔倒 |
| 0.46 | 0.12191 | 0.45373 | +0.6983 | 稳定（**行走支下沿**） |
| 0.52 | 0.13781 | 0.51291 | +0.8390 | 稳定 |
| 0.58 | 0.15372 | 0.57209 | +0.9766 | 稳定 |
| 0.70 | 0.18552 | 0.69046 | +1.2673 | 稳定 |
| 0.82 | 0.21732 | 0.80882 | +1.5914 | 稳定 |
| 1.00 | 0.26503 | 0.98637 | +2.0306 | 稳定 |
| 1.02~1.62 | （`amp_ca` 被夹在 0.98655） | | 2.022 → **1.734** | 稳定但**速度反降** |

两条结论：

1. **只有两支稳定**：`s<=0.28` 的「原地小步」（0.037 m/s）与 `s>=0.46` 的
   「行走」（≥0.70 m/s），中间 `s∈[0.34,0.44]` **全部摔倒**。所以不存在
   「缩幅值就慢下来」这种线性直觉。
2. **缩放上限是 1.0**：超过之后 `normalize()` 的 `|off| + amp <= 1` 触发**整体
   再缩放**，`amp_ca` 被夹住而偏置继续变大，速度反而下降。

### 20.2 中速区是空的（不是没找到参数，是结构上不存在）

在 `(amp, freq)` 平面上扫 `amp∈[0.30,0.70] × freq∈[1.5,7.0]`，再细扫
`amp∈[0.30,0.52] × freq∈[2.5,7.0]`；落进 `[0.15, 0.65] m/s` 的稳定点
**只有两个**：`amp=0.44 / freq=7.0 → +0.5791`、`amp=0.46 / freq=7.0 → +0.6213`。
其余格点要么是原地小步（0.04~0.07），要么摔倒（`amp` 0.34~0.44 全灭），
要么落在行走支的上沿。**常值偏置 trot 结构上走不到 0.1~0.6 m/s。**

这也解释了此前所有「让速度可指令化」的尝试为什么都失败：命令区间 `[0.2,0.6]`
的下半段根本不在这类步态的可达集里，策略面对的是「走 0.7+」或「不动」的二选一。

### 20.3 实现：一条反解出来的线性律

行走支上做最小二乘（10 个点）得到

```
v(s) = 2.4975 * s - 0.4657        最大残差 0.0153 m/s
```

反解即 `s = clip((v_cmd + 0.4657) / 2.4975, 0.46, 1.00)`（`scale_for_speed`）。
实现只缩放 `amp_th` / `amp_ca`，**不动 `off_th` / `off_ca`**：偏置是站立姿态的
一部分，缩它会改变机身高度与腿的负载分配，实测直接前倾摔倒。动态缩放存在一个
运行期字段 `_scale_dyn` 里，`radians(t)` 临时应用、`finally` 还原，而
`to_dict()` 会把它去掉（它不进配置，也不进 `summary.json`）。

**必须先反解 `v(s)`，不能把 `s` 拟合在 `v` 上。** 第一版写成
`s = 0.40 + 0.44 * v`，等价于 `v_prior ≈ 0.88 * v_cmd + 0.54` —— 先验永远比指令
快 0.54 m/s，残差需求恒为 −0.60 rad，策略又一次退化成常值动作
（`|action|` 恒 0.92、速度锁死 0.39）。两个方向的「最小二乘」给出的是不同的直线。

零残差实测：`v_cmd = 0.90 → 0.9003`、`1.50 → 1.4997`（**±0.0003 m/s**）；
而 `v_cmd = 0.20 / 0.45 / 0.60` 全部被裁剪成同一个 **0.6985**。

### 20.4 下沿以下只能靠残差刹车，而刹车是过刹的

零残差先验固定走 0.6985 m/s，要压到 0.2~0.6 必须给大腿一个常值负偏置。
实测（`action_scale=0.60`，各档稳定）：

| 大腿偏置 [rad] | 归一化动作 | 稳态 `vx` |
| --- | --- | --- |
| −0.104 | −0.17 | +0.3756 |
| −0.209 | −0.35 | +0.1735 |
| −0.315 | −0.52 | **−0.1385**（过刹） |
| −0.420 | −0.70 | **−0.2234**（过刹） |
| −0.526 | −0.88 | **−0.1830**（过刹，且回弹） |

三种 `action_scale`（0.30 / 0.60 / 0.90）下实测值**逐位相同** —— 一旦按
「偏置 / 缩放」换算成归一化动作并 clip，实际偏置就一样了。所以 `action_scale`
只能决定**够不够得到**：要覆盖 0.2~0.4 档必须 ≥0.60（0.30 时需要 −1.05~−1.75，
全部超出 ±1）。而够得到之后，偏置与速度是**非线性且过刹**的。

### 20.5 第一组对照：先验在指令带内 vs 带外

> ⚠ **本节的结论已被 §20.7 推翻**：下表的「B 臂学不动」是评测脚本用
> `env_config_for_model` 反推配置（拿到 `gait_prior=False`、`action_scale=0.30`
> 的默认值）造成的，不是被控对象的性质。用训练时的课程 yaml 重测，B 臂在
> 450k 步就把 `[0.20,0.60]` 跟到 ±0.12 m/s（斜率 +0.89、r +0.97）。
> 本节保留原文，作为「一个配置口径错误能造出多完整的假因果」的样本。

两条单级平地臂，各 800k 步、`n_envs=8`、`action_scale=0.60`、
`gait_amp_scale=1.0`、`gait_speed_track=true`、`forward_track_window=0.10`、
`termination=-5000`，**唯一差别是指令带**：

| 臂 | 指令带 | 先验能否表达 | 观测（同一步数附近） |
| --- | --- | --- | --- |
| A | `[0.70, 1.20]` | 能（带内精确） | `success_rate 0.44`、`ep_len 912`、`explained_variance 0.94` |
| B | `[0.20, 0.60]` | 不能（被裁成常数 0.6985） | `success_rate 0`、`ep_len 611~711`、`explained_variance 0.80` |

这是「速度真的变成控制量」的**第一个正面证据**：先验能表达的那一段，策略学得动；
表达不了的那一段，无论怎么训都是 0。同时也说明**当前 trot 的速度下限约
0.70 m/s**，要在 `[0.2, 0.6]` 上做真跟踪，需要换一类步态（带支撑相占空比的
walk / crawl），而不是继续调这条 trot 的幅值与频率。

> 上面这段推论**两半都错**：B 臂并不是「学不动」，而 `[0.2,0.6]` 也不需要换步态。
> 详见 §20.7。

### 20.6 一条被推翻的旧结论（写在这里免得再犯）

早期用 `episode_length_s=20`（1000 步）做过一轮「先验稳定性」扫描，得到
「`s=0.28` 稳定、`s∈[0.31,0.45]` 在 240~660 步内前倾摔倒、`s>=0.50` 又稳定」。
把回合拉长到 100 s 并**分开判 `terminated` / `truncated`** 之后，真相是：
那些「摔倒」全部是**超时**被算成了摔倒，`s>=0.46` 的行走支一直是稳定的。
教训与 §5 的那一类同源 —— **探针必须把 `terminated` 与 `truncated` 分开判**，
否则「没跑完」会被记成「摔了」，而两者的处置完全相反。

### 20.7 ⭐ 上面 20.5 的结论**当时就测错了**：先验在带外照样能跟

20.5 那张表里的「B 臂学不动」是**探针口径错误**造成的，不是被控对象的性质。
原因是评测脚本用 `env_config_for_model(ckpt)` 去拿配置：训练中途的 checkpoint
旁边**没有** `_env.json`（`scripts/train.py` 只在阶段末写一次），于是它退回
「按观测维度反推开关」的路径，拿到的是一堆 **dataclass 默认值** ——
`gait_prior=False`、`action_scale=0.30`。这两个默认值和训练时（`gait_prior=true`、
`action_scale=0.60`）完全不是一回事：关掉先验之后动作流水线退化成
`home + 0.3·action`，机器人根本迈不开腿，于是**任何检查点在这套配置下都测出
「vx≈0、斜率≈0」**。

同一个检查点、同一段代码，只换配置来源：

| 配置来源 | `gait_prior` | `action_scale` | 指令 0.20/0.35/0.50/0.60 → 实测 vx | 斜率 / r |
| --- | --- | --- | --- | --- |
| `env_config_for_model`（维度反推，**错**） | `False` | `0.30` | −0.0001 / −0.0002 / −0.0002 / −0.0001 | **−0.0001 / −0.44** |
| 训练用的课程 yaml（**对**） | `True` | `0.60` | +0.3182 / +0.4125 / +0.5110 / +0.5864 | **+0.667 / +0.9993** |

也就是说：**「策略锁死在常数速度、指令不是控制量」这个结论是假的**，
它是「用默认配置去回放一个在另一套配置下训出来的权重」的产物。用正确配置
重测后，B 臂（指令带 `[0.20,0.60]`，正是 20.5 说「表达不了」的那一段）在
450k 步就学到了**单调、高相关的速度跟踪**：

| 检查点 | vx 指令 0.10 / 0.20 / 0.35 / 0.50 / 0.60 / 0.70 / 0.90 | 斜率 / r |
| --- | --- | --- |
| `runs/zzparam_lo` @100k | — / +0.361 / +0.358 / +0.653 / +0.631 / — / — | +0.831 / +0.890（乱跳） |
| `runs/zzparam_lo` @250k | — / +0.234 / +0.311 / +0.456 / +0.353 / — / — | +0.403 / +0.760 |
| `runs/zzparam_lo` @450k | +0.258 / +0.319 / +0.413 / +0.512 / +0.584 / +0.680 / +1.029 | **+0.890 / +0.967** |

A 臂（`[0.70,1.20]`，先验在带内）@450k 更精确：0.70→0.7087、0.90→0.9057、
1.00→1.0115、1.20→1.2418，**最大误差 0.042 m/s，斜率 +1.066、r +0.9991**。
所以 20.5 最后那句「当前 trot 的速度下限约 0.70 m/s，要做 `[0.2,0.6]` 真跟踪
必须换一类步态」**也是错的** —— 不需要换步态，`[0.2,0.6]` 靠「速度参数化的
先验 + 常值残差刹车」就能跟到 ±0.12 m/s。

**真正的教训**（三条，都是可复用的）：

1. **回放权重必须用训练时那一套配置**。`_env.json` 只在阶段末写（而
   `CheckpointCallback` 每 50k 步存一次中间检查点），所以「训练中途的检查点 +
   维度反推」这条路**天然拿到默认值**。评测脚本一律从**课程 yaml**
   （`.tmp/courses/*.yaml`）取配置，或在训练时就给每个中间检查点写一份
   `_env.json`。
2. **反推维度根本无法确定开关组合**。`gait_prior` / `action_scale` 这些字段
   **不进观测**，所以任何「按 51 维反推」的方案都不可能恢复它们 ——
   它不是「精度差一点」，是**结构性不可恢复**。
3. **一个「什么都没变」的测量结果，先怀疑测量本身**。当时三个完全不同的
   检查点（不同奖励、不同课程、不同臂）在探针里给出**几乎逐位相同**的
   「常数输出」曲线，这本身就是极强的不对劲信号：不同的权重不该给出同一个
   数字。把这条当判据，可以省下后面几轮全部基于错结论的分析。

---

## 21. 换掉先验的腿相位：trot → walk，3/4/5/6 cm 零残差全通（2026-10-12）

### 21.1 结论先说

**先验的 `leg_phase` 是上楼问题的真正杠杆。** 把出货先验从 trot `(0, π, π, 0)` 换成
walk `(0, π, π/2, 3π/2)`（幅值 / 相位 / 频率一起重搜，落地为
`artifacts/walk_stairs.json`），**零残差**（策略输出全 0）就能在 3/4/5/6 cm 上全部走通，
而且上楼速度落在 `command_vx` 的 `[0.2, 0.6]` 指令带内：

| 台阶 | 结局 | 上楼均速 [m/s] | 零残差回报 |
| --- | --- | --- | --- |
| 3 cm | 4/4 成功 | 0.475~0.482 | +3251~+5556 |
| 4 cm | 2/2 成功 | 0.415~0.438 | +4241~+6142 |
| 5 cm | 2/2 成功 | 0.345~0.348 | +6384~+6464 |
| 6 cm | 3/3 到顶 | 0.231~0.255 | +6139~+6753 |
| 9 cm | 0/3（卡在第一道立面） | 0.031~0.037 | −371~+599 |

对照旧先验（trot）：`amp 1.0` 下 3 cm 虽 8/8 成功但上楼 **1.19 m/s**（带外）；
而 `gait_speed_track` 实际给到的 `amp 0.46` 下**连第一道立面都过不去**
（0 级、被阻挡 0.95、机身系 vx +0.028）。

### 21.2 两个被推翻的旧结论（不要再引用）

1. ~~「带内的慢速爬升在物理上不可行」~~ —— **对 trot 成立，对 walk 不成立。**
   同高度下 trot 的「带内速度」与「可靠性」互斥：`amp 0.80 / freq 6.0` 能 4/4 成功
   但上楼 0.962 m/s（带外）；`amp 0.75 / freq 7.0` 在带内（0.555）却只有 2/4。
   换腿相位后两个指标同时满足。
2. ~~「6 cm 是这条先验的天花板」~~ —— 那是 **trot** 的天花板。walk 在 6 cm 上仍然 3/3。

### 21.3 怎么搜出来的

`.tmp/gait_search.py` —— 离线参数搜索，目标函数是本仓特有的「带内」约束：

    得分 = 踩到级数/总级数 − 5·摔倒 − 3·超时 − 2·max(0, 上楼均速 − 0.60)
           + 6.0 · clip((apex_min − 台阶高)/0.05, 0, 1)

`--patterns` 覆盖 trot / pace / bound / walk / walk_lat 五种腿相位。150 次试探的
**前三名全是非 trot**（walk、walk、bound），得分打满 7.0 —— 这条搜索一开始就该
包含腿相位，只调幅值 / 频率是搜不到的。

### 21.4 ⚠ 必须同时关掉 `gait_speed_track`

`scale_for_speed` 在整条 `[0.2, 0.6]` 指令带上都返回下沿 `SPEED_SCALE_MIN = 0.46`
—— 也就是说这条「速度参数化」在本仓的指令带里其实是一个**常数**，并没有在跟随速度
（`v(s) = 2.4975 s − 0.4657` 反解出的 `s` 在 `v ≤ 0.6` 时都低于 0.46）。而 0.46 的
振幅对楼梯太小：walk 先验被裁到 46% 后**同样爬不上去**（实测 0/2、被阻挡 1.00、
摆动峰值只剩 5.8~7.1 cm）。

平地段的代价可忽略：trot（缩放 0.46）与 walk（满幅）的平地稳态速度分别是
**0.685 与 0.686 m/s**，几乎相同。

### 21.5 ⭐ 但 RL 训练会把这份好先验弄坏：先验已解决的任务不该再训

九级课程（2,150,000 步）跑完，**楼梯段全部不达标**：

| 阶段 | 成功率 | 摔倒率 |
| --- | --- | --- |
| `stair3` | 24% | 74% |
| `stair6` | 0% | 70% |
| `stair9` | 0% | 46% |
| `stair12` | 0% | 0% |
| `stair15` | 0% | 2% |

**同一个 3 cm 地形，零残差先验是 4/4 成功、0 摔倒。** 即训练出来的策略**比自己
什么都不做还差**。台阶越高摔倒越少、回报越高 —— 是「爬不上去就站着不动」的退化解。

> 注：训练日志里的 `均回报` 是 `VecNormalize` **归一化后**的值，幅度不可与原始回报横比；
> 成功率与摔倒率是原始的。

确定性回放（每格 3~5 回合，权重自带 `_env.json`）证实这不是采样噪声：

| 权重 | 3 cm 上 | 爬到 |
| --- | --- | --- |
| `flat_4`（热启动源） | 0/3 成功、3/3 摔倒 | 4~5 级 |
| `stair3@50k` | 0/3、3/3 摔倒 | 3~7 级 |
| `stair3@100k` | 0/3、2/3 摔倒（1 次差 0.35 m 超时） | 8 级 |
| `stair3_final` | 0/5、5/5 摔倒 | 3~7 级 |
| **零残差 walk 先验** | **4/4 成功、0 摔倒** | 8 级 |

### 21.6 病因：残差没有理由停在 0

量残差幅值（`.tmp/act_stats.py`，`mean|a|` 与它在关节角上的贡献）：

| 权重 | 地形 | `mean|a|` | 大腿残差 | 占先验幅值 |
| --- | --- | --- | --- | --- |
| `flat_4` | 平地 | 0.149 | ±0.123 rad | **32%** |
| `stair3` | 3 cm | 0.143 | ±0.116~0.124 rad | **30~32%** |

**两者几乎一样**：策略在楼梯上没有学到任何新东西，它把平地上那份残差**原封不动**
带了过去；同一份残差在平地上好用（3/3 成功）、在楼梯上就摔。

为什么压不下去：

* **奖励里没有任何「动作幅值」项** —— 只有 `action_rate`（罚*变化*）与 `torque`
  （罚力矩平方），都不驱策残差本身趋零；
* 在先验已给出近优解的地方，优势函数几乎全是噪声，策略没有梯度理由把均值推回 0；
* `ent_coef=0.002` + `log_std_init=-2.0` 一直保着探索，`tanh(mean)` 的非零均值
  于是被保留了下来。

**这与历史最优 `ab_hist1`（在 trot 先验上学到 62%）恰好互补**：那条先验**不够用**，
RL 有活可干，于是加分；这条先验**已经够用**，RL 只剩把轨迹推离最优的余地，于是减分。

### 21.7 顺带：换先验把平地的指令跟随也拉起来了

| 阶段 | 旧（`course9`：trot + `gait_speed_track`） | 新（`walk1`：walk，无参数化） |
| --- | --- | --- |
| `flat` | 0% | 0% |
| `flat_2` | 0% | **12%** |
| `flat_3` | 0% | **40%** |
| `flat_4` | 0% | **44%** |

### 21.8 下一步（按证据强度排序）

1. **先验已经解决的级不要训**（3/6 cm），或者给它加一条**残差幅值惩罚**
   （`action_l2`，按阶段用 `curriculum.reward_overrides` 下发：3/6 cm 强、9/12/15 弱或零），
   让「输出 0」重新成为**严格**最优。
2. **把预算挪到先验真的过不去的级**（9/12/15 cm），并在那里继续搜腿相位 / 抬腿裕量，
   而不是加大 `action_scale` —— 21.6 显示残差幅度不是瓶颈。
3. *（假设，未测）* 复核 `VecNormalize` 的观测统计量是否只在平地分布上标定过：
   楼梯上的归一化可能把观测推到训练分布之外。

### 21.9 复现命令

```bat
.venv\Scripts\python.exe .tmp\gait_search.py 3 --trials 150 --episodes 1 --out .tmp\gait_3cm_band.json
.venv\Scripts\python.exe .tmp\diag_gait_file.py artifacts\walk_stairs.json 3 --episodes 4
.venv\Scripts\python.exe .tmp\eval_on_stairs.py runs\walk1\checkpoints\stair3_final.zip 3 --episodes 5
.venv\Scripts\python.exe .tmp\act_stats.py runs\walk1\checkpoints\stair3_final.zip 3 --episodes 3
.venv\Scripts\python.exe .tmp\diag_prior.py 3 --cfg runs\walk1\checkpoints\stair3_final.zip
```

### 21.10 方向修正（同日，用户要求）：先验**不得**成为楼梯的解

§21.1 那份 walk 先验让"零残差就已经爬得上去"，这等于**把答案写进先验**，RL 只剩
把它弄坏的余地。用户明确要求「不希望依靠先验将上楼梯解决」，所以本节之后做了三件
事，先验退回"通用底座"的定位：

1. **`config.yaml` 的 `gait_file` 退回 `artifacts/trot_stairs.json`，
   `gait_speed_track` 放回 `true`。** trot @ 缩放 0.46 零残差**爬不上 3 cm**
   （0/2、被阻挡 0.95）—— 稳、但不解题。`walk_stairs.json` 保留为**实测发现**，
   不作为出货配置。
2. **预算挪到先验真的过不去的级**：`timesteps` 由等预算改为
   `[150000, 250000, 250000, 250000, 4096, 4096, 500000, 500000, 500000]`。
   3/6 cm 各留 4096 步不是删级 —— 保住九级课程形状（`play.py` 按阶段名找 checkpoint）。
3. **让爬升在账面上有分量**：`stair_progress` 由 `3.0` 提到 `1200.0`。
   理由是本节量化出来的量级失衡：`terrain_height_under_base` 是阶梯函数，爬完整条
   15 cm 楼梯在原权重下总共只付 **+3.60**，而 9 cm 上零残差先验站着不动一个回合的
   净账是 −371、`command_follow` 一项就白付 +4616。提到 1200 后每级 15 cm 值 180、
   爬完值 1440，与逐步项同量级。

**背面还有一个未落地的发现**（留着，别丢）：`command_follow` 的横向/偏航两项用的是
**相对**量程（`|误差| / cmd.vy_max`），而前进项用的是**绝对**窗口，且前进项的注释
明写"必须是绝对量程 —— 相对误差下指令带一窄，站着不动就几乎满分"。横向/偏航项恰恰
违反这条：零输出时误差恒为量程的一半，`track_shape(0.5) = 0.5`，于是**完全静止的
机器人白拿 `0.5 × (6.0 + 4.0) = +5.0/步`**（1000 步 +5000），且这个 0.5 与量程无关。
修法是**按比例收窄**这两项的窗口：``window = frac × 量程``，``frac = 0.5``。代入可
解析得到零输出期望恒为 **−0.125**（×6.0 = −0.75/步），且**与量程无关** —— 与那条
"与量程无关的白拿分"正好对称。

⚠ **不要用"固定绝对窗口"**（本节早先的提法，已被 `.tmp/quant_track_window.py` 否掉）：
对**窄**指令带反而更糟 —— vy_max=0.10 时零输出期望由 0.500 涨到 0.667（w=0.15）~
0.833（w=0.30），因为窄带里的指令本来就小、绝对窗口显得很宽松。实测量级：

    关掉先验 + 动作钉 0（机器人真的不动），400 步均值：
        flat_2 / stair3 / stair9 / stair15 的 command_follow = +6.57 / +6.25 / +6.25 / +6.25
        同一批回合的 forward_vel         = −8.58（alive +0.15）
    解析（零输出期望得分，满分 1.0）：
        vy_max 0.10 → 旧 0.500 | w=0.15 0.667 | w=0.20 0.750 | w=0.30 0.833
        vy_max 0.30 → 旧 0.500 | w=0.15 −0.083 | w=0.20 0.208 | w=0.30 0.500
        vy_max 0.50 → 旧 0.500 | w=0.15 −0.450 | w=0.20 −0.267 | w=0.30 0.100

**未落地**是因为它需要新增字段 + 重训，且要过一次回归门。

前进项另加了一处速度口径修正（`robotdog/reward.py` 的 `_forward_speed`）：把
**机身系**前进速度换成**水平面内沿机身朝向的地面前进速度**
`v_h = (v_body[0] − v_world[2]·e_x[2]) / |h|`（`e_x[2] = R[2,0]`）。平地段由地形门控
直接返回 `v_body[0]`，所以「平地逐位不变」是可测的强断言。

⚠ **这一项我先写错过，记在这里免得再犯**：第一版写成「除以 `cos(pitch)`」，理由是
「机身抬头 θ 时机身系前进速度 = 世界水平前进 × `cos θ`」。**那个推导把符号搞反了** ——
`v_body[0] = v_world·e_x = v_world_x·cos θ + v_world_z·sin θ`，爬楼时 `v_world_z > 0`
那一项是**加**的，所以原始读数并不是「偏小」，而是略微**偏大**；除以 `cos` 把偏差
又放大了一次。实测（walk 先验、3 cm 成功、`|pitch|` 中位 0.112；与独立算出的
水平参考量相比）：

| 口径 | 3 cm 平均绝对误差 / 偏置 | 6 cm |
| --- | --- | --- |
| 原始 `v_body[0]` | 0.00684 / −0.00023 | 0.02299 / −0.00173 |
| 只除以 `cos`（第一版） | 0.00704 / **+0.00259** | 0.02281 / **+0.00342** |
| `_forward_speed`（现在） | **0.00000 / +0.00000** | **0.00000 / +0.00000** |

也就是说第一版**不是**修得更准，而是把偏差换了个方向放大（爬楼时高估前进速度）。
这条同时被 `.tmp/` 的数值实验与一轮独立代码审查指出 —— 而**独立审查的价值正在于此**：
我自己那版"等价性证明"当时只断言了"平地不变"，恰好绕过了这个错误。

### 21.11 ⭐ 判别实验：先验**不是**坐在刀刃上 —— 病因在优化器一侧

给「能爬的 walk 先验」叠加**常值大腿偏置**（归一化动作单位，×0.6 rad 得弧度），
零残差跑 3 cm、每档 2 回合：

    .venv\Scripts\python.exe .tmp\diag_prior.py 3 \
        --cfg runs/walk1/checkpoints/stair3_final.zip --bias 0.0 0.10 -0.10 0.15 -0.15 0.20 -0.20

| 大腿偏置 | 结局 | 步数 | 踩到级 | 机身系 vx | 回报 |
| --- | --- | --- | --- | --- | --- |
| **0.00** | 成功 / 成功 | 480 / 480 | 8/8 | +0.475 | +5360 / +3174 |
| +0.10 | 成功 / 成功 | 360 / 364 | 8/8 | +0.637 | −2 / −1771 |
| **−0.10** | 成功 / 成功 | 747 / 770 | 8/8 | +0.302 | **+7578 / +7601** |
| +0.15 | 成功 / 成功 | 324 / 325 | 8/8 | +0.707 | −1653 / −2844 |
| −0.15 | 超时 / 超时（**已到顶**） | 1000 | 8/8 | +0.217 | +7289 / +7971 |
| +0.20 | **摔倒** / 成功 | 279 / 295 | 8/8 | +0.726 | −7231 / −3199 |
| −0.20 | 超时 / 超时（**已到顶**） | 1000 | 8/8 | +0.205 | +6918 / +7954 |

两条结论，都改正了 §21.6 的措辞：

1. **先验很结实。** ±0.10 的常值偏置 —— 正是被训坏那个策略实测的残差量级
   （``mean|a|`` = 0.143、大腿残差为先验幅值的 ~32%，见 §21.6）—— **照样 2/2 成功、
   8/8 到顶**，只有 ±0.20 才开始出问题（1 次摔倒）。所以「策略把好先验弄坏」**不是**
   因为「先验容不下扰动」。**病因在优化器一侧：RL 是没停在近零残差上，而不是不能停。**
2. **奖励景观里有一个明确的、RL 找反了方向的最优点。** 偏置 −0.10 的回报（+7578）
   远高于 0（+5360）—— 因为它把机器人刹到 0.30 m/s、更贴指令带（0.2~0.6）；
   而正偏置（+0.10 / +0.15）把它推到 0.64~0.71 m/s，超速被罚成负回报。
   但训练出来的策略收敛到**正**偏置（thigh +0.0168，见 §21.6）—— **方向是反的。**

⇒ 这对「不靠先验、让 RL 解楼梯」的方向是**好消息**：先验换成爬不上去的通用底座之后，
优化器有一件真活要干（学出 −0.1~−0.2 那档刹车 + 抬腿 + 把身体推上去），而不是被要求
「保持不动」—— 那恰恰是策略梯度方法最不擅长的事（近优点的优势函数几乎全是噪声，
见 §21.6 的三条理由）。**这条推论尚未被训练验证**，`walk2` 正在跑。

### 21.12 一轮独立代码审查的收获（2026-10-12）

对同一批改动做了一次**六维度独立审查 + 每条发现 3 视角对抗性复核**（`Workflow`；维度：
前进项口径 / `stair_progress` 标定 / 先验回退与预算 / 文档一致性 / 等价性证明本身 /
隐藏缺陷）。13 条被确认、0 条被推翻。最有价值的三条：

1. **`reward.py` 的"俯仰补偿"方向是反的** —— 见 §21.10 末尾的更正。要点不只是那个
   符号错，而是：**我自己那版等价性证明只断言了"平地不变"，恰好绕过了这个错误**。
   只证"没改坏"的证明，对"改对了没有"是盲的；是审查把它挖出来的。
2. **`config.yaml` 的课程注释自相矛盾**（同一段既写"四个平地处热"又写"前三级全留在
   平地"）—— 那是我自己改出来的，已修。
3. **文档行号引用大面积漂移，而门检测不到。** `docs/robotdog/env.md` 成员表的"谁在用"
   那一列用的是**裸 `:NNN` 引用**（同一行没有文件名），`check_lines` 与 `ref_truth`
   都解析不到它们 —— 所以它们红着，八道门也照样全绿。抽查（`sed -n`）证实：表里给
   `terrain_height_under_base` 列的 `env.py:1007/1052`、给 `foot_pos` 列的 `env.py:339`、
   给 `foot_clearance` 列的 `env.py:911` 都指到了无关代码上（分别是 `_track_cross`
   初始化、`_gait_action` 的零返回、`_contact_fatal` 的 `return False`）。
   **根治办法是换成符号式引用**（`` `_track_swing_peak` ``、`` `reset()` `` 这种）——
   符号不会漂移，这也正是 `.tmp/check_my_refs.py` 自己 docstring 里写的偏好
   （"符号式引用（无行号，不会漂移）"）。**本轮只修了能定位到符号的那些**
   （`env.md` 6 处 + `gait.md` 6 处，全部按 `grep -n` 取到的真值），成员表的**用例级**
   引用留给下一次：它们不影响行为，而改错的风险高于收益。

> 一条方法论：**行号引用不能凭"看起来旧"或"按差值推"来改。** 本轮我自己就推错过一次 ——
> 把 `_target_from_action` 写成 `1067`（那是按 `env.py` 还处于 +8 行的旧状态算的），
> 真值在行数中性之后是 **1059**。凡是要动行号，先用 `grep -n` 取真值。

### 21.13 ⭐ 修正：`stair_progress` 的参照面由**躯干 x** 改成**四足支撑面**（2026-10-12）

§21.10 里我写过「这一项是**一次性、按历史最高结算、总量有界**的，不存在 §12 记过的
『站着不动每步白拿』问题」—— **那句话只对了一半。** 一次性结算确实挡掉了「每步都付」，
却挡不住**换一个刷分姿势**：参照面原本取 `terrain_height_under_base`
= `height_at(base_pos[0])`，**只看躯干 x、不看 y、不看脚**。于是策略只要把躯干**探过**
一道立面就按级收钱，四只脚可以全在地面上。

这是 **§2.7 逐字记过的同一条退化路径**（标题即「`stair_progress` 奖励『躯干冲线』
而不是『脚上了一级』」）。它当年的修法是足端参照面
（`stair_progress_foot_ref` + `stair_progress_min_loaded_feet`），而那两项**已随重构被删**。
我把权重从 3.0 提到 1200，等于给这条路径发了 **400 倍**的薪水 —— 是那轮独立代码审查
把这条挖出来的，我自己两次写到这一项都没看出来。

**实测**（零残差、walk 先验、每格 900 步）：

| 地形 | 躯干下方地形最高 | 四足支撑面最高 | 超前 ≥ 一格 的步占比 | `stair_progress` 累计（修前 → 修后） |
| --- | --- | --- | --- | --- |
| 6 cm（真爬上去了） | 48.0 cm | 48.0 cm | 0.36 | +576 → **+576** |
| **9 cm（爬不上去）** | **9.0 cm** | **0.0 cm** | **0.91** | **+108 → +0.00** |

即 9 cm 上**四只脚全程在地面**、躯干读数却有 9 cm，白拿 +108。

**修法**：`climbed` 的参照面换成 `env._support_height()`（四足下方最低地形，即真正撑住
机器人的那个面），`reset()` 里 `_peak_terrain_height` 的初值同步换。四处全是
**1 行换 1 行**，`env.py` / `reward.py` 正文行数不变（`git diff -U0` 的 hunk 头逐条
核对：`-190 +190` / `-222 +222` / `-349,2 +349,2` / `-352 +352`，另加末尾纯追加），
所以 323 条文档行号引用全部继续有效。

为什么不叠当年的 `stair_progress_min_loaded_feet`：取**四足最小值**本身已是最严口径 ——
只要还有一只脚在下一级，支撑面就不涨、就不付钱，等价于「整只机器人真的上了一级」。

**平地不受影响**：平地上支撑面恒 0 ⇒ `stair_progress` 恒 0（等价性证明的 P4 已覆盖）。

> 方法论：**「总量有界」不等于「不可刷分」。** 一次性结算只保证「同一高度不重复付钱」，
> 不保证「拿到那个高度的**方式**是对的」。一个势函数项要真不可刷，**参照量本身必须是
> 那个任务真正想度量的事件** —— 这里是「脚踩上去了」，不是「躯干探过去了」。

### 21.14 ⭐ 前进项在楼梯级改用**课程方向**参照（2026-10-12）

`walk2`（2,408,192 步、九级、新奖励 + 修好的参照面）跑完，**楼梯五级仍全 0%**，
平地正常（`flat_4` 46%）。诊断出的机制与「能力」无关，是**奖励在指反方向**。

**实测**（`walk2` 检查点、确定性回放、1000 步回合）：

| 权重 | world x | yaw | 机身系 vx | 世界系 vx | 朝课程走的步占比 | `forward_vel` 回合累计 |
| --- | --- | --- | --- | --- | --- | --- |
| `stair15` | 0.98 → **−1.68** | −0.095 → **+3.02 rad (173°)** | **+0.340** | **−0.221** | 0.16 | **+7480** |
| `stair12` | 前进 **−1.7 ~ −5.0 m** | — | +0.23~+0.33 | — | — | +6248 |
| `stair9` | 0.98 → **1.37**（立面在 1.50） | 末 26° | +0.044 | +0.033 | 0.58 | −5150 |

官方评估（`play.py --mode eval`，100 回合，15 cm）：成功率 **0.0%**、
**平均前进 −6.33 m**、平均速度 −0.32 m/s、超时 100%。

**机制**：`forward_vel` 跟踪的是**机身系**速度，而「前方」是鼻子指哪算哪；出生点背后
是**无障碍平地**，于是最优解就是掉头 173°、以指令带内的速度走开。账面：「走开」一整
回合 **+10667**，而爬完整条 15 cm 楼梯只值 `success 50 + stair_progress 1200×1.20
≈ +1490` —— **走开比爬上去多赚 7 倍**。就算它会爬，奖励也让它选走开。

`stair9` 是同一件事的另一半：它停在 `x = 1.37~1.48`，**比第一道立面（1.50）还差一点**
（而 walk 先验本来能走到 1.57）—— 不是上不去，是**主动选择不靠近**：靠近了就要吃
−10/步 的前进罚。

**修法**：`_forward_speed` 按地形分参照系 —— 楼梯级返回 `base_lin_vel_world[0]`
（课程方向），平地级返回 `base_lin_vel_body[0]`。楼梯是按 `start_x` 沿 +x 铺开的一条
课程，所以 `vx` 只能是「沿课程前进的速度」；平地级不动，那里转向是指令的一部分。
**不需要新配置**：地形自己就说明了任务是什么（与既有的地形门控同一套思路）。

**离线预验**（同一批真实轨迹上两种参照的 `forward_vel` 回合累计）：

| 场景 | 机身系（旧） | 课程系（新） | 差 |
| --- | --- | --- | --- |
| 真爬上去（3 / 6 cm 成功） | +2301 / +2731 | +2363 / +2655 | +62 / −76（≈2%） |
| **掉头走开（stair15 / 12 训练后）** | +7480 / +6248 | **−7222 / −5561** | **−14703 / −11810** |
| 卡在立面前（stair9） | −5150 | −5490 | −340 |
| 先验在 15 cm 卡住 | −5958 | −5931 | +27 |

**只打漏洞、不打正解。**

**顺带删掉的**：`_forward_speed` 里那层「俯仰投影补偿」。课程方向是世界系量、本来就
不含俯仰投影误差，所以它在楼梯级被完全取代；留在平地上只有约 1% 的效应，却会破坏
「平地逐位不变」这条可测断言 —— 而那条断言正是上一轮抓出它符号错误的东西。

> 方法论：**「跟踪得好」不等于「在做任务」。** 一个跟踪项只要参照系选错，就存在一个
> 「满足它但完全不做任务」的退化解 —— 这里就是掉头走开。检查办法很便宜：看**世界系
> 位移**与**朝向**，不要只看奖励值。奖励涨了不等于任务在推进。

### 21.15 ⭐ walk3 的早期读数 + 两条量具陷阱 + 先验振幅这条路是堵死的（2026-10-12）

`walk3`（改后奖励、同一套课程）训到第 8/9 级时先量了已经落盘的 `stair9`，顺手做了
对照与可达性实验。四件事，前三件是**工具错**，第四件是**结论**。

#### 21.15.1 `stair9` 分不开 `walk2` 和 `walk3`

先跑 `walk3` 看到 `前进 +0.36~+0.58m`、`世界系 vx +0.019~+0.030`、`forward_vel −4650~−5490`，
差点写成「机制修好了」。**但 §21.14 第 1737 行早就记了 walk2 的 stair9**：世界系 vx
**+0.033**、机身系 +0.044、末位 x 1.37、`forward_vel −5150`。补一次对照（`.tmp/eval_on_stairs.py`
跑 `runs/walk2/checkpoints/stair9_final.zip`）得到 +0.023~+0.027 / 摆动峰值 6.1~6.2cm /
被阻挡 0.90~0.94：**两个 run 在 9 cm 上统计上不可区分。** 掉头只发生在 12/15 cm
（walk2 `stair15`：世界系 vx −0.221、偏航 173°、`forward_vel` +7480）。

> 教训：**判据要在能区分开的那个高度上读。** 同一条判据在 9 cm 上「通过」不携带信息，
> 因为基线也通过。选高度之前先看基线在各高度的取值。

#### 21.15.2 量具陷阱 A：`diag_stair4.py` **不报世界系 vx**

预登记的主判据第一条是「世界系 vx 均值 > 0」，而 `.tmp/eval_on_stairs.py` 转发的
`diag_stair4.report` 打的那行标签是「机身系 vx 均值」—— 它取的是
`rows["vx"] = env.base_lin_vel_body[0]`（`robotdog/env.py:609`），**世界系那一列根本没被记录**。
补一行 `vx_world=float(env.base_lin_vel_world[0])`（`env.py:605` 的属性）后才量得出主判据。

**这次补列顺带验证了量具**：同一把尺子量 walk2 `stair15` 得世界系 vx **−0.188/−0.280/−0.249**
（§21.14 记的是 −0.221）、机身系 **+0.352/+0.331/+0.349**（记的是 +0.340），独立复现。

> 顺带记一笔**两表不可混读**：§21.14 写 walk2 `stair15` 的 `forward_vel = +7480`，那是
> 它**在旧奖励下**的值；把同一个旧权重放进**新奖励**去评，得到的是 **−8760**。
> 两句话都对，说的是两回事。混读会看成矛盾。

#### 21.15.3 量具陷阱 B：没有平地对照 + 速度列，会把「原地空转」读成「抬得高」

`.tmp/sweep_prior_reach.py` 量「零残差下摆动峰值随 `gait_amp_scale` 怎么变」（口径与训练
一致：指令带 0.2~0.6、`speed_track=on`、seed 0、800 步、`artifacts/trot_stairs.json`）。
**第一版只扫台阶、不扫平地**，得到 4.3~4.6 cm 且几乎不随振幅变 —— 看上去像「振幅不是杠杆」。

加了**平地对照行**和**末段速度列**之后：

| `gait_amp_scale` | `amp_th`* | `amp_ca`* | 平地峰值 | 平地末位 x | 平地末段速度 | 平地结局 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.46 | 0.1219 | 0.4537 | 14.8 cm | 1.44 | **0.008 m/s** | 走满 |
| 0.70 | 0.1855 | 0.6905 | 22.6 cm | 1.97 | 0.104 m/s | 摔倒 |
| 0.85 | 0.2253 | 0.8384 | 22.4 cm | 2.74 | 0.270 m/s | 摔倒 |
| **1.00** | 0.2650 | **0.9864** | **4.5 cm** | **11.83** | **0.692 m/s** | 走满 |
| 1.15 | 0.3048 | 0.9865 | 4.6 cm | 11.98 | 0.702 m/s | 走满 |
| 1.50 | 0.3975 | 0.9865 | 4.6 cm | 12.25 | 0.719 m/s | 走满 |

`amp 0.46` 那行 14.8 cm 是**假象**：机器人以 0.008 m/s 原地乱蹬，腿在空转，离地间隙虚高。
**真正在走的三档（1.00/1.15/1.50）峰值全是 4.5~4.6 cm。** 不加对照与速度列，就会得出
「先验抬得出 14.8 cm，够得着 9 cm 立面」这个**方向相反**的结论。

#### 21.15.4 ⭐ `gait_amp_scale` 在 **calf 通道**上没有放大余量 —— 这条路是堵死的

上表第 3 列是关键：`amp_ca` 在 `gait_amp_scale ≥ 1.0` 时**卡在 0.9864~0.9866 不动**，
而 `amp_th` 从 0.2650 一路涨到 0.3975。原因在 `robotdog/gait.py:200-210` 的 `normalize()`：

```python
if abs(o) + a > 1.0:
    s = 1.0 / max(abs(o) + a, 1e-9)
    setattr(self, off, o * s); setattr(self, amp, a * s)
```

artifact（`artifacts/trot_stairs.json`）里 **`|off_ca| + amp_ca` 恰好 == 1.0000**
（`amp_ca 0.9864` + `|off_ca| 0.0136`）——**calf 通道出厂就顶在天花板上**。于是
`config.py:281-286` 把 `gait_amp_scale` 乘上去之后，`normalize()` 立刻把它 clip 回 1.0
再归一化回 0.9866；`amp_th` 有余量（0.5584），所以只有它在涨。

**后果**：抬腿高度由 calf 主导，而 calf 振幅**不可放大**。所以「把楼梯级的先验振幅
抬到 0.60~0.65」这句话在本 artifact 上是**结构性地不可行** —— 不是「不够大」。
（原话出自 `.tmp/walk3_acceptance.md:43-44`，那条阶梯（0.46→0 级 / 0.60→1 级 /
0.65→2 级 / 0.70→6~8 级）**全仓查不到支撑工件**，且与 `.tmp/step1_findings.md` §5 唯一
真实的网格矛盾 —— 6 cm 上 `amp 0.80/1.00/1.15` × `freq 4~7` **九组全失败**。
该条已作废并划掉。）

**训练口径下的硬事实**：`gait_amp_scale = 1.0`（训练用的那一档）时，零残差先验在
**平地上以 0.692 m/s 正常行走**、摆动峰值 **4.5 cm**；到立面前同样是 4.5 cm。
**4.5 < 9** —— 零残差先验物理上过不了 9 cm 台阶，与「它是不是够努力」无关。

> 教训：**先验的「振幅」不是一个自由度。** 它是 `(amp, off)` 在 `|off|+amp ≤ 1`
> 这个单纯形上的一个点，而 artifact 的 calf 顶点已经取到了边界。要真的改抬腿高度，
> 只能动**结构**（`leg_phase` / `calf_shape` / `freq`，即 §21.1 那条「腿相位是杠杆」），
> 或者让残差去补 —— 不能靠乘一个全局系数。

#### 21.15.5 顺带确认：`command_follow` 在近乎静止时每步仍白付

`walk3 stair9` 的奖励前五项里，`command_follow` 是**第一正项**（+5148~+5921 / 1000 步
≈ **+5.2~+5.9/步**），而当时机身系 vx 均值只有 +0.023~+0.032 —— 机器人几乎不动。
`walk2 stair9` 同一位置是 +5127~+5853。这条账**仓库自己记过**（`robotdog/reward.py:57-63`
的注释：「配上 `command_follow` 的 `+3.76/步` 与 `alive +0.15`，站桩合计 `+4.93/步`」），
修法写的是「改成绝对窗口」—— 但那次绝对窗口只落到了**前进项**
（`RewardConfig.forward_track_window = 0.25`），`compute_reward` 里横向/偏航两项
（`reward.py:255-269`）**仍然是相对量程**（`cmd.vy_max` / `yaw_range`）。
**那个修法是半截的**，站桩收入从这条路继续漏。

**未改**（本轮保持单变量：`walk2`/`walk3` 只差 `_forward_speed` 一处）。
候选方案是把横向/偏航的归一化量程改成 `0.5 × 量程`（尺度无关，零输出 ⇒ −0.125 而不是 +0.5）。
**注意这不碰 `forward_track_window`** —— 那是前进项的绝对窗口，是另一件事，
且按用户要求不再提放宽它。


---

## 22. 文档重组与结构清理（2026-10-12，代码审查会话）

本轮做了五件事：修一个奖励语义 bug、两层死代码清理、结构整合、类聚整理、文档体系重建。
每件的验证方式都是同一条：**450 步 × 3 场景（平地+先验 / 楼梯+三轴指令+滤波+probe / 纯残差）
的逐位等价回归**（`.tmp/eq_check.py`，产 JSON 后逐项对比 `reward_terms`）。

### 22.1 ⭐ `action_rate` 奖励算的是「与上上步的差」

`env.step()` 里奖励结算发生在动作历史更新**之前**，而更新顺序是
`_prev_action ← _last_action; _last_action ← action`。推演时序：step 2 的奖励读到
`_prev_action = 0`（应为 a₁），step 3 起读到 a₁（应为 a₂）—— 从第 3 步起
`action_rate = −0.02·‖aₜ − aₜ₋₂‖²`，惩罚的是**隔两步**的差。意图（字段名、docstring）
明显是相邻步差。

**修法**：奖励改读 `env._last_action`（奖励结算时刻它恰好是 aₜ₋₁），删除 `_prev_action`
字段。修复用两个独立手段验证：① 微测试 20 步逐项比对奖励值与解析公式精确到 1e-12；
② 等价回归里除 `action_rate` 外其余项逐位不变、`action_rate` 首个差异步恰为第 2 步
（与 bug 时序推演一致）。

**教训**：**「在结算之前更新状态」的代码，其字段语义要按时序推演验证，不能按名字信任。**
已训权重是在错误语义下训出来的；修复后任何「逐位不变」断言在这一项上会有预期差异。

### 22.2 死代码第一层（grep 证实零调用者）

删除：`set_stair()`（连同 `_build_model` 里只为它存在的渲染器重建守卫；渲染器陷阱
改写成「模型只在 `__init__` 编译一次」的不变量注释）、`config._bools()`（服务的
`needs_progress` 字段已不存在）、`GaitParams.targets()` / `speed_for_scale()`、
`Go2Meta` 的 12 个构造后无人读取的字段（含 `extract_meta` 里的对应提取代码）、
`SceneOptions.extra_visual/nconmax/njmax` 及其两个死分支、`compile_scene(write_to=)`、
`build_scene_xml(include_lights=)`、`play.EpisodeOutcome.final_x`、`scripts/__init__.py`。

**推翻一条旧判据**：experience §9.1 曾把 `set_stair` 归入「文档化公开 API 必须留」。
本轮用户批准删除 —— 「文档把它当公开 API」不是保留的理由，**没有调用者且需求不存在
（课程每级新建环境）就该删**；文档跟着代码改。

**抓漏**：删 `SceneOptions.extra_visual` 字段时漏删了 `build_scene_xml` 里的消费分支，
第一次等价检查当场 `AttributeError`。**教训：删字段必须 grep 它的全部读点，删完先跑
等价检查再继续。**

### 22.3 结构整合（行为保持）

`_contact_flags` + `_foot_contact_mask` 两遍 `data.ncon` 循环合并为一次
`_contact_scan()`（返回三元组）；5 个相机的 if/elif 链改为 `_CAMERA_VIEW` 预设表 +
注视点查表；play/teleop 三份「读 json→失败给默认」收敛为 `read_json_or()`。
净 −24 行。**诚实记录**：预估 −100，实际 −24 —— 预估没算进必须保留的 docstring
与共享辅助函数自身的行数。这类整合的价值在消除会漂移的拷贝，不在行数。

### 22.4 类聚整理：探测机制整体搬进 reward.py

按「代码尽可能出现在调用的地方」：`_track_swing_peak` / `_update_probe_state`
（probe_lift 的两个势函数状态机）从 env.py 搬到 reward.py（`track_swing_peak(env, ..)` /
`update_probe_state(env)`，沿 `_forward_speed` 先例）；`probe_target` 属性（纯转发、
只有 reward 读）删除，reward 直读 `rc.probe_lift_target`；`_contact_fatal` 内联进
`_terminated`。势函数**状态字段**仍留在 env（reset 要清零、info 要上报），搬的是**机制**。

**踩到一个 Python 作用域坑**：`step()` 里写 `reward.track_swing_peak(self, ...)`，
而 step 后面有局部变量 `reward, terms = ...` —— `reward` 被提升为全函数局部名，
当场 `UnboundLocalError`。改成裸函数调用（与 `compute_reward` 一致）。

### 22.5 记录在案的潜伏问题（未修）

`command_for_stage` 对 `vx 量程为 (0,0)` 的定值级回放 `vx=0`，而训练时那一级用的是
`env.command.vx` 基线（0.45）。当前 `config.yaml` 无定值级，不触发；将来加回定值级时
要让 `command_for_stage` 接收基线。

### 22.6 §21.15.5 的后续：横向/偏航「0.5 × 量程」已落地

§21.15.5 记录时还是「未改」状态；同日晚些时候已按候选方案落地：`command_follow` 的
横向/偏航窗口取 `0.5 × 量程`（`reward.py`），零输出期望从 +0.5 变为 ≈ −0.083/轴
（尾项被 `max(-1,·)` 截断）。

### 22.7 `.tmp/eq_check.py`：新的行为回归门

比旧 §6.3 的等价证明更好用：同样「固定种子 + 固定动作序列、逐项对比 `reward_terms`」，
但三场景覆盖（先验开/关、指令/滤波/probe/楼梯/终止路径）、一条命令跑完、产 JSON 可
程序化对比。**任何碰 env/reward/gait 的改动之后都要跑一次。**

---

## 23. sub-9 课程实验：trot + 2026-10-12 奖励下，3 cm 学出了真爬楼但困在两种局部最优之间（2026-10-13）

### 23.1 设置

课程插入 7.5 cm 中间级（十级），3/6/7.5 cm 预算从 4096 恢复为 250k 真训
（旧分配是 walk 先验时代「零残差就能过」的产物，对 trot 不成立）。先验
trot_stairs.json @ speed_track 下沿 0.46。run：`runs/trot_sub9`（stair6 起手动停止）。

### 23.2 结果

* **平地四级 900k 步，指令跟随成功率恒 0%**（回报 6069→6184→5806→4443 递减，
  摔倒率 2%）。相关系数门槛（最差轴 ≥0.30）一次都没过。
* **stair3：训练成功率 0% / 摔倒率 90%**，但轨迹是一个「改进→崩塌」循环：
  摔倒率 100% → 12%（101k 步，回合长 908）→ 回升到 94~96%（回合缩到 400）。
* ⭐ **确定性回放揭示训练统计掩盖的真进展**：`stair3_final` 在 3 cm 上
  **踩到第 5/8 级**、80% 时间足在台阶上、摆动峰值 8.5~11.9 cm、世界系 vx
  0.29~0.43 落在指令带内 —— 对照零残差先验（0/8、被阻挡 0.95、一步不爬），
  这是**质变**：stair_progress=1200 + probe_lift + foot_lift 的梯度确实
  教出了爬楼行为（§21.15 的「9 cm 以下无解」被推翻到 3 cm 这一层）。
* 100k 中段 checkpoint 是**另一种**策略：被阻挡 0.50~0.83、probe 每回合触发
  23~93 次、卡在第 2 级超时（末段钉住 0.70）。即训练在「谨慎卡住（超时）」
  与「激进冲楼（中段摔）」之间演化，两种局部最优的账面价值接近
  （超时 ≈ −3300~−5000；爬 5 级后摔 = +144 − 5000 ≈ −4900），**「爬完」
  （+50 + 1440）两头都够不着**。

### 23.3 机制分析

摔在半路（第 5 级）的账 = stair_progress +144 − 5000；回到平地重爬时
`_peak_terrain_height` 已随回合重置，已爬的高度不带走。于是奖励景观里有
两个价值相近的吸引子，梯度在一者间振荡（KL 早停全程活跃，explained_variance
0.895 —— 优化器本身健康）。** stair_progress 的一次性结算只付「增量」不罚
「跌落」**：从 15 cm 摔下来没有代价，峰值势能不构成对「活着待在高处」的激励。

### 23.4 平地跟随 0% 的疑点（未解）

walk1（walk 先验）flat_2 起就有 12%→44%，trot 下四级全 0%。候选解释：
① trot 是纯前进步态，横移/偏航要靠残差主动推翻（§17 零策略基线），
walk 腿相位天然留了侧向作动空间；② 最差轴门槛过严（三轴同时 ≥0.30）。
未做分辨实验。

### 23.5 诊断脚本 API 同步

`.tmp/diag_stair4.py` / `diag_prior.py` 引用旧 `_foot_contact_mask()`，
已批量改为 `env._contact_scan()[2]`（§22.4 改名的下游）。

### 23.6 回放速度修复与 2 Hz 自然先验（2026-10-13）

**回放 2× 加速的确认与修复**：`gif_max_frames=300` 把取帧稀疏化到每 4 个控制步
一帧，而帧间隔写死 40 ms（25 fps）→ 20 s 仿真播成 10 s。修法：帧间隔改为
`take × control_dt`（`EpisodeOutcome.frame_ms`），取帧越稀疏播放间隔越长，
永远真实时间。观感「步态过快」= 2× 播放 × 5.48 Hz 高频步态，两因叠加。

**2 Hz 自然先验（`artifacts/trot_natural.json`）的标定**：用户要求步频减半到
2 Hz、步幅稍增、姿态抬高（贴标称 [0.0, 0.9, −1.8]）。12 组网格 × 3 seed × 40 s
的结论：

* `off_th`（大腿前偏置）**不是姿态的敌人，是行进的驱动力**：off=0 时 vx≈0
  且俯仰 10~13°；原值 0.29 就是低趴前倾的来源（base_z 0.245~0.262、pitch 4~8°）。
* 40 s 长回合下稳定盆地很窄：off ≥ 0.15 全部在 40 s 内摔倒（12 s 短测看不出来
  —— **短测会漏掉慢性失稳**）；唯一 3 seed 全存活点 `off=0.12, amp_th=0.25,
  amp_ca=0.50`：vx 0.22（指令带下沿）、base_z 0.263（旧 0.247）、pitch 2.3°
  （旧 4.7°）、步幅 2 倍于旧运行时。
* 先验不解题 ✓：3 cm 零残差从不踏上台阶（最高 0 cm、blocked 1.00、0 成功），
  约 13 s 后摔（固定步态不适应楼梯，预期内 —— RL 的存在意义）。
* `gait_speed_track` 相应改 **false**：v(s) 标定曲线是 5.48 Hz 特定的，2 Hz 下
  失效；重启用前必须重做 §20 标定。

### 23.7 ⭐ 指令跟随的跨 checkpoint 趋势：wz 部分跟随一直在，vx 学会后遗忘，vy 恒漂（2026-10-13）

对 trot_sub9 的 5 个 checkpoint 各跑同一套分段恒定指令扫描（vy/wz/vx 各 3 档，
每段 70 步取后 50 步均值，`.tmp/track_trend.py`）：

| checkpoint | vy 斜率 (r) | wz 斜率 (r) | vx 斜率 (r) |
|---|---|---|---|
| flat | −0.025 (−0.69) | **0.373 (0.94)** | **0.349 (0.99)** |
| flat_2 | −0.068 (−0.50) | **0.478 (0.95)** | −0.027 (−0.14) |
| flat_3 | −0.033 (−0.23) | **0.493 (0.95)** | 0.186 (0.53) |
| flat_4 | −0.033 (−0.72) | 0.237 (0.96) | 0.017 (0.14) |
| stair3 | −0.073 (−0.80) | 0.356 (0.96) | −0.096 (−0.97) |

四个结论，其中前两个**修正本文件前面的说法**：

1. **「零响应」是错的 —— wz 通道有部分跟随**（增益 0.24~0.49、r 0.94+，
   全 checkpoint 稳定存在）。§23.6 的 cmd_demo 判读把它漏了：那次只有 2 个
   非零档 + 恒漂淹没，斜率回归被漂移污染。
2. **vx 跟随在第 1 级就学会了（增益 0.35、r 0.99），加入 vy 轴后遗忘**
   （flat_2 起塌到 ≈0，再未恢复）—— 典型的课程内梯度竞争/干扰，不是「学不出」。
3. **vy 恒定右漂 ≈ +0.3 m/s 是头号敌人**：它不吃 corr 门（vy 未学，本就无
   跟随可言），但平地 miss 判据是 |vy 实测 − 0| > 0.30 —— 漂移量正好骑在容差
   边界上，~50% 步脱靶，**单凭它就把四级平地的 miss 门全部顶爆**（corr 门
   其实是过的：wz 0.94/vx 0.99）。注意零残差先验 vy ≈ 0（§17 基线）——
   漂移是**策略残差**自己加的，且Reward 上恒漂每步要吃 ≈ −6 的
   lateral_vel 罚（0.5×窗口下 x=2 → shape −1），策略宁可挨罚也不撤，
   强烈暗示「撤漂移」与其它更值钱的东西（稳定？wz 调制？）耦合。
4. **课程进展没有带来单调改善**：flat_4 比 flat 更差（vx corr 0.99→0.14、
   wz 增益 0.37→0.24）。

方法论：**训练期统计（成功率）是门槛的合取，单看它分不清是哪一条门失败的**；
把每个 checkpoint 拉出来做分段指令扫描，才能把「没学会 / 学会后遗忘 /
被恒漂顶爆」这三种病因分开。`.tmp/track_trend.py` 可复用。

### 23.8 观测结构化 + 奖励逐项审计与 scaling（2026-10-13）

**观测**：`gait_phase` 打开（2 Hz 慢步态下单步支撑时间长，残差需要相位信息与
先验配合）；`base_lin_vel_body` / `command` 从「开关 + 构造期校验」改为
**结构固定块**（字段删除，`_obs_frame` 直接写死，`frame_size` 恒 +6）——
指令跟踪的两测在结构上不可移除。观测 51 → **53 维**（48 结构 + gait 2 +
step 1 + vel 2）。`_obs_config_for_dim` 只枚举三个可选开关（注意
`itertools.product` 最后位翻转最快，`gait_phase` 必须放最前才能让 51/53 维
命中合理组合）。

**奖励逐项实测**（平地/楼梯零残差 + 已训策略回合，全表见 doc/reward.md）：

* 逐步项全部被 `max(−1,·)` / `overspeed_cap` 约束在 ±10×权重带内 ——
  **整体尺度不过大**，绝对量由 VecNormalize 吸收。
* **command_follow 的 yaw 零指令基线是最大的恒定负项**（−3.5/步）：误差
  ×1.5 放大 × 0.15 窗口，滤波 wz 摆幅 0.1 rad/s 即饱和。它有梯度
  （「跟住不许转」值 +8/步），但也是账面里最大的常数偏移 —— 已把硬编码的
  1.5 提为 `RewardConfig.yaw_zero_gain`（行为不变），后续可调。
* **四个死项**（base_height/joint_vel/action_rate/joint_limit ≈ 0）是
  瞬态保险丝，2 Hz 稳态下不工作，保留。
* **scaling 已应用**：`success 50 → 500`（§23.7 的「爬完 vs 摔在半路」账面
  失衡；登顶与 stair_progress 总量同量级，仍为 termination 的 1/10）。
  **不 scaling**：stair_progress 1200（爬楼梯度刚建立）、termination −5000
  （§19.2 下界）。两者待下轮训练验证。

### 23.9 坐标系统一 + 指令量程缩减 + 缓变扰动（2026-10-13，用户指令）

三项改动，为下一轮训练铺路：

1. **观测/指令坐标系统一到机体系**。排查结论：除 `base_rpy`（世界系欧拉，是
   姿态表示而非「待统一的向量」）外，全部观测量已是机体系；**唯一混系的是
   指令** —— 它被当机身系目标用，但语义应是世界系（上层规划器「沿课程 +x 走」）。
   修法：`env.command_body()` 每步把世界系指令按 yaw 投影到机体系，奖励/观测/
   脱靶判据统一读投影值。**§21.14 的 `_forward_speed` 地形分支（楼梯=世界系/
   平地=机身系）由此删除** —— 世界系指令 + 机体系投影在结构上关闭「掉头走开」
   漏洞（掉头后机体系指令变成「后退」，误差直接惩罚），不再需要按地形切换。
2. **指令量程缩减 + 容差防骑边界**：vy ≤ 0.15、wz ≤ 0.6（§23.7 的恒漂 +0.3
   骑在 0.30 容差边界上的教训）；miss 容差 vy 0.30→0.12、wz 0.70→0.40，取值
   原则 = ≈2× 真在跟策略的预期误差，不放在典型误差恰好落地的值上。
3. **指令缓变扰动**（OU，std 0.03 / τ 1.5 s，稳态 ≈0.026、限幅 ±0.08）：模拟
   真实操作手的轻微无规律输入；hold=True（回放/遥操作）不生效。

### 23.10 ⭐ 测量伪影更正 §23.7 + 漂移机制确诊 + 横向去门控（2026-10-13）

**（a）§23.7 的趋势表被测量伪影污染。** track_trend 原版在段间不重置朝向：
wz 段（±0.6 持续 4 s）会把机身转几十度，后续段的「机体系指令」混入未受控
转向 —— vx 段测出 r=−1.000 的完美负相关就是这么来的。修法：每段 reset +
钉死 yaw=0。修正后重测 trot_sub9：

| checkpoint | vy 斜率 (r) | wz 斜率 (r) | vx 斜率 (r) |
|---|---|---|---|
| flat | −0.015 (−0.82) | 0.524 (0.999) | 0.105 (0.999) |
| flat_2 | −0.046 (−1.00) | 0.762 (0.999) | 0.617 (1.000) |
| flat_3 | +0.038 (0.998) | 0.738 (1.000) | 0.701 (1.000) |
| flat_4 | −0.288 (−1.000) | 0.463 (0.971) | 0.407 (0.999) |

**§23.7 的「vx 学会后遗忘」结论作废**：真实叙事是 vx/wz 跟随**稳定进步**
（vx 0.105→0.701、wz 0.52→0.74，flat_4 略回落），成功率恒 0% 的真凶是
**vy 恒漂顶爆 miss 门**。教训：**跨段测量必须控住朝向**；r 接近 1 而斜率
接近 0 = 近零方差下的数值噪声，别当成跟随证据。

**（b）漂移机制确诊**。sub9_v2 flat_1 后漂移 +0.173，但**零残差 trot_natural
的 vy = +0.001±0.039 —— 先验不漂，漂移是策略残差学的**。为什么学它？
机制：flat_1 的 `vy_max = 0` 使横向项被 `if cmd.vy_max:` 门控 → **整级没有
任何横向惩罚** → 漂移自由生长，flat_2 起再花容量戒。偏航通道免疫此病
（yaw_range_floor 让零指令也有价钱），横向缺对称设计。

**（c）修法：横向去门控**。新增 `RewardConfig.lateral_range_floor = 0.15`，
横向窗口 = `0.5·max(vy_max, 0.15)` = 全级统一 0.075 —— 横向与偏航对称，
vy_max=0 的级别也有「不许横漂」的价钱（直线行走 +6/步、漂 0.17 → −6/步，
摆幅 −12/步）。附带效果：平地直线行走的 command_follow 从 −3.54 回到 ≈0
（横向 +6 抵消 yaw 基线）。sub9_v2 因此中止（flat_2 完成、flat_3 进行中），
修正后以 **sub9_v3** 重启。

### 23.11 track_trend 扫描的第二个伪影与"以训练门为真值"（2026-10-13）

稳态化改造（一次 reset、行走中钉朝向）后 flat_3 测出 wz 反向响应（−0.5）——
与训练期成功率 10% 矛盾。排查：钉朝向的实现把 roll/pitch 一并清零，等于行走中
瞬移躯干姿态，测的是被瞬移污染的行为。**扫描方法两连败**（①朝向漂移、
②起动瞬态淹没调制、③行走中钉姿态 = 瞬移），暂停打磨：**训练期的
成功率/miss/corr（训练口径本身）才是地面真值**，分段扫描只用于量级评估。

### 23.11 ⭐ 阶段名带点 = warm-start 链的地雷（2026-10-13，sub9_v3 崩溃）

sub9_v3 在 stair9 开级时 `FileNotFoundError: stair7.5_final.zip.zip` 崩溃。
根因：**阶段名 `stair7.5` 里的点** —— SB3 的 `model.save` 对带点路径做
`with_suffix` 处理，`stair7.5_final` 被写成无扩展名的 zip、`_env.json` 被
断成 `stair7_env.json`（stem 在最后一个点截断）；而 warm-start 按
`stair7.5_final.zip` 找 → 再补一个 `.zip` → 双 `.zip.zip` → 炸。

这个雷是 2026-10-12 往课程里插 7.5 cm 中间级时埋下的 —— sub9_v2 在
flat_3 就被手动停止，从未走到 stair7.5，所以一直没炸。**修法**：
`config.py` 的阶段名生成统一把 `.` 换成 `_`（`stair7.5` → `stair7_5`）。
教训：**给会进入文件路径的任何名字加"字符白名单"约束，要在创建它的
同一刻做**，而不是等它第一次撞上路径处理。

### 23.12 sub9_v4 全课程结果与爬楼深度天花板（2026-10-13）

全量数据见 `train.md` 的 sub9_v4 节。三条结论：

1. **平地跟随首次建立**（flat_4 56%，历史最好）—— 四项修正的合力：
   横向去门控（§23.10）、量程缩减防骑边界（§23.9）、success 500（§23.8）、
   坐标统一 + gait_phase（§23.9）。修正后的 miss 容差体系首次「可达」。
2. **楼梯侧全面「摔得少、撑得久」但成功率只破 3 cm**（2%）。
3. ⭐ **爬楼深度天花板 = 连续 2~3 级 / 绝对高度 ~18-24 cm，与台阶高无关**
   （6cm×3、7.5cm×3、9cm×2、12cm×2、15cm×1-2）。单级抬腿不是限制
   （每级都能登上第一、二级），**连续登踏的稳定性**才是 —— 第 3~4 次
   逼近-抬起循环时摔倒。这与 §23 的两局部最优（登顶经济性）互补：
   即使修好账面（success 500），连续动作链的可靠性仍是独立瓶颈。
   stair15 的「被挡 → probe 站住」模式（摔倒率全级最低）说明 probe_lift
   高档位会诱导「站着蹭」而非「尝试登踏」—— 高台阶的 probe 权重需要
   重新权衡。

### 23.13 楼梯级直线指令 + 摔倒罚分级 + 进度指标（2026-10-13，用户观察 GIF 后指令）

用户从 GIF 读出三个问题，排查与处置：

1. **「摔倒 = 从楼梯侧边掉下去」**——根因是 **楼梯级仍在下发转指令**：
   课程给 stair 级 wz ±0.5~0.6（GIF 实录 stair3 wz 指令 +0.44），狗边爬边转、
   头部偏离楼梯方向、从侧边走下台阶。**修法：楼梯五级 vy=wz=0**（「沿楼梯
   直线行走」由外部指令给定），平地四级保留量程负责跟随能力；防漂压力由
   lateral_range_floor / yaw_zero_gain 的零指令基线维持。
2. **「穿模卡住」排查**：直线指令下 10 回合确定性重放 **0 个硬卡死**
   （50 步冻结判据）；转指令口径下回合快速摔倒、无卡住窗口。GIF 里的
   「卡住感」= 斜身抵立面枢转 + mm 级接触穿透（MuJoCo 软接触的正常量级，
   视觉可见但不影响物理）。结论：无网格级穿模 bug，先修指令再观察。
3. **二元判标 → 分级**：(a) 进度指标 `info["steps_mounted"]`（登上第几级，
   0~8）进 info 与 GIF HUD（「登踏 n/8 级」）；(b) 摔倒罚按爬升占比分级：
   `terminal = termination × (1 − relief × frac)`，frac = 峰值支撑面/顶端
   （平地 frac=0 → 行为与旧版逐位一致）；relief=0.5 → 脚下摔 −5000、
   顶端摔 −2500。成功保持二值 +500（分级成功已由 stair_progress 逐级
   +36~+180 承担）。§19.2 防早摔约束只依赖 frac=0 全额，不受影响。

### 23.14 楼梯段闭环外部引导（2026-10-13，用户架构指令）

用户指出核心缺陷：**训练与播放时没有外部控制器** —— 开环的 vy=0/wz=0 对
漂移/卡死毫无反应。teleop.py 的 StairController（横移拉回中线 + 偏航对准）
早已实现却只用于演示。修法：`env._guidance_command()` 把它机制化进环境 ——

* 楼梯级（n_up > 0）每步：`vy = clip(−1.2·(base_y − center_y), ±0.15)`、
  `wz = clip(1.5·wrap(−yaw), ±0.6)`（增益/限幅在 EnvConfig.guidance_*）；
* vx 仍由课程采样；缓变扰动叠加在引导值上（真实操作手也有手抖）；
* 平地级不受影响（指令跟随照旧）。

跟随引导指令 = 保持中线与航向：狗走偏时指令**主动纠正**，跟随它就是回到
正确路线上 —— 这是「指令成为状态的函数」的架构修正，也是 §23.7 以来
「跟随」任务的终极形态。配套验证：引导律单元测试（右偏/yaw 偏 → 反向
指令）+ 全回路（右偏 0.2m → 有效指令 vy −0.149 拉回）。

### 23.15 ⭐「足和楼梯穿模、无法抬起」的物理根因：足端 solimp 允许厘米级深穿透（2026-10-13）

用户从 GIF 观察到前脚与楼梯穿模、后续无法抬起。仪器化重放（逐步记录全部
接触的穿透深度）证实：**每回合 2000~3000 步处于 >5mm 深度穿透，最大 32.7mm
（整级台阶高）**，位置在下一级立面的「踏面×立面」凹角 —— 15kg 全身重量经
单足压在立面上，软接触合规性允许持续厘米级下陷，足端几何上陷进 box 内
自然「无法抬起」（要先从墙里退出来）。

**排查过程有两层教训**：
1. 只刚化楼梯 geom（terrain solref 0.005）**完全无效** —— 接触参数由两个
   geom 混合决定，vendored go2.xml 的足端 solimp [0.015, 1.0, 0.022] 主导了
   混合结果。决定性实验：直接改写全模型 solref → 11mm；再改 solimp →
   **3.8mm**。罪魁是 solimp 阻抗曲线，不是 solref 时间常数。
2. 修法落地为 `compile_scene` 里的编译期改写：group 3（全部碰撞图元）
   `solref = [0.005, 1]`、`solimp = [0.99, 0.999, 0.001, 0.5, 2]`；楼梯与地面
   geom 在 XML 层带同样参数。修复后：>5mm 深穿透 **0 步**、最大 5.5mm
   （仅落地冲击瞬间）。

**注意**：这是**有意的物理变更**（接触刚度影响动力学）—— 已训 checkpoint
的行为会轻微偏移，下轮训练（sub9_v5）在新物理上从头学习。等价基线已
重采（eq_baseline.json,2026-10-13 物理版）。
