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
| **框架回调** | `ProgressCallback._on_step`（`scripts/train.py:96`）、`CheckpointCallback._on_step`（`scripts/train.py:119`） | SB3 的 `BaseCallback` 按固定名字回调，从不被本仓代码显式调用 |
| **文档化的公开 API** | `GaitParams.static`（`robotdog/gait.py:170`）、`GaitParams.targets`（`robotdog/gait.py:90`）、`policy.Policy`（`robotdog/policy.py:21`）、`StairSpec.down_end_x`（`robotdog/tools/terrain.py:108`）、`Go2StairsEnv.yaw`（`robotdog/env.py:629-630`）、`Go2StairsEnv.set_stair`（`robotdog/env.py:389-395`） | 设计文档把它们的签名与理由逐条写下来了，是给人用的接口面。删掉等于同时删两处文档，而它们各只有 1–5 行 |

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




