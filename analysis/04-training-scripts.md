# 04 · 训练层与脚本层：`training.py` / `policy.py` / `curriculum.py` / `scripts/*.py`

审计对象：`robotdog/training.py`（**323 行**）、`robotdog/policy.py`（**341 行**）、`robotdog/curriculum.py`（**332 行**）、`scripts/train.py`（**396 行**）、`scripts/play.py`（**667 行**）、`scripts/teleop.py`（**288 行**）、`scripts/cmd_demo.py`（**192 行**）、`scripts/download_go2.py`（**152 行**）。

编号在本分册内为 `S-*`（scripts / training），与 `00-index.md` 的汇总表对应。

---

## S-1（blocker）`scripts/play.py` 的 GIF 模式把**函数对象**当 device 传进去 —— 默认模式下每个阶段必崩

### 现状

`scripts/play.py:47` 从配置模块导入了一个**函数**：

```python
from robotdog.config import (
    DEFAULT_CONFIG_PATH,
    command_for_stage,
    config_to_play,
    config_to_stages,
    device,          # ← robotdog.config.device(raw) -> str，是函数，不是字符串
    load_config,
)
```

`main()` 在 `:607` 正确地先把函数**求值**成字符串：

```python
dev = device(cfg)          # play.py:607  → "cpu"（config.yaml 的 ppo.device）
```

`--mode eval` 分支用的是求值后的变量（`:613`），**正确**。但 GIF 循环里用的是那个函数本身：

```python
# play.py:646-660   （gif 分支，逐阶段）
for i, stage in enumerate(plan):
    ...
    stair = stair_from_height(stage.height_cm)
    policy, env, _venv = open_runner(target, stair=stair, device=device)   # ← :654
```

`:654` 的 `device` 解析为模块级导入的 `robotdog.config.device` 函数对象（`eval` 分支的局部变量 `dev` 不叫 `device`，所以这个名字没有被遮蔽）。`:613` 与 `:654` 只差一个字符，静态读代码极易漏掉。

### 实测证据（`.tmp/zz_probe_playdev.py`，只读探针，不写盘）

```text
type(scripts.play.device) = <class 'function'>
same object as robotdog.config.device: True
play.py:614:         policy, env, _venv = open_runner(target, stair=stair, device=dev)     ← eval 分支，对
play.py:654:         policy, env, _venv = open_runner(target, stair=stair, device=device)  ← gif 分支，错

torch.device(function) raises: TypeError | device() received an invalid combination of arguments
    - got (function), but expected one of: (torch.device device) / (str type, int index = -1)

RESULT: load_policy raised TypeError
   robotdog/training.py:187  load_policy → build_ppo(venv, seed=0, device=device, net_arch=arch)
   robotdog/training.py:131  build_ppo → PPO("MlpPolicy", ...)
   (torch 在 PPO 构造里做 device 解析时抛出)
```

探针直接用 `load_policy(path, cfg=..., stair=stair_from_height(3.0), device=scripts.play.device)` 复现了 `:654` 的调用形状，**在 `Go2StairsEnv` 建立之前**就抛 `TypeError`，因此没有任何文件被创建。

### 影响

* `python scripts/play.py --run <run>`（`--mode` 默认就是 `gif`）**对任何权重、任何阶段都必然失败**：进入 `for` 循环第一轮就抛 `TypeError`，一个 GIF 也写不出来。
* 唯一能用的模式是 `--mode eval`（`:613` 用 `dev`）—— 也就是**默认模式坏、非默认模式好**，而 README/`play.py` 的模块 docstring 都把 GIF 当主用途介绍。
* `except` 只包了 `env.close()` 的 `finally`，没有兜住 `open_runner` 抛出的异常，所以不会出现「部分阶段成功、报错被吞」。

### 修复

```python
policy, env, _venv = open_runner(target, stair=stair, device=dev)   # :654 改成 dev
```

顺手建议：把 `robotdog.config.device` 的导入改名（如 `device_of`），让「函数名 vs 局部变量名」不再只有一个字符之差；或干脆在 `main()` 开头把 `dev` 直接改名为 `device`，消除同名。

---

## S-2（high）`--mode eval` 把统计量算好了却不落盘

`play.py:436-452` 构造了完整的 `payload`：

```python
payload = {
    "model": str(model_path or ""), "episodes": n,
    "success_rate": ..., "fall_rate": ..., "timeout_rate": ...,
    "mean_distance": ..., "mean_height_gain": ..., "mean_speed": ...,
    "mean_steps": ..., "wall_seconds": ...,
}
```

随后只有 `print(...)`（`:453-461`）→ `return 0`（`:462`）。**`payload` 没有任何 `json.dump` / 写文件路径**，`model_path` 参数唯一用途就是填进这个不落盘的字典。

对比：同一仓库里 `runs/course22_ref/eval_report.json` 是**存在的**（旧脚本 `.tmp/eval_run.py` 的产物，`play.py:310` 的注释还引用着它）。也就是说这不是「设计上不需要」，而是**迁移到 `scripts/play.py` 时丢了写盘**。

### 影响

* 「评估至少跑 40 回合」（`play.py` docstring）的结论只存在于终端回滚缓冲里：无法跨 run 比较、无法进 CI、无法被 `summary.json` 汇总。
* 训练侧有 `summary.json`（`curriculum.write_summary`），评估侧什么都没有 —— 产物链在最后一步断掉。
* `payload["model"]` 说明作者本来就打算写盘（否则不必记模型路径）。

### 修复

在 `print` 之后补：

```python
report = Path(model_path).with_name(f"{Path(model_path).stem}_eval.json") if model_path else None
if report:
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
```

`play.py` 已经 `import json`（`:29`），无需新依赖。

---

## S-3（high）`teleop.py` 与 `cmd_demo.py` 的 `--config` 是**死参数**

```python
# scripts/teleop.py
ap.add_argument("--config", ...)                     # :207  定义了
play = config_to_play(load_config())                 # :204  没传 args.config

# scripts/cmd_demo.py
ap.add_argument("--config", ...)                     # :140  定义了
play = config_to_play(load_config())                 # :137  没传 args.config
```

`scripts/train.py:296` 的 `raw = load_config(args.config)` 是**唯一**正确使用该参数的脚本；`scripts/play.py:603` 的 `load_config(args.config)` 也正确。这两个脚本会静默读仓根 `config.yaml`。

### 影响

「超参一律进 `config.yaml`，没有别的开关」（`agent.md §4`）意味着用户会自然地把 `config.yaml` 复制一份、改几个渲染参数、用 `--config my.yaml` 指向它 —— 在 teleop/cmd_demo 上**改动全部无效、且没有任何提示**（连"忽略未知参数"的警告都没有，因为参数被 argparse 正常接受了）。

### 修复

`play = config_to_play(load_config(args.config))`（两个脚本各一处）。

---

## S-4（high）每级 `VecNormalize` 统计**不随 warm-start 迁移**

`scripts/train.py` 的课程循环里，**每一级都新建一个向量环境**：

```python
def make_env(self, stage, *, warm_start=None):
    stage_cfg = copy.deepcopy(self.cfg)
    venv = make_vec_env(stage.spec, n_envs, cfg=stage_cfg, seed=run["seed"],
                        norm_reward=ppo.get("norm_reward", True))
    ...
```

而 `make_vec_env` 内部（`robotdog/training.py`）对每个新 `venv` 包一层**全新的** `VecNormalize(training=True, norm_obs=True, norm_reward=...)`，统计从 `mean=0 / var=1` 起步。

跨级唯一被搬运的是**权重**：

```python
if warm_start:
    model.set_parameters(str(warm_start))     # scripts/train.py → 只搬 policy/value 参数
```

`restore_vecnormalize(...)` 在 `robotdog/training.py` 里**只被 `load_policy()` 调用**（即只服务于评估/回放路径），训练循环从不加载上一级的 `*_vecnormalize.pkl` —— 尽管 `CheckpointCallback` 每 `save_every` 步都把 `save_vecnormalize(model, base)` 写出来了（`<阶段>_<步数>_steps_vecnormalize.pkl` 就在磁盘上）。

### 定量证据（`.tmp/zz_audit_cfg_mutation.py` 探针 B）

以 `joint_vel` 段为例，取原始值 `[-6, -4.91, …, +6]`：

| 归一化用的统计 | 结果 | 与训练期统计的偏差 |
| --- | --- | --- |
| identity（`mean=0, var=1`，clip ±10） | `[-6, -4.91, …]` | 最大差 **4.5** |
| 训练期统计（`std=4`） | `[-1.5, …, +1.5]` | 基准 |

⇒ 换级后的头几千步，策略收到的观测**与它刚训练完的那套分布整体差 3~4 倍尺度**（`joint_vel` 在 `config.yaml` 的观测里是 ±20 rad/s 量级的原始量）。这不是「饱和到 clip」，而是每个维度的尺度都变了。

### 影响

* 系统性风险：**每次换级都会出现一段性能跌落，然后被新一级的 PPO 重新爬回来**。跌落的代价是各级预算的前几千步被浪费，而 `config.yaml` 的台阶预算只有 250k 步/级。
* 更隐蔽的一面：`norm_reward=True`（`config.yaml` 的 `ppo.norm_reward`）也随新 `VecNormalize` 重置 ⇒ 级间**奖励尺度也跳变**，与观测尺度跳变叠加。
* 这与仓库自己的历史教训同类：`experience.md` 记录过「不 warm-start 成功率 100%→0%」；这里的问题是「warm-start 只做了一半（权重），归一化统计没跟上」。

### 修复方向（择一）

1. **搬统计**（最小改动）：`train_stage` 拿到 `warm_start` 路径后，用同目录的 `*_vecnormalize.pkl` 调 `restore_vecnormalize(venv, path)` 再 `learn()`。要注意统计的 `num_timesteps` 与 `clip_obs` 语义。
2. **课程内共享一个 venv**：把「换地形」做成环境方法（`set_stair` 已经存在，`env.py:320-334`，但 docstring 自承当前无调用点），而不是每级重建 venv —— 这样统计天然连续，也省掉每级一次 MJCF 重编译。
3. 至少把「不迁移」写成显式行为 + 文档（现状是隐式的，`summary.json` 与日志里都看不出来）。

---

## S-5（medium）产物里的「逐级配置」是错的：每级写盘的都是**基线**

`scripts/train.py:246-247`：

```python
save_env_json(target, self.cfg, step_height=stage.step_height, stage=stage.name)
```

写进去的是 `self.cfg`（= `load_config` 出来的基线 `EnvConfig`），而真正喂给环境的是 `make_env` 里 `copy.deepcopy(self.cfg)` 得到的 `stage_cfg` —— 两者的 `command.vy_max/yaw_max`、`reward.foot_lift`、`probe_lift`、`flat_success_distance` 完全不同（见下表）。`self.cfg` 只在 `env.set_commands()` 那条路径上被环境**就地改写**，所以它实际记录的不是「基线」而是「最后一次被改写成什么样」。

实测（`.tmp/zz_audit_stage_cfg.py`）：

```text
基线 cfg（写进 <stage>_final_env.json 与 summary.json 的那一份）
  command: vy_max=0.0  yaw_max=0.0  vx=0.55
  reward:  foot_lift=0.9  probe_lift=0.0
  flat_success_distance=3.0  needs_progress=True  reward_overrides={}

课程里每级真正下发的值
  flat    vy 0.00  yaw 0.00  vx 0.45  flat_d 2.0  {foot_lift 0.9, ratio 1.4, probe 0.0}
  flat_2  vy 0.15  yaw 0.15  vx 0.45  flat_d 2.0  {0.9, 1.4, 0.0}
  flat_3  vy 0.35  yaw 0.50  vx 0.45  flat_d 2.5  {0.9, 1.5, 0.4}
  stair3  vy 0.30  yaw 0.35  vx 0.00  flat_d 0.0  {1.0, 1.6, 0.6}
  stair6  vy 0.35  yaw 0.55  vx 0.00  flat_d 0.0  {1.2, 1.8, 0.9}
  stair9  vy 0.40  yaw 0.70  vx 0.00  flat_d 0.0  {1.4, 2.0, 1.2}
```

`summary.json` 的 `stages[i]` 只存 `is_flat / label / learning_rate / name / spec / step_height / timesteps` —— **既没有指令量程，也没有成功距离、`needs_progress`、奖励覆盖**。唯一可靠落盘的逐级字段是 `step_height`（`stage.step_height` 是显式传参）。

### 后果链（不是纯文档问题，会改变行为）

1. **teleop 被误导**：`scripts/teleop.py:233-234` 从 `target.cfg`（= `_env.json` / `summary.json` 的 `env_config`）读 `cfg.command.vy_max/yaw_max` 作为遥操作限幅，读到的却是 `0.0/0.0` ⇒ `:235-240` 的「两个都为 0」分支**恒成立**，于是把限幅强制放开到 `vy ±0.25 / wz ±0.35`，并打印一条**误导性警告**（「该权重的指令课程范围是 0（只训过直走）」）。对真正用指令课程训出来的权重（stair9 的 `vy_max=0.40 / yaw_max=0.70`），这是**静默降级**：用户以为在测 ±0.70 的偏航跟踪能力，实际只下发到 ±0.35。
2. **`step_height_from_model()` 仍然工作**（它读的是显式传入的 `report.step_height`），所以 `play.py` / `teleop.py` 的地形选择是对的 —— 问题只出在**指令量程与奖励覆盖**这两类字段。
3. 历史归档 `runs/course3/summary.json` 里 `env_config.command = {vx:0.45, vy_max:0.4, yaw_max:0.7}`、`reward.foot_lift=1.4`、`probe_lift=1.2`（= **最后一级**的值，旧代码产物）；`env_config.reward_overrides` 永远是 `{}`。⇒ 同一个键在旧归档里是「最后一级」、在新归档里是「基线」，**两代产物语义不同**，任何按 `env_config` 重建配置的消费者（`env_config_for_model` / `env_config_from_summary` / teleop / play）都要面对这个歧义。

### 修复

* `train.py` 改为把 `stage_cfg`（或 `asdict(stage_cfg)`）交给 `save_env_json`，让每个 checkpoint 的 `_env.json` 真正描述它自己的训练条件；`summary.json` 的 `stages[i]` 补上 `command_vy_max / command_yaw_max / command_vx / flat_success_distance / flat_success_needs_progress / reward_overrides`。
* `env_config_for_model()` 的优先级值得复核：现状是 `summary.json` 优先于 `<model>_env.json` —— 对 `checkpoints/<阶段>_final.zip` 而言，旁边那份 `_env.json` 才是它自己的，`summary.json` 是全 run 级的。

---

## S-6（medium）**所有外部控制脚本**在 `set_command_now()` 之后都不重取观测

`env.set_command_now()` 的语义（`env.py:392-419`）是：写入 `cfg.command._now`，并**刷新 `_hist[-1]`**。但 `obs` 是 numpy 数组，是**值**，刷新内部 `_hist` **改不到调用方手里那一份**。

| 脚本 | 位置 | 是否重取 `obs = env._observation()` |
| --- | --- | --- |
| `play.py` | `:319-324` `run_episode` | **有**（`obs = env._observation()`，且 docstring `:316-317` 专门解释了为什么要这样） |
| `teleop.py` | `:150-199` `run_teleop` | **没有**：`cmd = controller(env, step)` → `set_command_now(...)` → `env.step(runner.predict(obs))` |
| `cmd_demo.py` | `:100-132` `run_demo` | **没有**：`:110` 设指令后直接 step；`:126-127` 的 `reset()` 分支同样不重取 |

### 影响

* 受影响的是**每段/每回合的第一帧**（以及每次 `reset` 后的第一帧）：该帧的动作是按**旧指令**算出来的。
* 在 `cmd_demo.py` 里这会**污染统计结论**：脚本的输出是逐段的 `slope_and_r(cmd_vy, vy)` 斜率与相关系数（`:87-97`），而每段第一帧的「指令-实测」对是错配的。对 `hold_steps` 很短的段落，一帧的权重不可忽略。
* `play.py` 已经给出了正确写法，属于**同一仓内已被解决、但没回灌到另外两个脚本**的问题。

### 修复

两个脚本在 `set_command_now(...)` 之后各补一行 `obs = env._observation()`（`reset` 分支同样补）。更彻底的做法是在 `env.set_command_now()` 里返回新的观测（现在返回的是 `list(self._command)`），让「改了指令就要重取观测」变成调用方绕不过去的形状。

---

## S-7（medium）`mode_eval` 不传 `command`：评估用的是**基线 vx 与随机 vy/wz**

`play.py:428-429`：

```python
out = run_episode(env, policy, episode=ep, seed=seed + ep, capture=False,
                  camera=CAMERA_EVAL, width=0, height=0)     # ← 没有 command=
```

于是 `run_episode` 走 `:321-322` 的兜底分支，用 `env.cfg.command` 当前那一条：`vx = cmd.vx`、`vy/wz = cmd._now[1:2]`。而 `_now` 在每次 `reset()` 里被 `_sample_command()` 重抽（`env.py:338-350`：在 `±vy_max / ±yaw_max` 上均匀采样）。

* **vx** 来自 `_env.json` / `summary.json` 里的基线（`0.55`）或旧归档的最后一级（`0.45`）—— **不是**该阶段训练时的 `curriculum.command_vx`。
* **vy / wz** 每回合随机，且 `split` 在 100 个回合上，与训练课程的分级结构无关。
* 对比 `gif` 模式：`:640` 的 `_stage_plan` 会用 `command_for_stage(st)`（`PLAY_COMMAND_FRAC = 0.8`）**逐级挑一条量程内的指令**，并把它传进 `mode_gif`。

### 影响

`--mode eval` 报出的成功率与 `--mode gif` 的画面**不是同一个测试条件**，而 `eval` 的数字看起来更权威（100 回合 vs 10 回合）。用 `eval` 判断「这个 checkpoint 会不会走楼梯」时，随机 `wz` 会让平地级的数字明显偏低，而台阶级的数字受 `vx=0.55`（比训练用的 `0.0` 大）影响。

### 修复

`mode_eval` 增加 `command` 形参，调用方按 `--stage`（或 `play.stages` 的第一项 / `_default_height_cm` 对应的级）用 `command_for_stage` 生成指令；或至少在打印里注明「本回合指令为随机采样，vx 取自产物基线」。

---

## S-8（low）小问题与卫生项

1. **`Trainer.make_env(warm_start=...)` 形参从未被使用**：真正的 warm-start 发生在 `train_stage()` 的 `model.set_parameters(str(warm_start))`。形参留着会让人以为「传进去就能从上一级地形起步」。
2. **`play.get("device")` 恒取不到值**：`config.yaml` 的 `play:` 段没有 `device` 键（设备在 `ppo.device`），而 `config_to_play()` 不校验未知键（C-1）。`play.py:607` 已经改成 `device(cfg)` 并在注释里说明，属于**已修但遗留读取点**；建议从 `play` 段字典里彻底删掉这个键的读取。
3. **`ProgressCallback` 只在回合结束且到 `log_every` 时才打印**（`train.py`：`due = self.n_calls - self._last_print >= log_every`，且要求 `new_episode`）⇒ 早期（回合长度可达 1000 步）可能长时间无输出，看起来像卡死。建议加一条「每 N 次调用至少打一次步数」的保底输出。
4. **`_stage_plan` 的 `"auto"` 会在 run 目录里挑 checkpoint**（`play.py:565-574`）：只要 `checkpoints/<阶段>_final.zip` 存在就渲染那些阶段，而这些 checkpoint 用的是**各自阶段的权重**，`main()` 却在每个阶段都加载**同一个 `target`**（`--run` 解析出来的那一个，通常是最后一级）。⇒ 默认行为下，「阶段列表来自 A 级 checkpoint」与「实际推理用最后的 `model_final.zip`」不一致（地形是对的，权重不是该级的）。不是崩溃，但会让 GIF 的地理与能力对不上。
5. **`cmd_demo.py` 的「超出课程范围」自谦是过头的**：`INPLACE_PHASES` / `WALK_PHASES` 的最大 |vy| = 0.35、最大 |wz| = 0.60，而课程上界是 0.40 / 0.70 —— **全部落在课程内**，但脚本在 `:156-157` 打印「课程范围之外的指令策略没见过，量到的不是课程内性能」。真正的局限是「分段恒定指令 ≠ 课程的分级结构」和上面 S-6 的统计污染。
6. **`cmd_demo.py` 的 docstring 用了旧字段名** `curriculum.command_vy_max`（`:18-19`）。
7. **`cmd_demo.py` 固定平地**（`:152` `flat_spec(tread=0.50, width=2.0, start_x=1.50)`），不跟随权重自带地形，与 `teleop.py` 的 `_stair_from_model()` 行为不一致（有意的可能性大，但没写）。
8. **`teleop.py` 的 `vy_limit/yaw_limit` 兜底值**（`0.25/0.35`）**硬编码**，不在 `config.yaml` 里；配合 S-5 会成为常态路径，建议改成可配置或直接取 `config.yaml` 课程的最大量程。
9. **`open_runner()` 的 `warn` 分支**（`play.py:168-169`）在缺 `*_vecnormalize.pkl` 时只打印一句警告就继续按原始观测推理；配合 S-4（级间不迁移统计）的语境，这条警告值得升级为「除非显式 `--allow-raw-obs` 否则报错」。
10. **`play.py` 的 GIF 帧全部留在内存**：`mode_gif` 把 10 个回合 × 每回合约 500 帧（`FRAME_STRIDE=2`，`episode_length_s=20` ⇒ 最多 500 帧/回合）累积进一个 `list`，`write_gif` 再统一缩放编码（`960×540×3 × 5000 帧 ≈ 7.8 GB` 量级）。当前默认参数下 `scale=2`（缩到 480×270）后仍有 ~1.9 GB。建议边生成边写 `PIL` 帧序列，或把 `frames` 落临时目录。

---

## S-9 已验证**没有**问题的地方

* **`build_ppo()` 的签名是纯显式关键字参数**：没有任何 `**kwargs`，所以 `config_to_ppo()` 带出的杂键不会被偷偷塞进 SB3 —— 它们只会被丢弃（这是 C-1/C-2 的另一面：静默但**不至于污染训练**）。
* **`load_policy()` 是唯一的评估加载路径**：`play.py` / `teleop.py` / `cmd_demo.py` 全部经它加载，且它同时读 `<model>_vecnormalize.pkl`（`ObsNorm`）与 `model_final.zip`，符合「一个加载函数」的契约。缺 `*_vecnormalize.pkl` 时会显式告警。
* **`Runner.predict()` 的归一化顺序正确**：先 `ObsNorm.normalize(obs)` 再 `model.predict(x, deterministic=True)`，与训练期 `VecNormalize(norm_obs=True)` 的位置一致；`model.policy.set_training_mode(False)` 也在构造时设了。
* **`CheckpointCallback` 同时保存权重与归一化统计**（`model.save(...)` + `save_vecnormalize(...)`），命名 `<阶段>_<步数>_steps` 可排序。
* **`train.py` 的「每级一份 cfg 副本」与「奖励覆盖从基线叠」是正确的**，且 docstring 记录了踩过的坑（共享 cfg 导致 `command_vx=0.0` 走 falsy 分支不压回、楼梯段沿用平地最后一级的 0.45）——这正是 S-5 里 `self.cfg` 被改写的根源，作者清楚这一点，只是没把它同步到写盘逻辑。
* **`download_go2.py` 无重要缺陷**：镜像回退（raw.githubusercontent → gh-proxy → jsdelivr）× 3 次重试、`assets/*.obj` 与 `go2.xml` 的存在性健全检查、`RESULT: OK/INCOMPLETE` 退出码都正常。仅两条环境事实：(a) 需要访问 GitHub API 与镜像，断网环境返回 1；(b) `WANT_FILES` 的条目 size 硬编码 0，`download()` 的 `if size and len(blob) != size` 对这四个文件**不做大小校验**，若镜像返回 200 + HTML 错误页会被当成正常文件写入（只有经 GitHub API 列出的 `assets/*` 才校验）。
* **`robotdog/tools/__init__.py:16` 只 import `console, paths, terrain`**，工具层没有任何 `robotdog.env` 依赖 ⇒ 分层契约（`scripts → robotdog 顶层 → robotdog/tools`）在运行时成立；`tools/__init__.md` 里出现的 `from robotdog.env import ...` 只存在于**文档**中。

---

## 复现命令

```powershell
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe .tmp\zz_probe_playdev.py        # S-1 的证据（只读，不写盘）
.\.venv\Scripts\python.exe .tmp\zz_audit_cfg_mutation.py   # S-4 的归一化幅度 + F2 撤回
.\.venv\Scripts\python.exe .tmp\zz_audit_stage_cfg.py      # S-5 的逐级配置对照
```
