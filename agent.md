# `agent.md` —— 在本仓里干活的须知

> 这份文件面向**在本仓做改动的 AI 代理与人类协作者**，只讲一件事：动手之前必须知道什么。
> 它不重复项目说明（那是 `README.md`）、不讲怎么训练（那是 `train.md`）、不讲踩过哪些坑的
> 来龙去脉（那是 `experience.md`）、也不讲每个模块怎么设计（那是 `docs/`）。
>
> **读这份文件不能替代读代码。**它的价值在于：让一次改动不会在验证阶段才炸，也不会重复
> 本仓已经付出过代价的错误。

---

## 1. 一句话定位

Unitree Go2 **盲走**箱式楼梯的残差强化学习实验仓库：`config.yaml` 是唯一的超参入口，
`robotdog/` 是 12 个模块，`docs/` 与之一一对应，四支校验脚本是回归门。

---

## 2. 职责边界

**这份文件负责**：仓库地图（该读哪份文件）、最短验证流程、代码与文档约定、
本仓特有的硬约束、以及「已经证明过不对的方向」。

**这份文件不负责**：

* 项目背景与训练结果 → `README.md`
* 训练/评估的详细流程与产物清单 → `train.md`
* 具体踩坑的完整复盘（症状 → 定位 → 修法）→ `experience.md`
* 单模块的 10 节设计文档 → `docs/robotdog/*.md`
* 领域知识（奖励每一项为什么存在）→ `docs/robotdog/reward.md` 与 `experience.md` §2

**红线**：不要新建与 `robotdog/` 平行的代码层；不要把超参再搬回 CLI；不要在
`robotdog/tools/` 里 import `robotdog.env`（工具层必须能被离线工具单独导入）。

---

## 3. 最短上手路径

```
README.md          项目是什么、怎么装、config.yaml 每个键什么意思
  ↓
train.md           怎么训练、产物在哪、冒烟跑法
  ↓
agent.md（本文）   动手改之前必须知道的约束与坑
  ↓
docs/robotdog/<你要改的模块>.md   那个模块的设计与公开 API
  ↓
robotdog/<你要改的模块>.py        代码本身（约 200~1100 行，可通读）
```

**仓库地图**

| 想改什么 | 看哪里 |
| --- | --- |
| 超参、课程高度、PPO 参数 | `config.yaml`（**唯一入口**） |
| 观测组成 | `robotdog/env.py` 的 `ObsConfig` + `_obs_frame()` |
| 奖励 | `robotdog/reward.py`（**全部奖励数学只在这一个文件**） |
| 台阶几何 | `robotdog/tools/terrain.py` |
| 步态先验 | `robotdog/gait.py` + `artifacts/trot_stairs.json` |
| 终止/成功判据 | `robotdog/env.py` 的 `_terminated()` / `_success_reached()` |
| 指令（横向/偏航）与外部控制器接口 | `robotdog/env.py` 的 `set_commands()` / `set_command_now()` + `scripts/teleop.py` |
| 指令跟踪的量化与演示 GIF | `scripts/cmd_demo.py`（分段恒定指令表 + 斜率/相关系数） |
| MJCF 组装、kp/kd、执行器 | `robotdog/go2_model.py` |
| 训练循环、产物、权重加载 | `robotdog/training.py` + `scripts/train.py` |
| 评估与 GIF | `scripts/play.py` |
| 读配置 | `robotdog/config.py` |

**代码规模**（`README.md` §6.1 有完整表）：最大的几个是 `robotdog/env.py` 1101、
`scripts/play.py` 856、`robotdog/config.py` 527、`robotdog/go2_model.py` 467、
`robotdog/policy.py` 339。最大的文件也可以直接通读。

---

## 4. 运行环境

* **解释器必须是 `.venv\Scripts\python.exe`**，系统 Python 是 3.13 且没有装依赖。
* 依赖见 `requirements.txt`（8 个包，与 `README.md` 的依赖表逐项一致）。
* **训练/播放只认两个分段 CLI**：

```bat
.venv\Scripts\python.exe scripts\train.py --run-name <名字> [--config <yaml>]
.venv\Scripts\python.exe scripts\play.py  --run <名字> --mode eval|gif [--config <yaml>]
```

  超参一律写进 `config.yaml`（或另一份同结构的 yaml 传给 `--config`），**没有别的开关**。
  用 `--config` 指一份小配置跑冒烟是最省时间的回归手段（见 §7）。

---

## 5. 硬约束：这台机器上的坑

这些坑本仓每一个都踩过，**它们是环境事实，不是代码缺陷**。

**① PowerShell 与中文**

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
```

不设这两行，`Get-Content` 会把 UTF-8 文件显示成 GBK 乱码（`鍐呴儴杈呭姪` 之类）。
**文件本身没问题**，别把乱码当成文档腐坏去"修"。

**② 不要内联 `python -c`**

`pwsh -Command` 会吃掉反引号和花括号，含 markdown 行内代码（`` `x` ``）或 dict 字面量时
必然 `SyntaxError`。**一律写 `.tmp/xxx.py` 再跑。**

**③ 重定向编码**

pwsh 的 `>` 写出的是 **UTF-16**。要保存 git 里的文本内容用：

```bat
cmd /c "git show HEAD:robotdog/env.py > .tmp/env_before.py"
```

否则再读进来会 `SyntaxError: source code string cannot contain null bytes`。

**④ 长任务必须在受管后台任务里跑**

`Start-Process` 起的长任务**不跨工具调用存活**（进程树被回收，日志停在几千步），
必须用受管后台任务并设 `PYTHONUNBUFFERED=1`，否则 stderr 会被块缓冲吞掉，
看起来像"训练停滞"。

**⑤ 并发上限是 2 臂**

本机 15.7 GB 内存。**4 臂 × 8 env = 32 个并发 MuJoCo 会打爆内存**，报出的是
`ValueError: Error: engine error: Could not allocate memory`，被
`robotdog/go2_model.py` 包成 `RuntimeError: MJCF compile failed; dumped to
.generated/failed_scene.xml` —— **这不是几何缺陷**，不要浪费时间去查地形。
同时跑 2 臂（16 env）是安全上限。

**⑥ 沙箱**

* 只支持 `DummyVecEnv`（沙箱禁止 Windows 命名管道，`SubprocVecEnv` 会
  `PermissionError: [WinError 5]`）。
* `pip install` 写系统临时目录会被拒（`PermissionError: [Errno 13]`），需要一次性提权。

**⑦ 行尾是 CRLF**

本仓 `core.autocrlf=true`，**约定 CRLF**。用 `write` 工具整篇重写文件会写出裸 LF，
提交前跑一次：

```bat
.venv\Scripts\python.exe .tmp\fix_eol.py
```

**⑧ 平地地形不能写 0 cm 台阶**

`demo_stairs_spec(step_height=0.0)` 会 `RuntimeError: MJCF compile failed`（零高度 box 非法）。
平地必须用 `flat_spec(n_up=0, n_down=0, tread=0.50, width=2.0, start_x=1.50)` ——
`robotdog/config.py` 的 `config_to_stages` 已经这么做了。

---

## 6. 代码与文档约定

**代码**

* 注释与 docstring **一律中文**，用真中文标点。（唯一允许留英文的是：装饰性分隔线
  `# ---- #`、`# noqa: ...`、形状注释 `# (12,)`、库名横幅 `# ---- Gymnasium`。）
* 公开 API 用 `__all__` 显式列出；**不用的 import 直接删**。
* 删除符号前的判据是**「有没有任何读者」**，而且要看三类**假死**（见 §8.4）。
* 模块依赖是**单向**的：`scripts → robotdog 顶层 → robotdog/tools`。

**文档**

* `docs/robotdog/<模块>.md` 与 `robotdog/<模块>.py` **严格一一对应**（含 `tools/` 子包），
  每份固定 **10 节**、顺序不变、标题必须含模板里的关键字（`audit_docs.py` 会检查）。
* 文档引用代码时，**优先写 `路径/文件.py` 加符号名（`` `robotdog/env.py` 的 `step()` ``）**，
  行号只在「要特指某一行实现」时才写。行号是最容易腐坏的东西（源文件加一段 docstring
  就能让全仓引用整体漂移），而 `check_lines.py` **抓不到漂移**——它只报「越界 / 是空行」。
  要抓漂移得看 `.tmp/ref_truth.py`（把每个 `文件.py:行号` 指向的**真实内容**打出来）。
* 根文档只有四份：`README.md`（是什么 + 怎么用 + 代码规模）、`train.md`（怎么训 + 产物
  + 实测成绩）、`experience.md`（症状 → 根因 → 修法的复盘）、`agent.md`（本文件）。

---

## 7. 改完之后必须跑的验证

`.tmp/` 里这六支脚本是回归门，改任何东西之后全跑一遍（都在 1 秒内）：

```bat
.venv\Scripts\python.exe -m compileall -q robotdog scripts
.venv\Scripts\python.exe .tmp\check_lines.py       :: 文档里的 文件.py:行号 是否越界/指向空行
.venv\Scripts\python.exe .tmp\ref_truth.py         :: 每个行号指向的真实内容（抓「行号漂移但没越界」）
.venv\Scripts\python.exe .tmp\check_doc_code.py    :: markdown 里的 ```python 块能否 ast.parse
.venv\Scripts\python.exe .tmp\check_xrefs.py       :: 文档之间的相互引用是否存在
.venv\Scripts\python.exe .tmp\audit_docs.py        :: 10 节结构 / 交叉引用 / 索引完整性
```

`check_lines.py` 有**两个盲点**，所以不能只靠它：①它发现不了「文档描述了一个不存在的 API」
（表格里一行的过期它看不见）；②它发现不了「行号整体漂移但没越界」——在文件上游插一行，
下游所有行号都会指到**别的函数**，而每一行都是合法的非空行。`ref_truth.py` 补第二个盲点：
它**现算**每个 ``文件.py:行号`` 指向的那一行是什么、并对照 AST 判「这一行是不是另一个
`def`/`class` 的定义」，所以不需要任何手工维护的符号快照（快照一旦过期，门本身就成了
伪报的来源）。
**改完 `robotdog/env.py`**（增删任何一行都会让 `Go2StairsEnv` 成员行号整体漂移）之后，
顺手跑一次幂等的 `.tmp/sync_env_md_lines.py`，它按符号名把 `docs/robotdog/env.md` 里
成员的定义行号刷成真值。

两支门都**能解析裸文件名**（`` env.py:592 `` 会按「仓根 → `robotdog/` → `scripts/` →
`docs/` → 全仓唯一同名」找到 `robotdog/env.py`）。这一点是 2026-02 才补上的：在此之前
它们只试 `ROOT/raw`，于是文档里几十条裸名引用**从未被检查过**。

全绿的样子（数字随仓库演化，**以当前实跑为准**，别照抄历史值）：

```
stale/blank line refs: 0   未解析的引用（可能指向已删文件）: 2
refs checked: 329   suspicious: 0   未解析（可能指向仓外或已删文件）: 2
python blocks: 71, unparsable-skipped: 38, BROKEN: 0
BROKEN CROSS-REFS: none
RESULT: ALL CHECKS PASSED
```

唯一允许出现在「未解析」里的，是**仓外引用**（`stable_baselines3/...` 那两条）——引用
本身正确，只是解析器不扫 `.venv`。除此之外 `check_lines.py` 与 `ref_truth.py` 都应当是
**零条目**；文档里**不要**再出现「历史上某个错行号」那种举例，否则它会被门一直报出来。

**行为回归**（改了环境/奖励/训练时必须跑）：

派生配置必须**现算**，不要留快照——`config.yaml` 的字段集合一变，旧快照就会被段键校验
拦下（实测 `无法识别的键 ['camera', 'fps', 'height', 'scale', 'width']`）。

```bat
:: 0) 先生成冒烟配置（六级各 2000 步；播放只跑 5 回合，默认 100 回合太慢）
.venv\Scripts\python.exe .tmp\mk_smoke_config.py
.venv\Scripts\python.exe .tmp\mk_smoke_play_config.py

:: 1) 训练冒烟：六级各 2000 步，约 30 秒
.venv\Scripts\python.exe scripts\train.py --run-name smoke_cfg --config .tmp\smoke_config.yaml

:: 2) 播放冒烟：自动读权重自带的 <模型>_env.json 选地形
.venv\Scripts\python.exe scripts\play.py --run smoke_cfg --mode eval --config .tmp\smoke_play.yaml
```

跑完**删掉** `runs/smoke_cfg` 与 `output/gifs`（`.gitignore` 不覆盖 `output/`）。

**改了奖励或观测数学，还要做等价性证明**：用 `git show HEAD:<path>` 存下改前版本，
以 `importlib.util.spec_from_file_location` 建私有模块，对固定动作序列
`np.random.default_rng(20261008).uniform(-1, 1, (400, 12))` 在 0/5/8/11 cm × 3 个 seed 上
并排跑，要求 `max|Δreward| = 0.000e+00`、`max|Δobs| = 0.000e+00`、终止标志零差异。
参考实现见 `experience.md` §5.9。

**`check_lines.py` 只报症状不报病因**：它只能发现"行号越界"和"这一行是空行"，
发现不了"函数根本不存在""常量少了一半"。**符号存废必须人工对着源码核**。

---

## 8. 本仓的领域事实（写文档或做判断前先看这里）

这些都是实测结论，不是推测。**不要重新论证，也不要与之相反的断言。**

### 8.1 观测

* 当前 `config.yaml`：`history_len: 1` ⇒ **48 维**（每帧 `base_frame_size=42` + `base_lin_vel_body` 3 + `step_height` 1 + **指令块 2**）。
* 观测是**盲走**的：48 维全部有真机出处（关节编码器 / IMU / 上一步动作 /
  本体感觉估计 / 上层指令）+ 一个标量任务参数 `step_height`；没有触觉、没有射线、没有接触力读数。
  逐项出处与「哪些量是真值但只进奖励」见 §8.6。
* **`obs.base_height` 被硬性禁止**（`ObsConfig.__post_init__` 抛 `ValueError`）：
  它含地形真值。见 §8.6。
* **`obs.command` 恒真**：那 2 维是 `[vy, wz]` 指令（`vx` 定值、不进观测）。PPO 的网络与
  RolloutBuffer 维数不可变，所以**课程只能改指令范围不能开关这 2 维**。
* **单帧（48 维）不亏**：课程内每级都不差，3 cm 62% → 92%、6 cm 0% → 22%；课程外
  3 cm / 5 cm 都是 100%（10 帧基准是 50% / 0%）。50 Hz 下相邻帧几乎相同，10 帧绝大部分冗余。
  代价是失去帧内的隐式速度信息。

### 8.2 步态先验不能去掉

* `env.gait_prior: false` ⇒ 完整课程**全 0% 成功、0% 摔倒、每回合跑满 1000 步**，
  `mean|a| ≈ 0.09`（基准 0.18）—— 机器人原地站到超时，一次都不摔。
* 原因是 `action_scale = 0.30 rad` 是**按残差**设计的；去掉先验后网络要同时从零学步态
  与平衡。真要去掉，得先把 `action_scale` 放大到接近 gait 自身的 1.0 rad 并重训。
* 这条已经写进 `config.yaml` 的注释里，**不要再当成一个"可以试试的开关"**。

### 8.3 能力带与健康指标

* 本仓 `config.yaml` 从零训练的能力带约 **5~6 cm**；参考 checkpoint（论文式 660 维 +
  专门调参）约 **10 cm**。
* `mean|a|` 是判断"学习是否真的发生"的关键量：**≈ 0.42 代表真的在学**，0.02~0.10 意味着
  策略被压回站桩。11 cm 一直过不去，且开环调参先验（18 组 × 3 高度）**全部失败** ——
  缺的是摆动相里的向前落脚，不是抬腿高度。

### 8.4 删代码时的三类假死

零引用 **≠** 可删。这三类必须留：

1. **字符串派发**：调用点写的是 `venv.env_method("某个方法", ...)`，静态扫描看不见。
   （本仓目前 **没有**这样的调用点，但这是最容易误删的一类，改 `DummyVecEnv` 相关代码时留意。）
2. **框架回调**：`scripts/train.py` 里两个 `_on_step` —— SB3 `BaseCallback` 钩子。
3. **文档化的公开 API**：`GaitParams.static()`（**文档里「怎么关掉步态先验」的唯一正确写法**，
   删了文档就在教一个不存在的函数）、`GaitParams.targets()`、`policy.Policy`、
   `StairSpec.down_end_x`、`Go2StairsEnv.yaw`、`Go2StairsEnv.set_stair()`。

   > `set_stair()` 的处境值得记一笔：它**曾经**有调用者（`training.set_terrain()`），
   > 而 `set_terrain` 是零读者所以被删了，于是 `set_stair` 变成零调用点。
   > 但「换地形」本身是真需求（当前课程每级是 `make_vec_env` 新建环境，不是换地形），
   > 而且 `docs/robotdog/env.md` 把它当公开 API 记录，所以**保留**。
   > 判据是「零读者 **且** 不是文档化契约」才算死。

**反过来**：`GO2_SCENE_XML`、`ensure_dir()`、`ensure_output_dirs()`、`load_ppo()`、
`mean_abs_action()`、`set_terrain()` 是真的零读者，已经删除。**不要把它们加回来。**

### 8.5 其他已确认事实

* **kp / kd 是事后核算的**，来自 vendored Go2 MJCF 的作动器风格，本仓没做自动整定：
  hip/thigh 用 `(30, 0.6)`、calf 用 `(45, 0.9)`；对应阻尼比 0.30 / 0.31 / 0.53。
  推导表在 `README.md` §3.1。
* **归档 checkpoint 与新代码不兼容**：`runs/course22_ref/*` 是 660 维、
  `runs/course22_improved/*` 是 740 维、`runs/normab_*` 是 420 维，而当前是 48 维；
  且它们**都没有 `_env.json`**。**加载它们会 shape mismatch，这是预期行为，不是回归。**
  端到端评估只能用本轮自己训出来的 run（如 `runs/ab_base`、`runs/ab_hist1`）。
* `robotdog/training.py` 的 `load_policy()` **会把 `*_vecnormalize.pkl` 装回模型自带的
  `VecNormalize`**。曾经没装，导致 `model.get_env()` 评估等于喂未归一化观测，
  与 `play.py` 结论相反。**新增任何评估路径都要走 `load_policy()`，不要自己搭。**
* **躯干接触不再终止回合**（`_contact_fatal()` 恒 `False`；`terminate_contact_steps` 字段已删）。
  实测那条「连续 10 步」规则最长只到 8 步、从未生效，却随时会误杀「上台架时蹭立面」。
  `terminate_min_height` 现在是 `0.15`（躯干碰撞体最低点在 0.118 m）。
* **`body_collision` 从 `-4.0` 降到 `-1.2`**：原值让「贴着立面蹭」比站桩还亏，GIF 复盘里
  6/12/15 cm 的策略学会试两下就放弃。
* **奖励里的指令跟踪项必须门控**（`if cmd.vy_max:` / `if cmd.yaw_max:`）。不加门控时
  机器人直走就能白拿约 +1.0/步（实测 5 cm 上 125 步拿到 `lateral_vel=+139.73`），
  **等于又造一个「原地拖延」正项**。`env.yaw_rate` 也不是 `qvel[5]`，见 `experience.md` §13。
* **`probe_lift` 的起算线只跟策略当前能力比，不跟台阶高走。** 实测在 15 cm 立面之前
  被阻挡时，默认步态摆动峰值只到 0.09~0.12 m，而 `action_scale=0.30` 下搜遍固定动作
  最好也只到 0.148 m —— 原来的「`1.6 × 台阶高` = 0.24 m」门槛**不可达、梯度恒为零**。
  现在是势函数塑形（只在被阻挡时按「峰值又提高了多少」结算，起算线 0.06 m），
  实测梯度 9 cm 上 0.39~1.19/回合、15 cm 上 0.14~0.91/回合。见 `experience.md` §15。

### 8.6 策略能看到什么、看不到什么（信息来源清单）

本仓是**盲走**的。任何人碰 `_obs_frame()` / `reward.py` / `_fall_reference_height()`
之前，先看这张表 —— 「这个量是真机拿得到的吗」是本仓最容易悄悄搞错的一件事。

**`_obs_frame()` 里的 48 维，逐项都有真机出处：**

| 观测块 | 维 | 真机出处 |
|---|---|---|
| `joint_pos` | 12 | 关节编码器 |
| `joint_vel` | 12 | 编码器微分 |
| `base_rpy` | 3 | IMU 姿态（重力对齐） |
| `base_ang_vel_body` | 3 | IMU 陀螺 |
| `_last_action` | 12 | 上一步自己下发的指令 |
| `base_lin_vel_body` | 3 | 本体感觉估计（腿式里程计） |
| `[vy, wz]` 指令 | 2 | **上层控制器下发的指令**（遥控） |
| `step_height` | 1 | **任务参数，不是传感器读数** |

`step_height` 是唯一一个非传感器项，但它是**标量任务参数**：真机上可以随任务下发
（「现在爬 8 cm 的楼梯」），所以可接受。它**不含**「下一级立面在哪」。指令那 2 维同理：
它是别人给机器人的目标，不是机器人偷看的环境真值。

**五个使用真值、但只进奖励/终止、绝不进观测的量**（这是允许的：奖励与终止本来就
是上帝视角；**一旦写进 `_obs_frame` 就破坏盲走**）：

| 量 | 真值来源 | 只被谁用 |
|---|---|---|
| `_contact_flags()` | MuJoCo 接触对（躯干/腿 × 地形） | `body_collision` / `leg_collision` 奖励（**都不再终止回合**） |
| `_foot_contact_mask()` | MuJoCo 接触对（足端 × 地形） | `air_time` 奖励、腾空时间统计 |
| `_support_height()` | `stair.height_at()` 地形几何 | `_fall_reference_height()` → `_terminated` |
| `_support_height_mean()` | 同上 | 同上（可选分支） |
| `foot_clearance` | 足端位置 − 地形高 | `foot_lift` 奖励、`probe_lift` 奖励的摆动峰值 |

接触类两项在真机上有标准的等价估计（关节力矩残差 / 电流尖峰 / IMU 冲击），
属于可接受的 sim-to-real 折衷；**地形几何类三项（`foot_clearance`、两个
`_support_height*`）是真特权信息**，策略只能学到「按 `step_height` 抬够高」
这条相关性，拿不到逐点地形。

**`obs.base_height` 现在是硬错误**（`ObsConfig.__post_init__` 抛 `ValueError`）。
它原来的实现是 `base_pos[2] - _height_reference()`，而 `_height_reference()` 就是
`terrain_height_under_base` —— 直接把地形真值塞进观测。该分支与
`_height_reference()` 本身都已删除。**不要再把它加回来。**

---

## 9. 已知限制与坑

* **`check_lines.py` 抓不到结构性腐坏**（§7 已述）：文档整段描述了一个不存在的 API
  时，行号检查可能一条都不报。所以**改完 API 要人工读一遍对应文档**。
* **`deadcode` 类脚本必然误报 dunder 与框架钩子**（`__post_init__` / `__repr__` /
  `_on_step`），别照着它的输出删。当前 `.tmp/dead3.py` 报的 7 条**全是假死**
  （详见 §8.4）。
* **`.tmp/` 是 gitignored 的**（`configs/`、`runs/`、`logs/`、`.generated/` 也是）：它是
  工具与备份的家（`.tmp/knob_backup/` 存着已删旧开关的术与补丁，是唯一的恢复途径），
  但**不要指望提交它能进版本库**——重要的人工结论要写进 `docs/`、`experience.md`
  或本文，而不是留在 `.tmp/` 的脚本注释里。（它已经攒了 570 个条目，是事实上的
  ledger，这也是「结论必须落进文档」的原因。）
* **`output/gifs/` 不在 `.gitignore` 里**（它是交付物），所以冒烟必须自己清干净，
  否则会把渲染中间产物当成改动提交。
* **文档行号会随任何编辑漂移**。改完源码行数就重跑 `check_lines.py`；批量替换时用
  「唯一子串 + assert 命中一次」的脚本，不要按行号盲改。
* **`experience.md` 是复盘文档**，里面的旧架构名字（`Curriculum`、`StageTrainer`、
  `--resume`、`--stage`、`action_mode`、`touch.py`）是历史记录，**不要照着实现**。
  当前架构见 `docs/`。

---

## 10. 一次典型改动应该长什么样

需求：「把 `foot_lift` 的目标高度从我 0.08 m 改成 0.12 m。」

```bat
:: 0) 先确认这个参数在哪、有没有别的读者
rg -n "foot_lift_target" robotdog/ config.yaml docs/
::    -> robotdog/reward.py 的 RewardConfig（定义）
::    -> config.yaml 的 reward 段（值）
::    -> docs/robotdog/reward.md（文档）

:: 1) 只改 config.yaml 的值就够了（它本来就是可配项）
::    config.yaml: reward.foot_lift_target: 0.12

:: 2) 确认配置真的生效（不会因为键名拼错被静默忽略）
.venv\Scripts\python.exe -c "from robotdog.config import load_config, config_to_env; print(config_to_env(load_config()).reward.foot_lift_target)"
::    -> 0.12

:: 3) 行为回归
.venv\Scripts\python.exe scripts\train.py --run-name smoke_cfg --config .tmp\smoke_config.yaml
.venv\Scripts\python.exe scripts\play.py --run smoke_cfg --mode eval

:: 4) 文档回归
.venv\Scripts\python.exe .tmp\fix_eol.py
.venv\Scripts\python.exe .tmp\check_lines.py
.venv\Scripts\python.exe .tmp\check_xrefs.py
.venv\Scripts\python.exe .tmp\audit_docs.py

:: 5) 清掉冒烟产物（runs/ 被 .gitignore 忽略，output/gifs/ 不是）
rd /s /q runs\smoke_cfg
rd /s /q output\gifs
```

**如果第 1 步发现要改的是代码里的默认值**（而不是配置值），那就多两步：
用 `git show HEAD:` 存基线、跑 §7 的等价性证明，并在 `experience.md` 里补一节
记录"为什么改"。
