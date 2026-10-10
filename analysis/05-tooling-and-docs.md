# 05 · 工具层、文档与仓库卫生

审计对象：`robotdog/tools/{__init__,console,paths,terrain}.py`、`robotdog/gait.py`、`robotdog/go2_model.py`、`agent.md`、`README.md`、`train.md`、`experience.md`、`docs/`（14 份）、`.tmp/` 里的六支自查脚本、`.gitignore`、`requirements.txt`、`runs/` 与 `output/` 的产物形状。

编号 `T-*`（tooling / docs / tests），与 `00-index.md` 汇总表对应。

---

## T-1（high）**没有任何行为断言**：仓库的「验证」全部是文档结构校验

### 现状

* 仓内**没有 `tests/` 目录，也没有任何 `test_*.py` / `*_test.py`**（全仓 `glob "**/test*.py"` 只命中 `.venv/` 里的第三方包）。
* 仓库自带的验证手段是 `.tmp/` 下的六支脚本（`agent.md §7/§9` 引用它们）：`check_lines.py`、`check_doc_code.py`、`check_xrefs.py`、`audit_docs.py`、`ref_truth.py`、`fix_eol.py`（另有 `sync_env_md_lines.py`）。`.dsh/skills/legged-sim-rl-build/assets/gates/` 是同一套的副本。

### 本会话实跑结果（`.venv\Scripts\python.exe`，cwd=仓根）

| 门 | 检查什么 | 结果 | 退出码 |
| --- | --- | --- | --- |
| `check_lines.py` | 文档里引用的 `<文件>:<行号>` 是否存在、是否空行、是否有符号锚点 | **`stale/blank line refs: 4`** | **1（失败）** |
| `check_doc_code.py` | 文档里 ```python 块能否 `ast.parse` | `python blocks: 71, unparsable-skipped: 38, BROKEN: 0` | 0 |
| `check_xrefs.py` | 文档间的交叉引用目标是否存在 | `BROKEN CROSS-REFS: none` | 0 |
| `audit_docs.py` | 13 份文档的 10 节骨架 / 交叉引用 / 索引完整性 | `RESULT: ALL CHECKS PASSED` | 0 |
| `ref_truth.py` | 文档引用的**行内容**是否与断言一致（唯一做内容比对的门） | `refs checked: 244   suspicious: 17` | **0（即使有 17 条也可疑）** |

`check_lines.py` 的失败明细（引用指向**空行**，说明这些行号已经漂移）：

```text
experience.md:502            scripts/play.py:232 是空行
docs\robotdog\env.md:599     scripts/play.py:157 是空行
docs\robotdog\tools\paths.md:65   scripts/play.py:232 是空行
docs\robotdog\tools\paths.md:157  scripts/play.py:232 是空行
```

实测 `scripts/play.py:157`（`:156` 与 `:158` 之间）与 `:232` 确实是空行；被引用的 `open_runner` 现在在 `:158`、GIF 写出相关代码已经挪到别处。⇒ **`agent.md`/`README.md` 里「验证门全绿」的说法与实测不符**：`check_lines` 现在是红的（4 处），而它对 `scripts/play.py` 的引用集中在这两个已漂移的行号上 —— 恰好与 S-1（`play.py:654` 的 device 传参错误）发生在同一个文件、同一轮改动里。

### `ref_truth.py` 的两个结构性问题

1. **它在有 `BAD` 条目时仍然 `exit 0`**（`refs checked: 244   suspicious: 17`，退出码 0）。也就是说它**不能当回归门用**，只能当人工排查工具；`agent.md §7` 把它列为验证门会给人「跑了就安全」的错觉。
2. **它对「指到 `def`/`class` 行」的引用一律降级为 `?` 并把内容比对判为可疑**，实测 17 条里绝大多数是这类**误判**（文档引的是函数入口，内容确实就是那一行的 `def`）：

   ```text
   ?  README.md:130    robotdog/env.py:893  'def _target_from_action(...)'
   ?  README.md:144    robotdog/env.py:253  'self._kp = np.tile(self.cfg.kp, 4)'
   ?  env.md:22        robotdog/env.py:585  'def _compute_reward(...)'
   ?  env.md:65        robotdog/env.py:883  'def _gait_action(self) -> np.ndarray:'
   ?  reward.md:53     robotdog/env.py:585  'def _compute_reward(...)'
   ?  gait.md:75/399   robotdog/env.py:893  'def _target_from_action(...)'
   ?  training.md:85   robotdog/env.py:320  'def set_stair(...)'
   ```

   真正有信息量的只有两类：`BAD` 里的**空行**（= `check_lines` 已捕获）与**期望符号与实际不符**：

   ```text
   BAD  docs\robotdog\env.md:318  robotdog/env.py:210  期望含 'def _build_model'  实际: 'reward: RewardConfig = field(default_factory=RewardConfig)'
   ```

   `_build_model` 实测在 `robotdog/env.py:303`，所以 `env.md:318` 引用的 `env.py:210` **漂移了 ~93 行**（这正是「加了几行 docstring，全仓行号整体下移」的典型后果，`README.md:566-567` 与 `experience.md:566` 都记录过同类事故）。

### 结论与建议

* 当前四支「结构门」+ 一支「内容门」的组合，只能保证**文档骨架与代码块的语法**，**不能**保证：
  * 奖励数值/恒等式（`reward == sum(reward_terms.values())`）、
  * 观测维数与逐项来源、
  * 动作流水线的唯一性、
  * 课程各级的指令/奖励覆盖真的生效、
  * `play.py --mode gif` 真的能跑完一个回合。
* 后三条恰恰是本次审计发现 blocker/high 的地方（S-1、S-4、S-5）。**一次冒烟测试（`play.py --mode gif --stage flat` 跑 1 个回合 + 断言 GIF 文件存在）就能挡住 S-1**。
* 建议至少加一个 `tests/test_smoke.py`（pytest 已在 `requirements.txt` 的语境里可用）：`Go2StairsEnv` 建环境 → `step` 3 次 → 断言 `reward == sum(terms.values())`、`obs.shape == (cfg.obs.size,)`、`open_runner(...)` 能被调用（不加载真实权重，用 `--run` 的最小形状）。再把 `check_lines.py`（退出码 1 时）接进 CI。

---

## T-2（high）文档的行号锚点已经腐烂，而「规模表」刚刚被修好

### 已经修好的部分（不要再说它错）

审计开始时（见 `00-index.md` 的「快照状态」）`agent.md §1` 的规模表写 `env.py 895`、`play.py 462`，`README.md §6.1` 写 `play.py 482`。**现在 `agent.md:68-69` 已经改成 `robotdog/env.py` 1066、`scripts/play.py` 665**，`README.md:472-484` 的完整表也改成了 1066 / 665 / 460 / 443 / 375，与实测一致（`play.py` 在审计期间又从 665 长到 **667**，属于正常开销）。

⇒ **规模表这一条应记为「已随本轮改动修正」**。仍留在文档里的自相矛盾是：

* `README.md:512` 的小节标题仍是「**`env.py` 为什么还有 895 行**」，正文 `:514` 又写「`env.py` 的构成是…」—— 标题的 895 与同表格里的 1066 冲突；
* `README.md:535` 讨论「`render()` 的 59 行」：实测 `robotdog/env.py` 的 `render()` 是 `:1001-1059`（**59 行**，这条对）；但表格里 `env.py` 是 1066 行，说明作者**只改了数字没改正文**。

### 仍然腐烂的部分（实测）

* **行号锚点**：`check_lines.py` 报 4 处空行引用（T-1 的表里已列），全部集中在 `scripts/play.py`（`:157`、`:232`）。被引用方正是**审计期间被改动过的文件**。
* **`env.md:318`** 的 `robotdog/env.py:210` 期望 `def _build_model`，实际 `_build_model` 在 `:303`（`env.md:319` 接着引用的 `env.py:37` 是对的，所以只错这一条）。
* **`docs/robotdog/env.md` 的行号表**与实测的对应关系仍然成立（抽查 `:556` 的 `joint_pos → env.py:482` 与 `:574` 的 `_observation() → :580`：源码里 `joint_pos` property 在 `:490-493`、`_observation` 在 `:580`，**后者对、前者偏 8 行**）。
* **`README.md:130/144/145`** 引用的 `robotdog/env.py:893-911`（`_target_from_action`）、`:253-254`（`_kp/_kd`）、`:178-179`（`kp/kd` 元组）—— 实测这三处**都对**（`:893`/`:253`/`:178` 逐行命中），属于 `ref_truth.py` 的误判，不是文档的问题。

### 建议

把行号锚点的维护成本降到 0：文档里**优先写符号名**（`` `robotdog/env.py` 的 `_target_from_action()` ``），行号只在有自动同步机制时保留。`agent.md:170` 已经写了这条约定，但 `README.md`/`docs/` 里仍有大量裸行号（`ref_truth.py` 数出的 244 条引用里绝大多数是行号）。`fix_eol.py`/`sync_env_md_lines.py` 这种「机械同步行号」的脚本无法判断语义，只能推迟腐烂。

---

## T-3（medium）漂移类问题的固定模式：**加了 docstring，行号整体下移**

三份文档各自记录了同一次事故（`README.md:566-567`、`experience.md:566`、`agent.md §6`）：给 `robotdog/env.py` 的 `set_stair` docstring 加长 5 行，全仓 14 处行号一起失效（`env.py:531` / `env.py:827` 全部指到了别的函数）。

本次实测的同类残留（都可归入这一类）：

```text
env.md:318   robotdog/env.py:210  期望 def _build_model   → 实际在 :303
play.py 的 4 处引用指向空行（:157 / :232）
```

⇒ 结论不是「作者不认真」，而是**用行号做锚点的方案本身在高频改动的大文件（`env.py` 1066 行、`play.py` 667 行）上不可持续**。建议把「引用必须写成 `路径:符号名`」升级为硬约定（`agent.md:170` 已有雏形），并让 `check_lines.py` 在发现裸行号引用时**至少打警告**（现在它只验行号是否为空行）。

---

## T-4（medium）`.tmp/` 是事实上的实验台账（约 500 个文件），与 `agent.md §9` 的约定冲突

* `.tmp/` 在 `.gitignore` 里（第 7 行），所以**不进版本库**；但实测目录里有约 500 个文件：`diag_*` / `probe_*` / `fix_*` / `verify_*` / `audit_*` 一次性脚本，以及 `*_train.log` / `smoke*.log` / `ab_*.log` 训练日志，还有本轮审计新增的 `zz_*` 探针。
* `agent.md §9` 明确写「**重要结论不要留在 `.tmp/`**」，而实际上一批重要结论（A/B 对照 `ab_*.log`、`runs/course22_*` 的来源分析、`search_trot` 的产物来源）只在 `.tmp/` 里可查。
* 这与 `.gitignore` 的注释（「训练/评估产物：全部可以重建，不进版本库」）也冲突：`.tmp/` 里的实验**不能重建**（脚本是一次性的、日志是运行结果）。

### 建议

把「结论」搬进 `experience.md`（它已经是 1151 行的实验日志），把「可复跑的实验」搬进 `scripts/` 或 `tests/`，`.tmp/` 只留本次会话的临时文件。这条纯粹是卫生问题，但它让「这个仓的证据链在哪里」变得不可回答。

---

## T-5（medium）产物形状与文档口径不一致：`runs/`、`output/gifs/`、`gifs/<阶段>_<批次>` 三套命名同时存在

实测：

* `config.py:430-432` 的注释说 `play.batch` 的产物落在 `runs/<run>/gifs/<阶段>/<batch>.gif`（**阶段目录 + 批次文件**）；
* `play.py:523-529` 的 `_gif_dir_for()` 实现的是 `gifs_root / stage.label`，而 `gifs_root` 在 `:628-638` 会被拼上 `batch` ⇒ 实际是 `runs/<run>/gifs/<批次>/<阶段>_<高度>cm/<阶段>_<高度>cm.gif`（**批次目录 + 阶段目录**）；
* `play.py:485` 的 `--batch` help 文本又写「产物落 `runs/<run>/gifs/<阶段>_<批次>/`」（**第三种说法**，且与 `:529`+`:636` 的组合不符）。

三处说法互不相同，而 `output/gifs/` 里现存的两个文件（`course3_lateral_hold.gif`、`course3_yaw_hold.gif`）是 `teleop.py` 的产物（`paths.OUTPUT_GIFS_DIR`），**不在任何一套 `runs/<run>/gifs/...` 命名里** —— 也就是说仓库里同时存在三套 GIF 落点。

### 建议

以 `play.py` 的实现为准，把 `config.py` 的注释与 `--batch` 的 help 改成实测路径；或反过来把实现改成注释描述的形状。**关键是选一套**，并让 `--gifs-dir` 的行为在 help 里写清楚（它覆盖的是「根」，不是最终目录）。

---

## T-6（low）`runs/` 的归档完整性参差，`play.py` 的兜底链会把缺件静默掩盖

* 21 个 run 里有 **2 个没有 `model_final.zip`**：`runs/ab_both`（只有 `checkpoints/`，12 个文件）、`runs/cmdtrack3`（3 个文件）。对它们跑 `play.py --run ab_both` 会走 `_resolve_model()` 的兜底链（`model_final.zip` → `artifacts/best_stage.zip` → `model_best.zip` → `sorted(root.glob("*.zip"))[0]` → `checkpoints/*_final.zip` 的最后一个），**不会报错**，但用户以为自己评估的是 `ab_both` 的最终模型，实际拿到的是某个中间 checkpoint。
* 归档形状不统一：`runs/course22_improved` 有 `model_best.zip` + `summary.json` 但没有 `_env.json`；`runs/course22_ref` 有 `best_medium.zip`/`medium_final.zip`/`eval_report.json` 但没有 `_env.json`（与 `agent.md §8.5`「归档 checkpoint 没有 `_env.json`」一致）。
* `summary.json` 的 `obs_config` 缺 `vel_filter` 键（旧代码产物）——今天 `ObsConfig` 有 8 个字段，任何 `ObsConfig(**obs_config)` 式的重建在这些旧 run 上会**静默**使用 `vel_filter=False` 的默认值（恰好等价，但属于运气）。

### 建议

`_resolve_model()` 的兜底链在**没找到精确名字**时打印一条显式提示（「未找到 `model_final.zip`，改用 `checkpoints/stair9_final.zip`」），而不是静默降级。

---

## T-7（low）`requirements.txt` 与 `.gitignore` 的现状（核对无问题 + 一条提示）

* `requirements.txt`（**28 行**）：`mujoco==3.12.0`、`gymnasium==1.3.0`、`stable-baselines3==2.9.0`、`torch==2.14.1+cpu`、`numpy==2.2.4`、`tensorboard==2.21.0`、`pyyaml==6.0.3`、可选 `pillow==10.4.0`（`play.py --mode gif` 才需要）。注释声明 Python 3.13.0 / Windows / CPU。
  * **实测全仓（排除 `.venv`/`.tmp`）无 `matplotlib` / `onnx` / `onnxruntime` 残留 import** —— 与 `docs/README.md:101-107`「`robotdog/evaluate.py` 不再存在、触觉/测距逻辑已删除、GIF 与拼图改由 `scripts/play.py` 用 Pillow 完成」一致。
  * 提示：`pillow` 是**可选**依赖，而 `play.py --mode gif` 在 `write_gif()` 里是**函数内 `from PIL import Image`**（`:230`）⇒ 缺 pillow 时报的是 `ModuleNotFoundError` 而不是「请安装 pillow」；`scripts/play.py` 的 `--mode gif` 是默认模式（且在 S-1 修好之前还跑不到那一步）。
* `.gitignore`（**20 行**）：`.venv/`、`runs/`、`logs/`、`configs/`、`.generated/`、`.tmp/`、`output/{stills,frames,reports,exports}/`、`__pycache__/`、`*.py[cod]`、`*$py.class`、`.vscode/`、`.idea/`、`Thumbs.db`、`desktop.ini`、`.dsh-diag`、`.dsh`；注释明确「渲染中间产物忽略，**交付物 `output/gifs/` 要留**」。
  * 实测 `git check-ignore -v .venv/Scripts/python.exe` → `.gitignore:2:.venv/`（生效）；`.venv` 未被跟踪；`core.autocrlf=true`（仓库约定 CRLF）。
  * **本轮审计新建的 `analysis/` 不在 `.gitignore` 里**（按「不改动任何现有文件」的约束，我没有改它）⇒ 它会以未跟踪状态出现。若要把审计报告排除在版本库外，需要在 `.gitignore` 补一行 `analysis/`（这是唯一一处我建议但**没有**执行的改动）。

---

## T-8 已验证**没有**问题的地方

* **分层契约（工具层不 import 环境层）成立**：`robotdog/tools/__init__.py:16` 只有 `from robotdog.tools import console, paths, terrain`；`console.py` / `paths.py` / `terrain.py` 全都不 import `robotdog.env` 或 `robotdog` 顶层。`tools/__init__.md` 里出现的 `from robotdog.env import ...` **只存在于文档**。全仓对 `robotdog.tools.touch` 的引用**只出现在 docstring**（`env.py:232`、`terrain.py:59/173`），没有真实 import。
* **`console.setup_utf8_console()` 的实现是防御性的**（`robotdog/tools/console.py:18-29`）：try/except 包住 `reconfigure`，流已分离/已关闭时安静跳过（docstring 也写了「无能为力」）。这条对 Windows 下的中文输出是必要的。
* **`go2_model.py` 的场景编译路径正确**：`compile_scene()` 用 `mujoco.MjModel.from_xml_string(xml, assets=go2_assets())`，`go2_assets()` 把 `models/unitree_go2/assets/*.obj` 读成 `{name: bytes}` —— **必须这样编译**，否则 `ValueError: Error opening file 'assets/thigh_0.obj'`；失败时会 dump `.generated/failed_scene.xml`（仓内现存这一份 = 历史一次编译失败/内存打爆的残留，与 `agent.md §5⑤` 的记载一致）。
* **地形 geom 的可见性**：`TERRAIN_GEOM_GROUP` 是 `1`（在 MuJoCo 默认可见集 `[1,1,1,0,0,0]` 内），`env.render()` 还把 `geomgroup[0..5]` 全打开（`env.py:1022-1025`）⇒ 「GIF 里楼梯消失」的旧事故已被两层防护覆盖。`go2_model.py:65` 的 `COLLISION_GEOM_GROUP = 3` 用于碰撞体，默认不可见是**刻意的**。
* **`paths.py` 的根目录推导正确**：`WORKSPACE: Path = Path(__file__).resolve().parents[2]`（`robotdog/tools/paths.py:10` ⇒ 仓根），`GO2_ASSETS_DIR = GO2_DIR / "assets"`、`GENERATED_DIR = WORKSPACE / ".generated"`、`RUNS_DIR`、`OUTPUT_GIFS_DIR` 都据此派生，没有硬编码绝对路径。

---

## T-9（low–medium）`gait.py` 与 `robotdog/tools/terrain.py` 的补测

`go2_model.py` 的编译/元信息路径已在 T-8 里核对通过，本节只补两份**从未被单独审计过**的文件（`robotdog/gait.py` 191 行、`robotdog/tools/terrain.py` 242 行），以及一条仓库卫生项。

### T-9a（low）步态先验的复现链路**已断**，且文档比代码更清楚这件事；剩下的风险是「schema 判定靠一个字符串」

> 先纠正一个容易写错的判断：这条**不是**「文档在撒谎」。`docs/robotdog/gait.md:294-311` 已经用一张表把「出货先验经 `load_gait()` 之后的真实值（`top[0]`）」逐字段列了出来（`freq 5.477741882247174`、`period 0.18255697721736439 s → 183 ms`、`summary()` 原样字符串），`:375-388` 还明确写了「搜索脚本 `scripts/search_trot.py` 该文件从未入库……所以 `artifacts/trot_stairs.json` 是**不可重建的 artifact**」。`train.md:43` 的启动横幅也把它打印出来。也就是说 `gait.md` 是**准确的**，不准确的只有 `robotdog/gait.py:154` 那句 docstring（它仍把 `search_trot.py` 当作一个存在的脚本介绍）。

* 事实：`scripts/` 下真的没有 `search_trot.py`（只有 `__init__.py`、`cmd_demo.py`、`download_go2.py`、`play.py`、`teleop.py`、`train.py`）；`agent.md:59` 与 `README.md:88` 把 `artifacts/trot_stairs.json` 记作「唯一不可重建的 artifact」。**证据链是自洽的，只是不可复现**。
* 真正的代码风险在 `load_gait()` 的分支判定（`gait.py:181-191`）：`if "top" in data:` → `from_search_json()`，否则整份文件交给 `from_dict()`。而 `from_dict()` 只保留 `__dataclass_fields__` 里的键（`:144-145`），顶层的 `objective/target_speed/step_height/tread/clearance_weight/seconds/trials/refine/best/top` 全部丢弃。
* 实测两种分支的差距（`.tmp/zz_probe_gait_t9b.py`）：
  ```text
  from_search_json       : 5.48 Hz, ±0.265 thigh / ±0.986 calf (norm. units), shape=sin, scale=1.00 rad, period=183 ms
  from_dict(整个文件)     : 2.00 Hz, ±0.000 thigh / ±0.000 calf (norm. units), shape=sin, scale=0.70 rad, period=500 ms
  from_dict(best.gait)   : 5.48 Hz, ±0.265 thigh / ±0.986 calf (norm. units), shape=sin, scale=1.00 rad, period=183 ms
  action_scale: search=1.0  whole-file=0.7  best=1.0
  ```
  即：**只要产物里没有 `"top"` 键（比如将来重搜时把 schema 改成 `results`/`candidates`），先验会静默退化成「原地不摆腿、膝部空摆」**（`amp_th=0.0`、`freq` 取类默认 2.0 Hz、`action_scale` 取类默认 0.7 rad），训练照跑、日志照出、策略学不出步态 —— 与 `config.yaml:22` 那句「`gait_prior` 别关，六级全部给 0」描述的症状完全一样，但**根因换成了 schema**。
* `artifacts/trot_stairs.json` **同时含 `"top"`（10 条）与 `"best"`**，且 `top[0]` 与 `best` 逐字段相同（实测 `rank`/`score`/`gait`/`action_scale` 全部一致），所以今天恰好走对分支。这份冗余（`best` 只是 `top[0]` 的副本）意味着两个键的取舍没有任何约束。
* 建议：`load_gait()` 在没有 `"top"` 时**不要**静默走 `from_dict(整份文件)`；应当显式区分「单条 `GaitParams` JSON」与「搜索结果文件」，后者缺 `top` 就报错（现在的行为等价于把「schema 不匹配」伪装成「先验被关掉」）。

### T-9b（low）`GaitParams.normalize()` 会**静默裁剪**越界步态，与 `config.py` 的「未知/非法就硬报错」风格相反

* `normalize()`（`gait.py:110-128`）做三件静默的事：`freq` 裁到 `[0, 10]`；`amp_th/amp_ca/off_th/off_ca` 裁到 `[-1, 1]`；`|off| + amp > 1` 时**按比例缩小到恰好 1**。没有任何警告或异常（只有 `calf_shape` 不在 `CALF_SHAPES` 时抛 `ValueError`）。
* 实测（`.tmp/zz_probe_gait_t9.py`）：
  ```text
  freq 99.0 -> 10.0   amp_th 5.0 -> 0.5   off_th -3.0 -> -0.5
  calf_shape=5 -> IndexError: tuple index out of range      # 注意：不是 ValueError
  ```
  注意后一条：`from_dict()` 在 `:148-149` 用 `CALF_SHAPES[int(...)]` 做历史格式映射，越界的整数得到的是 `IndexError`（既不是 `normalize()` 里那个 `ValueError`，也没有可读消息）—— 与 `config.py` 的报错风格差异很大。
* 边界值实测：交付物 `best.gait` 满足 `|off_th| + amp_th = 0.55838736888824592`（不动）；`|off_ca| + amp_ca = 1.000000000000000019`（比 1 大一个 ULP ⇒ 每次加载都会触发一次比例 `0.99999999999999998` 的缩小，方向保持、数值上不可见）。实测 `load_gait()` 之后 `|off_ca| + amp_ca` 恰好打印成 `1`。所以「契约被违反」这件事在出货文件里**只差一个 ULP**，靠运气落在安全侧。
* 结论：`actions()` 里 `np.clip(a, -1.0, 1.0)`（`:88`）因此是个**恒等操作**（`normalize()` 已经保证 `|off|+amp <= 1`）—— 对今天的数据没有影响，但它掩盖了「谁负责保证边界」这个问题。

### T-9c（medium）`from_dict()` 静默丢弃未知键 ⇒ 步态文件里的拼写错误永远不会被发现

* `gait.py:144-145`：`known = {f for f in cls.__dataclass_fields__}`；`kw = {k: v for k, v in d.items() if k in known}` —— 其余键**无声丢弃**。
* 实测（`.tmp/zz_probe_gait_t9.py`，把 `amp_th` 拼成 `amp_thigh`）：
  ```text
  amp_th after misspelling = 0.0 (应为 0.0 = 静默退回类默认)
  summary: 5.48 Hz, ±0.000 thigh / ±0.986 calf (norm. units), shape=sin, scale=0.70 rad, period=183 ms
  ```
  加载成功、`action_scale` 也悄悄从 1.0 掉回类默认 0.7，**没有任何提示**。若这次拼错发生在 `freq` 上，症状就是「步态频率变了但没人知道」。
* 这与 C-1 是同一类病症（「配置键的拼写正确性是唯一校验面」），只是载体从 YAML 换成了 JSON，而 `gait.py` 没有 `_apply()` 那样的白名单检查。`config.yaml` 的 `reward:` 段把 `forward_vel_sigma` 拼错会**硬报错**，同样性质的错误在 `env.gait_file` 指向的 JSON 里却**静默通过**。

### T-9d（low）`robotdog/tools/terrain.py`：解析高度剖面与几何在**收尾边界**上不一致

* `StairSpec.height_at(x)`（`terrain.py:189-205`）在只上行的楼梯上：`x < up_end_x + top_landing` ⇒ 返回 `crest_height`；否则返回 `0.0`（`:196-199`）。
* 而 `step_blocks()`（`:156-166`）在 `elif self.top_landing > 0.0 and self.n_up > 0:` 分支里才生成 `_top` 平台块 —— 两个分支的**使能条件不同**：
  * `n_up > 0` 且 `top_landing > 0`：几何与剖面一致；
  * `n_up == 0`（`flat_spec()` 的平地）且 `top_landing > 0`：`step_blocks()` 不生成任何块（`:156` 的 `n_up > 0` 挡住），但 `height_at()` 在 `x < up_end_x + top_landing` 区间里仍返回 `crest_height` = `0 * step_height` = **0.0**（恰好也是 0，暂时无害）；
  * 更一般地：`n_up == 0` 时 `up_end_x == start_x`，所以 `height_at()` 的 `x < self.start_x ⇒ 0.0` 分支先接管，上述区间为空 ⇒ 目前**没有实际错配**。
* 真正值得记录的是一处**契约不对称**：`to_xml()` 的 docstring（`:169-176`）与 `step_blocks()` 的注释都以「几何 = 剖面」为前提，而 `height_at()` 里没有 `n_up > 0` 这个守卫。今天两条路径给出相同的数，但理由是「`crest_height == 0`」和「区间为空」，不是「条件一致」（两个分支的差异与实测见 T-9g-1）。任何人把 `n_up=0` 与 `top_landing>0` 组合起来用于「带平台的地面」，就会得到**剖面说有平台、几何里没有**的地形（观测里的 `terrain_height_under_base` 会用这个解析值）。
* `demo_stairs_spec()` 的 `kw.setdefault("top_landing", 1.20 if n_down == 0 else 0.0)`（`:228`）在 `n_down > 0` 时把 `top_landing` 设成 0 —— 但 `n_down > 0` 分支根本不读 `top_landing`，这个 setdefault 是**无效果的**（无害，属于可读性）。

### T-9e（low）仓库根有一份**被 git 跟踪**的 MuJoCo 运行日志，内容是数值发散告警

* `MUJOCO_LOG.TXT`（121 B，2 行，**tracked**，`git ls-files --error-unmatch MUJOCO_LOG.TXT` 命中，且**不在** `.gitignore` 里）：
  ```text
  WARNING Sat Oct 10 12:32:58 2026: Nan, Inf or huge value in QACC at DOF 0. The simulation is unstable. Time = 4.1360.
  ```
* 这条是 MuJoCo 在**物理发散**时写入的告警，说明至少有一次运行出现过 `QACC` 爆成 NaN/Inf（4.136 s 处）。它是运行时产物，与 `.gitignore` 的原则（「训练/评估产物：全部可以重建，不进版本库」）冲突，而且 `Nan/Inf in QACC` 这类事件在 PPO 训练里通常对应「一次坏 rollout」，值得在训练日志里被显式捕获而不是躺在仓根。
* 顺带：`.gitignore:16-19` 忽略了 `.dsh-diag` 与 `.dsh`，**没有**忽略 MuJoCo 的日志文件名，`MUJOCO_LOG.TXT` 会一直被跟踪下去。

---

## T-9f（medium）`flat_spec()` **不会**把 `step_height` 归零 —— 平地演示脚本因此给策略喂了一个「5 cm 台阶」

* `robotdog/tools/terrain.py:213-216`：
  ```python
  def flat_spec(**kw) -> StairSpec:
      kw.setdefault("n_up", 0)
      kw.setdefault("n_down", 0)
      return StairSpec(**kw)
  ```
  它**只**补 `n_up`/`n_down`，不动 `step_height`；而 `StairSpec.step_height` 的类默认是 `0.05`（`terrain.py:78`）。实测 `flat_spec(...).step_height == 0.05`。
* 训练链路是**对的**，因为 `config.py:293` 显式写了 `flat_spec(n_up=0, n_down=0, step_height=0.0, tread=tread)`（`policy.py:324` 同理）。**错的**是这两个仓内调用方：
  * `scripts/teleop.py:77` 的 `flat_spec_for(...)`；
  * `scripts/cmd_demo.py:152` 的 `flat_spec(tread=0.50, width=2.0, start_x=1.50)`。
* 后果链条：`env.py:559-562` 无条件 `if obs_cfg.step_height: parts.append(np.array([float(self.stair.step_height)]))`（`ObsConfig.step_height` 默认 True，`config.yaml:81` 也是 true）⇒ 这两个脚本**每次平地演示**都把 `0.05` 塞进观测第 45 维，而它们自己的 HUD 同时打印「地形 平地（无台阶）」（`play.py` 的 `hud_for`）。策略是**盲走**的，它唯一被允许知道的地形信息就是这个标量 —— 于是「平地」与「5 cm 台阶」在观测里是同一个东西。
* 奖励侧**没有**连带影响（实测）：`reward.py:278-280` 取 `max(foot_lift_target, foot_lift_step_ratio * step_height)` = `max(0.08, 1.4*0.05=0.07)` = `0.08`，两种取值同解。所以这是一个**只污染观测、不污染奖励**的静默缺陷。
* 顺带修掉一个陈旧注释：`reward.py:276-277` 写「在 5 cm 处它与原来的常量相等」，但 `1.4 × 0.05 = 0.07 ≠ 0.08`（`reward.py:74` 与 `config.yaml:115` 都是 0.08）—— 这条注释正是「为什么 phantom step 在奖励上无害」的解释，而它算错了。
* 建议：`flat_spec()` 里补 `kw.setdefault("step_height", 0.0)`（一处改动同时修掉两个脚本），或把 `StairSpec.step_height` 的类默认改为 `0.0`、让 `demo_stairs_spec()` 保留 `0.05`（它本来就是「造楼梯」的函数）。

## T-9g（medium）`terrain.py` 的几何参数没有任何校验 ⇒ 非法尺寸只在 MuJoCo 编译期以 `RuntimeError` 暴露

* **T-9g-1**：`terrain.py:136` 的 `if self.n_down > 0:` 分支里的 crest 块**没有** `crest_height > 0` / `n_up > 0` 守卫，而它的兄弟分支（`:156` 的 `elif self.top_landing > 0.0 and self.n_up > 0:`）有，注释 `:158-159` 还专门解释了「平地段 crest 高度为 0，顶层平台会退化成零高度方块」这个危险。于是 `StairSpec(n_up=0, n_down=3)` 会产出 `size="0.250000 1.000000 0.000000"` 的非法零高度 box（实测）⇒ 报的是 `RuntimeError: MJCF compile failed; dumped to .generated/failed_scene.xml`（`go2_model.py:405-408`），而不是一句「n_up=0 却给了 n_down>0」。
* **T-9g-2**：`demo_stairs_spec(step_height=0.0)` 同样会产出零高度 box（实测），`tread` 为负会产出负半尺寸，**全程静默**，直到编译失败。`terrain.py:70-94` 的 `StairSpec` 没有 `__post_init__`（对比 `ObsConfig.__post_init__` 会硬拦 `base_height`），这是同一份代码里两种风格。
* **T-9g-3**：`height_at()` 的 `:202` 是裸的 `j = int((x - self.up_end_x - self.landing) // self.tread)`，`tread=0` 且 `n_down>0` 时抛 `ZeroDivisionError: float floor division by zero`（实测）；`:194` 的同款表达式在 `tread=0` 时不可达（那时 `up_end_x == start_x`，前面 `x < self.start_x` 先接管）。仓库内没有任何 `n_down>0` 的调用点 ⇒ `:136-155`、`:200-205` 与 `:228` 的三元表达式**从未被使用过**（金字塔地形是「写了但没跑过」的代码路径）。
* **T-9g-4**：`down_end_x` 的 docstring（`:109`）「下降段回到地面高度处的 x」在 `n_up != n_down` 时为假（例如 8 上 6 下，最后一块离地面还差一道 10 cm 立面）。同样属于「未被使用所以没人发现」。

## T-9h（low）`terrain.py` / `go2_model.py` 里有一条**不存在的模块**的注释链

* `terrain.py:31-32`、`:59`、`:173` 与 `go2_model.py:232` 都提到 `robotdog.tools.touch` / `touch.riser_graze` / 「`env.py` 的楼梯测距 `mj_ray` 前向探针」，并引用了一个探针脚本 `.tmp/stall_probe.py`。
* 实测：`robotdog/tools/` 下只有 `__init__.py`、`console.py`、`paths.py`、`terrain.py`，**没有 `touch.py`**；全仓 grep `mj_ray|mj_multiRay` 在**注释之外零命中**；`riser_graze` 在 `reward.py` 里也不存在；`.tmp/stall_probe.py` 同样不存在。
* 关键点：这些注释所支撑的**结论仍然成立**（`TERRAIN_GEOM_GROUP = 1` 在 `mjv_defaultOption` 的默认可见集 `[1,1,1,0,0,0]` 内 —— 这一点已核实，且 `env.py:1022-1025` 还会强制打开全部六个 group），只是「给一个不存在的前向探针专用 group」这个**理由**是虚构的。这是文档层的问题，但它恰好位于那条最容易踩坑的常量旁边。

## T-9i（low）`go2_model.py`：地形 geom 的识别靠名字前缀，换一种地形就会被漏掉

* `extract_meta()`（`go2_model.py:331-335`）用
  ```python
  stair_ids = [g for g in range(model.ngeom)
               if (mj_id2name(model, mjOBJ_GEOM, g) or "").startswith("stair_")]
  ```
  来收集地形 geom，而 `env.py:606-608` 与 `:728-733` 用它（连同 `floor_geom_id`）构造「地形」集合，把地形接触从 `trunk_contact` / `leg_contact` 里排除。
* 实测：`build_scene_xml(custom, floor_geom=default_floor_geom())` 里放一个 `name="obstacle_0"` 的 box ⇒ `meta.stair_geom_ids` 为空 —— 也就是**任何不走 `StairSpec`（`name_prefix="stair"`）的地形都会漏进 `trunk_contact`**。`build_scene_xml()` 的 docstring 明确说自己支持「楼梯、heightfield、障碍物以及任何逐回合变化的几何」，所以这个前缀耦合与文档承诺不一致。
* 同一函数里 `floor_id = mj_name2id(model, mjOBJ_GEOM, "floor")` **不做 -1 检查**（对比 `_resolve_id()` 会抛 `KeyError`）：实测 `build_scene_xml("", floor_geom=None)` 编译后 `meta.floor_geom_id == -1`，于是 `terrain` 集合变成 `{-1} ∪ stair_*` —— 一个不存在的 geom id 被当成地形，而 `floor_geom_id` 的语义（「要排除的地面」）失效。当前正式路径总是传 `default_floor_geom()`，所以没被触发。
* 几何分组的实测数字（`.tmp/zz_probe_go2_t9.py`，8 级 9 cm 楼梯 + 默认地面）：`group 1: n=10`（`floor` + 9 个 `stair_*`）、`group 2: n=33`（纯视觉）、`group 3: n=23`；而 `trunk_geom_ids=[15,16,17]`（3 个）+ `leg_geom_ids`（16 个）+ `foot_geom_ids=[29,41,53,65]`（4 个）= **23**，与 group 3 总数**完全相等** ⇒ 「group 3 恰好等价于可接触地形的机器人几何」这一论断成立，`trunk/leg` 的切分也没有漏项。
* 已确认无问题：`total_mass = float(model.body_mass[1:].sum())` 确实排除了 world body（实测 15.206408 kg，`body_mass[0] == 0`）；`COLLISION_GEOM_GROUP = 3` 与 `models/unitree_go2/go2.xml` 的 `class visual=2 / class collision=3` 一致（躯干盒 + 膝圆柱 + 四个脚球），`trunk/leg/foot` 三个集合恰好把 group 3 的机器人部分切完；`viewer` 用的 `lru_cache(maxsize=1)` 资源字典按**裸 basename** 建键（含注释警告重复文件名会被 MuJoCo 拒绝）；非 ASCII 路径问题（`meshdir` 走内存 VFS）在模块 docstring 里解释得很完整。

## T-9j（low）`console.py` 与 `paths.py` 的边角

* `console.py:27-29` 的 `except (ValueError, OSError): pass` **静默吞掉** `reconfigure()` 失败 ⇒ 这个模块存在的唯一理由（把 GBK 控制台的输出变成 UTF-8，防止重定向日志乱码）在最需要它的时候可以无声失效。
* `console.py:7-8` 的 docstring 只点名 `scripts/train.py` 与 `scripts/play.py`，实际调用方还有 `scripts/cmd_demo.py:136` 与 `scripts/teleop.py:203`（「第一行就调用」这个说法本身对 `train.py:294` / `play.py:602` 成立）。
* `paths.py:12/:13/:21` 的 `MODELS_DIR` / `GO2_DIR` / `OUTPUT_DIR` 在文件外**没有消费者**（只喂给同文件里的其它常量）。它们有对应文档（`docs/robotdog/tools/paths.md`），属于「刻意的公开 API 面」，记为信息而非缺陷。

### T-9 已验证无问题（补测部分）

* 工具层 import 契约成立：`terrain.py`/`paths.py`/`console.py` 只 import stdlib 与同层模块，`tools/__init__.py` 只聚合 `console, paths, terrain`，全仓找不到 `robotdog/tools/` 内 import `robotdog.env` 或顶层 `robotdog` 的语句（契约 11 成立）。
* `terrain.py:150-151` 的 `if top <= 1e-9: continue` 是**必要的**（否则 `n_up == n_down` 的金字塔会产生零高度 box），且与 `height_at` 的 `max(0.0, ...)`（`:205`）一致；`flat_spec()` 产生 0 个 geom（`build_terrain_xml` 返回 `""`）且 `height_at ≡ 0.0`，**平地阶段没有零高度 box**。
* 4001 点扫描（平地 / 8 上 / 8 上 6 下 / 8 上 8 下 / `top_landing=0`）显示解析剖面与几何**在所有块内部完全一致**，仅在半开区间的共享端面上有 measure-zero 差异（例如默认楼梯 `height_at(6.7)=0.0` 而顶块仍覆盖到 `x<=6.7`）。
* crest→down 的接缝**不会**复现 `CREST_SEAM_OVERLAP` docstring 里那个「共面楔住」的陷阱（两块顶面相差一个 `step_height`，不存在零高度缝）—— 也就是说那条宝贵的教训只适用于「crest 与最后一级踏面同高」这一种情形，不能被误推广。
* 先验幅值与 `EnvConfig.action_scale` 解耦（契约 6 成立）：`gait.py:103 return self.action_scale * self.actions(t)` vs `env.py:898 target = home + gait_gain * gait.radians(t) + action_scale * action`；出货 `GaitParams.action_scale = 1.0`、`EnvConfig.action_scale = 0.30`，实测先验 `max|radians| = 0.99996`。
* `GaitParams.radians` 的直接构造会绕过 `normalize()`（`gait.py:88` 的 `np.clip` 只能在 `radians()` 内部兜底，实测 `GaitParams(amp_ca=1.5)` 的 `radians` 被压到 0.7 rad 而不是 1.05 rad）—— `docs/robotdog/gait.md:432` 的示例正是这种直接构造，所以这条「兜底」在实践中会被踩到，但它本身**不改变已加载路径**的行为。
* `TROT_PHASE = (0.0, π, π, 0.0)`（`gait.py:34`）与 `normalize()` 里强制的 `(FL, FR, RL, RR)`（`:126-127`）一致；`normalize()` 的比例缩放（`:118`）不会除零（只在该分支 `abs(o)+a > 1` 时进入）。

---

## 复现命令

```powershell
$env:PYTHONIOENCODING="utf-8"
.\.venv\Scripts\python.exe .tmp\check_lines.py      # 退出码 1：4 处空行引用
.\.venv\Scripts\python.exe .tmp\check_doc_code.py
.\.venv\Scripts\python.exe .tmp\check_xrefs.py
.\.venv\Scripts\python.exe .tmp\audit_docs.py
.\.venv\Scripts\python.exe .tmp\ref_truth.py        # 17 条 suspicious，注意退出码仍是 0
```

> 注：`ref_truth.py` 直接跑会因 Windows GBK 控制台 + `⭐` 字符抛 `UnicodeEncodeError: 'gbk' codec can't encode character '\u2b50'`，必须先设 `$env:PYTHONIOENCODING="utf-8"`（也算一个小的可用性缺陷）。
