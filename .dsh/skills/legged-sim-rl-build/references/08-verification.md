# 回归门：改完代码必须过的检查

> 对应 `SKILL.md` 阶段 10、§4 清单。素材取自 upstair_core 的六支校验脚本与等价性证明流程，
> 脚本本体已复制到 `assets/gates/`（换仓时只改脚本顶部的 `ROOT` / `DOCS` 常量）。

设计原则：**六支门全部在 1 秒内跑完、都不需要训练权重**。回归门要能在「改一行代码」的心智成本内
跑完，否则它一定被跳过。

---

## 1. 六支门与全绿基线

| # | 脚本 | 查什么 | 全绿输出 |
| --- | --- | --- | --- |
| 0 | `python -m compileall -q robotdog scripts` | 语法 | 无输出（退出码 0） |
| 1 | `assets/gates/check_lines.py` | 文档里 `file.py:行号` 是否**越界**或指向**空行**（裸文件名也能解析） | `stale/blank line refs: 0` |
| 2 | `assets/gates/ref_truth.py` | 行号指向的**真实内容**是什么、是否落在另一个 `def`/`class` 上（抓「漂移但未越界」） | `refs checked: N   suspicious: 0` |
| 3 | `assets/gates/check_doc_code.py` | markdown 里的 ```` ```python ```` 块能否 `ast.parse` | `python blocks: N, unparsable-skipped: N, BROKEN: 0` |
| 4 | `assets/gates/check_xrefs.py` | 根文档之间的交叉引用是否指向存在的标题 | `BROKEN CROSS-REFS: none` |
| 5 | `assets/gates/audit_docs.py` | ① 每个 `.py` 是否有对应 `.md`；② 文档 10 节结构与标题关键字；③ 文档里引用的 `experience.md §N` / `train.md §N` 是否存在；④ 索引完整性 | `RESULT: ALL CHECKS PASSED` |

辅助脚本（幂等，可反复跑）：

* `assets/gates/fix_eol.py`——把整篇重写产生的裸 LF 统一回仓约定的 CRLF。**每次用「整篇 write」改完
  文件都跑一次**，否则下一次 `git diff` 会把整个文件算成改动。
* `assets/gates/sync_env_md_lines.py`——改了环境类的成员后，按**符号名**刷新对应模块文档里的行号，
  不用手工数行。

**为什么要有 2 号门**：1 号门只能发现「越界/空行」。文档里 14 处 `env.py:531` 在源码里给某个函数
的 docstring 加长 5 行之后仍然「有效」——只是全部指到了别的函数。2 号门把「行号 → 这个位置是不是
另一个函数的定义」判出来，才能抓到这种静默漂移；判据由 **AST 现算**，不再依赖手工期望值。

---

## 2. 门禁抓不到的两类腐坏（必须靠纪律）

1. **文档描述了一个源码里不存在的 API**。行号检查只看「这一行有没有内容」，不看「这个符号还有没有
   存在」。对策：删代码时同步删文档（见第 5 节），并定期用「符号 → 源码 grep」的方式抽查。
2. **行号整体漂移但未越界**（上面 2 号门就是为它准备的）。

顺带两条同类教训：

* **别把行号当稳定标识**：引用代码优先写 `路径/文件.py` + **符号名**；只有在特指某一行（例如
  「`self._kp = np.tile(cfg.kp, 4)` 这一行」）时才写行号。
* **「能删就删」的判据要精确**：零读者 **且** 不是文档化契约才算死代码。三类「假死」必须保留——
  ① 字符串派发（`venv.env_method("set_commands")`，`rg` 看不到调用）；② 框架回调（SB3 的
  `_on_step`）；③ 文档化公开 API（文档里写进公开签名的函数/属性）。

---

## 3. 改了奖励或观测数学：等价性证明

**目的**：证明「重构没有改变数值行为」。奖励公式重写时最容易静默改掉数值（分母漏平方、常数阈值、
量程变化），只看成功率是发现不了的。

**流程**（参考实现在 `experience.md` §5.9）：

1. 取**改前**版本到 `.tmp/`：
   `cmd /c "git show HEAD:robotdog/reward.py > .tmp/reward_before.py"`
   ——用 `cmd /c` 而不是 PowerShell 的 `>`，后者写出来是 UTF-16，再读进来会报
   `SyntaxError: source code string cannot contain null bytes`。
2. 用 `importlib.util.spec_from_file_location` 把改前版本加载成**私有模块**（不要 `import`，
   否则拿到的是已经导入的新版本）。
3. 固定随机动作序列（参考实现用 `np.random.default_rng(20261008).uniform(-1, 1, (400, 12))`），
   在**多个地形高度 × 多个 seed**（0 / 5 / 8 / 11 cm × 3 seed）上并排跑两份实现。
4. 判据（必须全为 0，不是「接近」）：

   * `max|Δreward| = 0.000e+00`
   * `max|Δobs|    = 0.000e+00`
   * 终止标志零差异（逐回合比对 `terminated` / `truncated`）

参考实现验证过两次：`foot_lift_target` 改值、`body_collision` 从 −4.0 改到 −1.2（后者还额外核了
每一步的动力学与接触数），都要求逐位一致。

**如果做不到逐位一致**：那就是**行为改动**，不是重构。此时必须重新跑训练并在文档里写清
「这次改了什么、为什么、预期影响」，不能悄悄合并。

---

## 4. 行为回归（冒烟）

```powershell
$env:PYTHONUNBUFFERED="1"
.venv\Scripts\python.exe scripts\train.py --run-name smoke_cfg --config .tmp\smoke_config.yaml
.venv\Scripts\python.exe scripts\play.py  --run smoke_cfg --mode eval
```

* 冒烟配置放 `.tmp/`（gitignored），**不要**把试验配置写进仓根；六级各 2000 步约 45 秒。
* 跑完**手工删掉** `runs/smoke_cfg` 与 `output/gifs` —— `output/` 是交付物目录，**不在 `.gitignore`
  里**，冒烟产物会直接进版本库。
* 长任务必须用**受管后台任务**并设 `PYTHONUNBUFFERED=1`：用 `Start-Process` 起的长任务不跨工具
  调用存活（进程树被回收，日志停在几千步），stderr 的块缓冲还会让它看起来像训练停滞。

**单次训练的成功率不是结论**：同配置只换 seed，250k 步单级在 36%~72% 之间摆动。比较两个改动时，
用**同一组 seed 各跑 ≥3 次**，并以 GIF 里的姿态是否学到位为主判据（数字只作辅助）。

---

## 5. 文档同步（改了代码就得改，同一批提交里完成）

* `docs/<模块>.md` 与 `<模块>.py` 严格一一对应（含子包）：新增/删除模块 ⇒ 同步映射与索引，
  否则 5 号门报 `MISSING` / `NOT-IN-INDEX`。
* 固定 10 节、顺序不变，标题必须含模板关键字；某一节确实没有内容时**写明「无」并说明原因**，
  不要删节——结构一致性本身就是被检查的对象。
* 源码里已经不存在的符号，文档里**删掉**，不要保留「历史说明」：历史结论写进经验文档
  （`experience.md` / `train.md`）并注明日期与判据。
* 改了环境类成员 ⇒ 跑 `sync_env_md_lines.py`；改了任何文档 ⇒ 跑 1/2/3/4 号门。

---

## 6. 把回归门搬到一个新仓

* 五支脚本都是「读文档里的 `路径:行号` → 对源码」的形式，换仓只改顶部的 `ROOT`（和
  `check_xrefs.py` 的 `DOCS` 列表、`audit_docs.py` 的 `REQUIRED` 关键字表）。
* 起步最小集合：`compileall` + 1 号（行号边界）+ 3 号（文档代码块可解析）。
  等 `docs/` 采用 10 节模板后再加 5 号；等根文档之间开始互相引用再加 4 号。
* 2 号门需要**逐条录入期望值**（「这一行应该是什么」），建议在文档稳定后一次性录入，
  之后它才是有价值的回归网。

---

## 7. 六支门自身的已知弱点（搬用前先修）

这些是实测出来的，不是推测。**原样搬走会带着同样的盲区**：

1. **4 号门永远返回 0**：`check_xrefs.py` 最后一行只有 `print("BROKEN CROSS-REFS:", ...)`，
   没有 `sys.exit(1 if broken else 0)`（对比 `check_lines.py` 末尾就有）。⇒ 当 CI 门用时它
   **永远不会失败**。搬用时补上退出码。
2. **1 号门不扫 `agent.md`**：`check_lines.py` 的 `docs` 列表写死为
   `["README.md", "experience.md", "train.md"] + docs/**/*.md`，而 `agent.md` 里有大量
   `文件.py:行号` 引用 ⇒ 那一份的行号漂移无人检查。把它加进列表。
3. **3 号门的覆盖率有限**：被跳过的代码块（示例片段、伪代码、带省略号的片段）与能校验的
   数量级相同，所以 `BROKEN: 0` 不能理解为「文档里的代码都正确」。
4. **所有门都抓不到「数值口径过期」**：参考仓的文档曾经长期写着旧的奖励形状与旧权重
   （高斯核、2.5/1.5/1.0），源码早已是线性死区 10.0/6.0/4.0——六支门**全绿**。
   门只查「引用是否有效」「结构是否齐全」，不查「文档里的数字是否还是当前代码的数字」。
   对策只能是纪律：把**数值**写进配置文件、文档只引用键名；或给关键常量加一条断言式检查。
5. **从 `assets/gates/` 原地运行会静默失效**：脚本顶部 `ROOT = Path(__file__).resolve().parents[1]`
   是相对**仓根**写的，原地跑时 `parents[1]` 会逃出被检仓——`ref_truth.py` 打印
   `refs checked: 0   suspicious: 0` 并 **exit 0**（看起来全绿！），其余几支直接
   `FileNotFoundError`。必须复制到仓根下第一级目录再跑。

**两条历史弱点已在参考实现里修掉**：① 2 号门的手工 `EXPECT` 快照会自己腐坏（符号换行后把
**正确**的引用判成 `BAD`），现在「这一行是不是某个 `def`/`class` 的定义」由 AST 现算；
② 1、2 号门只按 `ROOT/raw` 解析路径，文档里几十条**裸文件名**引用（`` env.py:585 ``）因此
**从未被检查过**，现在按「仓根 → `robotdog/` → `scripts/` → `docs/` → 全仓唯一同名」解析。
