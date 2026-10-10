# 回归门脚本（参考实现原件）

这些是 upstair_core 里实际在用的六支校验脚本 + 两个幂等辅助脚本，**逐字复制**，未做任何改写。
配套说明见 `references/08-verification.md`。

## 放置位置

脚本顶部用 `ROOT = Path(__file__).resolve().parents[1]` 定位仓根，所以它们必须放在**仓根下第一级
目录**（参考实现放在 `.tmp/`，该目录 gitignored）。想放进 `tools/dev/` 这类更深的目录，就把
`parents[1]` 改成 `parents[N]`，或直接写死绝对路径。

`.tmp/` 在本仓被 `.gitignore` 忽略，所以脚本不会被提交；如果希望回归门进版本库，
把它们放到 `tools/dev/` 并同步改 `ROOT`。

## 各脚本一句话

| 脚本 | 作用 | 依赖 |
| --- | --- | --- |
| `check_lines.py` | 文档里 `file.py:行号` 是否越界或指向空行（**能解析裸文件名**） | 无 |
| `ref_truth.py` | 行号指向的**真实内容**是什么、是否落在另一个 `def`/`class` 上 | 无（定义行靠 AST 现算） |
| `check_doc_code.py` | markdown 里的 ```` ```python ```` 块能否 `ast.parse` | 无 |
| `check_xrefs.py` | 根文档（`DOCS` 列表）中的交叉引用是否存在 | 改 `DOCS` 列表 |
| `audit_docs.py` | 文档/源码一一对应 + 10 节结构 + 章节引用 + 索引完整性 | 改 `REQUIRED` 关键字表与文档根 |
| `fix_eol.py` | 整篇重写产生的裸 LF 统一回 CRLF | 无（幂等） |
| `sync_env_md_lines.py` | 按符号名刷新模块文档里的行号 | 需按新仓改符号—文档映射 |

## 运行

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

.venv\Scripts\python.exe -m compileall -q robotdog scripts
.venv\Scripts\python.exe .tmp\check_lines.py
.venv\Scripts\python.exe .tmp\ref_truth.py
.venv\Scripts\python.exe .tmp\check_doc_code.py
.venv\Scripts\python.exe .tmp\check_xrefs.py
.venv\Scripts\python.exe .tmp\audit_docs.py
```

全绿基线（**数字随仓库演化，以实跑为准**）：

```
stale/blank line refs: 0   未解析的引用（可能指向已删文件）: 0
refs checked: N   suspicious: 0
python blocks: N, unparsable-skipped: N, BROKEN: 0
BROKEN CROSS-REFS: none
RESULT: ALL CHECKS PASSED
```

两处允许长期存在的例外：① 参考实现刻意把「行号漂移但没越界」的历史举例留在
`README.md`（引用 `robotdog/env.py:827`，该行是空行），所以 `check_lines` 会一直报它一条；
② `ref_truth` 的「未解析」里允许出现仓外引用（如 `stable_baselines3/...`）。

## 已知弱点（原样搬走会带着同样的盲区）

下面这些问题在新仓里同样存在：

| 脚本 | 弱点 | 一行修法 |
| --- | --- | --- |
| `check_xrefs.py` | 末尾只有 `print(...)`，**没有 `sys.exit`** ⇒ 永远退出码 0，不能当 CI 门 | 末尾加 `sys.exit(1 if broken else 0)` |
| `check_lines.py` | `docs` 列表写死 3 份根文档 + `docs/**/*.md`，**不扫 `agent.md`** | 把 `agent.md` 加进 `docs` 列表 |
| `check_doc_code.py` | `unparsable-skipped` 与校验块数同量级（示例/伪代码被跳过）⇒ 覆盖有限 | 无自动修法，靠人工复核被跳过的块 |
| 全部 | 抓不到**数值口径过期**（文档写旧权重、源码已改，六支门仍全绿） | 数值只写在 `config.yaml`；文档引用键名而非数值 |
| 全部 | 从 `assets/gates/` **原地运行会静默失效**：`ROOT = parents[1]` 会逃出被检仓，`ref_truth.py` 打印 `refs checked: 0` 并 exit 0 | 复制到仓根下第一级目录再跑（参考实现放 `.tmp/`） |

**已经修掉的两条历史弱点**（旧版曾有、参考实现现已改写）：① `ref_truth.py` 的手工
`EXPECT` 快照——符号换行号后它会**假报 BAD**，现在定义行由 AST 现算；② 两支门只按
`ROOT/raw` 解析路径，于是文档里几十条**裸文件名**引用（`` env.py:585 ``）从未被检查过，
现在按「仓根 → `robotdog/` → `scripts/` → `docs/` → 全仓唯一同名」解析。

详见 `references/08-verification.md` §7。
