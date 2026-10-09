# `robotdog/__init__.py` —— 包入口

## 1. 一句话定位

包入口层，19 行：把工具层的三个叶子模块与 `setup_utf8_console` 重导出到包命名空间，
并暴露 `__version__`。它不依赖任何业务模块，因此位于依赖图最上游。

## 2. 职责边界

**做**：

* `from robotdog.tools import console, paths, terrain` —— 重导出三个工具模块
  （`robotdog/__init__.py:14`）；
* `from robotdog.tools.console import setup_utf8_console` —— 重导出该函数（`:15`）；
* 定义 `__version__ = "0.2.0"`（`:17`）；
* 用 `__all__` 把 `*` 导入限定为 `["console", "paths", "setup_utf8_console", "terrain"]`（`:19`）。

**不做**：

* **不在模块体里调用 `setup_utf8_console()`**——`import robotdog` **不会**修控制台编码，
  必须由脚本显式调用（见 §3 D1）；模块 docstring（`:6-10`）把这一点写成了显式警告；
* **不重导出环境**：`Go2StairsEnv` / `EnvConfig` / `ObsConfig` / `RewardConfig`
  都必须写 `from robotdog.env import Go2StairsEnv`（`README.md` §2 结尾的原话）；
* 不重导出 `gait` / `go2_model` / `training` / `curriculum` / `policy` / `env`；
* 不建目录、不读配置、不配 logging、不解析命令行、不定义 `main`。

## 3. 关键设计决策与理由

### D1 —— 控制台修复由脚本显式调用，不做成 import 副作用

本文件**只导入** `setup_utf8_console`，不调用它。真正调用它的是
`scripts/train.py:235` 与 `scripts/play.py:429`，两处都在 `main()` 的第一行。

**为什么不做成 import 副作用**：`setup_utf8_console()` 会改进程级的
`sys.stdout` / `sys.stderr`。把它挂在包导入上，意味着任何第三方
`import robotdog.gait`（只想读一个 JSON 的小脚本、离线分析、pytest fixture）都会
**顺带改进程 I/O**——副作用范围远超「导入一个数据结构模块」的合理预期。
显式调用把「我确实要输出中文」这个意图留在调用点上，代价只是一行。

Windows 上重定向到文件/管道时 `sys.stdout.encoding` 退化为 cp936，中文被写成 GBK
字节，属于「不调就一定踩」的坑，所以两个脚本都调，并且 `main()` 第一行就调
（任何 print 之前）。

安全性由被调用方保证：`setup_utf8_console()` 幂等，没有 `reconfigure` 的流直接跳过，
`ValueError/OSError` 静默 `pass`——所以重复调用无害。

### D2 —— 绝不重导出 `robotdog.env`

三条理由，第 1 条是硬约束：

1. **依赖方向**。`docs/README.md` 的「分层规则」要求 `训练层 → 环境层 → 工具层` 单向。
   本文件第 14 行已经会触发 `robotdog.tools`（见 §5），若它再导入 `robotdog.env`，
   那么连最底层的 `import robotdog.tools.terrain` 也要先执行 `robotdog.env`，
   `tools ↔ env` 立刻成环——这正是 `render.py` 当年被留在顶层而没有放进 `tools/`
   的同一条理由（该模块已在「盲走」重构中删除，但分层约束不变）。
2. **import 成本**。实测（本仓 `.venv`，`python -c "import sys; sys.path.insert(0,'.'); import robotdog; ..."`）：
   `numpy=False`、`mujoco=False`、`gymnasium=False`、`torch=False`、
   `stable_baselines3=False`、`robotdog.env=False`、`dataclasses=True`。包入口当前
   **只碰标准库**，**不碰** numpy / Gymnasium / PyTorch；一旦重导出 `env`，
   任何 `import robotdog.gait` 都会连带拉起 Gymnasium 与 `env.py` 的全部配置机械。
3. 只想读 `__version__` 或 `GaitParams` 的离线工具（例如检查 JSON 的小脚本）不该被迫
   构造一个完整的 Gymnasium 环境。

**被否决的方案**：在 `__init__.py` 里写 `from robotdog.env import Go2StairsEnv`，让
`robotdog.Go2StairsEnv` 可用。否决原因即上面第 1、2 条：与分层规则直接冲突，且代价是
每个工具导入都变重。

### D3 —— `__all__` 列四个名字，与重导出集合严格一致

`__all__` 必须与 `:14` / `:15` 两条 import 重导出的名字**逐字一致**（`console`、`paths`、
`setup_utf8_console`、`terrain`）。它同时是一份契约：任何「多导出一个业务类」的改动都会
先撞到这个列表，从而被迫回答 D2 里的分层问题。

## 4. 公开 API

### `__version__: str`

包版本，字面量 `"0.2.0"`（`robotdog/__init__.py:17`）。只被脚本与报告引用，
不参与任何逻辑分支。

### `setup_utf8_console() -> None`（重导出）

真身在 `robotdog/tools/console.py:18`。语义、参数、异常详见
`docs/robotdog/tools/console.md`，本文不重复：它**不返回值**、**不抛异常**，
对 `("stdout", "stderr")` 逐个执行 `reconfigure(encoding="utf-8", errors="replace")`。

### 完整源码（19 行）

```python
"""Go2 上下楼梯强化学习的最小核心包。

把工具层的三个叶子模块与 :func:`robotdog.tools.console.setup_utf8_console`
重导出到包命名空间。

注意：**导入本包不会自动修控制台编码** —— 脚本必须自己调一次
``console.setup_utf8_console()``（``scripts/train.py`` / ``scripts/play.py``
都在 ``main()`` 的第一行这么做），否则 Windows 下打印中文 HUD / 报表会撞
``UnicodeEncodeError``。
"""

from __future__ import annotations

from robotdog.tools import console, paths, terrain
from robotdog.tools.console import setup_utf8_console

__version__ = "0.2.0"

__all__ = ["console", "paths", "setup_utf8_console", "terrain"]
```

## 5. 内部结构与数据流

`import robotdog` 的执行链——注意第 1 步会**连带把三个工具模块全部加载**：

```
import robotdog
  → robotdog/__init__.py 模块体开始执行
    → :14 from robotdog.tools import console, paths, terrain
      → import robotdog.tools                # 执行 robotdog/tools/__init__.py
        → from robotdog.tools import console, paths, terrain
          → robotdog.tools.console  → setup_utf8_console 定义
          → robotdog.tools.paths    → WORKSPACE = Path(__file__).resolve().parents[2]（模块级求值）
          → robotdog.tools.terrain  → import dataclasses（不 import numpy / mujoco）
    → :15 from robotdog.tools.console import setup_utf8_console
    → :17 __version__ = "0.2.0"
    → :19 __all__ = ["console", "paths", "setup_utf8_console", "terrain"]
```

没有「调用 `setup_utf8_console()`」的那一步——这是 D1 的直接结果。

**推论**：`import robotdog` 只加载 `pathlib`、`sys`、`dataclasses` 这类标准库，
**不加载 `numpy`、不加载 `mujoco`、不加载 `gymnasium`、不加载 `env.py`**（实测见 §8）。
（触觉模块 `tools/touch.py` 曾在这里，它把 `mujoco` 也一起拉进来；该模块已在
「盲走」重构中删除。）

## 6. 关键常量与默认值

| 名字 | 值 | 位置 |
| --- | --- | --- |
| `__version__` | `"0.2.0"` | `robotdog/__init__.py:17` |
| `__all__` | `["console", "paths", "setup_utf8_console", "terrain"]` | `robotdog/__init__.py:19` |
| 文件行数 | 19 行 | — |

## 7. 与其它模块的关系

* **上游**：无（它是依赖图最上游）。
* **下游**：`robotdog.tools`（`console` / `paths` / `terrain`）。
* **绝不指向**：`robotdog.env` 及其任何依赖（D2）。
* **被谁触发**：任何 `import robotdog.*` 都会先执行本文件（Python 包语义）。
* **被谁显式调用**：`scripts/train.py:235`、`scripts/play.py:429`（`setup_utf8_console()`）。

## 8. 实测与经验教训

* **DOC-ROT 样本**：本文档曾描述「模块体第 11 行直接调用 `setup_utf8_console()`」，
  并贴过一份 13 行的旧源码（`__version__ = "0.1.0"`、`__all__` 只有两个名字）。
  真实文件从未在模块体里调用过它——**「导入即生效」从头到尾只存在于 docstring 与文档里**。
  发现它靠的是 `.tmp/check_lines.py` 的行号检查（`:7` 指向空行）加人工核对 `__all__`
  与 import 行；这类「副作用根本不存在」的事实，行号检查抓不到，必须靠
  「把文档里每个声称的行为回到源码里找那一行」。
* **连带修掉**：`robotdog/tools/console.py` 的模块 docstring 也写着「由
  `robotdog/__init__.py` 自动调用，所以只要 `import robotdog` 就生效」——同样是假的，
  已改为「由脚本在启动时显式调用一次」。**同一句假话写在两个文件里，是因为它们当初
  一起被写下、之后一起没被读。**
* `import robotdog` 的实测开销：`numpy=False`、`mujoco=False`、`gymnasium=False`、
  `torch=False`、`robotdog.env=False`、`dataclasses=True`（见 D2 第 2 条）。
  **注意**：`robotdog.tools` 的模块 docstring 说「只加载 numpy」，那是错的——`terrain`
  删掉两个射线函数后已不 import numpy，`paths` 只用 `pathlib`，`console` 只用 `sys`。

## 9. 已知限制与坑

* **不要在这儿加业务重导出**。每加一个都会把 `import robotdog.gait` 的代价推高，
  并且立刻违反 D2。
* **不要为了「方便」在模块体里调用 `setup_utf8_console()`**。D1 保留了显式调用，
  重复把副作用搬回导入路径会重新制造「导入一个纯数据模块却改了 stdout」的问题。
* `__all__` 与两条 import 必须同步；`from robotdog import *` 只会带出这四个名字。
* 包入口没有 `__getattr__` 懒加载：`terrain` 是 eager 的，所以 `import robotdog` 必然把
  三个工具模块放进 `sys.modules`（只含标准库，见 D2 第 2 条）。

## 10. 示例用法

```python
import robotdog
from robotdog import setup_utf8_console

# 想输出中文就自己调一次（幂等）；不调也不会乱码崩溃，只是重定向时可能乱码。
setup_utf8_console()

print(robotdog.__version__)          # 0.2.0
print(robotdog.paths.WORKSPACE)      # 仓库根
print(robotdog.terrain.TERRAIN_GEOM_GROUP)   # 1

# 业务类必须从各自的模块导入 —— 包入口故意不重导出它们。
from robotdog.env import EnvConfig, Go2StairsEnv  # noqa: E402
```
