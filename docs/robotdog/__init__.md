# `robotdog/__init__.py` —— 包入口

## 1. 一句话定位

包入口层，13 行：导入时修复 UTF-8 控制台、暴露版本号、把 `from robotdog import *`
限制成两个名字。它不依赖任何业务模块（只依赖工具层的一个叶子），因此位于依赖图最上游。

## 2. 职责边界

**做**：

* `from robotdog.tools.console import setup_utf8_console` —— 重导出该函数；
* 定义 `__version__ = "0.1.0"`；
* 在模块体（import 阶段）**直接调用** `setup_utf8_console()`；
* 用 `__all__` 把 `*` 导入限定为 `["__version__", "setup_utf8_console"]`。

**不做**：

* **不重导出环境**：`Go2StairsEnv` / `EnvConfig` / `ObsConfig` / `RewardConfig`
  都必须写 `from robotdog.env import Go2StairsEnv`（`README.md` §2 结尾的原话）；
* 不重导出 `gait` / `go2_model` / `training` / `curriculum` / `evaluate` / `render`；
* 不建目录、不读配置、不配 logging、不解析命令行、不定义 `main`。

## 3. 关键设计决策与理由

### D1 —— 把「修控制台」做成 import 副作用

`setup_utf8_console()` 在模块体第 11 行被直接调用，所以只要有一行 `import robotdog`，
stdout/stderr 就已是 UTF-8。**为什么放这里**：本仓所有入口（`scripts/train.py`、
`scripts/eval.py`、`scripts/render_checkpoint.py`）第一步都要导入 `robotdog` 系模块，
而且 Python 的包语义保证任何 `import robotdog.gait` 也会先执行本文件——这是唯一能覆盖
全部导入路径的位置。Windows 上重定向到文件/管道时 `sys.stdout.encoding` 退化为 cp936，
中文被写成 GBK 字节，属于「不修就一定踩」的坑。

安全性由被调用方保证：`setup_utf8_console()` 幂等，没有 `reconfigure` 的流直接跳过，
`ValueError/OSError` 静默 `pass`——所以把它做成副作用不会让 import 失败。

**被否决的方案**：在每个脚本里各写一行 `setup_utf8_console()`。否决原因——漏写就是静默
乱码，而且第三方 `import robotdog.gait` 拿不到修复；分散的调用点无法穷举。

### D2 —— 绝不重导出 `robotdog.env`

三条理由，第 1 条是硬约束：

1. **依赖方向**。`docs/README.md` 的「分层规则」要求 `训练层 → 环境层 → 工具层` 单向。
   本文件第 7 行已经会触发 `robotdog.tools`（见 §5），若它再导入 `robotdog.env`，
   那么连最底层的 `import robotdog.tools.terrain` 也要先执行 `robotdog.env`，
   `tools ↔ env` 立刻成环——这正是 `render.py` 被留在顶层而没有放进 `tools/` 的同一条理由。
2. **import 成本**。实测（本仓 `.venv`，`python -c "import sys; import robotdog"`）：
   `mujoco=True`、`numpy=True`，而 `gymnasium=False`、`torch=False`、
   `stable_baselines3=False`、`robotdog.env=False`。包入口当前**不碰** Gymnasium / PyTorch；
   一旦重导出 `env`，任何 `import robotdog.gait` 都会连带拉起 Gymnasium 与 `env.py`
   的全部配置机械。
3. 只想读 `__version__` 或 `GaitParams` 的离线工具（例如检查 JSON 的小脚本）不该被迫
   构造一个完整的 Gymnasium 环境。

**被否决的方案**：在 `__init__.py` 里写 `from robotdog.env import Go2StairsEnv`，让
`robotdog.Go2StairsEnv` 可用。否决原因即上面第 1、2 条：与分层规则直接冲突，且代价是
每个工具导入都变重。

### D3 —— `__all__` 显式只列两个名字

避免 `from robotdog import *` 泄漏 `__init__` 命名空间里的其它名字。这个列表同时是一份
**契约**：`README.md` §2 那句「`robotdog\__init__.py` **只**导出 `setup_utf8_console`」
就是靠它保证的。

## 4. 公开 API

### `__version__: str`

包版本，字面量 `"0.1.0"`（`robotdog/__init__.py:9`）。只被脚本与报告引用，
不参与任何逻辑分支。

### `setup_utf8_console() -> None`（重导出）

真身在 `robotdog/tools/console.py:17`。语义、参数、异常详见
`docs/robotdog/tools/console.md`，本文不重复：它**不返回值**、**不抛异常**，
对 `("stdout", "stderr")` 逐个执行 `reconfigure(encoding="utf-8", errors="replace")`。

### 完整源码（13 行）

```python
"""Robot-dog stair-climbing RL project (MuJoCo + Gymnasium + Stable-Baselines3).

Importing this package also repairs the console encoding, so Chinese log output
survives Windows redirection (see :mod:`robotdog.tools.console`).
"""

from robotdog.tools.console import setup_utf8_console

__version__ = "0.1.0"

setup_utf8_console()

__all__ = ["__version__", "setup_utf8_console"]
```

## 5. 内部结构与数据流

`import robotdog` 的执行链——注意第 2 步会**连带把五个工具模块全部加载**：

```
import robotdog
  → robotdog/__init__.py 模块体开始执行
    → from robotdog.tools.console import setup_utf8_console
      → import robotdog.tools                # 执行 robotdog/tools/__init__.py
        → from robotdog.tools import console, paths, png, terrain, touch
          → robotdog.tools.console  → setup_utf8_console 定义
          → robotdog.tools.paths    → WORKSPACE = Path(__file__).resolve().parents[2]（模块级求值）
          → robotdog.tools.png      → import struct, zlib, numpy
          → robotdog.tools.terrain  → import numpy
          → robotdog.tools.touch    → import mujoco, numpy
    → __version__ = "0.1.0"
    → setup_utf8_console()                   # 副作用发生在这里
    → __all__ = ["__version__", "setup_utf8_console"]
```

**推论**：`robotdog.tools` 是重的（eager 导入 `terrain` 与 `touch`，于是 `mujoco`
必然进 `sys.modules`），而它的 `__init__` 恰好被本文件第 7 行提前触发。所以
「避免 MuJoCo 开销」这件事上，包入口本身已经做不到——见 §9。

## 6. 关键常量与默认值

| 名字 | 值 | 位置 |
| --- | --- | --- |
| `__version__` | `"0.1.0"` | `robotdog/__init__.py:9` |
| `__all__` | `["__version__", "setup_utf8_console"]` | `robotdog/__init__.py:13` |
| 文件行数 | 13 行 | — |
| import 副作用 | `setup_utf8_console()` 调用 1 次 | `robotdog/__init__.py:11` |
| UTF-8 目标流 | `("stdout", "stderr")` | `robotdog/tools/console.py:19` |
| 编码 / 错误策略 | `encoding="utf-8"`、`errors="replace"` | `robotdog/tools/console.py:25` |

## 7. 与其它模块的关系

**上游（本文件依赖谁）**：

* `robotdog.tools.console` —— 唯一 import。理由：UTF-8 修复是所有其它模块打印中文的
  前提，必须最先生效。

**下游（谁依赖本文件）**：

* `robotdog.gait` / `env` / `training` / `curriculum` / `evaluate` / `render` / `tools`：
  **全部**。Python 的包语义决定任何 `import robotdog.X` 都要先执行 `robotdog/__init__.py`。
* `scripts/*.py`：靠 `import robotdog.<mod>` 顺带拿到控制台修复。

**方向为什么是这样**：本文件处在依赖图最上游，只向下依赖工具层的一个叶子，因此不会引入
环；反过来若它依赖环境层，`tools → env` 的环由 §3 D2 第 1 条立即触发。

## 8. 实测与经验教训

* `README.md` §2 结尾明确「`robotdog\__init__.py` **只**导出 `setup_utf8_console`，
  脚本要显式 `from robotdog.env import ...`」——本文 §2/§3 是这条约束的展开。
* `docs/README.md` 的「分层规则」定义了 `tools` 不 import `robotdog.env`、
  依赖单向 `训练层 → 环境层 → 工具层`；包入口的重导出决策必须服从这条规则。
* 重定向相关的流/编码坑（`| Tee-Object` 丢掉 stderr、Windows `>` 写成 UTF-16、
  要用 `Out-File -Encoding utf8`）见 `experience.md` §5.6 与 §5.8；本文不复述其结论。

## 9. 已知限制与坑

* **`robotdog.Go2StairsEnv` 不存在**。`from robotdog import Go2StairsEnv` 会得到
  `ImportError: cannot import name 'Go2StairsEnv' from 'robotdog'`；
  正确写法是 `from robotdog.env import Go2StairsEnv`。
* **`import robotdog` 不是免费的**：它连带加载 `robotdog.tools`，实测 `mujoco`、`numpy`
  已经进 `sys.modules`。`robotdog/tools/__init__.py` 一次性导入五个工具（实测
  `mujoco=True`、`gymnasium=False`），所以只想写一张 PNG 也会付出加载 `mujoco` 的代价。
  想要真正轻量的调用方只能绕开这个包。
* **副作用不可关**：没有 `disable_console_fix()` 之类的开关，修复永远发生。依赖 stdout
  原始编码的调用方必须在 import 之后自行再 `reconfigure` 回去。
* **`errors="replace"` 静默替换**：无法编码的字符变成 `?` 而不是报错，排查乱码时
  不能只看异常。
* **没有 PEP 562 兜底**：本文件未定义 `__getattr__`，任何未列出的名字都是 `AttributeError`。

## 10. 示例用法

```python
import robotdog                          # 副作用：stdout/stderr 变成 UTF-8

print(robotdog.__version__)              # 0.1.0
print(robotdog.__all__)                  # ['__version__', 'setup_utf8_console']

robotdog.setup_utf8_console()            # 幂等，可重复调用

from robotdog.env import Go2StairsEnv    # 环境必须显式导入
from robotdog.gait import GaitParams

env = Go2StairsEnv()
```
