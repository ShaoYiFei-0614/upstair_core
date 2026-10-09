# `robotdog/tools/__init__.py` —— 工具层子包入口

## 1. 一句话定位

**工具层**（`robotdog/tools/`）的子包入口：把工具层由哪五个模块组成这件事写成代码，并因此成为「谁可以依赖谁」这条分层规则的强制执行点。它不解决任何计算问题。

## 2. 职责边界

**做**：

* 用一行绝对导入把五个子模块绑进 `robotdog.tools` 的命名空间：`console`、`paths`、`png`、`terrain`、`touch`。
* 用 `__all__` 声明包级公开名。
* 在模块 docstring 里列出五个模块各自的职责（一行一个），并声明「工具层不 import `robotdog.env`」。

**不做**：

* **不做符号聚合**：`StairSpec` / `TouchConfig` / `WORKSPACE` 等符号**没有**被提升到包级。`from robotdog.tools import StairSpec` 会失败。
* **不做包装**：不 re-export 函数、不写 `__getattr__`、不做懒加载（`importlib.util.LazyLoader` / 模块级 `__getattr__`）。
* **不产生副作用**：不调用 `setup_utf8_console()`、不建目录、不打印、不读环境变量。
* **不知道 PPO / 课程 / 训练循环**，也不 import `robotdog.env`。

## 3. 关键设计决策与理由

### 3.1 只导出**模块对象**，不导出符号

```python
from robotdog.tools import console, paths, png, terrain, touch

__all__ = ["console", "paths", "png", "terrain", "touch"]
```

理由：包级命名空间保持成一张**稳定的名称表**，而不是第二套 API 面。全仓消费者都写精确导入 —— `robotdog/env.py:43-48` 写 `from robotdog.tools.terrain import StairSpec, build_terrain_xml`、`from robotdog.tools.touch import (TouchConfig, TouchReading, TouchSensor, empty_reading)`；`scripts/train.py:53` 写 `from robotdog.tools.paths import LOGS_DIR`。精确导入让「这个符号来自哪个文件」在阅读时不需要跳转。

**被否决的方案**：

* `from robotdog.tools import *` —— 否决，因为 `*` 会把五个模块的全部公开名（含 `np`、`mujoco` 这类再导出）灌进调用方命名空间，改子模块就会无意地改调用方的作用域。
* 把 `StairSpec`、`TouchConfig`、`WORKSPACE` 提升到包级 —— 否决，因为这会让包入口与子模块的**内部命名**耦合：子模块改一个类名就要同步改包入口和它的 `__all__`，而收益只是少打一段导入路径。
* 懒加载（模块级 `__getattr__` + `LazyLoader`）—— 否决，因为工具层模块之间只有一条真实依赖（`touch → terrain`），五个模块全量导入的成本主要来自 `numpy`/`mujoco`，而这两个库在**任何**真实使用场景里都必然被加载；懒加载换来的只是「拿路径常量不付 MuJoCo 加载代价」这一种边缘收益，却引入 import 顺序相关的隐蔽行为。

### 3.2 依赖顺序天然正确，不需要手工排序

导入行按字母序写 `console, paths, png, terrain, touch`，`__all__` 用同一顺序。工具层内部的真实依赖只有一条：`robotdog/tools/touch.py:53` 的 `from robotdog.tools.terrain import TERRAIN_GEOM_GROUP`。字母序里 `terrain` 在 `touch` 之前，因此这一行同时也是合法的加载顺序。

理由：把「顺序正确」做成字母序的副产品，而不是靠一句注释约定。**被否决的方案**：手工把 `terrain` 提前并写注释解释原因 —— 否决，因为注释会随重构失效，而字母序不会。

### 3.3 不在这里调用 `setup_utf8_console()`

理由：编码修复必须在**任何输出发生之前**生效，而 `robotdog/tools/__init__.py` 会在 `robotdog/__init__.py` 第 7 行的导入语句期间就被执行（导入 `robotdog.tools.console` 会先触发父包 `robotdog.tools` 的 `__init__`）。调用点放在 `robotdog/__init__.py:11`，因为那是「整个进程里第一个 robotdog 代码」的确定位置。

**被否决的方案**：在 `robotdog/tools/__init__.py` 里调用 —— 否决有两条：(1) 它让「导入一个路径常量」这种纯数据操作产生进程级副作用；(2) 顺序会变成「谁先 import 谁决定」，而 `robotdog.tools.console` 也可以被单独导入，那条路径下 `robotdog/__init__` 已经在更早的位置跑过了，重复调用虽然安全（见 `console.md` §3），但语义上变成了巧合而不是设计。

### 3.4 docstring 里显式写出五个模块的职责与分层禁令

模块 docstring 第 3-6 行声明「everything in this subpackage is a *tool*: it knows nothing about PPO, the curriculum or the training loop, and none of it imports `robotdog.env`」。

理由：这条规则没法用代码表达（Python 没有分层检查），所以把它写在**唯一一个必然被每个工具层消费者读到的文件**里。它是「新增一个工具模块之前必须回答的问题」：它依赖 `env` 吗？依赖就不许放进来 —— `robotdog/render.py` 正是因此留在顶层（它需要驱动 `Go2StairsEnv`），否则会形成 `tools ↔ env` 环。

## 4. 公开 API

签名逐字来自源码：

```python
from __future__ import annotations

from robotdog.tools import console, paths, png, terrain, touch

__all__ = ["console", "paths", "png", "terrain", "touch"]
```

| 公开名 | 类型 | 语义 |
| --- | --- | --- |
| `robotdog.tools.console` | module | `robotdog.tools.console` 模块对象（UTF-8 控制台修复） |
| `robotdog.tools.paths` | module | 路径常量模块 |
| `robotdog.tools.png` | module | 无依赖 PNG 读写 |
| `robotdog.tools.terrain` | module | 参数化楼梯几何 + MJCF 地形 |
| `robotdog.tools.touch` | module | 接触力 + 前方测距 |
| `__all__` | `list[str]` | 上面五项，顺序即 `__all__` 顺序 |

* 参数：无。
* 返回值：无（模块级语句）。
* 抛出的异常：只有导入期异常，即任一子模块导入失败时原样冒泡。本文件自身不 `raise`、不 `try/except`。
* 副作用：**没有**（不建目录、不改 `sys.stdout`、不打印）。唯一可观察的副作用是「五个子模块被加载进 `sys.modules`」，这会连带加载 `numpy`（`png`/`terrain`/`touch`）与 `mujoco`（`touch` 顶层导入）。

## 5. 内部结构与数据流

最常见的入口是 `robotdog/__init__.py:7`，此时四个文件按下面的顺序执行：

```
import robotdog
  → robotdog/__init__.py:7  from robotdog.tools.console import setup_utf8_console
      → (父包) robotdog/tools/__init__.py
          → from robotdog.tools import console, paths, png, terrain, touch
              → robotdog/tools/console.py   (仅 import sys)
              → robotdog/tools/paths.py     (仅 import pathlib)
              → robotdog/tools/png.py       (import struct, zlib, pathlib, numpy)
              → robotdog/tools/terrain.py   (import dataclasses, numpy)
              → robotdog/tools/touch.py     (import dataclasses, mujoco, numpy;
                                             from robotdog.tools.terrain import TERRAIN_GEOM_GROUP
                                             → 命中已加载的 terrain 模块，不重复执行)
      → console.setup_utf8_console() 已可取用
  → robotdog/__init__.py:11  setup_utf8_console()   ← 副作用在这里发生，不在工具层
```

从任一子模块进入时，父包 `robotdog.tools` 必然先执行，所以上表对 `from robotdog.tools.paths import LOGS_DIR` 同样成立：

```
from robotdog.tools.paths import LOGS_DIR
  → robotdog/tools/__init__.py  → 五个模块全部加载（含 mujoco）
  → robotdog/tools/paths.py     → LOGS_DIR 已可用
```

## 6. 关键常量与默认值

本文件不含数值常量。可列举的字面量只有：

| 项 | 值 |
| --- | --- |
| `__all__` 元素个数 | 5（`"console"`, `"paths"`, `"png"`, `"terrain"`, `"touch"`） |
| 导入的模块数 | 5 |
| 数值常量 | 无 |
| 默认值 | 无 |

## 7. 与其它模块的关系

**上游**：无 robotdog 依赖。只有 stdlib（`from __future__ import annotations`）。

**下游**（消费者，全部是「按名字取子模块」，不依赖包级再导出）：

| 位置 | 用法 |
| --- | --- |
| `robotdog/__init__.py:7` | `from robotdog.tools.console import setup_utf8_console` |
| `robotdog/env.py:43-48` | `from robotdog.tools.terrain import ...`、`from robotdog.tools.touch import ...` |
| `robotdog/go2_model.py:41-42` | `from robotdog.tools.paths import GO2_ASSETS_DIR, GO2_XML`、`from robotdog.tools.terrain import TERRAIN_GEOM_GROUP` |
| `robotdog/go2_model.py:423` | `from robotdog.tools.paths import GENERATED_DIR`（函数内延迟导入） |
| `robotdog/curriculum.py:27` | `from robotdog.tools.terrain import StairSpec, demo_stairs_spec, flat_spec` |
| `robotdog/evaluate.py:28-29` | `from robotdog.tools.terrain import ...` |
| `robotdog/render.py:31-32` | `from robotdog.tools.png import downscale, hstack, write_png`、`from robotdog.tools.terrain import StairSpec` |
| `robotdog/training.py:29-31, 357` | `paths.RUNS_DIR`、`terrain.StairSpec`、`from robotdog.tools.touch import TouchConfig` |
| `scripts/train.py:53` | `from robotdog.tools.paths import LOGS_DIR` |
| `scripts/render_checkpoint.py:46-65` | `import robotdog`、`from robotdog.tools import paths`、`from robotdog.tools.png import downscale, write_png`、`from robotdog.tools.terrain import StairSpec, flat_spec` |
| `scripts/eval.py:32-33, 147, 216` | `import robotdog`、`from robotdog.tools import paths`、函数内 `from robotdog.tools.terrain import flat_spec` / `from robotdog.tools.png import downscale, hstack, write_png` |
| `tools/ab_pair.py:46-49` | `from robotdog.tools.console import setup_utf8_console`、`from robotdog.tools.terrain import StairSpec` |

**为什么方向是这样**：分层是单向的 `训练层 → 环境层 → 工具层`（`docs/README.md`「分层规则」表）。工具层被所有层依赖，所以它自己不能依赖任何一层 —— 一旦 `tools/` 里出现 `import robotdog.env` 就成环。包入口就是这条规则的落点：它列出的五个模块**就是**工具层的全部内容，新增模块必须回答「它依赖 `env` 吗」。

顶层**没有**同名模块：`robotdog/console.py`、`paths.py`、`png.py`、`terrain.py`、`touch.py` 都不存在，唯一的导入路径是 `robotdog.tools.<name>`。早期版本放过一层 `sys.modules[__name__] = _mod` 的向后兼容替身，已删除（替身文件本身没有逻辑，报错 traceback 与 IDE 静态分析都会指向它）。

## 8. 实测与经验教训

* 控制台编码与重定向类坑见 `experience.md` §5.6、§5.8（本文档不重复结论）。
* 「工具层不 import `env`」这条规则的来龙去脉与 `render.py` 为什么留在顶层，见 `README.md` §2、§3 与 `docs/README.md`「分层规则」。
* 训练分层（`DummyVecEnv → VecMonitor → VecNormalize`）与逐阶段课程组织见 `train.md` §1。

## 9. 已知限制与坑

* **包级没有符号**。写 `from robotdog.tools import StairSpec` 得到
  `ImportError: cannot import name 'StairSpec' from 'robotdog.tools'`。正确写法是 `from robotdog.tools.terrain import StairSpec`。
* **`import robotdog.tools` 会加载 `mujoco`**。因为 `touch.py` 顶层 `import mujoco`，而 `__init__` 立即导入全部五个模块。只想拿路径常量也会付出这个代价。
* **`__all__` 只影响 `from robotdog.tools import *`**。`robotdog.tools.console` 等属性无论如何都可访问。
* **模块名就是文件路径**。规范路径 `robotdog.tools.terrain` 对应 `robotdog/tools/terrain.py`；报错信息里的模块路径不会指向别的文件，因为顶层没有兼容层。
* **本文件不调用 `setup_utf8_console()`**。任何 `import robotdog.*`（包括 `import robotdog.tools.paths`）都会先执行 `robotdog/__init__.py`，所以实际使用中编码修复总会生效；但「导入工具层就等于修好了编码」是父包的副作用，不是工具层的契约。

## 10. 示例用法

```python
# 推荐：精确导入符号
from robotdog.tools.paths import LOGS_DIR, OUTPUT_GIFS_DIR
from robotdog.tools.terrain import StairSpec, flat_spec

# 需要整个模块时才这样写（会连带加载 numpy 与 mujoco）
from robotdog.tools import paths, terrain

print(paths.WORKSPACE)                 # D:\...\upstair_core
print(terrain.flat_spec().n_up)        # 0
print(StairSpec().crest_height)        # 0.4
```
