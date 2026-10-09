# `robotdog/tools/console.py` —— 控制台编码修复

## 1. 一句话定位

**工具层**（`robotdog/tools/`）里的进程级 I/O 适配器：把 `sys.stdout` / `sys.stderr` 强制改成 UTF-8，解决 Windows 下 Python 输出被重定向到文件或管道时 `sys.stdout.encoding` 退化成本地代码页（cp936）、中文日志写成 GBK 字节、读取方按 UTF-8 解码得到乱码的问题。它不产生任何业务数据，只保证已经存在的输出**可读**。

## 2. 职责边界

**做**：

* 对 `"stdout"`、`"stderr"` 两个流各调用一次 `stream.reconfigure(encoding="utf-8", errors="replace")`。
* 幂等：可以重复调用；对任意流都安全（没有 `reconfigure` 属性就跳过）。
* 吞掉 `ValueError` / `OSError`（流已分离或已关闭）。

**不做**：

* **不解析编码**：不检测当前编码、不比较新旧编码、不做条件判断。
* **不改 `PYTHONIOENCODING`**、不改环境变量、不改 `sys.stdin`。
* **不设置 `sys.setdefaultencoding`**、不碰 locale、不调用 Windows API（`SetConsoleOutputCP` 之类）。
* **不安装日志 handler**、不包装 `print`、不写终端颜色。
* **不 import `robotdog.env`**，不知道 PPO / 课程 / 训练循环。
* **不在本模块里自我调用**。它是一个被调用的函数，不是 import 副作用。

## 3. 关键设计决策与理由

### 3.1 只碰 stdout/stderr，不碰 stdin

理由：乱码问题只发生在**输出**方向 —— 中文写出去被当 UTF-8 读。输入侧本仓没有交互式中文输入（全部是 CLI 参数与 YAML/JSON 配置），加进去只会扩大影响面。**被否决的方案**：三个流一起改 —— 否决，因为在探测/管道场景下动 `stdin` 可能改变父进程看到的缓冲行为，而收益为零。

### 3.2 用 `stream.reconfigure()` 而不是 `io.TextIOWrapper` 重新包一层

`reconfigure(encoding=..., errors=...)` 是 Python 3.7+ 在已有 `TextIOWrapper` 上原地改编码的官方接口，不需要替换 `sys.stdout` 对象本身，因此任何已经持有 `sys.stdout` 引用的代码（包括 `logging` 早期创建的 handler）都能看到新编码。

**被否决的方案**：

* `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")` —— 否决，因为它**换掉了对象**：任何缓存的旧引用继续用旧编码写，日志文件一半 UTF-8 一半 GBK；而且旧 wrapper 在 GC 时可能被关闭，导致后续写入报 `ValueError: I/O operation on closed file`。
* `sys.stdout.buffer.write` 逐字节转发 —— 否决，要自己实现包装类，等于重新发明 `TextIOWrapper`。

### 3.3 `reconfigure` 用 `getattr` + `None` 检查，而不是 `hasattr` / `try`

```python
reconfigure = getattr(stream, "reconfigure", None)
if reconfigure is None:
    continue
```

理由：把「属性不存在」和「属性存在但调用失败」这两种情况**分开处理**。`reconfigure` 不存在是正常的（例如被替换成自定义流对象、IPython 的 `OutStream`），静默跳过；调用抛异常才是异常路径，用 `try` 兜。用 `hasattr` 再取属性会做两次查找，且无法区分属性值是 `None` 的情况。

**被否决的方案**：直接用 `try: stream.reconfigure(...) except AttributeError: pass` —— 否决，因为 `AttributeError` 可能来自 `reconfigure` **内部**（某个流实现里的 bug），会被误吞成「这个流不支持」。

### 3.4 异常只兜 `(ValueError, OSError)`，不兜 `Exception`

`getattr(sys, stream_name, None)` 一起兜掉了「解释器没有 `sys.stdout`」（pythonw、`-c` 下 `sys.stdout` 为 `None`）的情况 —— 此时 `stream` 是 `None`，`getattr(None, "reconfigure", None)` 返回 `None`，走 `continue`。

`reconfigure` 已知会在两种情形抛异常并写入源码注释 `# 例如流已分离或已关闭；这里无能为力`（`robotdog/tools/console.py:27`）：

* `ValueError`：流已关闭。
* `OSError`：流已分离（detached）。

二者都是「环境已经坏了，但程序还能继续跑」，所以吞掉。**被否决的方案**：兜 `Exception` —— 否决，因为这会掩盖真正的 bug（例如某天有人给流对象传了错误的参数类型），而编码修复失败的后果只是乱码，不是崩溃。

### 3.5 调用点是脚本，不是 `robotdog/__init__.py`

`robotdog/__init__.py:15` 只**导入**它、**从不调用**它。真实调用点是
`scripts/train.py:235` 与 `scripts/play.py:429`，两处都在 `main()` 的第一行。

理由：编码必须在**任何输出发生之前**生效，而脚本的 `main()` 第一行正是这个确定位置；
放成包导入的副作用会让「导入一个纯数据模块」改进程 I/O，范围超出合理预期（见
`docs/robotdog/__init__.md` §3 D1）。放在 `robotdog/tools/__init__.py` 更糟：它会变成
「谁先 import 谁决定」，而且让「导入一个路径常量」这种纯数据操作产生全局副作用。

**被否决的方案**：在 `console.py` 模块层直接调用自己 —— 否决，因为那样连
`from robotdog.tools.console import setup_utf8_console` 都会改进程 I/O，而这个函数
**需要**能被显式调用（`scripts/train.py:235` / `scripts/play.py:429` 正是这么用的），
显式与隐式两条路径会叠加。

**历史腐坏**：本文档与 `robotdog/__init__.py`、`robotdog/tools/console.py` 三处曾一致声称
「导入 `robotdog` 即自动修控制台编码」，`git show HEAD:robotdog/__init__.py` 证明该调用
从来不存在。三处文案已改成事实；源码里**故意没有补上这次调用**（补调用会改变
`import robotdog` 的进程级 I/O 行为，超出「删无用代码 + 中文化注释」的范围）。

## 4. 公开 API

```python
from __future__ import annotations

import sys

__all__ = ["setup_utf8_console"]


def setup_utf8_console() -> None:
    """让 stdout/stderr 输出 UTF-8；可重复调用，对任何流都安全。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # 例如流已分离或已关闭；这里无能为力
            pass
```

（`robotdog/tools/console.py:18-29`。模块 docstring 在 `:1-9`，`__all__` 在 `:15`。）

| 项 | 值 |
| --- | --- |
| 签名 | `setup_utf8_console() -> None` |
| 参数 | 无 |
| 返回 | `None` |
| 异常 | **不抛出任何异常**。`ValueError` / `OSError` 被就地吞掉 |
| 副作用 | 把 `sys.stdout`、`sys.stderr` 的编码设为 `utf-8`、错误处理设为 `replace` |
| 模块级公开名 | 仅 `setup_utf8_console`（`__all__` 同时是它） |

* **语义细节**：`errors="replace"` 意味着无法编码的字符写成 `?`（U+FFFD）而不是抛 `UnicodeEncodeError`。这是刻意的 —— 渲染 HUD、打印路径时遇到生僻字不应中断训练进程。
* **语义细节**：只对当前 `sys.stdout` / `sys.stderr` 生效。之后若有人替换这两个对象（例如 `contextlib.redirect_stdout`），新对象不受本函数约束，需要重新调用。
* 本模块不含类、不含 dataclass、不含常量。

## 5. 内部结构与数据流

一次真实调用（脚本在 `main()` 第一行主动调用）：

```
scripts/train.py:54   from robotdog.tools import console  # noqa: E402
scripts/train.py:235  console.setup_utf8_console()        # main() 的第一条语句
  → setup_utf8_console()
      → for stream_name in ("stdout", "stderr")
          → getattr(sys, stream_name, None)          # 得到 TextIOWrapper 或 None
          → getattr(stream, "reconfigure", None)     # 得到绑定方法或 None
          → reconfigure(encoding="utf-8", errors="replace")
              → (内部重建 TextIOWrapper 的编解码器)
          → except (ValueError, OSError): pass
```

`scripts/play.py:48` 精确导入同一个模块，`scripts/play.py:429` 在 `main()` 的第一行
以同样方式调用：

```
scripts/play.py:48    from robotdog.tools import console, paths  # noqa: E402
scripts/play.py:429   console.setup_utf8_console()               # main() 的第一条语句
```

**没有任何「靠导入触发」的路径**：`import robotdog` 只执行
`robotdog/__init__.py:14`（三个工具模块）与 `:15`（导入本函数），不调用它。
`robotdog/__init__.py` 的模块 docstring（`:6-9`）把这一点写成了显式警告。

本模块内部无进一步调用链（只调 stdlib 的 `getattr` 与流的 `reconfigure`）。

## 6. 关键常量与默认值

| 常量 / 字面量 | 值 | 说明 |
| --- | --- | --- |
| 被处理的流名元组 | `("stdout", "stderr")` | 顺序固定，stdout 先 |
| `encoding` | `"utf-8"` | 强制值，不做条件判断 |
| `errors` | `"replace"` | 不可编码字符写为替代字符，不抛异常 |
| 捕获的异常类型 | `(ValueError, OSError)` | 两个，仅此两个 |
| 数值常量 / 阈值 / 默认参数 | **无** | 本模块没有任何数值参数 |

## 7. 与其它模块的关系

**上游**：仅 stdlib `sys`（以及 `from __future__ import annotations`）。**不依赖 numpy、不依赖 mujoco**，是整个工具层里最轻的模块 —— 这也是它必须能被任何脚本安全导入的原因。

**下游**：

| 位置 | 用法 | 方向理由 |
| --- | --- | --- |
| `robotdog/__init__.py:15` | **只导入，不调用** | 供 `from robotdog import setup_utf8_console` 使用；导入本身不产生副作用，所以 `import robotdog` 不会改进程 I/O |
| `scripts/train.py:54, 235` | 显式导入模块并在 `main()` 首行调用 | 独立脚本可能不经 `import robotdog` 启动，所以自己兜一次（幂等） |
| `scripts/play.py:48, 429` | 同上 | 同上 |

**为什么方向是「工具层被所有人依赖」**：编码修复是最底层的横切关注点，它不需要任何项目知识，因此可以放在依赖图的最底端被所有层调用；反过来它绝不 import 任何项目模块，否则就从最底端变成环的一环。

**为什么它不放进 `robotdog/tools/paths.py` 之类**：路径常量是**纯数据**，导入它们不应改变进程 I/O 状态；把两者混在一起会让「只想查一个目录」的代码产生全局副作用。

## 8. 实测与经验教训

* 后台 run 的 stderr 丢失、以及把输出重定向到文件时的编码现象，见 `experience.md` §5.6。
* Windows 下 `>` 重定向会写出 UTF-16、正确姿势是 `Out-File -Encoding utf8` 一类小坑，见 `experience.md` §5.8。
* 训练命令如何组织、日志落在哪里，见 `train.md` §1（本文档不重复其结论）。

## 9. 已知限制与坑

* **只对当前流对象生效**。已被别人缓存的旧 `TextIOWrapper` 引用不受影响；被替换掉的新流对象也不受约束。若某处做了 `sys.stdout = MyStream()`，需要再调一次 `setup_utf8_console()`。
* **不会让终端本身的代码页变 UTF-8**。如果 Windows 控制台活动代码页是 936，UTF-8 字节仍可能显示为乱码 —— 本函数解决的是「重定向到文件/管道后被读成乱码」，不是「控制台字体/代码页」。控制台侧要看 `chcp 65001`。
* **`errors="replace"` 会静默丢字**：无法编码的字符变成替代字符，日志里不会出现任何警告。想严格报错必须绕过本函数。
* **覆盖了调用方的显式设置**。若某脚本刻意把 `PYTHONIOENCODING` 设为别的编码，只要它（或它导入的 `robotdog` 脚本）调用了本函数，就会被改成 UTF-8。注意**光 `import robotdog` 不会**触发——本函数只被脚本显式调用。
* **不监控 `sys.stdout` 的后续替换**，也没有 `atexit` 钩子做还原。
* **`reconfigure` 内部抛出的 `AttributeError` / `TypeError` 不会被捕获**（只兜 `ValueError, OSError`），会直接冒泡。
* **模块名与文件路径一致**：本模块的规范导入路径是 `robotdog.tools.console`，仓里**没有**同名的顶层 `robotdog/console.py`，所以 traceback 里出现的模块名就是真实文件位置。

## 10. 示例用法

```python
# 1) 显式调用（独立工具脚本的标准开头）
from robotdog.tools.console import setup_utf8_console

setup_utf8_console()
print("中文日志写入管道/文件后不再是乱码")

# 2) 经包入口拿到同一个函数（等价写法）
from robotdog import setup_utf8_console
from robotdog.tools.paths import RUNS_DIR

setup_utf8_console()          # 再次调用是安全的（幂等）
print(RUNS_DIR)

# 3) 输出被重定向到文件，用 UTF-8 读回来
import sys

setup_utf8_console()
sys.stdout.write("校验\n")
```
