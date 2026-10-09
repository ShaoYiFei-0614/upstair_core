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
