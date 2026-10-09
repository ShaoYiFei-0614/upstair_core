"""Go2 上下楼梯强化学习的最小核心包。

导入本包会自动把控制台切成 UTF-8（见 :func:`robotdog.tools.console.setup_utf8_console`），
否则 Windows 下打印中文 HUD / 报表会撞 ``UnicodeEncodeError``。
"""

from __future__ import annotations

from robotdog.tools import console, paths, png, terrain
from robotdog.tools.console import setup_utf8_console

__version__ = "0.2.0"

__all__ = ["console", "paths", "png", "setup_utf8_console", "terrain"]
