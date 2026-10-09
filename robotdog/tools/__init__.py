"""可复用、依赖很轻的工具层。

这里的模块都是**工具**：不知道 PPO、课程或训练循环的存在，而且**都不 import**
:mod:`robotdog.env`。核心的环境 / 训练模块在上一层 :mod:`robotdog` 里，依赖方向
永远是 ``训练层 → 环境层 → 工具层``，不会成环。

* :mod:`robotdog.tools.paths`   —— 规范文件系统位置
* :mod:`robotdog.tools.console` —— 控制台 UTF-8 修复
* :mod:`robotdog.tools.terrain` —— 参数化楼梯几何 + MJCF 地形

每个模块的设计说明在 ``docs/robotdog/tools/``。
"""

from __future__ import annotations

from robotdog.tools import console, paths, terrain

__all__ = ["console", "paths", "terrain"]
