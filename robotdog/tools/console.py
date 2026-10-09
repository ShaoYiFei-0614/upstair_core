"""控制台编码修复。

Windows 上 Python 重定向到文件/管道时, ``sys.stdout.encoding`` 会退化为本地代码页
(cp936), 中文输出被写成 GBK 字节, 读取方按 UTF-8 解码就得到乱码。本模块把
stdout/stderr 强制改成 UTF-8, 让日志与工具输出在任何重定向下都可读。

由脚本在启动时显式调用一次（``scripts/train.py`` 与 ``scripts/play.py`` 都在
``main()`` 的第一行调用），所以不依赖 ``import robotdog`` 的副作用。
"""

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
