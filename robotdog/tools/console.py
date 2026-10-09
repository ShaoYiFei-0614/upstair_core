"""控制台编码修复。

Windows 上 Python 重定向到文件/管道时, ``sys.stdout.encoding`` 会退化为本地代码页
(cp936), 中文输出被写成 GBK 字节, 读取方按 UTF-8 解码就得到乱码。本模块把
stdout/stderr 强制改成 UTF-8, 让日志与工具输出在任何重定向下都可读。

由 ``robotdog/__init__.py`` 自动调用, 所以只要 ``import robotdog`` 就生效。
"""

from __future__ import annotations

import sys

__all__ = ["setup_utf8_console"]


def setup_utf8_console() -> None:
    """Make stdout/stderr emit UTF-8; safe to call repeatedly and on any stream."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # e.g. detached or already-closed stream; nothing we can do
            pass
