# `robotdog/tools/console.py` — 控制台 UTF-8 修复

对应代码：`robotdog/tools/console.py`（约 30 行）。

## 职责

Windows 上 Python 输出重定向到文件/管道时，`sys.stdout.encoding` 退化为本地代码页
（cp936），中文日志变成 GBK 字节、读取方按 UTF-8 解码即乱码。`setup_utf8_console()`
把 stdout/stderr 强制重配为 UTF-8。

## 设计约定

- **导入本包不产生副作用**：必须由脚本在 `main()` 第一行显式调用一次
  （`scripts/train.py` / `play.py` / `teleop.py` / `cmd_demo.py` 都这么做）。
- 可重复调用、对任何流安全：逐流探测 `reconfigure` 是否存在，失败静默跳过
  （流已分离/已关闭时无能为力，也不该为此崩溃）。
