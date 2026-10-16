# `robotdog/tools/paths.py` — 路径常量

对应代码：`robotdog/tools/paths.py`（约 40 行，**只有常量没有函数**）。

## 职责

全仓文件系统位置的唯一定义点。所有「东西放在哪」的问题都引用这里的常量，
不允许在别处手拼相对路径。

## 常量

| 常量 | 指向 | 说明 |
| --- | --- | --- |
| `WORKSPACE` | 仓根 | 由本文件位置上溯三级推得；**层级变了这里会静默指错**，冒烟测试必须覆盖 |
| `GO2_XML` / `GO2_ASSETS_DIR` | vendored Go2 模型 | 只读，永不修改 |
| `RUNS_DIR` | `runs/` | 训练产物（gitignored，可整体重建） |
| `OUTPUT_DIR` / `OUTPUT_VIDEOS_DIR` | `output/` | 人要看的东西；`videos/` 是交付件（**不在 gitignore 里**，冒烟后要清理） |
| `GENERATED_DIR` | `.generated/` | 动态生成的 MJCF 转储（编译失败时的现场） |

## 约定

工具层不知道任何上层模块；本文件被 `robotdog/` 全体与 `scripts/` 引用，
自身只依赖标准库。
