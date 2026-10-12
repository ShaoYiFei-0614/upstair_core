# `scripts/download_go2.py` — 拉取 Go2 模型资源

对应代码：`scripts/download_go2.py`。

## 职责

从 `google-deepmind/mujoco_menagerie` 下载 Unitree Go2 的 MJCF 与网格到
`models/unitree_go2/`，产出自包含的模型目录。只在初次建仓或 `--force` 时需要跑。

## 设计要点

- **多镜像回退**：GitHub raw → gh-proxy → jsDelivr，逐镜像重试。
- **完整性校验**：按 GitHub API 报出的 size 逐文件核对；结束时健全性检查
  `go2.xml` 与 `assets/*.obj` 必须同时存在。
- **增量下载**：已存在且大小一致的文件跳过（`--force` 强制重下）。
- 已存在的文件不会破坏 —— `compile_scene` 只读 `go2.xml` 与网格，本脚本可以
  随时安全重跑。
