"""工程的规范文件系统位置。"""

from __future__ import annotations

from pathlib import Path

# .../upstair_core —— 本文件在 <workspace>/robotdog/tools/ 下，所以往上走三级。
# （它原本是 robotdog/paths.py，只走两级；下沉到 tools 子包后所有派生路径都指向了
# robotdog/ 里面，直到冒烟测试才被抓住，故留这条注释。）
WORKSPACE: Path = Path(__file__).resolve().parents[2]

MODELS_DIR: Path = WORKSPACE / "models"
GO2_DIR: Path = MODELS_DIR / "unitree_go2"
GO2_XML: Path = GO2_DIR / "go2.xml"
GO2_ASSETS_DIR: Path = GO2_DIR / "assets"

RUNS_DIR: Path = WORKSPACE / "runs"

#: 人要**看**的东西放这里。只有 ``videos/`` 是交付件（``output/videos/<名字>.mp4``），
#: 其余是工作目录，谁写谁按需创建。
OUTPUT_DIR: Path = WORKSPACE / "output"
OUTPUT_VIDEOS_DIR: Path = OUTPUT_DIR / "videos"

#: 动态生成的 MJCF 场景（不污染 vendored 的模型目录）。
GENERATED_DIR: Path = WORKSPACE / ".generated"


__all__ = [
    "GENERATED_DIR",
    "GO2_ASSETS_DIR",
    "GO2_DIR",
    "GO2_XML",
    "MODELS_DIR",
    "OUTPUT_DIR",
    "OUTPUT_VIDEOS_DIR",
    "RUNS_DIR",
    "WORKSPACE",
]
