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
GO2_SCENE_XML: Path = GO2_DIR / "scene.xml"
GO2_ASSETS_DIR: Path = GO2_DIR / "assets"

RUNS_DIR: Path = WORKSPACE / "runs"

#: 人要**看**的东西放这里。只有 ``gifs/`` 是交付件（``output/gifs/<名字>.gif``），
#: 其余是工作目录，谁写谁按需创建。
OUTPUT_DIR: Path = WORKSPACE / "output"
OUTPUT_GIFS_DIR: Path = OUTPUT_DIR / "gifs"

#: 动态生成的 MJCF 场景（不污染 vendored 的模型目录）。
GENERATED_DIR: Path = WORKSPACE / ".generated"


def ensure_dir(path: Path) -> Path:
    """``mkdir -p``，并把路径还回来（方便一行写完）。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dirs() -> None:
    """创建交付需要的 ``output/`` 与 ``output/gifs/``。"""
    for path in (OUTPUT_DIR, OUTPUT_GIFS_DIR):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "GENERATED_DIR",
    "GO2_ASSETS_DIR",
    "GO2_DIR",
    "GO2_SCENE_XML",
    "GO2_XML",
    "MODELS_DIR",
    "OUTPUT_DIR",
    "OUTPUT_GIFS_DIR",
    "RUNS_DIR",
    "WORKSPACE",
    "ensure_dir",
    "ensure_output_dirs",
]
