"""Canonical filesystem locations for the project."""

from __future__ import annotations

from pathlib import Path

# .../upstair_core  -- this file lives in <workspace>/robotdog/tools/, so walk
# three levels up.  (It used to be robotdog/paths.py and walked two levels;
# moving it into the tools subpackage silently broke every derived path until
# the smoke test caught it, hence the explicit comment.)
WORKSPACE: Path = Path(__file__).resolve().parents[2]

MODELS_DIR: Path = WORKSPACE / "models"
GO2_DIR: Path = MODELS_DIR / "unitree_go2"
GO2_XML: Path = GO2_DIR / "go2.xml"
GO2_SCENE_XML: Path = GO2_DIR / "scene.xml"
GO2_ASSETS_DIR: Path = GO2_DIR / "assets"

CONFIGS_DIR: Path = WORKSPACE / "configs"
SCRIPTS_DIR: Path = WORKSPACE / "scripts"

ARTIFACTS_DIR: Path = WORKSPACE / "artifacts"
RUNS_DIR: Path = WORKSPACE / "runs"
LOGS_DIR: Path = WORKSPACE / "logs"

#: Everything a human is meant to *look at* ends up here.  Only
#: :data:`OUTPUT_GIFS_DIR` is a **shipped deliverable** (``output/`` holds the
#: GIFs and nothing else); the other four sub-directories are working
#: directories that the tools create on demand — see
#: ``scripts/render_checkpoint.py``.
OUTPUT_DIR: Path = WORKSPACE / "output"
OUTPUT_GIFS_DIR: Path = OUTPUT_DIR / "gifs"
#: 以下四个都不是交付物，只在对应的工具真的写东西时才被创建。
OUTPUT_STILLS_DIR: Path = OUTPUT_DIR / "stills"
OUTPUT_FRAMES_DIR: Path = OUTPUT_DIR / "frames"
OUTPUT_REPORTS_DIR: Path = OUTPUT_DIR / "reports"
OUTPUT_EXPORTS_DIR: Path = OUTPUT_DIR / "exports"

# Scratch generated MJCF scenes (kept out of the vendored model directory).
GENERATED_DIR: Path = WORKSPACE / ".generated"


def ensure_dirs() -> None:
    for path in (
        MODELS_DIR,
        CONFIGS_DIR,
        ARTIFACTS_DIR,
        RUNS_DIR,
        LOGS_DIR,
        GENERATED_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> Path:
    """``mkdir -p`` and give the path back (handy for one-liner call sites)."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dirs() -> None:
    """Create the **deliverable** part of ``output/`` (``output/`` + ``gifs/``).

    The other sub-directories (``stills/ frames/ reports/ exports/``) are not
    deliverables: whoever writes into them creates them on demand with
    :func:`ensure_dir`, so a plain GIF render never leaves empty directories
    behind.
    """
    for path in (OUTPUT_DIR, OUTPUT_GIFS_DIR):
        path.mkdir(parents=True, exist_ok=True)
