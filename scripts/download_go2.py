"""从 google-deepmind/mujoco_menagerie 下载 Unitree Go2 的 MJCF + 网格。

写入  models/unitree_go2/ ，产出一个自包含的模型目录：

    models/unitree_go2/
        go2.xml
        scene.xml
        go2.png
        LICENSE
        assets/*.obj

每个文件都走 HTTPS 拉取，带重试与镜像回退，再对着 GitHub API 报出的 size 校验。

运行:
    python scripts/download_go2.py
    python scripts/download_go2.py --force     # 重新下载全部
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
DEST = WS / "models" / "unitree_go2"

REPO = "google-deepmind/mujoco_menagerie"
REF = "main"
SRC_DIR = "unitree_go2"

# 镜像按顺序尝试。{path} 是文件在分支上的仓内相对路径。
MIRRORS = [
    "https://raw.githubusercontent.com/{repo}/{ref}/{path}",
    "https://gh-proxy.com/https://raw.githubusercontent.com/{repo}/{ref}/{path}",
    "https://cdn.jsdelivr.net/gh/{repo}@{ref}/{path}",
]

# 需要的（仓内相对）目录；go2.png 被 scene.xml 引用作为头部纹理，
# *_mjx.png 那几张预览图不需要。
WANT_FILES = [f"{SRC_DIR}/go2.xml", f"{SRC_DIR}/scene.xml", f"{SRC_DIR}/LICENSE", f"{SRC_DIR}/go2.png"]
WANT_DIRS = [f"{SRC_DIR}/assets"]

API = f"https://api.github.com/repos/{REPO}/contents"


def api_ls(path: str) -> list[dict]:
    url = f"{API}/{path}?ref={REF}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def download(path: str, size: int, use_mirror_from: int = 0) -> bytes:
    last: Exception | None = None
    order = MIRRORS[use_mirror_from:] + MIRRORS[:use_mirror_from]
    for tmpl in order:
        url = tmpl.format(repo=REPO, ref=REF, path=path)
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    blob = resp.read()
                if size and len(blob) != size:
                    raise IOError(f"size mismatch: got {len(blob)} want {size}")
                return blob
            except Exception as exc:  # noqa: BLE001 - 跨镜像重试
                last = exc
                time.sleep(0.7 * (attempt + 1))
    raise RuntimeError(f"all mirrors failed for {path}: {last}")


def collect_manifest() -> list[tuple[str, str, int]]:
    """返回 [(repo_path, local_relpath, size)]。"""
    manifest: list[tuple[str, str, int]] = []
    for path in WANT_FILES:
        name = path.split("/")[-1]
        manifest.append((path, name, 0))
    for directory in WANT_DIRS:
        for entry in api_ls(directory):
            if entry["type"] != "file":
                continue
            rel = f"{directory.split('/')[-1]}/{entry['name']}"
            manifest.append((entry["path"], rel, entry.get("size", 0)))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    print(f"destination: {DEST}\n")

    try:
        manifest = collect_manifest()
    except Exception as exc:
        print(f"ERROR: could not list remote repository: {type(exc).__name__}: {exc}")
        return 1

    print(f"{len(manifest)} files to consider\n")

    fetched = skipped = failed = 0
    total_bytes = 0

    for repo_path, rel, size in manifest:
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists() and not args.force and target.stat().st_size > 0:
            if size == 0 or target.stat().st_size == size:
                print(f"  skip   {rel}  ({target.stat().st_size:,} bytes, present)")
                skipped += 1
                total_bytes += target.stat().st_size
                continue

        try:
            blob = download(repo_path, size)
        except Exception as exc:
            print(f"  FAIL   {rel}: {exc}")
            failed += 1
            continue

        target.write_bytes(blob)
        total_bytes += len(blob)
        fetched += 1
        print(f"  got    {rel}  ({len(blob):,} bytes)")

    print(f"\nfetched={fetched} skipped={skipped} failed={failed}  total={total_bytes / 2**20:.1f} MiB")

    if failed:
        print("\nRESULT: INCOMPLETE")
        return 1

    # 健全性检查：MJCF 与它的网格目录必须都存在。
    xml = DEST / "go2.xml"
    assets = DEST / "assets"
    if not xml.exists() or not any(assets.glob("*.obj")):
        print("\nRESULT: INCOMPLETE - go2.xml or assets missing")
        return 1

    print("\nRESULT: OK - Unitree Go2 model downloaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
