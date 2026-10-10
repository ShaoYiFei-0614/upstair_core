"""把工作区里被写成 LF 的文件恢复成 CRLF（本仓约定 core.autocrlf=true）。

只处理确实含有 bare LF 的文本文件；已经全 CRLF 或二进制文件跳过。

**必须包含仓根文件**：早期版本只扫 `robotdog/`、`scripts/`、`docs/` 三个目录，
于是 `README.md`、`train.md`、`experience.md`、`agent.md`、`config.yaml`、
`requirements.txt` 这些直接在根下的文件永远扫不到 —— 它们被 `write` 工具整篇重写后
会一直保持 LF，`git status` 则一直警告 `LF will be replaced by CRLF`。

用法：
  .venv\\Scripts\\python.exe .tmp\\fix_eol.py            # 修
  .venv\\Scripts\\python.exe .tmp\\fix_eol.py --check    # 只报告，不写盘
"""
from __future__ import annotations

import pathlib
import sys

# 目录（递归）+ 仓根上的单个文件
ROOTS = ["robotdog", "scripts", "docs"]
ROOT_FILES = ["README.md", "train.md", "experience.md", "agent.md", "config.yaml",
              "requirements.txt"]
EXTS = {".py", ".md", ".yaml", ".yml", ".txt", ".json"}


def targets() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for root in ROOTS:
        for p in pathlib.Path(root).rglob("*"):
            if p.is_file() and p.suffix.lower() in EXTS:
                out.append(p)
    for name in ROOT_FILES:
        p = pathlib.Path(name)
        if p.is_file() and p.suffix.lower() in EXTS:
            out.append(p)
    return out


def main(argv: list[str]) -> int:
    check = "--check" in argv
    fixed: list[tuple[str, int]] = []
    for p in targets():
        b = p.read_bytes()
        crlf = b.count(b"\r\n")
        bare = b.count(b"\n") - crlf
        if bare == 0:
            continue
        if check:
            fixed.append((str(p), bare))
            continue
        norm = b.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        p.write_bytes(norm)
        fixed.append((str(p), bare))
    for name, n in fixed:
        print(f"{'WOULD-FIX' if check else 'FIXED'} {name}  ({n} bare LF -> CRLF)")
    print(f"total {'to fix' if check else 'fixed'}: {len(fixed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
