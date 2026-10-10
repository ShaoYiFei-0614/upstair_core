"""校验文档里 `文件.py:行号` 引用是否指向真实存在的行。

用法: .venv\\Scripts\\python.exe .tmp/check_lines.py
只做「行号是否越界」与「该行是否空行」两种硬检查；不判断语义。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 匹配 `path.py:12`、`path.py:12-18`、`path.py:12, 34`、`path.py:12/34`
REF = re.compile(r"([A-Za-z0-9_./\\-]+\.py):((?:\d+)(?:[-/,]\s*\d+)*)")


def line_numbers(spec: str) -> list[int]:
    out: list[int] = []
    for part in re.split(r"[/,]", spec):
        part = part.strip()
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                out.extend((int(a), int(b.strip())))
            except ValueError:
                pass
        elif part.isdigit():
            out.append(int(part))
    return out


def resolve(raw: str) -> Path | None:
    """把文档里的引用解析成仓内文件路径。

    文档里存在大量**裸文件名**引用（``env.py:585``、``reward.py:311``）。
    以前这里只试 ``ROOT / raw``，于是那些引用永远解析不到文件、被 ``continue``
    静默跳过 —— 门的盲区正好覆盖了成员表和调用栈图。现在按「仓根 → robotdog
    → scripts → docs → 全仓唯一同名」的顺序找，找不到才算「可能已删除」。
    """
    rel = raw.replace("\\", "/")
    for cand in (ROOT / rel, ROOT / "robotdog" / rel, ROOT / "scripts" / rel,
                 ROOT / "docs" / rel):
        if cand.is_file():
            return cand
    name = Path(rel).name
    hits = [p for p in ROOT.rglob(name)
            if p.is_file() and ".venv" not in p.parts and ".git" not in p.parts
            and ".tmp" not in p.parts and "runs" not in p.parts]
    return hits[0] if len(hits) == 1 else None


def main() -> int:
    docs = [ROOT / "README.md", ROOT / "experience.md", ROOT / "train.md"]
    docs += sorted((ROOT / "docs").rglob("*.md"))
    cache: dict[Path, list[str]] = {}
    problems = 0
    unresolved = 0
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        for m in REF.finditer(text):
            raw, spec = m.group(1), m.group(2)
            cand = resolve(raw)
            if cand is None:
                unresolved += 1
                continue  # 引用的可能是已删除的文件，由 check_xrefs 管
            lines = cache.setdefault(cand, cand.read_text(encoding="utf-8").splitlines())
            for n in line_numbers(spec):
                if n < 1 or n > len(lines):
                    problems += 1
                    print(f"{doc.relative_to(ROOT)}: {raw}:{n} 越界（文件共 {len(lines)} 行）")
                elif not lines[n - 1].strip():
                    problems += 1
                    print(f"{doc.relative_to(ROOT)}: {raw}:{n} 是空行")
    print(f"stale/blank line refs: {problems}   未解析的引用（可能指向已删文件）: {unresolved}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
