"""Audit the generated docs/ tree.

Checks per file:
  * the 10 required `## N. <title>` sections are present, in order;
  * every `experience.md` / `train.md` section number cited actually exists;
  * every `.py` under robotdog/ has a matching docs file.

Usage::

    python .tmp/audit_docs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

REQUIRED = [
    "一句话定位",
    "职责边界",
    "关键设计决策与理由",
    "公开 API",
    "内部结构与数据流",
    "关键常量与默认值",
    "与其它模块的关系",
    "实测与经验教训",
    "已知限制与坑",
    "示例用法",
]

SECTION_HEAD = re.compile(r"^##\s+(\d+)\.\s*(.+?)\s*$", re.M)
CITE = re.compile(r"`?(experience|train)\.md`?\s*(?:的)?\s*§\s*([0-9]+(?:\.[0-9]+)?)")


def real_sections(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+([0-9]+(?:\.[0-9]+)?)", line)
        if m:
            out.add(m.group(1))
    return out


def heading_numbers(path: Path, top: int) -> set[str]:
    """Collect the numbers of `#`*top-level headings, recursively (e.g. 3.4)."""
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(#{2,4})\s+([0-9]+(?:\.[0-9]+)*)", line)
        if m:
            depth = len(m.group(1)) - 1  # ## -> 1, ### -> 2
            parts = m.group(2).split(".")
            if len(parts) >= top:
                out.add(".".join(parts[:top]))
                for k in range(1, len(parts)):
                    out.add(".".join(parts[:k]))
            else:
                out.add(m.group(2))
    return out


def expected_docs() -> dict[Path, Path]:
    """docs file -> the .py it documents."""
    mapping: dict[Path, Path] = {}
    for py in sorted((ROOT / "robotdog").rglob("*.py")):
        rel = py.relative_to(ROOT / "robotdog")
        mapping[DOCS / "robotdog" / rel.with_suffix(".md")] = py
    return mapping


def main() -> int:
    problems = 0
    train_nums = heading_numbers(ROOT / "train.md", 2)
    exp_nums = heading_numbers(ROOT / "experience.md", 2)

    mapping = expected_docs()
    missing = [md for md in mapping if not md.exists()]

    print(f"== structure ({len(mapping)} expected docs) ==")
    for md, py in mapping.items():
        if not md.exists():
            print(f"MISSING  {md.relative_to(ROOT)}   (for {py.relative_to(ROOT)})")
            problems += 1
            continue
        text = md.read_text(encoding="utf-8")
        nums = []
        titles = []
        for m in SECTION_HEAD.finditer(text):
            nums.append(int(m.group(1)))
            titles.append(m.group(2))
        bad = []
        if nums != list(range(1, 11)):
            bad.append(f"sections={nums}")
        if not bad:
            for n, want, got in zip(nums, REQUIRED, titles):
                if want not in got:
                    bad.append(f"§{n} title '{got}' != '{want}'")
        if bad:
            print(f"STRUCT   {md.relative_to(ROOT)}: {'; '.join(bad)}")
            problems += 1

    print(f"\n== cross-references ({len(mapping)} files) ==")
    for md in sorted(DOCS.rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        for doc, num in CITE.findall(text):
            pool = train_nums if doc == "train" else exp_nums
            if num not in pool:
                line = next(
                    (i for i, l in enumerate(text.splitlines(), 1) if f"§{num}" in l),
                    0,
                )
                print(f"BADCITE  {md.relative_to(ROOT)}:{line}  {doc}.md §{num} does not exist")
                problems += 1

    print(f"\n== index completeness ==")
    index = (DOCS / "README.md").read_text(encoding="utf-8")
    for md in sorted(mapping):
        name = md.name
        if name not in index:
            print(f"NOT-IN-INDEX  {md.relative_to(ROOT)}")
            problems += 1
    for md in sorted(mapping):
        if not md.exists():
            print(f"NOT-WRITTEN   {md.relative_to(ROOT)}")
    if missing:
        print(f"\n{len(missing)} doc(s) still missing.")

    print(f"\nRESULT: {'ALL CHECKS PASSED' if problems == 0 and not missing else f'{problems} problem(s)'}")
    return 0 if problems == 0 and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
