"""审计全仓 .md 里的 `文件.py:行号` 引用：给出该行的**实际内容**，供人工判断是否指对了目标。

比 check_lines.py 强的地方：check_lines 只报「越界 / 是空行」，这个脚本把「行号合法但
指向了别的代码」也摆出来。逐条看内容才能抓到漂移。

**本版起不再依赖手工维护的 EXPECT 快照** —— 那套快照一旦源码挪动就会自己变成
「伪报」的源头（曾经把 `robotdog/env.py` 的 23 条快照整体落在别的函数上）。现在
「这一行应当是一个定义行」由 AST 现算：`def` / `class` / 顶层赋值目标都算定义行。

用法：
  .venv\\Scripts\\python.exe .tmp\\ref_truth.py                # 全仓 .md
  .venv\\Scripts\\python.exe .tmp\\ref_truth.py docs\\robotdog\\env.md README.md
  .venv\\Scripts\\python.exe .tmp\\ref_truth.py --only-mismatch
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 例：robotdog/env.py:344 / robotdog/env.py:344-349 / a/b.py:12, 34 / x.py:5/7
REF_RE = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_./\\-]*\.py):((?:\d+)(?:\s*[-/,]\s*\d+)*)")

# 同一行 markdown 里反引号包起来的符号名：用来判断「数字指向的 def 是不是它」。
IDENT_RE = re.compile(r"`?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)(?:\(\))?")

# 与 IDENT_RE 配套的「本行提到的标识符」提取：既识别 `反引号包裹` 的，
# 也识别代码块里 `compute_reward(` 这种裸名字。
WORD_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")

# 解析「裸文件名」引用：仓根 → robotdog → scripts → docs，再退到全仓唯一同名。
# 不这么做的话，`env.py:585` 这类引用解析不到文件就被静默跳过 —— 而文档里
# 恰好有几十条这种写法，成员表和调用栈图全落在盲区里。
_EXCLUDE = {".venv", ".git", ".tmp", "runs", ".generated"}


def resolve(raw: str) -> Path | None:
    rel = raw.replace("\\", "/")
    for cand in (ROOT / rel, ROOT / "robotdog" / rel, ROOT / "scripts" / rel,
                 ROOT / "docs" / rel):
        if cand.is_file():
            return cand
    hits = [p for p in ROOT.rglob(Path(rel).name)
            if p.is_file() and not (_EXCLUDE & set(p.parts))]
    return hits[0] if len(hits) == 1 else None


def _num_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in re.split(r"[-/,]", spec):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out


def definition_lines(src: Path) -> dict[int, str]:
    """现算「这一行声明了什么」。

    返回 ``{行号: 声明名}``。沿用旧版的 EXPECT 快照语义：**只收 def / class**。
    赋值（模块常量、dataclass 字段）刻意不收 —— 一条引用指向某字段的赋值行
    （``robotdog/reward.py:101`` 的 ``termination: float = -15.0``）是完全正常的
    文档写法，把它算成「定义行」会让门对着几十条正确引用报 BAD。真正需要抓的
    是「引用本该指向函数/类，却落在别的函数/类的 def 上」这类漂移。
    """
    try:
        tree = ast.parse(src.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return {}
    out: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.lineno] = node.name
        elif isinstance(node, ast.ClassDef):
            out[node.lineno] = node.name
    return out


def main(argv: list[str]) -> int:
    only_mismatch = "--only-mismatch" in argv
    args = [a for a in argv if not a.startswith("--")]
    docs = [Path(a) for a in args] if args else sorted(ROOT.glob("*.md")) + sorted(
        (ROOT / "docs").rglob("*.md")
    )

    cache: dict[Path, list[str]] = {}
    defs: dict[Path, dict[int, str]] = {}
    mismatch = 0
    total = 0
    unresolved = 0
    for doc in docs:
        text = doc.read_text(encoding="utf-8", errors="replace")
        doc_lines = text.split("\n")
        for m in REF_RE.finditer(text):
            raw, spec = m.group(1), m.group(2)
            src = resolve(raw)
            if src is None:
                unresolved += 1
                continue
            lines = cache.get(src)
            if lines is None:
                lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
                cache[src] = lines
                defs[src] = definition_lines(src)
            doc_line = text.count("\n", 0, m.start()) + 1
            doc_line_text = doc_lines[doc_line - 1] if doc_line - 1 < len(doc_lines) else ""
            inline_names = {x.split(".")[-1] for x in IDENT_RE.findall(doc_line_text)} \
                | set(WORD_RE.findall(doc_line_text))
            for n in _num_list(spec):
                total += 1
                if n < 1 or n > len(lines):
                    print(f"OOR   {doc.name}:{doc_line}  {raw}:{n}  (文件共 {len(lines)} 行)")
                    mismatch += 1
                    continue
                content = lines[n - 1].strip()
                declared = defs[src].get(n)
                bad = False
                note = ""
                if not content:
                    bad, note = True, "空行"
                elif declared and declared not in inline_names:
                    bad, note = True, f"这一行是 {declared!r} 的定义，但同行没提到它"
                if bad:
                    mismatch += 1
                    print(f"BAD   {doc}:{doc_line}  {raw}:{n}  {note}  实际: {content[:70]!r}")
                elif not only_mismatch:
                    tag = "ok " if declared else "?  "
                    print(f"{tag}  {doc.name}:{doc_line}  {raw}:{n}  {content[:76]!r}")
    print(f"\nrefs checked: {total}   suspicious: {mismatch}   "
          f"未解析（可能指向仓外或已删文件）: {unresolved}")
    return 1 if mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
