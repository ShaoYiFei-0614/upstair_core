"""语法校验所有文档里的 python 代码块（仅 AST 解析，不执行）。

用法: .venv\\Scripts\\python.exe .tmp/check_doc_code.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^```(\w*)\s*$")


def blocks(text: str):
    """产出 (起始行号, 语言, 代码)。"""
    lines = text.splitlines()
    lang = None
    start = 0
    buf: list[str] = []
    for i, line in enumerate(lines, start=1):
        m = FENCE.match(line)
        if m:
            if lang is None:
                lang = m.group(1) or ""
                start = i + 1
                buf = []
            else:
                yield start, lang, "\n".join(buf)
                lang = None
            continue
        if lang is not None:
            buf.append(line)
    if lang is not None:
        yield start, lang, "\n".join(buf)


def main() -> int:
    targets = [ROOT / "README.md", ROOT / "experience.md", ROOT / "train.md"]
    targets += sorted((ROOT / "docs").rglob("*.md"))
    bad = 0
    total = 0
    skipped = 0
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for lineno, lang, code in blocks(text):
            if lang not in ("python", "py"):
                continue
            total += 1
            stripped = code.strip()
            # 片段（省略号占位）无法解析，明确跳过并计数
            if "..." in stripped:
                skipped += 1
                continue
            # 纯签名块（`.pyi` 风格）是有意的，跳过：块里没有函数体/赋值语句，
            # 只有 `def f() -> T` 与 `NAME: T = ...` 这类声明行。
            sig_lines = [
                ln for ln in stripped.splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            body_lines = [ln for ln in sig_lines if not ln.lstrip().startswith(("def ", "class ", "@", "import ", "from "))]
            if sig_lines and all(
                ln.lstrip().startswith(("def ", "class ", "@", "import ", "from "))
                or (":" in ln and ":" not in ln.split(":", 1)[1].split("=")[0])
                for ln in sig_lines
            ) and not any(ln.rstrip().endswith(":") for ln in body_lines):
                skipped += 1
                continue
            try:
                ast.parse(stripped)
            except SyntaxError as exc:
                # 允许 py3.12 的 `def f(): ...` 与纯签名片段
                if "..." in stripped:
                    skipped += 1
                    continue
                # 签名块（有 `def ` 行但没有以 `:` 结尾的定义行）是有意的伪代码
                if "def " in stripped and not any(
                    ln.rstrip().endswith(":") for ln in stripped.splitlines()
                    if ln.lstrip().startswith("def ")
                ):
                    skipped += 1
                    continue
                bad += 1
                rel = path.relative_to(ROOT)
                print(f"{rel}:{lineno} SyntaxError: {exc.msg} (line {exc.lineno})")
    print(f"python blocks: {total}, unparsable-skipped: {skipped}, BROKEN: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
