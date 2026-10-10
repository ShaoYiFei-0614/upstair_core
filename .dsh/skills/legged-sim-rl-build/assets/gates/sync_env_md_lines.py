"""幂等地把 docs/robotdog/env.md 里 `Go2StairsEnv` 成员的行号刷成 `robotdog/env.py` 的真值。

为什么不沿用「旧值 -> 新值」的映射表：那张表只能跑一次，跑第二次就会把已经改好的数字
再当成旧值改一遍（级联），本轮就因此写出过 `env.py:284-281` 这种降序区间。
本脚本改为**按符号查真值**，所以可以随便重跑：

  * 表格行形态 ``| `name()` | `:NNN` | ... |``  -> 把该单元格刷成真值；
  * 正文形态  `` `name`（`path/env.py:NNN`）``   -> 刷成真值。

用法：
  .venv\\Scripts\\python.exe .tmp\\sync_env_md_lines.py [--check]
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "robotdog" / "env.md"
SRC = ROOT / "robotdog" / "env.py"


def real_defs() -> dict[str, int]:
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    return {
        n.name: n.lineno
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def main(argv: list[str]) -> int:
    check = "--check" in argv
    defs = real_defs()
    lines = DOC.read_text(encoding="utf-8").split("\n")
    changed = 0
    for i, line in enumerate(lines):
        # 表格行：第一个反引号里的成员名（去掉 ()）
        m = re.match(r"\|\s*`([A-Za-z_][A-Za-z0-9_]*)\(?\)?`\s*\|\s*`:(\d+)`", line)
        if m and m.group(1) in defs:
            want = defs[m.group(1)]
            if int(m.group(2)) != want:
                lines[i] = line[: m.start(2)] + str(want) + line[m.end(2):]
                print(f"  L{i + 1}  {m.group(1):26s} :{m.group(2)} -> :{want}")
                changed += 1
            continue
        # 正文形态：`name`（`robotdog/env.py:NNN`
        m2 = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)`（`robotdog/env\.py:(\d+)", line)
        if m2 and m2.group(1) in defs:
            want = defs[m2.group(1)]
            if int(m2.group(2)) != want:
                lines[i] = line[: m2.start(2)] + str(want) + line[m2.end(2):]
                print(f"  L{i + 1}  {m2.group(1):26s} :{m2.group(2)} -> :{want}")
                changed += 1
    text = "\n".join(lines)
    if changed and not check:
        DOC.write_text(text, encoding="utf-8")
        print(f"\nWROTE {DOC}  共改 {changed} 处")
    elif check:
        print(f"\nCHECK-ONLY：将改 {changed} 处")
    else:
        print("\n已与源码一致，无需修改")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
