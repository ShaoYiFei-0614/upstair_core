"""Check that every `<doc>.md` §X.Y cross-reference resolves to a real heading."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ["README.md", "experience.md", "train.md"]

docs = {f: (ROOT / f).read_text(encoding="utf-8") for f in DOCS}
heads = {}
for f, t in docs.items():
    heads[f] = [m.group(1).strip() for m in re.finditer(r"^#{2,3} (.+)$", t, re.M)]

pat = re.compile(r"(README\.md|experience\.md|train\.md)[`\s]*§([0-9]+(?:\.[0-9]+)?)")

def check(where: str, text: str, broken: list) -> None:
    for m in pat.finditer(text):
        target, sec = m.group(1), m.group(2)
        if not any(h.startswith(sec) for h in heads[target]):
            broken.append((where, m.group(0)))

broken: list = []
for f, t in docs.items():
    check(f, t, broken)

for p in ROOT.rglob("*.py"):
    s = str(p)
    if ".venv" in s or ".tmp" in s:
        continue
    check(str(p.relative_to(ROOT)).replace("\\", "/"), p.read_text(encoding="utf-8"), broken)

print("BROKEN CROSS-REFS:", broken if broken else "none")
