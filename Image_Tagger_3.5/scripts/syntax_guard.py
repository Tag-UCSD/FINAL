"""Syntax Guard

Enterprise GO gate. Ensures that all live Python code in critical folders
parses via AST, and that no embedded truncation placeholders made it into
the tree (e.g., 'np.fl...' from truncated edits).

Critical folders:
- backend/
- scripts/
- tests/

Archive/ and other excluded directories are ignored.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
CRITICAL_DIRS = ("backend", "scripts", "tests")
EXCLUDE_DIR_PARTS = {"archive", "__pycache__", ".venv", "node_modules", ".git"}

# Detect embedded truncations that AST might not catch (e.g., np.fl...)
TRUNCATION_REGEXES = [
    re.compile(r"\bnp\.[A-Za-z_]+\.\.\.\b"),   # np.fl...
    re.compile(r"\b\w+\.\w+\.\.\.\b"),      # any dotted token ending with ...
    re.compile(r"astype\(\s*np\.[A-Za-z_]*\.\.\."),  # astype(np.fl...
]


def iter_py_files() -> Iterable[Path]:
    for d in CRITICAL_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            if any(part in EXCLUDE_DIR_PARTS for part in p.parts):
                continue
            yield p


def scan_truncations(text: str) -> List[Tuple[int, str]]:
    bad: List[Tuple[int, str]] = []
    for ln, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Only scan code-ish lines: ignore obvious docstring-only lines
        if stripped.startswith(('"""', "'''")):
            continue
        for rx in TRUNCATION_REGEXES:
            if rx.search(line):
                bad.append((ln, stripped))
                break
    return bad


def main() -> None:
    syntax_errors = []
    trunc_errors = []

    for p in iter_py_files():
        text = p.read_text(encoding="utf-8")
        try:
            ast.parse(text, filename=str(p))
        except SyntaxError as e:
            rel = p.relative_to(ROOT)
            syntax_errors.append((str(rel), e.lineno or 0, e.msg))

        bad = []
        if p.name != "syntax_guard.py":
            bad = scan_truncations(text)
        if bad:
            rel = p.relative_to(ROOT)
            for ln, snippet in bad:
                trunc_errors.append((str(rel), ln, snippet))

    if syntax_errors or trunc_errors:
        print("[syntax_guard] NO-GO", file=sys.stderr)
        if syntax_errors:
            print("  Syntax errors:", file=sys.stderr)
            for rel, ln, msg in syntax_errors:
                print(f"    {rel}:{ln}: {msg}", file=sys.stderr)
        if trunc_errors:
            print("  Truncation placeholders:", file=sys.stderr)
            for rel, ln, snippet in trunc_errors:
                print(f"    {rel}:{ln}: {snippet}", file=sys.stderr)
        raise SystemExit(1)

    print("[syntax_guard] OK: all critical python files parse cleanly and contain no truncations.")


if __name__ == "__main__":
    main()
