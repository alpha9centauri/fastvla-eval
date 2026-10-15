"""No tracked source file may be empty.

This exists because one was, silently, for several commits. An edit written as

    open(p, "w").write(open(p).read().replace(a, b))

truncates the file before the read, because Python evaluates `open(p, "w")` first. The
result is a zero-byte file that imports, parses and lints clean, so neither ruff nor
pytest noticed; `audit/t2_breakdown.py` stayed empty until a run produced no output.

Deliberately-empty package markers are listed in ALLOWED_EMPTY.
"""

import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ALLOWED_EMPTY = {"fastvla/__init__.py"}


def tracked_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [f for f in out.splitlines() if f]


def test_no_empty_tracked_sources():
    empty = []
    for f in tracked_files():
        if not f.endswith((".py", ".sh", ".md", ".cpp", ".hpp", ".toml")):
            continue
        if f in ALLOWED_EMPTY:
            continue
        p = os.path.join(ROOT, f)
        if os.path.isfile(p) and os.path.getsize(p) == 0:
            empty.append(f)
    assert not empty, f"empty tracked source files: {empty}"


def test_key_scripts_have_plausible_size():
    """A file that lost its body but kept a docstring would pass the emptiness check."""
    minimums = {
        "audit/t2_breakdown.py": 4000,
        "audit/t1_summarize.py": 3000,
        "fastvla/labeler/patch.py": 3000,
        "fastvla/labeler/src/gjk.hpp": 5000,
        "fastvla/labeler/src/labeler.hpp": 6000,
    }
    small = {
        f: os.path.getsize(os.path.join(ROOT, f))
        for f, m in minimums.items()
        if os.path.exists(os.path.join(ROOT, f)) and os.path.getsize(os.path.join(ROOT, f)) < m
    }
    assert not small, f"suspiciously small: {small}"
