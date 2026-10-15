"""No tracked source file may be empty.

This exists because one was, silently, for several commits. An edit written as

    open(p, "w").write(open(p).read().replace(a, b))

truncates the file before the read, because Python evaluates `open(p, "w")` first. The
result is a zero-byte file that imports, parses and lints clean, so neither ruff nor
pytest noticed; `audit/t2_breakdown.py` stayed empty until a run produced no output.

Deliberately-empty package markers are listed in ALLOWED_EMPTY.
"""

import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Nothing is allowed to be empty: every package marker carries a one-line docstring,
# which is cheap and means an emptied file is always a real finding.
ALLOWED_EMPTY = set()


def tracked_files():
    """Tracked files per git.

    Skips only when git is genuinely unavailable -- no binary, or no repository, as in the
    GPU image which installs no git. An unexpected git FAILURE is raised, not skipped: a
    guard that swallowed any non-zero exit here would have hidden a real image bug, where
    /opt/fastvla was root-owned and `git ls-files` returned 128 with "detected dubious
    ownership". That is the sort of thing this test exists to surface.
    """
    if shutil.which("git") is None:
        pytest.skip("git binary not available (e.g. inside the GPU image)")
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        pytest.skip("no .git directory (repository not in the build context)")
    r = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise AssertionError(
            f"git ls-files failed in {ROOT} (exit {r.returncode}): {r.stderr.strip()}"
        )
    return [f for f in r.stdout.splitlines() if f]


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
