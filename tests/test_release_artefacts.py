"""Files a clone needs must actually be in the clone.

This exists because it happened twice in one release. `.gitignore` carried a blanket
`*.gif`, `git add -A` skipped the two README GIFs without a word, and the pushed README
showed an empty gap where the rollout clip should have been. The same rule for `*.npz`
had quietly left out the fitted probe -- the file RuntimeMonitor loads by default, so a
clone could not run MONITOR at all.

An ignore rule beating an intended add is invisible in `git status`: there is nothing to
report, because nothing was staged. So it needs a test rather than care.
"""

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DOCS = [
    "README.md",
    "CONTRIBUTING.md",
    "docs/PROVENANCE.md",
    "docs/BENCHMARK.md",
    "docs/CORRECTNESS.md",
    "docs/ARCHITECTURE.md",
    "fastvla/monitor/EVAL.md",
]
MEDIA = (".png", ".gif", ".jpg", ".jpeg", ".svg", ".webp")
# markdown ![alt](path) and html <img src="path">
PAT = re.compile(r'!\[[^\]]*\]\(([^)\s]+)\)|<img[^>]*\ssrc="([^"]+)"')


def tracked_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return set(out.splitlines())


def referenced():
    """(document, image path relative to the repo root) for every local image."""
    found = []
    for doc in DOCS:
        p = os.path.join(ROOT, doc)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            text = fh.read()
        for m in PAT.finditer(text):
            src = m.group(1) or m.group(2)
            if src.startswith(("http://", "https://", "data:")):
                continue
            if not src.lower().endswith(MEDIA):
                continue
            # a relative path resolves against the document's own directory
            rel = os.path.normpath(os.path.join(os.path.dirname(doc), src))
            found.append((doc, rel))
    return found


def test_there_are_images_to_check():
    assert referenced(), "no local images found; the regex or the doc list is wrong"


def test_every_referenced_image_exists_on_disk():
    missing = [(d, r) for d, r in referenced() if not os.path.exists(os.path.join(ROOT, r))]
    assert not missing, f"documents point at images that do not exist: {missing}"


def test_every_referenced_image_is_tracked_by_git():
    t = tracked_files()
    untracked = [(d, r) for d, r in referenced() if r not in t]
    assert not untracked, (
        f"documents display images git is not tracking (a .gitignore rule is winning): {untracked}"
    )


def test_the_default_probe_is_tracked():
    """RuntimeMonitor loads this path when MONITOR is log or stop; a clone needs it."""
    from fastvla.monitor.monitor import DEFAULT_PROBE

    rel = os.path.relpath(DEFAULT_PROBE, ROOT)
    assert os.path.exists(DEFAULT_PROBE), f"{rel} is missing"
    assert rel in tracked_files(), (
        f"{rel} is not tracked -- a .gitignore rule is winning, and a clone cannot run "
        f"MONITOR=log or MONITOR=stop"
    )


def test_bench_artefacts_behind_the_table_are_tracked():
    """The table claims to be generated from these; a reader has to be able to check."""
    import re as _re

    with open(os.path.join(ROOT, "docs", "BENCHMARK.md"), encoding="utf-8") as fh:
        readme = fh.read()
    cited = set(_re.findall(r"bench/out/[A-Za-z0-9_.\-]+\.json", readme))
    assert cited, "no bench/out artefacts cited in docs/BENCHMARK.md"
    t = tracked_files()
    missing = sorted(c for c in cited if c not in t)
    assert not missing, f"the table cites artefacts that are not in the repo: {missing}"
