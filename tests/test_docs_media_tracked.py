"""Every local image a document displays must be committed, or it renders as a blank box.

This exists because it happened: `.gitignore` carried a blanket `*.gif`, `git add -A`
skipped the two README GIFs without a word, and the pushed README showed an empty gap
where the rollout clip should have been. An ignore rule silently winning over an intended
add is not visible in `git status`, so it needs a test.
"""

import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = [
    "README.md",
    "CONTRIBUTING.md",
    "docs/stories.md",
    "docs/PROVENANCE.md",
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
