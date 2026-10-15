"""The copied watchdog must stay byte-identical to the research original.

audit/run_with_alert.py is a verbatim copy (see audit/PROVENANCE.txt). A formatter or a
well-meaning fix silently changing it would mean the audit ran under a different wrapper
than the one named in the handoff -- ruff already tried once, splitting its single-line
import. This test is the guard; the file is also in ruff's extend-exclude.
"""

import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
COPY = os.path.join(HERE, "..", "audit", "run_with_alert.py")
SOURCE = "/data_new/vchandr2/vla-safety-interp/src/displacement/run_with_alert.py"
HEADER_LINES = 12
SOURCE_MD5 = "3d17e208520dfb7d70a02b8569e3ee86"


def test_source_unchanged():
    """If this fails the research file itself moved; re-copy deliberately, do not edit."""
    if not os.path.exists(SOURCE):
        import pytest

        pytest.skip("research tree not present")
    with open(SOURCE, "rb") as fh:
        assert hashlib.md5(fh.read()).hexdigest() == SOURCE_MD5


def test_copy_body_matches_source():
    if not os.path.exists(SOURCE):
        import pytest

        pytest.skip("research tree not present")
    with open(COPY, "rb") as fh:
        body = b"".join(fh.readlines()[HEADER_LINES:])
    with open(SOURCE, "rb") as fh:
        assert body == fh.read()
