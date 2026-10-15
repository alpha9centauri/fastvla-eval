"""Make the project and the research reference modules importable from tests.

`convex_distance` is the Python implementation the C++ port is checked against. It lives
in the research repo and is reached through the symlink at /data_new/vchandr2 (the handoff
records that both copies were replaced by symlinks to src/physics/). It needs only numpy
and scipy, so it imports without mujoco or the rollout stubs.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = "/data_new/vchandr2"

for p in (ROOT, os.path.join(ROOT, "fastvla", "labeler"), REFERENCE):
    if p not in sys.path:
        sys.path.insert(0, p)
