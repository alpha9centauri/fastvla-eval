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

# FASTVLA_NO_REFERENCE=1 simulates a CI runner, which has the repository but not the
# research tree. Used to verify that the suite's skip paths actually work before relying
# on them in CI, rather than discovering there that a test errors instead of skipping.
_paths = [ROOT, os.path.join(ROOT, "fastvla", "labeler")]
if os.environ.get("FASTVLA_NO_REFERENCE", "0") != "1":
    _paths.append(REFERENCE)

for p in _paths:
    if p not in sys.path:
        sys.path.insert(0, p)
