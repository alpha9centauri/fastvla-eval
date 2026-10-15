"""Parity on 1000 real saved states, asserted against the artefact.

The measurement itself needs MuJoCo, EGL and GPU 1, so it runs under
`bench/parity_run.sh` (which goes through run_with_alert.py like every other GPU job) and
writes `bench/out/m2_parity_real.json`. This test is the gate on that artefact, so the
acceptance criterion lives in the test suite rather than only in a log.
"""

import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ARTEFACT = os.path.join(HERE, "..", "bench", "out", "m2_parity_real.json")
TOL = 1e-6


@pytest.fixture(scope="module")
def parity():
    if not os.path.exists(ARTEFACT):
        pytest.skip("run bench/parity_run.sh on GPU 1 to produce m2_parity_real.json")
    with open(ARTEFACT) as fh:
        return json.load(fh)


@pytest.mark.slow
def test_enough_states_measured(parity):
    assert parity["n_clearance_calls"] >= 1000


@pytest.mark.slow
def test_max_abs_diff_under_tolerance(parity):
    assert parity["max_abs_diff_m_overall"] < TOL, (
        f"max |python - c++| = {parity['max_abs_diff_m_overall']:.3e} m"
    )


@pytest.mark.slow
def test_both_gjk_call_sites_covered(parity):
    """S3 (union-hull lower bound) and S5a (exact pairs) are separate call sites."""
    assert set(parity["sites"]) == {"s3", "s5"}
    for site in parity["sites"].values():
        assert site["n_calls"] > 0


@pytest.mark.slow
def test_no_intersection_disagreements(parity):
    """A 0.0 from one side and not the other would flip a contact verdict."""
    for name, site in parity["sites"].items():
        assert site["n_intersection_disagreements"] == 0, name
