"""C++ GJK port vs the Python reference: analytic cases and randomised hulls.

Acceptance for M2 (plan section 6, "Labeler parity"): distances within 1e-6 m of the
Python implementation, identical nearest-pair identities, and the analytic cases from the
original validation pass.

These tests need no GPU and no MuJoCo. The companion check on 1000 real saved states is
in test_gjk_real_states.py, which asserts on an artefact produced on GPU 1.
"""
import numpy as np
import pytest

TOL = 1e-6

fastgjk = pytest.importorskip(
    "fastgjk", reason="run fastvla/labeler/build.sh to build the extension")
from convex_distance import _closest_on_simplex as py_closest  # noqa: E402
from convex_distance import gjk_distance as py_gjk  # noqa: E402


def unit_cube(centre=(0.0, 0.0, 0.0)):
    """Axis-aligned cube of side 1, i.e. half-extents 0.5, as GeomHulls builds a box."""
    c = np.asarray(centre, dtype=np.float64)
    return np.array([[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5)
                     for sz in (-0.5, 0.5)], dtype=np.float64) + c


# ---------------------------------------------------------------- analytic cases ----
@pytest.mark.parametrize("name,offset,expected", [
    ("separated 1 m along x", (2.0, 0.0, 0.0), 1.0),
    ("separated 0.25 m along x", (1.25, 0.0, 0.0), 0.25),
    ("separated 1 mm along x", (1.001, 0.0, 0.0), 0.001),
    ("face touching", (1.0, 0.0, 0.0), 0.0),
    ("edge touching", (1.0, 1.0, 0.0), 0.0),
    ("corner touching", (1.0, 1.0, 1.0), 0.0),
    ("overlapping halfway", (0.5, 0.0, 0.0), 0.0),
    ("fully coincident", (0.0, 0.0, 0.0), 0.0),
    ("separated along the diagonal", (2.0, 2.0, 2.0), np.sqrt(3.0)),
])
def test_two_unit_cubes_analytic(name, offset, expected):
    """The C++ port must match the closed-form separation, not merely match Python."""
    A, B = unit_cube(), unit_cube(offset)
    got = fastgjk.gjk_distance(A, B)
    assert got == pytest.approx(expected, abs=TOL), name
    assert got == pytest.approx(py_gjk(A, B), abs=TOL), f"{name}: disagrees with Python"


def test_touching_returns_exactly_zero():
    """0.0 is the contact signal; it must be exact, not merely small."""
    A, B = unit_cube(), unit_cube((1.0, 0.0, 0.0))
    assert fastgjk.gjk_distance(A, B) == 0.0
    assert py_gjk(A, B) == 0.0


def test_sphere_pads_subtracted():
    """distance_padded reproduces DistanceEngine.distance: GJK minus both swept radii."""
    A = np.zeros((1, 3))
    B = np.array([[1.0, 0.0, 0.0]])
    assert fastgjk.distance_padded(A, 0.2, B, 0.3) == pytest.approx(0.5, abs=TOL)
    # overlapping pads clamp at zero rather than going negative
    assert fastgjk.distance_padded(A, 0.7, B, 0.6) == 0.0


def test_degenerate_inputs():
    """Single points, duplicated vertices and collinear sets must not diverge."""
    pt = np.zeros((1, 3))
    far = np.array([[3.0, 4.0, 0.0]])
    assert fastgjk.gjk_distance(pt, far) == pytest.approx(5.0, abs=TOL)
    dup = np.tile(np.array([[1.0, 1.0, 1.0]]), (5, 1))
    assert fastgjk.gjk_distance(pt, dup) == pytest.approx(np.sqrt(3.0), abs=TOL)
    line = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    assert fastgjk.gjk_distance(line, line + [0.0, 1.0, 0.0]) == pytest.approx(1.0, abs=TOL)


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        fastgjk.gjk_distance(np.zeros((4, 2)), unit_cube())
    with pytest.raises(ValueError):
        fastgjk.gjk_distance(np.zeros((0, 3)), unit_cube())


# ---------------------------------------------------------------- randomised --------
def test_random_hulls_match_python():
    """4000 random pairs, mixed separated and overlapping, across hull sizes."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(4000):
        na, nb = int(rng.integers(1, 9)), int(rng.integers(1, 9))
        A = rng.normal(size=(na, 3))
        B = rng.normal(size=(nb, 3)) + rng.normal(scale=2.0, size=3)
        worst = max(worst, abs(py_gjk(A, B) - fastgjk.gjk_distance(A, B)))
    assert worst < TOL, f"worst |python - c++| = {worst:.3e}"


def test_random_hulls_agree_on_intersection():
    """Disagreeing about 0.0 would change a collision verdict, so check it separately."""
    rng = np.random.default_rng(1)
    for _ in range(3000):
        A = rng.normal(size=(int(rng.integers(2, 9)), 3))
        B = rng.normal(size=(int(rng.integers(2, 9)), 3)) + rng.normal(scale=0.6, size=3)
        assert (py_gjk(A, B) == 0.0) == (fastgjk.gjk_distance(A, B) == 0.0)


def test_closest_on_simplex_matches_including_subset():
    """The surviving subset drives the next iteration, so its identity must match too."""
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(3000):
        W = rng.normal(size=(int(rng.integers(1, 5)), 3))
        pv, pi = py_closest(W)
        cv, ci = fastgjk.closest_on_simplex(W)
        worst = max(worst, float(np.abs(np.asarray(pv) - np.asarray(cv)).max()))
        assert list(pi) == list(ci), f"subset differs on W={W!r}"
    assert worst < TOL, f"worst coordinate difference = {worst:.3e}"


def test_scale_invariance_small_and_large():
    """Scene geometry spans millimetres to tens of centimetres; check both ends."""
    for scale in (1e-3, 1e-2, 1.0, 10.0):
        A, B = unit_cube() * scale, unit_cube((2.0, 0, 0)) * scale
        assert fastgjk.gjk_distance(A, B) == pytest.approx(1.0 * scale, rel=1e-9)
