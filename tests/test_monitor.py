"""The monitor's contract, on a synthetic probe: no GPU, no policy, no checkpoint."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastvla.monitor import probe as P  # noqa: E402
from fastvla.monitor.monitor import POSE_DIMS, RuntimeMonitor  # noqa: E402


@pytest.fixture
def fake_probe(tmp_path):
    p = tmp_path / "probe.npz"
    d = 4
    np.savez(
        p,
        w=np.ones(d),
        mu=np.zeros(d),
        sd=np.ones(d),
        ybar=np.array([0.0]),
        alpha=np.array([1.0]),
        threshold=np.array([0.5]),
    )
    return str(p)


def test_off_mode_loads_nothing():
    m = RuntimeMonitor("off")
    assert not m.enabled and m.w is None


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        RuntimeMonitor("warn")


def test_score_sign_is_minus_log_clearance(fake_probe):
    m = RuntimeMonitor("log", fake_probe)
    # the probe predicts log clearance, so a smaller clearance must score HIGHER
    near = m.score(np.array([-2.0, 0, 0, 0]))
    far = m.score(np.array([2.0, 0, 0, 0]))
    assert near > far


def test_log_mode_never_holds(fake_probe):
    m = RuntimeMonitor("log", fake_probe)
    for _ in range(3):
        _s, hold = m.observe("k", np.array([-9.0, 0, 0, 0]))
        assert hold is False
    assert m.n_alarms == 3, "log still counts the alarm it does not act on"


def test_stop_latches_per_rollout_and_persists(fake_probe):
    m = RuntimeMonitor("stop", fake_probe)
    _s, hold = m.observe("a", np.array([-9.0, 0, 0, 0]))
    assert hold
    _s, hold = m.observe("a", np.array([9.0, 0, 0, 0]))  # score now far below threshold
    assert hold, "the brake holds for the rest of the rollout"
    _s, hold = m.observe("b", np.array([9.0, 0, 0, 0]))
    assert not hold, "another rollout is unaffected"


def test_hold_zeroes_pose_keeps_gripper_and_copies():
    c = np.arange(56, dtype=float).reshape(8, 7)
    h = RuntimeMonitor.hold(c)
    assert (h[:, :POSE_DIMS] == 0).all()
    assert (h[:, POSE_DIMS] == c[:, POSE_DIMS]).all()
    assert (c == np.arange(56).reshape(8, 7)).all(), "must not mutate the policy's chunk"


def test_threshold_override(fake_probe):
    assert RuntimeMonitor("log", fake_probe, threshold=2.5).threshold == 2.5


def test_ridge_recovers_a_linear_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 12))
    w = rng.normal(size=12)
    y = X @ w - 3.0
    f = P.fit_ridge(X, y, 1e-3)
    assert P.spearman(P.predict(f, X), y) > 0.999


def test_auroc_is_oriented():
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert P.auroc(s, np.array([0, 0, 1, 1], bool)) == 1.0
    assert P.auroc(s, np.array([1, 1, 0, 0], bool)) == 0.0
