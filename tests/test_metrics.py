"""The metrics endpoint emits valid Prometheus exposition format.

Worth testing because it is hand-rolled (no prometheus_client, see metrics.py for why) and
because a malformed histogram is the kind of thing that only shows up when a scrape
silently drops the series.
"""

import re

from fastvla.server import metrics as MET


def test_render_has_expected_series():
    m = MET.Metrics()
    m.observe_forward(1, 0.184, [0.798])
    m.observe_preprocess(0.021)
    m.set_queue_depth(3)
    m.set_workers(8)
    out = m.render()
    for name in (
        "fastvla_requests_total",
        "fastvla_forwards_total",
        "fastvla_queue_depth",
        "fastvla_workers_active",
        "fastvla_forward_latency_seconds_bucket",
        "fastvla_queue_wait_seconds_bucket",
        "fastvla_preprocess_seconds_bucket",
        "fastvla_batch_size_bucket",
    ):
        assert name in out, name
    assert "fastvla_queue_depth 3" in out
    assert "fastvla_workers_active 8" in out
    assert "fastvla_requests_total 1" in out


def test_histogram_buckets_are_cumulative_and_end_at_inf():
    m = MET.Metrics()
    for v in (0.02, 0.12, 0.19, 0.4, 7.0):
        m.observe_forward(1, v, [])
    out = m.render()
    buckets = re.findall(r'fastvla_forward_latency_seconds_bucket\{le="([^"]+)"\} (\d+)', out)
    assert buckets, "no forward-latency buckets rendered"
    counts = [int(c) for _le, c in buckets]
    assert counts == sorted(counts), f"buckets must be cumulative, got {counts}"
    assert buckets[-1][0] == "+Inf"
    assert counts[-1] == 5, "every observation must land in +Inf"
    assert "fastvla_forward_latency_seconds_count 5" in out


def test_every_type_line_precedes_its_series():
    m = MET.Metrics()
    m.observe_forward(2, 0.2, [0.1, 0.2])
    out = m.render().splitlines()
    declared = {ln.split()[2] for ln in out if ln.startswith("# TYPE")}
    for ln in out:
        if ln.startswith("#") or not ln.strip():
            continue
        base = ln.split()[0].split("{")[0]
        for suffix in ("_bucket", "_sum", "_count"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        assert base in declared, f"{base} has no # TYPE line"


def test_render_is_parseable_as_text():
    """Each non-comment line must be `name value` or `name{labels} value`."""
    m = MET.Metrics()
    m.observe_forward(1, 0.1, [0.2])
    for ln in m.render().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        assert re.match(r"^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? -?[0-9.eE+]+$", ln), ln
