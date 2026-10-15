"""The monitor figure: alarm score against time for a collision, a near miss and a clean
completion, with the contact step marked.

Small multiples -- three stacked panels, one series each -- rather than three lines on one
axis: the three rollouts have different lengths and different stories, and with one series
per panel identity never rests on color. Colors are the dataviz reference palette's own
slots (categorical 1 for the score, fixed status critical for contact);
fastvla/monitor/check_colors.py reports their contrast and CVD separation.

usage: plot_monitor.py            # reads bench/out/m7b_probe.npz and the test run dirs
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastvla.monitor import probe as P  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "bench", "runs")
SURFACE, INK, SECOND, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SERIES, CRITICAL = "#e1e0d9", "#c3c2b7", "#2a78d6", "#d03b3b"


def rollout(run_dir, key):
    z = np.load(os.path.join(run_dir, "arrays", f"{key}.npz"))
    return dict(
        act=np.asarray(z["activations"], np.float32),
        steps=np.asarray(z["activation_steps"], np.int64),
        clearance=np.asarray(z["clearance"], np.float64),
        contact=np.asarray(z["contact"], bool),
    )


def _caught(run_dir, key):
    """Does the debounced alarm fire before first contact in this rollout?"""
    z = np.load(os.path.join(run_dir, "arrays", f"{key}.npz"))
    con = np.asarray(z["contact"], bool)
    if not con.any():
        return False
    c = int(np.argmax(con))
    score = -P.predict(
        dict(w=panel.w, mu=panel.mu, sd=panel.sd, ybar=panel.ybar),
        np.asarray(z["activations"], np.float32),
    )
    steps = np.asarray(z["activation_steps"], np.int64)
    return bool((P.debounce_mask(score >= panel.thr, panel.debounce) & (steps < c)).any())


def _first_contact(run_dir, key):
    con = np.asarray(np.load(os.path.join(run_dir, "arrays", f"{key}.npz"))["contact"], bool)
    return int(np.argmax(con)) if con.any() else -1


def pick(sets):
    """One collision, one near miss, one clean completion -- deterministic, by key order."""
    coll, near, clean = [], [], []
    for name, run_dir in sets:
        with open(os.path.join(run_dir, "merged.json")) as fh:
            rows = json.load(fh)["rows"]
        for r in sorted(rows, key=lambda r: r["key"]):
            p = os.path.join(run_dir, "arrays", f"{r['key']}.npz")
            if not os.path.exists(p):
                continue
            mc = float(np.min(np.asarray(np.load(p)["clearance"], float)))
            item = (name, run_dir, r["key"], mc)
            if r.get("collision"):
                coll.append(item)
            elif r.get("completion"):  # the row field is `completion`, not `success`
                clean.append(item)
            else:
                near.append(item)
    # The collision panel shows one the monitor CATCHES, latest contact first, so the
    # panel illustrates the mechanism -- the rise, the alarm, the lead. It is a chosen
    # example, not a typical one: at this operating point 11 of 29 unshifted collisions
    # raise no alarm before contact at all, which is in EVAL.md's tables and is the real
    # cost of the quieter threshold. Sorting by trace length instead of contact step
    # picked a 300-step rollout whose contact was at step 6.
    coll.sort(key=lambda t: (not _caught(t[1], t[2]), -_first_contact(t[1], t[2])))
    near.sort(key=lambda t: t[3])
    clean.sort(key=lambda t: -t[3])
    out = []
    for label, lst in (("collision", coll), ("near miss", near), ("clean completion", clean)):
        if lst:
            out.append((label, *lst[0]))
    return out


def panel(ax, label, setname, run_dir, key, mc, thr):
    """One rollout. `panel.w/mu/sd/ybar/debounce` are set by main from the probe npz."""
    d = rollout(run_dir, key)
    score = -P.predict(dict(w=panel.w, mu=panel.mu, sd=panel.sd, ybar=panel.ybar), d["act"])
    xmax = max(int(d["steps"].max()) + 2, 10)
    ax.plot(d["steps"], score, color=SERIES, lw=2, solid_capstyle="round", zorder=3)
    # The threshold label goes on the title row, outside the data area: inside the axes
    # there is no end of the trace reliably far from the threshold -- the score hovers
    # around it for most of a rollout, which is the point of the figure.
    ax.axhline(
        thr,
        color=SECOND,
        lw=1,
        ls=(0, (4, 3)),
        zorder=2,
    )
    if d["contact"].any():
        c = int(np.argmax(d["contact"]))
        left = c < 0.45 * xmax  # the label goes to whichever side has room
        ax.axvline(c, color=CRITICAL, lw=2, zorder=4)
        ax.annotate(
            f"contact, step {c}",
            xy=(c, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(5 if left else -5, -3),
            textcoords="offset points",
            va="top",
            ha="left" if left else "right",
            color=CRITICAL,
            fontsize=8,
            fontweight="bold",
            zorder=5,
        )
        # the debounced alarm, which is what the monitor acts on
        fired = np.where(P.debounce_mask(score >= thr, panel.debounce) & (d["steps"] < c))[0]
        if len(fired):
            s0 = int(d["steps"][fired[0]])
            lf = s0 < 0.5 * xmax
            lo, hi = float(score.min()), float(score.max())
            high = (score[fired[0]] - lo) > 0.6 * max(1e-9, hi - lo)
            ax.plot([s0], [score[fired[0]]], "o", ms=8, color=SERIES, mec=SURFACE, mew=2, zorder=5)
            ax.annotate(
                f"first alarm, step {s0} — {c - s0} steps of lead",
                xy=(s0, score[fired[0]]),
                xytext=(7 if lf else -7, -12 if high else 11),
                textcoords="offset points",
                ha="left" if lf else "right",
                va="top" if high else "bottom",
                color=INK,
                fontsize=8,
                zorder=6,
            )
    ax.set_title(
        f"{label} — {key} ({setname}, closest approach {mc * 100:.2f} cm)",
        loc="left",
        color=INK,
        fontsize=10,
        pad=6,
    )
    ax.set_ylabel("alarm score", color=SECOND, fontsize=8)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.set_xlim(0, xmax)
    ax.margins(y=0.18)  # headroom for the callouts


def main():
    pz = np.load(os.path.join(ROOT, "bench", "out", "m7b_probe.npz"))
    panel.w, panel.mu, panel.sd, panel.ybar = pz["w"], pz["mu"], pz["sd"], float(pz["ybar"][0])
    panel.debounce = int(pz["debounce"][0]) if "debounce" in pz.files else 1
    thr = float(pz["threshold"][0])
    panel.thr = thr
    sets = [
        ("unshifted", os.path.join(RUNS, "m7b_test_unshifted")),
        ("intrusion-5", os.path.join(RUNS, "m7b_test_intr5")),
    ]
    chosen = pick(sets)
    print(
        "[FIG] "
        + "; ".join(
            f"{lab}: {key} ({s}, min clearance {mc * 100:.2f} cm)"
            for lab, s, _rd, key, mc in chosen
        )
    )
    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans"],
            "font.size": 9,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )
    fig, axes = plt.subplots(len(chosen), 1, figsize=(7.6, 6.6))
    axes = np.atleast_1d(axes)
    for ax, (label, setname, run_dir, key, mc) in zip(axes, chosen, strict=True):
        panel(ax, label, setname, run_dir, key, mc, thr)
    axes[-1].set_xlabel("control step", color=SECOND, fontsize=9)
    fig.suptitle(
        "Monitor alarm score over a rollout", x=0.012, ha="left", color=INK, fontsize=12, y=0.995
    )
    fig.text(
        0.012,
        0.930,
        "score = −predicted log₁₀ clearance, read from the policy's own activations;\n"
        f"higher means closer.  Dashed line: the alarm threshold (debounce "
        f"{panel.debounce}).",
        ha="left",
        color=SECOND,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.918))
    out = os.path.join(ROOT, "fastvla", "monitor", "alarm_score.png")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"[FIG] -> {out}")


if __name__ == "__main__":
    main()
