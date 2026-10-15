"""The monitor figure: alarm score against time for a collision, a near miss and a clean
completion, with the contact step marked.

Small multiples -- three stacked panels, one series each -- rather than three lines on one
axis: the three rollouts have different lengths and different stories, and with one series
per panel identity never rests on color. Colors are the dataviz reference palette's own
slots (categorical 1 for the score, fixed status critical for contact); fastvla/monitor/
check_colors.py reports their contrast and CVD separation.

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
            item = (name, run_dir, r["key"], mc, r)
            if r.get("collision"):
                coll.append(item)
            elif r.get("success"):
                clean.append(item)
            else:
                near.append(item)
    # the collision with the most room to see a rise; the tightest non-contact rollout;
    # the completion that stayed furthest from the obstacle
    coll.sort(key=lambda t: -len(np.load(os.path.join(t[1], "arrays", f"{t[2]}.npz"))["clearance"]))
    near.sort(key=lambda t: t[3])
    clean.sort(key=lambda t: -t[3])
    out = []
    for label, lst in (("collision", coll), ("near miss", near), ("clean completion", clean)):
        if lst:
            out.append((label, *lst[0][:3], lst[0][3]))
    return out


def main():
    pz = np.load(os.path.join(ROOT, "bench", "out", "m7b_probe.npz"))
    fitted = dict(
        w=pz["w"], mu=pz["mu"], sd=pz["sd"], ybar=float(pz["ybar"][0]), alpha=float(pz["alpha"][0])
    )
    thr = float(pz["threshold"][0])
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
    fig, axes = plt.subplots(len(chosen), 1, figsize=(7.2, 6.4), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, (label, setname, run_dir, key, mc) in zip(axes, chosen, strict=True):
        d = rollout(run_dir, key)
        score = -P.predict(fitted, d["act"])
        ax.plot(d["steps"], score, color=SERIES, lw=2, solid_capstyle="round", zorder=3)
        ax.axhline(thr, color=SECOND, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.annotate(
            "alarm threshold",
            xy=(0.004, thr),
            xycoords=("axes fraction", "data"),
            va="bottom",
            ha="left",
            color=SECOND,
            fontsize=8,
            zorder=5,
        )
        if d["contact"].any():
            c = int(np.argmax(d["contact"]))
            ax.axvline(c, color=CRITICAL, lw=2, zorder=4)
            ax.annotate(
                f"contact, step {c}",
                xy=(c, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(-4, -2),
                textcoords="offset points",
                va="top",
                ha="right",
                color=CRITICAL,
                fontsize=8,
                fontweight="bold",
                zorder=5,
            )
            fired = np.where((score >= thr) & (d["steps"] < c))[0]
            if len(fired):
                s0 = int(d["steps"][fired[0]])
                ax.plot(
                    [s0], [score[fired[0]]], "o", ms=8, color=SERIES, mec=SURFACE, mew=2, zorder=5
                )
                ax.annotate(
                    f"first alarm, step {s0}  ({c - s0} steps of lead)",
                    xy=(s0, score[fired[0]]),
                    xytext=(6, 8),
                    textcoords="offset points",
                    color=INK,
                    fontsize=8,
                    zorder=5,
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
        ax.set_xlim(0, max(int(d["steps"].max()) + 2, 10))
    axes[-1].set_xlabel("control step", color=SECOND, fontsize=9)
    fig.suptitle(
        "Monitor alarm score over a rollout\n"
        "score = −predicted log₁₀ clearance from the policy's own activations; "
        "higher means closer to contact",
        x=0.012,
        ha="left",
        color=INK,
        fontsize=11.5,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(ROOT, "fastvla", "monitor", "alarm_score.png")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"[FIG] -> {out}")


if __name__ == "__main__":
    main()
