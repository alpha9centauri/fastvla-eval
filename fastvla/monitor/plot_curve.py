"""The operating curve: recall against false alarms, with the two operating points marked.

Two panels, one per test set, because the two sets have different false-alarm scales and
putting them on one axis would hide the intrusion-5 curve under the unshifted one. Two
series per panel (debounce 1 and 2) with a legend and direct labels on the marked points,
so identity never rests on colour alone. Slots 1 and 2 of the dataviz reference palette;
fastvla/monitor/check_colors.py reports their separation.

usage: plot_curve.py        # reads bench/out/m7b_probe.json
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SURFACE, INK, SECOND, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"
TITLES = {"test_unshifted": "unshifted, 60 starts", "test_intr5": "intrusion-5, 48 starts"}


def main():
    with open(os.path.join(ROOT, "bench", "out", "m7b_probe.json")) as fh:
        res = json.load(fh)
    chosen_t = res["threshold"]["target_recall"]
    cmp_t = res["threshold"]["comparison_recall90"]["target_recall"]

    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans"],
            "font.size": 9,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.5))
    for ax, name in zip(axes, ("test_unshifted", "test_intr5"), strict=True):
        sweep = res["sets"][name]["recall_sweep"]
        for db, color in ((1, S1), (2, S2)):
            xs = [r[f"debounce{db}"]["alarms_per_100_control_steps"] for r in sweep]
            ys = [r[f"debounce{db}"]["test_recall_k16"] for r in sweep]
            ax.plot(
                xs,
                ys,
                "-o",
                color=color,
                lw=2,
                ms=5,
                mec=SURFACE,
                mew=1.2,
                label=f"debounce {db}" + (" \u2014 act on one crossing" if db == 1 else ""),
                zorder=3,
            )
        # the two operating points, each on the curve it belongs to
        marked = []
        for target, db, color, text in (
            (chosen_t, 2, S2, f"chosen\n{chosen_t:.0%} recall, debounce 2"),
            (cmp_t, 1, S1, f"first pass\n{cmp_t:.0%} recall, no debounce"),
        ):
            r = next(r for r in sweep if abs(r["train_recall_target"] - target) < 1e-9)
            x = r[f"debounce{db}"]["alarms_per_100_control_steps"]
            y = r[f"debounce{db}"]["test_recall_k16"]
            ax.plot([x], [y], "o", ms=11, mfc="none", mec=color, mew=2.2, zorder=5)
            marked.append((x, y, text))
        # Placed after both points are known, so each label goes to the side with room:
        # left-hand points label to the right, right-hand ones to the left. Writing them
        # inline put "chosen" off the left edge, over the y-axis label.
        xr = max(mx for mx, _my, _mt in marked)
        for mx, my, mtext in marked:
            left = mx < 0.4 * xr
            ax.annotate(
                mtext,
                xy=(mx, my),
                xytext=(12 if left else -12, 10 if left else -12),
                textcoords="offset points",
                ha="left" if left else "right",
                va="bottom" if left else "top",
                color=INK,
                fontsize=8,
                zorder=6,
            )
        ax.set_title(TITLES[name], loc="left", color=INK, fontsize=10.5, pad=6)
        ax.set_xlabel(
            "false alarms per 100 control steps (clean rollouts)", color=SECOND, fontsize=8.5
        )
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(left=0)
        ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=SECOND)
    axes[0].set_ylabel("recall: contact within 16 control steps", color=SECOND, fontsize=8.5)
    fig.suptitle(
        "What the monitor costs to catch what it catches",
        x=0.008,
        ha="left",
        color=INK,
        fontsize=12,
        y=0.985,
    )
    fig.text(
        0.008,
        0.925,
        "each point is a threshold chosen on the training set for a recall target, "
        "then measured on held-out scenes",
        ha="left",
        color=SECOND,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = os.path.join(ROOT, "fastvla", "monitor", "operating_curve.png")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"[FIG] -> {out}")


if __name__ == "__main__":
    main()
