"""The parts of the dataviz validator that apply to a one-series chart, in Python.

The skill ships scripts/validate_palette.js, but node here is v10.19 -- too old for its
ESM and for `??=` -- and there is no newer node on the box. Rather than assert the colors
pass, this is a faithful port of the two checks that have any content for a single-series
figure: WCAG contrast against the chart surface, and OKLab Delta E (unsimulated and under
the Machado 2009 protan/deutan transforms) between the series color and the status color
that shares the plot. Constants and conversions copied from the validator so the numbers
are comparable to it; adjacency and chroma checks are omitted because one series has no
adjacent pair.
"""

import itertools
import math

CVD_TARGET, CVD_FLOOR, NORMAL_FLOOR, CONTRAST_MIN = 8.0, 6.0, 15.0, 3.0
MACHADO = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
}


def _lin(h):
    h = h.strip().lstrip("#")
    return [
        ((c / 255) / 12.92 if (c / 255) <= 0.04045 else (((c / 255) + 0.055) / 1.055) ** 2.4)
        for c in (int(h[i : i + 2], 16) for i in (0, 2, 4))
    ]


def contrast(a, b):
    def lum(h):
        r, g, bl = _lin(h)
        return 0.2126 * r + 0.7152 * g + 0.0722 * bl

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _oklab(rgb):
    r, g, b = rgb
    ll = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * ll + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * ll - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * ll + 0.7827717662 * m - 0.8086757660 * s,
    )


def _sim(h, kind):
    r, g, b = _lin(h)
    return [min(1.0, max(0.0, row[0] * r + row[1] * g + row[2] * b)) for row in MACHADO[kind]]


def delta_e(h1, h2, kind=None):
    a = _oklab(_sim(h1, kind) if kind else _lin(h1))
    b = _oklab(_sim(h2, kind) if kind else _lin(h2))
    return 100 * math.dist(a, b)


def report(colors, surface, label=""):
    ok = True
    print(f"-- {label} on {surface}")
    for name, hexv in colors.items():
        c = contrast(hexv, surface)
        state = "pass" if c >= CONTRAST_MIN else "WARN"
        if c < CONTRAST_MIN:
            ok = False
        print(f"   contrast {name:<10} {hexv} {c:5.2f}:1  {state}")
    for (n1, c1), (n2, c2) in itertools.combinations(colors.items(), 2):
        nd = delta_e(c1, c2)
        cvd = min(delta_e(c1, c2, "protan"), delta_e(c1, c2, "deutan"))
        cs = "pass" if cvd >= CVD_TARGET else ("floor" if cvd >= CVD_FLOOR else "fail")
        ns = "pass" if nd >= NORMAL_FLOOR else "fail"
        print(f"   dE {n1}/{n2}: normal {nd:5.1f} ({ns}), min(protan,deutan) {cvd:5.1f} ({cs})")
    return ok


if __name__ == "__main__":
    # The figure's ink: one series (categorical slot 1) plus the fixed status critical for
    # the contact marker. Both are the reference palette's own hexes, unmodified.
    report(
        {"series-1": "#2a78d6", "critical": "#d03b3b", "secondary": "#52514e", "muted": "#898781"},
        "#fcfcfb",
        "fastvla monitor figure, light",
    )
