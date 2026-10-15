"""Render a timestamped terminal transcript into a GIF, as a faithful fast-forward.

The transcript is `<elapsed seconds>\\t<line>` per line, captured from a real run. Frame i
shows every line whose timestamp is at or before `i / fps * speed`, scrolled to the last
ROWS lines -- so the GIF is the terminal replayed quicker, with nothing added, reordered
or invented. The speed-up is printed and belongs in the caption.

Colours are the repository's chart palette on its dark surface; the status green is used
only where the line itself says PASS, so the colour never carries meaning alone.

usage: render_terminal_gif.py <transcript> <out.gif> [target_seconds] [fps] [cols]
"""

import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BG, FG = "#1a1a19", "#c3c2b7"
WHITE, BLUE, GREEN, DIM = "#ffffff", "#3987e5", "#0ca30c", "#898781"
ROWS, PAD, SIZE, LEAD = 20, 5, 10, 13


def colour(line):
    if line.startswith("$ "):
        return WHITE
    if "PASS" in line:
        return GREEN
    if line.startswith("[RUN]"):
        return BLUE
    if line.startswith("[PREFLIGHT]"):
        return FG
    if line.startswith("[WATCH]"):
        return DIM
    return FG


def main():
    src, out = sys.argv[1], sys.argv[2]
    target = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    fps = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    cols = int(sys.argv[5]) if len(sys.argv) > 5 else 78

    events = []
    with open(src) as fh:
        for raw in fh:
            if "\t" not in raw:
                continue
            t, line = raw.rstrip("\n").split("\t", 1)
            # Wrap rather than truncate: a real terminal wraps, and the preflight's
            # longest line ends in the "-> ok" that is the whole point of printing it.
            chunks = [line[i : i + cols] for i in range(0, max(len(line), 1), cols)]
            for c in chunks:
                events.append((float(t), c))
    span = events[-1][0] if events else 1.0
    speed = span / target
    font = ImageFont.truetype(FONT, SIZE)
    adv = font.getlength("M")
    w = int(2 * PAD + cols * adv)
    h = 2 * PAD + ROWS * LEAD
    print(
        f"[GIF] {len(events)} lines over {span:.1f} s -> {target:.0f} s at {fps} fps "
        f"({speed:.1f}x), {w}x{h}, {cols} cols"
    )

    n_frames = int(target * fps)
    with tempfile.TemporaryDirectory() as td:
        for i in range(n_frames):
            now = (i / fps) * speed
            shown = [ln for t, ln in events if t <= now][-ROWS:]
            img = Image.new("RGB", (w, h), BG)
            d = ImageDraw.Draw(img)
            for r, ln in enumerate(shown):
                d.text((PAD, PAD + r * LEAD), ln, font=font, fill=colour(ln))
            img.save(os.path.join(td, f"f{i:05d}.png"))
        pal = os.path.join(td, "pal.png")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                os.path.join(td, "f%05d.png"),
                "-vf",
                "palettegen=max_colors=64",
                pal,
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(td, "f%05d.png"),
                "-i",
                pal,
                "-lavfi",
                "paletteuse",
                "-loop",
                "0",
                out,
            ],
            check=True,
        )
    print(f"[GIF] -> {out} ({os.path.getsize(out) / 1048576:.2f} MB)")


if __name__ == "__main__":
    main()
