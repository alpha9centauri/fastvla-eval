#!/usr/bin/env bash
# Build docs/media/monitor_stop.gif: the same start run twice, MONITOR=off beside
# MONITOR=stop, from the two demo runs' saved videos. Labels and the event captions are
# burned in at the control step they actually happen, read from the rollout records.
#
# usage: make_monitor_gif.sh <off_run> <stop_run> <key> <contact_step> <brake_step> <out.gif>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OFF=$1; STOP=$2; KEY=$3; CONTACT=$4; BRAKE=$5; OUT=$6
F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT
SPEED=1.8        # 300 control steps at 20 fps -> 8.3 s
W=380            # per panel; two panels side by side

# Colours are RRGGBB: 0xd03b3b is the palette status red, 0x0ca30c the status green.
# Getting the order backwards once rendered the contact caption blue.
pane () {  # <mp4> <title> <event text> <event frame> <event colour>
  echo "[0:v]scale=${W}:${W}:flags=lanczos,pad=${W}:$((W+64)):0:64:color=0x1a1a19,\
drawtext=fontfile=$F:text='$2':x=14:y=14:fontsize=22:fontcolor=white,\
drawtext=fontfile=$F:text='step %{eif\\:n\\:d}':x=(w-tw)-14:y=18:fontsize=17:fontcolor=0x898781,\
drawtext=fontfile=$F:text='$3':x=14:y=40:fontsize=17:fontcolor=$5:enable='gte(n\,$4)'[v]"
}

ffmpeg -y -v error -i "$OFF/videos/$KEY.mp4" -filter_complex \
  "$(pane x 'MONITOR = off' "contact at step $CONTACT" "$CONTACT" 0xd03b3b)" -map "[v]" -an $TMP/l.mp4
ffmpeg -y -v error -i "$STOP/videos/$KEY.mp4" -filter_complex \
  "$(pane x 'MONITOR = stop' "arm held from step $BRAKE" "$BRAKE" 0x0ca30c)" -map "[v]" -an $TMP/r.mp4

ffmpeg -y -v error -i $TMP/l.mp4 -i $TMP/r.mp4 -filter_complex \
  "[0:v][1:v]hstack=inputs=2,setpts=PTS/${SPEED}[v]" -map "[v]" -r 12 -an $TMP/pair.mp4
ffmpeg -y -v error -i $TMP/pair.mp4 -filter_complex \
  "[0:v]split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3" \
  -loop 0 "$OUT"
printf '[GIF] %s  %.2f MB\n' "$OUT" "$(stat -c%s "$OUT" | awk '{print $1/1048576}')"
