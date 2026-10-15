#!/usr/bin/env bash
# One sweep point: start a dmon sampler, run the manifest, stop the sampler.
# usage: sweep.sh <manifest> <N> <B> <D> <tag>
set -euo pipefail
ROOT=/data_new/vchandr2/fastvla-eval
A=$ROOT/audit
MAN=$1; N=$2; B=$3; D=$4; TAG=$5
mkdir -p "$A/logs"
nvidia-smi dmon -s um -d 1 -o T > "$A/logs/M3_${TAG}_dmon.txt" &
DMON=$!
trap 'kill $DMON 2>/dev/null || true' EXIT
RUN_ID="$TAG" bash "$ROOT/bench/m3a_run.sh" "$MAN" "$N" "$B" "$D"
