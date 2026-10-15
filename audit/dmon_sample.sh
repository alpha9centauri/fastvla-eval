#!/usr/bin/env bash
# Sample both GPUs once a second for the duration of a run. Column `sm` is the
# utilisation number the benchmark table wants; GPU 0 is sampled only to show it was
# left alone. Writes raw samples to disk so the summary can be recomputed.
# usage: dmon_sample.sh <outfile>
exec nvidia-smi dmon -s um -d 1 -o T > "$1"
