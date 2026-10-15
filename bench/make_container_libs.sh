#!/usr/bin/env bash
# Build bench/triton_libs: the graphics libraries the NVIDIA container images lack.
#
# Regenerated rather than committed (28 MB of binaries). Needed by both
# bench/triton_serve.sh and bench/incontainer_run.sh.
#
# Two rules learned the hard way in M3b:
#  * ONLY graphics libraries. Putting the conda env's whole lib/ on LD_LIBRARY_PATH
#    shadows the container's libtinfo and libstdc++, and numpy's C extension then fails
#    to load with the misleading "you should not try to import numpy from its source
#    directory".
#  * Host libraries must be COPIED, not symlinked. A symlink into
#    /usr/lib/x86_64-linux-gnu dangles inside the container, where that path is the
#    container's own.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENVROOT=${ENVROOT:-/data_new/vchandr2/miniconda3/envs/libero-vla}
HOSTLIB=${HOSTLIB:-/usr/lib/x86_64-linux-gnu}
OUT="$ROOT/bench/triton_libs"
mkdir -p "$OUT"
rm -f "$OUT"/*

# from the conda env (bound into the container, so symlinks are fine)
for pat in 'libX*' 'libxcb*' 'libGL*' 'libEGL*' 'libOSMesa*' 'libXau*' 'libXdmcp*' \
           'libdrm*' 'libgbm*' 'libglapi*' 'libwayland*'; do
  for f in "$ENVROOT"/lib/$pat; do [ -e "$f" ] && ln -sf "$f" "$OUT/"; done
done
# from the host, for what the env lacks (libXext and friends) -- copied, not linked
for pat in 'libX*' 'libxcb*' 'libXext*' 'libXrender*' 'libXrandr*' 'libXinerama*' \
           'libXcursor*' 'libXi*' 'libXfixes*' 'libXxf86vm*' 'libXt*' 'libSM*' 'libICE*'; do
  for f in "$HOSTLIB"/$pat; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    [ -e "$OUT/$b" ] && continue
    cp -L "$f" "$OUT/$b"
  done
done
echo "bench/triton_libs: $(ls "$OUT" | wc -l) libraries, $(du -sh "$OUT" | cut -f1)"
