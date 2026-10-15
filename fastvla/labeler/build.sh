#!/usr/bin/env bash
# Build the fastgjk extension in place.
#
# Flags: -O3 -fopenmp. Threading is available but defaults to one thread (see DESIGN.md
# step 4c); -fopenmp at one thread measured no slower than a build without it.
#
# No -ffast-math. No -march=native either: it was measured in step 4d at 4.7% faster with
# parity intact, and rejected anyway, because M5 ships this in a Docker image and a
# native build would fault or silently change results on a different host. 4.7% of a
# stage that is now ~1% of a control step is not worth a non-portable binary. Pass
# EXTRA_CXXFLAGS to override for a local experiment.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/data_new/vchandr2/miniconda3/envs/libero-vla/bin/python}"
INC="$($PY -c 'import pybind11;print(pybind11.get_include())')"
PYINC="$($PY -c 'import sysconfig;print(sysconfig.get_paths()["include"])')"
EXT="$($PY -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')"
FLAGS="${EXTRA_CXXFLAGS:--O3 -fopenmp}"
set -x
g++ -std=c++17 -shared -fPIC $FLAGS -fvisibility=hidden \
    -I"$INC" -I"$PYINC" -I"$HERE/src" \
    "$HERE/src/bindings.cpp" -o "$HERE/fastgjk$EXT"
set +x
echo "built: $HERE/fastgjk$EXT  (flags: $FLAGS)"
