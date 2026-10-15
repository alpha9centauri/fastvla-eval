#!/usr/bin/env bash
# Build the fastgjk extension in place.
#
# No -ffast-math and no -march=native: both change floating-point results, and M2's
# acceptance is 1e-6 m parity against the Python reference. -march=native is revisited
# as a measured optimisation in step 4, with parity re-run.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/data_new/vchandr2/miniconda3/envs/libero-vla/bin/python}"
INC="$($PY -c 'import pybind11;print(pybind11.get_include())')"
PYINC="$($PY -c 'import sysconfig;print(sysconfig.get_paths()["include"])')"
EXT="$($PY -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')"
FLAGS="${EXTRA_CXXFLAGS:--O3}"
set -x
g++ -std=c++17 -shared -fPIC $FLAGS -fvisibility=hidden \
    -I"$INC" -I"$PYINC" -I"$HERE/src" \
    "$HERE/src/bindings.cpp" -o "$HERE/fastgjk$EXT"
set +x
echo "built: $HERE/fastgjk$EXT  (flags: $FLAGS)"
