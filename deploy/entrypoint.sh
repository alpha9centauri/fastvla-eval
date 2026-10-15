#!/usr/bin/env bash
# One image, two roles. The GPU pre-flight runs first for either, exactly as it does on
# the host -- see fastvla/server/CONTAINERS.md section 8 for why nvidia-smi is not the
# check that matters.
set -euo pipefail
cd /opt/fastvla
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  python3 fastvla/server/preflight_gpu.py || {
    echo "GPU pre-flight failed; refusing to start" >&2
    exit 3
  }
fi
case "${1:-fastvla-server}" in
  fastvla-server) exec python3 -u fastvla/server/server.py ;;
  fastvla-worker) exec python3 -u fastvla/sim/worker.py ;;
  fastvla-runner) exec python3 -u fastvla/eval/runner.py ;;
  bash|sh)        exec "$@" ;;
  *)              exec "$@" ;;
esac
