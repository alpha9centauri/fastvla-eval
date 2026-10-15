#!/usr/bin/env bash
# Stop the Triton server started by triton_serve.sh, by pid. Never by pgrep pattern:
# a pattern like "triton_serve.sh" also matches the command line of whatever is doing
# the killing, which has twice taken out the caller instead of the server.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE=$ROOT/bench/out/triton_serve.pid
if [ -f "$PIDFILE" ]; then
  P=$(cat "$PIDFILE")
  if kill -0 "$P" 2>/dev/null; then
    pkill -TERM -P "$P" 2>/dev/null || true
    kill -TERM "$P" 2>/dev/null || true
    sleep 3
    kill -KILL "$P" 2>/dev/null || true
    echo "stopped triton_serve.sh pid $P"
  else
    echo "pid $P not running"
  fi
  rm -f "$PIDFILE"
else
  echo "no pidfile"
fi
