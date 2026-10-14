#!/usr/bin/env bash
# Point this clone's git hooks at the tracked .githooks/ directory.
# Run once after cloning: tools/install-hooks.sh
set -euo pipefail
cd "$(dirname "$0")/.."
chmod +x .githooks/*
git config --local core.hooksPath .githooks
echo "hooks installed: core.hooksPath=$(git config --local core.hooksPath)"
