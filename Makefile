# Targets are added as the thing they run comes to exist. A target with nothing
# to run says so and fails, rather than passing silently and looking green.

RUFF ?= ruff
PYTEST ?= pytest
PY ?= python3

.PHONY: help lint format test bench bench-quick bench-table bench-check container-libs

help:
	@echo "lint    - ruff check + format check"
	@echo "format  - ruff format (writes)"
	@echo "test    - pytest"
	@echo "bench         - reproduce every number in the README table (~25 min, GPU 1)"
	@echo "bench-quick   - the end-to-end rows only (~8 min)"
	@echo "bench-table   - re-render the README table from bench/out artefacts"
	@echo "bench-check   - fail if the README table is stale"
	@echo "container-libs- build the graphics libs the NVIDIA images lack"

lint:
	@command -v $(RUFF) >/dev/null 2>&1 || { \
	  echo "ruff not found. Install it in the env you lint from:"; \
	  echo "  pip install ruff"; exit 1; }
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) format .
	$(RUFF) check --fix .

test:
	@if [ ! -d tests ]; then \
	  echo "No tests/ directory yet. The parity suite (plan section 6) is the first"; \
	  echo "thing that lands here: labeler parity, then Tier A rollout parity."; \
	  exit 1; \
	fi
	$(PYTEST)

bench:
	bash bench/bench_all.sh

# The end-to-end rows only: about 8 minutes instead of 25.
bench-quick:
	BENCH_QUICK=1 bash bench/bench_all.sh

# Re-render the README table from the artefacts already in bench/out, without re-running.
bench-table:
	$(PY) bench/render_table.py

# Fail if the README table does not match the artefacts.
bench-check:
	$(PY) bench/render_table.py --check

# The graphics libraries the NVIDIA container images lack; needed for the
# in-container and Triton benchmarks only.
container-libs:
	bash bench/make_container_libs.sh
