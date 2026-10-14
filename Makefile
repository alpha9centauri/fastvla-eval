# Targets are added as the thing they run comes to exist. A target with nothing
# to run says so and fails, rather than passing silently and looking green.

RUFF ?= ruff
PYTEST ?= pytest

.PHONY: help lint format test bench

help:
	@echo "lint    - ruff check + format check"
	@echo "format  - ruff format (writes)"
	@echo "test    - pytest"
	@echo "bench   - reproduce every number in the README (not yet implemented)"

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
	@echo "Not implemented. bench/ is built in M1-M5; it must reproduce every"
	@echo "number in the README benchmark table (plan section 8) from scratch."
	@exit 1
