# neurofaune — developer gate
#
# THE GATE (blocking — must pass before a change is tagged / used in research):
#     make check   →  unit tests + regression (behavior-preservation) tests
#
# ADVISORY (informational — never blocks, never rewrites files on its own):
#     make advisory →  ruff + mypy, reported but allowed to fail
#
# See plans/main-plan.md for the IRL implementation-testing loop this gate serves.

UV  ?= uv
PKG := neurofaune

.DEFAULT_GOAL := help

.PHONY: help sync test regression check integration lint typecheck advisory format clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

sync:  ## Install package + dev deps from uv.lock (reproducible)
	$(UV) sync --extra dev

# ---------------------------------------------------------------------------
# THE GATE (blocking)
# ---------------------------------------------------------------------------
test:  ## Run fast unit tests
	$(UV) run --extra dev pytest -m "not slow and not regression and not integration" -q

regression:  ## Run fast behavior-preservation tests vs frozen golden outputs
	$(UV) run --extra dev pytest -m "regression and not slow and not integration" -q

check: test regression  ## THE GATE — tests + regression must pass before promotion (hermetic, every PR)
	@echo "✅ gate passed — candidate preserves behavior, safe to tag / pin in research"

integration:  ## SLOW tier — real-tool (ANTs/FSL) end-to-end vs derived-metric golden; run before a release tag
	$(UV) run --extra dev pytest -m "integration" -q

# ---------------------------------------------------------------------------
# ADVISORY (never blocks — leading '-' ignores non-zero exit)
# ---------------------------------------------------------------------------
lint:  ## ruff lint (advisory)
	-$(UV) run --extra dev ruff check $(PKG)

typecheck:  ## mypy type check (advisory)
	-$(UV) run --extra dev mypy $(PKG)

advisory: lint typecheck  ## Run all advisory checks (informational only)
	@echo "ℹ  advisory checks complete — these do NOT gate promotion"

# ---------------------------------------------------------------------------
# Manual maintenance (opt-in — not part of the gate)
# ---------------------------------------------------------------------------
format:  ## Apply ruff --fix + black (manual; review the diff before committing)
	$(UV) run --extra dev ruff check --fix $(PKG)
	$(UV) run --extra dev black $(PKG)

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
