.PHONY: help setup check lint typecheck test contracts format clean \
	    data sft dpo eval export serve demo

# Default target — prints the menu.
help:
	@echo ""
	@echo "  AlignForge -- available targets"
	@echo "  ------------------------------------------------"
	@echo "  Gate (must pass before every push):"
	@echo "    make check      -- lint + typecheck + test + contracts"
	@echo ""
	@echo "  Dev:"
	@echo "    make setup      -- create venv and install editable (CPU extras)"
	@echo "    make format     -- auto-format in place"
	@echo "    make clean      -- remove build/cache artefacts"
	@echo ""
	@echo "  Workflow (added as parts land):"
	@echo "    make data       -- build the default dataset artifact"
	@echo "    make sft        -- run QLoRA SFT"
	@echo "    make dpo        -- run DPO alignment"
	@echo "    make eval       -- full evaluation pass"
	@echo "    make export     -- merge + GGUF + Ollama"
	@echo "    make serve      -- start the API + UI"
	@echo "    make demo       -- one-command demo from a clean clone"
	@echo ""

# ── Gate ──────────────────────────────────────────────────────────────────
check: lint typecheck test contracts

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest tests/ -m "not gpu and not slow and not integration" --tb=short

contracts:
	uv run lint-imports

# ── Dev ───────────────────────────────────────────────────────────────────
setup:
	uv venv --python 3.11
	uv pip install -e ".[dev,serve,ui]"
	uv run pre-commit install
	@echo "Done. Activate with:  source .venv/bin/activate"

format:
	uv run ruff check --fix .
	uv run ruff format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
	rm -rf src/*.egg-info

# ── Workflow (stubbed; filled by later parts) ─────────────────────────────
data:
	alignforge data build

sft:
	alignforge train sft

dpo:
	alignforge train dpo

eval:
	alignforge eval all

export:
	@echo "See Part 09."

serve:
	@echo "See Part 10."

demo:
	@echo "See Part 10 — one-command demo not available until the serve layer is built."

# Coverage targets.
coverage:
    uv run pytest tests/ -m "not gpu and not slow and not integration" \
      --cov=alignforge --cov-report=html --cov-report=term-missing -q
    @echo "HTML report: htmlcov/index.html"

# Run the slow/integration suite (requires network, no GPU).
test-integration:
    uv run pytest tests/ -m "integration" --tb=short -v

# Run everything including slow tests (local only, not CI).
test-all:
    uv run pytest tests/ --tb=short -q

# Full gate — includes integration tests (requires network, no GPU).
check-full: check test-integration

# Print a test count summary.
test-summary:
	@uv run pytest tests/ --co -q --no-header 2>/dev/null | tail -3

# Run only tests for one layer.
test-core:
	uv run pytest tests/ -k "config or registry or hardware or logging or seed" -v

test-serve:
	uv run pytest tests/ -k "serve or api or middleware or engine" -v

test-eval:
	uv run pytest tests/ -k "eval or judge or metrics" -v
