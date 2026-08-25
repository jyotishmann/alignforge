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
