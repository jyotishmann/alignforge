"""Load and validate evaluation suites from JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from alignforge.eval.schemas import EvalCase

log = structlog.get_logger()

# Canonical suite names and their expected minimum sizes.
SUITE_REGISTRY: dict[str, int] = {
    "domain_v1": 20,  # hand-written; minimum 20, target 50
    "mtbench_sub": 20,
    "alpacaeval_sub": 30,
}


def load_suite(
    suite_name: str,
    evals_dir: Path,
    limit: int | None = None,
) -> list[EvalCase]:
    """Load and validate one eval suite. Returns a list of EvalCase."""
    path = evals_dir / f"{suite_name}.jsonl"
    if not path.exists():
        log.warning("eval_suite_missing", suite=suite_name, path=str(path))
        return []

    cases: list[EvalCase] = []
    errors = 0
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                raw["suite"] = suite_name  # inject suite name
                case = EvalCase.model_validate(raw)
                cases.append(case)
            except Exception as exc:
                log.warning("eval_case_invalid", line=i + 1, error=str(exc))
                errors += 1

    if limit is not None:
        cases = cases[:limit]

    log.info(
        "suite_loaded",
        suite=suite_name,
        n_cases=len(cases),
        n_errors=errors,
        verifiable=sum(1 for c in cases if c.verifiable),
    )
    return cases


def load_all_suites(
    suite_names: list[str],
    evals_dir: Path,
    limit: int | None = None,
) -> dict[str, list[EvalCase]]:
    """Load multiple suites. Returns {suite_name: [EvalCase, ...]}."""
    return {name: load_suite(name, evals_dir, limit=limit) for name in suite_names}


def verify_suites_exist(suite_names: list[str], evals_dir: Path) -> list[str]:
    """Return names of suites that are missing or below minimum size."""
    problems: list[str] = []
    for name in suite_names:
        path = evals_dir / f"{name}.jsonl"
        if not path.exists():
            problems.append(f"{name}: file missing at {path}")
            continue
        cases = load_suite(name, evals_dir)
        min_size = SUITE_REGISTRY.get(name, 1)
        if len(cases) < min_size:
            problems.append(f"{name}: only {len(cases)} cases, expected >= {min_size}")
    return problems
