"""Evaluation metrics: win rate, bootstrap CIs, Bradley-Terry, length control."""

from __future__ import annotations

import contextlib
import json
import math
import random
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


# ── Win rate ──────────────────────────────────────────────────────────────


def compute_win_rate(
    judgements: list[dict[str, Any]],
    model_a: str,
    model_b: str,
) -> dict[str, float]:
    """Compute win rate for model_a vs model_b with ties as 0.5.

    win_rate = (wins_a + 0.5 * ties) / (wins_a + wins_b + ties)
    Failed judgements excluded.
    """
    wins_a = wins_b = ties = failed = 0

    for j in judgements:
        v = j.get("final_verdict", "")
        if v == "model_a":
            wins_a += 1
        elif v == "model_b":
            wins_b += 1
        elif v == "tie":
            ties += 1
        else:
            failed += 1

    n = wins_a + wins_b + ties
    if n == 0:
        return {"win_rate": 0.5, "wins": 0, "losses": 0, "ties": 0, "n": 0, "failed": failed}

    win_rate = (wins_a + 0.5 * ties) / n
    return {
        "win_rate": round(win_rate, 4),
        "wins": wins_a,
        "losses": wins_b,
        "ties": ties,
        "n": n,
        "failed": failed,
    }


def bootstrap_ci(
    judgements: list[dict[str, Any]],
    model_a: str,
    model_b: str,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for win_rate via percentile method.

    Resamples over cases (not individual verdicts) to respect case-level
    heterogeneity. Returns ci_low, ci_high, and the point estimate.
    """
    rng = random.Random(seed)
    valid = [j for j in judgements if j.get("final_verdict") in ("model_a", "model_b", "tie")]
    n = len(valid)
    if n == 0:
        return {"win_rate": 0.5, "ci_low": 0.0, "ci_high": 1.0, "n": 0}

    # Point estimate.
    point = compute_win_rate(valid, model_a, model_b)["win_rate"]

    # Bootstrap.
    boot_rates: list[float] = []
    for _ in range(n_resamples):
        sample = [rng.choice(valid) for _ in range(n)]
        wins = sum(1 for j in sample if j["final_verdict"] == "model_a")
        ties_count = sum(1 for j in sample if j["final_verdict"] == "tie")
        rate = (wins + 0.5 * ties_count) / n
        boot_rates.append(rate)

    boot_rates.sort()
    alpha = 1.0 - ci_level
    lo_idx = math.floor(alpha / 2 * n_resamples)
    hi_idx = math.ceil((1 - alpha / 2) * n_resamples) - 1
    hi_idx = min(hi_idx, n_resamples - 1)

    return {
        "win_rate": round(point, 4),
        "ci_low": round(boot_rates[lo_idx], 4),
        "ci_high": round(boot_rates[hi_idx], 4),
        "n": n,
        "ci_level": ci_level,
    }


# ── Bradley-Terry model ───────────────────────────────────────────────────


def fit_bradley_terry(
    pairwise_results: dict[str, dict[str, Any]],
    n_iter: int = 200,
    learning_rate: float = 0.1,
) -> dict[str, float]:
    """Fit a Bradley-Terry model to pairwise win counts.

    pairwise_results: {
        "model_a_vs_model_b": {"model_a": n_wins_a, "model_b": n_wins_b, "ties": n_ties}
    }
    Returns {model_id: elo_score}, centred at 1000 with std ~100.

    Algorithm: iterative MLE via gradient ascent on log-likelihood.
    The tie correction uses the Rao-Kupper extension: a tie counts as
    0.5 wins for each model.
    """
    # Collect all model names.
    models = set()
    for key in pairwise_results:
        a, b = key.split("_vs_")
        models.add(a)
        models.add(b)
    models_list = sorted(models)
    n = len(models_list)

    if n < 2:
        return dict.fromkeys(models_list, 1000.0)

    # Initialise scores at 0 (log scale).
    scores = dict.fromkeys(models_list, 0.0)

    # Build win matrix: wins[i][j] = n wins of model i over model j (ties = 0.5).
    wins: dict[str, dict[str, float]] = {m: dict.fromkeys(models_list, 0.0) for m in models_list}
    for key, result in pairwise_results.items():
        a, b = key.split("_vs_")
        wins[a][b] += result.get("wins_a", 0) + 0.5 * result.get("ties", 0)
        wins[b][a] += result.get("wins_b", 0) + 0.5 * result.get("ties", 0)

    # Iterative MLE.
    for _ in range(n_iter):
        new_scores = {}
        for m in models_list:
            numerator = sum(wins[m][o] for o in models_list if o != m)
            denominator = sum(
                (wins[m][o] + wins[o][m])
                / (math.exp(scores[m]) + math.exp(scores[o]))
                * math.exp(scores[m])
                for o in models_list
                if o != m
                if (wins[m][o] + wins[o][m]) > 0
            )
            if denominator > 0:
                new_scores[m] = math.log(numerator / denominator) if numerator > 0 else -10.0
            else:
                new_scores[m] = scores[m]
        scores = new_scores

    # Convert to Elo-like scale (mean=1000, scale factor 400/ln(10) ≈ 173.7).
    mean_s = sum(scores.values()) / len(scores)
    elo = {m: round(1000.0 + 173.7 * (scores[m] - mean_s), 1) for m in models_list}
    log.info("bradley_terry_fit", elo=elo)
    return elo


# ── Length-controlled win rate ────────────────────────────────────────────


def length_controlled_win_rate(
    judgements: list[dict[str, Any]],
    responses_a: dict[str, Any],  # {case_id: GeneratedResponse}
    responses_b: dict[str, Any],
    model_a: str,
    model_b: str,
    length_tolerance: float = 0.20,
    n_resamples: int = 10_000,
) -> dict[str, Any]:
    """Win rate after discarding cases where winner is >20% longer than loser.

    This controls for the well-documented DPO verbosity bias:
    longer responses often win judge comparisons for stylistic reasons.
    If the length-controlled rate is much lower than the raw rate, report both.
    """
    # Filter judgements where length difference is within tolerance.
    filtered = []
    for j in judgements:
        cid = j.get("case_id", "")
        resp_a = responses_a.get(cid)
        resp_b = responses_b.get(cid)
        if resp_a is None or resp_b is None:
            continue

        n_a = resp_a.n_tokens or len(resp_a.response.split())
        n_b = resp_b.n_tokens or len(resp_b.response.split())

        verdict = j.get("final_verdict", "")
        if verdict == "model_a" and n_a > n_b * (1 + length_tolerance):
            continue  # A won but was much longer — discard
        if verdict == "model_b" and n_b > n_a * (1 + length_tolerance):
            continue  # B won but was much longer — discard

        filtered.append(j)

    n_discarded = len(judgements) - len(filtered)
    log.info(
        "length_control_applied",
        n_original=len(judgements),
        n_filtered=len(filtered),
        n_discarded=n_discarded,
        tolerance_pct=int(length_tolerance * 100),
    )

    if not filtered:
        return {"win_rate": 0.5, "ci_low": 0.0, "ci_high": 1.0, "n": 0, "n_discarded": n_discarded}

    result = bootstrap_ci(filtered, model_a, model_b, n_resamples=n_resamples)
    result["n_discarded"] = n_discarded
    result["pct_discarded"] = round(100 * n_discarded / max(len(judgements), 1), 1)
    return result


# ── Verifiable exact-match ────────────────────────────────────────────────


def exact_match_score(
    judgements: list[dict[str, Any]],
    responses: dict[str, Any],  # {case_id: GeneratedResponse}
    cases: dict[str, Any],  # {case_id: EvalCase}
) -> dict[str, Any]:
    """For verifiable cases, check if the expected_answer appears in the response.

    This is a recall metric: does the response contain the key factual claim?
    Not a strict equality check — expected_answer is typically a short phrase.
    """
    n_verifiable = n_correct = 0
    failures = []

    for case_id, case in cases.items():
        if not case.verifiable or not case.expected_answer:
            continue
        resp = responses.get(case_id)
        if resp is None or resp.error:
            continue

        n_verifiable += 1
        # Case-insensitive substring check.
        expected_lower = case.expected_answer.lower().strip()
        response_lower = resp.response.lower()

        if any(phrase.strip() in response_lower for phrase in expected_lower.split("|")):
            n_correct += 1
        else:
            failures.append(
                {
                    "case_id": case_id,
                    "expected": case.expected_answer,
                    "got_preview": resp.response[:100],
                }
            )

    score = n_correct / n_verifiable if n_verifiable > 0 else 0.0
    log.info(
        "exact_match_score",
        n_verifiable=n_verifiable,
        n_correct=n_correct,
        score=round(score, 4),
    )
    return {
        "exact_match": round(score, 4),
        "n_verifiable": n_verifiable,
        "n_correct": n_correct,
        "failures": failures,
    }


# ── Position-bias rate ────────────────────────────────────────────────────


def position_bias_rate(judgements: list[dict[str, Any]]) -> float:
    """Fraction of pairs where the verdict changed when order was swapped."""
    biased = sum(1 for j in judgements if j.get("position_bias_observed", False))
    total = len(judgements)
    rate = biased / total if total > 0 else 0.0
    log.info("position_bias_rate", rate=round(rate, 4), biased=biased, total=total)
    return round(rate, 4)


# ── Aggregated metrics runner ─────────────────────────────────────────────


def compute_all_metrics(
    model_ids: list[str],
    suite_names: list[str],
    judgements_dir: Path,
    responses_dir: Path,
    evals_dir: Path,
    n_resamples: int = 10_000,
) -> dict[str, Any]:
    """Load all judgements and compute the full metrics suite.

    Returns a dict suitable for passing directly to the report renderer.
    """
    import itertools

    from alignforge.eval.suites.loader import load_suite

    all_results: dict[str, Any] = {
        "models": model_ids,
        "suites": suite_names,
        "pairwise": {},
        "elo": {},
        "position_bias_rate": {},
        "length_controlled": {},
        "exact_match": {},
        "per_suite": {},
    }

    pairwise_for_bt: dict[str, dict[str, Any]] = {}

    for model_a, model_b in itertools.combinations(model_ids, 2):
        pair_key = f"{model_a}_vs_{model_b}"
        pair_judgements: list[dict[str, Any]] = []

        for suite_name in suite_names:
            jfile = judgements_dir / pair_key / f"{suite_name}.jsonl"
            if jfile.exists():
                with jfile.open() as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            with contextlib.suppress(json.JSONDecodeError):
                                pair_judgements.append(json.loads(line))

        if not pair_judgements:
            continue

        # Win rate + CI.
        wr = bootstrap_ci(pair_judgements, model_a, model_b, n_resamples=n_resamples)
        # Position bias.
        pb_rate = position_bias_rate(pair_judgements)

        # Load responses for length control.
        responses_a = _load_response_dict(responses_dir, model_a, suite_names)
        responses_b = _load_response_dict(responses_dir, model_b, suite_names)
        lc_wr = length_controlled_win_rate(
            pair_judgements, responses_a, responses_b, model_a, model_b, n_resamples=n_resamples
        )

        all_results["pairwise"][pair_key] = wr
        all_results["position_bias_rate"][pair_key] = pb_rate
        all_results["length_controlled"][pair_key] = lc_wr
        pairwise_for_bt[pair_key] = {
            "wins_a": wr["wins"],
            "wins_b": wr["losses"],
            "ties": wr["ties"],
        }

        # Per-suite breakdown.
        for suite_name in suite_names:
            suite_js = [j for j in pair_judgements if j.get("suite") == suite_name]
            if suite_js:
                suite_wr = compute_win_rate(suite_js, model_a, model_b)
                all_results["per_suite"].setdefault(suite_name, {})[pair_key] = suite_wr

        # Exact match (for all models, across verifiable cases).
        cases_dict = {}
        for suite_name in suite_names:
            for case in load_suite(suite_name, evals_dir):
                cases_dict[case.id] = case
        for mid, resp_dict in [(model_a, responses_a), (model_b, responses_b)]:
            em = exact_match_score(pair_judgements, resp_dict, cases_dict)
            all_results["exact_match"][mid] = em

    # Bradley-Terry Elo.
    if pairwise_for_bt:
        all_results["elo"] = fit_bradley_terry(pairwise_for_bt)

    return all_results


def _load_response_dict(
    responses_dir: Path, model_id: str, suite_names: list[str]
) -> dict[str, Any]:
    from alignforge.eval.schemas import GeneratedResponse

    result: dict[str, Any] = {}
    for suite in suite_names:
        p = responses_dir / model_id / f"{suite}.jsonl"
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = GeneratedResponse.model_validate_json(line)
                    if r.error is None:
                        result[r.case_id] = r
                except Exception:
                    pass
    return result
