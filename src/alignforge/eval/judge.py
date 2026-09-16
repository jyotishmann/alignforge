"""Pairwise LLM judge with position-bias correction and structured output."""

from __future__ import annotations

import itertools
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import structlog

from alignforge.eval.rubric import DEVELOPER_ASSISTANT_RUBRIC, Rubric
from alignforge.eval.schemas import EvalCase, GeneratedResponse

log = structlog.get_logger()

# ── Prompt builder ────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert technical evaluator assessing two AI assistant responses to
    a developer question. Your task is to determine which response better serves
    a software developer asking the question.

    You MUST respond with valid JSON only. No preamble, no explanation outside the JSON.
    Do not copy or repeat the responses in your output.
""").strip()


def build_judge_prompt(
    case: EvalCase,
    response_a: GeneratedResponse,
    response_b: GeneratedResponse,
    rubric: Rubric = DEVELOPER_ASSISTANT_RUBRIC,
) -> str:
    """Build the judge user prompt for one pairwise comparison."""
    dim_scores_schema = ", ".join(f'"{d.name}": <int 1-5>' for d in rubric.dimensions)
    return textwrap.dedent(f"""
        ## Question

        {case.prompt}

        ## What this question is testing

        {case.intent}

        ## Evaluation criteria

        {rubric.to_prompt_text()}

        ## Response A

        {response_a.response}

        ## Response B

        {response_b.response}

        ## Your task

        Evaluate both responses against the criteria above.
        Respond with JSON in exactly this format:

        {{
          "verdict": "A" | "B" | "tie",
          "rationale": "<2-3 sentences explaining the key difference>",
          "dimension_scores": {{
            {dim_scores_schema}
          }},
          "confidence": "high" | "medium" | "low"
        }}

        Rules:
        - "tie" only if responses are genuinely equivalent in quality.
        - The rationale must reference specific content from the responses.
        - If Response A has a factual error, it cannot win even if better formatted.
        - Consider the intent: e.g. if the intent says "trivially short", a padded
          correct answer should score lower on Concision than a brief correct one.
    """).strip()


# ── Structured output parsing ─────────────────────────────────────────────

VALID_VERDICTS = {"A", "B", "tie"}


def parse_judge_response(raw: str) -> dict[str, Any] | None:
    """Parse the judge's JSON output. Returns None if parsing fails."""
    # Strip markdown code fences if present.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(line for line in lines if not line.startswith("```"))

    try:
        parsed = json.loads(cleaned)
        verdict = parsed.get("verdict", "")
        if verdict not in VALID_VERDICTS:
            log.warning("judge_invalid_verdict", verdict=verdict, raw=raw[:200])
            return None
        return dict(parsed)
    except json.JSONDecodeError as exc:
        log.warning("judge_json_parse_error", error=str(exc), raw=raw[:200])
        return None


def parse_with_retries(
    raw: str,
    retry_fn: Any,
    max_retries: int = 2,
) -> dict[str, Any] | None:
    """Try parsing; if it fails, call retry_fn for a simplified prompt."""
    result = parse_judge_response(raw)
    if result is not None:
        return result

    for attempt in range(max_retries):
        log.info("judge_retry", attempt=attempt + 1)
        retry_raw = retry_fn(raw, attempt)
        result = parse_judge_response(retry_raw)
        if result is not None:
            return result

    log.error("judge_parse_failed_all_retries", raw=raw[:500])
    return None


# ── Position-bias-corrected comparison ────────────────────────────────────


@dataclass
class JudgementRecord:
    """One position-bias-corrected judgement of a pair of responses."""

    case_id: str
    suite: str
    model_a: str
    model_b: str
    # Individual verdicts from each ordering.
    verdict_ab: str | None  # judge saw A then B
    verdict_ba: str | None  # judge saw B then A (A/B labels swapped)
    # Debiased verdict: winner only if consistent across orderings.
    final_verdict: str  # "model_a" | "model_b" | "tie" | "failed"
    rationale_ab: str = ""
    rationale_ba: str = ""
    position_bias_observed: bool = False  # verdicts were inconsistent
    dimension_scores_ab: dict[str, int] | None = None
    dimension_scores_ba: dict[str, int] | None = None
    rubric: str = "developer_assistant_v1"


def adjudicate_pair(
    case: EvalCase,
    resp_a: GeneratedResponse,
    resp_b: GeneratedResponse,
    call_judge_fn: Any,  # (system, user) -> str
    rubric: Rubric = DEVELOPER_ASSISTANT_RUBRIC,
) -> JudgementRecord:
    """Judge one pair with position swapping for bias correction.

    Order 1: A first, B second.
    Order 2: B first, A second (labels relabelled to avoid confusion).
    A wins the debiased comparison only if it wins in both orderings.
    """
    # Order 1: A presented as "A", B presented as "B".
    prompt_ab = build_judge_prompt(case, resp_a, resp_b, rubric)
    raw_ab = call_judge_fn(JUDGE_SYSTEM_PROMPT, prompt_ab)
    parsed_ab = parse_judge_response(raw_ab)

    verdict_ab = parsed_ab.get("verdict") if parsed_ab else None
    rationale_ab = parsed_ab.get("rationale", "") if parsed_ab else ""
    scores_ab = parsed_ab.get("dimension_scores") if parsed_ab else None

    # Order 2: B presented as "A", A presented as "B" (swapped).
    prompt_ba = build_judge_prompt(case, resp_b, resp_a, rubric)
    raw_ba = call_judge_fn(JUDGE_SYSTEM_PROMPT, prompt_ba)
    parsed_ba = parse_judge_response(raw_ba)

    # In order 2, "A" is actually model_b and "B" is actually model_a.
    # Translate verdict back to (model_a, model_b) space.
    verdict_ba_raw = parsed_ba.get("verdict") if parsed_ba else None
    if verdict_ba_raw == "A":
        verdict_ba = "B"  # "A" in swapped order = model_b in original space
    elif verdict_ba_raw == "B":
        verdict_ba = "A"  # "B" in swapped order = model_a in original space
    else:
        verdict_ba = str(verdict_ba_raw) if verdict_ba_raw is not None else "tie"

    rationale_ba = parsed_ba.get("rationale", "") if parsed_ba else ""
    scores_ba = parsed_ba.get("dimension_scores") if parsed_ba else None

    # Debiased verdict: consistent win across both orderings.
    if verdict_ab is None or verdict_ba is None:
        final = "failed"
        bias_observed = False
    elif verdict_ab == verdict_ba:
        final = verdict_ab.lower().replace("a", "model_a").replace("b", "model_b")
        bias_observed = False
    else:
        # Inconsistent — count as tie, flag bias.
        final = "tie"
        bias_observed = True

    log.info(
        "pair_judged",
        case_id=case.id,
        model_a=resp_a.model_id,
        model_b=resp_b.model_id,
        verdict_ab=verdict_ab,
        verdict_ba=verdict_ba,
        final=final,
        bias=bias_observed,
    )

    return JudgementRecord(
        case_id=case.id,
        suite=case.suite,
        model_a=resp_a.model_id,
        model_b=resp_b.model_id,
        verdict_ab=verdict_ab,
        verdict_ba=verdict_ba,
        final_verdict=final,
        rationale_ab=rationale_ab,
        rationale_ba=rationale_ba,
        position_bias_observed=bias_observed,
        dimension_scores_ab=scores_ab,
        dimension_scores_ba=scores_ba,
    )


# ── Full judging run ──────────────────────────────────────────────────────


def run_judging(
    model_ids: list[str],
    suite_names: list[str],
    responses_dir: Path,
    judgements_dir: Path,
    judge_fn: Any,
    rubric: Rubric = DEVELOPER_ASSISTANT_RUBRIC,
    evals_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """Judge all pairwise model comparisons across all suites.

    Compares every ordered pair (A, B) where A != B.
    Each case judged twice (position swap) inside adjudicate_pair.
    Checkpointed per judgement.
    """
    # from alignforge.eval.suites.loader import load_suite
    from alignforge.eval.schemas import EvalCase

    judgements_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}

    pairs = list(itertools.combinations(model_ids, 2))
    log.info("judging_start", n_models=len(model_ids), n_pairs=len(pairs), suites=suite_names)

    for model_a, model_b in pairs:
        pair_key = f"{model_a}_vs_{model_b}"
        pair_dir = judgements_dir / pair_key
        pair_dir.mkdir(parents=True, exist_ok=True)
        total_judged = 0

        for suite_name in suite_names:
            # Load responses for both models.
            responses_a = _load_responses(responses_dir, model_a, suite_name)
            responses_b = _load_responses(responses_dir, model_b, suite_name)

            # Load cases for intent field.
            cases_by_id: dict[str, Any] = {}
            if evals_dir:
                from alignforge.eval.suites.loader import load_suite as _ls

                for case in _ls(suite_name, evals_dir):
                    cases_by_id[case.id] = case

            # Align responses by case_id.
            common_ids = set(responses_a) & set(responses_b)
            if limit:
                common_ids = set(list(common_ids)[:limit])

            out_file = pair_dir / f"{suite_name}.jsonl"
            ckpt_file = pair_dir / f"{suite_name}.ckpt.json"
            completed_ids = _load_judge_checkpoint(ckpt_file)

            log.info(
                "suite_judging_start",
                pair=pair_key,
                suite=suite_name,
                total=len(common_ids),
                remaining=len(common_ids) - len(completed_ids),
            )

            with out_file.open("a") as f:
                for case_id in sorted(common_ids):
                    if case_id in completed_ids:
                        continue

                    resp_a = responses_a[case_id]
                    resp_b = responses_b[case_id]

                    raw_case = cases_by_id.get(case_id)
                    case = (
                        cast(EvalCase, raw_case)
                        if raw_case is not None
                        else EvalCase(
                            id=case_id,
                            suite=suite_name,
                            prompt=resp_a.prompt,
                            intent="No intent recorded.",
                        )
                    )

                    record = adjudicate_pair(case, resp_a, resp_b, judge_fn, rubric)
                    f.write(_judgement_to_json(record) + "\n")
                    f.flush()

                    completed_ids.add(case_id)
                    _save_judge_checkpoint(ckpt_file, completed_ids)
                    total_judged += 1

        summary[pair_key] = total_judged

    log.info("judging_complete", summary=summary)
    return summary


def _load_responses(responses_dir: Path, model_id: str, suite_name: str) -> dict[str, Any]:
    """Load responses for one model/suite into a {case_id: GeneratedResponse} dict."""
    from alignforge.eval.schemas import GeneratedResponse

    path = responses_dir / model_id / f"{suite_name}.jsonl"
    if not path.exists():
        return {}
    responses: dict[str, Any] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = GeneratedResponse.model_validate_json(line)
                if r.error is None:  # skip failed generations
                    responses[r.case_id] = r
            except Exception:
                pass
    return responses


def _load_judge_checkpoint(path: Path) -> set[str]:
    import json as _json

    if path.exists():
        try:
            return set(_json.loads(path.read_text()))
        except Exception:
            pass
    return set()


def _save_judge_checkpoint(path: Path, completed: set[str]) -> None:
    import json as _json

    path.write_text(_json.dumps(sorted(completed)))


def _judgement_to_json(record: JudgementRecord) -> str:
    import json as _json

    return _json.dumps(
        {
            "case_id": record.case_id,
            "suite": record.suite,
            "model_a": record.model_a,
            "model_b": record.model_b,
            "verdict_ab": record.verdict_ab,
            "verdict_ba": record.verdict_ba,
            "final_verdict": record.final_verdict,
            "rationale_ab": record.rationale_ab,
            "position_bias_observed": record.position_bias_observed,
            "dimension_scores_ab": record.dimension_scores_ab,
            "rubric": record.rubric,
        }
    )
