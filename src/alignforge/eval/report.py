"""Evaluation report renderer: Markdown + HTML with regression gallery."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import structlog

from alignforge.eval.schemas import EvalCase

log = structlog.get_logger()


def render_report(
    metrics: dict[str, Any],
    judgements_dir: Path,
    responses_dir: Path,
    evals_dir: Path,
    output_dir: Path,
    eval_id: str,
    judge_name: str = "gpt-4o-mini",
    rubric_name: str = "developer_assistant_v1",
) -> tuple[Path, Path]:
    """Generate eval_<id>.md and eval_<id>.html. Returns (md_path, html_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"eval_{eval_id}.md"
    html_path = output_dir / f"eval_{eval_id}.html"

    md_content = _render_markdown(
        metrics, judgements_dir, responses_dir, evals_dir, eval_id, judge_name, rubric_name
    )
    html_content = _md_to_html(md_content, eval_id)

    md_path.write_text(md_content)
    html_path.write_text(html_content)
    log.info("report_written", md=str(md_path), html=str(html_path))
    return md_path, html_path


def _render_markdown(
    metrics: dict[str, Any],
    judgements_dir: Path,
    responses_dir: Path,
    evals_dir: Path,
    eval_id: str,
    judge_name: str,
    rubric_name: str,
) -> str:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    models = metrics.get("models", [])
    suites = metrics.get("suites", [])
    pairwise = metrics.get("pairwise", {})
    lc = metrics.get("length_controlled", {})
    elo = metrics.get("elo", {})
    pb = metrics.get("position_bias_rate", {})

    lines: list[str] = [
        "# AlignForge Evaluation Report",
        "",
        f"**Eval ID:** `{eval_id}`  ",
        f"**Generated:** {ts}  ",
        f"**Judge:** {judge_name}  ",
        f"**Rubric:** {rubric_name}  ",
        f"**Models evaluated:** {', '.join(f'`{m}`' for m in models)}  ",
        f"**Suites:** {', '.join(suites)}  ",
        "",
        "---",
        "",
        "## Headline Results",
        "",
        "Win rate (model_a vs model_b) with 95% bootstrap CIs, n cases per comparison.",
        "Ties counted as 0.5. Failed judgements excluded.",
        "",
        "| Comparison | Win Rate | 95% CI | n | Length-Controlled WR |",
        "|---|---|---|---|---|",
    ]

    for pair_key, wr in sorted(pairwise.items()):
        lc_wr = lc.get(pair_key, {})
        ci_str = f"[{wr.get('ci_low', '?')}, {wr.get('ci_high', '?')}]"
        lc_str = (
            f"{lc_wr.get('win_rate', '—')} [{lc_wr.get('ci_low', '?')}, {lc_wr.get('ci_high', '?')}]"
            if lc_wr
            else "—"
        )
        lines.append(
            f"| {pair_key.replace('_vs_', ' vs ')} "
            f"| **{wr.get('win_rate', '?')}** "
            f"| {ci_str} "
            f"| {wr.get('n', '?')} "
            f"| {lc_str} |"
        )

    # Elo scores.
    if elo:
        lines += [
            "\n### Bradley-Terry Elo Scores \n",
            "Single latent strength score per model (higher is better; centred at 1000).\n",
            "| Model | Elo |",
            "|---|---|",
        ]
        for model, score in sorted(elo.items(), key=lambda x: -x[1]):
            lines.append(f"| `{model}` | {score} |")

    # Position bias disclosure.
    if pb:
        lines += ["", "### Position Bias Disclosure", ""]
        lines.append(
            "Fraction of pairs where verdict changed when response order was swapped. "
            ">30% indicates the judge is unreliable for these comparisons."
        )
        lines += ["", "| Comparison | Bias Rate |", "|---|---|"]
        for pair_key, rate in sorted(pb.items()):
            flag = " ⚠️" if rate > 0.30 else ""
            lines.append(f"| {pair_key.replace('_vs_', ' vs ')} | {rate}{flag} |")

    # Per-suite breakdown.
    per_suite = metrics.get("per_suite", {})
    if per_suite:
        lines += ["", "## Per-Suite Breakdown", ""]
        for suite_name, suite_results in sorted(per_suite.items()):
            lines += [f"### {suite_name}", ""]
            lines += ["| Comparison | Win Rate | n |", "|---|---|---|"]
            for pair_key, wr in sorted(suite_results.items()):
                lines.append(
                    f"| {pair_key.replace('_vs_', ' vs ')} "
                    f"| {wr.get('win_rate', '?')} "
                    f"| {wr.get('n', '?')} |"
                )
            lines.append("")

    # Regression gallery.
    gallery = _build_regression_gallery(metrics, judgements_dir, responses_dir, evals_dir, n=10)
    if gallery:
        lines += [
            "",
            "## Regression Gallery",
            "",
            "Cases where the DPO model lost most decisively to the SFT model.",
            "These are the cases that most need investigation or additional preference data.",
            "",
        ]
        for i, entry in enumerate(gallery, 1):
            lines += [
                f"### Case {i}: `{entry['case_id']}`",
                "",
                f"**Prompt:** {entry['prompt']}",
                "",
                f"**Intent:** {entry['intent']}",
                "",
                f"**SFT response:** {entry['winner_response'][:300]}{'...' if len(entry['winner_response']) > 300 else ''}",
                "",
                f"**DPO response:** {entry['loser_response'][:300]}{'...' if len(entry['loser_response']) > 300 else ''}",
                "",
                f"**Judge rationale:** {entry['rationale']}",
                "",
                "---",
                "",
            ]

    lines += [
        "",
        "## Methodology Notes",
        "",
        "- **Position-bias correction:** every pair judged twice with order swapped. "
        "Verdict only counted when consistent across both orderings.",
        "- **Ties:** counted as 0.5 wins for each model in the win-rate calculation.",
        "- **Length control:** cases where the winning response is >20% longer than the "
        "losing response are excluded from the length-controlled win rate.",
        "- **Bootstrap CI:** 10,000 resamples over cases (not comparisons), percentile method.",
        "- **Exact match:** verifiable cases checked for expected answer as substring (case-insensitive).",
        f"- **Judge disclosure:** judge model is {judge_name}. Models in the same family "
        "as evaluated models may exhibit self-enhancement bias.",
        "",
        "---",
        f"*Generated by AlignForge eval pipeline. Eval ID: `{eval_id}`.*",
    ]

    return "\n".join(lines)


def _build_regression_gallery(
    metrics: dict[str, Any],
    judgements_dir: Path,
    responses_dir: Path,
    evals_dir: Path,
    n: int = 10,
) -> list[dict[str, Any]]:
    """Find the N cases where DPO lost most decisively to SFT."""
    models = metrics.get("models", [])
    # Find the DPO model ID (heuristic: the one with 'dpo' in the name).
    dpo_model = next((m for m in models if "dpo" in m.lower()), None)
    sft_model = next((m for m in models if "sft" in m.lower()), None)
    if not dpo_model or not sft_model:
        return []

    # The pair we want: SFT vs DPO, cases where DPO loses.
    pair_key = f"{sft_model}_vs_{dpo_model}"
    alt_key = f"{dpo_model}_vs_{sft_model}"

    cases_where_dpo_lost: list[dict[str, Any]] = []

    for pk in [pair_key, alt_key]:
        for suite_file in (judgements_dir / pk).glob("*.jsonl"):
            with suite_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        j = json.loads(line)
                        # DPO lost: if pair is sft_vs_dpo and verdict is model_a (SFT wins)
                        # or if pair is dpo_vs_sft and verdict is model_b (SFT wins).
                        is_dpo_loss = (pk == pair_key and j.get("final_verdict") == "model_a") or (
                            pk == alt_key and j.get("final_verdict") == "model_b"
                        )
                        if is_dpo_loss:
                            cases_where_dpo_lost.append(j)
                    except json.JSONDecodeError:
                        pass

    # Sort by presence of rationale (cases with rationale are more auditable).
    cases_where_dpo_lost = [j for j in cases_where_dpo_lost if j.get("rationale_ab")]

    gallery: list[dict[str, Any]] = []
    from alignforge.eval.suites.loader import load_suite

    cases_map: dict[str, Any] = {}
    for suite_name in metrics.get("suites", []):
        for case in load_suite(suite_name, evals_dir):
            cases_map[case.id] = case

    sft_responses = _load_response_dict_flat(responses_dir, sft_model, metrics.get("suites", []))
    dpo_responses = _load_response_dict_flat(responses_dir, dpo_model, metrics.get("suites", []))

    for j in cases_where_dpo_lost[:n]:
        case_id = j.get("case_id", "")
        raw_case = cases_map.get(case_id)
        if not raw_case:
            continue
        case = cast(EvalCase, raw_case)
        sft_resp = sft_responses.get(case_id)
        dpo_resp = dpo_responses.get(case_id)
        if not sft_resp or not dpo_resp:
            continue
        gallery.append(
            {
                "case_id": case_id,
                "prompt": case.prompt,
                "intent": case.intent,
                "winner_response": sft_resp.response,
                "loser_response": dpo_resp.response,
                "rationale": j.get("rationale_ab", ""),
            }
        )

    return gallery[:n]


def _load_response_dict_flat(
    responses_dir: Path, model_id: str, suite_names: list[str]
) -> dict[str, Any]:
    """Load all responses for a model across suites."""
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
                    if not r.error:
                        result[r.case_id] = r
                except Exception:
                    pass
    return result


def _md_to_html(md: str, eval_id: str) -> str:
    """Convert Markdown to a self-contained HTML page with inline CSS."""
    # Basic MD-to-HTML: headers, bold, tables, code. Not a full parser.
    import re

    html_body = md
    # Headers.
    for level in [4, 3, 2, 1]:
        prefix = "#" * level
        html_body = re.sub(
            rf"^{re.escape(prefix)} (.+)$",
            rf"<h{level}>\1</h{level}>",
            html_body,
            flags=re.MULTILINE,
        )
    # Bold.
    html_body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html_body)
    # Code inline.
    html_body = re.sub(r"`([^`]+)`", r"<code>\1</code>", html_body)
    # Tables (rough).
    html_body = re.sub(r"^\|---.*$", "", html_body, flags=re.MULTILINE)
    html_body = re.sub(
        r"^\|(.+)\|$",
        lambda m: (
            "<tr>" + "".join(f"<td>{c.strip()}</td>" for c in m.group(1).split("|")) + "</tr>"
        ),
        html_body,
        flags=re.MULTILINE,
    )
    html_body = re.sub(r"(<tr>.+</tr>\n)+", r"<table>\g<0></table>", html_body)
    # Paragraphs.
    html_body = re.sub(r"\n\n", r"</p><p>", html_body)
    html_body = f"<p>{html_body}</p>"
    # HR.
    html_body = html_body.replace("---", "<hr>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AlignForge Eval Report {eval_id}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 900px; margin: 40px auto; padding: 0 20px; color: #24292f; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  td, th {{ border: 1px solid #d0d7de; padding: 8px 12px; text-align: left; }}
  tr:nth-child(even) {{ background: #f6f8fa; }}
  code {{ background: #f6f8fa; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }}
  h1 {{ border-bottom: 2px solid #d0d7de; padding-bottom: 8px; }}
  h2 {{ border-bottom: 1px solid #d0d7de; padding-bottom: 4px; margin-top: 32px; }}
  hr {{ border: none; border-top: 1px solid #d0d7de; margin: 24px 0; }}
  strong {{ font-weight: 600; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""
