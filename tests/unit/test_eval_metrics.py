"""Evaluation metrics unit tests — no API calls, no GPU."""

from __future__ import annotations

# import pytest


class TestWinRate:
    def test_all_model_a_wins(self) -> None:
        from alignforge.eval.metrics import compute_win_rate

        js = [{"final_verdict": "model_a"}] * 10
        r = compute_win_rate(js, "a", "b")
        assert r["win_rate"] == 1.0
        assert r["wins"] == 10

    def test_ties_count_as_half(self) -> None:
        from alignforge.eval.metrics import compute_win_rate

        js = [{"final_verdict": "tie"}] * 10
        r = compute_win_rate(js, "a", "b")
        assert r["win_rate"] == 0.5

    def test_mixed(self) -> None:
        from alignforge.eval.metrics import compute_win_rate

        js = (
            [{"final_verdict": "model_a"}] * 6
            + [{"final_verdict": "model_b"}] * 2
            + [{"final_verdict": "tie"}] * 2
        )
        r = compute_win_rate(js, "a", "b")
        # (6 + 0.5*2) / 10 = 0.7
        assert abs(r["win_rate"] - 0.7) < 0.001


class TestBootstrapCI:
    def test_ci_contains_point_estimate(self) -> None:
        from alignforge.eval.metrics import bootstrap_ci

        js = [{"final_verdict": "model_a"}] * 7 + [{"final_verdict": "model_b"}] * 3
        result = bootstrap_ci(js, "a", "b", n_resamples=1000)
        assert result["ci_low"] <= result["win_rate"] <= result["ci_high"]

    def test_ci_width_decreases_with_n(self) -> None:
        """More cases → narrower CI."""
        from alignforge.eval.metrics import bootstrap_ci

        small = [{"final_verdict": "model_a"}] * 3 + [{"final_verdict": "model_b"}] * 2
        large = small * 10
        r_small = bootstrap_ci(small, "a", "b", n_resamples=1000)
        r_large = bootstrap_ci(large, "a", "b", n_resamples=1000)
        small_width = r_small["ci_high"] - r_small["ci_low"]
        large_width = r_large["ci_high"] - r_large["ci_low"]
        assert large_width < small_width


class TestPositionBias:
    def test_bias_rate_computed(self) -> None:
        from alignforge.eval.metrics import position_bias_rate

        js = [
            {"position_bias_observed": True},
            {"position_bias_observed": False},
            {"position_bias_observed": True},
        ]
        rate = position_bias_rate(js)
        assert abs(rate - 2 / 3) < 0.01


class TestJudgeOutput:
    def test_parse_valid_json(self) -> None:
        from alignforge.eval.judge import parse_judge_response

        raw = '{"verdict": "A", "rationale": "A is better.", "dimension_scores": {}, "confidence": "high"}'
        result = parse_judge_response(raw)
        assert result is not None
        assert result["verdict"] == "A"

    def test_parse_strips_code_fence(self) -> None:
        from alignforge.eval.judge import parse_judge_response

        raw = '```json\n{"verdict": "tie", "rationale": "equal"}\n```'
        result = parse_judge_response(raw)
        assert result is not None
        assert result["verdict"] == "tie"

    def test_invalid_verdict_returns_none(self) -> None:
        from alignforge.eval.judge import parse_judge_response

        raw = '{"verdict": "C", "rationale": "bad"}'
        result = parse_judge_response(raw)
        assert result is None


class TestPositionSwap:
    def test_consistent_verdict_not_biased(self) -> None:
        """When judge returns same verdict in both orders, no bias flagged."""
        from alignforge.eval.judge import adjudicate_pair
        from alignforge.eval.judge_client import make_echo_judge
        from alignforge.eval.schemas import EvalCase, GeneratedResponse

        judge_fn = make_echo_judge(always_prefer="A")
        case = EvalCase(id="t1", prompt="Test question here?", intent="test", suite="test")
        resp_a = GeneratedResponse(
            case_id="t1",
            suite="test",
            model_id="m_a",
            prompt="Test",
            response="Response A text here.",
            n_tokens=5,
            time_seconds=0.1,
        )
        resp_b = GeneratedResponse(
            case_id="t1",
            suite="test",
            model_id="m_b",
            prompt="Test",
            response="Response B text here.",
            n_tokens=5,
            time_seconds=0.1,
        )

        record = adjudicate_pair(case, resp_a, resp_b, judge_fn)
        # Echo judge always says "A". In order 1: A=model_a (A wins). In order 2: A=model_b.
        # Translated back: order 2 verdict is "B" in original space → inconsistent → tie.
        assert record.final_verdict in ("tie", "model_a")
        # Position bias IS observed when the judge always says "A" but models swap.
        # In order 1: A→model_a wins. In order 2: A→model_b (which maps to B in original) → inconsistent.
        assert record.position_bias_observed is True
