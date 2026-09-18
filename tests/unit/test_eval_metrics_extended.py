"""Extended eval metrics tests — edge cases and three-model Bradley-Terry."""

from __future__ import annotations

from alignforge.eval.metrics import (
    # bootstrap_ci,
    # compute_win_rate,
    exact_match_score,
    fit_bradley_terry,
    length_controlled_win_rate,
    # position_bias_rate,
)


class TestBradleyTerry:
    def test_three_model_ranking(self) -> None:
        """Three models with clear ordering produce consistent Elo scores."""
        pairwise = {
            "base_vs_sft": {"wins_a": 2, "wins_b": 8, "ties": 0},
            "base_vs_dpo": {"wins_a": 1, "wins_b": 9, "ties": 0},
            "sft_vs_dpo": {"wins_a": 4, "wins_b": 6, "ties": 0},
        }
        elo = fit_bradley_terry(pairwise)
        assert set(elo.keys()) == {"base", "sft", "dpo"}
        # DPO should have the highest Elo, base the lowest.
        assert elo["dpo"] > elo["sft"] > elo["base"]

    def test_single_model_returns_1000(self) -> None:
        elo = fit_bradley_terry({"a_vs_a": {"wins_a": 0, "wins_b": 0, "ties": 1}})
        for v in elo.values():
            assert abs(v - 1000.0) < 1.0

    def test_elo_centred_at_1000(self) -> None:
        pairwise = {
            "a_vs_b": {"wins_a": 6, "wins_b": 4, "ties": 0},
        }
        elo = fit_bradley_terry(pairwise)
        mean_elo = sum(elo.values()) / len(elo)
        assert abs(mean_elo - 1000.0) < 50.0  # centred approximately at 1000


class TestLengthControl:
    def test_discards_long_winner(self) -> None:
        """Cases where the winner is much longer are discarded."""
        from alignforge.eval.schemas import GeneratedResponse

        def make_resp(case_id: str, model: str, n_tok: int) -> GeneratedResponse:
            return GeneratedResponse(
                case_id=case_id,
                suite="test",
                model_id=model,
                prompt="q",
                response="r" * n_tok,
                n_tokens=n_tok,
                time_seconds=0.1,
            )

        judgements = [
            {"case_id": "d01", "final_verdict": "model_a"},  # A wins
        ]
        responses_a = {"d01": make_resp("d01", "sft", 500)}  # A is very long
        responses_b = {"d01": make_resp("d01", "dpo", 100)}  # B is short

        result = length_controlled_win_rate(
            judgements,
            responses_a,
            responses_b,
            "sft",
            "dpo",
            length_tolerance=0.20,
            n_resamples=100,
        )
        # The case where A won but A was 5x longer should be discarded.
        assert result["n_discarded"] == 1
        assert result["n"] == 0  # nothing left to compute

    def test_keeps_similar_length_cases(self) -> None:
        from alignforge.eval.schemas import GeneratedResponse

        def make_resp(cid: str, model: str, n_tok: int) -> GeneratedResponse:
            return GeneratedResponse(
                case_id=cid,
                suite="test",
                model_id=model,
                prompt="q",
                response="r",
                n_tokens=n_tok,
                time_seconds=0.1,
            )

        judgements = [{"case_id": f"d{i:02d}", "final_verdict": "model_a"} for i in range(5)]
        responses_a = {f"d{i:02d}": make_resp(f"d{i:02d}", "sft", 100) for i in range(5)}
        responses_b = {f"d{i:02d}": make_resp(f"d{i:02d}", "dpo", 110) for i in range(5)}

        result = length_controlled_win_rate(
            judgements,
            responses_a,
            responses_b,
            "sft",
            "dpo",
            length_tolerance=0.20,
            n_resamples=100,
        )
        assert result["n_discarded"] == 0
        assert result["n"] == 5


class TestExactMatch:
    def test_substring_match(self) -> None:
        from alignforge.eval.schemas import EvalCase, GeneratedResponse

        cases = {
            "v01": EvalCase(
                id="v01",
                prompt="What is Python?",
                intent="factual",
                verifiable=True,
                expected_answer="interpreted language",
            )
        }
        responses = {
            "v01": GeneratedResponse(
                case_id="v01",
                suite="test",
                model_id="dpo",
                prompt="What is Python?",
                response="Python is an interpreted language known for readability.",
                n_tokens=10,
                time_seconds=0.1,
            )
        }
        result = exact_match_score([], responses, cases)
        assert result["n_correct"] == 1
        assert result["exact_match"] == 1.0

    def test_pipe_alternatives(self) -> None:
        """expected_answer with | separator — any match counts."""
        from alignforge.eval.schemas import EvalCase, GeneratedResponse

        cases = {
            "v02": EvalCase(
                id="v02",
                prompt="How to reverse?",
                intent="code",
                verifiable=True,
                expected_answer="list.reverse()|reversed()",
            )
        }
        responses = {
            "v02": GeneratedResponse(
                case_id="v02",
                suite="test",
                model_id="dpo",
                prompt="How?",
                response="Use reversed() for a new list.",
                n_tokens=7,
                time_seconds=0.1,
            )
        }
        result = exact_match_score([], responses, cases)
        assert result["exact_match"] == 1.0

    def test_wrong_answer_scores_zero(self) -> None:
        from alignforge.eval.schemas import EvalCase, GeneratedResponse

        cases = {
            "v03": EvalCase(
                id="v03",
                prompt="Is MD5 ok?",
                intent="security",
                verifiable=True,
                expected_answer="bcrypt|argon2",
            )
        }
        responses = {
            "v03": GeneratedResponse(
                case_id="v03",
                suite="test",
                model_id="base",
                prompt="Is MD5 ok?",
                response="MD5 is fine for passwords.",  # wrong!
                n_tokens=6,
                time_seconds=0.1,
            )
        }
        result = exact_match_score([], responses, cases)
        assert result["exact_match"] == 0.0
        assert len(result["failures"]) == 1
