"""The evaluation rubric — specification for what 'better' means.

This rubric appears verbatim in the judge prompt (eval/judge.py).
Any change here must be reflected in the prompt and vice versa.
The rubric IS the specification of the evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RubricDimension:
    name: str
    description: str
    weight: float = 1.0
    scale: tuple[str, str] = ("1 = worst", "5 = best")


@dataclass(frozen=True)
class Rubric:
    dimensions: tuple[RubricDimension, ...] = field(default_factory=tuple)
    name: str = "developer_assistant_v1"
    version: str = "1.0"

    def to_prompt_text(self) -> str:
        """Render the rubric as the criteria section of a judge prompt."""
        lines = ["Evaluate each response on these dimensions (1-5 scale):\n"]
        for dim in self.dimensions:
            lines.append(f"**{dim.name}** (weight {dim.weight})")
            lines.append(f"  {dim.description}")
            lines.append(f"  Scale: {dim.scale[0]}, {dim.scale[1]}\n")
        return "\n".join(lines)

    def dimension_names(self) -> list[str]:
        return [d.name for d in self.dimensions]


# ── The canonical rubric for the developer assistant domain ──────────────

DEVELOPER_ASSISTANT_RUBRIC = Rubric(
    name="developer_assistant_v1",
    version="1.0",
    dimensions=(
        RubricDimension(
            name="Correctness",
            description=(
                "Is the technical claim true? If code is present, would it run "
                "and produce the expected output? Factual errors or broken code "
                "score 1-2 regardless of other dimensions."
            ),
            weight=2.0,  # correctness weighted double - a wrong answer is never good
        ),
        RubricDimension(
            name="Directness",
            description=(
                "Does the answer start with the answer? No 'Great question!', "
                "no restating the prompt, no preamble. Score 5 if the first "
                "sentence is substantive."
            ),
            weight=1.0,
        ),
        RubricDimension(
            name="Concision",
            description=(
                "Is the length proportional to the complexity of the question? "
                "A one-line question deserves a short answer. Padding, repetition, "
                "or unnecessary elaboration reduces the score."
            ),
            weight=1.0,
        ),
        RubricDimension(
            name="Actionability",
            description=(
                "Can the reader do something immediately after reading? "
                "Concrete steps, runnable code, or a clear decision point score "
                "high. Vague generalities score low."
            ),
            weight=1.0,
        ),
        RubricDimension(
            name="Calibration",
            description=(
                "Is uncertainty stated explicitly when the question is ambiguous, "
                "the answer is version-dependent, or the premise is wrong? "
                "A response that confidently answers an ambiguous question as if "
                "it were unambiguous scores 1-2."
            ),
            weight=1.0,
        ),
    ),
)
