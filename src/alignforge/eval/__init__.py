"""Evaluation: generation, judging, metrics, and reporting."""

from alignforge.eval.generation import run_generation
from alignforge.eval.schemas import EvalCase, GeneratedResponse

__all__ = ["EvalCase", "GeneratedResponse", "run_generation"]
