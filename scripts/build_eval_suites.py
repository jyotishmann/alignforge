"""Extract MT-Bench and AlpacaEval subsets into our eval suite format.

Run once and commit the outputs:
    uv run python scripts/build_eval_suites.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def build_mtbench_sub(
    output_path: Path,
    n: int = 40,
    categories: list[str] | None = None,
    seed: int = 42,
) -> None:
    """Download MT-Bench questions and extract a subset."""
    from datasets import load_dataset

    # MT-Bench questions are in the HF dataset 'HuggingFaceH4/mt_bench_prompts'.
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    categories = categories or ["coding", "reasoning", "extraction"]

    random.seed(seed)
    cases = []
    for row in ds:
        cat = row.get("category", "")
        if cat not in categories:
            continue
        # Use only the first turn (single-turn for consistency with our setup).
        turns = row.get("turns", [])
        if not turns:
            continue
        prompt = turns[0]
        if len(prompt) < 20:
            continue
        cases.append(
            {
                "id": f"mt_{row['question_id']}",
                "prompt": prompt,
                "intent": f"MT-Bench {cat} category, question {row['question_id']}.",
                "difficulty": 2,
                "verifiable": False,
                "category": cat,
                "suite": "mtbench_sub",
            }
        )

    # Sample to target size.
    if len(cases) > n:
        cases = random.sample(cases, n)
    elif len(cases) < n:
        print(f"Warning: only {len(cases)} MT-Bench cases in categories {categories}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")
    print(f"Written {len(cases)} MT-Bench cases to {output_path}")


def build_alpacaeval_sub(
    output_path: Path,
    n: int = 60,
    seed: int = 42,
) -> None:
    """Sample AlpacaEval instructions as a general instruction-following test."""
    from datasets import load_dataset

    # tatsu-lab/alpaca_eval contains 805 instructions with reference outputs.
    ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")

    random.seed(seed)
    all_rows = list(ds)
    sample = random.sample(all_rows, min(n, len(all_rows)))

    cases = []
    for i, row in enumerate(sample):
        instruction = row.get("instruction", "")
        if not instruction or len(instruction) < 10:
            continue
        cases.append(
            {
                "id": f"ae_{i:03d}",
                "prompt": instruction,
                "intent": "AlpacaEval general instruction-following.",
                "difficulty": 2,
                "verifiable": False,
                "category": "general",
                "suite": "alpacaeval_sub",
            }
        )

    with output_path.open("w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")
    print(f"Written {len(cases)} AlpacaEval cases to {output_path}")


if __name__ == "__main__":
    evals_dir = Path("evals")
    build_mtbench_sub(evals_dir / "mtbench_sub.jsonl", n=40)
    build_alpacaeval_sub(evals_dir / "alpacaeval_sub.jsonl", n=60)
    print("Done. Commit the generated JSONL files.")
