"""Two-stage domain filter: keyword recall → embedding precision."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

log = structlog.get_logger()

# ── Stage 1: keyword/regex recall filter ─────────────────────────────────

# Technical lexicon — tuned for "developer support assistant" domain.
_TECHNICAL_TERMS = [
    # Languages
    r"\bpython\b",
    r"\bjavascript\b",
    r"\btypescript\b",
    r"\bjava\b",
    r"\bc\+\+\b",
    r"\brust\b",
    r"\bgo\b",
    r"\bruby\b",
    r"\bswift\b",
    r"\bkotlin\b",
    r"\bsql\b",
    r"\bhtml\b",
    r"\bcss\b",
    r"\bbash\b",
    r"\bshell\b",
    r"\bphp\b",
    r"\bscala\b",
    r"\br\b(?=\s+(?:script|code|package))",
    # Tools and concepts
    r"\bgit\b",
    r"\bdocker\b",
    r"\bkubernetes\b",
    r"\blinux\b",
    r"\bapi\b",
    r"\brest\b",
    r"\bgraphql\b",
    r"\bhttp\b",
    r"\bdatabase\b",
    r"\bpostgres\b",
    r"\bmysql\b",
    r"\bmongodb\b",
    r"\bredis\b",
    r"\baws\b",
    r"\bazure\b",
    r"\bgcp\b",
    r"\bcloud\b",
    r"\bci/cd\b",
    r"\bterraform\b",
    r"\bansible\b",
    r"\bmachine\s+learning\b",
    r"\bdeep\s+learning\b",
    r"\bneural\s+net",
    r"\btransformer\b",
    r"\bpytorch\b",
    r"\btensorflow\b",
    r"\bnpm\b",
    r"\bpip\b",
    r"\bcargo\b",
    r"\byarn\b",
    # Code patterns
    r"```",  # code fences
    r"\bdef\s+\w+\s*\(",  # Python function definitions
    r"\bfunction\s+\w+\s*\(",  # JS function definitions
    r"\bclass\s+\w+",  # class definitions
    r"\bimport\s+\w+",  # import statements
    r"\berror\b.*\b(?:at|in|on)\b",  # error-message shapes
    r"(?:Error|Exception|Traceback)\b",
    r"\bstack\s*(?:trace|overflow)\b",
    r"\bdebug\b",
    r"\bcompile\b",
    r"\brun(?:time|ning)?\b",
    r"\bcommand\s+(?:line|not\s+found)\b",
]

_TECHNICAL_RE = re.compile("|".join(_TECHNICAL_TERMS), re.IGNORECASE)


def keyword_filter(text: str) -> bool:
    """Stage 1: return True if the text matches the technical lexicon."""
    return bool(_TECHNICAL_RE.search(text))


class EmbeddingFilter:
    """Stage 2: cosine similarity to domain eval centroids."""

    def __init__(
        self,
        eval_prompts: list[str],
        n_clusters: int = 5,
        threshold: float = 0.45,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold

        # Embed eval prompts and compute centroids.
        log.info("embedding_filter_init", n_prompts=len(eval_prompts), n_clusters=n_clusters)
        eval_embeddings = self.model.encode(eval_prompts, normalize_embeddings=True)
        km = KMeans(n_clusters=min(n_clusters, len(eval_prompts)), n_init=10, random_state=42)
        km.fit(eval_embeddings)
        # Normalise centroids for cosine similarity via dot product.
        self.centroids = km.cluster_centers_
        norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
        self.centroids = self.centroids / np.maximum(norms, 1e-8)

    def score(self, text: str) -> float:
        """Max cosine similarity to any centroid."""
        emb = self.model.encode([text], normalize_embeddings=True)
        sims = emb @ self.centroids.T
        return float(np.max(sims))

    def passes(self, text: str) -> bool:
        return self.score(text) >= self.threshold

    def batch_filter(self, texts: list[str], batch_size: int = 256) -> list[bool]:
        """Batch-encode for efficiency; returns a mask."""
        all_embs = self.model.encode(texts, normalize_embeddings=True, batch_size=batch_size)
        sims = all_embs @ self.centroids.T  # (N, k)
        max_sims = np.max(sims, axis=1)  # (N,)
        return [bool(s >= self.threshold) for s in max_sims]


def load_eval_prompts(evals_dir: Path, suites: list[str] | None = None) -> list[str]:
    """Load evaluation prompts from JSONL files for decontamination and filtering."""
    prompts: list[str] = []
    suites = suites or ["domain_v1"]
    for suite in suites:
        path = evals_dir / f"{suite}.jsonl"
        if not path.exists():
            log.warning("eval_suite_missing", path=str(path))
            continue
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                prompts.append(row.get("prompt", row.get("instruction", "")))
    log.info("eval_prompts_loaded", n=len(prompts))
    return prompts


def tune_threshold(
    filter_obj: EmbeddingFilter,
    candidates: list[str],
    labels: list[bool],
    thresholds: list[float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Find the threshold that maximises F1 against hand labels.

    Returns (best_threshold, {threshold: f1, ...}).
    """
    thresholds = thresholds or [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    scores = [filter_obj.score(t) for t in candidates]
    results: dict[str, float] = {}
    best_t, best_f1 = 0.45, 0.0

    for t in thresholds:
        preds = [s >= t for s in scores]
        f1 = f1_score(labels, preds, zero_division=0.0)
        results[str(t)] = round(f1, 3)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    log.info("threshold_tuned", best_threshold=best_t, best_f1=round(best_f1, 3))
    return best_t, results


def apply_domain_filter(
    examples: list[Any],
    eval_prompts: list[str],
    threshold: float = 0.45,
    skip_embedding: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    """Two-stage domain filter. Returns (survivors, filter_stats).

    Args:
        examples: SFTExample or PreferenceExample instances.
        eval_prompts: prompts from the eval suites, for embedding centroids.
        threshold: cosine similarity threshold for Stage 2.
        skip_embedding: if True, only run Stage 1 (useful for --limit debugging).
    """
    n_input = len(examples)

    # Stage 1: keyword recall.
    def _get_text(ex: Any) -> str:
        return ex.prompt_text if hasattr(ex, "prompt_text") else ex.prompt

    stage1 = [ex for ex in examples if keyword_filter(_get_text(ex))]
    n_after_kw = len(stage1)
    log.info("filter_stage1_keyword", before=n_input, after=n_after_kw)

    if skip_embedding or not eval_prompts:
        return stage1, {
            "input": n_input,
            "after_keyword": n_after_kw,
            "after_embedding": n_after_kw,
            "threshold": None,
            "embedding_skipped": True,
        }

    # Stage 2: embedding precision.
    emb_filter = EmbeddingFilter(eval_prompts, threshold=threshold)
    texts = [_get_text(ex) for ex in stage1]
    mask = emb_filter.batch_filter(texts)
    stage2 = [ex for ex, keep in zip(stage1, mask, strict=False) if keep]
    n_after_emb = len(stage2)
    log.info("filter_stage2_embedding", before=n_after_kw, after=n_after_emb, threshold=threshold)

    stats = {
        "input": n_input,
        "after_keyword": n_after_kw,
        "after_embedding": n_after_emb,
        "threshold": threshold,
        "embedding_skipped": False,
    }
    return stage2, stats
