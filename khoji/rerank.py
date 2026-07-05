"""Reranker — the precision stage of retrieval.

A cross-encoder scores (query, passage) *jointly*, so it can tell "this passage is about
Bohr" from "this passage is about Rutherford" — the distinction a dense bi-encoder blurs
(and the direct cause of the Bohr↔Rutherford wrong-chunk answers). It runs over the fused
dense∪BM25 pool and reorders it before the min-score gate.

Swappable backend (same discipline as persona.py). Default: FlashRank, which runs small
ONNX cross-encoders on the already-installed `onnxruntime` — no PyTorch, ~100 MB RAM, so it
can't page the resident Phi model. `rerank_model` can be swapped for a heavier cross-encoder
(e.g. bge-reranker) on a machine with more RAM.
"""

from __future__ import annotations

import logging

from khoji.config import settings

log = logging.getLogger("khoji")

_ranker = None      # lazily-loaded FlashRank Ranker (loads the model on first use)
_failed = False     # once loading fails, stop retrying — serving falls back to RRF order


def _flashrank():
    global _ranker
    if _ranker is None:
        from flashrank import Ranker
        settings.rerank_cache_dir.mkdir(parents=True, exist_ok=True)
        _ranker = Ranker(model_name=settings.rerank_model,
                         cache_dir=str(settings.rerank_cache_dir))
    return _ranker


def rerank(query: str, passages: list[str]) -> list[float] | None:
    """Score each passage against the query (input order preserved).

    Returns None if the reranker is unavailable (not installed / model can't load) so the
    caller keeps the RRF ordering instead of crashing.
    """
    global _failed
    if _failed or not passages:
        return None
    try:
        from flashrank import RerankRequest
        ranker = _flashrank()
        req = RerankRequest(query=query,
                            passages=[{"id": i, "text": p} for i, p in enumerate(passages)])
        scores = [0.0] * len(passages)
        for r in ranker.rerank(req):
            scores[int(r["id"])] = float(r["score"])
        return scores
    except Exception as e:
        _failed = True
        log.warning("reranker unavailable — falling back to RRF ordering: %s", e)
        return None


def warmup() -> None:
    """Preload the cross-encoder at boot so the first real query isn't cold."""
    if not settings.rerank:
        return
    if rerank("warmup query", ["a warmup passage"]) is not None:
        log.info("reranker warmed up (%s)", settings.rerank_model)
