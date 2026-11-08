"""BM25 ranking for Steam game retrieval."""

from __future__ import annotations

from typing import Any

import numpy as np

from ir.bm25_index import BM25Index


def rank_games_bm25(
    query: str,
    index: BM25Index,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Rank Steam games using BM25 relevance scores."""

    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    if not query.strip() or top_k == 0:
        return []

    scores = np.asarray(index.score_query(query), dtype=float)

    if scores.size == 0:
        return []

    # Use App ID as a stable tie-breaker when BM25 scores are equal.
    app_ids = np.asarray(index.app_ids)
    order = np.lexsort((app_ids, -scores))[:top_k]

    ranked = []

    for position in order:
        position = int(position)

        ranked.append({
            **index.games[position],
            "bm25_score": float(scores[position]),
        })

    return ranked