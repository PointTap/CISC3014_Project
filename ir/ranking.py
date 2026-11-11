"""Cosine-similarity ranking with optional normalized quality/tag boosts."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ir.preprocessing import tokenize
from ir.tfidf_index import TfidfIndex

COSINE_WEIGHT = 0.8
RATING_WEIGHT = 0.1
TAG_WEIGHT = 0.1


def rank_games(query: str, index: TfidfIndex, top_k: int = 10) -> list[dict[str, Any]]:
    """Rank game records; cosine similarity remains the dominant score."""
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not query.strip() or top_k == 0 or index.matrix is None:
        return []

    query_vector = index.transform_query(query)
    if query_vector is None or query_vector.nnz == 0:
        return []
    similarities = cosine_similarity(query_vector, index.matrix).ravel()
    query_terms = set(tokenize(query))
    tag_overlaps = []
    rating_scores = []

    for game in index.games:
        tag_terms = set(tokenize(" ".join(game.get("tags") or [])))
        tag_overlap = len(query_terms & tag_terms) / len(query_terms) if query_terms else 0.0
        tag_overlaps.append(tag_overlap)
        reviews = game.get("reviews") or {}
        rating = reviews.get("rating", 0.0)
        try:
            rating_score = min(1.0, max(0.0, float(rating)))
        except (TypeError, ValueError):
            rating_score = 0.0
        rating_scores.append(rating_score)

    scores = (
        COSINE_WEIGHT * similarities
        + RATING_WEIGHT * np.asarray(rating_scores)
        + TAG_WEIGHT * np.asarray(tag_overlaps)
    )
    app_ids = np.asarray(index.app_ids)
    order = np.lexsort((app_ids, -scores))[:top_k]
    ranked = []
    for position in order:
        position = int(position)
        ranked.append({
            **index.games[position],
            "cosine_similarity": float(similarities[position]),
            "tag_overlap": float(tag_overlaps[position]),
            "final_score": float(scores[position]),
        })
    return ranked
