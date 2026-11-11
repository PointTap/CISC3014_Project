"""Standard ranked-retrieval metrics for binary relevance judgments."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def precision_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_ids = set(relevant)
    return sum(app_id in relevant_ids for app_id in ranked[:k]) / k


def recall_at_k(ranked: Sequence[int], relevant: Iterable[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_ids = set(relevant)
    if not relevant_ids:
        return 0.0
    return sum(app_id in relevant_ids for app_id in ranked[:k]) / len(relevant_ids)


def average_precision(ranked: Sequence[int], relevant: Iterable[int]) -> float:
    relevant_ids = set(relevant)
    if not relevant_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    seen: set[int] = set()
    for rank, app_id in enumerate(ranked, start=1):
        if app_id in relevant_ids and app_id not in seen:
            hits += 1
            precision_sum += hits / rank
            seen.add(app_id)
    return precision_sum / len(relevant_ids)


def ndcg_at_k(ranked: Sequence[int], relevance: Mapping[int, int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    gains = [max(0, int(relevance.get(app_id, 0))) for app_id in ranked[:k]]
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal = sorted((max(0, int(score)) for score in relevance.values()), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def mean_average_precision(
    rankings: Mapping[str, Sequence[int]],
    judgments: Mapping[str, Iterable[int]],
) -> float:
    """Average AP over judged queries; missing results count as empty rankings."""
    if not judgments:
        return 0.0
    return sum(
        average_precision(rankings.get(query, ()), relevant)
        for query, relevant in judgments.items()
    ) / len(judgments)


def evaluate_ranking(
    rankings: Mapping[str, Sequence[int]],
    judgments: Mapping[str, Iterable[int]],
    k: int = 10,
) -> dict[str, float]:
    """Compute macro Precision@K, Recall@K, MAP, and binary NDCG@K."""
    if k <= 0:
        raise ValueError("k must be positive")
    queries = list(judgments)
    if not queries:
        return {"precision@k": 0.0, "recall@k": 0.0, "map": 0.0, "ndcg@k": 0.0}
    precision = []
    recall = []
    ndcg = []
    for query in queries:
        relevant = set(judgments[query])
        ranked = rankings.get(query, ())
        precision.append(precision_at_k(ranked, relevant, k))
        recall.append(recall_at_k(ranked, relevant, k))
        ndcg.append(ndcg_at_k(ranked, {item: 1 for item in relevant}, k))
    return {
        "precision@k": sum(precision) / len(queries),
        "recall@k": sum(recall) / len(queries),
        "map": mean_average_precision(rankings, judgments),
        "ndcg@k": sum(ndcg) / len(queries),
    }
