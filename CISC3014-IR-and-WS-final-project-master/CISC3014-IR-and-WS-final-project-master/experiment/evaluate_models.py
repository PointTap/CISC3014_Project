"""Evaluate Steam retrieval models against manual relevance judgments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow imports from the project root when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.search_engine import SearchEngine
from ir.bm25_field_index import FieldBM25Index
from ir.bm25_improved_index import ImprovedBM25Index
from ir.bm25_index import BM25Index
from ir.bm25_ranking import rank_games_bm25
from ir.evaluation import evaluate_ranking


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "games.json"
JUDGMENTS_PATH = PROJECT_ROOT / "experiment" / "judgments.json"

TOP_K = 10


def load_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def extract_ids(results: list[dict]) -> list[int]:
    """Convert ranked game records into App ID rankings."""
    return [int(game["app_id"]) for game in results]


def main() -> None:
    print("=" * 76)
    print("STEAM GAME RETRIEVAL EVALUATION")
    print("=" * 76)

    games = load_json(DATA_PATH)
    raw_judgments = load_json(JUDGMENTS_PATH)

    # Convert manual binary judgments into:
    # query_id -> set of relevant App IDs.
    judgments: dict[str, set[int]] = {}
    query_texts: dict[str, str] = {}

    incomplete = []

    for query_id, query_data in raw_judgments.items():
        query_texts[query_id] = query_data["query"]

        relevant_ids = set()

        for game in query_data["games"]:
            relevance = game.get("relevant")

            if relevance is None:
                incomplete.append(
                    (query_id, game["app_id"], game["title"])
                )
            elif relevance is True:
                relevant_ids.add(int(game["app_id"]))

        judgments[query_id] = relevant_ids

    if incomplete:
        print()
        print("ERROR: Some relevance judgments are incomplete.")

        for query_id, app_id, title in incomplete:
            print(
                f"  {query_id}: {app_id} - {title}"
            )

        print()
        print("Finish judging before running evaluation.")
        raise SystemExit(1)

    print(f"\nLoaded {len(games)} Steam games.")
    print(f"Loaded {len(judgments)} judged queries.")

    total_relevant = sum(
        len(items)
        for items in judgments.values()
    )

    print(f"Total relevant judgments: {total_relevant}")

    print("\nBuilding retrieval indexes...")

    tfidf_engine = SearchEngine(games)
    bm25_index = BM25Index(games)
    field_index = FieldBM25Index(games)
    improved_index = ImprovedBM25Index(games)

    print("Indexes ready.")

    rankings: dict[str, dict[str, list[int]]] = {
        "TF-IDF": {},
        "BM25": {},
        "BM25-Field": {},
        "BM25-Improved": {},
    }

    print("\nRunning queries...")

    for query_id, query in query_texts.items():
        print(f"  {query_id}: {query}")

        tfidf_results = tfidf_engine.search(
            query,
            TOP_K,
        )

        bm25_results = rank_games_bm25(
            query,
            bm25_index,
            TOP_K,
        )

        field_results = rank_games_bm25(
            query,
            field_index,
            TOP_K,
        )

        improved_results = rank_games_bm25(
            query,
            improved_index,
            TOP_K,
        )

        rankings["TF-IDF"][query_id] = extract_ids(
            tfidf_results
        )

        rankings["BM25"][query_id] = extract_ids(
            bm25_results
        )

        rankings["BM25-Field"][query_id] = extract_ids(
            field_results
        )

        rankings["BM25-Improved"][query_id] = extract_ids(
            improved_results
        )

    print()
    print("=" * 76)
    print("OVERALL RESULTS")
    print("=" * 76)

    all_metrics = {}

    for model_name, model_rankings in rankings.items():
        metrics = evaluate_ranking(
            model_rankings,
            judgments,
            k=TOP_K,
        )

        all_metrics[model_name] = metrics

    print(
        f"{'MODEL':<18}"
        f"{'P@10':>12}"
        f"{'R@10':>12}"
        f"{'MAP':>12}"
        f"{'NDCG@10':>14}"
    )

    print("-" * 68)

    for model_name, metrics in all_metrics.items():
        print(
            f"{model_name:<18}"
            f"{metrics['precision@k']:>12.4f}"
            f"{metrics['recall@k']:>12.4f}"
            f"{metrics['map']:>12.4f}"
            f"{metrics['ndcg@k']:>14.4f}"
        )

    print()
    print("=" * 76)
    print("PER-QUERY PRECISION@10")
    print("=" * 76)

    for query_id, query in query_texts.items():
        print()
        print(f"{query_id}: {query}")
        print("-" * 76)

        relevant = judgments[query_id]

        for model_name, model_rankings in rankings.items():
            ranked = model_rankings[query_id]

            relevant_found = sum(
                app_id in relevant
                for app_id in ranked[:TOP_K]
            )

            precision = relevant_found / TOP_K

            print(
                f"  {model_name:<16} "
                f"{precision:.4f} "
                f"({relevant_found}/{TOP_K})"
            )

    print()
    print("=" * 76)

    best_model = max(
        all_metrics,
        key=lambda name: all_metrics[name]["ndcg@k"],
    )

    print(
        f"Best model by NDCG@10: {best_model} "
        f"({all_metrics[best_model]['ndcg@k']:.4f})"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()