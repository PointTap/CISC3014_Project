"""Build a pooled set of games for manual relevance judgment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow this script to import project modules when run from experiment/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.search_engine import SearchEngine
from ir.bm25_field_index import FieldBM25Index
from ir.bm25_improved_index import ImprovedBM25Index
from ir.bm25_index import BM25Index
from ir.bm25_ranking import rank_games_bm25


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "games.json"
QUERY_PATH = PROJECT_ROOT / "experiment" / "queries.json"
OUTPUT_PATH = PROJECT_ROOT / "experiment" / "pool.json"

POOL_DEPTH = 10


def load_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    games = load_json(DATA_PATH)
    queries = load_json(QUERY_PATH)

    tfidf_engine = SearchEngine(games)

    bm25_index = BM25Index(games)
    field_index = FieldBM25Index(games)
    improved_index = ImprovedBM25Index(games)

    pool = {}

    for query_id, query in queries.items():
        rankings = {
            "tfidf": tfidf_engine.search(query, POOL_DEPTH),
            "bm25": rank_games_bm25(
                query,
                bm25_index,
                POOL_DEPTH,
            ),
            "bm25-field": rank_games_bm25(
                query,
                field_index,
                POOL_DEPTH,
            ),
            "bm25-improved": rank_games_bm25(
                query,
                improved_index,
                POOL_DEPTH,
            ),
        }

        candidates = {}

        for model_name, results in rankings.items():
            for game in results:
                app_id = int(game["app_id"])

                if app_id not in candidates:
                    candidates[app_id] = {
                        "app_id": app_id,
                        "title": game.get("title", ""),
                        "url": game.get("url", ""),
                        "retrieved_by": [],
                    }

                candidates[app_id]["retrieved_by"].append(
                    model_name
                )

        pool[query_id] = {
            "query": query,
            "candidates": sorted(
                candidates.values(),
                key=lambda item: (
                    item["title"].lower(),
                    item["app_id"],
                ),
            ),
        }

        print(
            f"{query_id}: {query} -> "
            f"{len(candidates)} unique pooled games"
        )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            pool,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Pool written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()