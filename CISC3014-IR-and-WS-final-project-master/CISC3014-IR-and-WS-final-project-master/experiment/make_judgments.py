"""Create a relevance-judgment template from the pooled search results."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

POOL_PATH = PROJECT_ROOT / "experiment" / "pool.json"
OUTPUT_PATH = PROJECT_ROOT / "experiment" / "judgments.json"


def main() -> None:
    with POOL_PATH.open(encoding="utf-8") as file:
        pool = json.load(file)

    judgments = {}

    for query_id, query_data in pool.items():
        judgments[query_id] = {
            "query": query_data["query"],
            "games": [
                {
                    "app_id": game["app_id"],
                    "title": game["title"],
                    "relevant": None,
                }
                for game in query_data["candidates"]
            ],
        }

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            judgments,
            file,
            ensure_ascii=False,
            indent=2,
        )

    total = sum(
        len(query["games"])
        for query in judgments.values()
    )

    print(f"Queries: {len(judgments)}")
    print(f"Games to judge: {total}")
    print(f"Written to: {OUTPUT_PATH}")
    print()
    print("Change every relevant value:")
    print("  null -> true   if relevant")
    print("  null -> false  if not relevant")


if __name__ == "__main__":
    main()