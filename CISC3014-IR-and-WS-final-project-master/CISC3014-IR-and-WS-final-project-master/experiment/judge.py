"""Interactive relevance judging for the Steam retrieval experiment."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "games.json"
JUDGMENTS_PATH = PROJECT_ROOT / "experiment" / "judgments.json"


def load_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def shorten(text: str, limit: int = 700) -> str:
    text = " ".join(str(text or "").split())

    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + "..."


def format_list(value) -> str:
    if not value:
        return "(none)"

    if isinstance(value, list):
        return ", ".join(str(item) for item in value)

    return str(value)


def count_progress(judgments) -> tuple[int, int]:
    total = 0
    completed = 0

    for query_data in judgments.values():
        for game in query_data["games"]:
            total += 1

            if game["relevant"] is not None:
                completed += 1

    return completed, total


def main() -> None:
    games = load_json(DATA_PATH)
    judgments = load_json(JUDGMENTS_PATH)

    games_by_id = {
        int(game["app_id"]): game
        for game in games
    }

    print()
    print("=" * 72)
    print("STEAM GAME RELEVANCE JUDGMENT")
    print("=" * 72)
    print()
    print("Controls:")
    print("  y = relevant")
    print("  n = not relevant")
    print("  s = skip for now")
    print("  q = save and quit")
    print()
    print(
        "Judge whether the GAME substantially satisfies "
        "the WHOLE QUERY."
    )
    print()

    for query_id, query_data in judgments.items():

        query = query_data["query"]

        for judgment in query_data["games"]:

            if judgment["relevant"] is not None:
                continue

            app_id = int(judgment["app_id"])
            game = games_by_id.get(app_id)

            completed, total = count_progress(judgments)

            print()
            print("=" * 72)
            print(
                f"Progress: {completed}/{total}"
            )
            print(
                f"Query ID: {query_id}"
            )
            print(
                f"QUERY: {query}"
            )
            print("-" * 72)

            print(
                f"Game: {judgment['title']}"
            )

            print(
                f"App ID: {app_id}"
            )

            if game is None:
                print()
                print(
                    "WARNING: Game record was not found "
                    "in games.json."
                )
            else:
                print()
                print(
                    "Genres:"
                )
                print(
                    format_list(game.get("genres"))
                )

                print()
                print(
                    "Categories:"
                )
                print(
                    format_list(game.get("categories"))
                )

                print()
                print(
                    "Tags:"
                )
                print(
                    format_list(game.get("tags"))
                )

                print()
                print(
                    "Description:"
                )
                print(
                    shorten(
                        game.get("description", "")
                    )
                )

            print()
            print("-" * 72)

            while True:

                answer = input(
                    "Relevant? [y/n/s/q]: "
                ).strip().lower()

                if answer == "y":
                    judgment["relevant"] = True
                    save_json(
                        JUDGMENTS_PATH,
                        judgments,
                    )
                    break

                if answer == "n":
                    judgment["relevant"] = False
                    save_json(
                        JUDGMENTS_PATH,
                        judgments,
                    )
                    break

                if answer == "s":
                    break

                if answer == "q":
                    save_json(
                        JUDGMENTS_PATH,
                        judgments,
                    )

                    completed, total = count_progress(
                        judgments
                    )

                    print()
                    print(
                        f"Saved. Progress: "
                        f"{completed}/{total}"
                    )

                    return

                print(
                    "Please enter y, n, s, or q."
                )

    save_json(
        JUDGMENTS_PATH,
        judgments,
    )

    completed, total = count_progress(
        judgments
    )

    print()
    print("=" * 72)
    print(
        f"Judging complete: {completed}/{total}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()