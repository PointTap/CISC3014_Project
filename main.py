"""Command-line entry point for crawling Steam games and searching locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from app.search_engine import SearchEngine
from crawler.steamdb_chart import DEFAULT_CHART_URL, SteamDBChartCrawler
from crawler.steamdb_crawler import SteamCrawler
from crawler.tag_expansion import expand_from_seed
from database.analytics import calculate_price_analytics
from database.storage import GameDatabase
from prediction.price_predictor import (
    InsufficientPriceHistoryError,
    PricePredictor,
)
from visualization.price_history import plot_price_history

DEFAULT_DATA = Path("data/raw/games.json")
DEFAULT_DATABASE = Path("data/steam_games.db")


def _read_app_ids(args: argparse.Namespace) -> list[int]:
    values = list(args.app_ids or [])
    if args.app_id_file:
        with Path(args.app_id_file).open(encoding="utf-8") as file:
            values.extend(
                line.strip() for line in file
                if line.strip() and not line.lstrip().startswith("#")
            )
    try:
        app_ids = [int(value) for value in values]
    except ValueError as error:
        raise ValueError("App IDs must be integers") from error
    if not app_ids:
        raise ValueError("Provide at least one App ID or --app-id-file")
    if any(app_id <= 0 for app_id in app_ids):
        raise ValueError("App IDs must be positive integers")
    return app_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Steam game information retrieval")
    commands = parser.add_subparsers(dest="command", required=True)

    crawl = commands.add_parser("crawl", help="Fetch Steam Store details for App IDs")
    crawl.add_argument("--app-ids", nargs="*", default=[])
    crawl.add_argument("--app-id-file")
    crawl.add_argument(
        "--steamdb-top",
        nargs="?",
        const=100,
        type=int,
        metavar="COUNT",
        help="discover and crawl up to COUNT games from SteamDB's current-player chart (default: 100)",
    )
    crawl.add_argument(
        "--steamdb-chart-url",
        default=DEFAULT_CHART_URL,
        help="SteamDB current-player chart URL",
    )
    crawl.add_argument(
        "--store-search",
        metavar="QUERY",
        help="discover matching App IDs using Steam Store search (typically up to 10 results)",
    )
    crawl.add_argument(
        "--search-limit",
        type=int,
        default=10,
        help="maximum Store search results to crawl (default: 10)",
    )
    crawl.add_argument(
        "--tag-expand",
        type=int,
        metavar="APP_ID",
        help="recursively discover games using the seed game's Steam tags",
    )
    crawl.add_argument(
        "--depth",
        type=int,
        default=2,
        help="tag expansion depth: seed is depth 0 (default: 2)",
    )
    crawl.add_argument(
        "--max-games",
        type=int,
        default=100,
        help="maximum games in this expansion, including the seed (default: 100)",
    )
    crawl.add_argument(
        "--per-tag-limit",
        type=int,
        default=10,
        help="maximum Steam Store title-search results per tag (default: 10)",
    )
    crawl.add_argument(
        "--tags-per-game",
        type=int,
        default=5,
        help="maximum tags searched for each game (default: 5)",
    )
    crawl.add_argument(
        "--tag-state",
        help="JSON file storing tag search results (default: <output>.tag-search.json)",
    )
    crawl.add_argument(
        "--reset-tag-history",
        action="store_true",
        help="ignore and replace saved tag search results for this expansion",
    )
    crawl.add_argument("--output", default=str(DEFAULT_DATA))
    crawl.add_argument(
        "--database",
        default=str(DEFAULT_DATABASE),
        help="SQLite database for metadata and observed price snapshots",
    )
    crawl.add_argument("--delay", type=float, default=1.5)
    crawl.add_argument("--refresh", action="store_true")

    search = commands.add_parser("search", help="Search the locally crawled games")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--model", choices=("tfidf", "bm25"), default="tfidf")
    search.add_argument("--data", default=str(DEFAULT_DATA))

    sync = commands.add_parser(
        "sync-db",
        help="Import the local JSON corpus and its observed prices into SQLite",
    )
    sync.add_argument("--data", default=str(DEFAULT_DATA))
    sync.add_argument("--database", default=str(DEFAULT_DATABASE))

    menu = commands.add_parser("menu", help="Start the interactive CLI")
    menu.add_argument("--data", default=str(DEFAULT_DATA))
    menu.add_argument("--database", default=str(DEFAULT_DATABASE))
    return parser


def main(argv: list[str] | None = None) -> int:
    # Preserve unsupported Unicode from game titles in legacy Windows consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = build_parser().parse_args(argv)
    if args.command == "crawl":
        if args.search_limit <= 0:
            raise ValueError("--search-limit must be positive")
        selected_sources = sum((
            args.steamdb_top is not None,
            args.store_search is not None,
            args.tag_expand is not None,
            bool(args.app_ids or args.app_id_file),
        ))
        if selected_sources != 1:
            raise ValueError(
                "Choose exactly one source: --app-ids/--app-id-file, "
                "--steamdb-top, --store-search, or --tag-expand"
            )
        if args.tag_expand is not None:
            crawler = SteamCrawler(delay=args.delay)
            summary = expand_from_seed(
                seed_app_id=args.tag_expand,
                output_path=args.output,
                crawler=crawler,
                depth=args.depth,
                per_tag_limit=args.per_tag_limit,
                max_games=args.max_games,
                tags_per_game=args.tags_per_game,
                refresh=args.refresh,
                state_path=args.tag_state,
                reset_tag_history=args.reset_tag_history,
                on_progress=lambda level, added, total: print(
                    f"Depth {level}: discovered {added} game(s); "
                    f"local games now: {total}"
                ),
            )
            database = GameDatabase(args.database)
            database.upsert_games(_load_games(args.output), record_price=True)
            print(
                f"Expanded from App ID {summary['seed_app_id']}: "
                f"{summary['games_discovered']} game(s), "
                f"{summary['tags_searched']} tag(s) searched, "
                f"{summary['tags_failed']} failed tag search(es), "
                f"depth {summary['depth_reached']}; "
                f"downloaded {summary['games_downloaded']} new game(s) "
                f"to {args.output}. Local games now: {summary['local_games']}"
            )
            return 0
        if args.steamdb_top is not None:
            app_ids = SteamDBChartCrawler(delay=args.delay).fetch_top_app_ids(
                args.steamdb_top, args.steamdb_chart_url
            )
            if not app_ids:
                raise ValueError("SteamDB returned no App IDs from its current-player chart")
            print(f"Discovered {len(app_ids)} App IDs from SteamDB.")
            crawler = SteamCrawler(delay=args.delay)
        elif args.store_search is not None:
            crawler = SteamCrawler(delay=args.delay)
            app_ids = crawler.search_app_ids(args.store_search, args.search_limit)
            print(f"Steam Store search found {len(app_ids)} App IDs.")
        else:
            app_ids = _read_app_ids(args)
            crawler = SteamCrawler(delay=args.delay)
        games = crawler.crawl(app_ids, args.output, refresh=args.refresh)
        database = GameDatabase(args.database)
        database.upsert_games(_load_games(args.output), record_price=True)
        print(
            f"Fetched {len(games)} new game(s) from {len(app_ids)} selected App IDs; "
            f"data saved to {args.output}; SQLite database updated at "
            f"{args.database}"
        )
        return 0

    if args.command == "sync-db":
        games = _load_games(args.data)
        GameDatabase(args.database).upsert_games(games, record_price=True)
        print(
            f"Imported {len(games)} game record(s) and available observed "
            f"prices into {args.database}"
        )
        return 0

    if args.command == "menu":
        _run_menu(args.data, args.database)
        return 0

    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    engine = SearchEngine.from_json(args.data)
    results = engine.search(args.query, args.top_k, model=args.model)
    if not results:
        print("No matching games found.")
        return 0
    print(f"Model: {args.model.upper()}")
    for rank, game in enumerate(results, start=1):
        score = (
            game["final_score"]
            if args.model == "tfidf"
            else game["bm25_score"]
        )
        print(
            f"{rank}. {game.get('title', '(untitled)')} "
            f"(App ID {game['app_id']}) - {score:.4f}"
        )
        if game.get("url"):
            print(f"   {game['url']}")
    return 0


def _load_games(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as file:
        games = json.load(file)
    if not isinstance(games, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return games


def _run_menu(data_path: str | Path, database_path: str | Path) -> None:
    """Run the five-option course-project interface over local data."""
    games = _load_games(data_path)
    games_by_id = {int(game["app_id"]): game for game in games}
    database = GameDatabase(database_path)
    database.upsert_games(games, record_price=True)
    engine = SearchEngine(games)

    predictor = PricePredictor()
    try:
        predictor.fit(database.get_all_price_histories())
    except InsufficientPriceHistoryError:
        predictor = None

    while True:
        print(
            "\nSteam Game Search and Price Prediction\n"
            "1. Search Games\n"
            "2. View Game Details\n"
            "3. View Price History\n"
            "4. Predict Next Discount\n"
            "5. Exit"
        )
        choice = input("Select an option: ").strip()
        if choice == "1":
            query = input("Search: ").strip()
            if not query:
                print("Enter a non-empty natural-language query.")
                continue
            model = input("Ranking model [tfidf/bm25] (tfidf): ").strip().lower()
            model = model or "tfidf"
            if model not in {"tfidf", "bm25"}:
                print("Choose tfidf or bm25.")
                continue
            results = engine.search(query, top_k=10, model=model)
            if not results:
                print("No matching games found.")
                continue
            for rank, game in enumerate(results, start=1):
                print(f"{rank}. {game.get('title', '(untitled)')} ({game['app_id']})")
            selected = input("Select a result number, or press Enter to return: ").strip()
            if selected:
                try:
                    position = int(selected) - 1
                    if not 0 <= position < len(results):
                        raise ValueError
                except ValueError:
                    print("That result number is not in the list.")
                    continue
                game = games_by_id[int(results[position]["app_id"])]
                _show_game_details(game, database)
                _show_price_history(game, database, show_chart=True)
                _show_prediction(game, database, predictor, model="random_forest")
        elif choice == "2":
            game = _prompt_game(games_by_id)
            if game is not None:
                _show_game_details(game, database)
        elif choice == "3":
            game = _prompt_game(games_by_id)
            if game is not None:
                _show_price_history(game, database, show_chart=True)
        elif choice == "4":
            game = _prompt_game(games_by_id)
            if game is not None:
                model = input("Model [linear/random_forest] (random_forest): ").strip()
                _show_prediction(
                    game,
                    database,
                    predictor,
                    model=model or "random_forest",
                )
        elif choice == "5":
            print("Goodbye.")
            return
        else:
            print("Choose a menu option from 1 to 5.")


def _prompt_game(games_by_id: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    value = input("Enter game App ID: ").strip()
    try:
        appid = int(value)
    except ValueError:
        print("App ID must be an integer.")
        return None
    game = games_by_id.get(appid)
    if game is None:
        print(f"App ID {appid} is not in the local corpus.")
    return game


def _show_game_details(game: dict[str, Any], database: GameDatabase) -> None:
    appid = int(game["app_id"])
    history = database.get_price_history(appid)
    analytics = calculate_price_analytics(history)
    print(f"\n{game.get('title', '(untitled)')} (App ID {appid})")
    print(f"Release date: {game.get('release_date') or 'Unknown'}")
    print(f"Developer: {', '.join(game.get('developers') or []) or 'Unknown'}")
    print(f"Publisher: {', '.join(game.get('publishers') or []) or 'Unknown'}")
    print(f"Genres: {', '.join(game.get('genres') or []) or 'Unknown'}")
    print(f"Tags: {', '.join(game.get('tags') or []) or 'Unknown'}")
    print(f"Description: {game.get('description') or 'No description available.'}")
    if analytics["current_price"] is None:
        print("Current price: No price observations recorded")
        return
    currency = analytics["currency"] or "USD"
    print(
        f"Current observed price: {analytics['current_price']:.2f} {currency} "
        f"(observed {analytics['last_observed']})"
    )
    print(
        f"Historical lowest: {analytics['historical_lowest_price']:.2f} {currency}"
    )
    _print_analytics(analytics)


def _show_price_history(
    game: dict[str, Any],
    database: GameDatabase,
    *,
    show_chart: bool,
) -> None:
    appid = int(game["app_id"])
    history = database.get_price_history(appid)
    if not history:
        print("No price snapshots recorded yet. Run a crawl to start tracking.")
        return
    print(f"\nPrice history for {game.get('title', appid)}:")
    for row in history:
        print(
            f"{row['date']}  {row['price']:.2f} {row['currency']}  "
            f"discount {row['discount_percentage']:.0f}%"
        )
    if show_chart:
        plot_price_history(history, str(game.get("title", appid)), show=True)


def _show_prediction(
    game: dict[str, Any],
    database: GameDatabase,
    predictor: PricePredictor | None,
    *,
    model: str,
) -> None:
    if predictor is None:
        print(
            "Prediction unavailable: collect more price snapshots, including "
            "at least two sale-to-sale examples across the corpus."
        )
        return
    try:
        result = predictor.predict(
            database.get_price_history(int(game["app_id"])),
            model=model,
        )
    except ValueError as error:
        print(f"Prediction unavailable: {error}")
        return
    print(f"\nDiscount prediction for {game.get('title', game['app_id'])}:")
    if result["estimated_discount_percentage"] is None:
        print(result["note"])
        return
    print(f"Model: {result['model']}")
    print(f"Estimated next sale date: {result['estimated_next_sale_date']}")
    print(
        f"Estimated discount: {result['estimated_discount_percentage']:.1f}%"
    )
    probability = result["probability_sale_within_30_days"]
    if probability is None:
        print("Sale probability within 30 days: insufficient interval data")
    else:
        print(f"Sale probability within 30 days: {probability:.1%}")
    print(result["note"])


def _print_analytics(analytics: dict[str, Any]) -> None:
    average_discount = analytics["average_discount"]
    days_since = analytics["days_since_last_discount"]
    interval = analytics["average_sale_interval"]
    print(
        "Average discount: "
        + (f"{average_discount:.1f}%" if average_discount is not None else "N/A")
    )
    print(
        "Days since last discount: "
        + (str(days_since) if days_since is not None else "No discounts observed")
    )
    print(f"Discounts in past year: {analytics['discounts_in_past_year']}")
    print(
        "Average sale interval: "
        + (f"{interval:.0f} days" if interval is not None else "N/A")
    )


if __name__ == "__main__":
    raise SystemExit(main())
