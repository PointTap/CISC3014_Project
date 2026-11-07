"""Command-line entry point for crawling Steam games and searching locally."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.search_engine import SearchEngine
from crawler.steamdb_chart import DEFAULT_CHART_URL, SteamDBChartCrawler
from crawler.steamdb_crawler import SteamCrawler
from crawler.tag_expansion import expand_from_seed

DEFAULT_DATA = Path("data/raw/games.json")


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
        const=1000,
        type=int,
        metavar="COUNT",
        help="discover and crawl up to COUNT games from SteamDB's current-player chart (default: 1000)",
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
    crawl.add_argument("--delay", type=float, default=1.5)
    crawl.add_argument("--refresh", action="store_true")

    search = commands.add_parser("search", help="Search the locally crawled games")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--data", default=str(DEFAULT_DATA))
    return parser


def main(argv: list[str] | None = None) -> int:
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
        print(
            f"Fetched {len(games)} new game(s) from {len(app_ids)} selected App IDs; "
            f"data saved to {args.output}"
        )
        return 0

    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    engine = SearchEngine.from_json(args.data)
    results = engine.search(args.query, args.top_k)
    if not results:
        print("No matching games found.")
        return 0
    for rank, game in enumerate(results, start=1):
        print(
            f"{rank}. {game.get('title', '(untitled)')} "
            f"(App ID {game['app_id']}) - {game['final_score']:.4f} "
            f"[cosine {game['cosine_similarity']:.4f}]"
        )
        if game.get("url"):
            print(f"   {game['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
