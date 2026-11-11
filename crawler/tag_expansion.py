"""Bounded breadth-first expansion of a game corpus through Steam tags."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

import requests

from crawler.steamdb_crawler import SteamCrawler

logger = logging.getLogger(__name__)


def expand_from_seed(
    seed_app_id: int,
    output_path: str | Path,
    crawler: SteamCrawler,
    depth: int = 2,
    per_tag_limit: int = 10,
    max_games: int = 100,
    tags_per_game: int = 5,
    refresh: bool = False,
    state_path: str | Path | None = None,
    reset_tag_history: bool = False,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, int]:
    """Discover games via each frontier game's tags up to a fixed depth.

    The seed has depth zero. Each following layer searches Steam Store for
    title matches to tags on the previous layer. `max_games` includes the seed.
    """
    if seed_app_id <= 0:
        raise ValueError("seed_app_id must be positive")
    if depth < 0:
        raise ValueError("depth must be non-negative")
    if per_tag_limit <= 0 or max_games <= 0 or tags_per_game <= 0:
        raise ValueError("per-tag limit, max games, and tags per game must be positive")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_file = Path(state_path) if state_path else path.with_suffix(
        path.suffix + ".tag-search.json"
    )
    state = {} if reset_tag_history else _load_state(state_file)
    searched_tags: dict[str, dict[str, Any]] = state.setdefault("searched_tags", {})
    if not isinstance(searched_tags, dict):
        raise ValueError(f"Invalid searched_tags in {state_file}")

    seed_downloads = crawler.crawl([seed_app_id], path, refresh=refresh)
    records = _load_records(path)
    games_by_id = {int(game["app_id"]): game for game in records}
    if seed_app_id not in games_by_id:
        raise ValueError(f"Seed App ID {seed_app_id} was not saved to {path}")
    if not games_by_id[seed_app_id].get("tags"):
        seed_downloads += crawler.crawl([seed_app_id], path, refresh=True)
        games_by_id = {
            int(game["app_id"]): game for game in _load_records(path)
        }

    reached = {seed_app_id}
    frontier = [seed_app_id]
    discovered_downloads = len(seed_downloads)
    depth_reached = 0

    for level in range(1, depth + 1):
        candidate_ids: list[int] = []
        for app_id in frontier:
            game_tags = games_by_id[app_id].get("tags")
            if not isinstance(game_tags, list):
                continue
            for tag in game_tags[:tags_per_game]:
                query = str(tag).strip()
                key = query.casefold()
                if not query or query == "+":
                    continue
                if len(reached) >= max_games:
                    break
                tag_result = searched_tags.get(key)
                if tag_result is None:
                    try:
                        matches = crawler.search_app_ids(
                            query, limit=per_tag_limit, allow_empty=True
                        )
                    except requests.exceptions.TooManyRedirects as error:
                        searched_tags[key] = {
                            "query": query,
                            "app_ids": [],
                            "error": str(error),
                        }
                        _save_state(state_file, state)
                        logger.warning(
                            "Skipping tag %r after too many redirects: %s",
                            query,
                            error,
                        )
                        continue
                    tag_result = {
                        "query": query,
                        "app_ids": [int(app_id) for app_id in matches],
                        "error": None,
                    }
                    searched_tags[key] = tag_result
                    _save_state(state_file, state)

                matches = tag_result.get("app_ids", [])
                for match_id in matches:
                    match_id = int(match_id)
                    if match_id <= 0 or match_id in reached:
                        continue
                    if len(reached) >= max_games:
                        break
                    reached.add(match_id)
                    candidate_ids.append(match_id)
                if len(reached) >= max_games:
                    break
            if len(reached) >= max_games:
                break

        if not candidate_ids:
            break
        fetched = crawler.crawl(candidate_ids, path, refresh=refresh)
        discovered_downloads += len(fetched)
        games_by_id.update(
            (int(game["app_id"]), game)
            for game in _load_records(path)
        )
        frontier = [app_id for app_id in candidate_ids if app_id in games_by_id]
        depth_reached = level
        if on_progress:
            on_progress(level, len(frontier), len(_load_records(path)))
        if not frontier:
            break

    return {
        "seed_app_id": seed_app_id,
        "depth_reached": depth_reached,
        "games_discovered": len(reached),
        "games_downloaded": discovered_downloads,
        "tags_searched": len(searched_tags),
        "tags_failed": sum(
            bool(result.get("error")) for result in searched_tags.values()
        ),
        "local_games": len(_load_records(path)),
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def _load_state(path: Path) -> dict[str, Any]:
    state = _read_state_file(path)
    recovery_path = _recovery_path(path)
    recovery_state = _read_state_file(recovery_path)
    merged_tags = dict(state.get("searched_tags", {}))
    merged_tags.update(recovery_state.get("searched_tags", {}))
    state.update(recovery_state)
    state["searched_tags"] = merged_tags
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    recovery_path = _recovery_path(path)
    if recovery_path.exists():
        _atomic_write_json(recovery_path, state)
        return

    try:
        _atomic_write_json(path, state)
    except PermissionError:
        try:
            _atomic_write_json(recovery_path, state)
        except PermissionError as recovery_error:
            raise PermissionError(
                f"Cannot save tag-search state to {path} or its recovery file "
                f"{recovery_path}. Close programs holding these files open and retry."
            ) from recovery_error
        logger.warning(
            "Could not replace %s because Windows denied access; saved tag "
            "search state to recovery file %s. It will be loaded automatically.",
            path,
            recovery_path,
        )


def _read_state_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"searched_tags": {}}
    with path.open(encoding="utf-8") as file:
        state = json.load(file)
    if not isinstance(state, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    searched_tags = state.get("searched_tags", {})
    if not isinstance(searched_tags, dict):
        raise ValueError(f"Invalid searched_tags in {path}")
    return state


def _recovery_path(path: Path) -> Path:
    return path.with_name(path.name + ".recovery")


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")

        for attempt in range(5):
            try:
                temporary_path.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2 * (attempt + 1))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
