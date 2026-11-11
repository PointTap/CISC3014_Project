"""Robots-aware, rate-limited crawler for Steam Store appdetails."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Iterable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from crawler.parser import parse_game

logger = logging.getLogger(__name__)

STORE_API = "https://store.steampowered.com/api/appdetails/"
STORE_SEARCH_API = "https://store.steampowered.com/api/storesearch/"
USER_AGENT = "SteamGameSearchCourseProject/1.0 (polite educational crawler)"


class SteamCrawler:
    """Fetch one appdetails record at a time while respecting robots and delay."""

    def __init__(self, delay: float = 1.5, timeout: float = 20.0,
                 session: requests.Session | None = None) -> None:
        if delay < 0:
            raise ValueError("delay must be non-negative")
        self.delay = delay
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request_at = 0.0
        self._seen_urls: set[str] = set()

    def _robots_for(self, url: str) -> RobotFileParser:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            robots_url = f"{origin}/robots.txt"
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            response = self.session.get(robots_url, timeout=self.timeout)
            self._last_request_at = time.monotonic()
            response.raise_for_status()
            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(response.text.splitlines())
            self._robots[origin] = parser
        return self._robots[origin]

    def _get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        request = requests.Request("GET", url, params=params).prepare()
        request_url = request.url
        if request_url in self._seen_urls:
            raise ValueError(f"Duplicate URL refused: {request_url}")
        if not self._robots_for(request_url).can_fetch(USER_AGENT, request_url):
            raise PermissionError(f"robots.txt disallows crawling: {request_url}")
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        response = self.session.get(url, params=params, timeout=self.timeout)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        self._seen_urls.add(request_url)
        return response

    def fetch_game(self, app_id: int) -> dict[str, Any]:
        if app_id <= 0:
            raise ValueError(f"App ID must be positive: {app_id}")
        response = self._get(STORE_API, {"appids": app_id, "cc": "us", "l": "en"})
        payload = response.json()
        app = payload.get(str(app_id), {})
        if not app.get("success") or not isinstance(app.get("data"), dict):
            raise ValueError(f"Steam Store returned no details for App ID {app_id}")
        store_page_url = f"https://store.steampowered.com/app/{app_id}/"
        store_page_response = self._get(
            store_page_url, {"cc": "us", "l": "en"}
        )
        return parse_game(app_id, app["data"], store_html=store_page_response.text)

    def search_app_ids(self, query: str, limit: int = 10,
                       allow_empty: bool = False) -> list[int]:
        """Find game App IDs through Steam Store's public title-search endpoint."""
        if not query.strip():
            raise ValueError("Search query must not be empty")
        if limit <= 0:
            raise ValueError("Search result limit must be positive")
        response = self._get(
            STORE_SEARCH_API,
            {"term": query, "l": "english", "cc": "US"},
        )
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ValueError("Steam Store search returned an invalid response")
        app_ids = []
        for item in payload["items"]:
            if not isinstance(item, dict) or item.get("type") != "app":
                continue
            try:
                app_id = int(item["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if app_id > 0 and app_id not in app_ids:
                app_ids.append(app_id)
            if len(app_ids) >= limit:
                break
        if not app_ids and not allow_empty:
            raise ValueError(f"Steam Store search returned no games for: {query}")
        return app_ids

    def crawl(self, app_ids: Iterable[int], output_path: str | Path,
              refresh: bool = False) -> list[dict[str, Any]]:
        """Crawl unique IDs and persist valid records after each successful fetch."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = _load_records(path)
        by_id = {int(row["app_id"]): row for row in existing if "app_id" in row}
        results: list[dict[str, Any]] = []

        for app_id in dict.fromkeys(int(value) for value in app_ids):
            if app_id in by_id and not refresh:
                continue
            try:
                game = self.fetch_game(app_id)
            except requests.exceptions.TooManyRedirects as error:
                logger.warning(
                    "Skipping App ID %s after too many redirects: %s",
                    app_id,
                    error,
                )
                continue
            by_id[app_id] = game
            _save_records(path, by_id.values())
            results.append(game)
        return results


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def _save_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(sorted(records, key=lambda row: row["app_id"]), file,
                  ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)
