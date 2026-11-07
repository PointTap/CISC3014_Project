"""Discover App IDs from SteamDB's current-player chart."""

from __future__ import annotations

import re
import time
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from crawler.steamdb_crawler import USER_AGENT

DEFAULT_CHART_URL = "https://steamdb.info/charts/"
_APP_PATH = re.compile(r"^/app/(\d+)(?:/|$)")


class SteamDBChartCrawler:
    """Fetch permitted chart pages and extract unique Steam application IDs."""

    def __init__(self, delay: float = 2.0, timeout: float = 20.0,
                 session: requests.Session | None = None) -> None:
        if delay < 0:
            raise ValueError("delay must be non-negative")
        self.delay = delay
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._robots: dict[str, RobotFileParser] = {}
        self._seen_urls: set[str] = set()
        self._last_request_at = 0.0

    def _robots_for(self, url: str) -> RobotFileParser:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            robots_url = f"{origin}/robots.txt"
            self._wait()
            response = self.session.get(robots_url, timeout=self.timeout)
            self._last_request_at = time.monotonic()
            response.raise_for_status()
            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(response.text.splitlines())
            self._robots[origin] = parser
        return self._robots[origin]

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

    def _fetch_page(self, url: str) -> str:
        if url in self._seen_urls:
            raise ValueError(f"Duplicate SteamDB chart URL refused: {url}")
        if not self._robots_for(url).can_fetch(USER_AGENT, url):
            raise PermissionError(f"robots.txt disallows crawling: {url}")
        self._wait()
        response = self.session.get(url, timeout=self.timeout)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        self._seen_urls.add(url)
        return response.text

    def fetch_top_app_ids(self, limit: int = 1000,
                          chart_url: str = DEFAULT_CHART_URL) -> list[int]:
        """Return App IDs in current-player rank order, up to `limit`."""
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")

        parsed_url = urlparse(chart_url)
        query = dict(parse_qsl(parsed_url.query, keep_blank_values=True))
        app_ids: list[int] = []
        seen_ids: set[int] = set()
        page = 1

        while len(app_ids) < limit:
            page_query = dict(query)
            if page > 1:
                page_query["page"] = str(page)
            page_url = urlunparse(parsed_url._replace(query=urlencode(page_query)))
            html = self._fetch_page(page_url)
            page_ids = _extract_chart_app_ids(html)
            if not page_ids:
                if page == 1:
                    raise ValueError(
                        "SteamDB response did not contain a current-player chart; "
                        "the page format may have changed or access may be restricted."
                    )
                break

            new_ids = [app_id for app_id in page_ids if app_id not in seen_ids]
            if not new_ids:
                break
            for app_id in new_ids:
                if len(app_ids) == limit:
                    break
                app_ids.append(app_id)
                seen_ids.add(app_id)
            page += 1

        return app_ids


def _extract_chart_app_ids(html: str) -> list[int]:
    soup = BeautifulSoup(html, "html.parser")
    chart_table = None
    for table in soup.find_all("table"):
        headers = " ".join(
            header.get_text(" ", strip=True).lower()
            for header in table.find_all(["th"])
        )
        if "current players" in headers or "players" in headers:
            chart_table = table
            break
    if chart_table is None:
        return []

    app_ids: list[int] = []
    for row in chart_table.find_all("tr"):
        for anchor in row.find_all("a", href=True):
            match = _APP_PATH.match(urlparse(anchor["href"]).path)
            if match:
                app_ids.append(int(match.group(1)))
                break
    return app_ids
