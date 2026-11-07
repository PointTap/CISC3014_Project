"""Normalize Steam Store API and game-page data into searchable records."""

from __future__ import annotations

from datetime import datetime, timezone
from html import unescape
import re
from typing import Any

from bs4 import BeautifulSoup


def _plain_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return BeautifulSoup(unescape(value), "html.parser").get_text(" ", strip=True)


def parse_game(app_id: int, store_data: dict[str, Any],
               store_html: str = "") -> dict[str, Any]:
    """Convert Steam Store appdetails JSON and page markup to the game schema."""
    app_id = int(app_id)
    genres = [
        item["description"]
        for item in store_data.get("genres", [])
        if isinstance(item, dict) and item.get("description")
    ]
    categories = [
        item["description"]
        for item in store_data.get("categories", [])
        if isinstance(item, dict) and item.get("description")
    ]
    tags, reviews = _parse_store_page(store_html)
    if reviews is None:
        reviews = {
            "positive": None,
            "negative": None,
            "total": None,
            "rating": None,
        }

    price_data = store_data.get("price_overview")
    price = None
    if isinstance(price_data, dict):
        price = {
            "currency": price_data.get("currency", ""),
            "initial": price_data.get("initial", 0),
            "final": price_data.get("final", 0),
            "final_formatted": price_data.get("final_formatted", ""),
        }

    release = store_data.get("release_date")
    release_date = release.get("date", "") if isinstance(release, dict) else ""
    title = str(store_data.get("name", "")).strip()

    return {
        "app_id": app_id,
        "url": f"https://store.steampowered.com/app/{app_id}/",
        "title": title,
        "description": _plain_text(
            store_data.get("detailed_description")
            or store_data.get("short_description", "")
        ),
        "genres": genres,
        "categories": categories,
        "tags": tags,
        "reviews": reviews,
        "release_date": str(release_date),
        "developers": _string_list(store_data.get("developers")),
        "publishers": _string_list(store_data.get("publishers")),
        "price": price,
        "last_crawled": datetime.now(timezone.utc).isoformat(),
    }


def _parse_store_page(html: str) -> tuple[list[str], dict[str, Any] | None]:
    if not html:
        return [], None
    soup = BeautifulSoup(html, "html.parser")
    tags = list(dict.fromkeys(
        element.get_text(" ", strip=True)
        for element in soup.select(".app_tag")
        if element.get_text(" ", strip=True)
    ))

    summary = soup.select_one("#appReviewsAll_responsive")
    if summary is None:
        summary = next(
            (
                row for row in soup.select(".user_reviews_summary_row")
                if "All Reviews" in row.get_text(" ", strip=True)
            ),
            None,
        )
    if summary is None:
        return tags, None

    match = re.search(
        r"(\d{1,3})%\s+of\s+([\d,]+)",
        summary.get_text(" ", strip=True),
    )
    if match is None:
        return tags, None
    rating_percent = int(match.group(1))
    total = _nonnegative_int(match.group(2).replace(",", ""))
    if total == 0 or rating_percent > 100:
        return tags, None
    positive = round(total * rating_percent / 100)
    return tags, {
        "positive": positive,
        "negative": total - positive,
        "total": total,
        "rating": positive / total,
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
