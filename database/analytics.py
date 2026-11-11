"""Price-history statistics derived only from recorded observations.

Discount episodes are counted on transitions into a discounted period rather
than once per daily snapshot. This avoids treating a week-long sale as seven
independent sales.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from statistics import mean
from typing import Any, Iterable


def extract_discount_events(
    history: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse consecutive discounted snapshots into sale episodes."""
    observations = _ordered_history(history)
    events: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None

    for observation in observations:
        discount = _discount(observation.get("discount_percentage"))
        observed_date = _parse_date(observation["date"])
        if discount > 0:
            if active is None:
                active = {
                    "date": observed_date,
                    "last_date": observed_date,
                    "discount_percentage": discount,
                }
            else:
                active["last_date"] = observed_date
                active["discount_percentage"] = max(
                    active["discount_percentage"], discount
                )
        elif active is not None:
            events.append(active)
            active = None

    if active is not None:
        events.append(active)
    return events


def calculate_price_analytics(
    history: Iterable[dict[str, Any]],
    *,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Calculate current/lowest prices and sale-frequency statistics."""
    today = as_of or date.today()
    observations = [
        row for row in _ordered_history(history)
        if _parse_date(row["date"]) <= today
    ]
    if not observations:
        return {
            "current_price": None,
            "historical_lowest_price": None,
            "average_discount": None,
            "days_since_last_discount": None,
            "discounts_in_past_year": 0,
            "average_sale_interval": None,
            "last_observed": None,
            "currency": None,
        }

    events = extract_discount_events(observations)
    event_dates = [event["date"] for event in events]
    intervals = [
        (event_dates[index] - event_dates[index - 1]).days
        for index in range(1, len(event_dates))
    ]
    latest = observations[-1]
    latest_date = _parse_date(latest["date"])
    last_discount_date = events[-1]["last_date"] if events else None
    recent_cutoff = today - timedelta(days=365)

    return {
        "current_price": float(latest["price"]),
        "historical_lowest_price": min(float(row["price"]) for row in observations),
        "average_discount": (
            mean(event["discount_percentage"] for event in events)
            if events
            else 0.0
        ),
        "days_since_last_discount": (
            max(0, (today - last_discount_date).days)
            if last_discount_date is not None
            else None
        ),
        "discounts_in_past_year": sum(
            recent_cutoff <= event["date"] <= today for event in events
        ),
        "average_sale_interval": mean(intervals) if intervals else None,
        "last_observed": latest_date.isoformat(),
        "currency": str(latest.get("currency") or "USD"),
    }


def _ordered_history(
    history: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = list(history)
    for row in rows:
        if "date" not in row or "price" not in row:
            raise ValueError("Each price observation needs date and price fields")
    return sorted(rows, key=lambda row: _parse_date(row["date"]))


def _parse_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return date.fromisoformat(value[:10])
            except ValueError as error:
                raise ValueError(f"Invalid price-history date: {value!r}") from error
    raise ValueError(f"Invalid price-history date: {value!r}")


def _discount(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0
