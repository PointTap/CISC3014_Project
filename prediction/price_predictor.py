"""Train price-discount regressors and estimate future sale timing.

Each training example uses only sales observed before its target sale. The
regressors estimate the next event's discount; its date is estimated separately
from observed intervals. The 30-day probability is an empirical conditional
frequency over observed sale intervals, not a classifier confidence score.
"""

from __future__ import annotations

from datetime import date, timedelta
from statistics import mean
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from database.analytics import extract_discount_events

FEATURE_NAMES = (
    "month",
    "day_of_year",
    "days_since_last_discount",
    "average_discount",
    "average_sale_interval",
    "previous_discount_percentage",
)


class InsufficientPriceHistoryError(ValueError):
    """Raised when there are not enough observed sale events to train."""


class PricePredictor:
    """Fit linear and random-forest models from observed discount episodes."""

    def __init__(self) -> None:
        self.linear_model = LinearRegression()
        self.random_forest_model = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
        )
        self.training_samples = 0
        self._fitted = False
        self._global_interval: float | None = None

    def fit(
        self,
        histories: Mapping[int, list[dict[str, Any]]],
    ) -> "PricePredictor":
        """Train on next-sale discount examples from every game in the corpus."""
        features: list[list[float]] = []
        targets: list[float] = []
        all_intervals: list[int] = []

        for history in histories.values():
            events = extract_discount_events(history)
            event_dates = [event["date"] for event in events]
            all_intervals.extend(
                (event_dates[index] - event_dates[index - 1]).days
                for index in range(1, len(event_dates))
            )
            for target_index in range(1, len(events)):
                prior = events[:target_index]
                target_event = events[target_index]
                interval = (target_event["date"] - prior[-1]["date"]).days
                prior_intervals = [
                    (prior[index]["date"] - prior[index - 1]["date"]).days
                    for index in range(1, len(prior))
                ]
                features.append(
                    _feature_vector(
                        target_event["date"],
                        interval,
                        mean(event["discount_percentage"] for event in prior),
                        mean(prior_intervals) if prior_intervals else interval,
                        prior[-1]["discount_percentage"],
                    )
                )
                targets.append(float(target_event["discount_percentage"]))

        self.training_samples = len(targets)
        self._global_interval = mean(all_intervals) if all_intervals else None
        if self.training_samples < 2:
            self._fitted = False
            raise InsufficientPriceHistoryError(
                "At least two historical sale-to-sale examples across the "
                "corpus are required to train the discount models"
            )

        matrix = np.asarray(features, dtype=float)
        target_values = np.asarray(targets, dtype=float)
        self.linear_model.fit(matrix, target_values)
        self.random_forest_model.fit(matrix, target_values)
        self._fitted = True
        return self

    def predict(
        self,
        history: list[dict[str, Any]],
        *,
        model: str = "random_forest",
        as_of: date | None = None,
    ) -> dict[str, Any]:
        """Estimate the next sale date, discount and empirical 30-day chance."""
        if not self._fitted:
            raise InsufficientPriceHistoryError(
                "Call fit() with a corpus containing enough sale history first"
            )
        if model not in {"linear", "random_forest"}:
            raise ValueError("model must be 'linear' or 'random_forest'")

        events = extract_discount_events(history)
        if not events:
            return {
                "model": model,
                "estimated_discount_percentage": None,
                "estimated_next_sale_date": None,
                "probability_sale_within_30_days": None,
                "note": "No discount event has been observed for this game.",
            }

        event_dates = [event["date"] for event in events]
        intervals = [
            (event_dates[index] - event_dates[index - 1]).days
            for index in range(1, len(event_dates))
        ]
        interval = mean(intervals) if intervals else self._global_interval
        if interval is None or interval <= 0:
            return {
                "model": model,
                "estimated_discount_percentage": None,
                "estimated_next_sale_date": None,
                "probability_sale_within_30_days": None,
                "note": "Not enough observed sale intervals to estimate timing.",
            }

        today = as_of or date.today()
        average_discount = mean(event["discount_percentage"] for event in events)
        previous_discount = float(events[-1]["discount_percentage"])
        predicted_date = events[-1]["date"] + timedelta(days=max(1, round(interval)))
        while predicted_date <= today:
            predicted_date += timedelta(days=max(1, round(interval)))

        feature_row = np.asarray(
            [
                _feature_vector(
                    predicted_date,
                    interval,
                    average_discount,
                    interval,
                    previous_discount,
                )
            ],
            dtype=float,
        )
        estimator = (
            self.linear_model if model == "linear" else self.random_forest_model
        )
        predicted_discount = float(estimator.predict(feature_row)[0])
        predicted_discount = min(100.0, max(0.0, predicted_discount))

        probability = _conditional_sale_probability(
            intervals,
            elapsed_days=max(0, (today - events[-1]["date"]).days),
            currently_discounted=(
                float(
                    max(history, key=lambda row: str(row["date"])).get(
                        "discount_percentage", 0.0
                    )
                    or 0.0
                )
                > 0
                if history
                else False
            ),
        )
        return {
            "model": model,
            "estimated_discount_percentage": predicted_discount,
            "estimated_next_sale_date": predicted_date.isoformat(),
            "probability_sale_within_30_days": probability,
            "training_samples": self.training_samples,
            "note": (
                "Forecasts are estimates from observed snapshots; they are not "
                "Steam or publisher announcements."
            ),
        }


def _feature_vector(
    target_date: date,
    days_since_last_discount: float,
    average_discount: float,
    average_sale_interval: float,
    previous_discount_percentage: float,
) -> list[float]:
    return [
        float(target_date.month),
        float(target_date.timetuple().tm_yday),
        float(days_since_last_discount),
        float(average_discount),
        float(average_sale_interval),
        float(previous_discount_percentage),
    ]


def _conditional_sale_probability(
    intervals: list[int],
    *,
    elapsed_days: int,
    currently_discounted: bool,
) -> float | None:
    if not intervals:
        return None
    if currently_discounted:
        return 0.0
    remaining = [interval for interval in intervals if interval > elapsed_days]
    if not remaining:
        return None
    due_within_window = sum(
        interval <= elapsed_days + 30 for interval in remaining
    )
    return due_within_window / len(remaining)
