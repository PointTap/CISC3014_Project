"""Tests for persisted price observations and derived sale statistics."""

from datetime import date
import tempfile
import unittest
from pathlib import Path

from database.analytics import calculate_price_analytics, extract_discount_events
from database.storage import GameDatabase


class GameDatabaseTests(unittest.TestCase):
    def test_upsert_records_one_price_per_game_per_day(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = GameDatabase(Path(directory) / "games.db")
            game = {
                "app_id": 42,
                "title": "Example",
                "price": {
                    "currency": "USD",
                    "initial": 1000,
                    "final": 750,
                },
                "last_crawled": "2025-01-02T12:00:00+00:00",
            }
            database.upsert_game(game)
            game["price"]["final"] = 500
            database.upsert_game(game)

            history = database.get_price_history(42)
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["date"], "2025-01-02")
            self.assertEqual(history[0]["price"], 5.0)
            self.assertEqual(history[0]["discount_percentage"], 50.0)
            self.assertEqual(database.get_game(42)["title"], "Example")

    def test_free_game_records_zero_price_when_observed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = GameDatabase(Path(directory) / "games.db")
            database.upsert_game(
                {
                    "app_id": 5,
                    "title": "Free",
                    "is_free": True,
                    "last_crawled": "2025-01-03T00:00:00+00:00",
                }
            )
            self.assertEqual(database.get_price_history(5)[0]["price"], 0.0)

    def test_metadata_import_can_skip_price_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = GameDatabase(Path(directory) / "games.db")
            database.upsert_game(
                {
                    "app_id": 6,
                    "title": "Metadata only",
                    "price": {"initial": 1000, "final": 800},
                },
                record_price=False,
            )
            self.assertEqual(database.get_price_history(6), [])


class PriceAnalyticsTests(unittest.TestCase):
    def test_consecutive_discount_snapshots_are_one_event(self) -> None:
        history = [
            {"date": "2025-01-01", "price": 8, "discount_percentage": 20},
            {"date": "2025-01-02", "price": 8, "discount_percentage": 20},
            {"date": "2025-01-03", "price": 10, "discount_percentage": 0},
            {"date": "2025-01-31", "price": 6, "discount_percentage": 40},
            {"date": "2025-02-01", "price": 10, "discount_percentage": 0},
        ]
        events = extract_discount_events(history)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["date"], date(2025, 1, 1))
        self.assertEqual(events[0]["discount_percentage"], 20)
        stats = calculate_price_analytics(history, as_of=date(2025, 2, 1))
        self.assertEqual(stats["current_price"], 10)
        self.assertEqual(stats["historical_lowest_price"], 6)
        self.assertEqual(stats["average_discount"], 30)
        self.assertEqual(stats["average_sale_interval"], 30)
        self.assertEqual(stats["days_since_last_discount"], 1)
        self.assertEqual(stats["discounts_in_past_year"], 2)

    def test_empty_history_returns_explicit_missing_values(self) -> None:
        stats = calculate_price_analytics([])
        self.assertIsNone(stats["current_price"])
        self.assertIsNone(stats["average_discount"])
        self.assertEqual(stats["discounts_in_past_year"], 0)


if __name__ == "__main__":
    unittest.main()
