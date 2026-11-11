"""Tests for sale-event features and both discount regression models."""

from datetime import date
import unittest

from prediction.price_predictor import PricePredictor


def _history(discounts: list[tuple[str, float]]) -> list[dict[str, object]]:
    rows = []
    for day, percentage in discounts:
        rows.append(
            {
                "date": day,
                "price": 20 * (1 - percentage / 100),
                "discount_percentage": percentage,
                "currency": "USD",
            }
        )
        if percentage:
            next_day = date.fromisoformat(day).toordinal() + 1
            rows.append(
                {
                    "date": date.fromordinal(next_day).isoformat(),
                    "price": 20,
                    "discount_percentage": 0,
                    "currency": "USD",
                }
            )
    return rows


class PricePredictorTests(unittest.TestCase):
    def test_fits_and_predicts_with_both_regressors(self) -> None:
        histories = {
            1: _history(
                [
                    ("2024-01-01", 20),
                    ("2024-02-01", 0),
                    ("2024-03-01", 30),
                    ("2024-04-01", 0),
                    ("2024-05-01", 40),
                ]
            ),
            2: _history(
                [
                    ("2024-01-05", 15),
                    ("2024-02-05", 0),
                    ("2024-03-05", 25),
                    ("2024-04-05", 0),
                    ("2024-05-05", 35),
                ]
            ),
        }
        predictor = PricePredictor().fit(histories)
        self.assertGreaterEqual(predictor.training_samples, 2)

        for model in ("linear", "random_forest"):
            result = predictor.predict(
                histories[1],
                model=model,
                as_of=date(2024, 5, 10),
            )
            self.assertEqual(result["model"], model)
            self.assertEqual(result["estimated_next_sale_date"], "2024-06-30")
            self.assertGreaterEqual(result["estimated_discount_percentage"], 0)
            self.assertLessEqual(result["estimated_discount_percentage"], 100)
            self.assertIsNotNone(result["probability_sale_within_30_days"])

    def test_no_sale_event_returns_no_game_specific_forecast(self) -> None:
        histories = {
            1: _history(
                [
                    ("2024-01-01", 20),
                    ("2024-02-01", 0),
                    ("2024-03-01", 30),
                    ("2024-04-01", 0),
                ]
            ),
            2: _history(
                [
                    ("2024-01-01", 10),
                    ("2024-02-01", 0),
                    ("2024-03-01", 20),
                    ("2024-04-01", 0),
                ]
            ),
        }
        predictor = PricePredictor().fit(histories)
        result = predictor.predict(
            [{"date": "2024-01-01", "price": 10, "discount_percentage": 0}]
        )
        self.assertIsNone(result["estimated_next_sale_date"])
        self.assertIsNone(result["estimated_discount_percentage"])


if __name__ == "__main__":
    unittest.main()
