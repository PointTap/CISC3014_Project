"""Smoke tests that ensure both retrieval models remain usable."""

import unittest

from app.search_engine import SearchEngine


class SearchEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SearchEngine(
            [
                {
                    "app_id": 1,
                    "title": "Fantasy RPG",
                    "description": "Explore an open world and battle bosses.",
                    "genres": ["RPG"],
                    "tags": ["Fantasy", "Open World"],
                },
                {
                    "app_id": 2,
                    "title": "Cozy Farm",
                    "description": "Grow crops and make friends.",
                    "genres": ["Casual"],
                    "tags": ["Farming"],
                },
            ]
        )

    def test_tfidf_and_bm25_retrieve_matching_game(self) -> None:
        for model in ("tfidf", "bm25"):
            results = self.engine.search(
                "open world fantasy RPG with bosses",
                top_k=1,
                model=model,
            )
            self.assertEqual(results[0]["app_id"], 1)

    def test_unknown_search_model_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.engine.search("RPG", model="unknown")

    def test_bm25_handles_an_empty_corpus(self) -> None:
        self.assertEqual(SearchEngine([]).search("RPG", model="bm25"), [])


if __name__ == "__main__":
    unittest.main()
