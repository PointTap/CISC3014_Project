"""Load game records, create an in-memory index, and serve ranked searches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ir.ranking import rank_games
from ir.tfidf_index import TfidfIndex


class SearchEngine:
    """Application workflow: load -> preprocess/index -> query -> rank."""

    def __init__(self, games: list[dict[str, Any]]) -> None:
        game_table = pd.DataFrame.from_records(games)
        self.index = TfidfIndex(game_table.to_dict(orient="records"))

    @classmethod
    def from_json(cls, path: str | Path) -> "SearchEngine":
        with Path(path).open(encoding="utf-8") as file:
            games = json.load(file)
        if not isinstance(games, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return cls(games)

    def search(
        self,
        query: str,
        top_k: int = 10,
        model: str = "tfidf",
    ) -> list[dict[str, Any]]:
        """Search with TF-IDF/cosine or BM25 over the same game documents."""
        if model == "tfidf":
            return rank_games(query, self.index, top_k)
        if model == "bm25":
            from ir_withBM25.bm25_index import BM25Index
            from ir_withBM25.bm25_ranking import rank_games_bm25

            if not hasattr(self, "_bm25_index"):
                self._bm25_index = BM25Index(self.index.games)
            return rank_games_bm25(query, self._bm25_index, top_k)
        raise ValueError("model must be 'tfidf' or 'bm25'")
