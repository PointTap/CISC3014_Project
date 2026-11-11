"""BM25 index for Steam game retrieval."""

from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi

from ir.preprocessing import game_document, tokenize


class BM25Index:
    """Build a BM25 index using the same Steam documents as TF-IDF."""

    def __init__(self, games: list[dict[str, Any]]) -> None:
        self.games = games
        self.app_ids = [int(game["app_id"]) for game in games]

        # Keep preprocessing identical to the TF-IDF baseline so that
        # later evaluation compares retrieval models rather than different data.
        self.documents = [
            tokenize(game_document(game))
            for game in games
        ]

        self.bm25 = BM25Okapi(self.documents) if any(self.documents) else None

    def score_query(self, query: str):
        """Return a BM25 relevance score for every indexed game."""
        query_tokens = tokenize(query)

        if not query_tokens or self.bm25 is None:
            return []

        return self.bm25.get_scores(query_tokens)