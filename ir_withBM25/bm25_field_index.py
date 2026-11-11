"""Field-weighted BM25 index for Steam game retrieval."""

from __future__ import annotations

from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from ir.preprocessing import tokenize


class FieldBM25Index:
    """BM25 indexes for separate Steam game metadata fields."""

    # Experimental field weights.
    # Structured metadata receives more weight because genres and tags
    # usually describe game relevance more directly than long descriptions.
    TITLE_WEIGHT = 1.5
    DESCRIPTION_WEIGHT = 1.0
    GENRE_WEIGHT = 2.0
    CATEGORY_WEIGHT = 1.0
    TAG_WEIGHT = 2.5

    def __init__(self, games: list[dict[str, Any]]) -> None:
        self.games = games
        self.app_ids = [int(game["app_id"]) for game in games]

        self.title_documents = [
            tokenize(str(game.get("title", "")))
            for game in games
        ]

        self.description_documents = [
            tokenize(str(game.get("description", "")))
            for game in games
        ]

        self.genre_documents = [
            tokenize(self._join_values(game.get("genres")))
            for game in games
        ]

        self.category_documents = [
            tokenize(self._join_values(game.get("categories")))
            for game in games
        ]

        self.tag_documents = [
            tokenize(self._join_values(game.get("tags")))
            for game in games
        ]

        self.title_bm25 = BM25Okapi(
            self.title_documents
        )

        self.description_bm25 = BM25Okapi(
            self.description_documents
        )

        self.genre_bm25 = BM25Okapi(
            self.genre_documents
        )

        self.category_bm25 = BM25Okapi(
            self.category_documents
        )

        self.tag_bm25 = BM25Okapi(
            self.tag_documents
        )

    def score_query(self, query: str) -> np.ndarray:
        """Return combined field-weighted BM25 relevance scores."""

        query_tokens = tokenize(query)

        if not query_tokens:
            return np.zeros(
                len(self.games),
                dtype=float,
            )

        title_scores = self.title_bm25.get_scores(
            query_tokens
        )

        description_scores = self.description_bm25.get_scores(
            query_tokens
        )

        genre_scores = self.genre_bm25.get_scores(
            query_tokens
        )

        category_scores = self.category_bm25.get_scores(
            query_tokens
        )

        tag_scores = self.tag_bm25.get_scores(
            query_tokens
        )

        # Combine the independent BM25 scores from each Steam field.
        combined_scores = (
            self.TITLE_WEIGHT * title_scores
            + self.DESCRIPTION_WEIGHT * description_scores
            + self.GENRE_WEIGHT * genre_scores
            + self.CATEGORY_WEIGHT * category_scores
            + self.TAG_WEIGHT * tag_scores
        )

        return combined_scores

    @staticmethod
    def _join_values(value: object) -> str:
        """Convert Steam metadata lists into searchable text."""

        if isinstance(value, str):
            return value

        if isinstance(value, list):
            return " ".join(
                str(item)
                for item in value
                if item
            )

        return ""