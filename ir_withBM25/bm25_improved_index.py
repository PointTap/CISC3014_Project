"""Improved field-weighted BM25 with query-term coverage."""

from __future__ import annotations

from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from ir.preprocessing import game_document, tokenize


class ImprovedBM25Index:
    """Field-weighted BM25 with a query coverage adjustment."""

    TITLE_WEIGHT = 1.5
    DESCRIPTION_WEIGHT = 1.0
    GENRE_WEIGHT = 2.0
    CATEGORY_WEIGHT = 1.0
    TAG_WEIGHT = 2.5

    COVERAGE_BASE = 0.5
    COVERAGE_WEIGHT = 0.5

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

        # Full-document token sets are used only for query coverage.
        self.document_term_sets = [
            set(tokenize(game_document(game)))
            for game in games
        ]

        self.title_bm25 = BM25Okapi(self.title_documents)
        self.description_bm25 = BM25Okapi(self.description_documents)
        self.genre_bm25 = BM25Okapi(self.genre_documents)
        self.category_bm25 = BM25Okapi(self.category_documents)
        self.tag_bm25 = BM25Okapi(self.tag_documents)

    def score_query(self, query: str) -> np.ndarray:
        """Return field-weighted BM25 scores adjusted by query coverage."""

        query_tokens = tokenize(query)

        if not query_tokens:
            return np.zeros(len(self.games), dtype=float)

        title_scores = self.title_bm25.get_scores(query_tokens)
        description_scores = self.description_bm25.get_scores(query_tokens)
        genre_scores = self.genre_bm25.get_scores(query_tokens)
        category_scores = self.category_bm25.get_scores(query_tokens)
        tag_scores = self.tag_bm25.get_scores(query_tokens)

        field_scores = (
            self.TITLE_WEIGHT * title_scores
            + self.DESCRIPTION_WEIGHT * description_scores
            + self.GENRE_WEIGHT * genre_scores
            + self.CATEGORY_WEIGHT * category_scores
            + self.TAG_WEIGHT * tag_scores
        )

        query_term_set = set(query_tokens)

        coverages = np.asarray(
            [
                len(query_term_set & document_terms) / len(query_term_set)
                for document_terms in self.document_term_sets
            ],
            dtype=float,
        )

        coverage_multiplier = (
            self.COVERAGE_BASE
            + self.COVERAGE_WEIGHT * coverages
        )

        return field_scores * coverage_multiplier

    @staticmethod
    def _join_values(value: object) -> str:
        """Convert Steam metadata fields into searchable text."""

        if isinstance(value, str):
            return value

        if isinstance(value, list):
            return " ".join(
                str(item)
                for item in value
                if item
            )

        return ""