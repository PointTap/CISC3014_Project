"""TF-IDF vector-space index over game description documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from ir.preprocessing import game_document, preprocess_text


class TfidfIndex:
    """Map each App ID to its L2-normalized TF-IDF document vector."""

    def __init__(self, games: list[dict[str, Any]]) -> None:
        self.games = games
        self.app_ids = [int(game["app_id"]) for game in games]
        self.vectorizer = TfidfVectorizer(
            preprocessor=preprocess_text,
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
            norm="l2",
        )
        documents = [game_document(game) for game in games]
        normalized_documents = [preprocess_text(document) for document in documents]
        self.matrix = (
            self.vectorizer.fit_transform(documents)
            if any(normalized_documents)
            else None
        )

    def transform_query(self, query: str):
        """Convert a natural-language query into the corpus vocabulary space."""
        if self.matrix is None:
            return None
        return self.vectorizer.transform([query])

    def save(self, path: str | Path) -> None:
        """Persist the fitted model and vectors for reuse."""
        import joblib

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "games": self.games,
                "app_ids": self.app_ids,
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TfidfIndex":
        """Restore a previously saved fitted index."""
        import joblib

        payload = joblib.load(path)
        index = cls.__new__(cls)
        index.games = payload["games"]
        index.app_ids = payload["app_ids"]
        index.vectorizer = payload["vectorizer"]
        index.matrix = payload["matrix"]
        return index
