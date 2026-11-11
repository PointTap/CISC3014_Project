"""Tokenize and lemmatize document, query, and tag text consistently."""

from __future__ import annotations

import re

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LEMMATIZER = WordNetLemmatizer()


def tokenize(text: str) -> list[str]:
    """Lowercase, tokenize, remove stopwords, then lemmatize English terms."""
    terms = _TOKEN_RE.findall(text.lower())
    normalized = []
    for term in terms:
        if term in ENGLISH_STOP_WORDS:
            continue
        try:
            verb_lemma = _LEMMATIZER.lemmatize(term, pos="v")
            lemma = (
                verb_lemma
                if verb_lemma != term
                else _LEMMATIZER.lemmatize(term, pos="n")
            )
        except LookupError as error:
            raise RuntimeError(
                "NLTK WordNet data is missing. Run "
                "'python -m nltk.downloader wordnet' before searching."
            ) from error
        normalized.append(lemma)
    return normalized


def preprocess_text(text: str) -> str:
    """Return a whitespace-delimited normalized string for the TF-IDF analyzer."""
    return " ".join(tokenize(text))


def game_document(game: dict) -> str:
    """Combine the fields whose content describes game relevance."""
    parts = [
        game.get("title", ""),
        game.get("description", ""),
        _join_values(game.get("genres")),
        _join_values(game.get("categories")),
        _join_values(game.get("tags")),
    ]
    return " ".join(str(part) for part in parts if part)


def _join_values(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    return ""
