"""SQLite storage for game metadata and observed Steam price snapshots.

The Store API exposes a current price, not a complete historical series. This
module therefore records an observation on the date each game was crawled.
Repeated writes for one game on the same day update that day's observation
instead of creating duplicates.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Generator, Iterable


class GameDatabase:
    """Small SQLite repository for game metadata and price observations."""

    def __init__(self, path: str | Path = "data/steam_games.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Commit successful work and always release the file handle."""
        connection = sqlite3.connect(self.path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the requested schema and a uniqueness constraint per day."""
        with self._connection() as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS games (
                    appid INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    genres TEXT NOT NULL DEFAULT '[]',
                    tags TEXT NOT NULL DEFAULT '[]',
                    release_date TEXT NOT NULL DEFAULT '',
                    developer TEXT NOT NULL DEFAULT '',
                    publisher TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    appid INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    price REAL NOT NULL,
                    discount_percentage REAL NOT NULL,
                    currency TEXT NOT NULL DEFAULT 'USD',
                    FOREIGN KEY (appid) REFERENCES games(appid) ON DELETE CASCADE,
                    UNIQUE (appid, date)
                );

                CREATE INDEX IF NOT EXISTS idx_price_history_app_date
                    ON price_history(appid, date);
                """
            )

    def upsert_game(
        self,
        game: dict[str, Any],
        *,
        record_price: bool = True,
        observed_on: date | str | None = None,
    ) -> None:
        """Insert or update one game and optionally save its observed price."""
        appid = int(game.get("app_id", game.get("appid", 0)))
        if appid <= 0:
            raise ValueError("A positive app_id is required")

        with self._connection() as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                """
                INSERT INTO games (
                    appid, title, description, genres, tags, release_date,
                    developer, publisher
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(appid) DO UPDATE SET
                    title = excluded.title,
                    description = excluded.description,
                    genres = excluded.genres,
                    tags = excluded.tags,
                    release_date = excluded.release_date,
                    developer = excluded.developer,
                    publisher = excluded.publisher
                """,
                (
                    appid,
                    str(game.get("title", "")),
                    str(game.get("description", "")),
                    _json_list(game.get("genres")),
                    _json_list(game.get("tags")),
                    str(game.get("release_date", "")),
                    _join_names(game.get("developers", game.get("developer"))),
                    _join_names(game.get("publishers", game.get("publisher"))),
                ),
            )

            observation = _price_observation(game)
            if record_price and observation is not None:
                date_text = _observation_date(game, observed_on)
                price, discount, currency = observation
                connection.execute(
                    """
                    INSERT INTO price_history (
                        appid, date, price, discount_percentage, currency
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(appid, date) DO UPDATE SET
                        price = excluded.price,
                        discount_percentage = excluded.discount_percentage,
                        currency = excluded.currency
                    """,
                    (appid, date_text, price, discount, currency),
                )

    def upsert_games(
        self,
        games: Iterable[dict[str, Any]],
        *,
        record_price: bool = True,
    ) -> None:
        """Persist a batch of records using each record's crawl date."""
        for game in games:
            self.upsert_game(game, record_price=record_price)

    def get_game(self, appid: int) -> dict[str, Any] | None:
        """Return one normalized metadata record, or ``None`` if absent."""
        with self._connection() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM games WHERE appid = ?", (int(appid),)
            ).fetchone()
        if row is None:
            return None
        game = dict(row)
        game["genres"] = json.loads(game["genres"])
        game["tags"] = json.loads(game["tags"])
        return game

    def get_price_history(self, appid: int) -> list[dict[str, Any]]:
        """Return chronological observations for one game."""
        with self._connection() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT date, price, discount_percentage, currency
                FROM price_history
                WHERE appid = ?
                ORDER BY date, id
                """,
                (int(appid),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_all_price_histories(self) -> dict[int, list[dict[str, Any]]]:
        """Return all histories grouped by App ID for model training."""
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT appid, date, price, discount_percentage, currency
                FROM price_history
                ORDER BY appid, date, id
                """
            ).fetchall()

        histories: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            appid = int(row["appid"])
            histories.setdefault(appid, []).append(
                {
                    "date": row["date"],
                    "price": row["price"],
                    "discount_percentage": row["discount_percentage"],
                    "currency": row["currency"],
                }
            )
        return histories


def _price_observation(
    game: dict[str, Any],
) -> tuple[float, float, str] | None:
    """Convert Steam's US-cent prices into USD and derive the discount."""
    price_data = game.get("price")
    if isinstance(price_data, dict):
        final = price_data.get("final")
        if isinstance(final, (int, float)):
            initial = price_data.get("initial", final)
            try:
                initial_value = max(0.0, float(initial))
            except (TypeError, ValueError):
                initial_value = float(final)
            final_value = max(0.0, float(final))
            discount = (
                max(0.0, min(100.0, 100.0 * (initial_value - final_value) / initial_value))
                if initial_value > 0
                else 0.0
            )
            return final_value / 100.0, discount, str(price_data.get("currency") or "USD")

    if game.get("is_free") is True:
        return 0.0, 0.0, "USD"
    return None


def _observation_date(
    game: dict[str, Any],
    observed_on: date | str | None,
) -> str:
    """Use a supplied date, otherwise the crawler's timestamp, then today."""
    value: date | str | None = observed_on
    if value is None:
        value = game.get("last_crawled")
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.date().isoformat()
        except ValueError:
            try:
                return date.fromisoformat(value[:10]).isoformat()
            except ValueError as error:
                raise ValueError(f"Invalid observation date: {value!r}") from error
    return datetime.now(timezone.utc).date().isoformat()


def _json_list(value: Any) -> str:
    """Serialize a list-like metadata field without losing Unicode text."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    return json.dumps([str(item) for item in values if item], ensure_ascii=False)


def _join_names(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return ""
