# Steam Game Search

A small course-project search engine that ranks Steam games for natural-language
queries. It uses a TF-IDF vector-space model with cosine similarity as the
primary score, then applies optional rating and tag-match boosts.

## Architecture

```text
App IDs -> crawler -> Steam Store appdetails API -> parser -> data/raw/games.json
                                                         |
game JSON + query -> preprocessing -> TF-IDF index -> cosine + boosts -> Top-K
                                                         |
                                  relevance judgments -> evaluation metrics
```

The crawler searches Steam Store by title to discover App IDs, then uses Steam
Store appdetails JSON and Store pages for descriptions, genres, categories,
user-defined tags, and review summaries. It requests robots.txt before
crawling, refuses endpoints disallowed by robots rules, deduplicates requests
and App IDs, and enforces a delay between requests. Steam Store search usually
returns up to 10 matches per query. SteamDB chart discovery remains an optional
source, limited to the requested top 1000 entries (100 by default); no SteamDB login or cookies
are used.

## Data schema

`data/raw/games.json` is a JSON array with one object per game:

| Field | Type | Meaning |
|---|---|---|
| `app_id` | integer | Stable Steam application ID |
| `url` | string | Canonical Steam Store URL |
| `title`, `description` | string | Name and searchable description |
| `genres`, `categories`, `tags` | array of strings | Genres, Steam features, and user-defined tags |
| `reviews` | object | `positive`, `negative`, `total`, `rating` (0–1) |
| `release_date` | string | Steam's display value |
| `developers`, `publishers` | array of strings | Credits |
| `price` | object or null | Store currency, amount in minor units, and display text |
| `last_crawled` | string | UTC timestamp |

Unavailable review summaries are represented by `null` values, not fabricated
values. Review percentages are rounded to the integer percentage Steam displays.

## Setup

Use Python 3.10 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Crawl and search

Pass App IDs directly or read them from a newline-separated text file:

```powershell
python main.py crawl --app-ids 413150 367520 1091500
python main.py crawl --app-id-file app_ids.txt --delay 2
python main.py crawl --steamdb-top 100 --delay 2
python main.py crawl --store-search "cozy farming" --search-limit 10
python main.py crawl --tag-expand 413150 --depth 2 --max-games 100
python main.py search --query "cozy farming game similar to Stardew Valley" --top-k 10
```

The crawler writes and incrementally updates `data/raw/games.json`. It skips
already stored App IDs unless `--refresh` is used. Network access is required
for crawling; searching works offline after data has been collected. SteamDB
may restrict automated access; the crawler does not bypass access controls and
reports HTTP errors such as 403. The SteamDB chart must contain its current-
players column; if its page layout changes, chart discovery stops instead of
silently collecting a different list. Crawling 100 games is intentionally
slow due to the configured per-request delay.

Steam Store search is a public title-search endpoint, not a semantic search
over full descriptions. Use several topic queries (for example, `farming`,
`open world RPG`, and `story horror`) to gather candidates; the local TF-IDF
engine then searches their descriptions, genres, and tags for natural-language
queries. This keeps live web discovery separate from the course project's
information-retrieval ranking.

`--tag-expand` starts with a saved or newly fetched seed game (depth 0), searches
Steam Store by up to five of that game's tags, fetches the matching games, then
repeats from those games until the depth is reached. The crawl is iterative
breadth-first traversal, with duplicate App IDs and tag queries skipped.
`--max-games` caps the total games discovered in that run (including the seed);
`--per-tag-limit` and `--tags-per-game` bound request volume. Steam Store's
search endpoint searches titles rather than its full tag catalog, so tag
expansion uses each tag as a title-search term; the retrieved games' Store tags
are then included in the local TF-IDF index.
Searched tag results are cached in `<output>.tag-search.json` and reused in
later runs, so successful tag searches are not repeated and can still support
deeper expansion later. Use `--reset-tag-history` to discard the saved query
cache for a run. A tag query that hits a redirect loop is logged and cached as
failed instead of stopping the entire crawl. If fetching an individual game's
details hits a redirect loop, that App ID is skipped and later games continue;
failed games are excluded from the next expansion depth. The command prints
the local corpus size after each depth and at completion. Tag-state saves use unique
temporary files and retry briefly if Windows temporarily locks the state
file;
if the primary state file remains locked, the cache is saved to a `.recovery`
sidecar and merged automatically on the next run.

## Retrieval model

The searchable document concatenates title, description, genres, tags, and
Steam categories as an additional metadata field.
Preprocessing lowercases text, tokenizes it, removes English stopwords, and
lemmatizes terms with NLTK's WordNet lemmatizer. Pandas holds the game-record
table, NumPy combines scores, and scikit-learn builds TF-IDF vectors.
For query vector **q** and game vector **d**, cosine similarity is:

`cos(q, d) = (q · d) / (||q|| ||d||)`

The final score uses `0.8 * cosine + 0.1 * rating + 0.1 * tag_overlap`.
Both boosts are normalized to [0, 1]; rating is the positive-review fraction
and tag overlap is the fraction of processed query terms found in game tags.
Set either boost weight to zero in `ir/ranking.py` to use pure cosine ranking.

## Project modules and IR concepts

| Module | Responsibility and concept |
|---|---|
| `crawler/steamdb_chart.py` | Reads SteamDB's public current-player chart and discovers ranked App IDs. It does not provide historical pricing. |
| `crawler/steamdb_crawler.py` | Rate-limited, robots-aware requests to public Steam Store endpoints. |
| `crawler/parser.py` | Parses and normalizes HTML/API responses into game documents and price observations. |
| `database/storage.py` | SQLite repository for game metadata and dated price snapshots; one observation per game per day. |
| `database/analytics.py` | Computes historical low, mean discount, sale episodes, sale interval, and days since a discount. |
| `ir/preprocessing.py` | Tokenization, stopword removal, and WordNet lemmatization shared by queries and documents. |
| `ir/tfidf_index.py` | Builds sparse TF-IDF vectors; inverse document frequency reduces the weight of common terms. |
| `ir/ranking.py` | Uses cosine similarity to compare query/document vectors and produce ranked results. |
| `ir_withBM25/bm25_index.py` | BM25Okapi ranks term matches with term-frequency saturation and document-length normalization. |
| `ir/evaluation.py` | Precision@K, Recall@K, MAP, and NDCG measure ranked retrieval against manual judgments. |
| `prediction/price_predictor.py` | Trains linear regression and random-forest regressors on earlier discount events. |
| `visualization/price_history.py` | Matplotlib chart of observed price and discount percentages. |
| `app/search_engine.py` | Connects a local corpus and either TF-IDF or BM25 ranking to the application. |
| `main.py` | Crawl, search, database synchronization, and interactive CLI menu. |

WordNet is an NLTK data corpus and is installed separately from Python packages.
After installing requirements, run:

```powershell
python -m nltk.downloader wordnet
```

## Price snapshots and prediction limits

SteamDB chart discovery supplies App IDs. The crawler then reads public Steam
Store appdetails for current metadata and price; these endpoints do not expose
a complete historical price series. SQLite records the prices the project
actually observes, using the crawl date. Existing JSON records can be imported
with `python main.py sync-db`. Re-crawling over time builds the history; a
single import is not enough to train a meaningful predictor, and no older
prices are fabricated.

Prices are saved in the selected Store currency (the included crawler requests
the US Store). A sale episode begins when an observed discount changes from
zero to positive; multiple observations during one sale count as one sale.
The estimated next sale date is the most recent sale date plus the average
observed sale interval. The 30-day sale probability is an empirical
conditional frequency from a game's observed sale intervals, not a calibrated
classifier. Predictions are estimates, not publisher announcements.

## Interactive CLI

Start the five-option menu with `python main.py menu`. It supports:

1. Natural-language game search (choose TF-IDF or BM25), then details, price
   history, chart, and an available prediction for the selected game.
2. View metadata and calculated price analytics by App ID.
3. Print and plot recorded price observations by App ID.
4. Predict the next discount with linear regression or random forest.
5. Exit.

For example, `python main.py crawl --steamdb-top 100` discovers up to 100
current-player chart entries and updates both the JSON corpus and SQLite
database. Run repeated crawls on different dates to collect price history.

## Evaluation

`ir/evaluation.py` provides Precision@K, Recall@K, Average Precision, MAP, and
NDCG. Supply judged App IDs per query and the corresponding ranked App IDs.
The metrics use binary relevance (relevant / not relevant).

## Layout

```text
crawler/          polite HTTP client, SteamDB chart discovery, parsing
database/         SQLite persistence and price analytics
data/raw/         crawled game records (created on first crawl)
experiment/       relevance judgments and TF-IDF/BM25 evaluation scripts
ir/               preprocessing, TF-IDF, cosine ranking, evaluation
ir_withBM25/      BM25 and field-weighted BM25 indexes
prediction/       linear regression and random-forest price forecasts
visualization/    Matplotlib price-history charts
app/              search workflow and ranking-model selection
main.py           crawl, search, sync-db, and interactive CLI
```
