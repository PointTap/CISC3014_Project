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
source, limited to the requested top 1000 entries; no SteamDB login or cookies
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
python main.py crawl --steamdb-top 1000 --delay 2
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
silently collecting a different list. Crawling 1000 games is intentionally
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

The searchable document concatenates title, description, genres, and tags.
Preprocessing lowercases text, tokenizes it, removes English stopwords, and
stems tokens with NLTK's Porter stemmer. Pandas holds the game-record table,
NumPy combines scores, and scikit-learn builds TF-IDF vectors.
For query vector **q** and game vector **d**, cosine similarity is:

`cos(q, d) = (q · d) / (||q|| ||d||)`

The final score uses `0.8 * cosine + 0.1 * rating + 0.1 * tag_overlap`.
Both boosts are normalized to [0, 1]; rating is the positive-review fraction
and tag overlap is the fraction of processed query terms found in game tags.
Set either boost weight to zero in `ir/ranking.py` to use pure cosine ranking.

## Evaluation

`ir/evaluation.py` provides Precision@K, Recall@K, Average Precision, MAP, and
NDCG. Supply judged App IDs per query and the corresponding ranked App IDs.
The metrics use binary relevance (relevant / not relevant).

## Layout

```text
crawler/       polite HTTP client and response parser
crawler/tag_expansion.py  bounded tag-based corpus expansion
data/raw/      crawled game records (created on first crawl)
data/processed/ optional index artifacts
ir/            preprocessing, TF-IDF, ranking, evaluation
app/           search workflow
main.py        crawl and search CLI
scraper.py     convenience entry point for crawling
```
