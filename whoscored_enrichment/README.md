# WhoScored scraper and validated import

Visible Chrome successfully collected live 2026/27 data on **21 November 2026**.
Headless Chrome received HTTP 403. The scraper uses ordinary visible Chrome and
WhoScored’s own Detailed statistics feed; it does not reconstruct totals from rounded
per-game averages. Original experiment files and notebooks remain unchanged.

The completed run contains **2,326 player–club records**: Premier League 464, La Liga 494,
Bundesliga 425, Serie A 507 and Ligue 1 436. Eleven statistical columns are available.
Passing accuracy is missing for 12 records with no attempted passes; their count values
and identities remain in both exports. The 28 importer/scraper/browser tests passed.

## Run

From the enclosing `FootballAIProject` directory:

```sh
.venv/bin/python AI_Football_Project/scrape_whoscored_stats.py --league all
```

Google Chrome must be installed. To recreate the Python dependencies:

```sh
.venv/bin/python -m pip install -r AI_Football_Project/requirements-whoscored.txt
```

League keys: `epl`, `laliga`, `bundesliga`, `seriea`, `ligue1`, or `all`.
The browser uses a separate profile in `~/.cache/footballai/whoscored-browser`.
Your personal Chrome profile is not copied or inspected.

A successful all-league run writes:

- `data/players_advanced_whoscored_2026_2027_totals.csv`: identity, exposure and source totals/percentages.
- `data/players_advanced_whoscored_2026_2027_per90.csv`: the same records with count statistics divided by source minutes and multiplied by 90. Percentages remain percentages.
- `whoscored_enrichment/collection_manifest.json`: source directories, collection time and league coverage.

Each league also has a unique directory under `data/whoscored_sources/` containing
`raw_capture.json`, `players.csv`, `players_per90.csv` and `metadata.json`.
Raw captures preserve source rows, request URLs, unit-option mappings and pagination counts.
No player CSV is written from a blocked or incomplete capture. Existing completed captures
remain available when a later run fails.

## What is collected

| Export column | Source or calculation |
| --- | --- |
| `key_passes` | `keyPassesTotal` |
| `passes` | `passTotal` |
| `accurate_passes` | `shortPassAccurate + passLongBallAccurate` |
| `pass_completion_pct` | `100 * accurate_passes / passTotal`; missing when no passes were attempted |
| `long_balls` | `passLongBallAccurate + passLongBallInaccurate` |
| `successful_dribbles` | `dribbleWon` |
| `aerial_duels_won` | `duelAerialWon` |
| `clearances` | `clearanceTotal` |
| `blocks` | `outfielderBlock`, specifically the source’s shot-block count |
| `dispossessions` | `dispossessed` |
| `poor_controls` | `turnover`; retain this source-specific definition when interpreting it |

The four accurate/inaccurate short/long pass components must sum to `passTotal` before
passing accuracy is calculated. These are weighted season totals, not an average of
match-level percentages. Missing values are retained. Source zeros are observed zeros.
`apps` in the JSON feed is total appearances: `subOn` is not added again.
Goalkeepers and low-minute players are retained in these source exports; the primary
experiment’s exclusions are applied only at its preprocessing stage.

Ratings, source positions, height/weight, nationality and other profile fields from the
feed are not imported into these feature exports. xA is not claimed or inferred.

## How collection is verified

1. Open the league through its tournament route and select the visible **2026/2027** season.
2. Open Player Statistics and its **Detailed** view. Read its unit labels and option values.
   The live page mapped Per game to `0`, Per 90 mins to `1`, and Total to `2`; the collector
   reads the mapping rather than assuming that `1` means totals.
3. Request the Detailed view’s same-origin statistics feed with that unit parameter and
   the selected season/stage and tournament IDs. Collect Key passes, Passes, Dribbles,
   Aerial, Clearances, Blocks and Possession loss. Use all positions and appearances,
   overall location and the full 0–90 minute interval.
4. Validate every returned row’s season and tournament. Follow pagination until unique
   player/club records equal `totalResults`, including leagues with more than 500 records.
5. Join categories using source player and club IDs. Reject conflicting exposure totals,
   repeated records/pages, unrecognized schema and partial responses.

Requests are paced. HTTP 401, 403 or 429 stops collection and preserves diagnostics.
The script does not use proxy rotation, automation-hiding patches or challenge solvers.

## Browser bookmark exporter

Open `WhoScored_exporter.html` in your regular browser and drag its blue link to the
bookmarks bar. On an accessible WhoScored **league Player Statistics** page, select
2026/2027 and click that bookmark. It runs the same Detailed collector and downloads JSON.
It reads the public page/feed and does not send your data to an external service.

Convert the download with:

```sh
.venv/bin/python AI_Football_Project/scrape_whoscored_stats.py --from-capture /path/to/capture.json
```

Rebuild the bookmark after changing the collector:

```sh
.venv/bin/python AI_Football_Project/scrape_whoscored_stats.py --build-bookmarklet
```

Other useful options: `--units per90`, `--categories "Key passes,Passes"`,
`--interactive` for manual page/filter setup, `--url` for a specific league page,
and `--max-pages` to cap pagination. Reaching the cap is an incomplete capture.
`--mode dom` retains a rendered-table fallback; use it only when the page has a clearly
active Total or Per 90 control. The standard Summary/Passing tables are per-game averages
and are not an appropriate substitute for the Detailed totals.

## Reporting period and the existing experiment

**Collection is complete only for the current source snapshot; it is not an automatic
merge into the earlier experiment.** Source metadata initially has `cutoff_date: null`
and `import_ready: false`. Retrieval time is not the last included match date.

Your local original CSV matches Kaggle **version 10**, published **2 November 2026**:
SHA256 `a404aac3864ef5a1e51b68af7d8e0b66bb5da505276977aca4c9411f53833b69`.
It contains 2,295 records and no explicit last-match date. It is not silently replaced
with a later Kaggle version. Current WhoScored totals can include additional appearances.
Do not relabel a current table as an older reporting window merely to satisfy validation.

`import_whoscored_stats.py` requires actual source reporting-period evidence and exact
player/club/league, appearances and minute reconciliation. Only after verifying a compatible
cutoff, record that date and its evidence in the export metadata and run:

```sh
.venv/bin/python AI_Football_Project/import_whoscored_stats.py --csv /path/to/players.csv --metadata /path/to/metadata.json
```

Repeat both options to supply several leagues in one import. The importer preserves the
existing original/biometric files, previous features and 180-minute cohort. It writes new
files ending `_with_biometrics_whoscored.csv`, plus an audit and manifest. Such files are
not created when the source period cannot be verified. Exact exposure agreement corroborates
compatible observations; it is not independent proof of every underlying match.

## Validation

```sh
.venv/bin/python -m unittest discover -s AI_Football_Project/tests -p 'test_whoscored*.py' -v
```

Tests use explicitly synthetic tables/feed responses and intercept all browser requests.
They cover unit mapping, rate calculations, passing accuracy, appearances, pagination,
category joins, blocked/partial captures, the generated bookmark download and preservation
of the existing experiment. Live captures are checked separately and retained as evidence.
