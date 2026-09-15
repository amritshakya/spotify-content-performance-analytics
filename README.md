# Spotify Content Performance Analytics

A reproducible, professional-tooling analytics project built on a Spotify
track-catalog extract, focused on data-quality engineering, a documented
metric layer, and SQL-based descriptive analysis of content, artist, genre,
and release-period characteristics.

**Status: Phase 2 (data foundation + SQL analytics layer) complete,
including `sql/02_kpi_summary.sql` and `sql/03_segment_analysis.sql`.** No
modeling, clustering, or dashboards have been built — see
[Planned Modeling](#planned-modeling).

## Why This Project Exists

This project extends an earlier CS251 coursework exploratory analysis
(`../cs251-data-analysis-projects/project02`, preserved unchanged) into a
separate, independent professional analytics project built with
industry-standard tooling.

- The original coursework implemented a custom NumPy-based `Data`/`Analysis`
  class hierarchy from scratch and used it to perform categorical-variable
  exploratory data analysis (counting, filtering, bar charts) on the same
  Spotify extract, as a class assignment on building data-structure and
  EDA fundamentals.
- This repository does not reuse that class hierarchy. It is a fresh
  pipeline built with pandas, DuckDB, and pytest, organized as a
  reproducible loader → cleaning → SQL-analytics workflow with real tests
  and documented metric definitions.
- The original CS251 repository is untouched by this work: no files there
  were modified, and its `git status` was verified clean both before and
  after this project was built.

## Business / Analytical Question

> How do content, artist, genre, and release characteristics differ across
> higher- and lower-popularity tracks in this dataset, and how do those
> relationships vary across segments and release periods?

This is observational catalog data. **This project does not make causal
claims.** An observed association between a track/artist/genre/release
characteristic and `popularity_score` is never described as that
characteristic *causing* popularity, and this is not framed as paid-media,
user-acquisition, or growth-marketing analytics — it is descriptive analysis
of a fixed data extract.

## Dataset

| | |
|---|---|
| Shape (raw) | 277,938 rows × 34 columns |
| Shape (primary analytical dataset) | 277,937 rows (1 exact duplicate removed) |
| Shape (sensitivity dataset) | 277,846 rows (one row per `(track, artist_1)`) |
| Source file | `data/raw/spotify.csv` (gitignored, 47 MB) |
| SHA-256 | `4045adda473d6cd4fde6f3a11253faf37d7ee26fd122bcfd5c8a6c1fc5441974` |

**Key variables:** `track`, `artist_1`, `artist_1_pop`, `artist_1_genre_1..3`,
`release_date`, `popularity_score` (0–100, Spotify's algorithmic popularity
score), `popularity` (course-provided 3-level banding of `popularity_score`
— see below), `album_type`, `tempo`, `duration_ms`, and course-provided
3-level bins of several Spotify audio features (`danceability`, `energy`,
`speechiness`, `acousticness`, `instrumentalness`, `liveness`).

**Provenance status:** the CS251 course materials describe this dataset as
*"collected from the Spotify music service ... obtained using its web API."*
That is the course's own description provided alongside the file — it has
not been independently verified by this project (no API request log or
collection script accompanies the extract). See `data/README.md` for the
full provenance statement and known sampling limitations.

**Unknown sampling frame:** there is no evidence this ~278K-row extract is a
random or representative sample of Spotify's catalog. Every
percentage/share figure in this project is scoped to "tracks in this
dataset," never generalized to "Spotify tracks" broadly.

**Unknown collection time:** `popularity_score` is a snapshot at whatever
moment Spotify was queried to assemble this extract. That moment is not
recorded in the data and is **not** inferred from filesystem timestamps
(those reflect when the course copy was written to disk, not when Spotify
was queried).

**Important missingness:** `artist_2` and `artist_3` (artist *name* fields)
are **100% missing** in this extract; `artist_1_genre_1` is missing for
60.3% of tracks, and genre missingness is not random with respect to
popularity or release period (see Measurement Caveats below).

**`popularity_score` interpretation:** an integer 0–100 Spotify popularity
score. The course-provided categorical `popularity` column (`0low` /
`1moderate` / `2high`) was found to be an exact banding of it —
`0low = 0–33`, `1moderate = 34–66`, `2high = 67–100` — recovered empirically
by cross-tabulation, not assumed (see `reports/data_quality_report.md` §16).

## Measurement Caveats

These constraints propagate through every analysis, notebook, SQL query, and
metric definition in this project:

- **Snapshot nature of popularity.** `popularity_score` reflects play volume
  and recency of plays *as of the dataset's unknown collection time*. It is
  never described as a track's "current" popularity in the reader's present,
  nor as a stable, time-invariant property of the track.
- **Release-period recency/survivorship.** `release_year` / `release_date`
  are entangled with recency and survivorship: an old track still present in
  this dataset, with whatever popularity score it has, is not a random
  sample of everything released in that decade — it's whatever subset
  survived catalog curation and is still being played enough to register a
  score. Decade comparisons are phrased as "current popularity of tracks
  released in each decade" or "popularity at the dataset's collection time,"
  never as "performance by decade."
- **Sampling-frame limitation.** See Dataset section above — no share or
  percentage is generalized beyond "tracks in this dataset."
- **`artist_1_pop` / track popularity entanglement.** `artist_1_pop` is
  itself a Spotify-computed popularity signal for the primary artist, and it
  is not independent of the track's own `popularity_score` — popular artists
  mechanically tend to have popular tracks via correlated underlying
  algorithmic signals. `artist_1_pop` is used descriptively throughout this
  project and never treated as an independent causal predictor of track
  popularity.
- **Genre-availability selection issue.** Tracks with an observed
  `artist_1_genre_1` differ systematically from tracks without one: higher
  median/mean `popularity_score`, higher mean `artist_1_pop`, and a
  different release-decade mix (see
  `reports/data_quality_report.md` §15 for exact figures). Every
  genre-conditioned claim in this project is phrased as "among tracks in
  this dataset with an observed primary genre," never as an unconditional
  statement about all tracks.

## Analytical Workflow

```
data/raw/spotify.csv (gitignored)
        │  src/load_data.py   -- SHA-256 validation, schema validation, missing-value standardization
        ▼
   raw DataFrame (277,938 rows × 34 cols)
        │  src/clean_data.py  -- exact-dedup, derived fields, quality flags
        ▼
   primary analytical dataset (277,937 rows) ──┐
        │  build_sensitivity_dataset()          │
        ▼                                       │
   sensitivity dataset (277,846 rows)            │
        │                                        │
        └──────────────┬─────────────────────────┘
                        ▼
              src/build_database.py
                        │  writes Parquet (data/processed/, gitignored)
                        │  builds DuckDB database (data/processed/spotify.duckdb, gitignored)
                        ▼
              sql/01_data_quality.sql, 02_kpi_summary.sql, 03_segment_analysis.sql
                        ▼
              notebooks/01_data_quality_and_cleaning.ipynb
              notebooks/02_sql_business_analysis.ipynb
```

## Data Quality Findings

Full findings, with every number recomputed programmatically (never
hard-coded), are in [`reports/data_quality_report.md`](reports/data_quality_report.md).
Highlights:

- 1 exact full-row duplicate pair; 91 `(track, artist_1)` repeat groups
  (182 rows), retained by policy rather than deduplicated on popularity.
- `artist_2` / `artist_3` are 100% missing as name fields — yet
  `artist_2_pop` / `artist_3_pop` are populated for 12.5% / 3.4% of rows
  respectively, describing artists this extract cannot identify by name.
- 90 rows with `tempo == 0`; 13 rows with `release_date == 1899`; both
  flagged, neither assumed invalid, neither deleted.
- 16.86% of tracks have `popularity_score == 0`, and the bottom 10% of the
  distribution is entirely 0 — a pattern *consistent with* upstream
  pre-filtering, reported as suggestive evidence, not proof.
- Genre availability is associated with higher popularity, higher artist
  popularity, and a different release-decade mix (see Measurement Caveats).
- Primary-vs-sensitivity-dataset KPI deltas are trivial for every metric
  except `repeat_track_artist_share`, whose drop is a mechanical consequence
  of the sensitivity dataset's own definition, not a substantive finding.
- Genre coverage rises sharply and monotonically with popularity tier
  (32.8% low → 47.7% moderate → 71.75% high) — the strongest single
  confirmation of the genre-availability selection issue (see SQL Analysis).

## SQL Analysis

The SQL layer runs against a local DuckDB database built from the primary
analytical dataset (`src/build_database.py`; the `.duckdb` file itself is
gitignored and rebuilt on demand). All 19 statements across the three files
below run cleanly against that database.

- **`sql/01_data_quality.sql`**: reproduces the core data-quality audit in
  SQL as an independent check against the pandas computations above — row
  counts, missingness, exact/repeat duplicate counts, genre coverage,
  quality-flag counts, popularity distribution (via `PERCENTILE_CONT`), and
  release-decade coverage.
- **`sql/02_kpi_summary.sql`** (5 queries, Q1–Q5): current popularity by
  release decade; popularity distribution and high-popularity share by
  genre (conditioned on observed genre); genre coverage by decade
  (`HAVING` on small-decade groups); popularity metrics by album type.
- **`sql/03_segment_analysis.sql`** (Q6–Q10): primary-genre ranking within
  release decades (`RANK() OVER (PARTITION BY ...)`, chosen over
  `ROW_NUMBER()` so tied genres share a rank); genre share within decade
  (a window *aggregate*, `SUM() OVER (PARTITION BY ...)` with no `ORDER BY`
  inside the window, so it returns each decade's full total rather than a
  running sum); genre availability and top genres ranked by popularity tier;
  a primary-vs-sensitivity KPI comparison computed directly in SQL (matches
  the pandas figures above to 3+ decimal places); and a factual check of
  the album-type popularity-gap hypothesis raised in Q5.

Q1–Q7 were developed fully interactively: question stated in plain English
→ grain discussed → query hand-written by the project author → critiqued on
correctness/grain/null-handling/aggregation/filtering/window semantics/
partitioning/ordering → confirmed or revised → saved. Each query's file
comment records that critique, including the reasoning behind specific
choices such as `RANK()` over `ROW_NUMBER()` for tie handling (Q6) and why
the window in Q7 deliberately omits `ORDER BY` (it changes the frame from a
full-partition total to a running sum). Q8–Q10 were authored directly by
the assistant after the project author explicitly paused the interactive
process partway through Q7's review to finish the phase; they follow the
same documented rules (conditioning on observed genre, no causal claims,
etc.) but were not individually reviewed turn-by-turn the way Q1–Q7 were.
Full results and narration are in `notebooks/02_sql_business_analysis.ipynb`.

## Planned Modeling

Not yet implemented. A future phase may explore regression-style modeling
of `popularity_score` against content/genre/release features, purely as
descriptive/associational modeling consistent with this project's no-causal-
claims stance — it has not been started, attempted, or scaffolded in any way
in this phase.

## Repository Structure

```
spotify-content-performance-analytics/
    README.md
    .gitignore
    requirements.txt
    data/
        README.md              -- provenance, hashes, sampling limitations
        raw/                    -- gitignored raw CSV
        processed/              -- gitignored Parquet + DuckDB build artifacts
    sql/
        01_data_quality.sql     -- complete
        02_kpi_summary.sql      -- complete (Q1-Q5, built interactively)
        03_segment_analysis.sql -- complete (Q6-Q10, see SQL Analysis)
    src/
        load_data.py            -- raw CSV loader, hash + schema validation
        clean_data.py           -- cleaning policy, derived fields, sensitivity dataset
        build_database.py       -- Parquet + DuckDB build
        metrics.py               -- documented KPI/metric layer
        quality_audit.py         -- reusable data-quality audit computations
    notebooks/
        01_data_quality_and_cleaning.ipynb
        02_sql_business_analysis.ipynb
    reports/
        data_quality_report.md
    tests/
        test_load_data.py
        test_clean_data.py
        test_metrics.py
        test_database.py
```

## Reproducibility

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Copy the raw CSV into place (not committed to this repo):
cp /path/to/spotify.csv data/raw/spotify.csv

# 2. Build the processed datasets, Parquet files, and DuckDB database:
python -m src.build_database

# 3. Run the SQL layer:
duckdb data/processed/spotify.duckdb < sql/01_data_quality.sql
duckdb data/processed/spotify.duckdb < sql/02_kpi_summary.sql
duckdb data/processed/spotify.duckdb < sql/03_segment_analysis.sql

# 4. Run the test suite:
pytest

# 5. Execute the notebooks headlessly:
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/01_data_quality_and_cleaning.ipynb \
    notebooks/02_sql_business_analysis.ipynb
```

## Limitations

- This is a single, fixed extract of unknown sampling frame and unknown
  collection date — every finding is scoped to "this dataset," not to
  Spotify's catalog or to any real-world business outcome.
- `artist_2` / `artist_3` are unusable as artist-name fields; any
  collaboration-related analysis in this project is scoped to `artist_1`
  only.
- Genre coverage is 39.7% and is not missing at random — genre-conditioned
  findings describe the genre-labeled subset, not the full dataset.
- No causal claims are made anywhere in this project. Associations between
  content/genre/release characteristics and `popularity_score` are
  descriptive only. Where a plausible explanation is raised (e.g. release-
  period age for compilations' lower popularity, `sql/03_segment_analysis.sql`
  Q10), it is checked factually where possible and reported as partial or
  untested, never asserted as confirmed.
- Genre coverage also varies sharply by release decade (22%–66%) and is
  lowest for 2020s tracks (29.3%) despite that being one of the largest
  decade groups — genre-by-decade comparisons under-represent recent tracks.
- SQL queries Q1–Q7 (across `sql/02_kpi_summary.sql` and
  `sql/03_segment_analysis.sql`) were reviewed interactively per query;
  Q8–Q10 were authored directly after the project author paused the
  interactive process to finish the phase (see SQL Analysis) and were not
  individually critiqued turn-by-turn the same way.
- This project was not used to drive any real Spotify business decision —
  it is a portfolio analytics project built on a coursework-supplied data
  extract.
