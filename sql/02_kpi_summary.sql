-- ============================================================================
-- 02_kpi_summary.sql
--
-- Descriptive KPI queries against the PRIMARY analytical dataset
-- (tracks_primary). Each query was reviewed for grain, null handling,
-- aggregation semantics, and window-function behavior before being treated
-- as final -- see the comment above each query for that review, and
-- README.md "SQL Analysis" for the finished layer's scope.
--
-- Every popularity/share figure below describes "tracks in this dataset" at
-- an unknown collection time -- see README.md "Measurement Caveats" and
-- reports/data_quality_report.md for the full limitations this SQL layer
-- inherits.
--
-- Run with the DuckDB CLI if installed:
--   duckdb data/processed/spotify.duckdb < sql/02_kpi_summary.sql
-- or via the Python duckdb package already in requirements.txt -- see
-- README.md "Reproducibility" for a short runnable snippet.
-- ============================================================================


-- Q1. Current popularity of tracks by release decade
-- ----------------------------------------------------------------------------
-- Question: for tracks in this dataset, what does popularity_score look like
-- when grouped by release decade -- median, mean, and count per decade?
-- Grain: one row per release decade.
-- Null handling: the 4 tracks with an unknown release_decade_clean are
-- excluded from this decade-level view (WHERE, not HAVING, since we are
-- excluding individual rows before aggregation, not excluding a group based
-- on an aggregate condition). They remain in the underlying primary dataset
-- and are simply not part of this particular breakdown -- no "Unknown"
-- bucket is added because it would not add useful information to a
-- decade-over-decade popularity comparison.
-- Interpretation: read as "current popularity of tracks released in each
-- decade" / "popularity at the dataset's collection time," never as
-- "performance by decade" -- release period is entangled with recency and
-- survivorship (see README). Decades with very small n (1890s: 13, 1900s:
-- 10, 1910s: 18) have noisy medians and should not be over-interpreted.
SELECT
    release_decade_clean,
    COUNT(*) AS track_count,
    MEDIAN(popularity_score) AS median_popularity_score,
    ROUND(AVG(popularity_score), 2) AS mean_popularity_score
FROM tracks_primary
WHERE release_decade_clean IS NOT NULL
GROUP BY release_decade_clean
ORDER BY release_decade_clean;


-- Q2. Popularity distribution by primary genre
-- ----------------------------------------------------------------------------
-- Question: among tracks in this dataset with an observed primary genre, how
-- does popularity_score differ across genres -- count, median, and mean per
-- genre?
-- Grain: one row per artist_1_genre_1 value.
-- Null handling / filtering: WHERE artist_1_genre_1 IS NOT NULL restricts
-- this view to the 39.7% of tracks with an observed primary genre, per the
-- genre-availability bias finding (genre presence correlates with higher
-- popularity, so this is never generalized to "all tracks").
-- Sample size check: smallest genre group is 358 rows (latin), largest is
-- 28,410 (pop) -- all comfortably large enough for this descriptive
-- comparison, so no HAVING floor on track_count is needed.
-- Ordering: a secondary `artist_1_genre_1` key breaks ties on
-- median_popularity_score (several genres tie, e.g. rap/reggae at 39.0)
-- deterministically -- without it, tied rows' relative order is not
-- guaranteed to be stable across query executions.
-- Interpretation: "among tracks in this dataset with an observed primary
-- genre," never an unconditional claim about all tracks or about Spotify's
-- catalog.
SELECT
    artist_1_genre_1,
    COUNT(*) AS track_count,
    MEDIAN(popularity_score) AS median_popularity_score,
    ROUND(AVG(popularity_score), 2) AS mean_popularity_score
FROM tracks_primary
WHERE artist_1_genre_1 IS NOT NULL
GROUP BY artist_1_genre_1
ORDER BY median_popularity_score DESC, artist_1_genre_1;


-- Q3. High-popularity share by genre
-- ----------------------------------------------------------------------------
-- Question: among tracks in this dataset with an observed primary genre,
-- what share of tracks in each genre fall in the recovered "high" popularity
-- tier (popularity_score >= 67)?
-- Grain: one row per artist_1_genre_1 value. Same genre-availability filter
-- and caveat as Q2.
-- Technique: conditional aggregation -- SUM(CASE...) and AVG(CASE...) off
-- the same boolean expression, in a single pass per group, rather than a
-- separate filtered subquery or self-join.
-- Interpretation note: high_popularity_share_pct does not simply track
-- median_popularity_score from Q2 -- reggae has the highest high-popularity
-- share (13.66%) despite a lower median than pop, indicating a relatively
-- heavier upper tail rather than a uniformly higher distribution. Classical,
-- blues, and jazz have very few tracks reaching this threshold (4, 4, 11)
-- despite large group sizes -- a real, if thin, tail, not a sample-size
-- artifact.
SELECT
    artist_1_genre_1,
    COUNT(*) AS track_count,
    SUM(CASE WHEN popularity_score >= 67 THEN 1 ELSE 0 END) AS high_popularity_count,
    ROUND(
        100.0 * AVG(CASE WHEN popularity_score >= 67 THEN 1 ELSE 0 END),
        2
    ) AS high_popularity_share_pct
FROM tracks_primary
WHERE artist_1_genre_1 IS NOT NULL
GROUP BY artist_1_genre_1
ORDER BY high_popularity_share_pct DESC;


-- Q4. Genre coverage by decade
-- ----------------------------------------------------------------------------
-- Question: across ALL tracks in this dataset (not filtered to genre-labeled
-- ones -- this IS the coverage question), what fraction of tracks in each
-- release decade have an observed primary genre?
-- Grain: one row per release decade.
-- Null handling: WHERE release_decade_clean IS NOT NULL excludes the 4 rows
-- that cannot be assigned to any decade, applied before aggregation.
-- HAVING COUNT(*) >= 50 then excludes decade GROUPS whose total size is too
-- small for a stable coverage percentage (1890s: 13, 1900s: 10, 1910s: 18) --
-- correctly placed in HAVING, not WHERE, since it filters on an aggregate
-- (the group's row count), not on individual rows.
-- Technique: conditional aggregation, same SUM(CASE...)/AVG(CASE...) pattern
-- as Q3, applied to genre presence instead of a popularity threshold.
-- Interpretation: genre coverage is NOT stable across release decades
-- (22% in the 1940s up to 66% in the 1970s), and critically, 2020s coverage
-- is only 29.28% despite being one of the largest decade groups. This
-- reinforces that genre availability is systematically incomplete, and any
-- genre-by-decade comparison elsewhere in this project must be conditioned
-- on "tracks in this dataset with an observed primary genre," never treated
-- as representative of that decade's full output.
SELECT
    release_decade_clean,
    COUNT(*) AS track_count,
    SUM(
        CASE
            WHEN artist_1_genre_1 IS NOT NULL THEN 1
            ELSE 0
        END
    ) AS tracks_with_genre,
    ROUND(
        100.0 * AVG(
            CASE
                WHEN artist_1_genre_1 IS NOT NULL THEN 1
                ELSE 0
            END
        ),
        2
    ) AS genre_coverage_pct
FROM tracks_primary
WHERE release_decade_clean IS NOT NULL
GROUP BY release_decade_clean
HAVING COUNT(*) >= 50
ORDER BY release_decade_clean;


-- Q5. Popularity metrics by album type
-- ----------------------------------------------------------------------------
-- Question: for tracks in this dataset, how does popularity differ by
-- album_type (album / single / compilation) -- count, median, mean, and
-- high-popularity share?
-- Grain: one row per album_type.
-- Null handling: none needed -- album_type has zero missingness in this
-- dataset, and all three groups are large (23K-162K rows), so no WHERE/
-- HAVING filter is required.
-- Technique: combines a plain GROUP BY aggregation with the conditional-
-- aggregation high-popularity share pattern from Q3.
-- Interpretation (kept deliberately narrow):
--   - singles have the highest median popularity_score in this dataset (33,
--     vs. 27 for albums and 13 for compilations)
--   - albums have a slightly higher high-popularity share than singles
--     (3.12% vs. 2.91%) despite the lower median -- the same median-vs-tail
--     divergence pattern seen for genre in Q3
--   - compilations are lowest on every measure here
-- Compilations skewing toward older/reissued catalog content is a plausible
-- explanation for their lower popularity, NOT a confirmed cause -- it would
-- need to be checked directly against the release-date distribution before
-- being stated as established.
SELECT
    album_type,
    COUNT(*) AS track_count,
    MEDIAN(popularity_score) AS median_popularity_score,
    ROUND(AVG(popularity_score), 2) AS mean_popularity_score,
    SUM(CASE WHEN popularity_score >= 67 THEN 1 ELSE 0 END) AS high_popularity_count,
    ROUND(
        100.0 * AVG(
            CASE
                WHEN popularity_score >= 67 THEN 1
                ELSE 0
            END
        ),
        2
    ) AS high_popularity_share_pct
FROM tracks_primary
GROUP BY album_type
ORDER BY median_popularity_score DESC;
