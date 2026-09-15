-- ============================================================================
-- 03_segment_analysis.sql
--
-- Segment-analysis queries against the PRIMARY analytical dataset
-- (tracks_primary), plus a comparison against the SENSITIVITY dataset
-- (tracks_sensitivity). Developed interactively, same process as
-- 02_kpi_summary.sql: each query below was written by the project author,
-- reviewed for correctness/grain/null-handling/window-function-semantics/
-- partitioning/ordering, then saved -- for interview defensibility, not
-- authored directly by an LLM.
--
-- Every popularity/share figure below describes "tracks in this dataset" at
-- an unknown collection time -- see README.md "Measurement Caveats" and
-- reports/data_quality_report.md for the full limitations this SQL layer
-- inherits.
--
-- Run with:  duckdb data/processed/spotify.duckdb < sql/03_segment_analysis.sql
-- ============================================================================


-- Q6. Primary-genre ranking within release decades
-- ----------------------------------------------------------------------------
-- Question: within each release decade, which genres are most common (by
-- track count) among tracks in this dataset with an observed primary genre?
-- Grain: two steps. (1) CTE at (release_decade_clean, artist_1_genre_1)
-- grain -- one row per decade/genre combination with its track count.
-- (2) a window function ranking each genre WITHIN its own decade partition.
-- Null handling: both release_decade_clean and artist_1_genre_1 are filtered
-- to NOT NULL inside the CTE, before aggregation -- consistent with the
-- genre-availability conditioning used throughout this project.
-- Window-function choice: RANK(), not ROW_NUMBER(). Genres with identical
-- track counts within the same decade must receive the same rank --
-- ROW_NUMBER() would arbitrarily break the tie and imply an ordering the
-- data doesn't support. RANK() correctly leaves a gap after ties (e.g. two
-- genres tied at rank 2 are followed by rank 4, not rank 3).
-- Partitioning: PARTITION BY release_decade_clean, ORDER BY track_count DESC
-- -- ranks are computed independently within each decade.
-- Small-decade caveat (deliberately NOT filtered out): several early
-- decades have very few observed (decade, genre) combinations (e.g. the
-- 1890s has exactly one genre observed at all), making "rank 1" there
-- trivial/unstable. This is left unfiltered because the query is a complete
-- descriptive ranking -- removing small decades would silently discard
-- observed data rather than just caveat it. Note also that a
-- `HAVING COUNT(*) >= 50` inside the CTE would filter individual
-- (decade, genre) CELLS, not decade GROUPS -- it would not implement "drop
-- unstable decades" even if added.
WITH genre_counts AS (
    SELECT
        release_decade_clean,
        artist_1_genre_1,
        COUNT(*) AS track_count
    FROM tracks_primary
    WHERE release_decade_clean IS NOT NULL
      AND artist_1_genre_1 IS NOT NULL
    GROUP BY
        release_decade_clean,
        artist_1_genre_1
)

SELECT
    release_decade_clean,
    artist_1_genre_1,
    track_count,
    RANK() OVER (
        PARTITION BY release_decade_clean
        ORDER BY track_count DESC
    ) AS genre_rank
FROM genre_counts
ORDER BY
    release_decade_clean,
    genre_rank,
    artist_1_genre_1;


-- Q7. Genre share within decade
-- ----------------------------------------------------------------------------
-- Question: for each (decade, genre) combination among tracks in this
-- dataset with an observed primary genre, what percentage of that decade's
-- genre-labeled tracks does this genre represent?
-- Grain: same (decade, genre) grain as Q6, with each row also carrying its
-- decade's total genre-labeled count as a denominator.
-- Technique: a window AGGREGATE, not a ranking function --
-- SUM(track_count) OVER (PARTITION BY release_decade_clean) reused as both
-- a displayed total and a percentage denominator, avoiding a second query
-- or self-join against the decade totals.
-- Window-frame note (this is the detail that makes the query correct): the
-- OVER() clause deliberately has NO ORDER BY. Omitting ORDER BY makes the
-- default frame the entire partition (every genre row in that decade), so
-- SUM(track_count) returns each decade's full genre-labeled total on every
-- row. Adding ORDER BY track_count DESC inside that OVER() would silently
-- change the frame to a running/cumulative total up to the current row --
-- the wrong denominator entirely.
-- Validation: genre_share_pct sums to ~100% (rounding noise only) within
-- every decade -- checked directly against this query's output.
WITH genre_counts AS (
    SELECT
        release_decade_clean,
        artist_1_genre_1,
        COUNT(*) AS track_count
    FROM tracks_primary
    WHERE release_decade_clean IS NOT NULL
      AND artist_1_genre_1 IS NOT NULL
    GROUP BY
        release_decade_clean,
        artist_1_genre_1
)

SELECT
    release_decade_clean,
    artist_1_genre_1,
    track_count,
    SUM(track_count) OVER (
        PARTITION BY release_decade_clean
    ) AS decade_genre_labeled_total,
    ROUND(
        100.0 * track_count
        / SUM(track_count) OVER (
            PARTITION BY release_decade_clean
        ),
        2
    ) AS genre_share_pct
FROM genre_counts
ORDER BY
    release_decade_clean,
    genre_share_pct DESC,
    artist_1_genre_1;


-- Q8. Observed primary-genre availability by popularity tier, and top
--     genres ranked within each tier
-- ----------------------------------------------------------------------------
-- Question (8a): what fraction of tracks in each popularity_tier (low /
-- moderate / high -- the recovered thresholds from src/clean_data.py) have
-- an observed primary genre? Question (8b): among genre-labeled tracks,
-- which genres are most common within each tier?
-- Grain (8a): one row per popularity_tier. Grain (8b): one row per
-- (tier, genre), limited to each tier's top 5 by track count.
-- Null handling: popularity_tier IS NOT NULL and (for 8b)
-- artist_1_genre_1 IS NOT NULL are both filtered before aggregation.
-- Technique (8a): conditional aggregation, same pattern as Q3/Q4/Q5.
-- Technique (8b): RANK() OVER (PARTITION BY popularity_tier ORDER BY
-- track_count DESC), same justification as Q6 (ties should share a rank),
-- combined with QUALIFY to keep only the top 5 per tier without a wrapping
-- subquery.
-- Finding (8a) is one of the strongest in this project: genre coverage
-- rises monotonically and sharply with popularity tier -- 32.81% (low) ->
-- 47.69% (moderate) -> 71.75% (high). This is additional, tier-level
-- confirmation of the genre-availability bias documented in
-- reports/data_quality_report.md -- it is NOT evidence that having a genre
-- causes higher popularity; both are plausibly downstream of a common
-- factor such as an artist/track being established enough on the platform
-- to have complete metadata AND accumulated plays.
SELECT
    popularity_tier,
    COUNT(*) AS track_count,
    SUM(CASE WHEN artist_1_genre_1 IS NOT NULL THEN 1 ELSE 0 END) AS tracks_with_genre,
    ROUND(100.0 * AVG(CASE WHEN artist_1_genre_1 IS NOT NULL THEN 1 ELSE 0 END), 2) AS genre_coverage_pct
FROM tracks_primary
WHERE popularity_tier IS NOT NULL
GROUP BY popularity_tier
ORDER BY CASE popularity_tier WHEN 'low' THEN 0 WHEN 'moderate' THEN 1 WHEN 'high' THEN 2 END;

WITH tier_genre_counts AS (
    SELECT
        popularity_tier,
        artist_1_genre_1,
        COUNT(*) AS track_count
    FROM tracks_primary
    WHERE popularity_tier IS NOT NULL
      AND artist_1_genre_1 IS NOT NULL
    GROUP BY popularity_tier, artist_1_genre_1
)
SELECT
    popularity_tier,
    artist_1_genre_1,
    track_count,
    RANK() OVER (PARTITION BY popularity_tier ORDER BY track_count DESC) AS genre_rank_in_tier
FROM tier_genre_counts
QUALIFY genre_rank_in_tier <= 5
ORDER BY
    CASE popularity_tier WHEN 'low' THEN 0 WHEN 'moderate' THEN 1 WHEN 'high' THEN 2 END,
    genre_rank_in_tier;


-- Q9. Primary vs. sensitivity dataset -- core KPI comparison (SQL)
-- ----------------------------------------------------------------------------
-- Question: does collapsing to one row per (track, artist_1) --
-- tracks_sensitivity -- materially change the core descriptive KPIs versus
-- the primary dataset (tracks_primary), which retains repeats?
-- Grain: one row per dataset (2 rows total: primary, sensitivity).
-- Technique: two CTEs computing identical KPI expressions against each
-- table, combined with UNION ALL and a `dataset` label column -- keeps the
-- two computations visibly identical (copy-pasted expressions, not
-- re-derived logic) so a difference in the output reflects a difference in
-- the data, not a difference in how the metric was computed.
-- This is a second, independent (SQL, not pandas) confirmation of the same
-- comparison in reports/data_quality_report.md section 17 / src/metrics.py
-- -- and it matches those pandas-computed figures to 3+ decimal places.
-- Finding: deltas are trivial for every KPI shown here. The largest relative
-- mover project-wide, repeat_track_artist_share, is not included here
-- because it is close to zero by construction in the sensitivity dataset
-- (see reports/data_quality_report.md section 17) and is not a KPI about
-- content/genre/release characteristics.
WITH primary_kpis AS (
    SELECT
        'primary' AS dataset,
        COUNT(*) AS track_count,
        MEDIAN(popularity_score) AS median_popularity_score,
        ROUND(AVG(popularity_score), 4) AS mean_popularity_score,
        ROUND(100.0 * AVG(CASE WHEN popularity_score >= 67 THEN 1 ELSE 0 END), 4) AS high_popularity_share_pct,
        ROUND(100.0 * AVG(CASE WHEN artist_1_genre_1 IS NOT NULL THEN 1 ELSE 0 END), 4) AS primary_genre_coverage_pct
    FROM tracks_primary
),
sensitivity_kpis AS (
    SELECT
        'sensitivity' AS dataset,
        COUNT(*) AS track_count,
        MEDIAN(popularity_score) AS median_popularity_score,
        ROUND(AVG(popularity_score), 4) AS mean_popularity_score,
        ROUND(100.0 * AVG(CASE WHEN popularity_score >= 67 THEN 1 ELSE 0 END), 4) AS high_popularity_share_pct,
        ROUND(100.0 * AVG(CASE WHEN artist_1_genre_1 IS NOT NULL THEN 1 ELSE 0 END), 4) AS primary_genre_coverage_pct
    FROM tracks_sensitivity
)
SELECT * FROM primary_kpis
UNION ALL
SELECT * FROM sensitivity_kpis;


-- Q10. Release-period profile by album type (factual check on a Q5 hypothesis)
-- ----------------------------------------------------------------------------
-- Question: Q5 (sql/02_kpi_summary.sql) flagged "compilations skew toward
-- older/reissued content" as an untested explanation for their lower
-- popularity. This checks it directly: what is each album_type's release-
-- year profile in this dataset?
-- Grain: one row per album_type.
-- Null handling: WHERE release_year IS NOT NULL excludes the 4 rows with
-- unknown release date, before aggregation.
-- Finding (factual, not causal): compilations do skew markedly older --
-- median release year 2010, with 48.4% released in 2000 or earlier, versus
-- singles at median 2021 and only 3.5% that old. But albums also have a
-- large older-release share (40.8%, median 2012) while still having a much
-- higher median popularity_score than compilations (27 vs. 13, from Q5) --
-- so release-period age alone does not fully explain the compilation
-- popularity gap. The hypothesis is PARTIALLY supported, not confirmed as a
-- complete explanation, and no causal claim is made either way.
SELECT
    album_type,
    COUNT(*) AS track_count,
    MEDIAN(release_year) AS median_release_year,
    ROUND(100.0 * AVG(CASE WHEN release_decade_clean <= 2000 THEN 1 ELSE 0 END), 2) AS pct_released_2000_or_earlier
FROM tracks_primary
WHERE release_year IS NOT NULL
GROUP BY album_type
ORDER BY median_release_year;
