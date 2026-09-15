-- ============================================================================
-- 01_data_quality.sql
--
-- Data-quality checks against the PRIMARY analytical dataset (tracks_primary),
-- built by src/build_database.py from the primary cleaned dataset (exact
-- full-row duplicates removed; (track, artist_1) repeats retained; derived
-- fields and quality flags added -- see src/clean_data.py).
--
-- This file reproduces the core data-quality audit numbers in SQL, as a
-- second, independent check against the pandas-computed figures in
-- reports/data_quality_report.md. Every share/percentage below describes
-- "tracks in this dataset," not Spotify's catalog generally.
--
-- Run with:  duckdb data/processed/spotify.duckdb < sql/01_data_quality.sql
-- ============================================================================


-- 1. TOTAL OBSERVATIONS -------------------------------------------------------
-- Row count of the primary analytical dataset (raw extract minus the single
-- exact full-row duplicate removed during cleaning).
SELECT COUNT(*) AS total_observations
FROM tracks_primary;


-- 2. NULL / MISSING RATES -----------------------------------------------------
-- Missingness for the columns most relevant to this project's analytical
-- question. artist_2 / artist_3 are included to make their near-total
-- missingness visible directly from the database, not just in pandas.
SELECT
    ROUND(100.0 * SUM(CASE WHEN track            IS NULL THEN 1 ELSE 0 END) / COUNT(*), 3) AS pct_missing_track,
    ROUND(100.0 * SUM(CASE WHEN artist_1          IS NULL THEN 1 ELSE 0 END) / COUNT(*), 3) AS pct_missing_artist_1,
    ROUND(100.0 * SUM(CASE WHEN artist_2          IS NULL THEN 1 ELSE 0 END) / COUNT(*), 3) AS pct_missing_artist_2,
    ROUND(100.0 * SUM(CASE WHEN artist_3          IS NULL THEN 1 ELSE 0 END) / COUNT(*), 3) AS pct_missing_artist_3,
    ROUND(100.0 * SUM(CASE WHEN artist_1_genre_1  IS NULL THEN 1 ELSE 0 END) / COUNT(*), 3) AS pct_missing_primary_genre,
    ROUND(100.0 * SUM(CASE WHEN release_date      IS NULL THEN 1 ELSE 0 END) / COUNT(*), 3) AS pct_missing_release_date
FROM tracks_primary;


-- 3a. EXACT FULL-ROW DUPLICATES (post-cleaning validation) -------------------
-- Should be exactly 0: exact duplicates are removed before the primary
-- dataset is written, so this query is an integrity check confirming that.
SELECT
    SUM(CASE WHEN exact_duplicate_flag THEN 1 ELSE 0 END) AS remaining_exact_duplicates
FROM tracks_primary;


-- 3b. (track, artist_1) REPEAT COUNTS ----------------------------------------
-- Repeats are retained by policy (see src/clean_data.py docstring). This
-- reports how many groups repeat and how many rows that involves.
WITH grouped AS (
    SELECT track, artist_1, COUNT(*) AS n
    FROM tracks_primary
    GROUP BY track, artist_1
)
SELECT
    SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END)      AS repeat_groups,
    SUM(CASE WHEN n > 1 THEN n ELSE 0 END)      AS rows_in_repeat_groups,
    MAX(n)                                       AS largest_group_size
FROM grouped;


-- 4. PRIMARY GENRE COVERAGE ---------------------------------------------------
SELECT
    SUM(CASE WHEN has_primary_genre THEN 1 ELSE 0 END)                          AS n_with_primary_genre,
    SUM(CASE WHEN NOT has_primary_genre THEN 1 ELSE 0 END)                      AS n_without_primary_genre,
    ROUND(100.0 * AVG(CASE WHEN has_primary_genre THEN 1 ELSE 0 END), 3)        AS pct_with_primary_genre
FROM tracks_primary;


-- 5. QUALITY-FLAG COUNTS ------------------------------------------------------
SELECT
    SUM(CASE WHEN tempo_quality_flag = 'zero_tempo' THEN 1 ELSE 0 END)          AS n_zero_tempo,
    SUM(CASE WHEN tempo_quality_flag = 'missing'    THEN 1 ELSE 0 END)          AS n_missing_tempo,
    SUM(CASE WHEN suspicious_release_date_flag      THEN 1 ELSE 0 END)          AS n_suspicious_release_date_1899,
    SUM(CASE WHEN track_artist_duplicate_flag       THEN 1 ELSE 0 END)          AS n_rows_in_track_artist_repeats,
    SUM(CASE WHEN is_high_popularity                THEN 1 ELSE 0 END)          AS n_high_popularity
FROM tracks_primary;


-- 6. POPULARITY DISTRIBUTION --------------------------------------------------
-- Median is reported as the primary summary elsewhere in this project because
-- the distribution is right-skewed with a large mass at 0; mean is kept as a
-- secondary comparison. Quantiles below quantify that skew and the size of
-- the zero-popularity spike.
SELECT
    COUNT(*)                                                             AS n,
    SUM(CASE WHEN popularity_score = 0 THEN 1 ELSE 0 END)                AS n_zero,
    ROUND(100.0 * AVG(CASE WHEN popularity_score = 0 THEN 1 ELSE 0 END), 3) AS pct_zero,
    ROUND(AVG(popularity_score), 3)                                      AS mean_popularity_score,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY popularity_score)       AS p25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY popularity_score)       AS median_popularity_score,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY popularity_score)       AS p75,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY popularity_score)       AS p90,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY popularity_score)       AS p99
FROM tracks_primary;


-- 7. RELEASE-PERIOD COVERAGE --------------------------------------------------
-- Count of tracks in this dataset by release decade. Read as "current
-- popularity of tracks released in each decade" contexts elsewhere in this
-- project always pair this with popularity, never decade counts alone, and
-- never phrase it as "performance by decade" (release period is entangled
-- with recency, survivorship, and catalog composition -- see README).
SELECT
    release_decade_clean,
    COUNT(*) AS n_tracks
FROM tracks_primary
GROUP BY release_decade_clean
ORDER BY release_decade_clean;
