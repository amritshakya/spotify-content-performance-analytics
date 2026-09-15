"""Raw-data loading for the Spotify Content Performance Analytics project.

This module owns the ONLY code path that is allowed to read
``data/raw/spotify.csv``. It intentionally does not reuse the CS251 coursework
``Data`` class (see repository root README) -- it is a fresh, pandas-based
loader written for this project.

Responsibilities:
    * validate the raw file's identity via SHA-256 before ingesting it
    * skip the course-specific row-2 type-declaration row correctly
    * validate the resulting shape and column set against known-good values
    * surface malformed rows instead of silently dropping them
    * standardize known missing-value conventions
    * never write to, or otherwise mutate, the raw file

Nothing in this module removes rows, deduplicates, or derives analytical
fields -- that is the job of ``src/clean_data.py``. The loader's output is a
faithful, minimally-standardized reflection of the raw extract.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = REPO_ROOT / "data" / "raw" / "spotify.csv"

# Identity of the raw extract this project was built against. Computed with
# `shasum -a 256` against the byte-for-byte copy of the CS251 course file.
EXPECTED_SHA256 = "4045adda473d6cd4fde6f3a11253faf37d7ee26fd122bcfd5c8a6c1fc5441974"

# Row 2 of the raw file is a course-specific type-declaration row, not an
# observation -- it is always skipped. This is the count of real observations
# in the remaining rows.
EXPECTED_ROW_COUNT = 277_938

# Column order and names as they appear in the raw file's header row.
EXPECTED_COLUMNS = [
    "track",
    "artist_1",
    "artist_1_pop",
    "artist_2",
    "artist_2_pop",
    "artist_3",
    "artist_3_pop",
    "artist_1_genre_1",
    "artist_1_genre_2",
    "artist_1_genre_3",
    "artist_2_genre_1",
    "artist_2_genre_2",
    "artist_2_genre_3",
    "artist_3_genre_1",
    "artist_3_genre_2",
    "artist_3_genre_3",
    "release_date",
    "release_decade",
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "popularity",
    "duration",
    "album_type",
    "popularity_score",
    "key",
    "loudness",
    "valence",
    "tempo",
    "duration_ms",
    "time_signature",
]

# Columns downstream code depends on existing; a stricter subset of
# EXPECTED_COLUMNS used to fail fast with a clear message if the file has been
# tampered with in a way that preserves row/column *counts* but not names.
REQUIRED_COLUMNS = [
    "track",
    "artist_1",
    "artist_1_pop",
    "artist_1_genre_1",
    "release_date",
    "popularity",
    "popularity_score",
    "album_type",
    "tempo",
    "duration_ms",
    "time_signature",
]

# Literal string tokens observed in the raw extract that represent missing
# values but are not parsed as NaN by pandas' default `read_csv` (which only
# treats empty fields and a small builtin list as missing). Empty fields are
# already handled by pandas natively and are not included here.
LITERAL_MISSING_TOKENS = ["Missing", "missing", "MISSING", "N/A", "n/a", "NA"]


class RawDataError(Exception):
    """Base class for problems with the raw data file."""


class HashMismatchError(RawDataError):
    """Raised when the raw file's SHA-256 does not match the expected value."""


class SchemaError(RawDataError):
    """Raised when the raw file's shape or columns do not match expectations."""


def compute_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute the SHA-256 hex digest of a file, streaming in chunks."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw_file_hash(
    path: Path = DEFAULT_RAW_PATH, expected: str = EXPECTED_SHA256
) -> str:
    """Validate that ``path`` matches ``expected`` SHA-256. Returns the digest.

    Raises HashMismatchError on mismatch. This check exists so that a
    corrupted download, an accidental partial copy, or a swapped-in different
    extract is caught before any analysis is built on top of it.
    """
    actual = compute_sha256(path)
    if actual != expected:
        raise HashMismatchError(
            f"SHA-256 mismatch for {path}.\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "The raw file does not match the extract this project was built "
            "against. Refusing to load it -- verify the correct file is at "
            "this path (see data/README.md)."
        )
    return actual


def load_raw(
    path: Path = DEFAULT_RAW_PATH,
    validate_hash: bool = True,
    validate_shape: bool = True,
) -> pd.DataFrame:
    """Load the raw Spotify CSV into a DataFrame.

    Parameters
    ----------
    path:
        Location of the raw CSV. Defaults to ``data/raw/spotify.csv``.
    validate_hash:
        If True (default), verify the file's SHA-256 against
        ``EXPECTED_SHA256`` before parsing. Set False only for deliberate
        testing against a different, known extract.
    validate_shape:
        If True (default), verify the parsed row/column counts and required
        column names against expected values.

    Returns
    -------
    pandas.DataFrame with 34 raw columns, one row per observation, row 2
    (the course type-declaration row) skipped, and known literal missing-value
    tokens standardized to NaN. Raw values are otherwise preserved unmodified
    -- no rows are dropped, deduplicated, or type-coerced beyond pandas'
    default CSV type inference.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    HashMismatchError
        If ``validate_hash`` is True and the file's hash does not match.
    SchemaError
        If ``validate_shape`` is True and the row count, column count, or
        required column names do not match expectations.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}\n"
            "Copy the original spotify.csv into data/raw/ (see data/README.md)."
        )

    if validate_hash:
        validate_raw_file_hash(path)

    malformed_lines: list[str] = []

    def _on_bad_line(bad_line: list[str]) -> None:
        malformed_lines.append(bad_line)
        return None

    df = pd.read_csv(
        path,
        skiprows=[1],  # course-specific type-declaration row, not data
        on_bad_lines=_on_bad_line,
        engine="python",
    )

    if malformed_lines:
        # Surface, don't swallow: malformed rows were skipped by the parser.
        # This project has never observed any in the known-good extract, so
        # any occurrence is worth a loud warning rather than silent loss.
        import warnings

        warnings.warn(
            f"{len(malformed_lines)} malformed row(s) were skipped while "
            f"parsing {path}. First example: {malformed_lines[0]!r}",
            stacklevel=2,
        )

    if validate_shape:
        if list(df.columns) != EXPECTED_COLUMNS:
            missing = set(EXPECTED_COLUMNS) - set(df.columns)
            extra = set(df.columns) - set(EXPECTED_COLUMNS)
            raise SchemaError(
                "Raw file columns do not match expected schema.\n"
                f"  missing: {sorted(missing)}\n"
                f"  extra:   {sorted(extra)}\n"
                f"  order matches: {list(df.columns) == EXPECTED_COLUMNS}"
            )
        missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_required:
            raise SchemaError(f"Required columns missing: {missing_required}")
        if len(df) != EXPECTED_ROW_COUNT:
            raise SchemaError(
                f"Expected {EXPECTED_ROW_COUNT} observations, got {len(df)}. "
                "The raw file may have been altered."
            )

    df = _standardize_missing_values(df)
    return df


def _standardize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace known literal missing-value tokens with real NaN.

    Pandas already treats empty CSV fields as NaN by default. This handles
    the additional literal-string convention observed in this extract (e.g.
    the string "Missing" appearing in a handful of `track` values).
    """
    df = df.copy()
    object_cols = df.select_dtypes(include="object").columns
    df[object_cols] = df[object_cols].replace(LITERAL_MISSING_TOKENS, pd.NA)
    return df


def malformed_row_report(path: Path = DEFAULT_RAW_PATH) -> dict:
    """Parse the raw file and report on malformed rows without raising.

    Returns a dict with `count` and up to 5 example raw lines. Used by the
    data-quality audit; not part of the normal load path's control flow.
    """
    path = Path(path)
    malformed_lines: list[str] = []

    def _on_bad_line(bad_line: list[str]) -> None:
        malformed_lines.append(bad_line)
        return None

    pd.read_csv(
        path,
        skiprows=[1],
        on_bad_lines=_on_bad_line,
        engine="python",
    )
    return {"count": len(malformed_lines), "examples": malformed_lines[:5]}


if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns from {DEFAULT_RAW_PATH}")
    print(f"SHA-256 validated: {EXPECTED_SHA256}")
