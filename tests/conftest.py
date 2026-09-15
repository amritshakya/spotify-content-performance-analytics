"""Shared fixtures for the test suite.

Loading and cleaning the full 278K-row raw file is not free (a few seconds),
so the real dataset is loaded once per test session and shared across test
modules via these fixtures.
"""

import pytest

from src.clean_data import build_primary_dataset, build_sensitivity_dataset
from src.load_data import load_raw


@pytest.fixture(scope="session")
def raw_df():
    return load_raw()


@pytest.fixture(scope="session")
def primary_df_and_stats(raw_df):
    return build_primary_dataset(raw_df)


@pytest.fixture(scope="session")
def primary_df(primary_df_and_stats):
    return primary_df_and_stats[0]


@pytest.fixture(scope="session")
def dedup_stats(primary_df_and_stats):
    return primary_df_and_stats[1]


@pytest.fixture(scope="session")
def sensitivity_df(primary_df):
    return build_sensitivity_dataset(primary_df)
