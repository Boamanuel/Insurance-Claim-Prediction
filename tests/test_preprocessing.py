"""
Lightweight unit tests using synthetic data — don't require the real
(595k-row, non-committed) train.csv. Run with: pytest
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.preprocessing import select_features, split_data


@pytest.fixture
def sample_df():
    n = 200
    rng = np.random.default_rng(config.RANDOM_STATE)
    data = {
        "id": range(n),
        "ps_ind_01": rng.integers(0, 5, n),
        "ps_calc_04": rng.integers(0, 5, n),  # should get dropped
        "ps_calc_06": rng.integers(0, 5, n),  # should get dropped
        "target": rng.choice([0, 1], size=n, p=[0.9, 0.1]),
    }
    return pd.DataFrame(data)


def test_select_features_drops_id_and_low_correlation_columns(sample_df):
    result = select_features(sample_df)
    assert "id" not in result.columns
    assert "ps_calc_04" not in result.columns
    assert "ps_calc_06" not in result.columns
    assert "ps_ind_01" in result.columns
    assert "target" in result.columns


def test_split_data_shapes(sample_df):
    df = select_features(sample_df)
    x_train, x_test, y_train, y_test = split_data(df)

    assert len(x_train) + len(x_test) == len(df)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    # roughly matches configured test_size
    assert abs(len(x_test) / len(df) - config.TEST_SIZE) < 0.05
