"""
Preprocessing pipeline: drop unwanted columns, split, and balance.

Mirrors notebook sections 4.3 (feature selection), 5.1-5.3 (split +
SMOTE). Kept as pure functions that take/return dataframes or arrays,
so they're independently testable and reusable at inference time.
"""

from typing import Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from src import config
from src.logger import get_logger

logger = get_logger(__name__)


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the id column and the low-correlation calc features."""
    df = df.copy()

    if config.ID_COLUMN in df.columns:
        df = df.drop(columns=[config.ID_COLUMN])
        logger.info("Dropped id column")

    cols_to_drop = [c for c in config.LOW_CORRELATION_COLUMNS if c in df.columns]
    missing = set(config.LOW_CORRELATION_COLUMNS) - set(cols_to_drop)
    if missing:
        logger.warning("Configured low-correlation columns not found in data: %s", missing)

    df = df.drop(columns=cols_to_drop)
    logger.info("Dropped %d low-correlation columns. Remaining shape: %s", len(cols_to_drop), df.shape)

    return df


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split on features vs. target."""
    x = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    logger.info(
        "Split data: x_train=%s x_test=%s y_train=%s y_test=%s",
        x_train.shape, x_test.shape, y_train.shape, y_test.shape,
    )
    return x_train, x_test, y_train, y_test


def balance_with_smote(
    x_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """Oversample the minority class in the training set only."""
    logger.info("Class distribution before SMOTE: %s", y_train.value_counts().to_dict())

    smote = SMOTE(random_state=config.RANDOM_STATE)
    x_smote, y_smote = smote.fit_resample(x_train, y_train)

    logger.info("Class distribution after SMOTE: %s", pd.Series(y_smote).value_counts().to_dict())
    return x_smote, y_smote
