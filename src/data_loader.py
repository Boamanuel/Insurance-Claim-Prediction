"""
Data loading and basic validation.

Mirrors the "Load Dataset" / "Basic Checks" sections of the original
notebook, but as a reusable, logged function instead of ad-hoc cells.
"""

from pathlib import Path

import pandas as pd

from src import config
from src.logger import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when the input data doesn't match the expected schema."""


def load_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV and run sanity checks that used to be manual
    notebook cells (.shape, .info(), .isnull().sum(), target column
    presence). Raises DataValidationError early instead of failing
    obscurely three steps into preprocessing.
    """
    path = Path(path)
    logger.info("Loading data from %s", path)

    if not path.exists():
        logger.error("Data file not found at %s", path)
        raise FileNotFoundError(f"No data file at {path}")

    df = pd.read_csv(path)
    logger.info("Loaded dataframe with shape %s", df.shape)

    if config.TARGET_COLUMN not in df.columns:
        logger.error("Target column '%s' missing from data", config.TARGET_COLUMN)
        raise DataValidationError(
            f"Expected target column '{config.TARGET_COLUMN}' not found"
        )

    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls > 0:
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        logger.warning("Found %d null values across columns: %s", total_nulls, cols_with_nulls)
    else:
        logger.info("No null values found in dataset")

    class_counts = df[config.TARGET_COLUMN].value_counts().to_dict()
    logger.info("Target class distribution: %s", class_counts)

    return df
