"""
Inference entry point — loads the saved model and scores new data.

Run as:
    python -m src.predict --input-path data/new_customers.csv --output-path predictions.csv
"""

import argparse
import json

import joblib
import pandas as pd

from src import config
from src.logger import get_logger

logger = get_logger(__name__)


class ModelNotFoundError(Exception):
    """Raised when predict is called before a model has been trained/saved."""


def load_model():
    if not config.MODEL_PATH.exists():
        logger.error("No trained model found at %s. Run `python -m src.train` first.", config.MODEL_PATH)
        raise ModelNotFoundError(f"No model found at {config.MODEL_PATH}")

    model = joblib.load(config.MODEL_PATH)

    with open(config.MODEL_METADATA_PATH) as f:
        metadata = json.load(f)

    logger.info(
        "Loaded model '%s' (trained %s, %s=%.4f)",
        metadata["model_name"],
        metadata["trained_at_utc"],
        config.PRIMARY_METRIC,
        metadata["metrics"][config.PRIMARY_METRIC],
    )
    return model, metadata


def prepare_input(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Align incoming data to the exact feature set/order the model expects."""
    missing = set(feature_columns) - set(df.columns)
    if missing:
        logger.error("Input data is missing required columns: %s", missing)
        raise ValueError(f"Missing required columns: {missing}")

    extra = set(df.columns) - set(feature_columns) - {config.ID_COLUMN, config.TARGET_COLUMN}
    if extra:
        logger.warning("Ignoring unexpected columns not seen during training: %s", extra)

    return df[feature_columns]


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Return the input dataframe with prediction and probability columns appended."""
    model, metadata = load_model()
    x = prepare_input(df, metadata["feature_columns"])

    logger.info("Scoring %d rows", len(x))
    predictions = model.predict(x)

    result = df.copy()
    result["predicted_target"] = predictions

    if hasattr(model, "predict_proba"):
        result["predicted_probability"] = model.predict_proba(x)[:, 1]

    logger.info("Prediction complete. Positive rate: %.4f", (predictions == 1).mean())
    return result


def main(input_path: str, output_path: str) -> None:
    logger.info("Loading input data from %s", input_path)
    df = pd.read_csv(input_path)

    result = predict(df)

    result.to_csv(output_path, index=False)
    logger.info("Wrote predictions to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score new data with the trained model")
    parser.add_argument("--input-path", type=str, required=True, help="CSV file to score")
    parser.add_argument("--output-path", type=str, default="predictions.csv", help="Where to write predictions")
    args = parser.parse_args()
    main(args.input_path, args.output_path)
