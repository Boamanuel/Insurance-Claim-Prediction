"""
Training entry point.

Trains every classifier used in the original notebook on the SMOTE-
balanced training set, evaluates each on the held-out test set, logs
every result, and persists the best-performing model + its metadata
to models/. Run as:

    python -m src.train
"""

import argparse
import json
import time
from datetime import datetime, timezone

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data_loader import load_data
from src.logger import get_logger
from src.preprocessing import balance_with_smote, select_features, split_data

logger = get_logger(__name__)


def get_candidate_models() -> dict:
    """Same model set as the notebook, defined once in one place."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=config.KNN_N_NEIGHBORS),
        "decision_tree": DecisionTreeClassifier(**config.DECISION_TREE_PARAMS),
        "gradient_boosting": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
        "random_forest": RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
        "naive_bayes": GaussianNB(),
        "svm": SVC(probability=True, random_state=config.RANDOM_STATE),
    }


def evaluate_model(model, x_test, y_test) -> dict:
    """Compute accuracy, minority-class F1, and ROC-AUC for a fitted model."""
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_minority_class": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    else:
        metrics["roc_auc"] = float("nan")

    logger.debug(
        "Classification report:\n%s",
        classification_report(y_test, y_pred, zero_division=0),
    )
    return metrics


def train_all_models(x_train, y_train, x_test, y_test) -> dict:
    """Train + evaluate every candidate model, logging progress and results."""
    results = {}
    models = get_candidate_models()

    for name, model in models.items():
        logger.info("Training model: %s", name)
        start = time.time()
        try:
            model.fit(x_train, y_train)
        except Exception:
            logger.exception("Training failed for model '%s' — skipping", name)
            continue
        elapsed = time.time() - start

        metrics = evaluate_model(model, x_test, y_test)
        metrics["train_seconds"] = round(elapsed, 2)
        results[name] = {"model": model, "metrics": metrics}

        logger.info(
            "Finished %s in %.2fs — accuracy=%.4f f1_minority=%.4f roc_auc=%.4f",
            name, elapsed, metrics["accuracy"], metrics["f1_minority_class"], metrics["roc_auc"],
        )

    return results


def select_best_model(results: dict) -> tuple:
    """Pick the model with the highest value of config.PRIMARY_METRIC."""
    if not results:
        raise RuntimeError("No models trained successfully — nothing to select from")

    best_name = max(
        results, key=lambda name: results[name]["metrics"][config.PRIMARY_METRIC]
    )
    logger.info(
        "Selected best model: %s (%s=%.4f)",
        best_name, config.PRIMARY_METRIC, results[best_name]["metrics"][config.PRIMARY_METRIC],
    )
    return best_name, results[best_name]


def save_model(name: str, model, metrics: dict, feature_columns: list) -> None:
    """Persist the trained model and metadata (for reproducibility/auditing)."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, config.MODEL_PATH)
    logger.info("Saved model to %s", config.MODEL_PATH)

    metadata = {
        "model_name": name,
        "metrics": metrics,
        "feature_columns": feature_columns,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
    }
    with open(config.MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to %s", config.MODEL_METADATA_PATH)


def main(data_path=config.RAW_DATA_PATH) -> None:
    logger.info("=== Starting training pipeline ===")
    try:
        df = load_data(data_path)
        df = select_features(df)
        x_train, x_test, y_train, y_test = split_data(df)
        x_smote, y_smote = balance_with_smote(x_train, y_train)

        results = train_all_models(x_smote, y_smote, x_test, y_test)
        best_name, best_result = select_best_model(results)

        save_model(
            best_name,
            best_result["model"],
            best_result["metrics"],
            feature_columns=list(x_train.columns),
        )
        logger.info("=== Training pipeline completed successfully ===")

    except Exception:
        logger.exception("Training pipeline failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train insurance claim prediction models")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(config.RAW_DATA_PATH),
        help="Path to the training CSV file",
    )
    args = parser.parse_args()
    main(data_path=args.data_path)
