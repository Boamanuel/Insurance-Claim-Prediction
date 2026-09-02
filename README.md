# Insurance Claim Prediction

![tests](https://github.com/your-username/insurance-claim-prediction/actions/workflows/tests.yml/badge.svg)

A machine learning pipeline that predicts which customers are likely to buy an
insurance product, built on an imbalanced, high-dimensional dataset
(~595K rows, 59 features, ~3.6% positive class).

This started as exploratory work in a Jupyter notebook (kept in
[`notebooks/`](notebooks/) for reference) and was refactored here into a
modular, logged, testable pipeline suitable for deployment or handing off to
another engineer.

## What it does

1. **Loads and validates** the raw data (schema checks, null checks, class
   distribution).
2. **Preprocesses**: drops the `id` column and low-correlation features
   identified during EDA, splits into train/test, and balances the training
   set with SMOTE.
3. **Trains 7 candidate classifiers** — Logistic Regression, KNN, Decision
   Tree, Gradient Boosting, Random Forest, Naive Bayes, SVM — and evaluates
   each on accuracy, minority-class F1, and ROC-AUC.
4. **Selects and persists the best model** (by ROC-AUC, since accuracy is a
   misleading metric on this imbalanced a dataset) along with metadata
   (metrics, feature list, training timestamp) for reproducibility.
5. **Serves predictions** on new data via a separate inference script.

Every step logs to console and to a rotating file (`logs/pipeline.log`)
instead of relying on notebook `print()` output that disappears when the
kernel restarts.

## Project structure

```
insurance-claim-prediction/
├── src/
│   ├── config.py          # paths, hyperparameters, constants
│   ├── logger.py           # shared logging setup
│   ├── data_loader.py       # load + validate raw data
│   ├── preprocessing.py     # feature selection, split, SMOTE
│   ├── train.py             # trains all models, saves the best one
│   └── predict.py           # inference on new data
├── tests/
│   └── test_preprocessing.py
├── notebooks/
│   └── Insurance_claim_prediction_model_final.ipynb   # original EDA/notebook
├── data/                   # not committed — see Data section
├── models/                 # trained model + metadata land here (not committed)
├── logs/                   # pipeline.log (not committed)
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/<your-username>/insurance-claim-prediction.git
cd insurance-claim-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data

The raw dataset (`train.csv`) is **not committed** to this repo

### Train

```bash
python -m src.train
# or point at a different file:
python -m src.train --data-path path/to/other_train.csv
```

This logs progress for every model, then saves the best-performing model to
`models/best_model.joblib` and its metadata to `models/model_metadata.json`.

### Predict

```bash
python -m src.predict --input-path data/new_customers.csv --output-path predictions.csv
```

### Run tests

```bash
pytest
```

Tests use small synthetic data, so they run without needing the real
dataset.



## Known limitations / next steps

- The PCA exploration from the original notebook is not part of the
  production pipeline (it was exploratory, not something the final models
  depend on) — see the notebook if you want that analysis.
- No hyperparameter tuning (e.g. `GridSearchCV`) is included yet — each
  model uses either notebook-derived or default parameters.
- No CI workflow yet (see `.github/workflows/` suggestion below) — tests
  currently need to be run manually.
- The SVM's `probability=True` setting adds training overhead; drop it if
  you don't need probability outputs and want faster training.
