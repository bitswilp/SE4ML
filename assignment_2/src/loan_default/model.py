"""Model training, selection, and persistence (Objective 1.1 + 1.3).

Ports the candidate-model comparison from Assignment I. Each candidate is a
full ``Pipeline([features, model])`` so the fitted artifact is self-contained:
raw application JSON in, prediction out, with preprocessing baked in.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import CONFIG
from .features import FeatureEngineer
from .logging_config import get_logger
from .metrics import model_quality

logger = get_logger(__name__)

RANDOM_STATE = CONFIG["data"]["random_state"]
TEST_SIZE = CONFIG["data"]["test_size"]
ARTIFACT_PATH = CONFIG["model"]["artifact_path"]
RF_N_ESTIMATORS = CONFIG["model"].get("rf_n_estimators", 200)
SELECTION_METRIC = CONFIG["model"].get("selection_metric", "pr_auc")


def candidate_models() -> dict:
    """The three Assignment I candidates.

    ``class_weight`` balancing addresses the ~75/25 class imbalance so the
    minority (default) class is not ignored.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }


def build_pipeline(classifier) -> Pipeline:
    """Compose feature engineering + a classifier into one fitted object."""
    return Pipeline([("features", FeatureEngineer()), ("model", classifier)])


class ModelTrainer:
    """Trains the candidates, selects the best by the configured metric, saves it."""

    def __init__(self, selection_metric: str = SELECTION_METRIC) -> None:
        self.selection_metric = selection_metric
        self.results_: pd.DataFrame | None = None
        self.best_name_: str | None = None
        self.best_pipeline_: Pipeline | None = None

    def train_and_select(self, X: pd.DataFrame, y) -> Pipeline:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
        logger.info(
            "Split -> train %d | test %d (stratified)", len(X_train), len(X_test)
        )

        rows, fitted = [], {}
        for name, clf in candidate_models().items():
            logger.info("Training candidate: %s", name)
            pipe = build_pipeline(clf)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            scores = model_quality(y_test, y_pred, y_proba)
            scores["model"] = name
            rows.append(scores)
            fitted[name] = pipe

        self.results_ = pd.DataFrame(rows).set_index("model")
        self.best_name_ = self.results_[self.selection_metric].idxmax()
        self.best_pipeline_ = fitted[self.best_name_]
        logger.info("Best model by %s: %s", self.selection_metric, self.best_name_)
        return self.best_pipeline_

    def save(self, path: str | Path | None = None) -> Path:
        if self.best_pipeline_ is None:
            logger.error("save() called before train_and_select().")
            raise RuntimeError("Nothing to save; call train_and_select() first.")
        path = Path(path or ARTIFACT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_pipeline_, path)
        logger.info("Saved best model '%s' to %s", self.best_name_, path)
        return path
