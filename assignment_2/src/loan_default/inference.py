"""Inference: load the trained pipeline and score loan applications.

The saved artifact is a full ``Pipeline([features, model])``, so scoring takes
raw application records (dicts / DataFrame) straight in -- feature engineering
runs inside the pipeline, guaranteeing the same transforms as training.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .config import CONFIG
from .logging_config import get_logger

logger = get_logger(__name__)

ARTIFACT_PATH = CONFIG["model"]["artifact_path"]
DECLINE_AT = CONFIG["decision"]["decline_at"]
REFER_AT = CONFIG["decision"]["refer_at"]


class Predictor:
    """Loads a saved pipeline and turns applications into risk decisions."""

    def __init__(
        self,
        artifact_path: str | Path | None = None,
        decline_at: float = DECLINE_AT,
        refer_at: float = REFER_AT,
    ) -> None:
        self.artifact_path = Path(artifact_path or ARTIFACT_PATH)
        self.decline_at = decline_at
        self.refer_at = refer_at
        self.model = self._load()

    def _load(self):
        if not self.artifact_path.exists():
            logger.error("Model artifact missing: %s", self.artifact_path)
            raise FileNotFoundError(self.artifact_path)
        logger.info("Loading model from %s", self.artifact_path)
        return joblib.load(self.artifact_path)

    @staticmethod
    def _to_frame(records) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            return records
        if isinstance(records, dict):
            records = [records]
        return pd.DataFrame(records)

    def predict_proba(self, records):
        """Return P(default) for each record."""
        return self.model.predict_proba(self._to_frame(records))[:, 1]

    def predict(self, records):
        """Return the hard 0/1 label for each record."""
        return self.model.predict(self._to_frame(records))

    def decide(self, prob: float) -> str:
        """Map a probability to the business decision."""
        if prob >= self.decline_at:
            return "decline"
        if prob >= self.refer_at:
            return "refer"
        return "approve"

    def score(self, records) -> list[dict]:
        """Full scoring: probability + label + decision, per record."""
        probs = self.predict_proba(records)
        return [
            {
                "default_probability": round(float(p), 4),
                "prediction": int(p >= 0.5),
                "decision": self.decide(float(p)),
            }
            for p in probs
        ]
