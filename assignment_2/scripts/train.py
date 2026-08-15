"""Entry point: run the end-to-end training pipeline.

Usage:  python scripts/train.py

Absolute imports (not relative) because this is a script, not part of the
``loan_default`` package. ``pyproject.toml`` puts ``src`` on the path for
pytest; the bootstrap below does the same for a plain ``python`` run, so no
PYTHONPATH or editable install is needed.
"""

import sys
from pathlib import Path

# parents[1] is the project root (this file is <root>/scripts/train.py), so this
# puts <root>/src on the import path. It must run *before* the loan_default
# imports below, which is why they carry noqa: E402 (import not at top of file).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loan_default.config import CONFIG  # noqa: E402
from loan_default.data import SERVING_FEATURES, TARGET, DataIngestion  # noqa: E402
from loan_default.data_quality import DataQualityChecker  # noqa: E402
from loan_default.logging_config import get_logger  # noqa: E402
from loan_default.metrics import data_quality, drift_report  # noqa: E402
from loan_default.model import ModelTrainer  # noqa: E402

# Named "train" rather than __name__ (which would be "__main__" here) so log
# lines identify the job.
logger = get_logger("train")


def main() -> None:
    """Orchestrate ingest -> validate -> train -> select -> persist.

    Deliberately thin: each step delegates to the package, so the same logic is
    unit-testable without invoking this script.
    """
    # config.py already resolved this to an absolute path, so the run behaves
    # the same from any working directory.
    raw_path = CONFIG["data"]["raw_path"]

    # Hard-fails with SchemaValidationError if a required column is missing;
    # unknown extra columns and a non-binary target only warn.
    df = DataIngestion().load_and_validate(raw_path)

    # Observability only — missingness/shape stats recorded in the run log.
    # Nothing downstream branches on this.
    logger.info("Data quality: %s", data_quality(df))

    # The gate. FAIL means the data is corrupt and training is pointless, so
    # DataQualityError propagates and the run dies here. WARN means the data
    # is intact but the model may be quietly wrong; those are logged and the
    # run continues, because a human -- not the script -- has to judge them.
    checks = DataQualityChecker().run_all(df)
    print("\nData quality checks:")
    print(DataQualityChecker.report(checks))
    DataQualityChecker.raise_on_fail(checks)

    # astype(int): the raw column can load as float/object when NaNs are
    # present, and the stratified split expects discrete labels.
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])  # leakage/ID/Gender dropped inside FeatureEngineer

    # Fits all three candidates and keeps the best by CONFIG's selection_metric
    # (pr_auc by default — preferred over accuracy given the ~75/25 imbalance).
    trainer = ModelTrainer()
    trainer.train_and_select(X, y)

    # Drift baseline. Comparing the two halves of the training file is a
    # self-check that the metric is wired up and that the file is internally
    # homogeneous; in production the reference is this training sample and the
    # current sample is live traffic. Scoped to SERVING_FEATURES because ID is
    # sequential and would register enormous, meaningless drift.
    midpoint = len(X) // 2
    drift = drift_report(X.iloc[:midpoint], X.iloc[midpoint:], columns=SERVING_FEATURES)
    logger.info(
        "Drift self-check: %s (max PSI %.4f on %s)",
        drift["verdict"],
        drift["max_psi"],
        drift["drifted_column"],
    )

    # print, not logger: this table is the operator-facing result of the run,
    # as opposed to the progress logging above.
    print("\nModel comparison (test set):")
    print(trainer.results_.round(4).to_string())

    # Persists the winning Pipeline (feature engineering + classifier) to
    # CONFIG's artifact_path. This file is what inference.py / the API load.
    trainer.save()


if __name__ == "__main__":
    main()
