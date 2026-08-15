"""Data ingestion: load the raw loan-default file and validate its schema.

Objective 1.1 (modular design) + 1.3 (error handling / logging).

Research-code equivalent: the ``load_data`` function and inline column lists
in the Assignment I ``loan_default_pipeline.py`` notebook/script. Here the same
responsibility is a class with explicit schema validation, typed errors, and
levelled logging instead of ``print``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import CONFIG
from .logging_config import get_logger

logger = get_logger(__name__)

# Full expected schema of the RAW file (34 columns), verified against the
# Kaggle loan-default dataset (148,670 x 34).
KNOWN_COLUMNS: list[str] = [
    "ID",
    "year",
    "loan_limit",
    "Gender",
    "approv_in_adv",
    "loan_type",
    "loan_purpose",
    "Credit_Worthiness",
    "open_credit",
    "business_or_commercial",
    "loan_amount",
    "rate_of_interest",
    "Interest_rate_spread",
    "Upfront_charges",
    "term",
    "Neg_ammortization",
    "interest_only",
    "lump_sum_payment",
    "property_value",
    "construction_type",
    "occupancy_type",
    "Secured_by",
    "total_units",
    "income",
    "credit_type",
    "Credit_Score",
    "co-applicant_credit_type",
    "age",
    "submission_of_application",
    "LTV",
    "Region",
    "Security_Type",
    "Status",
    "dtir1",
]

TARGET: str = CONFIG["data"]["target_column"]
_DROP = set(CONFIG["data"]["id_sensitive_columns"]) | set(
    CONFIG["data"]["leakage_columns"]
)

# The 27 features the model actually consumes (everything the pipeline keeps),
# plus the target. These MUST be present or ingestion fails hard.
SERVING_FEATURES: list[str] = [
    c for c in KNOWN_COLUMNS if c not in _DROP and c != TARGET
]
REQUIRED_COLUMNS: list[str] = SERVING_FEATURES + [TARGET]

# The numeric / categorical split of SERVING_FEATURES, consumed by the
# ColumnTransformer in features.py. Declared explicitly rather than inferred
# from dtypes: pandas types depend on what happens to be in a given batch, and
# a small serving payload could silently classify a column the other way.
# Note total_units ("1U") and age ("25-34") are string-coded and so categorical.
NUMERIC_FEATURES: list[str] = [
    "year",
    "loan_amount",
    "term",
    "property_value",
    "income",
    "Credit_Score",
    "LTV",
    "dtir1",
]
CATEGORICAL_FEATURES: list[str] = [
    c for c in SERVING_FEATURES if c not in NUMERIC_FEATURES
]

# Guard: the split must cover the serving set exactly (8 + 19 = 27).
assert set(NUMERIC_FEATURES) <= set(SERVING_FEATURES), "unknown numeric feature"
assert len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) == len(SERVING_FEATURES)


class SchemaValidationError(ValueError):
    """Raised when the ingested data is missing required columns (hard FAIL)."""


class DataIngestion:
    """Loads the loan-default dataset and validates its structure.

    Design note (FAIL vs WARN): a missing *required* column is a hard FAIL
    that raises and blocks the pipeline (corrupt input). Unexpected extra
    columns or a non-binary target are WARN-level: logged for human review
    but not fatal, because they may be benign schema drift.
    """

    def __init__(self, target_column: str = TARGET) -> None:
        self.target_column = target_column

    def load(self, path: str | Path) -> pd.DataFrame:
        """Read the CSV into a DataFrame.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the file cannot be parsed as CSV.
        """
        path = Path(path)
        if not path.exists():
            logger.error("Data file not found: %s", path)
            raise FileNotFoundError(path)
        try:
            logger.info("Loading data from %s", path)
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to parse CSV %s: %s", path, exc)
            raise ValueError(f"Could not read CSV: {path}") from exc

        logger.info("Loaded %d rows x %d cols", df.shape[0], df.shape[1])
        if self.target_column in df.columns:
            balance = df[self.target_column].value_counts(normalize=True)
            logger.info(
                "Target balance -> 0: %.1f%% | 1: %.1f%%",
                balance.get(0, 0) * 100,
                balance.get(1, 0) * 100,
            )
        return df

    def validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check the DataFrame against the expected schema.

        Returns the same DataFrame on success (so it can be chained).
        Raises ``SchemaValidationError`` if any required column is absent.
        """
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            logger.error("Schema validation FAILED. Missing columns: %s", missing)
            raise SchemaValidationError(f"Missing required columns: {missing}")

        unexpected = [c for c in df.columns if c not in KNOWN_COLUMNS]
        if unexpected:
            logger.warning("Unexpected (unknown) columns present: %s", unexpected)

        target_values = set(df[self.target_column].dropna().unique())
        if not target_values <= {0, 1}:
            logger.warning(
                "Target %s is not binary {0,1}; found values: %s",
                self.target_column,
                sorted(target_values),
            )

        logger.info(
            "Schema validation passed (%d required columns present).",
            len(REQUIRED_COLUMNS),
        )
        return df

    def load_and_validate(self, path: str | Path) -> pd.DataFrame:
        """Convenience: load then validate in one call."""
        return self.validate_schema(self.load(path))
