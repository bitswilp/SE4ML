"""Shared pytest fixtures (synthetic data — no external files needed).

Nothing here touches ``data/raw/`` or ``models/model.joblib``: the suite must
pass on a clean checkout, before anyone has run ``scripts/train.py``. The
model used by the inference and API tests is therefore trained here, on
synthetic data with a signal deliberately planted in it (see ``signal_df``).
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from loan_default.data import KNOWN_COLUMNS, TARGET
from loan_default.inference import Predictor
from loan_default.model import build_pipeline

# Representative category values per categorical column (from sample_request.json).
_CATEGORICAL_SAMPLE = {
    "loan_limit": "cf",
    "Gender": "Male",
    "approv_in_adv": "nopre",
    "loan_type": "type1",
    "loan_purpose": "p1",
    "Credit_Worthiness": "l1",
    "open_credit": "nopc",
    "business_or_commercial": "nob/c",
    "Neg_ammortization": "not_neg",
    "interest_only": "not_int",
    "lump_sum_payment": "not_lpsm",
    "construction_type": "sb",
    "occupancy_type": "pr",
    "Secured_by": "home",
    "total_units": "1U",
    "credit_type": "CIB",
    "co-applicant_credit_type": "CIB",
    "age": "25-34",
    "submission_of_application": "to_inst",
    "Region": "south",
    "Security_Type": "direct",
}
_NUMERIC_SAMPLE = {
    "ID": 1,
    "year": 2019,
    "loan_amount": 116500,
    "rate_of_interest": 4.0,
    "Interest_rate_spread": 0.2,
    "Upfront_charges": 500.0,
    "term": 360.0,
    "property_value": 118000.0,
    "income": 1740.0,
    "Credit_Score": 758,
    "LTV": 98.7,
    "Status": 0,
    "dtir1": 45.0,
}


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A small DataFrame with the full 34-column raw schema."""
    rng = np.random.default_rng(42)
    n = 50
    row = {**_NUMERIC_SAMPLE, **_CATEGORICAL_SAMPLE}
    data = {}
    for col in KNOWN_COLUMNS:
        if col == "ID":
            data[col] = np.arange(n)
        elif col == "Status":
            data[col] = rng.integers(0, 2, n)
        elif col in _NUMERIC_SAMPLE:
            data[col] = rng.normal(row[col], abs(row[col]) * 0.1 + 1, n)
        else:
            data[col] = [row[col]] * n
    return pd.DataFrame(data)[KNOWN_COLUMNS]


# --------------------------------------------------------------------------
# Fixtures with a planted signal, for inference / API tests
# --------------------------------------------------------------------------
# raw_df above has a RANDOM target, which is right for schema and
# overfit-a-small-batch tests but useless for directional tests: a model
# fitted on noise has no direction to check. signal_df instead generates
# Status from a known monotone rule, so "worse credit score => higher
# P(default)" is a property the model MUST have learned if the pipeline is
# wired correctly. The planted rule is the test oracle, not a claim about
# real-world credit risk.

SIGNAL_ROWS = 600

# Columns stored as int64 in the raw CSV (verified against the file).
_INTEGER_COLUMNS = {"year", "Credit_Score", "loan_amount"}


@pytest.fixture(scope="session")
def signal_df() -> pd.DataFrame:
    """Synthetic frame whose target depends monotonically on Credit_Score/LTV."""
    rng = np.random.default_rng(7)
    n = SIGNAL_ROWS
    row = {**_NUMERIC_SAMPLE, **_CATEGORICAL_SAMPLE}

    # Integer-valued, matching the real file: year, Credit_Score and
    # loan_amount are int64 in the CSV, and LoanApplication types the first
    # two as `int`. A float fixture would have the tests failing validation
    # over an artefact of the fixture rather than a defect in the code.
    credit_score = rng.integers(500, 850, n)
    ltv = rng.uniform(30, 120, n)

    # Standardise, then combine: high LTV pushes risk up, high score pulls it
    # down. Coefficients are large so the relationship is easy to learn and the
    # directional assertions are not fighting sampling noise.
    z_score = (credit_score - credit_score.mean()) / credit_score.std()
    z_ltv = (ltv - ltv.mean()) / ltv.std()
    logit = 2.5 * z_ltv - 2.5 * z_score
    prob = 1 / (1 + np.exp(-logit))
    status = (rng.uniform(0, 1, n) < prob).astype(int)

    data = {}
    for col in KNOWN_COLUMNS:
        if col == "ID":
            data[col] = np.arange(n)
        elif col == TARGET:
            data[col] = status
        elif col == "Credit_Score":
            data[col] = credit_score
        elif col == "LTV":
            data[col] = ltv
        elif col in _INTEGER_COLUMNS:
            data[col] = rng.normal(row[col], abs(row[col]) * 0.1 + 1, n).round()
            data[col] = data[col].astype(int)
        elif col in _NUMERIC_SAMPLE:
            data[col] = rng.normal(row[col], abs(row[col]) * 0.1 + 1, n)
        else:
            data[col] = [row[col]] * n
    return pd.DataFrame(data)[KNOWN_COLUMNS]


@pytest.fixture(scope="session")
def artifact_path(signal_df, tmp_path_factory):
    """Train the real Pipeline on signal_df and persist it like train.py does.

    Session-scoped: fitting once and reusing keeps the suite fast, and every
    test that consumes it only reads.
    """
    y = signal_df[TARGET].astype(int)
    X = signal_df.drop(columns=[TARGET])
    pipeline = build_pipeline(HistGradientBoostingClassifier(random_state=0))
    pipeline.fit(X, y)

    path = tmp_path_factory.mktemp("models") / "model.joblib"
    joblib.dump(pipeline, path)
    return path


@pytest.fixture(scope="session")
def predictor(artifact_path) -> Predictor:
    """A Predictor backed by the freshly trained test artifact."""
    return Predictor(artifact_path)


@pytest.fixture(scope="session")
def base_record(signal_df) -> dict:
    """One representative application record (no target, no NaNs)."""
    return signal_df.drop(columns=[TARGET]).iloc[0].to_dict()
