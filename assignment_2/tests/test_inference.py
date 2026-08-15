"""Inference tests (Objective 2.7b).

Shape, range, invariance and directional checks on the served model. These are
behavioural tests: they assert properties that must hold for ANY acceptable
model, rather than pinning a metric value that would break on every retrain.
"""

import numpy as np
import pytest

from loan_default.data import TARGET
from loan_default.inference import Predictor

N_BATCH = 20


# --------------------------------------------------------------------------
# Loading / error handling
# --------------------------------------------------------------------------
def test_missing_artifact_raises(tmp_path):
    """Objective 1.3: a missing model must fail loudly at load, not at score."""
    with pytest.raises(FileNotFoundError):
        Predictor(tmp_path / "nope.joblib")


# --------------------------------------------------------------------------
# Output shape
# --------------------------------------------------------------------------
def test_prediction_shape(predictor, signal_df):
    """One probability, one label and one result dict per input row."""
    records = signal_df.drop(columns=[TARGET]).head(N_BATCH)

    probs = predictor.predict_proba(records)
    labels = predictor.predict(records)
    scored = predictor.score(records)

    assert probs.shape == (N_BATCH,)
    assert labels.shape == (N_BATCH,)
    assert len(scored) == N_BATCH


def test_single_dict_is_accepted(predictor, base_record):
    """A serving payload is one dict, not a DataFrame — it must work directly."""
    scored = predictor.score(base_record)

    assert len(scored) == 1
    assert set(scored[0]) == {"default_probability", "prediction", "decision"}


# --------------------------------------------------------------------------
# Output range
# --------------------------------------------------------------------------
def test_probability_in_range(predictor, signal_df):
    """predict_proba outputs must lie within [0, 1]."""
    probs = predictor.predict_proba(signal_df.drop(columns=[TARGET]).head(N_BATCH))

    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert not np.isnan(probs).any()


def test_labels_are_binary(predictor, signal_df):
    labels = predictor.predict(signal_df.drop(columns=[TARGET]).head(N_BATCH))
    assert set(np.unique(labels)) <= {0, 1}


def test_decisions_are_from_the_allowed_set(predictor, signal_df):
    scored = predictor.score(signal_df.drop(columns=[TARGET]).head(N_BATCH))
    assert {r["decision"] for r in scored} <= {"approve", "refer", "decline"}


# --------------------------------------------------------------------------
# Invariance
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "column, new_value",
    [
        ("Gender", "Female"),  # protected attribute — fair-lending requirement
        ("ID", 999_999),  # identifier — must carry no signal
        ("rate_of_interest", 99.0),  # excluded leakage column
        ("credit_type", "EQUI"),  # the 100%-default leakage category
    ],
)
def test_invariance_to_excluded_columns(predictor, base_record, column, new_value):
    """Changing an excluded column must not move the score by even a hair.

    This is the executable form of the fairness/leakage claim in the README:
    the exclusions are enforced by the pipeline, not merely documented. The
    Gender case in particular is the one a reviewer will ask about.
    """
    baseline = predictor.predict_proba(base_record)[0]

    modified = dict(base_record)
    modified[column] = new_value
    after = predictor.predict_proba(modified)[0]

    assert after == pytest.approx(baseline, abs=1e-12)


# --------------------------------------------------------------------------
# Directional expectations
# --------------------------------------------------------------------------
# The fixture plants the rule "high LTV / low Credit_Score => more likely to
# default", so a correctly wired pipeline MUST reproduce it. If these fail
# while the shape tests pass, features are reaching the model scrambled.
def test_lower_credit_score_increases_risk(predictor, base_record):
    good = dict(base_record, Credit_Score=830.0)
    poor = dict(base_record, Credit_Score=520.0)

    assert predictor.predict_proba(poor)[0] > predictor.predict_proba(good)[0]


def test_higher_ltv_increases_risk(predictor, base_record):
    low = dict(base_record, LTV=35.0)
    high = dict(base_record, LTV=115.0)

    assert predictor.predict_proba(high)[0] > predictor.predict_proba(low)[0]


def test_risk_is_monotone_across_the_credit_score_range(predictor, base_record):
    """Stronger than a two-point check: the whole curve should trend downward.

    Tree ensembles are step functions, so exact monotonicity is not guaranteed;
    the assertion is on the overall trend (ends differ in the right direction
    and no large reversal in between).
    """
    scores = [520.0, 600.0, 680.0, 760.0, 830.0]
    probs = [
        predictor.predict_proba(dict(base_record, Credit_Score=s))[0] for s in scores
    ]

    assert probs[0] > probs[-1]
    assert all(later - earlier < 0.05 for earlier, later in zip(probs, probs[1:]))


# --------------------------------------------------------------------------
# Business decision banding
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prob, expected",
    [
        (0.05, "approve"),
        (0.39, "approve"),
        (0.40, "refer"),  # boundary: refer_at is inclusive
        (0.69, "refer"),
        (0.70, "decline"),  # boundary: decline_at is inclusive
        (0.99, "decline"),
    ],
)
def test_decide_bands(predictor, prob, expected):
    """Thresholds come from config.yaml; the banding logic is pinned here."""
    assert predictor.decide(prob) == expected


def test_thresholds_are_configurable(artifact_path):
    """A stricter policy must change decisions without retraining."""
    strict = Predictor(artifact_path, decline_at=0.30, refer_at=0.10)
    assert strict.decide(0.35) == "decline"
