"""Unit tests for feature engineering (Objective 2.6 — unit tests).

These pin the *contract* FeatureEngineer must honour at serving time, because
a silent breach of any of them is train/serve skew: the model keeps returning
plausible numbers computed from the wrong columns.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from loan_default.data import SERVING_FEATURES, TARGET
from loan_default.features import FeatureEngineer

LEAKAGE_AND_ID = ["ID", "Gender", "rate_of_interest", "credit_type"]


def test_transform_has_no_nulls(raw_df):
    """Imputation must leave the matrix fully dense.

    NaNs are injected first: the raw file genuinely has missing values, so a
    clean fixture would let a missing imputation step pass unnoticed.
    """
    dirty = raw_df.copy()
    dirty.loc[0:9, "income"] = np.nan
    dirty.loc[5:14, "loan_purpose"] = np.nan

    out = FeatureEngineer().fit(dirty).transform(dirty)

    assert not np.isnan(out).any()


def test_transform_returns_one_row_per_input_row(raw_df):
    out = FeatureEngineer().fit(raw_df).transform(raw_df)
    assert out.shape[0] == len(raw_df)


def test_leakage_and_identifier_columns_are_dropped(raw_df):
    """The 27 serving features only — no ID, protected attribute, or leakage.

    Checked against the transformer's configured input columns by EXACT name.
    Substring matching would be wrong here: 'credit_type' is banned but
    'co-applicant_credit_type' is a legitimate serving feature.
    """
    fe = FeatureEngineer().fit(raw_df)

    consumed = set()
    for _name, _transformer, columns in fe.preprocessor_.transformers_:
        consumed.update(columns)

    assert consumed == set(SERVING_FEATURES)
    for banned in LEAKAGE_AND_ID + [TARGET]:
        assert banned not in consumed, f"{banned} reached the model"


def test_output_width_is_stable_for_a_single_row(raw_df):
    """A one-row serving payload must produce the same width as training.

    Width is fixed at fit time by the one-hot encoder. If it were recomputed
    per batch, a single-row request would produce fewer columns and the model
    would raise (or worse, silently misalign).
    """
    fe = FeatureEngineer().fit(raw_df)
    width = fe.transform(raw_df).shape[1]

    single = raw_df.iloc[[0]]
    assert fe.transform(single).shape == (1, width)


def test_unknown_category_does_not_raise(raw_df):
    """handle_unknown='ignore': an unseen code encodes as all-zeros.

    Serving reality — a new region or product code appears in production. The
    API must degrade (weaker prediction) rather than 500.
    """
    fe = FeatureEngineer().fit(raw_df)
    width = fe.transform(raw_df).shape[1]

    unseen = raw_df.iloc[[0]].copy()
    unseen["Region"] = "ATLANTIS"

    out = fe.transform(unseen)
    assert out.shape == (1, width)
    assert not np.isnan(out).any()


def test_extra_columns_are_ignored(raw_df):
    """Unexpected columns in a payload must not change the output."""
    fe = FeatureEngineer().fit(raw_df)
    expected = fe.transform(raw_df.iloc[[0]])

    noisy = raw_df.iloc[[0]].copy()
    noisy["some_new_upstream_column"] = 123

    np.testing.assert_allclose(fe.transform(noisy), expected)


def test_missing_serving_column_is_imputed_not_fatal(raw_df):
    """A dropped column becomes NaN, then gets imputed — same width, no NaN."""
    fe = FeatureEngineer().fit(raw_df)
    width = fe.transform(raw_df).shape[1]

    incomplete = raw_df.iloc[[0]].drop(columns=["income"])
    out = fe.transform(incomplete)

    assert out.shape == (1, width)
    assert not np.isnan(out).any()


def test_column_order_does_not_matter(raw_df):
    """Selection is by name, so a reordered payload scores identically."""
    fe = FeatureEngineer().fit(raw_df)
    expected = fe.transform(raw_df.iloc[[0]])

    shuffled = raw_df.iloc[[0]][list(reversed(raw_df.columns))]
    np.testing.assert_allclose(fe.transform(shuffled), expected)


def test_transform_before_fit_raises(raw_df):
    with pytest.raises(NotFittedError):
        FeatureEngineer().transform(raw_df)


def test_serving_feature_count_is_27():
    """Guards the derivation in data.py against an accidental config edit."""
    assert len(SERVING_FEATURES) == 27


def test_numeric_features_are_standardised(raw_df):
    """Post-scaling, numeric columns should be roughly zero-mean/unit-variance.

    Loose tolerances: the point is to catch a scaler that never ran, not to
    re-test scikit-learn's arithmetic.
    """
    fe = FeatureEngineer().fit(raw_df)
    out = pd.DataFrame(fe.transform(raw_df), columns=fe.get_feature_names_out())
    numeric_cols = [c for c in out.columns if c.startswith("num__")]

    means = out[numeric_cols].mean().abs()
    assert (means < 1e-6).all()
