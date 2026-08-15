"""Data-validation tests for the quality checker and drift metric (Obj 2.8b).

Each check is tested twice: once on data that should PASS, once on data
deliberately broken in the specific way that check exists to catch. A check
that never fires is worse than no check, because it manufactures confidence.
"""

import numpy as np
import pandas as pd
import pytest

from loan_default.data_quality import (
    CheckResult,
    DataQualityChecker,
    DataQualityError,
    Status,
)
from loan_default.metrics import (
    categorical_stability_index,
    data_quality,
    drift_report,
    population_stability_index,
)


@pytest.fixture
def checker() -> DataQualityChecker:
    return DataQualityChecker()


@pytest.fixture
def clean_df(raw_df) -> pd.DataFrame:
    """raw_df with its one known blemish repaired.

    raw_df draws LTV, loan_amount and property_value independently, so the
    LTV identity does not hold in it. Recomputing LTV makes the frame a valid
    baseline for consistency, without weakening any other check.
    """
    df = raw_df.copy()
    df["LTV"] = df["loan_amount"] / df["property_value"] * 100
    return df


# --------------------------------------------------------------------------
# DQ001 schema (FAIL severity)
# --------------------------------------------------------------------------
def test_schema_passes_on_full_frame(checker, clean_df):
    assert checker.check_schema(clean_df).status is Status.PASS


def test_schema_fails_when_required_column_missing(checker, clean_df):
    result = checker.check_schema(clean_df.drop(columns=["loan_amount"]))

    assert result.status is Status.FAIL
    assert result.code == "DQ001"
    assert "loan_amount" in result.detail


# --------------------------------------------------------------------------
# DQ002 completeness (WARN severity)
# --------------------------------------------------------------------------
def test_completeness_passes_when_below_threshold(checker, clean_df):
    assert checker.check_completeness(clean_df).status is Status.PASS


def test_completeness_warns_on_a_mostly_empty_column(checker, clean_df):
    df = clean_df.copy()
    df.loc[df.index[:45], "income"] = np.nan  # 45/50 = 90% missing

    result = checker.check_completeness(df)

    assert result.status is Status.WARN
    assert "income" in result.detail


def test_completeness_threshold_is_configurable(clean_df):
    """A stricter threshold must change the verdict on unchanged data."""
    df = clean_df.copy()
    df.loc[df.index[:10], "income"] = np.nan  # 20% missing

    assert DataQualityChecker().check_completeness(df).status is Status.PASS
    strict = DataQualityChecker(max_missing_fraction=0.1)
    assert strict.check_completeness(df).status is Status.WARN


# --------------------------------------------------------------------------
# DQ003 uniqueness (FAIL severity)
# --------------------------------------------------------------------------
def test_uniqueness_passes_on_distinct_rows(checker, clean_df):
    assert checker.check_uniqueness(clean_df).status is Status.PASS


def test_uniqueness_fails_on_duplicated_rows(checker, clean_df):
    doubled = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)

    result = checker.check_uniqueness(doubled)

    assert result.status is Status.FAIL
    assert result.metrics["duplicate_rows"] == 1


def test_uniqueness_fails_on_repeated_ids(checker, clean_df):
    df = clean_df.copy()
    df.loc[df.index[1], "ID"] = df.loc[df.index[0], "ID"]

    assert checker.check_uniqueness(df).status is Status.FAIL


# --------------------------------------------------------------------------
# DQ004 validity (FAIL severity)
# --------------------------------------------------------------------------
def test_validity_passes_on_plausible_values(checker, clean_df):
    assert checker.check_validity(clean_df).status is Status.PASS


def test_validity_fails_on_negative_loan_amount(checker, clean_df):
    df = clean_df.copy()
    df.loc[df.index[0], "loan_amount"] = -1000

    result = checker.check_validity(df)

    assert result.status is Status.FAIL
    assert "loan_amount" in result.detail


def test_validity_fails_on_impossible_credit_score(checker, clean_df):
    df = clean_df.copy()
    df.loc[df.index[0], "Credit_Score"] = 5000

    assert checker.check_validity(df).status is Status.FAIL


def test_validity_fails_on_non_binary_target(checker, clean_df):
    df = clean_df.copy()
    df.loc[df.index[0], "Status"] = 7

    assert checker.check_validity(df).status is Status.FAIL


def test_zero_income_is_reported_but_not_fatal(checker, clean_df):
    """Zero income is possible, so it is a note rather than a failure."""
    df = clean_df.copy()
    df.loc[df.index[0], "income"] = 0

    result = checker.check_validity(df)

    assert result.status is Status.PASS
    assert result.metrics["zero_income"] == 1


# --------------------------------------------------------------------------
# DQ005 consistency (WARN severity)
# --------------------------------------------------------------------------
def test_consistency_passes_when_ltv_identity_holds(checker, clean_df):
    assert checker.check_consistency(clean_df).status is Status.PASS


def test_consistency_warns_when_ltv_contradicts_the_other_columns(checker, clean_df):
    df = clean_df.copy()
    df["LTV"] = df["LTV"] + 25  # e.g. an upstream unit or join error

    assert checker.check_consistency(df).status is Status.WARN


# --------------------------------------------------------------------------
# DQ006 target balance (WARN severity)
# --------------------------------------------------------------------------
def test_target_balance_passes_on_a_reasonable_split(checker, clean_df):
    assert checker.check_target_balance(clean_df).status is Status.PASS


def test_target_balance_warns_when_a_class_nearly_vanishes(checker, clean_df):
    df = clean_df.copy()
    df["Status"] = 0
    df.loc[df.index[0], "Status"] = 1  # 2% minority, below the 5% floor

    assert checker.check_target_balance(df).status is Status.WARN


# --------------------------------------------------------------------------
# DQ007 missingness leakage (WARN severity)
# --------------------------------------------------------------------------
def test_missingness_leakage_passes_when_holes_are_random(checker, clean_df):
    df = clean_df.copy()
    df.loc[df.index[::2], "income"] = np.nan  # every other row, independent of target

    assert checker.check_missingness_leakage(df).status is Status.PASS


def test_missingness_leakage_warns_when_nan_pattern_encodes_the_target(
    checker, clean_df
):
    """The Assignment I finding, reproduced: NaN wherever the loan defaulted."""
    df = clean_df.copy()
    df.loc[df["Status"] == 1, "income"] = np.nan

    result = checker.check_missingness_leakage(df)

    assert result.status is Status.WARN
    assert "income" in result.metrics["suspects"]
    assert result.metrics["suspects"]["income"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# DQ008 category leakage (WARN severity)
# --------------------------------------------------------------------------
def test_category_leakage_warns_on_a_perfectly_predictive_value(checker, clean_df):
    """The credit_type == 'EQUI' pattern, reproduced on synthetic data."""
    df = clean_df.copy()
    df["Region"] = "south"
    df.loc[df["Status"] == 1, "Region"] = "ATLANTIS"  # 100% default

    result = checker.check_category_leakage(df)

    assert result.status is Status.WARN
    assert any("ATLANTIS" in key for key in result.metrics["suspects"])


def test_category_leakage_ignores_tiny_groups(checker, clean_df):
    """A single perfectly-predictive row is noise, not leakage."""
    df = clean_df.copy()
    df["Status"] = 0
    df.loc[df.index[0], ["Region", "Status"]] = ["ATLANTIS", 1]

    strict = DataQualityChecker(leakage_min_support=0.5)
    assert strict.check_category_leakage(df).status is Status.PASS


# --------------------------------------------------------------------------
# Orchestration and gating
# --------------------------------------------------------------------------
def test_run_all_returns_eight_distinct_checks(checker, clean_df):
    results = checker.run_all(clean_df)

    assert len(results) == 8
    assert len({r.code for r in results}) == 8
    assert all(isinstance(r, CheckResult) for r in results)


def test_clean_data_produces_no_failures(checker, clean_df):
    assert checker.failures(checker.run_all(clean_df)) == []


def test_raise_on_fail_blocks_corrupt_data(checker, clean_df):
    """FAIL is a gate: the pipeline must stop."""
    broken = clean_df.drop(columns=["loan_amount"])

    with pytest.raises(DataQualityError, match="DQ001"):
        checker.raise_on_fail(checker.run_all(broken))


def test_raise_on_fail_allows_warnings_through(checker, clean_df):
    """WARN is not a gate: leaky-but-intact data still trains, loudly."""
    df = clean_df.copy()
    df.loc[df["Status"] == 1, "income"] = np.nan

    results = checker.run_all(df)

    assert checker.warnings(results)  # something did warn
    checker.raise_on_fail(results)  # and it did not raise


def test_report_summarises_every_check(checker, clean_df):
    report = checker.report(checker.run_all(clean_df))

    assert report.count("\n") == 8  # 8 checks + the summary line
    assert "passed" in report


# --------------------------------------------------------------------------
# Drift metric (PSI)
# --------------------------------------------------------------------------
def test_psi_is_zero_for_identical_samples():
    values = np.random.default_rng(0).normal(size=1000)
    assert population_stability_index(values, values) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)

    small = population_stability_index(reference, rng.normal(0.2, 1, 5000))
    large = population_stability_index(reference, rng.normal(2.0, 1, 5000))

    assert 0 <= small < large


def test_psi_is_non_negative_on_random_resamples():
    rng = np.random.default_rng(1)
    psi = population_stability_index(rng.normal(size=2000), rng.normal(size=2000))

    assert psi >= 0


def test_psi_handles_a_constant_reference_column():
    """No variation means no bins to compare — must return 0, not crash."""
    assert population_stability_index([5, 5, 5, 5], [5, 5, 6, 7]) == 0.0


def test_categorical_psi_detects_a_new_category():
    reference = ["north"] * 100 + ["south"] * 100
    current = ["north"] * 100 + ["ATLANTIS"] * 100

    assert categorical_stability_index(reference, current) > 0.25


def test_drift_report_calls_a_stable_sample_stable(clean_df):
    report = drift_report(clean_df, clean_df, columns=["loan_amount", "Region"])

    assert report["verdict"] == "stable"
    assert report["max_psi"] == pytest.approx(0.0, abs=1e-6)


def test_drift_report_flags_a_shifted_column(clean_df):
    shifted = clean_df.copy()
    shifted["Credit_Score"] = shifted["Credit_Score"] - 300

    report = drift_report(clean_df, shifted, columns=["Credit_Score", "loan_amount"])

    assert report["verdict"] == "significant_drift"
    assert report["drifted_column"] == "Credit_Score"


def test_drift_report_ignores_columns_absent_from_either_frame(clean_df):
    report = drift_report(clean_df, clean_df.drop(columns=["income"]))

    assert "income" not in report["psi"]


# --------------------------------------------------------------------------
# Summary metrics
# --------------------------------------------------------------------------
def test_data_quality_summary_reports_shape_and_missingness(clean_df):
    summary = data_quality(clean_df)

    assert summary["n_rows"] == len(clean_df)
    assert summary["n_columns"] == clean_df.shape[1]
    assert 0.0 <= summary["missing_fraction"] <= 1.0


def test_data_quality_counts_duplicate_rows(clean_df):
    doubled = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)

    assert data_quality(doubled)["duplicate_rows"] == 1
