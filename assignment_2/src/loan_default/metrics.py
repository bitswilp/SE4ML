"""Model-quality and data-quality metrics (Objective 2.8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import CONFIG

QUALITY = CONFIG["quality"]


def model_quality(y_true, y_pred, y_proba=None) -> dict:
    """Model-quality metrics (Objective 2.8a).

    accuracy / precision / recall / f1 come from hard predictions; roc_auc and
    pr_auc need probabilities. PR-AUC (average precision) is the headline metric
    under class imbalance and is used for model selection.
    """
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
        out["pr_auc"] = average_precision_score(y_true, y_proba)
    return out


def data_quality(df) -> dict:
    """Data-quality metrics (Objective 2.8b).

    Cheap, always-computable summary statistics recorded on every run so that
    a change in the input data is visible in the logs after the fact.
    """
    missing = df.isna().mean()
    return {
        "missing_fraction": float(missing.mean()),
        "worst_column_missing_fraction": float(missing.max()) if len(missing) else 0.0,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
    }


# --------------------------------------------------------------------------
# Drift detection (Objective 2.8b, second metric)
# --------------------------------------------------------------------------
# Population Stability Index: bin a reference sample, apply the SAME bin edges
# to a later sample, and measure how much probability mass moved.
#
#     PSI = sum over bins of (actual_i - expected_i) * ln(actual_i / expected_i)
#
# PSI is 0 for identical distributions and grows as they diverge. It is
# symmetric and unbounded above. The conventional bands (<0.10 stable,
# 0.10-0.25 moderate, >0.25 significant) are long-standing credit-risk rules
# of thumb rather than a standard traceable to a primary source, so they live
# in config.yaml as adjustable thresholds -- treat them as a triage aid, not a
# decision rule.

PSI_EPSILON = 1e-6  # keeps ln() finite when a bin is empty on one side


def population_stability_index(reference, current, bins: int = 10) -> float:
    """PSI between a reference and a current sample of one NUMERIC column.

    Bin edges come from the reference quantiles and are then applied unchanged
    to the current sample -- re-binning each sample separately would compare
    two different rulers and could report 0 drift for a shifted distribution.
    """
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()
    if reference.empty or current.empty:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:  # constant reference column -- nothing to compare
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    expected = np.histogram(reference, bins=edges)[0] / len(reference)
    actual = np.histogram(current, bins=edges)[0] / len(current)
    expected = np.clip(expected, PSI_EPSILON, None)
    actual = np.clip(actual, PSI_EPSILON, None)

    return float(np.sum((actual - expected) * np.log(actual / expected)))


def categorical_stability_index(reference, current) -> float:
    """PSI variant for a CATEGORICAL column: bins are the category values.

    Categories present in only one sample still contribute, via the same
    epsilon flooring, so an entirely new category registers as drift instead
    of being silently ignored.
    """
    reference = pd.Series(reference).dropna().astype(str)
    current = pd.Series(current).dropna().astype(str)
    if reference.empty or current.empty:
        return float("nan")

    categories = sorted(set(reference) | set(current))
    expected = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
    actual = current.value_counts(normalize=True).reindex(categories, fill_value=0)
    expected = np.clip(expected.to_numpy(), PSI_EPSILON, None)
    actual = np.clip(actual.to_numpy(), PSI_EPSILON, None)

    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(reference_df, current_df, columns=None, bins: int = None) -> dict:
    """Per-column PSI plus an overall verdict against the configured bands.

    Intended use in production: persist the training sample as the reference,
    then run this against each batch of live traffic. A rising max PSI is the
    signal to retrain -- the model has not changed, the world has.
    """
    bins = bins or QUALITY.get("psi_bins", 10)
    warn_at = QUALITY.get("psi_warn", 0.10)
    fail_at = QUALITY.get("psi_fail", 0.25)

    shared = [c for c in reference_df.columns if c in current_df.columns]
    columns = [c for c in (columns or shared) if c in shared]

    per_column = {}
    for column in columns:
        reference = reference_df[column]
        if pd.api.types.is_numeric_dtype(reference):
            psi = population_stability_index(reference, current_df[column], bins)
        else:
            psi = categorical_stability_index(reference, current_df[column])
        per_column[column] = round(psi, 5)

    scored = {c: v for c, v in per_column.items() if not np.isnan(v)}
    max_psi = max(scored.values()) if scored else 0.0
    if max_psi >= fail_at:
        verdict = "significant_drift"
    elif max_psi >= warn_at:
        verdict = "moderate_drift"
    else:
        verdict = "stable"

    return {
        "psi": per_column,
        "max_psi": round(max_psi, 5),
        "drifted_column": max(scored, key=scored.get) if scored else None,
        "verdict": verdict,
        "thresholds": {"warn_at": warn_at, "fail_at": fail_at},
    }
