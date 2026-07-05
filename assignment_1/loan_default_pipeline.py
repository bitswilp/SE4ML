"""
loan_default_pipeline.py
========================
Loan Default Prediction - end-to-end ML pipeline.

Assignment: AIMLCZG546 - Software Engineering for Machine Learning
Group: <fill in your group no and member names>

Architectural pattern demonstrated: PIPE-AND-FILTER.
    Each stage is a "filter" that transforms the data and passes it to the next:
        raw data -> impute missing -> encode categoricals -> scale numerics -> model
    This is implemented with scikit-learn's ColumnTransformer + Pipeline, so the
    whole flow is a single composable object.

IMPORTANT NOTES / ASSUMPTIONS (verify these for your own submission):
    * Target column `Status`: this code assumes 1 = default, 0 = non-default.
      That is the conventional reading and is consistent with the data, but it is
      an INFERENCE - the dataset ships with no official data dictionary.
    * Leakage handling: `Interest_rate_spread`, `rate_of_interest`, and
      `Upfront_charges` are DROPPED. On this dataset their missingness aligns
      almost perfectly with the target (e.g. Interest_rate_spread is missing for
      exactly the defaulted rows), so keeping them would leak the answer.
    * `Gender` is dropped as a fairness / non-discrimination design choice.
    * `ID` is dropped (identifier, no predictive signal).
    * Reported metrics depend on these preprocessing choices, the fixed random
      seed, and the assumed target direction. They are not "the" answer - they
      are what THIS configuration produces.
"""

import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TARGET = "Status"
ID_AND_SENSITIVE = ["ID", "Gender"]            # identifier + protected attribute

# Two kinds of target leakage were found by inspecting the data:
#   (a) missingness leakage - these columns are missing almost exactly for the
#       defaulted rows (e.g. Interest_rate_spread is missing for 100% of defaults).
#   (b) category leakage - within `credit_type`, the value 'EQUI' has a 100%
#       default rate (all 15,298 EQUI rows default), so the column partly encodes
#       the target. Dropping the whole column is the safe choice; the remaining
#       bureaus carry only mild signal.
# NOTE: (b) being an "artifact" rather than a genuine predictor is an inference;
# the 100% default rate itself is a verified fact from the data.
LEAKAGE_COLS = [
    "Interest_rate_spread", "rate_of_interest", "Upfront_charges",  # (a)
    "credit_type",                                                  # (b)
]
RANDOM_STATE = 42
TEST_SIZE = 0.25


# --------------------------------------------------------------------------- #
# 1. DATA LOADING
# --------------------------------------------------------------------------- #
def load_data(path):
    print(f"[LOAD] Reading {path}")
    df = pd.read_csv(path)
    print(f"[LOAD] Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    balance = df[TARGET].value_counts(normalize=True).sort_index()
    print(f"[LOAD] Target balance -> "
          f"0 (non-default): {balance.get(0, 0):.1%} | "
          f"1 (default): {balance.get(1, 0):.1%}")
    return df


# --------------------------------------------------------------------------- #
# 2. DATA PROCESSING  (the Pipe-and-Filter preprocessing definition)
# --------------------------------------------------------------------------- #
def split_features_target(df):
    drop = [c for c in (ID_AND_SENSITIVE + LEAKAGE_COLS) if c in df.columns]
    print(f"[PREP] Dropping columns: {drop}")
    df = df.drop(columns=drop)

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    print(f"[PREP] {len(num_cols)} numeric, {len(cat_cols)} categorical features")
    return X, y, num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    # Numeric filter chain: impute median -> standardize
    numeric_filter = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    # Categorical filter chain: impute most-frequent -> one-hot encode
    categorical_filter = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_filter, num_cols),
        ("cat", categorical_filter, cat_cols),
    ])


# --------------------------------------------------------------------------- #
# 3. MODEL SELECTION
# --------------------------------------------------------------------------- #
def candidate_models():
    # class_weight balanced addresses the ~75/25 imbalance.
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE),
    }


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }


def select_models(X, y, num_cols, cat_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"[SPLIT] train {X_train.shape[0]:,} | test {X_test.shape[0]:,} "
          f"(stratified)")

    pre = build_preprocessor(num_cols, cat_cols)
    results, fitted = [], {}

    for name, clf in candidate_models().items():
        print(f"[TRAIN] {name} ...")
        pipe = Pipeline([("preprocess", pre), ("model", clf)])
        pipe.fit(X_train, y_train)
        results.append(evaluate(name, pipe, X_test, y_test))
        fitted[name] = pipe

    results_df = pd.DataFrame(results).set_index("model")
    return results_df, fitted, (X_test, y_test)


# --------------------------------------------------------------------------- #
# 4. OUTPUT
# --------------------------------------------------------------------------- #
def report(results_df, fitted, test_data):
    X_test, y_test = test_data
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    print("\n" + "=" * 70)
    print("MODEL COMPARISON  (test set)")
    print("=" * 70)
    print(results_df.to_string())

    # Selection rule: prioritise PR-AUC, the appropriate metric under imbalance.
    best_name = results_df["pr_auc"].idxmax()
    print(f"\n[SELECT] Best model by PR-AUC: {best_name}")
    print("         (PR-AUC / recall are prioritised over accuracy because the "
          "positive\n          class is the minority - accuracy alone is misleading.)")

    best = fitted[best_name]
    y_pred = best.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion matrix for {best_name}:")
    print("                 pred:0   pred:1")
    print(f"   actual 0    {cm[0, 0]:>8d} {cm[0, 1]:>8d}")
    print(f"   actual 1    {cm[1, 0]:>8d} {cm[1, 1]:>8d}")

    # Interpretability view (quality requirement). The best model by PR-AUC may
    # be a boosting model that exposes no importances, so we always show the
    # interpretable reference (Logistic Regression coefficients) plus Random
    # Forest importances where available.
    if "Logistic Regression" in fitted:
        print_top_features(fitted["Logistic Regression"], "Logistic Regression")
    if "Random Forest" in fitted:
        print_top_features(fitted["Random Forest"], "Random Forest")
    return best_name, best


def print_top_features(pipe, name, k=12):
    """Show the most influential features (interpretability quality req)."""
    try:
        feat_names = pipe.named_steps["preprocess"].get_feature_names_out()
    except Exception as e:
        print(f"\n[FEATURES] Could not extract feature names: {e}")
        return

    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        vals, label = model.feature_importances_, "importance"
    elif hasattr(model, "coef_"):
        vals, label = np.abs(model.coef_[0]), "|coefficient|"
    else:
        print("\n[FEATURES] Model exposes no importances/coefficients.")
        return

    order = np.argsort(vals)[::-1][:k]
    print(f"\nTop {k} features by {label} ({name}):")
    for i in order:
        print(f"   {feat_names[i]:40s} {vals[i]:.4f}")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main(path):
    df = load_data(path)
    X, y, num_cols, cat_cols = split_features_target(df)
    results_df, fitted, test_data = select_models(X, y, num_cols, cat_cols)
    best_name, best = report(results_df, fitted, test_data)

    import joblib
    out = "loan_default_model.joblib"
    joblib.dump(best, out)
    print(f"\n[SAVE] Selected model '{best_name}' saved to {out}")


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "loan_default_dataset.csv"
    main(data_path)
