"""Feature engineering: the pipe-and-filter preprocessing as a reusable transformer.

Objective 1.1 (modular design). Ports the ``build_preprocessor`` /
``split_features_target`` logic from the Assignment I ``loan_default_pipeline.py``
(the PIPE-AND-FILTER pattern) into a single scikit-learn transformer.

Why a transformer and not a loose function: fitting *selection + imputation +
encoding + scaling* into one object means the identical transform is applied at
training time and at inference time. This is what prevents train/serve skew --
the class of bug where the model sees differently-processed features in
production than it saw in training.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SERVING_FEATURES
from .logging_config import get_logger

logger = get_logger(__name__)


def build_preprocessor() -> ColumnTransformer:
    """Return the pipe-and-filter ColumnTransformer.

    Numeric filter:      median impute -> standard scale
    Categorical filter:  most-frequent impute -> one-hot encode (dense)

    One-hot uses ``handle_unknown="ignore"`` so categories unseen at fit time
    (e.g. a new region in a live request) yield an all-zero encoding instead of
    raising -- the output width stays fixed.
    """
    numeric_filter = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_filter = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_filter, NUMERIC_FEATURES),
            ("cat", categorical_filter, CATEGORICAL_FEATURES),
        ]
    )


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Selects the serving features, then applies the preprocessing filters.

    Behaviour on non-conforming input (mirrors the Assignment I service):
      * extra columns (ID, Gender, leakage columns, target) are DROPPED;
      * serving columns absent from the input become NaN and are filled by the
        imputers;
      * unknown categorical values are encoded as all-zeros.
    """

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        """Keep exactly the serving features, in a stable order."""
        return X.reindex(columns=SERVING_FEATURES)

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        logger.info("Fitting feature preprocessor on %d rows", len(X))
        self.preprocessor_ = build_preprocessor()
        self.preprocessor_.fit(self._select(X), y)
        return self

    def transform(self, X: pd.DataFrame):
        check_is_fitted(self, "preprocessor_")
        return self.preprocessor_.transform(self._select(X))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "preprocessor_")
        return self.preprocessor_.get_feature_names_out()
