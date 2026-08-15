"""Data-validation tests (Objective 2.6)."""

import pytest

from loan_default.data import (
    REQUIRED_COLUMNS,
    DataIngestion,
    SchemaValidationError,
)


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        DataIngestion().load("does/not/exist.csv")


def test_validate_schema_passes_on_valid_frame(raw_df):
    out = DataIngestion().validate_schema(raw_df)
    assert list(out.columns) == list(raw_df.columns)


def test_validate_schema_fails_on_missing_required_column(raw_df):
    broken = raw_df.drop(columns=["loan_amount"])
    with pytest.raises(SchemaValidationError):
        DataIngestion().validate_schema(broken)


def test_target_is_binary(raw_df):
    assert set(raw_df["Status"].unique()) <= {0, 1}


def test_required_columns_count():
    # 27 serving features + 1 target
    assert len(REQUIRED_COLUMNS) == 28
