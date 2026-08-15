"""Data quality checks (Objective 2.8b).

Eight named checks returning ``CheckResult`` objects with a quality code and a
PASS / WARN / FAIL status.

DESIGN: FAIL vs WARN
--------------------
The two levels answer different questions, and conflating them is what makes
data quality gates either useless (everything warns, nobody looks) or
unusable (everything fails, people disable the gate).

    FAIL -> the data is CORRUPT. Something is true of this file that cannot be
            true of valid loan data: a required column is gone, IDs repeat,
            an amount is negative. Training on it produces a meaningless
            model, so the pipeline stops.

    WARN -> the data is INTACT but the resulting MODEL may be quietly wrong:
            heavy missingness, a collapsed class balance, leakage. Training
            succeeds and the metrics may even look excellent -- which is
            precisely the danger. A human must read these.

Only FAIL blocks the run (see ``raise_on_fail``). WARN is recorded and logged.

The two leakage checks are the ones worth reading closely: they encode
findings verified quantitatively against this dataset in Assignment I, and
both still fire on the raw file today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from .config import CONFIG
from .data import REQUIRED_COLUMNS, TARGET
from .logging_config import get_logger

logger = get_logger(__name__)

QUALITY = CONFIG["quality"]

# Columns that must never be negative: a negative principal, property value,
# income or term is not "unusual data", it is impossible data.
NON_NEGATIVE_COLUMNS = [
    "loan_amount",
    "property_value",
    "income",
    "term",
    "LTV",
    "Credit_Score",
]


class Status(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class DataQualityError(RuntimeError):
    """Raised when a FAIL-severity check trips (hard gate)."""


@dataclass
class CheckResult:
    """Outcome of one check.

    ``code`` is a stable identifier (DQ001...DQ008) so a result can be
    referenced in a report or an alert without depending on the wording of
    ``detail``, which is written for humans and may change.
    """

    name: str
    status: Status
    detail: str = ""
    code: str = ""
    metrics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is not Status.FAIL

    def __str__(self) -> str:
        return f"[{self.status.value:4s}] {self.code} {self.name}: {self.detail}"


class DataQualityChecker:
    """Runs the eight checks over a raw loan-default DataFrame.

    Thresholds come from ``config.yaml`` and can be overridden per instance,
    which is what makes the checks testable: a test can set a threshold that
    the fixture is known to breach instead of having to manufacture 40%
    missingness.
    """

    def __init__(self, target_column: str = TARGET, **thresholds) -> None:
        self.target_column = target_column
        self.thresholds = {**QUALITY, **thresholds}

    # -- helpers ----------------------------------------------------------
    def _t(self, key: str):
        return self.thresholds[key]

    def _is_extreme(self, group_rate: float, base_rate: float) -> bool:
        """Is a group's default rate extreme RELATIVE TO the overall base rate?

        The comparison must be relative. On a dataset that is 98% non-default,
        an absolute test would flag every category as "perfectly predicts
        non-default" -- true, useless, and it buries the one group that
        actually leaks. A group is only interesting when it is near-certain in
        a direction the dataset as a whole is not.
        """
        high = self._t("leakage_rate")
        low = 1 - high
        if group_rate >= high and base_rate < high:
            return True
        return group_rate <= low and base_rate > low

    # -- FAIL-severity checks (data integrity) ----------------------------
    def check_schema(self, df: pd.DataFrame) -> CheckResult:
        """DQ001 — every required column is present."""
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            return CheckResult(
                "schema",
                Status.FAIL,
                f"{len(missing)} required column(s) absent: {missing}",
                "DQ001",
                {"n_missing_columns": len(missing)},
            )
        return CheckResult(
            "schema",
            Status.PASS,
            f"all {len(REQUIRED_COLUMNS)} required columns present",
            "DQ001",
            {"n_missing_columns": 0},
        )

    def check_uniqueness(self, df: pd.DataFrame) -> CheckResult:
        """DQ003 — no duplicate IDs and no duplicated rows.

        Duplicates are a FAIL rather than a WARN because they silently
        reweight training: a row appearing twice counts twice, and if it lands
        on both sides of the split it leaks the answer across the split.
        """
        dup_rows = int(df.duplicated().sum())
        dup_ids = int(df["ID"].duplicated().sum()) if "ID" in df.columns else 0
        metrics = {"duplicate_rows": dup_rows, "duplicate_ids": dup_ids}

        if dup_rows or dup_ids:
            return CheckResult(
                "uniqueness",
                Status.FAIL,
                f"{dup_rows} duplicate row(s), {dup_ids} duplicate ID(s)",
                "DQ003",
                metrics,
            )
        return CheckResult(
            "uniqueness", Status.PASS, "no duplicate rows or IDs", "DQ003", metrics
        )

    def check_validity(self, df: pd.DataFrame) -> CheckResult:
        """DQ004 — values lie inside their possible domain.

        Impossible values only (negative money, a target that is not 0/1, a
        credit score outside the configured range). Merely unusual values are
        left alone; this check must not fire on legitimate outliers.
        """
        problems: list[str] = []
        metrics: dict = {}

        for col in NON_NEGATIVE_COLUMNS:
            if col in df.columns:
                n_negative = int((df[col] < 0).sum())
                if n_negative:
                    problems.append(f"{col}: {n_negative} negative value(s)")
                    metrics[f"{col}_negative"] = n_negative

        low, high = self._t("credit_score_range")
        if "Credit_Score" in df.columns:
            out_of_range = int(
                ((df["Credit_Score"] < low) | (df["Credit_Score"] > high)).sum()
            )
            metrics["credit_score_out_of_range"] = out_of_range
            if out_of_range:
                problems.append(
                    f"Credit_Score: {out_of_range} value(s) outside [{low}, {high}]"
                )

        if self.target_column in df.columns:
            values = set(df[self.target_column].dropna().unique())
            metrics["target_values"] = sorted(str(v) for v in values)
            if not values <= {0, 1}:
                problems.append(f"target {self.target_column} is not binary {{0,1}}")
            n_null_target = int(df[self.target_column].isna().sum())
            metrics["null_targets"] = n_null_target
            if n_null_target:
                problems.append(f"{n_null_target} row(s) with a null target")

        if problems:
            return CheckResult(
                "validity", Status.FAIL, "; ".join(problems), "DQ004", metrics
            )

        # Zero income is possible (unemployed / undisclosed) so it is reported
        # for information rather than failing the run.
        note = "all values within their possible domain"
        if "income" in df.columns:
            n_zero = int((df["income"] == 0).sum())
            metrics["zero_income"] = n_zero
            if n_zero:
                note += f" (note: {n_zero} row(s) with zero income)"
        return CheckResult("validity", Status.PASS, note, "DQ004", metrics)

    # -- WARN-severity checks (ML readiness) ------------------------------
    def check_completeness(self, df: pd.DataFrame) -> CheckResult:
        """DQ002 — no column is missing more than ``max_missing_fraction``."""
        limit = self._t("max_missing_fraction")
        missing = df.isna().mean()
        offenders = missing[missing > limit].round(4).to_dict()
        metrics = {
            "overall_missing_fraction": round(float(missing.mean()), 4),
            "worst_column": (str(missing.idxmax()) if len(missing) else None),
            "worst_fraction": round(float(missing.max()), 4) if len(missing) else 0.0,
        }

        if offenders:
            return CheckResult(
                "completeness",
                Status.WARN,
                f"{len(offenders)} column(s) above {limit:.0%} missing: {offenders}",
                "DQ002",
                {**metrics, "offenders": offenders},
            )
        return CheckResult(
            "completeness",
            Status.PASS,
            f"no column above {limit:.0%} missing "
            f"(worst: {metrics['worst_column']} at {metrics['worst_fraction']:.2%})",
            "DQ002",
            metrics,
        )

    def check_consistency(self, df: pd.DataFrame) -> CheckResult:
        """DQ005 — LTV agrees with loan_amount / property_value.

        A cross-field identity: LTV should equal loan_amount / property_value
        x 100. VERIFIED to hold exactly on the raw Kaggle file, which makes it
        a sharp tripwire -- if it starts breaking, an upstream join or a unit
        change has gone wrong even though every column still looks valid on
        its own.
        """
        needed = {"LTV", "loan_amount", "property_value"}
        if not needed <= set(df.columns):
            return CheckResult(
                "consistency",
                Status.WARN,
                f"skipped; requires {sorted(needed)}",
                "DQ005",
                {},
            )

        usable = df[["LTV", "loan_amount", "property_value"]].dropna()
        usable = usable[usable["property_value"] != 0]
        if usable.empty:
            return CheckResult(
                "consistency", Status.WARN, "no comparable rows", "DQ005", {}
            )

        implied = usable["loan_amount"] / usable["property_value"] * 100
        deviation = (implied - usable["LTV"]).abs()
        breached = float((deviation > self._t("ltv_tolerance_pct")).mean())
        limit = self._t("max_inconsistent_fraction")
        metrics = {
            "inconsistent_fraction": round(breached, 4),
            "median_deviation_pct": round(float(deviation.median()), 4),
            "rows_compared": int(len(usable)),
        }

        if breached > limit:
            return CheckResult(
                "consistency",
                Status.WARN,
                f"{breached:.2%} of rows breach the LTV identity (limit {limit:.2%})",
                "DQ005",
                metrics,
            )
        return CheckResult(
            "consistency",
            Status.PASS,
            f"LTV identity holds for {1 - breached:.2%} of comparable rows",
            "DQ005",
            metrics,
        )

    def check_target_balance(self, df: pd.DataFrame) -> CheckResult:
        """DQ006 — the minority class is large enough to learn from."""
        if self.target_column not in df.columns:
            return CheckResult(
                "target_balance", Status.WARN, "target column absent", "DQ006", {}
            )

        shares = df[self.target_column].value_counts(normalize=True)
        minority = float(shares.min()) if len(shares) else 0.0
        floor = self._t("min_target_fraction")
        metrics = {
            "minority_fraction": round(minority, 4),
            "class_shares": {str(k): round(float(v), 4) for k, v in shares.items()},
        }

        if minority < floor:
            return CheckResult(
                "target_balance",
                Status.WARN,
                f"minority class is {minority:.2%}, below the {floor:.0%} floor",
                "DQ006",
                metrics,
            )
        return CheckResult(
            "target_balance",
            Status.PASS,
            f"minority class {minority:.2%} (>= {floor:.0%})",
            "DQ006",
            metrics,
        )

    def check_missingness_leakage(self, df: pd.DataFrame) -> CheckResult:
        """DQ007 — a column's MISSINGNESS must not predict the target.

        If P(default | column missing) is near 1, the NaN pattern itself
        encodes the answer. The column looks innocuous; the hole in it is the
        label. VERIFIED on this dataset for Interest_rate_spread,
        rate_of_interest and Upfront_charges (all dropped in config.yaml),
        and ALSO for property_value / LTV, which are still served.
        """
        if self.target_column not in df.columns:
            return CheckResult(
                "missingness_leakage", Status.WARN, "target absent", "DQ007", {}
            )

        support = self._t("leakage_min_support")
        target = df[self.target_column]
        base_rate = float(target.mean())
        suspects = {}

        for col in df.columns:
            if col == self.target_column:
                continue
            is_missing = df[col].isna()
            if is_missing.mean() < support or is_missing.all():
                continue
            default_rate = float(target[is_missing].mean())
            # Both directions leak: missingness that always means default, and
            # missingness that always means repaid.
            if self._is_extreme(default_rate, base_rate):
                suspects[col] = round(default_rate, 4)

        if suspects:
            return CheckResult(
                "missingness_leakage",
                Status.WARN,
                f"missingness predicts the target in {len(suspects)} column(s): "
                f"{suspects} (P(default | missing))",
                "DQ007",
                {"suspects": suspects},
            )
        return CheckResult(
            "missingness_leakage",
            Status.PASS,
            "no column's missingness predicts the target",
            "DQ007",
            {"suspects": {}},
        )

    def check_category_leakage(self, df: pd.DataFrame) -> CheckResult:
        """DQ008 — no category value is almost perfectly predictive.

        VERIFIED on this dataset: credit_type == 'EQUI' has a ~100% default
        rate across ~15K rows. Only groups with at least ``leakage_min_support``
        of the rows are considered, so genuinely rare categories are not
        flagged for having a small, noisy sample.
        """
        if self.target_column not in df.columns:
            return CheckResult(
                "category_leakage", Status.WARN, "target absent", "DQ008", {}
            )

        rate = self._t("leakage_rate")
        min_rows = max(1, int(self._t("leakage_min_support") * len(df)))
        base_rate = float(df[self.target_column].mean())
        suspects = {}

        for col in df.select_dtypes(include=["object", "string", "category"]).columns:
            grouped = df.groupby(col, observed=True)[self.target_column].agg(
                ["mean", "size"]
            )
            grouped = grouped[grouped["size"] >= min_rows]
            for value, row in grouped.iterrows():
                if self._is_extreme(float(row["mean"]), base_rate):
                    suspects[f"{col}={value}"] = {
                        "default_rate": round(float(row["mean"]), 4),
                        "rows": int(row["size"]),
                    }

        if suspects:
            return CheckResult(
                "category_leakage",
                Status.WARN,
                f"{len(suspects)} near-perfectly predictive category value(s): "
                f"{suspects}",
                "DQ008",
                {"suspects": suspects},
            )
        return CheckResult(
            "category_leakage",
            Status.PASS,
            f"no category value exceeds a {rate:.0%} default rate "
            f"with >= {min_rows} rows",
            "DQ008",
            {"suspects": {}},
        )

    # -- orchestration ----------------------------------------------------
    def run_all(self, df: pd.DataFrame) -> list[CheckResult]:
        """Run every check and log each outcome at its matching level.

        Ordered integrity-first: if the schema is broken, later checks would
        report confusing secondary symptoms of the same root cause.
        """
        logger.info("Running data quality checks on %d rows", len(df))
        results = [
            self.check_schema(df),
            self.check_completeness(df),
            self.check_uniqueness(df),
            self.check_validity(df),
            self.check_consistency(df),
            self.check_target_balance(df),
            self.check_missingness_leakage(df),
            self.check_category_leakage(df),
        ]

        for result in results:
            if result.status is Status.FAIL:
                logger.error("%s", result)
            elif result.status is Status.WARN:
                logger.warning("%s", result)
            else:
                logger.info("%s", result)
        return results

    @staticmethod
    def failures(results: list[CheckResult]) -> list[CheckResult]:
        return [r for r in results if r.status is Status.FAIL]

    @staticmethod
    def warnings(results: list[CheckResult]) -> list[CheckResult]:
        return [r for r in results if r.status is Status.WARN]

    @classmethod
    def raise_on_fail(cls, results: list[CheckResult]) -> None:
        """The gate: FAIL stops the pipeline, WARN does not."""
        failed = cls.failures(results)
        if failed:
            codes = ", ".join(f"{r.code} ({r.name})" for r in failed)
            raise DataQualityError(f"Data quality FAILED: {codes}")

    @staticmethod
    def report(results: list[CheckResult]) -> str:
        """Plain-text summary, for the run log or the assignment report."""
        lines = [str(r) for r in results]
        counts = {s.value: sum(r.status is s for r in results) for s in Status}
        lines.append(
            f"-- {counts['PASS']} passed, {counts['WARN']} warned, "
            f"{counts['FAIL']} failed"
        )
        return "\n".join(lines)
