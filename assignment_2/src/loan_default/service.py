"""REST API + UI exposing loan-default inference (Objective 1.5).

Endpoints
    GET  /         -> the scoring UI (static HTML)
    GET  /health   -> liveness + whether a trained artifact is present
    POST /predict  -> score one application, return probability + decision
    GET  /docs      -> auto-generated OpenAPI documentation (FastAPI)

Run:  python scripts/run_api.py   (or: uvicorn loan_default.service:app --reload)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import CONFIG
from .inference import Predictor
from .logging_config import get_logger

logger = get_logger(__name__)

ARTIFACT_PATH = Path(CONFIG["model"]["artifact_path"])
# index.html ships inside the package, next to this module, so the UI travels
# with the code rather than depending on a directory at the project root.
STATIC_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Loan Default Risk API",
    version="1.0.0",
    description="Scores a loan application and returns a default-risk decision.",
)

_predictor: Optional[Predictor] = None


def get_predictor() -> Predictor:
    """Lazy, cached model load. 503 if no artifact has been trained yet."""
    global _predictor
    if _predictor is None:
        try:
            _predictor = Predictor()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="Model not trained yet. Run scripts/train.py first.",
            ) from exc
    return _predictor


def _biz(risk: str, driver: bool = False) -> dict:
    """Attach the business reading of a field to its JSON schema.

    ``x_risk`` states the expected direction of travel between the field and
    default risk; ``x_driver`` marks the fields DATA_DICTIONARY.MD singles out
    as primary risk indicators. Both are custom OpenAPI keys, so /docs and the
    UI pick the reasoning up from this one definition.

    These are the *expected* domain relationships from the dictionary, not
    coefficients measured on this model. The model may well have learned
    something different -- see the caveats on LoanApplication.
    """
    extra: dict = {"x_risk": risk}
    if driver:
        extra["x_driver"] = True
    return extra


class LoanApplication(BaseModel):
    """One loan application to be scored: the 27 serving features.

    FOR THE REVIEWING MANAGER
    -------------------------
    Every field carries a plain-English description and its permitted codes,
    all of which render in the /docs page. Each also has a typical value
    pre-filled as its default, so you can score a realistic application by
    editing only the fields you actually have to hand.

    The response is a probability of default plus one of three decisions,
    banded by ``config/config.yaml``:

        below 0.40      -> approve
        0.40 to < 0.70  -> refer to an underwriter
        0.70 and above  -> decline

    Those cut-offs are illustrative policy, NOT tuned on the data. Change them
    in config.yaml to match the bank's risk appetite; /health reports the
    values currently in force.

    CAVEATS THAT AFFECT HOW FAR YOU SHOULD TRUST A SCORE
    ----------------------------------------------------
    * The source dataset ships no data dictionary. The code meanings below are
      the conventional mortgage-industry reading of each abbreviation, not an
      official definition. Confirm them against the bank's own product
      catalogue before relying on this for a real lending decision.
    * Codes are case-sensitive and must match the training data exactly
      ("south" is lower-case, "North" is capitalised). An unrecognised code
      does NOT raise an error -- it is silently treated as "no information",
      which weakens the prediction without warning you. Copy codes verbatim.
    * Some fields the bank does hold are deliberately not inputs. Gender is
      excluded as a protected attribute (fair-lending). rate_of_interest,
      Interest_rate_spread, Upfront_charges and credit_type are excluded
      because they leak the outcome: they are only known once the loan has
      been priced or has already gone bad, so including them would inflate
      accuracy in testing and collapse in production.
    * This model scores default RISK. It is a decision aid, not an approval
      authority, and gives no reason codes for an adverse-action notice.
    """

    model_config = {"populate_by_name": True}

    # --- Loan terms -------------------------------------------------------
    year: int = Field(
        2019,
        description=(
            "Origination year. Every training row is 2019, so this field "
            "carries no predictive signal; leave it as-is."
        ),
        json_schema_extra=_biz(
            "Origination year normally captures the economic and interest-rate "
            "environment, which shifts default rates over time. Here it is "
            "constant, so it cannot discriminate between applications."
        ),
    )
    loan_amount: float = Field(
        116500,
        ge=0,
        description=(
            "Principal advanced, in currency units. Training range roughly "
            "16,500 to 3,576,500 (median about 296,500)."
        ),
        json_schema_extra=_biz(
            "A larger advance means greater exposure and a heavier repayment "
            "burden, but on its own it says little: judge it against income "
            "and property value, i.e. through DTI and LTV."
        ),
    )
    term: float = Field(
        360.0,
        ge=0,
        description=(
            "Repayment term in MONTHS. 360 = a 30-year mortgage and is by far "
            "the most common value; the data runs from 96 to 360."
        ),
        json_schema_extra=_biz(
            "A longer term lowers each instalment (helping affordability) but "
            "raises total interest exposure. The effect on default is not "
            "necessarily one-directional."
        ),
    )
    loan_type: str = Field(
        "type1",
        description=(
            "Product family, anonymised in the source data: 'type1' (most "
            "common), 'type2', 'type3'. The underlying products are not "
            "identified, so treat this as an opaque product label."
        ),
        json_schema_extra=_biz(
            "Product families carry different eligibility criteria and risk "
            "profiles. Which way each code leans cannot be reasoned about "
            "here, because the products are not identified."
        ),
    )
    loan_purpose: str = Field(
        "p1",
        description=(
            "Purpose of borrowing, anonymised: 'p1', 'p2', 'p3', 'p4'. "
            "p3 and p4 dominate the data. Actual purposes are not disclosed."
        ),
        json_schema_extra=_biz(
            "Purpose shifts risk -- a purchase, a refinance and a home "
            "improvement attract different borrowers and properties. The codes "
            "are anonymised, so the direction cannot be reasoned about here."
        ),
    )
    loan_limit: str = Field(
        "cf",
        description=(
            "'cf' = conforming (meets standard lending limits/criteria), "
            "'ncf' = non-conforming. About 93% of the data is 'cf'."
        ),
        json_schema_extra=_biz(
            "Non-conforming loans sit outside standard limits and so are "
            "written under different underwriting conditions, which can carry "
            "a different risk profile."
        ),
    )
    approv_in_adv: str = Field(
        "nopre",
        description=(
            "Was the applicant pre-approved before applying? "
            "'pre' = yes, 'nopre' = no (about 84% of cases)."
        ),
        json_schema_extra=_biz(
            "Pre-approval means some preliminary underwriting has already "
            "happened. Whether that signals lower risk depends entirely on how "
            "the lender generates the flag."
        ),
    )
    business_or_commercial: str = Field(
        "nob/c",
        description=(
            "Is this business/commercial rather than personal lending? "
            "'b/c' = business or commercial, 'nob/c' = not (about 86%)."
        ),
        json_schema_extra=_biz(
            "Business and commercial lending is repaid from trading income "
            "rather than salary, so its risk characteristics differ from "
            "residential lending."
        ),
    )

    # --- Product features that raise or defer the repayment burden --------
    Neg_ammortization: str = Field(
        "not_neg",
        description=(
            "Negative amortisation -- payments can be lower than the interest "
            "accruing, so the balance GROWS. 'neg_amm' = yes (about 10% of "
            "the data), 'not_neg' = no. A recognised risk feature."
        ),
        json_schema_extra=_biz(
            "Raises risk. Unpaid interest is capitalised onto the principal, "
            "so the debt can grow instead of amortising and the borrower may "
            "never build equity."
        ),
    )
    interest_only: str = Field(
        "not_int",
        description=(
            "Interest-only period, with principal deferred. "
            "'int_only' = yes (about 5%), 'not_int' = no."
        ),
        json_schema_extra=_biz(
            "Eases payments now at the cost of later risk: nothing is repaid "
            "off the principal, so the burden jumps when the interest-only "
            "period ends."
        ),
    )
    lump_sum_payment: str = Field(
        "not_lpsm",
        description=(
            "Is a lump-sum / balloon repayment due at the end of the term? "
            "'lpsm' = yes (about 2%), 'not_lpsm' = no."
        ),
        json_schema_extra=_biz(
            "Raises risk at maturity. A large balloon payment must be met from "
            "savings or refinanced, and refinancing may not be available on "
            "acceptable terms when it falls due."
        ),
    )
    open_credit: str = Field(
        "nopc",
        description=(
            "Open credit line on the account. 'opc' = yes, 'nopc' = no. "
            "Very rare in training (556 of 148,670 rows), so a value of "
            "'opc' rests on almost no evidence."
        ),
        json_schema_extra=_biz(
            "Reflects the borrower's wider credit profile. Substantial "
            "available credit can mean flexibility, but also room to take on "
            "further exposure."
        ),
    )

    # --- Property and collateral ------------------------------------------
    property_value: float = Field(
        118000.0,
        ge=0,
        description=(
            "Appraised value of the property securing the loan. Training "
            "median about 418,000. Missing for roughly 10% of training rows."
        ),
        json_schema_extra=_biz(
            "The collateral behind the loan, and the denominator of LTV. A "
            "high value relative to the advance gives more protection if the "
            "loan has to be recovered."
        ),
    )
    LTV: float = Field(
        98.7,
        ge=0,
        description=(
            "Loan-to-value ratio as a PERCENTAGE (loan_amount / "
            "property_value x 100). Typically 60-90; above 100 means the debt "
            "exceeds the collateral. The training data contains extreme "
            "outliers (up to 7831), so its upper end is not clean."
        ),
        json_schema_extra=_biz(
            "Higher LTV means less borrower equity and a thinner collateral "
            "buffer, raising both the chance of default and the loss if it "
            "happens. Above 100% the debt exceeds the security outright.",
            driver=True,
        ),
    )
    occupancy_type: str = Field(
        "pr",
        description=(
            "'pr' = primary residence (93% of data), 'ir' = investment "
            "property, 'sr' = secondary/second home."
        ),
        json_schema_extra=_biz(
            "Borrowers tend to prioritise the home they live in. Investment "
            "and second properties are more readily given up when finances "
            "tighten."
        ),
    )
    total_units: str = Field(
        "1U",
        description=(
            "Number of dwelling units in the property, as a code: '1U', "
            "'2U', '3U', '4U'. Almost all training rows are '1U'."
        ),
        json_schema_extra=_biz(
            "Multi-unit properties often signal an investment profile, where "
            "repayment leans on rental income rather than salary."
        ),
    )
    construction_type: str = Field(
        "sb",
        description=(
            "'sb' = site-built, 'mh' = manufactured home. Effectively "
            "constant in training (only 33 'mh' rows of 148,670), so this "
            "field carries almost no signal."
        ),
        json_schema_extra=_biz(
            "Construction type affects how the collateral holds its value and "
            "how readily it can be insured or resold. Too rare here to "
            "contribute meaningfully."
        ),
    )
    Secured_by: str = Field(
        "home",
        description=(
            "Collateral type: 'home' or 'land'. As above, only 33 training "
            "rows are 'land' -- near-constant, so effectively uninformative."
        ),
        json_schema_extra=_biz(
            "Determines what asset actually backs the loan, and so what can be "
            "recovered on default. Too rare here to contribute meaningfully."
        ),
    )
    Security_Type: str = Field(
        "direct",
        description=(
            "Security arrangement: 'direct' or 'Indriect' (the misspelling "
            "is in the source data and must be reproduced exactly). "
            "Near-constant, so effectively uninformative."
        ),
        json_schema_extra=_biz(
            "Direct security can be enforced against the asset itself; "
            "indirect arrangements add a step and so more recovery risk. Too "
            "rare here to contribute meaningfully."
        ),
    )

    # --- Applicant --------------------------------------------------------
    income: float = Field(
        1740.0,
        ge=0,
        description=(
            "Applicant's gross MONTHLY income (the scale of the data implies "
            "monthly, not annual). Training median about 5,760. Missing for "
            "roughly 6% of training rows."
        ),
        json_schema_extra=_biz(
            "Higher stable income means greater capacity to keep servicing the "
            "debt, so lower default risk. It is also the denominator of the "
            "debt-to-income ratio.",
            driver=True,
        ),
    )
    Credit_Score: int = Field(
        758,
        ge=0,
        le=900,
        description=(
            "Credit score as recorded in this dataset, spanning 500-900 "
            "(note: not the 300-850 FICO scale). Higher is better."
        ),
        json_schema_extra=_biz(
            "The most direct summary of past repayment behaviour. A higher "
            "score generally means a lower probability of default.",
            driver=True,
        ),
    )
    Credit_Worthiness: str = Field(
        "l1",
        description=(
            "Lender's internal creditworthiness grade: 'l1' (96% of data, "
            "the stronger grade) or 'l2'. Grading criteria are not disclosed."
        ),
        json_schema_extra=_biz(
            "The lender's own grade. The stronger grade ('l1') generally "
            "indicates lower default risk, though the grading criteria behind "
            "it are not disclosed.",
            driver=True,
        ),
    )
    co_applicant_credit_type: str = Field(
        "CIB",
        alias="co-applicant_credit_type",
        description=(
            "Credit bureau that supplied the CO-APPLICANT's file: 'CIB' or "
            "'EXP'. Split roughly 50/50. Note the applicant's own equivalent "
            "(credit_type) is excluded as a leakage field."
        ),
        json_schema_extra=_biz(
            "A co-applicant adds repayment capacity, so their credit profile "
            "bears on overall risk. This field names only the bureau, not the "
            "quality of that profile."
        ),
    )
    age: str = Field(
        "25-34",
        description=(
            "Applicant age BAND, not a number: '<25', '25-34', '35-44', "
            "'45-54', '55-64', '65-74', '>74'. Must be one of these exact "
            "strings."
        ),
        json_schema_extra=_biz(
            "Age correlates with financial stability, accumulated assets and "
            "length of credit history. FAIRNESS NOTE: age is a sensitive "
            "attribute and, unlike Gender, it is NOT excluded from this model "
            "-- weigh any age-driven effect carefully."
        ),
    )
    dtir1: float = Field(
        45.0,
        description=(
            "Debt-to-income ratio as a PERCENTAGE of income committed to "
            "debt service. Training range 5-61, median 39. Missing for "
            "roughly 16% of training rows."
        ),
        json_schema_extra=_biz(
            "The core affordability test. The more of an income already "
            "committed to debt, the less room to absorb a shock, so higher DTI "
            "means higher default risk.",
            driver=True,
        ),
    )
    submission_of_application: str = Field(
        "to_inst",
        description=(
            "Channel: 'to_inst' = submitted directly to the institution "
            "(about 64%), 'not_inst' = submitted through an intermediary."
        ),
        json_schema_extra=_biz(
            "Channel can proxy for how the application was assembled and "
            "checked; broker-sourced and direct business often differ in "
            "profile."
        ),
    )
    Region: str = Field(
        "south",
        description=(
            "Geographic region. Exact strings, with the inconsistent casing "
            "of the source data: 'North', 'south', 'central', 'North-East'. "
            "'North' and 'south' cover 93% of the data."
        ),
        json_schema_extra=_biz(
            "Regional employment, property markets and local economic "
            "conditions all move default rates, and affect what collateral "
            "recovers if a loan goes bad."
        ),
    )

    def to_record(self) -> dict:
        """Dict keyed by the real dataset column names (alias form)."""
        return self.model_dump(by_alias=True)


class PredictionResponse(BaseModel):
    """What the manager reads back for a scored application."""

    default_probability: float = Field(
        description=(
            "Estimated probability that this loan defaults, 0.0 to 1.0. "
            "This is the number to reason about -- it is a risk ranking, not "
            "a guarantee about this individual borrower."
        )
    )
    prediction: int = Field(
        description=(
            "Hard label at the fixed 0.5 cut-off: 1 = predicted default, "
            "0 = predicted repay. Provided for reference/metrics only; it "
            "ignores the business thresholds, so prefer 'decision' below."
        )
    )
    decision: str = Field(
        description=(
            "The recommended action under current policy: 'approve' "
            "(probability below refer_at), 'refer' to an underwriter, or "
            "'decline'. Thresholds come from config.yaml and are reported by "
            "/health. A decision aid, not an automatic approval."
        )
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_artifact_present": ARTIFACT_PATH.exists(),
        "decline_at": CONFIG["decision"]["decline_at"],
        "refer_at": CONFIG["decision"]["refer_at"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    application: LoanApplication,
    predictor: Predictor = Depends(get_predictor),
) -> PredictionResponse:
    try:
        result = predictor.score(application.to_record())[0]
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    return PredictionResponse(**result)


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return (
            "<h1>Loan Default Risk API</h1>"
            "<p>UI file not found. See <a href='/docs'>/docs</a>.</p>"
        )
    return index.read_text(encoding="utf-8")
