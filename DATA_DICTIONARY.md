# Data Dictionary — Loan Default Dataset

Business-meaning reference for all 34 columns of the Kaggle loan-default dataset
(148,670 rows x 34 columns) used in this project.

## How to read this dictionary

- **Provenance.** Compiled by Group 40 as a domain reference. It is *not* an
  official data dictionary shipped with the dataset.
- **Business meanings** are the intended semantics — useful domain reasoning,
  not ground truth about this particular file.
- **Example values are illustrative.** Several do not match the dataset's actual
  encoded values (see "Actual encoded values" below). Currency amounts shown with
  `₹` are illustrative; the dataset values carry no stated unit.
- **Target encoding is NOT confirmed here.** The `Status` row itself says the
  meaning depends on "the dataset's label convention." This project assumes
  `1 = default`, but that remains an *assumption* — confirm against the Kaggle
  dataset description before relying on it.

## Column reference

| Column                      | Business meaning                                                               | Example                                    | Expected relationship with `Status` / default                                                                                                                                                                              |
| --------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ID`                        | Unique identifier for the loan/application                                     | 100234                                     | **No business predictive value.** Should generally be excluded from model features.                                                                                                                                        |
| `year`                      | Year in which the loan/application originated                                  | 2019                                       | Can capture economic conditions, interest-rate environment, lending policy changes, etc. A relationship with default may change over time.                                                                                 |
| `loan_limit`                | Whether the requested loan falls within the lender's applicable loan limit     | `cf`, `ncf`                                | Loans outside standard limits may represent different risk/underwriting conditions.                                                                                                                                        |
| `Gender`                    | Applicant's gender                                                             | Male/Female                                | Potentially correlated with historical outcomes, but is a **sensitive/protected attribute** and should be treated carefully. It may be excluded from decisioning and retained for fairness auditing.                       |
| `approv_in_adv`             | Whether the applicant received an approval/qualification in advance            | `Yes` / `No`                               | An advance approval indicates that some preliminary underwriting has already occurred. Its relationship with default depends on how the lender generates this flag.                                                        |
| `loan_type`                 | Category/type of mortgage product                                              | Type 1, Type 2, Type 3                     | Different loan products have different eligibility criteria and risk profiles.                                                                                                                                             |
| `loan_purpose`              | Why the borrower wants the loan                                                | Home purchase, refinance, home improvement | Purpose can affect risk. For example, refinancing and purchasing may have different borrower/property characteristics.                                                                                                     |
| `Credit_Worthiness`         | Lender's assessment/category of borrower's creditworthiness                    | `L1`, `L2`, etc.                           | **Higher creditworthiness generally indicates lower default risk.**                                                                                                                                                        |
| `open_credit`               | Whether/open availability of credit exists for the borrower                    | Yes/No                                     | Indicates aspects of the borrower's existing credit profile. Excessive available/open credit can sometimes indicate higher exposure.                                                                                       |
| `business_or_commercial`    | Whether the loan is residential or business/commercial in nature               | Business / Commercial                      | Different loan purposes and repayment mechanisms can have different risk characteristics.                                                                                                                                  |
| `loan_amount`               | Amount borrowed/requested                                                      | ₹50 lakh                                   | Larger debt exposure can increase repayment burden, but risk depends heavily on income/property value.                                                                                                                     |
| `rate_of_interest`          | Interest rate charged on the loan                                              | 8.5%                                       | Higher rates generally increase the borrower's repayment burden and can therefore increase default risk. However, rates may also be assigned based on perceived risk, creating a reverse relationship that needs analysis. |
| `Interest_rate_spread`      | Difference between the loan's interest rate and a reference/base rate          | 1.2%                                       | Often represents the risk/market premium attached to the loan. A larger spread can indicate either higher borrower risk or market conditions.                                                                              |
| `Upfront_charges`           | Fees/charges paid or assessed at loan origination                              | ₹50,000                                    | Can reflect loan product, lender pricing and borrower/loan characteristics. Usually a weaker direct risk variable.                                                                                                         |
| `term`                      | Loan repayment period                                                          | 360 months                                 | Longer terms reduce periodic payment but increase total interest exposure; relationship with default isn't necessarily monotonic.                                                                                          |
| `Neg_ammortization`         | Whether negative amortization is allowed/expected                              | Yes/No                                     | If unpaid interest is added to principal, outstanding debt can grow, potentially increasing default risk.                                                                                                                  |
| `interest_only`             | Whether borrower initially pays only interest rather than principal + interest | Yes/No                                     | Interest-only structures can reduce initial payments but may increase longer-term repayment risk.                                                                                                                          |
| `lump_sum_payment`          | Whether the loan requires a large lump-sum/balloon payment                     | Yes/No                                     | A large future payment can create repayment/refinancing risk.                                                                                                                                                              |
| `property_value`            | Estimated/appraised value of the property securing the mortgage                | ₹80 lakh                                   | Higher property value relative to loan amount generally provides more collateral protection.                                                                                                                               |
| `construction_type`         | Type of property construction                                                  | `sb` / `mh` etc.                           | Different construction types may have different collateral/insurance characteristics.                                                                                                                                      |
| `occupancy_type`            | How the property is occupied                                                   | Owner occupied / investment / secondary    | Owner-occupied properties may have different repayment behavior from investment properties.                                                                                                                                |
| `Secured_by`                | Asset securing the loan                                                        | Real estate                                | Determines collateral backing the loan.                                                                                                                                                                                    |
| `total_units`               | Number of housing units in the property                                        | 1, 2, 4                                    | More units can indicate a multi-unit/investment property and different risk characteristics.                                                                                                                               |
| `income`                    | Borrower's reported/verified income                                            | ₹12 lakh/year                              | **Generally, higher stable income → greater repayment capacity → lower default risk.**                                                                                                                                     |
| `credit_type`               | Type/source of credit history used                                             | `EXP`, `EQUI`, etc.                        | Different credit-reporting sources can capture different aspects of borrower history.                                                                                                                                      |
| `Credit_Score`              | Credit score of the borrower                                                   | 720                                        | **One of the most intuitive risk indicators. Higher score generally → lower default probability.**                                                                                                                         |
| `co-applicant_credit_type`  | Credit information/source/type for co-applicant                                | `CIB`, `EXP` etc.                          | A co-applicant's credit profile can affect overall repayment capacity/risk.                                                                                                                                                |
| `age`                       | Age category/age of borrower                                                   | 35–44                                      | Can correlate with financial stability, income and credit history, but should be treated carefully from a fairness perspective.                                                                                            |
| `submission_of_application` | How/when the application was submitted in the underwriting process             | Online/Not Presented                       | May capture application channel/process differences that correlate with risk.                                                                                                                                              |
| `LTV`                       | **Loan-to-Value ratio** = loan amount / property value                         | 80%                                        | **Very important mortgage-risk feature. Higher LTV generally means less borrower equity and less collateral buffer → potentially higher default/loss risk.**                                                               |
| `Region`                    | Geographic region of the property/borrower                                     | North, South, etc.                         | Regional economic, property-market and employment differences can influence default risk.                                                                                                                                  |
| `Security_Type`             | Type of security/collateral associated with the loan                           | Direct/Indirect                            | Different collateral structures may have different recovery risk.                                                                                                                                                          |
| `dtir1`                     | **Debt-to-Income ratio**: debt obligations relative to borrower income         | 40%                                        | **Very important affordability indicator. Higher DTI/DTIR generally means greater repayment burden → higher default risk.**                                                                                                |
| `Status`                    | Historical loan outcome / target                                               | 0 / 1                                      | **Target variable:** whether the loan defaulted/non-defaulted, depending on the dataset's label convention.                                                                                                                |

## Actual encoded values (verified against the 148,670-row file)

These are the real values the model consumes — use these, not the illustrative
examples above, for the API and UI.

**Numeric serving features (8):** `year`, `loan_amount`, `term`,
`property_value`, `income`, `Credit_Score`, `LTV`, `dtir1`.

**Categorical serving features (19):**

| Column | Encoded values |
| --- | --- |
| `loan_limit` | cf, ncf |
| `approv_in_adv` | nopre, pre |
| `loan_type` | type1, type2, type3 |
| `loan_purpose` | p1, p2, p3, p4 |
| `Credit_Worthiness` | l1, l2 |
| `open_credit` | nopc, opc |
| `business_or_commercial` | b/c, nob/c |
| `Neg_ammortization` | neg_amm, not_neg |
| `interest_only` | int_only, not_int |
| `lump_sum_payment` | lpsm, not_lpsm |
| `construction_type` | mh, sb |
| `occupancy_type` | ir, pr, sr |
| `Secured_by` | home, land |
| `total_units` | 1U, 2U, 3U, 4U |
| `co-applicant_credit_type` | CIB, EXP |
| `age` | <25, 25-34, 35-44, 45-54, 55-64, 65-74, >74 |
| `submission_of_application` | not_inst, to_inst |
| `Region` | North, North-East, central, south |
| `Security_Type` | direct, Indriect *(dataset's own spelling)* |

## Columns dropped before modelling (7)

| Column(s) | Reason |
| --- | --- |
| `ID` | Identifier — no predictive value. |
| `Gender` | Protected attribute — excluded from decisioning. |
| `rate_of_interest`, `Interest_rate_spread`, `Upfront_charges` | **Missingness leakage** — `Interest_rate_spread` is missing for 100% of `Status==1` rows (verified). |
| `credit_type` | **Category leakage** — value `EQUI` maps to a 100% `Status==1` rate across ~15.3K rows (verified). |

`Status` is the target and is not a model input.