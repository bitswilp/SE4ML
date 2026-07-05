# Loan Default Prediction System

## Overview

This project implements an **end-to-end machine learning pipeline** for predicting loan default risk. It demonstrates software engineering best practices for ML systems, including the **Pipe-and-Filter** architectural pattern for data preprocessing and the **Microservice** pattern for model serving.

**Course**: AIMLCZG546 - Software Engineering for Machine Learning  
**Dataset**: [Kaggle Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)

---

## Table of Contents

- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Components](#components)
- [Usage](#usage)
- [Key Design Decisions](#key-design-decisions)
- [Evaluation Metrics](#evaluation-metrics)

---

## Project Structure

```
assignment_1/
├── loan_default_pipeline.py           # ML pipeline with preprocessing & model training
├── loan_default_service.py            # Flask microservice for model serving
├── loan_default_model.joblib          # Trained model artifact (generated)
├── loan_default_architecture.png      # System architecture diagram
├── sample_request.json                # Example API requests
├── README.md                          # This file
└── Group_40.docx                      # Assignment submission document (download & edit)
```

### Submission Document

📄 **[Group_40.docx](./Group_40.docx)** — Download this file to view and edit the formal assignment submission with:
- Group member names and details
- Design documentation
- Implementation notes
- Evaluation results

---

## Architecture

### System Design Diagram

The system follows two key architectural patterns:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Pipe-and-Filter Pattern                        │
│                   (loan_default_pipeline.py)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [Raw Data] → [Impute Missing] → [Encode] → [Scale] → [Model]   │
│                                                                   │
│  • Handles missing values (median/most-frequent imputation)      │
│  • Categorical encoding (OneHotEncoder)                          │
│  • Numerical scaling (StandardScaler)                            │
│  • Trains multiple candidate models                              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Persist Model Artifact
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Microservice Pattern (loan_default_service.py)      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [Client] ──HTTP──> [Flask Service] ──> [Loaded Model]          │
│                          │                                        │
│                      GET /          → Health check               │
│                      POST /predict  → Scoring + Decision         │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dataset

**Source**: [Kaggle Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)

### Dataset Overview

- **Records**: ~614K loan applications
- **Features**: 45+ attributes (both numeric and categorical)
- **Target Variable**: `Status` (1 = Default, 0 = Non-Default)
- **Class Imbalance**: Approximately 75% non-default, 25% default

### Key Features

| Feature Category | Examples |
|---|---|
| **Loan Characteristics** | `loan_amount`, `term`, `loan_type`, `loan_purpose` |
| **Borrower Profile** | `Credit_Score`, `age`, `income`, `Credit_Worthiness` |
| **Property Information** | `property_value`, `construction_type`, `occupancy_type`, `LTV` |
| **Application Details** | `year`, `Region`, `submission_of_application`, `approv_in_adv` |

### Data Quality Considerations

**Leakage Columns Removed**:
- `Interest_rate_spread`, `rate_of_interest`, `Upfront_charges` — Missing almost exclusively for defaulted loans
- `credit_type` — Shows 100% default rate for 'EQUI' category (data artifact)

**Protected Attributes Removed**:
- `Gender` — Fairness/non-discrimination design choice

**Identifiers Removed**:
- `ID` — No predictive signal

---

## Quick Start

### Prerequisites

```bash
Python 3.8+
pip install scikit-learn pandas numpy flask joblib
```

### Step 1: Train the Model

```bash
python loan_default_pipeline.py loan_default_dataset.csv
```

**Output**:
- Console report with model comparisons and feature importance
- `loan_default_model.joblib` — Serialized trained model

### Step 2: Start the Microservice

```bash
python loan_default_service.py
```

Service runs on `http://localhost:5000`

### Step 3: Score a Loan Application

```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2019,
    "loan_amount": 116500,
    "Credit_Score": 758,
    "income": 1740,
    ...
  }'

# Batch predictions (POST with JSON array)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

### Step 4: Health Check

```bash
curl http://localhost:5000/
```

---

## Components

### 1. **loan_default_pipeline.py** — Training Pipeline

**Responsibility**: Data loading, preprocessing, model training, and evaluation

**Key Functions**:

| Function | Purpose |
|---|---|
| `load_data(path)` | Load CSV and report class balance |
| `split_features_target(df)` | Drop leakage & identifier columns, separate X/y |
| `build_preprocessor(num_cols, cat_cols)` | Create `ColumnTransformer` with parallel imputation & encoding |
| `candidate_models()` | Define three classifier candidates |
| `select_models(X, y, ...)` | Train all models with stratified cross-validation |
| `report(results_df, ...)` | Display metrics and top features |
| `main(path)` | Orchestrate full pipeline |

**Preprocessing Chain** (Pipe-and-Filter):

```python
ColumnTransformer:
  ├── Numeric Branch:
  │   ├── SimpleImputer(strategy='median')
  │   └── StandardScaler()
  └── Categorical Branch:
      ├── SimpleImputer(strategy='most_frequent')
      └── OneHotEncoder(handle_unknown='ignore')
```

**Models Trained**:
1. **Logistic Regression** — Fast, interpretable baseline
2. **Random Forest** — Ensemble, handles non-linearity
3. **Histogram Gradient Boosting** — State-of-the-art performance

**Selection Criterion**: Precision-Recall AUC (prioritizes minority class recall under imbalance)

---

### 2. **loan_default_service.py** — Microservice

**Responsibility**: Expose the trained model via HTTP REST API

**Endpoints**:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Health check, service metadata |
| `/predict` | POST | Score single or batch applications |

**Request Format** (JSON):

```json
{
  "year": 2019,
  "loan_limit": "cf",
  "loan_amount": 116500,
  "Credit_Score": 758,
  "income": 1740,
  ...
}
```

**Response Format**:

```json
{
  "default_probability": 0.3245,
  "decision": "approve"
}
```

Possible decisions:
- `"approve"` — probability < 0.40 (low risk)
- `"refer_to_manual_review"` — 0.40 ≤ probability < 0.70 (medium risk)
- `"decline"` — probability ≥ 0.70 (high risk)

**Decision Logic**:

```
IF probability >= 0.70  → "decline"
ELIF probability >= 0.40 → "refer_to_manual_review"
ELSE                     → "approve"
```

*(Thresholds are illustrative business policy; tune based on cost/benefit analysis)*

---

## Usage

### Training Workflow

```python
# 1. Data Loading
df = load_data("loan_default_dataset.csv")

# 2. Feature Engineering
X, y, num_cols, cat_cols = split_features_target(df)

# 3. Model Selection
results_df, fitted_models, test_data = select_models(X, y, num_cols, cat_cols)

# 4. Report & Save Best Model
best_model_name, best_model = report(results_df, fitted_models, test_data)
```

### Deployment Workflow

```bash
# Terminal 1: Start microservice
python loan_default_service.py

# Terminal 2: Make predictions
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2019,
    "loan_amount": 250000,
    "Credit_Score": 750,
    "income": 5000,
    ...
  }'
```

---

## Key Design Decisions

### 1. **Leakage Prevention**
- Identified columns where missingness/category distribution leaked the target
- Dropped rather than imputed to maintain statistical validity
- Documented reasoning explicitly in code comments

### 2. **Fairness & Non-Discrimination**
- Removed `Gender` to avoid algorithmic bias in lending
- Encourages compliance with fair lending regulations (ECOA, FHA)

### 3. **Imbalanced Data Handling**
- Applied `class_weight="balanced"` in classifiers
- Used stratified train-test split
- Prioritized PR-AUC over accuracy as selection metric

### 4. **Microservice Resilience**
- Model expects exactly specified columns
- Missing features are auto-imputed
- Unknown categories handled gracefully by `OneHotEncoder`
- Requests can be single objects or batches

### 5. **Interpretability**
- Reports top features by importance (Random Forest) or coefficients (Logistic Regression)
- Enables stakeholder understanding of model decisions

---

## Evaluation Metrics

The pipeline reports **7 metrics** on the test set:

| Metric | Formula | Use Case |
|---|---|---|
| **Accuracy** | (TP + TN) / Total | Overall correctness (misleading under imbalance) |
| **Precision** | TP / (TP + FP) | False alarm rate (cost of wrongly declining good loans) |
| **Recall** | TP / (TP + FN) | True positive rate (cost of approving bad loans) |
| **F1-Score** | 2 × (Precision × Recall) / (P + R) | Harmonic mean of P & R |
| **ROC-AUC** | Area under ROC curve | Discrimination ability across all thresholds |
| **PR-AUC** | Area under precision-recall curve | **(PRIMARY)** Recommended for imbalanced data |
| **Confusion Matrix** | 2×2 matrix | Fine-grained error breakdown |

**Example Output**:

```
MODEL COMPARISON  (test set)
                              accuracy   precision   recall      f1     roc_auc   pr_auc
model
Logistic Regression           0.7512      0.6234      0.5891     0.6058   0.7834   0.5742
Random Forest                 0.7698      0.6512      0.6145     0.6324   0.8021   0.6015
Hist Gradient Boosting        0.7825      0.6789      0.6389     0.6584   0.8156   0.6234  ← BEST

[SELECT] Best model by PR-AUC: Hist Gradient Boosting
```

---

## Model Artifacts

### loan_default_model.joblib

Serialized scikit-learn `Pipeline` containing:
- Preprocessing (`ColumnTransformer` with imputation & encoding)
- Fitted classifier (best model by PR-AUC)

**Created by**: `loan_default_pipeline.py`  
**Loaded by**: `loan_default_service.py`  
**Size**: ~5–10 MB (depends on model complexity)

---

## Assumptions & Caveats

1. **Target Direction**: Assumes `Status = 1` means "default" and `Status = 0` means "non-default"
   - Inferred from data patterns; dataset includes no official data dictionary
   - Verify this assumption with domain experts

2. **Decision Thresholds**: `DECLINE_AT = 0.70`, `REFER_AT = 0.40` are **illustrative only**
   - Should be calibrated by cost-benefit analysis (cost of false positive vs. false negative)
   - Tune based on business objectives and risk appetite

3. **Production Server**: Flask's built-in server is for development/demo
   - For production, use **FastAPI** + **gunicorn**/**uvicorn**
   - Add authentication, rate limiting, monitoring, and logging

4. **Model Drift**: No concept drift detection
   - Monitor model performance over time as loan portfolios evolve
   - Retrain periodically with fresh data

---

## Future Enhancements

- [ ] Add model versioning and A/B testing
- [ ] Implement SHAP for local feature attribution
- [ ] Add model drift detection and retraining pipelines
- [ ] Deploy with Docker + Kubernetes
- [ ] Add comprehensive logging and monitoring (Prometheus, ELK)
- [ ] Implement feature store for consistent preprocessing
- [ ] Add explainability dashboard for business stakeholders

---

## References

- **Scikit-Learn Documentation**: [Pipelines & Preprocessing](https://scikit-learn.org/stable/modules/compose.html)
- **ML System Design**: [ML Pipelines - Google ML Rules](https://developers.google.com/machine-learning/guides/rules-of-ml)
- **Fair Lending Regulations**: [ECOA & FHA Guidelines](https://www.consumerfinance.gov/fair-lending/)
- **Imbalanced Classification**: [Understanding Precision-Recall & ROC Curves](https://machinelearningmastery.com/roc-curves-and-auc/)

---

## Submission & Documentation

📄 **[Group_40.docx](./Group_40.docx)** — Download this file to:
- View the formal assignment submission document
- Add/edit group member names
- Add implementation notes and results
- Submit with assignment

---

## License

This project is part of the AIMLCZG546 course. See `Group_40.docx` for submission details and attribution.

---

**Last Updated**: 2026-07-05  
**Maintainer**: Group 40
