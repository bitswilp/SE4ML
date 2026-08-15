# Loan Default Risk — Assignment II (Group 40)

Production-oriented refactor of the Assignment I loan-default / credit-risk
model: modular code, a REST API, and a pytest-based test suite.

> Group No: 40 · Course: AIMLCZG546 — Software Engineering for Machine Learning
> Members / contribution table: see the submission report.

## Project structure

```
group40_assignment2/
├── config/config.yaml          # paths, hyperparameters, quality thresholds
├── data/{raw,processed}/       # datasets (gitignored)
├── models/                     # saved artifacts (gitignored)
├── notebooks/                  # research_prototype.ipynb (research code, Obj 1.2)
├── scripts/
│   ├── train.py                # run the training pipeline
│   └── run_api.py              # launch the REST API
├── src/loan_default/
│   ├── config.py               # load config.yaml
│   ├── logging_config.py       # centralised logging (Obj 1.3)
│   ├── data.py                 # ingestion + schema validation
│   ├── features.py             # feature engineering transformer
│   ├── model.py                # training + persistence
│   ├── inference.py            # load artifact + predict
│   ├── data_quality.py         # DataQualityChecker (ported from Asgmt I)
│   ├── metrics.py              # model + data quality metrics (Obj 2.8)
│   └── service.py              # FastAPI app (Obj 1.5)
├── tests/                      # pytest suite (Obj 2.6, 2.7)
├── pyproject.toml              # black / isort / pytest config (Obj 1.4)
└── .flake8                     # flake8 config (Obj 1.4)
```

## Requirement -> file map

| Objective | Where |
|-----------|-------|
| 1.1 Modular OOP design      | `src/loan_default/{data,features,model,inference}.py` |
| 1.2 Research vs production  | `notebooks/research_prototype.ipynb` vs `src/loan_default/` |
| 1.3 Logging + error handling| `logging_config.py`, used in data / model / inference |
| 1.4 Formatting + linting    | `pyproject.toml`, `.flake8` |
| 1.5 REST API                | `service.py`, `scripts/run_api.py` |
| 2.6 >= 2 test types         | `tests/` (data-validation, unit, integration) |
| 2.7 ML training/inference   | `tests/test_model.py`, `tests/test_inference.py` |
| 2.8 Model + data metrics    | `metrics.py` |
| 2.9 Production + security    | this README (sections below) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python scripts/train.py                 # train + save model.joblib
python scripts/run_api.py               # serve at http://127.0.0.1:8000/docs
pytest                                  # run the test suite
black . && isort . && flake8            # format + lint
```

## Objective 2.9 — production testing & security

Full write-up: [`docs/production_testing_and_security.md`](docs/production_testing_and_security.md).

- **Production testing strategy: shadow deployment**, promoting through canary to full
  rollout. Chosen over A/B testing because a default label matures over the loan term
  (mostly 360 months here), so no label-dependent method can gate a release. The
  challenger scores live traffic in a background task and its output is logged but never
  returned, giving score-agreement, decision-flip, latency and error comparisons on day
  one with zero applicant exposure.
- **Security consideration: input validation at the API boundary.** Pydantic already
  rejects malformed types and negative amounts with 422 before anything reaches
  scikit-learn. Verified gaps remain: numeric fields are unbounded above, and any string
  is accepted as a category code — which `handle_unknown="ignore"` then encodes as all
  zeros, silently dropping the feature while still returning a confident score. Fix is to
  constrain codes with `Literal`/`Enum` and bound numerics at the edge, keeping the
  encoder's tolerance as a batch-path safety net, plus authentication and rate limiting.