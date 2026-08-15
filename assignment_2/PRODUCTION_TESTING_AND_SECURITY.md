# Objective 2.9 — Testing in Production & Security

**AIMLCZG546 · Assignment II · Group 40 · Loan Default Risk Service**

---

## Part A — Testing and experimentation in production: shadow deployment

### Why shadow, and not canary or A/B

All three are viable, but one property of credit risk decides it: **the label arrives
years late.**

Whether a loan defaults is only known over the loan's term — the dataset's `term` column
is overwhelmingly 360 months. An A/B test measures a variant against a business outcome,
so an honest A/B test of a default model would need to wait months at minimum before its
primary metric existed. Canary release has the same dependency in weaker form: it limits
blast radius, but "is the canary healthy?" still ultimately means "are its decisions
right?", which is unanswerable at the moment of the release.

**Shadow deployment needs no labels at all.** The challenger scores real production
traffic in parallel with the champion, and its output is logged but never returned. Every
comparison it supports — score agreement, decision flips, latency, error rate — is
available on day one, from the same requests, with zero exposure to any applicant.

There is also a fairness argument specific to lending. A canary or A/B test *does* act on
real applicants, so a defective challenger declines people who should have been approved.
In a regulated credit process those are adverse actions against identifiable individuals.
Shadow mode makes that impossible by construction.

### How it maps onto this codebase

The existing design makes shadow mode cheap, because `Predictor` already takes an
artifact path and the API resolves it through a FastAPI dependency:

```python
# Sketch — champion serves, challenger observes.
champion  = Predictor(CONFIG["model"]["artifact_path"])
challenger = Predictor(CONFIG["model"]["shadow_artifact_path"])

@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication) -> PredictionResponse:
    record = application.to_record()
    result = champion.score(record)[0]

    # Never blocks and never affects the response.
    background.add_task(log_shadow, request_id, record, challenger, result)

    return PredictionResponse(**result)
```

Two properties matter. The challenger runs in a background task, so a slow or broken
challenger cannot delay or fail a live request. And only the champion's result reaches
`PredictionResponse`, so the contract enforced by `tests/test_api.py` is unchanged.

### What gets compared

| Signal | How | Why it matters |
|---|---|---|
| Score distribution | `population_stability_index()` on champion vs challenger probabilities | Already implemented in `metrics.py`. A large PSI means the challenger reasons differently, not just marginally better. |
| Decision flip rate | % of requests where the `approve` / `refer` / `decline` band changes | The business-visible difference. A 2% flip rate is a tuning change; 30% is a different product. |
| Flip direction | Flips split by direction | 5% approve→decline and 5% decline→approve net to zero but mean very different things. |
| Latency | p50 / p99 per model | Sizes the infrastructure before the challenger takes traffic. |
| Error rate | Exceptions per 1,000 requests | Catches serving-time faults no offline test reached. |
| Input quality | `DataQualityChecker` over batched live requests | Detects drift or upstream schema change independently of either model. |

### Promotion path

1. **Shadow** until the sample is large enough to be meaningful, and until flip rate and
   latency are within agreed bounds. No customer impact.
2. **Canary** to a small traffic share, monitored on immediate proxies (approval rate,
   referral volume, complaint rate) rather than default outcomes.
3. **Full rollout**, with the previous artifact retained for rollback.
4. **Retrospective A/B evaluation** once outcome labels mature — the only stage that can
   measure real predictive quality, and it happens after the decision, not before it.

### Honest limits of this approach

- Shadow mode measures **agreement, not correctness**. If the champion is wrong and the
  challenger agrees with it, shadow mode reports everything is fine.
- It cannot detect **feedback-loop bias**. Declined applicants never generate a repayment
  outcome, so the training data only contains loans that were approved — the selective
  labelling (or "reject inference") problem. No amount of production testing on served
  traffic fixes a bias that lives in what never got served.
- It doubles inference cost while running.

---

## Part B — Security consideration: input validation at the API boundary

The service exposes an unauthenticated HTTP endpoint that returns a continuous
probability. That combination is the classic setup for model extraction and probing, and
input validation is the first control.

### What already works

`LoanApplication` is a Pydantic model, so FastAPI rejects malformed requests with **422
before any data reaches scikit-learn** — this is the right architecture, and
`tests/test_api.py` pins it. Verified by probing the running service: `{"LTV": -500}`
returns 422, and `{"Credit_Score": "abc"}` returns 422 naming the offending field.

### Verified gaps

Probing the same endpoint also produced these, all returning **200 OK**:

| Request | Result | Cause |
|---|---|---|
| `{"income": 1e15}` | 200, `approve` | Only `ge=0`; no upper bound |
| `{"year": 9999}` | 200, `refer` | `year` has no bounds at all |
| `{"Region": "<script>alert(1)</script>"}` | 200, `refer` | Any string is accepted as a category code |

The third is the most interesting, and it is not a cross-site scripting problem — the
value is never rendered. It is a **silent degradation** problem. `OneHotEncoder` is
configured with `handle_unknown="ignore"`, so an unrecognised code encodes as all zeros:
the feature is quietly removed from the calculation and the caller still gets a confident
probability with no warning. An attacker can therefore switch individual features off
one at a time and watch the score move, which maps the model's sensitivity without ever
triggering an error.

That behaviour was a **deliberate robustness choice** — a genuinely new region appearing
in production should not return a 500 — and the fix is not to remove it. It is to move
the decision to the edge:

- **Constrain categorical fields** with `Literal` / `Enum` of known codes, so an unknown
  code is a 422 at the API while `handle_unknown="ignore"` stays as the last-resort
  safety net for batch scoring.
- **Bound every numeric field** with `le=` as well as `ge=`, using the training ranges
  already documented in each field's description.
- **Set `extra="forbid"`** so unexpected keys are rejected rather than silently dropped.
- **Add authentication and rate limiting** (neither is present). Unlimited anonymous
  access to a scoring endpoint is what makes extraction economically feasible.
- **Consider coarsening the response.** Returning `decision` plus a risk band, rather than
  a four-decimal probability, materially reduces what an attacker can reconstruct — at the
  cost of transparency the underwriter may legitimately need.

### Secondary: model artifact integrity

`Predictor._load()` calls `joblib.load()`, which unpickles. **Unpickling executes
arbitrary code**, so anyone who can write to `models/model.joblib` achieves remote code
execution on the serving host. The artifact is trusted here because it is produced by
`scripts/train.py`, but that trust is implicit and unenforced. Mitigations: restrict write
access to the artifact directory, record a checksum at training time and verify it at
load, and treat model files from any external source as untrusted input.

---

## Summary

| Requirement | Answer |
|---|---|
| Production testing approach | Shadow deployment, promoting through canary to full rollout; chosen because default labels take months to years to mature, so no label-dependent method can gate a release |
| Security consideration | Input validation at the API boundary — Pydantic already blocks malformed types and negative values; unbounded numerics and unconstrained category codes remain, the latter causing silent feature-dropping that enables model probing |