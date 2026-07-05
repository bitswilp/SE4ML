"""
loan_default_service.py
=======================
Model-as-a-Service (Microservice) for loan default prediction.

Assignment: AIMLCZG546 - Software Engineering for Machine Learning
Group: <fill in your group no and member names>

Architectural pattern demonstrated: MICROSERVICE / MODEL-AS-A-SERVICE.
    The trained model (produced by loan_default_pipeline.py, which implements the
    Pipe-and-Filter pattern) is packaged behind a small HTTP service. Any client
    - a loan-officer portal, a batch job, the decision engine - scores an
    application by POSTing JSON to /predict, without knowing anything about the
    model internals. The model can be retrained and the .joblib swapped without
    changing any client.

ENDPOINTS
    GET  /         -> health check + how many features the model expects
    POST /predict  -> accepts one application (JSON object) or many (JSON list),
                      returns default probability + a decision.

ASSUMPTIONS / CAVEATS (kept explicit on purpose):
    * "default_probability" is P(Status = 1). This rests on the inference that
      Status = 1 means "default"; the dataset ships no data dictionary, so verify.
    * The DECLINE_AT / REFER_AT thresholds below are ILLUSTRATIVE business policy,
      not values tuned or validated on data. Set them from your own cost analysis.
    * Model quality depends on loan_default_pipeline.py's preprocessing choices
      (leakage columns dropped, etc.). See that file's notes.
    * This uses Flask's built-in server, which is for development/demo only, not
      production. (FastAPI + a production server such as gunicorn/uvicorn would
      be a common alternative.)
"""

import joblib
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = "loan_default_model.joblib"

# Decision policy (non-ML business rules applied on top of the model score).
# These are placeholders - replace with thresholds your team can justify.
DECLINE_AT = 0.70   # probability at/above which the application is declined
REFER_AT = 0.40     # probability at/above which it goes to manual review

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
# The fitted preprocessor knows exactly which input columns it expects.
EXPECTED_COLUMNS = list(model.named_steps["preprocess"].feature_names_in_)


def decide(prob):
    if prob >= DECLINE_AT:
        return "decline"
    if prob >= REFER_AT:
        return "refer_to_manual_review"
    return "approve"


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "loan-default-scoring",
        "model_expects_n_features": len(EXPECTED_COLUMNS),
        "decline_at": DECLINE_AT,
        "refer_at": REFER_AT,
    })


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "Body must be JSON (an object or a list of objects)."}), 400

    records = payload if isinstance(payload, list) else [payload]

    # Build a DataFrame with exactly the columns the pipeline expects.
    # Any feature the caller omits becomes NaN and is handled by the pipeline's
    # imputers; unknown categories are handled by OneHotEncoder(handle_unknown).
    df = pd.DataFrame(records).reindex(columns=EXPECTED_COLUMNS)

    probs = model.predict_proba(df)[:, 1]
    results = [
        {"default_probability": round(float(p), 4), "decision": decide(p)}
        for p in probs
    ]
    return jsonify(results[0] if len(results) == 1 else results)


if __name__ == "__main__":
    # host=0.0.0.0 so it is reachable locally for the demo.
    app.run(host="0.0.0.0", port=5000)
