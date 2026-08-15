"""ML training tests (Objective 2.7a)."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from loan_default.model import build_pipeline


def test_pipeline_has_expected_steps():
    pipe = build_pipeline(RandomForestClassifier())
    assert list(dict(pipe.named_steps)) == ["features", "model"]


def test_overfit_small_batch(raw_df):
    """A high-capacity model must (over)fit a tiny batch to near-perfect
    TRAINING accuracy. If it can't, the pipeline is wired wrong (features not
    reaching the model, label misalignment, etc.).

    This is the 'overfit small batch' variant of 2.7a; scikit-learn tree
    ensembles expose no per-epoch loss curve, so 'loss decreases' does not
    apply to them.
    """
    batch = raw_df.copy()
    y = batch["Status"].astype(int)
    X = batch.drop(columns=["Status"])

    # Unrestricted forest = high capacity; it should memorise a 50-row batch.
    pipe = build_pipeline(RandomForestClassifier(n_estimators=50, random_state=0))
    pipe.fit(X, y)

    train_acc = accuracy_score(y, pipe.predict(X))
    assert train_acc >= 0.98
