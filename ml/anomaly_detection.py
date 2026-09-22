"""
MindfulTech – Anomaly Detection Module
=======================================
Uses Isolation Forest to detect unusual digital usage patterns compared
with the training dataset distribution.

This is a SECONDARY ML component.  An anomaly flag does NOT mean
addiction or a clinical condition — it simply means the input is
statistically unusual relative to the dataset.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
ISO_FOREST_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")


def train_isolation_forest(X_train, contamination: float = 0.08,
                           random_state: int = 42):
    """
    Train an Isolation Forest on the (scaled) training data.

    Parameters
    ----------
    X_train : array-like
        Scaled feature matrix.
    contamination : float
        Expected proportion of anomalies in the training data.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    IsolationForest
        Fitted model.
    """
    print(f"  Training Isolation Forest (contamination={contamination}) ...")
    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
    )
    iso.fit(X_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(iso, ISO_FOREST_PATH)
    print(f"  Saved Isolation Forest -> {ISO_FOREST_PATH}")

    return iso


def detect_anomaly(features_scaled: np.ndarray) -> dict:
    """
    Check whether a single user input is anomalous.

    Parameters
    ----------
    features_scaled : np.ndarray
        1-D or 2-D scaled feature vector (same shape as training data).

    Returns
    -------
    dict with keys:
        is_anomaly     : bool
        anomaly_score  : float  (lower = more anomalous, typically < 0)
        anomaly_message: str
    """
    if not os.path.exists(ISO_FOREST_PATH):
        return {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "anomaly_message": "",
        }

    iso = joblib.load(ISO_FOREST_PATH)
    if features_scaled.ndim == 1:
        features_scaled = features_scaled.reshape(1, -1)

    prediction = iso.predict(features_scaled)[0]   # 1 = normal, -1 = anomaly
    raw_score = iso.decision_function(features_scaled)[0]

    is_anomaly = prediction == -1
    message = ""
    if is_anomaly:
        message = (
            "Unusual digital usage detected compared with typical patterns. "
            "This does not indicate a clinical condition - it simply means "
            "your reported habits are statistically uncommon relative to "
            "the dataset."
        )

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(float(raw_score), 4),
        "anomaly_message": message,
    }
