"""
MindfulTech – Prediction / Inference Module
============================================
Loads the trained ML models and generates a complete prediction result
for a single user input.

Pipeline:
    raw user input
      → derive features
      → scale
      → classify (Random Forest)
      → cluster (K-Means)
      → regress wellbeing score (Linear Regression)
      → anomaly check (Isolation Forest)
      → pattern detection
      → recommendations
      → explanation
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Import shared feature lists and derived-feature logic
from ml.preprocessing import (
    PRIMARY_FEATURE_COLS,
    FEATURE_COLS,
    add_derived_features,
)
from ml.anomaly_detection import detect_anomaly
from recommendations.engine import (
    detect_patterns,
    generate_recommendations,
    get_explanation,
)


def _load(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


def predict(user_input: dict) -> dict:
    """
    Generate a complete prediction from raw user input.

    Parameters
    ----------
    user_input : dict
        Keys matching PRIMARY_FEATURE_COLS (13 fields).

    Returns
    -------
    dict with keys:
        risk_level, wellbeing_score, cluster_id, cluster_label,
        patterns, recommendations, explanation, anomaly
    """
    # --- load artefacts ---
    scaler = _load("scaler.pkl")
    label_encoder = _load("label_encoder.pkl")
    classifier = _load("random_forest_model.pkl")
    km_model = _load("kmeans_model.pkl")

    reg_model = None
    reg_path = os.path.join(MODEL_DIR, "linear_regression_model.pkl")
    if os.path.exists(reg_path):
        reg_model = _load("linear_regression_model.pkl")

    desc_path = os.path.join(MODEL_DIR, "cluster_descriptions.json")
    with open(desc_path) as f:
        cluster_desc = json.load(f)

    # --- build feature vector with derived features ---
    row_df = pd.DataFrame([{c: user_input.get(c, 0)
                            for c in PRIMARY_FEATURE_COLS}])
    row_df = add_derived_features(row_df)
    features = row_df[FEATURE_COLS].values
    features_scaled = scaler.transform(features)

    # --- classification ---
    risk_encoded = classifier.predict(features_scaled)[0]
    risk_level = label_encoder.inverse_transform([risk_encoded])[0]

    # --- regression (wellbeing score) ---
    wellbeing_score = None
    if reg_model is not None:
        wb = reg_model.predict(features_scaled)[0]
        wellbeing_score = round(float(max(0, min(100, wb))), 1)

    # --- clustering ---
    cluster_id = int(km_model.predict(features_scaled)[0])
    cluster_info = cluster_desc.get(str(cluster_id), {})
    cluster_label = cluster_info.get("label", "Unknown")

    # --- anomaly detection ---
    anomaly = detect_anomaly(features_scaled)

    # --- patterns & recommendations ---
    patterns = detect_patterns(user_input)
    recommendations = generate_recommendations(user_input, patterns)

    # --- assemble result ---
    result = {
        "risk_level": risk_level,
        "wellbeing_score": wellbeing_score,
        "cluster_id": cluster_id,
        "cluster_label": cluster_label,
        "patterns": patterns,
        "recommendations": recommendations,
        "anomaly": anomaly,
    }

    # --- explanation (template or Bedrock) ---
    result["explanation"] = get_explanation(
        result, user_input, recommendations
    )

    return result
