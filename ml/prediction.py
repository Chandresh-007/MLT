import os
import json
import numpy as np
import joblib
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
FEATURE_COLS = [
    "age", "screen_time", "social_media", "gaming", "short_video",
    "phone_unlocks", "notifications", "night_usage", "sleep",
    "focus_time", "exercise", "stress", "productivity",
]

def _load(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))

def predict(user_input: dict) -> dict:
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
    features = np.array([[user_input.get(c, 0) for c in FEATURE_COLS]])
    features_scaled = scaler.transform(features)
    risk_encoded = classifier.predict(features_scaled)[0]
    risk_level = label_encoder.inverse_transform([risk_encoded])[0]
    wellbeing_score = None
    if reg_model is not None:
        wb = reg_model.predict(features_scaled)[0]
        wellbeing_score = round(float(max(0, min(100, wb))), 1)
    cluster_id = int(km_model.predict(features_scaled)[0])
    cluster_info = cluster_desc.get(str(cluster_id), {})
    cluster_label = cluster_info.get("label", "Unknown")
    patterns = detect_patterns(user_input)
    recommendations = generate_recommendations(user_input, patterns)
    return {
        "risk_level": risk_level, "wellbeing_score": wellbeing_score,
        "cluster_id": cluster_id, "cluster_label": cluster_label,
        "patterns": patterns, "recommendations": recommendations,
    }

def detect_patterns(inp: dict) -> list[str]:
    patterns = []
    if inp.get("screen_time", 0) > 8: patterns.append("Very high daily screen time")
    elif inp.get("screen_time", 0) > 5: patterns.append("Above-average daily screen time")
    if inp.get("social_media", 0) > 4: patterns.append("High social media usage")
    if inp.get("gaming", 0) > 3: patterns.append("Significant gaming hours")
    if inp.get("short_video", 0) > 3: patterns.append("Considerable short-video consumption")
    if inp.get("phone_unlocks", 0) > 100: patterns.append("Frequent phone checking")
    elif inp.get("phone_unlocks", 0) > 60: patterns.append("Moderately frequent phone checking")
    if inp.get("notifications", 0) > 200: patterns.append("Very high notification volume")
    if inp.get("night_usage", 0) > 2: patterns.append("High evening/night screen usage")
    if inp.get("sleep", 0) < 6: patterns.append("Below-recommended sleep duration")
    if inp.get("focus_time", 0) < 2: patterns.append("Low daily focus/study time")
    if inp.get("exercise", 0) < 15: patterns.append("Minimal physical activity")
    if inp.get("stress", 0) > 7: patterns.append("Elevated self-reported stress")
    if inp.get("productivity", 0) < 4: patterns.append("Low self-reported productivity")
    if not patterns: patterns.append("No major concerning patterns detected")
    return patterns

def generate_recommendations(inp: dict, patterns: list[str]) -> list[str]:
    recs = []
    if inp.get("night_usage", 0) > 2:
        recs.append("Consider reducing recreational screen use before bedtime to support better sleep quality.")
    if inp.get("social_media", 0) > 4:
        recs.append("Consider scheduling short distraction-free periods during study or work hours.")
    if inp.get("focus_time", 0) < 2:
        recs.append("Try starting with a short focused session (e.g., 25 minutes) and gradually increase its duration.")
    if inp.get("sleep", 0) < 6:
        recs.append("Consider maintaining a consistent sleep schedule and aiming for 7-9 hours of sleep.")
    if inp.get("screen_time", 0) > 8:
        recs.append("Take regular screen breaks — a 5-minute break every hour can help reduce eye strain and mental fatigue.")
    if inp.get("phone_unlocks", 0) > 100:
        recs.append("Try batching your phone checks at set intervals rather than responding to every notification immediately.")
    if inp.get("exercise", 0) < 15:
        recs.append("Even a short daily walk or stretch routine can improve focus and reduce stress.")
    if inp.get("stress", 0) > 7:
        recs.append("Consider incorporating brief relaxation activities such as deep-breathing exercises into your daily routine.")
    if inp.get("gaming", 0) > 3 or inp.get("short_video", 0) > 3:
        recs.append("Setting a daily time limit for entertainment activities can help balance recreation with other priorities.")
    if inp.get("notifications", 0) > 200:
        recs.append("Reducing non-essential app notifications may help lower distractions throughout the day.")
    if not recs:
        recs.append("Your digital habits appear balanced — keep up the healthy routine!")
    return recs
