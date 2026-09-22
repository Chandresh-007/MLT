"""
MindfulTech – Flask Application
================================
Main web application serving all routes for the digital-wellbeing
analysis platform.
"""

import os
import json
import uuid
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, flash,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.secret_key = os.environ.get("SECRET_KEY", "mindfultech-dev-secret-key-change-in-prod")

# Load optional .env file (for Bedrock credentials)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from database.db import (
    init_db, save_habit, save_prediction,
    get_latest_entry, get_history, get_prediction_by_habit_id,
    get_dashboard_stats,
)
from ml.prediction import predict, FEATURE_COLS

init_db()


# --------------- helpers ---------------

def _session_id():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


# --------------- routes ---------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    entry = get_latest_entry(_session_id())
    return render_template("dashboard.html", entry=entry)


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "GET":
        return render_template("analyze.html")

    try:
        habit_data = {
            "age": int(request.form.get("age", 20)),
            "screen_time": float(request.form.get("screen_time", 0)),
            "social_media": float(request.form.get("social_media", 0)),
            "gaming": float(request.form.get("gaming", 0)),
            "short_video": float(request.form.get("short_video", 0)),
            "phone_unlocks": int(request.form.get("phone_unlocks", 0)),
            "notifications": int(request.form.get("notifications", 0)),
            "night_usage": float(request.form.get("night_usage", 0)),
            "sleep": float(request.form.get("sleep", 7)),
            "focus_time": float(request.form.get("focus_time", 0)),
            "exercise": int(request.form.get("exercise", 0)),
            "stress": int(request.form.get("stress", 5)),
            "productivity": int(request.form.get("productivity", 5)),
        }
    except (ValueError, TypeError):
        flash("Please enter valid numbers for all fields.", "danger")
        return redirect(url_for("analyze"))

    # --- server-side validation ---
    errors = []
    if not (10 <= habit_data["age"] <= 80):
        errors.append("Age must be between 10 and 80.")
    if not (0 <= habit_data["screen_time"] <= 24):
        errors.append("Screen time must be between 0 and 24 hours.")
    if not (0 <= habit_data["sleep"] <= 24):
        errors.append("Sleep must be between 0 and 24 hours.")
    if not (1 <= habit_data["stress"] <= 10):
        errors.append("Stress must be between 1 and 10.")
    if not (1 <= habit_data["productivity"] <= 10):
        errors.append("Productivity must be between 1 and 10.")
    if not (0 <= habit_data["exercise"] <= 1440):
        errors.append("Exercise must be between 0 and 1440 minutes.")

    if errors:
        for e in errors:
            flash(e, "danger")
        return redirect(url_for("analyze"))

    sid = _session_id()
    habit_id = save_habit(sid, habit_data)

    try:
        result = predict(habit_data)
    except Exception as e:
        flash(f"Prediction error: {e}. Make sure models are trained.", "danger")
        return redirect(url_for("analyze"))

    save_prediction(habit_id, result)
    return redirect(url_for("results", habit_id=habit_id))


@app.route("/results/<int:habit_id>")
def results(habit_id):
    entry = get_prediction_by_habit_id(habit_id)
    if entry is None:
        flash("Result not found.", "warning")
        return redirect(url_for("analyze"))
    return render_template("results.html", entry=entry)


@app.route("/analytics")
def analytics():
    return render_template("analytics.html")


@app.route("/history")
def history():
    entries = get_history(_session_id(), limit=100)
    return render_template("history.html", entries=entries)


@app.route("/models")
@app.route("/modules")
def models_page():
    metrics_path = os.path.join(BASE_DIR, "models", "model_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    cluster_path = os.path.join(BASE_DIR, "models", "cluster_descriptions.json")
    clusters = {}
    if os.path.exists(cluster_path):
        with open(cluster_path) as f:
            clusters = json.load(f)

    return render_template("models.html", metrics=metrics, clusters=clusters)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/api/chart-data")
def chart_data():
    history_list = get_history(_session_id(), limit=30, chronological=True)
    labels = []
    screen_time = []
    social_media = []
    focus_time = []
    sleep = []
    productivity = []
    risk_levels = []
    gaming = []
    short_video = []
    night_usage = []
    exercise = []
    stress = []

    risk_map = {"Low": 1, "Moderate": 2, "High": 3}

    for i, entry in enumerate(history_list):
        labels.append(f"Entry {i + 1}")
        screen_time.append(entry.get("screen_time", 0))
        social_media.append(entry.get("social_media", 0))
        focus_time.append(entry.get("focus_time", 0))
        sleep.append(entry.get("sleep", 0))
        productivity.append(entry.get("productivity", 0))
        risk_levels.append(risk_map.get(entry.get("risk_level", "Low"), 1))
        gaming.append(entry.get("gaming", 0))
        short_video.append(entry.get("short_video", 0))
        night_usage.append(entry.get("night_usage", 0))
        exercise.append(entry.get("exercise", 0))
        stress.append(entry.get("stress", 0))

    return jsonify({
        "labels": labels,
        "screen_time": screen_time,
        "social_media": social_media,
        "focus_time": focus_time,
        "sleep": sleep,
        "productivity": productivity,
        "risk_levels": risk_levels,
        "gaming": gaming,
        "short_video": short_video,
        "night_usage": night_usage,
        "exercise": exercise,
        "stress": stress,
    })


# --------------- Helpers for REST API ---------------

def _clean_for_json(obj):
    """Recursively convert NumPy scalars and arrays to native Python types."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _validate_habit_data(data: dict) -> tuple[dict | None, list[str]]:
    """Extract and validate 13 habit fields from dict payload."""
    errors = []
    try:
        habit_data = {
            "age": int(data.get("age", 20)),
            "screen_time": float(data.get("screen_time", 0)),
            "social_media": float(data.get("social_media", 0)),
            "gaming": float(data.get("gaming", 0)),
            "short_video": float(data.get("short_video", 0)),
            "phone_unlocks": int(data.get("phone_unlocks", 0)),
            "notifications": int(data.get("notifications", 0)),
            "night_usage": float(data.get("night_usage", 0)),
            "sleep": float(data.get("sleep", 7)),
            "focus_time": float(data.get("focus_time", 0)),
            "exercise": int(data.get("exercise", 0)),
            "stress": int(data.get("stress", 5)),
            "productivity": int(data.get("productivity", 5)),
        }
    except (ValueError, TypeError) as ex:
        return None, [f"Invalid input values: {ex}"]

    if not (10 <= habit_data["age"] <= 80):
        errors.append("Age must be between 10 and 80.")
    if not (0 <= habit_data["screen_time"] <= 24):
        errors.append("Screen time must be between 0 and 24 hours.")
    if not (0 <= habit_data["sleep"] <= 24):
        errors.append("Sleep must be between 0 and 24 hours.")
    if not (1 <= habit_data["stress"] <= 10):
        errors.append("Stress must be between 1 and 10.")
    if not (1 <= habit_data["productivity"] <= 10):
        errors.append("Productivity must be between 1 and 10.")
    if not (0 <= habit_data["exercise"] <= 1440):
        errors.append("Exercise must be between 0 and 1440 minutes.")

    return habit_data, errors


# --------------- REST API Endpoints (Phase 2) ---------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Direct ML prediction endpoint.
    Accepts JSON with 13 features and returns ML prediction results without persisting.
    """
    data = request.get_json(silent=True) or request.form.to_dict()
    if not data:
        return jsonify({"status": "error", "message": "Missing request body. Provide JSON with habit metrics."}), 400

    habit_data, errors = _validate_habit_data(data)
    if errors:
        return jsonify({"status": "error", "errors": errors}), 400

    try:
        result = predict(habit_data)
        clean_result = _clean_for_json(result)
        return jsonify({"status": "success", "prediction": clean_result}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"Prediction error: {str(e)}"}), 500


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Habit submission endpoint.
    Accepts JSON, saves to SQLite daily_habits, runs prediction, saves to predictions,
    and returns JSON with habit_id and prediction results.
    """
    data = request.get_json(silent=True) or request.form.to_dict()
    if not data:
        return jsonify({"status": "error", "message": "Missing request body. Provide JSON with habit metrics."}), 400

    habit_data, errors = _validate_habit_data(data)
    if errors:
        return jsonify({"status": "error", "errors": errors}), 400

    sid = request.headers.get("X-Session-ID") or data.get("session_id") or _session_id()

    try:
        habit_id = save_habit(sid, habit_data)
        result = predict(habit_data)
        save_prediction(habit_id, result)
        clean_result = _clean_for_json(result)
        return jsonify({
            "status": "success",
            "habit_id": habit_id,
            "prediction": clean_result,
        }), 201
    except Exception as e:
        return jsonify({"status": "error", "message": f"Processing error: {str(e)}"}), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    """Returns recent habit history entries and predictions as JSON."""
    limit = request.args.get("limit", 50, type=int)
    sid = request.headers.get("X-Session-ID") or _session_id()
    entries = get_history(sid, limit=limit, chronological=False)
    return jsonify({
        "status": "success",
        "count": len(entries),
        "history": entries,
    }), 200


@app.route("/api/history/<int:habit_id>", methods=["GET"])
def api_history_detail(habit_id):
    """Returns a specific habit entry and its prediction by habit ID."""
    entry = get_prediction_by_habit_id(habit_id)
    if not entry:
        return jsonify({"status": "error", "message": f"Habit record with ID {habit_id} not found."}), 404
    return jsonify({"status": "success", "entry": entry}), 200


@app.route("/api/dashboard", methods=["GET"])
def api_dashboard():
    """Returns summary metrics, KPIs, and latest entry for the user dashboard."""
    sid = request.headers.get("X-Session-ID") or _session_id()
    stats = get_dashboard_stats(sid)
    return jsonify({"status": "success", "dashboard": stats}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
