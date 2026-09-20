import os
import json
import uuid
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, flash,
)
app = Flask(__name__)
app.secret_key = "mindfultech-dev-secret-key-change-in-prod"
from database.db import init_db, save_habit, save_prediction, get_latest_entry, get_history, get_prediction_by_habit_id
from ml.prediction import predict, FEATURE_COLS
init_db()

def _session_id():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]

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
    sid = _session_id()
    habit_id = save_habit(sid, habit_data)
    result = predict(habit_data)
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

@app.route("/models")
@app.route("/modules")
def models_page():
    metrics_path = os.path.join("models", "model_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    cluster_path = os.path.join("models", "cluster_descriptions.json")
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
    history = get_history(_session_id(), limit=30)
    labels, screen_time, social_media, focus_time, sleep, productivity, risk_levels = [], [], [], [], [], [], []
    risk_map = {"Low": 1, "Moderate": 2, "High": 3}
    for i, entry in enumerate(history):
        labels.append(f"Entry {i + 1}")
        screen_time.append(entry.get("screen_time", 0))
        social_media.append(entry.get("social_media", 0))
        focus_time.append(entry.get("focus_time", 0))
        sleep.append(entry.get("sleep", 0))
        productivity.append(entry.get("productivity", 0))
        risk_levels.append(risk_map.get(entry.get("risk_level", "Low"), 1))
    return jsonify({
        "labels": labels, "screen_time": screen_time, "social_media": social_media,
        "focus_time": focus_time, "sleep": sleep, "productivity": productivity,
        "risk_levels": risk_levels,
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
