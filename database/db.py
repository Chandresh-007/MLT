"""
MindfulTech – Database Module (SQLite)
=======================================
Handles all database operations: initialisation, saving habits,
saving predictions, and retrieving history.
"""

import os
import sqlite3
import json
from datetime import datetime

# Serverless / Vercel compatibility: use /tmp when running in read-only environment
if os.environ.get("VERCEL"):
    DB_DIR = "/tmp"
    DB_PATH = os.path.join(DB_DIR, "mindfultech.db")
else:
    DB_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database"
    )
    DB_PATH = os.path.join(DB_DIR, "mindfultech.db")


def get_connection():
    global DB_DIR, DB_PATH
    try:
        os.makedirs(DB_DIR, exist_ok=True)
    except OSError:
        DB_DIR = "/tmp"
        DB_PATH = os.path.join(DB_DIR, "mindfultech.db")
        os.makedirs(DB_DIR, exist_ok=True)

    try:
        conn = sqlite3.connect(DB_PATH)
    except sqlite3.OperationalError:
        # Fallback to /tmp if database directory was read-only
        DB_DIR = "/tmp"
        DB_PATH = os.path.join(DB_DIR, "mindfultech.db")
        os.makedirs(DB_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row
    return conn



def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_habits (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            age             INTEGER,
            screen_time     REAL,
            social_media    REAL,
            gaming          REAL,
            short_video     REAL,
            phone_unlocks   INTEGER,
            notifications   INTEGER,
            night_usage     REAL,
            sleep           REAL,
            focus_time      REAL,
            exercise        INTEGER,
            stress          INTEGER,
            productivity    INTEGER,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            habit_id         INTEGER NOT NULL,
            risk_level       TEXT,
            wellbeing_score  REAL,
            cluster_label    TEXT,
            patterns         TEXT,
            recommendations  TEXT,
            explanation      TEXT,
            is_anomaly       INTEGER DEFAULT 0,
            anomaly_message  TEXT,
            created_at       TEXT NOT NULL,
            FOREIGN KEY (habit_id) REFERENCES daily_habits(id)
        )
    """)

    # Add columns if they don't exist (for upgrading existing databases)
    for col, coltype in [
        ("explanation", "TEXT"),
        ("is_anomaly", "INTEGER DEFAULT 0"),
        ("anomaly_message", "TEXT"),
    ]:
        try:
            cur.execute(
                f"ALTER TABLE predictions ADD COLUMN {col} {coltype}"
            )
        except sqlite3.OperationalError:
            pass  # column already exists

    conn.commit()
    conn.close()
    print("Database initialised")


def save_habit(session_id: str, habit_data: dict) -> int:
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat()

    cur.execute(
        "SELECT id FROM users WHERE session_id = ?", (session_id,)
    )
    row = cur.fetchone()
    if row:
        user_id = row["id"]
    else:
        cur.execute(
            "INSERT INTO users (session_id, created_at) VALUES (?, ?)",
            (session_id, now),
        )
        user_id = cur.lastrowid

    cur.execute("""
        INSERT INTO daily_habits
            (user_id, age, screen_time, social_media, gaming, short_video,
             phone_unlocks, notifications, night_usage, sleep,
             focus_time, exercise, stress, productivity, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        habit_data.get("age"), habit_data.get("screen_time"),
        habit_data.get("social_media"), habit_data.get("gaming"),
        habit_data.get("short_video"), habit_data.get("phone_unlocks"),
        habit_data.get("notifications"), habit_data.get("night_usage"),
        habit_data.get("sleep"), habit_data.get("focus_time"),
        habit_data.get("exercise"), habit_data.get("stress"),
        habit_data.get("productivity"), now,
    ))
    habit_id = cur.lastrowid
    conn.commit()
    conn.close()
    return habit_id


def save_prediction(habit_id: int, prediction: dict):
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat()

    anomaly = prediction.get("anomaly", {})

    cur.execute("""
        INSERT INTO predictions
            (habit_id, risk_level, wellbeing_score, cluster_label,
             patterns, recommendations, explanation,
             is_anomaly, anomaly_message, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        habit_id,
        prediction.get("risk_level"),
        prediction.get("wellbeing_score"),
        prediction.get("cluster_label"),
        json.dumps(prediction.get("patterns", [])),
        json.dumps(prediction.get("recommendations", [])),
        prediction.get("explanation", ""),
        1 if anomaly.get("is_anomaly") else 0,
        anomaly.get("anomaly_message", ""),
        now,
    ))
    conn.commit()
    conn.close()


def _row_to_dict(row) -> dict:
    """Convert a sqlite3.Row to dict and parse JSON fields."""
    d = dict(row)
    if d.get("patterns"):
        try:
            d["patterns"] = json.loads(d["patterns"])
        except (json.JSONDecodeError, TypeError):
            d["patterns"] = []
    if d.get("recommendations"):
        try:
            d["recommendations"] = json.loads(d["recommendations"])
        except (json.JSONDecodeError, TypeError):
            d["recommendations"] = []
    return d


def get_latest_entry(session_id: str) -> dict | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT dh.*, p.risk_level, p.wellbeing_score, p.cluster_label,
               p.patterns, p.recommendations, p.explanation,
               p.is_anomaly, p.anomaly_message,
               p.id AS prediction_id
        FROM daily_habits dh
        JOIN users u ON dh.user_id = u.id
        LEFT JOIN predictions p ON p.habit_id = dh.id
        WHERE u.session_id = ?
        ORDER BY dh.created_at DESC
        LIMIT 1
    """, (session_id,))
    row = cur.fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def get_history(session_id: str, limit: int = 30, chronological: bool = False) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT dh.*, p.risk_level, p.wellbeing_score, p.cluster_label,
               p.patterns, p.recommendations, p.explanation,
               p.is_anomaly, p.anomaly_message
        FROM daily_habits dh
        JOIN users u ON dh.user_id = u.id
        LEFT JOIN predictions p ON p.habit_id = dh.id
        WHERE u.session_id = ?
        ORDER BY dh.created_at DESC
        LIMIT ?
    """, (session_id, limit))
    rows = cur.fetchall()
    conn.close()
    items = [_row_to_dict(r) for r in rows]
    return list(reversed(items)) if chronological else items


def get_prediction_by_habit_id(habit_id: int) -> dict | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT dh.*, p.risk_level, p.wellbeing_score, p.cluster_label,
               p.patterns, p.recommendations, p.explanation,
               p.is_anomaly, p.anomaly_message,
               p.id AS prediction_id
        FROM daily_habits dh
        LEFT JOIN predictions p ON p.habit_id = dh.id
        WHERE dh.id = ?
        LIMIT 1
    """, (habit_id,))
    row = cur.fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def get_dashboard_stats(session_id: str) -> dict:
    """Compute summary statistics for user dashboard API."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(dh.id) as total_entries,
               AVG(dh.screen_time) as avg_screen_time,
               AVG(dh.sleep) as avg_sleep,
               AVG(dh.focus_time) as avg_focus_time,
               AVG(dh.stress) as avg_stress,
               AVG(dh.productivity) as avg_productivity
        FROM daily_habits dh
        JOIN users u ON dh.user_id = u.id
        WHERE u.session_id = ?
    """, (session_id,))
    agg = cur.fetchone()

    cur.execute("""
        SELECT p.risk_level, COUNT(p.id) as count
        FROM daily_habits dh
        JOIN users u ON dh.user_id = u.id
        JOIN predictions p ON p.habit_id = dh.id
        WHERE u.session_id = ?
        GROUP BY p.risk_level
    """, (session_id,))
    risk_rows = cur.fetchall()
    risk_breakdown = {
        row["risk_level"]: row["count"]
        for row in risk_rows if row["risk_level"]
    }
    conn.close()

    latest = get_latest_entry(session_id)

    return {
        "total_entries": agg["total_entries"] if agg else 0,
        "avg_screen_time": round(agg["avg_screen_time"], 1) if agg and agg["avg_screen_time"] else 0,
        "avg_sleep": round(agg["avg_sleep"], 1) if agg and agg["avg_sleep"] else 0,
        "avg_focus_time": round(agg["avg_focus_time"], 1) if agg and agg["avg_focus_time"] else 0,
        "avg_stress": round(agg["avg_stress"], 1) if agg and agg["avg_stress"] else 0,
        "avg_productivity": round(agg["avg_productivity"], 1) if agg and agg["avg_productivity"] else 0,
        "risk_breakdown": risk_breakdown,
        "latest_entry": latest,
    }
