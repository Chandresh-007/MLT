import os
import sqlite3
import json
from datetime import datetime
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database")
DB_PATH = os.path.join(DB_DIR, "mindfultech.db")

def get_connection():
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
            created_at       TEXT NOT NULL,
            FOREIGN KEY (habit_id) REFERENCES daily_habits(id)
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialised")

def save_habit(session_id: str, habit_data: dict) -> int:
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    cur.execute("SELECT id FROM users WHERE session_id = ?", (session_id,))
    row = cur.fetchone()
    if row:
        user_id = row["id"]
    else:
        cur.execute("INSERT INTO users (session_id, created_at) VALUES (?, ?)", (session_id, now))
        user_id = cur.lastrowid
    cur.execute("""
        INSERT INTO daily_habits
            (user_id, age, screen_time, social_media, gaming, short_video,
             phone_unlocks, notifications, night_usage, sleep,
             focus_time, exercise, stress, productivity, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, habit_data.get("age"), habit_data.get("screen_time"),
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
    cur.execute("""
        INSERT INTO predictions
            (habit_id, risk_level, wellbeing_score, cluster_label,
             patterns, recommendations, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        habit_id, prediction.get("risk_level"), prediction.get("wellbeing_score"),
        prediction.get("cluster_label"), json.dumps(prediction.get("patterns", [])),
        json.dumps(prediction.get("recommendations", [])), now,
    ))
    conn.commit()
    conn.close()

def get_latest_entry(session_id: str) -> dict | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT dh.*, p.risk_level, p.wellbeing_score, p.cluster_label,
               p.patterns, p.recommendations, p.id AS prediction_id
        FROM daily_habits dh
        JOIN users u ON dh.user_id = u.id
        LEFT JOIN predictions p ON p.habit_id = dh.id
        WHERE u.session_id = ?
        ORDER BY dh.created_at DESC
        LIMIT 1
    """, (session_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        d = dict(row)
        if d.get("patterns"): d["patterns"] = json.loads(d["patterns"])
        if d.get("recommendations"): d["recommendations"] = json.loads(d["recommendations"])
        return d
    return None

def get_history(session_id: str, limit: int = 30) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT dh.*, p.risk_level, p.wellbeing_score, p.cluster_label,
               p.patterns, p.recommendations
        FROM daily_habits dh
        JOIN users u ON dh.user_id = u.id
        LEFT JOIN predictions p ON p.habit_id = dh.id
        WHERE u.session_id = ?
        ORDER BY dh.created_at DESC
        LIMIT ?
    """, (session_id, limit))
    rows = cur.fetchall()
    conn.close()
    history = []
    for row in rows:
        d = dict(row)
        if d.get("patterns"): d["patterns"] = json.loads(d["patterns"])
        if d.get("recommendations"): d["recommendations"] = json.loads(d["recommendations"])
        history.append(d)
    return list(reversed(history))

def get_prediction_by_habit_id(habit_id: int) -> dict | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT dh.*, p.risk_level, p.wellbeing_score, p.cluster_label,
               p.patterns, p.recommendations, p.id AS prediction_id
        FROM daily_habits dh
        LEFT JOIN predictions p ON p.habit_id = dh.id
        WHERE dh.id = ?
        LIMIT 1
    """, (habit_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        d = dict(row)
        if d.get("patterns"): d["patterns"] = json.loads(d["patterns"])
        if d.get("recommendations"): d["recommendations"] = json.loads(d["recommendations"])
        return d
    return None
