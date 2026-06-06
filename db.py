"""
db.py — SQLite schema initialisation and seed data
Gym WhatsApp Agent
"""

import sqlite3
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "gym.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables and insert seed data."""
    conn = get_connection()
    cur = conn.cursor()

    # ── Members ──────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS members (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            phone         TEXT    UNIQUE NOT NULL,
            name          TEXT    NOT NULL,
            plan_type     TEXT    NOT NULL,
            start_date    TEXT    NOT NULL,
            expiry_date   TEXT    NOT NULL
        )
    """)

    # ── Class Schedule ────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS class_schedule (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            class_type TEXT NOT NULL,
            date       TEXT NOT NULL,
            time       TEXT NOT NULL,
            instructor TEXT NOT NULL,
            slots      INTEGER NOT NULL DEFAULT 15,
            booked     INTEGER NOT NULL DEFAULT 0
        )
    """)

    # ── Trial Bookings ────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trial_bookings (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            phone      TEXT    NOT NULL,
            name       TEXT,
            date       TEXT    NOT NULL,
            time       TEXT    NOT NULL,
            created_at TEXT    NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # ── Class Registrations ───────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS class_registrations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            phone      TEXT    NOT NULL,
            class_id   INTEGER NOT NULL,
            created_at TEXT    NOT NULL DEFAULT (datetime('now')),
            UNIQUE(phone, class_id)
        )
    """)

    conn.commit()

    # ── Seed demo member ──────────────────────────────────────────────────────
    demo_phone = os.getenv("DEMO_MEMBER_PHONE", "")
    if demo_phone:
        today = datetime.today()
        start = today - timedelta(days=30)
        expiry = today + timedelta(days=335)
        cur.execute("""
            INSERT OR IGNORE INTO members (phone, name, plan_type, start_date, expiry_date)
            VALUES (?, ?, ?, ?, ?)
        """, (
            demo_phone,
            "Demo Member",
            "Premium Annual",
            start.strftime("%Y-%m-%d"),
            expiry.strftime("%Y-%m-%d"),
        ))

    # ── Seed class schedule (next 14 days) ────────────────────────────────────
    classes = [
        ("Yoga",        1, "07:00", "Priya Sharma"),
        ("Yoga",        3, "07:00", "Priya Sharma"),
        ("Yoga",        5, "07:00", "Priya Sharma"),
        ("HIIT",        1, "18:00", "Rahul Verma"),
        ("HIIT",        3, "18:00", "Rahul Verma"),
        ("HIIT",        5, "18:00", "Rahul Verma"),
        ("Zumba",       2, "10:00", "Neha Kapoor"),
        ("Zumba",       4, "10:00", "Neha Kapoor"),
        ("Zumba",       6, "10:00", "Neha Kapoor"),
        ("Strength",    2, "17:00", "Arjun Mehta"),
        ("Strength",    4, "17:00", "Arjun Mehta"),
        ("Pilates",     1, "09:00", "Sonal Jain"),
        ("Pilates",     4, "09:00", "Sonal Jain"),
        ("Spin",        3, "06:30", "Vikram Singh"),
        ("Spin",        6, "06:30", "Vikram Singh"),
    ]
    today = datetime.today()
    for class_type, day_offset, time, instructor in classes:
        date = (today + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        cur.execute("""
            INSERT OR IGNORE INTO class_schedule (class_type, date, time, instructor)
            VALUES (?, ?, ?, ?)
        """, (class_type, date, time, instructor))

    conn.commit()
    conn.close()
    print("[db] Database initialised.")


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_member(phone: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM members WHERE phone = ?", (phone,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_next_class(class_type: str) -> dict | None:
    today = datetime.today().strftime("%Y-%m-%d")
    conn = get_connection()
    row = conn.execute("""
        SELECT * FROM class_schedule
        WHERE LOWER(class_type) = LOWER(?)
          AND date >= ?
        ORDER BY date ASC, time ASC
        LIMIT 1
    """, (class_type, today)).fetchone()
    conn.close()
    return dict(row) if row else None


def book_trial(phone: str, date: str, time: str, name: str = "") -> str:
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO trial_bookings (phone, name, date, time)
            VALUES (?, ?, ?, ?)
        """, (phone, name, date, time))
        conn.commit()
        return "success"
    except Exception as e:
        return f"error: {e}"
    finally:
        conn.close()


def register_for_class(phone: str, class_id: int) -> str:
    conn = get_connection()
    try:
        # Check class exists and has slots
        row = conn.execute(
            "SELECT * FROM class_schedule WHERE id = ?", (class_id,)
        ).fetchone()
        if not row:
            return "error: class not found"
        if row["booked"] >= row["slots"]:
            return "error: class is fully booked"

        conn.execute("""
            INSERT INTO class_registrations (phone, class_id)
            VALUES (?, ?)
        """, (phone, class_id))
        conn.execute(
            "UPDATE class_schedule SET booked = booked + 1 WHERE id = ?",
            (class_id,)
        )
        conn.commit()
        return "success"
    except sqlite3.IntegrityError:
        return "error: already registered for this class"
    except Exception as e:
        return f"error: {e}"
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
