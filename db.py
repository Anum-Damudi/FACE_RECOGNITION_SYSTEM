import json
import face_recognition
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path("face_security.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT UNIQUE,
                name TEXT NOT NULL,
                age INTEGER,
                occupation TEXT,
                status TEXT,
                face_encoding TEXT NOT NULL
            )
            """
        )
        conn.commit()


def user_exists(dataset_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM users WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
    return row is not None


def insert_user(
    dataset_id: str,
    name: str,
    age: int,
    occupation: str,
    status: str,
    face_encoding: List[float],
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO users(dataset_id, name, age, occupation, status, face_encoding)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                name,
                age,
                occupation,
                status,
                json.dumps(face_encoding),
            ),
        )
        conn.commit()


def get_all_users() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, dataset_id, name, age, occupation, status, face_encoding FROM users"
        ).fetchall()

    users: List[Dict[str, Any]] = []
    for row in rows:
        users.append(
            {
                "id": row["id"],
                "dataset_id": row["dataset_id"],
                "name": row["name"],
                "age": row["age"],
                "occupation": row["occupation"],
                "status": row["status"],
                "face_encoding": json.loads(row["face_encoding"]),
            }
        )
    return users


def get_user_by_dataset_id(dataset_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, dataset_id, name, age, occupation, status, face_encoding FROM users WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "dataset_id": row["dataset_id"],
        "name": row["name"],
        "age": row["age"],
        "occupation": row["occupation"],
        "status": row["status"],
        "face_encoding": json.loads(row["face_encoding"]),
    }
