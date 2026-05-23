"""
Persistence layer for detection events.

Primary implementation: MongoDB (document store fits flexible surveillance logs).
If MongoDB is unavailable at runtime, operations no-op with a warning so the API
still returns JSON (operators should monitor logs).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import config

logger = logging.getLogger(__name__)

_client = None
_db = None


def _get_collection():
    """
    Lazily connect to MongoDB and return the detection_logs collection.

    Returns None if pymongo is missing or connection fails.
    """
    global _client, _db
    if _db is not None:
        return _db[config.DETECTION_LOGS_COLLECTION]

    try:
        from pymongo import MongoClient
    except ImportError:
        logger.warning("pymongo not installed; detection logs will not be saved.")
        return None

    try:
        _client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=3000)
        _client.admin.command("ping")
        _db = _client[config.MONGODB_DB_NAME]
        logger.info(
            "MongoDB connected: %s / %s",
            config.MONGODB_DB_NAME,
            config.DETECTION_LOGS_COLLECTION,
        )
        return _db[config.DETECTION_LOGS_COLLECTION]
    except Exception as exc:
        logger.warning("MongoDB connection failed (%s); logs disabled until DB is up.", exc)
        _client = None
        _db = None
        return None


def save_detection_log(
    matched: bool,
    criminal_name: str,
    crime_type: str,
    confidence: float,
    alert_level: str,
    distance: float,
    screenshot_path: str,
    source: str = "api",
    extra: dict[str, Any] | None = None,
) -> bool:
    """
    Insert one detection record with UTC timestamp.

    Schema (MongoDB document):
      - matched: bool
      - criminal_name, crime_type, alert_level: str
      - confidence, distance: float
      - screenshot_path: str (relative or absolute path to saved image)
      - detected_at: ISODate
      - source: str (e.g. 'api', 'internal')
      - extra: optional subdocument for integrators
    """
    col = _get_collection()
    if col is None:
        return False

    doc: dict[str, Any] = {
        "matched": matched,
        "criminal_name": criminal_name,
        "crime_type": crime_type,
        "confidence": confidence,
        "distance": distance,
        "alert_level": alert_level,
        "screenshot_path": screenshot_path,
        "detected_at": datetime.now(timezone.utc),
        "source": source,
    }
    if extra:
        doc["extra"] = extra

    try:
        col.insert_one(doc)
        return True
    except Exception as exc:
        logger.error("MongoDB insert failed: %s", exc)
        return False
