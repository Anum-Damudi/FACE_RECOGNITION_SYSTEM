"""
Application configuration.

Loads settings from environment variables with sensible defaults for local development.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Project root (directory containing this file)
BASE_DIR = Path(__file__).resolve().parent

# Folder holding one subdirectory per criminal (images + metadata.json)
CRIMINALS_DIR = Path(os.environ.get("CRIMINALS_DIR", str(BASE_DIR / "criminals")))

# Saved detection screenshots and optional file logs
LOGS_DIR = Path(os.environ.get("LOGS_DIR", str(BASE_DIR / "logs")))

# Face model: Facenet512 is a strong balance of speed vs accuracy for DeepFace
DEEPFACE_MODEL = os.environ.get("DEEPFACE_MODEL", "Facenet512")

# Similarity metric passed to DeepFace.represent alignment
# Cosine distance: lower = more similar. Match if distance < threshold.
DISTANCE_METRIC = os.environ.get("DISTANCE_METRIC", "cosine")

# If cosine distance is below this value, treat as same person → HIGH ALERT
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.6"))

# MongoDB
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.environ.get("MONGODB_DB_NAME", "surveillance_alerts")
DETECTION_LOGS_COLLECTION = os.environ.get("DETECTION_LOGS_COLLECTION", "detection_logs")

# API
FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5000"))

# Alerts
ENABLE_SOUND_ALARM = os.environ.get("ENABLE_SOUND_ALARM", "true").lower() in (
    "1",
    "true",
    "yes",
)
ENABLE_POPUP = os.environ.get("ENABLE_POPUP", "true").lower() in ("1", "true", "yes")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
