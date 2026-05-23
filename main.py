"""
Flask API entrypoint for the Criminal Database & High Alert module.

Other surveillance components POST a face crop image; this service runs DeepFace
matching, persists logs + screenshots, and fires high-alert UX when the gallery match
is under the configured distance threshold.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, jsonify, request

import alert_system
import config
import database
from face_matcher import FaceMatcher, decode_upload_to_bgr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
matcher = FaceMatcher()


def _json_safe_distance(value: float) -> float | None:
    """JSON cannot represent inf/nan; expose null to API clients instead."""
    if math.isinf(value) or math.isnan(value):
        return None
    return value


def _ensure_runtime_dirs() -> None:
    """Create criminals gallery root, logs folder, and any missing parents."""
    config.CRIMINALS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _save_screenshot(bgr_image, matched: bool, name_slug: str) -> str:
    """
    Write the detection frame to disk under logs/.

    Returns the absolute path written.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in name_slug)[:80]
    prefix = "MATCH" if matched else "NOMATCH"
    filename = f"{prefix}_{ts}_{safe_slug}.jpg"
    path = config.LOGS_DIR / filename
    cv2.imwrite(str(path), bgr_image)
    return str(path.resolve())


@app.route("/health", methods=["GET"])
def health():
    """Liveness probe for orchestration (Kubernetes, load balancers)."""
    return jsonify({"status": "ok", "service": "criminal_alert_module"})


@app.route("/api/v1/reload-gallery", methods=["POST"])
def reload_gallery():
    """
    Re-scan criminals/ and rebuild embeddings after gallery changes.

    Prefer calling this after bulk import instead of restarting the process.
    """
    try:
        matcher.refresh_gallery()
        return jsonify({"ok": True, "criminals_indexed": matcher.gallery_count()})
    except Exception as exc:
        logger.exception("reload-gallery failed")
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/v1/detect", methods=["POST"])
def detect_face():
    """
    Accept a detected-face image and return JSON match verdict + metadata.

    Multipart form:
      - Field `image` (required): image file (JPEG/PNG/...).
    Optional form fields:
      - threshold: float override for distance threshold (lower = stricter match in cosine space).

    Raw body:
      - If Content-Type is image/*, raw bytes are treated as one image file.
    """
    _ensure_runtime_dirs()

    bgr = None
    threshold_override = None

    if request.files:
        if "image" in request.files:
            bgr = decode_upload_to_bgr(request.files["image"])
        elif "face" in request.files:
            bgr = decode_upload_to_bgr(request.files["face"])
        threshold_override = request.form.get("threshold", type=float)

    if bgr is None and request.content_type and request.content_type.startswith("image/"):
        try:
            arr = np.frombuffer(request.get_data(), dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            bgr = None

    if bgr is None:
        return (
            jsonify(
                {
                    "error": "No image found. Send multipart field 'image' or raw image body.",
                    "matched": False,
                    "criminal_name": None,
                    "confidence": 0.0,
                    "alert_level": "NONE",
                }
            ),
            400,
        )

    result = matcher.match_frame(bgr, threshold=threshold_override)
    detected_at = datetime.now(timezone.utc)
    detected_iso = detected_at.isoformat()

    screenshot_path = ""
    if result.matched:
        slug = result.criminal_name or "matched"
        screenshot_path = _save_screenshot(bgr, True, slug)

    log_distance = _json_safe_distance(result.distance)
    database.save_detection_log(
        matched=result.matched,
        criminal_name=result.criminal_name,
        crime_type=result.crime_type,
        confidence=result.confidence,
        alert_level=result.alert_level,
        distance=log_distance if log_distance is not None else -1.0,
        screenshot_path=screenshot_path,
        source="api",
        extra={"threshold": threshold_override or config.MATCH_THRESHOLD},
    )

    if result.matched:
        alert_system.trigger_high_alert(
            criminal_name=result.criminal_name,
            crime_type=result.crime_type,
            confidence=result.confidence,
            detection_time=detected_at,
        )
    else:
        alert_system.notify_no_match(result.confidence, result.distance)

    payload = {
        "matched": result.matched,
        "criminal_name": result.criminal_name or None,
        "crime_type": result.crime_type or None,
        "confidence": result.confidence,
        "alert_level": result.alert_level,
        "distance": _json_safe_distance(result.distance),
        "detection_time": detected_iso,
        "screenshot_path": screenshot_path or None,
    }
    return jsonify(payload)


def create_app() -> Flask:
    """Factory for WSGI servers (gunicorn, waitress)."""
    _ensure_runtime_dirs()
    return app


if __name__ == "__main__":
    _ensure_runtime_dirs()
    logger.info(
        "Starting Criminal Alert API — gallery=%s threshold=%s model=%s",
        config.CRIMINALS_DIR,
        config.MATCH_THRESHOLD,
        config.DEEPFACE_MODEL,
    )
    try:
        matcher.refresh_gallery()
    except Exception:
        logger.exception("Initial gallery indexing failed; will retry on first /detect")
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=False)
