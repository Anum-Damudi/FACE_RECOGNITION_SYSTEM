"""
Face matching against the criminal gallery using DeepFace embeddings.

Each criminal is stored as a folder under CRIMINALS_DIR with:
  - metadata.json (display_name, crime_type, optional id)
  - One or more face images (jpg/png/...)

Embeddings are computed on demand and cached in memory for the process lifetime.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from deepface import DeepFace
from deepface.commons import distance as dst

import config

logger = logging.getLogger(__name__)


@dataclass
class CriminalRecord:
    """In-memory representation of one criminal folder."""

    folder_name: str
    display_name: str
    crime_type: str
    image_paths: list[Path]
    embeddings: list[np.ndarray] | None = None


@dataclass
class MatchResult:
    """Outcome of comparing a probe face to the gallery."""

    matched: bool
    criminal_name: str
    crime_type: str
    confidence: float
    distance: float
    alert_level: str


class FaceMatcher:
    """
    Indexes criminals on disk, builds embeddings with DeepFace, and runs 1:N comparison.

    Distance uses the same metric as DeepFace (cosine by default): lower distance means
    a closer match. A match is declared when distance < MATCH_THRESHOLD, per requirements.
    """

    def __init__(self) -> None:
        self._records: list[CriminalRecord] = []
        self._indexed = False

    def gallery_count(self) -> int:
        """Number of criminals with at least one usable face embedding."""
        return sum(1 for r in self._records if r.embeddings)

    def refresh_gallery(self) -> None:
        """
        Scan CRIMINALS_DIR, load metadata, and pre-compute embeddings for all gallery images.

        Call after adding/removing criminals or images.
        """
        self._records = []
        config.CRIMINALS_DIR.mkdir(parents=True, exist_ok=True)

        for sub in sorted(config.CRIMINALS_DIR.iterdir()):
            if not sub.is_dir():
                continue
            meta_path = sub / "metadata.json"
            if not meta_path.is_file():
                logger.warning("Skipping %s: missing metadata.json", sub)
                continue

            with open(meta_path, encoding="utf-8") as f:
                meta: dict[str, Any] = json.load(f)

            display_name = meta.get("display_name") or meta.get("name") or sub.name
            crime_type = meta.get("crime_type") or "unknown"

            images = [
                p
                for p in sub.iterdir()
                if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
            ]
            if not images:
                logger.warning("Skipping %s: no face images", sub)
                continue

            rec = CriminalRecord(
                folder_name=sub.name,
                display_name=str(display_name),
                crime_type=str(crime_type),
                image_paths=sorted(images),
                embeddings=None,
            )
            self._records.append(rec)

        self._build_embeddings()
        self._indexed = True
        logger.info("Gallery indexed: %d criminals", len(self._records))

    def _build_embeddings(self) -> None:
        """Compute DeepFace embedding vectors for every gallery image."""
        for rec in self._records:
            vectors: list[np.ndarray] = []
            for path in rec.image_paths:
                try:
                    rep = DeepFace.represent(
                        img_path=str(path),
                        model_name=config.DEEPFACE_MODEL,
                        enforce_detection=True,
                        detector_backend="opencv",
                    )
                    if rep and isinstance(rep, list) and "embedding" in rep[0]:
                        vectors.append(np.array(rep[0]["embedding"], dtype=np.float32))
                except Exception as exc:
                    logger.error(
                        "Embedding failed for %s: %s",
                        path,
                        exc,
                    )
            rec.embeddings = vectors if vectors else None

    def _probe_embedding(self, bgr_image: np.ndarray) -> np.ndarray | None:
        """
        Extract embedding from an OpenCV BGR image (numpy array).

        DeepFace accepts ndarray; we pass a copy to avoid buffer issues.
        """
        try:
            rep = DeepFace.represent(
                img_path=bgr_image,
                model_name=config.DEEPFACE_MODEL,
                enforce_detection=True,
                detector_backend="opencv",
            )
            if rep and isinstance(rep, list) and "embedding" in rep[0]:
                return np.array(rep[0]["embedding"], dtype=np.float32)
        except Exception as exc:
            logger.error("Probe embedding failed: %s", exc)
        return None

    def match_frame(
        self,
        bgr_image: np.ndarray,
        threshold: float | None = None,
    ) -> MatchResult:
        """
        Compare the detected face image against all criminals.

        Returns MatchResult with matched flag, best identity, confidence-like score,
        raw distance, and alert_level (HIGH_ALERT / NONE).

        Confidence is reported as max(0, 1 - distance) for cosine-style distances in [0, 1].
        """
        if not self._indexed:
            self.refresh_gallery()

        threshold = threshold if threshold is not None else config.MATCH_THRESHOLD

        probe = self._probe_embedding(bgr_image)
        if probe is None:
            return MatchResult(
                matched=False,
                criminal_name="",
                crime_type="",
                confidence=0.0,
                distance=float("inf"),
                alert_level="NONE",
            )

        best_distance = float("inf")
        best_rec: CriminalRecord | None = None

        for rec in self._records:
            if not rec.embeddings:
                continue
            for emb in rec.embeddings:
                if config.DISTANCE_METRIC == "cosine":
                    d = float(dst.findCosineDistance(probe, emb))
                elif config.DISTANCE_METRIC == "euclidean":
                    d = float(dst.findEuclideanDistance(probe, emb))
                else:
                    d = float(dst.findCosineDistance(probe, emb))

                if d < best_distance:
                    best_distance = d
                    best_rec = rec

        if best_rec is None or best_distance >= threshold:
            conf = max(0.0, 1.0 - min(best_distance, 1.0)) if best_distance < float("inf") else 0.0
            return MatchResult(
                matched=False,
                criminal_name="",
                crime_type="",
                confidence=conf,
                distance=best_distance if best_distance < float("inf") else 0.0,
                alert_level="NONE",
            )

        confidence = max(0.0, 1.0 - best_distance)
        return MatchResult(
            matched=True,
            criminal_name=best_rec.display_name,
            crime_type=best_rec.crime_type,
            confidence=round(confidence, 4),
            distance=round(best_distance, 6),
            alert_level="HIGH_ALERT",
        )


def decode_upload_to_bgr(file_storage) -> np.ndarray | None:
    """
    Read an uploaded file-like object from Flask into a BGR ndarray for OpenCV/DeepFace.

    Returns None if decoding fails.
    """
    try:
        data = file_storage.read()
        if not data:
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return bgr
    except Exception as exc:
        logger.error("decode_upload_to_bgr failed: %s", exc)
        return None


def load_image_path_bgr(path: Path) -> np.ndarray | None:
    """Load an image from disk as BGR using OpenCV."""
    img = cv2.imread(str(path))
    if img is None:
        logger.error("OpenCV could not read: %s", path)
    return img
