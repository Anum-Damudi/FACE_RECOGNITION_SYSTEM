from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np

# More jitters = stabler 128-d vectors (slower). Helps cross-photo matching.
NUM_JITTERS = 7

# Upscale tiny LFW crops before CNN/HOG so landmarks are usable.
GALLERY_MIN_SIDE = 320


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    file_array = np.frombuffer(file_bytes, np.uint8)
    bgr_image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError("Unable to decode uploaded image.")
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image


def upscale_if_tiny(rgb: np.ndarray, min_side: int = GALLERY_MIN_SIDE) -> np.ndarray:
    h, w = rgb.shape[:2]
    m = min(h, w)
    if m >= min_side:
        return rgb
    scale = min_side / m
    return cv2.resize(
        rgb,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_CUBIC,
    )


def face_locations_robust(rgb: np.ndarray, use_cnn: bool) -> List[Tuple[int, int, int, int]]:
    """CNN when requested, then HOG with stronger upsampling."""
    if use_cnn:
        locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=1, model="cnn")
        if locs:
            return locs
    for ups in (2, 1):
        locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=ups, model="hog")
        if locs:
            return locs
    return []


def encode_single_face_rgb(rgb: np.ndarray, use_cnn: bool) -> Optional[np.ndarray]:
    """One face per image (gallery / tight crop). Same encoder settings as live detection."""
    rgb = upscale_if_tiny(rgb)
    h, w = rgb.shape[:2]
    locs = face_locations_robust(rgb, use_cnn=use_cnn)
    if not locs:
        locs = [(0, w, h, 0)]
    encs = face_recognition.face_encodings(rgb, locs, num_jitters=NUM_JITTERS)
    return encs[0] if encs else None


def detect_faces_and_encodings(
    rgb_image: np.ndarray,
    use_cnn: bool = True,
) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
    locs = face_locations_robust(rgb_image, use_cnn=use_cnn)
    encs = face_recognition.face_encodings(rgb_image, locs, num_jitters=NUM_JITTERS)
    return locs, encs


def _best_match_row(
    query_encoding: np.ndarray,
    known_users: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Pick identity by minimum distance among all rows for that name, then best identity."""
    if not known_users:
        return None, 1.0

    by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for u in known_users:
        by_name[u["name"]].append(u)

    best_row: Optional[Dict[str, Any]] = None
    best_dist = float("inf")

    for rows in by_name.values():
        encodings = [np.array(r["face_encoding"], dtype=np.float64) for r in rows]
        distances = face_recognition.face_distance(encodings, query_encoding)
        idx = int(np.argmin(distances))
        d = float(distances[idx])
        if d < best_dist:
            best_dist = d
            best_row = rows[idx]

    return best_row, best_dist


def best_match(
    query_encoding: np.ndarray,
    known_users: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], float]:
    return _best_match_row(query_encoding, known_users)


def match_face(
    query_encoding: np.ndarray,
    known_users: List[Dict[str, Any]],
    tolerance: float = 0.65,
) -> Tuple[Optional[Dict[str, Any]], float]:
    candidate, distance = _best_match_row(query_encoding, known_users)
    if candidate is None:
        return None, distance
    if distance <= tolerance:
        return candidate, distance
    return None, distance


def draw_results(
    rgb_image: np.ndarray,
    face_locations: List[Tuple[int, int, int, int]],
    labels: List[str],
) -> np.ndarray:
    image = rgb_image.copy()
    for (top, right, bottom, left), label in zip(face_locations, labels):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(image, (left, bottom - 28), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(
            image,
            label,
            (left + 6, bottom - 8),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
    return image
