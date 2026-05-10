from typing import Any, Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    file_array = np.frombuffer(file_bytes, np.uint8)
    bgr_image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError("Unable to decode uploaded image.")
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image


def detect_faces_and_encodings(
    rgb_image: np.ndarray,
) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return face_locations, face_encodings


def match_face(
    query_encoding: np.ndarray,
    known_users: List[Dict[str, Any]],
    tolerance: float = 0.48,
) -> Tuple[Optional[Dict[str, Any]], float]:
    if not known_users:
        return None, 1.0

    known_encodings = [np.array(user["face_encoding"], dtype=np.float64) for user in known_users]
    distances = face_recognition.face_distance(known_encodings, query_encoding)

    if len(distances) == 0:
        return None, 1.0

    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])

    if best_distance <= tolerance:
        return known_users[best_idx], best_distance
    return None, best_distance


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
