import cv2
from typing import Dict, List, Optional

from db import get_all_users, init_db
from face_engine import best_match, detect_faces_and_encodings, draw_results, match_face


DEFAULT_TOLERANCE = 0.65


def open_camera() -> Optional[cv2.VideoCapture]:
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        for idx in range(0, 4):
            try:
                capture = cv2.VideoCapture(idx, backend)
            except Exception:
                continue
            if capture is not None and capture.isOpened():
                print(f"Opened camera device {idx} with backend {backend}.")
                return capture
            if capture is not None:
                capture.release()
    return None


def run_live_camera(use_cnn: bool = True, tolerance: float = DEFAULT_TOLERANCE) -> None:
    init_db()
    known_users = get_all_users()
    if not known_users:
        print("Database is empty. Run `python seed_dataset.py` first.")
        return

    video = open_camera()
    if video is None:
        print("Unable to access the camera. Please connect a webcam, grant OS permissions, and try again.")
        return

    print("Press ESC or 'q' to quit the live camera view.")
    while True:
        success, frame = video.read()
        if not success or frame is None:
            print("Failed to capture camera frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations, face_encodings = detect_faces_and_encodings(
            rgb_frame,
            use_cnn=use_cnn,
        )

        labels: List[str] = []
        for encoding in face_encodings:
            matched_user, distance = match_face(encoding, known_users, tolerance=tolerance)
            if matched_user:
                labels.append(f"{matched_user['name']} ({distance:.2f})")
            else:
                closest, closest_d = best_match(encoding, known_users)
                labels.append(
                    f"Unknown (best {closest_d:.2f})" if closest else "Unknown"
                )

        result_frame = draw_results(rgb_frame, face_locations, labels)
        display_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Sentinel Live Camera", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_camera()