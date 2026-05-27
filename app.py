import base64
from typing import Any, Dict, List

import cv2
import numpy as np
from flask import Flask, redirect, render_template, request, session, url_for

from db import get_all_users, init_db
from face_engine import (
    best_match,
    decode_uploaded_image,
    detect_faces_and_encodings,
    draw_results,
    match_face,
)

app = Flask(__name__)

app.secret_key = "sentinel_neural_core"

USER_CREDENTIALS = {
    "admin": "password123"
}

DEFAULT_RECOGNITION_TOLERANCE = 0.55


def load_known_users() -> List[Dict[str, Any]]:

    init_db()

    return get_all_users()


def image_to_data_url(image: Any) -> str:

    success, buffer = cv2.imencode(
        ".png",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )

    if not success:

        raise RuntimeError("Unable to encode result image.")

    return base64.b64encode(buffer.tobytes()).decode("ascii")


@app.route("/")
def home():

    session.pop("user", None)

    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
@app.route("/login.html", methods=["GET", "POST"])
def login():

    error = None

    if request.method == "POST":

        username = request.form.get("username", "").strip()

        password = request.form.get("password", "").strip()

        if (
            username in USER_CREDENTIALS and
            USER_CREDENTIALS[username] == password
        ):

            session["user"] = username

            return redirect(url_for("dashboard"))

        error = "ACCESS DENIED"

    return render_template("login.html", error=error)


@app.route("/dashboard")
@app.route("/dashboard.html")
def dashboard():

    if "user" not in session:

        return redirect(url_for("login"))

    return render_template("dashboard.html")


@app.route("/recognition", methods=["GET", "POST"])
def recognition():

    if "user" not in session:

        return redirect(url_for("login"))

    known_users = load_known_users()

    registered_count = len(known_users)

    sample_names = (
        sorted({u["name"] for u in known_users})
        if known_users else []
    )

    error = None

    result_image = None

    recognized_people: List[Dict[str, Any]] = []

    unknown_hints: List[Dict[str, Any]] = []

    use_cnn = True

    tolerance = DEFAULT_RECOGNITION_TOLERANCE

    if request.method == "POST":

        uploaded_file = request.files.get("image_file")

        use_cnn = request.form.get("use_cnn") == "on"

        try:

            tolerance = float(
                request.form.get("tolerance", tolerance)
            )

        except ValueError:

            tolerance = DEFAULT_RECOGNITION_TOLERANCE

        if not uploaded_file or uploaded_file.filename == "":

            error = "Please upload a JPEG or PNG image."

        elif registered_count == 0:

            error = (
                "Database is empty. "
                "Please run `python seed_dataset.py` first."
            )

        else:

            try:

                file_bytes = uploaded_file.read()

                rgb_image = decode_uploaded_image(file_bytes)

                face_locations, face_encodings = detect_faces_and_encodings(
                    rgb_image,
                    use_cnn=use_cnn,
                )

                if not face_locations:

                    error = "No face detected in the uploaded image."

                else:

                    labels: List[str] = []

                    for encoding in face_encodings:

                        matched_user, distance = match_face(
                            encoding,
                            known_users,
                            tolerance=tolerance,
                        )

                        if matched_user:

                            labels.append(
                                f"{matched_user['name']} ({distance:.2f})"
                            )

                            recognized_people.append(
                                {
                                    **matched_user,
                                    "distance": distance
                                }
                            )

                        else:

                            closest, closest_d = best_match(
                                encoding,
                                known_users
                            )

                            labels.append(
                                f"Unknown (best {closest_d:.2f})"
                                if closest else "Unknown"
                            )

                            if closest:

                                unknown_hints.append(
                                    {
                                        **closest,
                                        "distance": closest_d
                                    }
                                )

                    result_image = image_to_data_url(

                        draw_results(
                            rgb_image,
                            face_locations,
                            labels
                        )
                    )

            except ValueError as exc:

                error = str(exc)

            except Exception as exc:

                error = f"Unable to process image: {exc}"

    return render_template(
        "recognition.html",
        registered_count=registered_count,
        sample_names=sample_names,
        error=error,
        result_image=result_image,
        recognized_people=recognized_people,
        unknown_hints=unknown_hints,
        use_cnn=use_cnn,
        tolerance=tolerance,
    )


@app.route('/api/scan', methods=['POST'])
def api_scan():

    if 'user' not in session:

        return {'error': 'Unauthorized'}, 401

    if 'frame' not in request.files:

        return {'error': 'Missing frame'}, 400

    frame_file = request.files['frame']

    image_bytes = frame_file.read()

    nparr = np.frombuffer(image_bytes, np.uint8)

    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:

        return {'error': 'Unable to decode image'}, 400

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations, face_encodings = detect_faces_and_encodings(
        rgb_image,
        use_cnn=False
    )

    if not face_locations:

        return {
            'state': 'idle',
            'face': {
                'top': 0,
                'right': 0,
                'bottom': 0,
                'left': 0,
            }
        }

    face_location = face_locations[0]

    face_encoding = face_encodings[0]

    known_users = load_known_users()

    matched_user, distance = match_face(
        face_encoding,
        known_users,
        tolerance=DEFAULT_RECOGNITION_TOLERANCE,
    )

    if matched_user:

        status_text = str(
            matched_user.get('status', '')
        ).lower()

        if any(
            word in status_text
            for word in ('criminal', 'wanted', 'threat')
        ):

            state = 'criminal'

        else:

            state = 'known'

    else:

        state = 'unknown'

    response = {

        'state': state,

        'face': {

            'top': int(face_location[0]),
            'right': int(face_location[1]),
            'bottom': int(face_location[2]),
            'left': int(face_location[3]),
        },

        'distance': float(distance),
    }

    if not matched_user:

        closest, closest_distance = best_match(
            face_encoding,
            known_users
        )

        if closest:

            response['closest_name'] = closest['name']

            response['closest_distance'] = float(
                closest_distance
            )

    if matched_user:

        response.update(
            {
                'name': matched_user.get(
                    'name',
                    'Unknown'
                ),

                'age': matched_user.get(
                    'age',
                    'N/A'
                ),

                'occupation': matched_user.get(
                    'occupation',
                    'N/A'
                ),

                'status': matched_user.get(
                    'status',
                    'Verified'
                )
            }
        )

    return response


@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect(url_for("login"))


if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )