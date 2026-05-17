import cv2
import face_recognition
import pickle
import numpy as np
import streamlit as st

# Load model
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)


def run_live_camera():

    st.subheader("Live Face Recognition Camera")

    FRAME_WINDOW = st.image([])

    # Open camera
    video = cv2.VideoCapture(0)

    # Better camera quality
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    video.set(cv2.CAP_PROP_FPS, 30)
    video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    process_this_frame = True

    while True:

        success, frame = video.read()

        if not success:
            st.error("Cannot access camera")
            break

        # Flip camera like mirror
        frame = cv2.flip(frame, 1)

        # Resize for faster processing
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=0.25,
            fy=0.25
        )

        rgb_small = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        if process_this_frame:

            face_locations = face_recognition.face_locations(
                rgb_small,
                model="hog"
            )

            face_encodings = face_recognition.face_encodings(
                rgb_small,
                face_locations
            )

            names = []

            for encoding in face_encodings:

                try:
                    prediction = model.predict([encoding])[0]

                    probabilities = model.predict_proba(
                        [encoding]
                    )[0]

                    confidence = np.max(probabilities)

                    if confidence < 0.65:
                        prediction = "Unknown"

                except:
                    prediction = "Unknown"

                names.append(prediction)

        process_this_frame = not process_this_frame

        # Draw results
        for (top, right, bottom, left), name in zip(
            face_locations,
            names
        ):

            # Scale back
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Green rectangle
            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            # Filled label box
            cv2.rectangle(
                frame,
                (left, bottom - 35),
                (right, bottom),
                (0, 255, 0),
                cv2.FILLED
            )

            # Name text
            cv2.putText(
                frame,
                name,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show in Streamlit
        FRAME_WINDOW.image(frame)

    video.release()