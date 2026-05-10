from typing import Any, Dict, List

import streamlit as st

from db import get_all_users, init_db
from face_engine import (
    decode_uploaded_image,
    detect_faces_and_encodings,
    draw_results,
    match_face,
)

st.set_page_config(page_title="AI Face Security Demo", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner=False)
def load_known_users() -> List[Dict[str, Any]]:
    init_db()
    return get_all_users()


st.title("AI Face Recognition Security System (Demo)")
st.caption(
    "Flow: Image Upload -> Face Detection -> Face Recognition -> Database Match -> Details / Unknown Alert"
)

with st.sidebar:
    st.header("System Status")
    known_users = load_known_users()
    st.metric("Registered Faces", len(known_users))
    st.info("Run `python seed_dataset.py` once to populate demo data from an online dataset (LFW).")

uploaded_file = st.file_uploader(
    "Upload image (jpg/jpeg/png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.warning("Upload an image to start recognition.")
    st.stop()

if len(known_users) == 0:
    st.error("Database is empty. Please run `python seed_dataset.py` first.")
    st.stop()

file_bytes = uploaded_file.read()

try:
    rgb_image = decode_uploaded_image(file_bytes)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

face_locations, face_encodings = detect_faces_and_encodings(rgb_image)

if not face_locations:
    st.error("No face detected in the uploaded image.")
    st.stop()

labels: List[str] = []
recognized_people: List[Dict[str, Any]] = []
unknown_count = 0

for encoding in face_encodings:
    matched_user, distance = match_face(encoding, known_users)
    if matched_user:
        labels.append(f"{matched_user['name']} ({distance:.2f})")
        recognized_people.append({**matched_user, "distance": distance})
    else:
        labels.append("Unknown Person")
        unknown_count += 1

result_img = draw_results(rgb_image, face_locations, labels)

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Processed Image")
    st.image(result_img, use_container_width=True)

with right:
    st.subheader("Recognition Output")
    if recognized_people:
        for person in recognized_people:
            with st.container(border=True):
                st.success(f"Match Found: {person['name']}")
                st.write(f"**Name:** {person['name']}")
                st.write(f"**Age:** {person['age']}")
                st.write(f"**Occupation:** {person['occupation']}")
                st.write(f"**ID / Status:** {person['status']}")
                st.caption(f"Face distance score: {person['distance']:.3f} (lower is better)")

    if unknown_count > 0:
        st.error(f"ALERT: {unknown_count} unknown face(s) detected!")
        st.warning("Unknown Person")

st.divider()
st.write("Tip: For best results, use clear frontal face images with good lighting.")
