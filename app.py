from typing import Any, Dict, List

from camera import run_live_camera
import streamlit as st

from db import get_all_users, init_db
from face_engine import (
    best_match,
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
    if st.button("Reload registered faces"):
        load_known_users.clear()
        st.rerun()
    known_users = load_known_users()
    st.metric("Registered Faces", len(known_users))
    use_cnn = st.checkbox(
        "CNN face detector (recommended)",
        value=True,
        help="More accurate than HOG; slower on CPU. Turn off for speed.",
    )
    match_tolerance = st.slider(
        "Match tolerance (higher = accept more as a match)",
        min_value=0.45,
        max_value=0.75,
        value=0.65,
        step=0.01,
        help="With multi-sample gallery, 0.63–0.68 usually works. Raise if same person stays Unknown.",
    )
    st.info(
        "Rebuild gallery with: `python seed_dataset.py` (clears DB by default; several encodings per person). "
        "Then upload a **different photo** of someone listed under sample names."
    )
    if known_users:
        sample_names = sorted({u["name"] for u in known_users})
        with st.expander("Sample names in your database"):
            st.caption(", ".join(sample_names[:25]) + (" …" if len(sample_names) > 25 else ""))

uploaded_file = st.file_uploader(
    "Upload image (jpg/jpeg/png)",
    type=["jpg", "jpeg", "png"],
)

camera_clicked = st.button("📷 Open Live Camera")

if camera_clicked:
    run_live_camera()

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

face_locations, face_encodings = detect_faces_and_encodings(rgb_image, use_cnn=use_cnn)

if not face_locations:
    st.error("No face detected in the uploaded image.")
    st.stop()

labels: List[str] = []
recognized_people: List[Dict[str, Any]] = []
unknown_hints: List[Dict[str, Any]] = []

for encoding in face_encodings:
    matched_user, distance = match_face(encoding, known_users, tolerance=match_tolerance)
    if matched_user:
        labels.append(f"{matched_user['name']} ({distance:.2f})")
        recognized_people.append({**matched_user, "distance": distance})
    else:
        closest, closest_d = best_match(encoding, known_users)
        labels.append(f"Unknown (best {closest_d:.2f})" if closest else "Unknown")
        if closest:
            unknown_hints.append({**closest, "distance": closest_d})

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
                st.caption(f"Face distance score: {person['distance']:.3f} (lower is better; ≤ {match_tolerance:.2f} counts as match)")

    if unknown_hints:
        st.error("No confident match for at least one face (above tolerance).")
        for person in unknown_hints:
            with st.container(border=True):
                st.warning(f"Closest in database: **{person['name']}**")
                st.write(f"**Name:** {person['name']}")
                st.write(f"**Age:** {person['age']}")
                st.write(f"**Occupation:** {person['occupation']}")
                st.write(f"**ID / Status:** {person['status']}")
                st.caption(
                    f"Distance **{person['distance']:.3f}** is above your tolerance **{match_tolerance:.2f}**. "
                    "Raise the slider slightly, use a clearer frontal photo, or pick someone from the name list in the sidebar."
                )

st.divider()
st.write("Tip: For best results, use clear frontal face images with good lighting.")
