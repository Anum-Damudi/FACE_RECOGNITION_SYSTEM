<<<<<<< HEAD
# AI Face Recognition Security System (Demo)

This demo project implements an image-upload-based AI face recognition security workflow with:

- Python + Streamlit frontend
- OpenCV + `face_recognition` for face detection and recognition
- SQLite database for storing/retrieving user profile data + face embeddings
- Online dataset bootstrap using LFW (Labeled Faces in the Wild)

## Program Flow

Image Upload -> Face Detection -> Face Recognition -> Database Matching -> Display User Information / Unknown Alert

## 1) Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Build Demo Face Database (Online Dataset)

```bash
python seed_dataset.py
```

This downloads the LFW dataset from the internet and stores face encodings and profile details in `face_security.db`.

## 3) Run Streamlit App

```bash
streamlit run app.py
```

## 4) Use the App

1. Upload a face image (`jpg/jpeg/png`)
2. App detects face(s)
3. Compares against stored face embeddings
4. If matched, shows:
   - Name
   - Age
   - Occupation
   - ID/Status
5. If not matched, shows Unknown Person + alert banner

## Files

- `app.py` - Streamlit UI and app flow
- `face_engine.py` - image decode, face detect/encode, matching logic
- `db.py` - SQLite database layer
- `seed_dataset.py` - online dataset ingestion and DB seeding

## Future Demo Upgrades

- Add webcam/CCTV live stream mode with OpenCV
- Replace SQLite with MongoDB/MySQL
- Add check-in logs and alert history table
- Add threshold tuning controls in sidebar
=======
# FACE_RECOGNITION_SYSTEM
>>>>>>> 087dbfc312de52f7b55072cffb5904f0d536a0fe
