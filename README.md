# AI Face Recognition Security System (Merged with Sentinel AI)

This repository now combines the original face recognition demo with the Sentinel AI web console.

Features:
- Flask-based security portal with login
- Image upload face recognition workflow
- SQLite user gallery and face encoding database
- Sentinel-inspired dashboard UI with registration count and result summaries

## 1) Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Build or seed the face database

```bash
python seed_dataset.py
```

If you already have a face database, you can skip this step. The database is stored in `face_security.db`.

## 3) Run the merged Flask app

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

The original Sentinel dashboard is still available at `/dashboard` after login.
The face recognition upload page is available at `/recognition` after login.

## 4) Login

Use the built-in demo credentials:
- Username: `admin`
- Password: `password123`

## 5) Use the app

1. Upload a face image (`jpg/jpeg/png`)
2. Select whether to use the CNN detector
3. Adjust tolerance if needed
4. View match results or unknown hints

## Files

- `app.py` - merged Flask app and recognition flow
- `face_engine.py` - image decode, detection, encoding, matching, and drawing
- `db.py` - SQLite database layer
- `camera.py` - fallback live camera helper
- `seed_dataset.py` - dataset ingestion and DB seeding
- `templates/` - login and dashboard UI templates
- `static/` - Sentinel UI CSS and JS assets
