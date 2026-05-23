# Criminal Database & High Alert Module

Python backend for an AI surveillance stack: accepts a **detected face crop** over HTTP, compares it to an on-disk **criminal gallery** using **DeepFace** (Facenet512 embeddings), and raises a **HIGH ALERT** when the best cosine distance is **below** a configurable threshold (default **0.6** — lower distance means a closer match).

## Features

- **1:N face matching** with embedding comparison (DeepFace / Facenet family).
- **One folder per criminal** under `criminals/` with `metadata.json` + reference images.
- **Flask REST API** for integration with other modules (`/api/v1/detect`).
- **MongoDB** logging of each check (works without MongoDB — logs a warning and continues).
- **Screenshots** of confirmed matches saved under `logs/`.
- **Windows popup** (`MessageBox`) and **optional system beep** on high alert (configurable).

## Project layout

```
criminal_alert_module/
  main.py           # Flask app & routes
  face_matcher.py   # DeepFace gallery + comparison
  database.py       # MongoDB persistence
  alert_system.py   # Sound + popup
  config.py         # Environment / defaults
  requirements.txt
  criminals/        # Gallery (you populate)
  sample_gallery/   # Example folder naming + metadata templates
  logs/             # Match screenshots (gitignored images)
  docs/schema_mysql.sql  # Optional SQL schema
```

## Installation

**Prerequisites:** Python 3.9+, [MongoDB](https://www.mongodb.com/) (optional but recommended for production).

```bash
cd criminal_alert_module
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

First run will download **DeepFace** model weights (and TensorFlow if needed); ensure outbound network is allowed.

Copy `.env.example` to `.env` and adjust paths, Mongo URI, and `MATCH_THRESHOLD` if needed.

### MongoDB quick start (Docker)

```bash
docker run -d --name mongo -p 27017:27017 mongo:7
```

## Criminal gallery

1. Copy structure from `sample_gallery/` into `criminals/` (or create new UUID-style folder names).
2. For each person add **one or more** clear frontal images (`.jpg` / `.png` / …).
3. Ensure `metadata.json` exists:

```json
{
  "display_name": "Jane Doe",
  "crime_type": "Armed robbery",
  "external_id": "WARRANT-2024-8891"
}
```

4. Reload embeddings without restarting:

```bash
curl -X POST http://127.0.0.1:5000/api/v1/reload-gallery
```

## Run the API

```bash
python main.py
```

Defaults: `http://0.0.0.0:5000` — see `config.py` / `.env`.

## Sample API request

**Multipart upload (recommended):**

```bash
curl -s -X POST http://127.0.0.1:5000/api/v1/detect ^
  -F "image=@C:\path\to\face_crop.jpg" ^
  -F "threshold=0.6"
```

(On Linux/macOS remove `^` and use `\` line continuation or a single line.)

**JSON response shape:**

```json
{
  "matched": true,
  "criminal_name": "Jane Doe",
  "crime_type": "Armed robbery",
  "confidence": 0.8512,
  "alert_level": "HIGH_ALERT",
  "distance": 0.1488,
  "detection_time": "2026-05-16T12:34:56.789012+00:00",
  "screenshot_path": "C:\\...\\criminal_alert_module\\logs\\MATCH_20260516T123456Z_Jane_Doe.jpg"
}
```

When **no match**: `matched` is `false`, `alert_level` is `NONE`, `screenshot_path` is `null` (screenshots are only stored for confirmed matches).

## Sample MongoDB schema

Collection: `detection_logs` (database `surveillance_alerts` by default).

```javascript
{
  "_id": ObjectId("..."),
  "matched": true,
  "criminal_name": "Jane Doe",
  "crime_type": "Armed robbery",
  "confidence": 0.8512,
  "distance": 0.1488,
  "alert_level": "HIGH_ALERT",
  "screenshot_path": "C:/.../logs/MATCH_....jpg",
  "detected_at": ISODate("2026-05-16T12:34:56.789Z"),
  "source": "api",
  "extra": { "threshold": 0.6 }
}
```

MySQL-style DDL is provided in `docs/schema_mysql.sql`.

## Threshold semantics

- **Metric:** cosine distance via DeepFace helpers (default).
- **Rule:** `matched == true` when **best distance \< MATCH_THRESHOLD**.
- **Confidence (API):** `max(0, 1 - distance)` for interpretation alongside your operational tuning.

Tune `MATCH_THRESHOLD` per camera quality and model; typical cosine thresholds often fall in **0.35–0.7** depending on setup.

## Production notes

- Run behind **gunicorn** / **waitress** + reverse proxy; use `create_app()` as the WSGI entry.
- Pre-warm the gallery at deploy time and call `/api/v1/reload-gallery` after bulk imports.
- For headless servers, set `ENABLE_POPUP=false` and forward alerts to your SIEM instead.

## License

Educational / integration template — ensure compliance with local law and privacy policy before deploying on real biometric data.
