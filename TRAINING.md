# Supervised Learning — Training Guide

This document describes the **supervised learning module** (`train_model.py`) for the AI Face Recognition Security System.

**Branch:** `supervised`  
**Owner:** Training pipeline (local dataset → `trained_model.pkl`)

---

## What this module does

1. Reads face images from `dataset/<person_name>/`
2. Detects faces and builds **128-dimensional encodings** (`face_recognition`)
3. Saves a gallery to **`trained_model.pkl`**
4. Runs **leave-one-out evaluation** and recommends a **match tolerance**
5. Writes **`training_report.json`** for reports and demos

No changes to `app.py`, `camera.py`, or `face_engine.py` are required for training.

---

## Dataset layout (required)

Create one folder per classmate. The **folder name** is the label shown on camera/UI.

```text
face_recognition_system/
  dataset/
    Ahmed_Khan/
      img_01.jpg
      img_02.jpg
      img_03.png
    Sara_Ali/
      photo_a.jpg
      photo_b.jpg
```

### Photo guidelines (for best accuracy)

| Rule | Recommendation |
|------|----------------|
| Photos per person | **3–8** clear images |
| Face | Frontal, eyes visible |
| Lighting | Bright, even (avoid heavy shadows) |
| Background | Simple is fine |
| Group photos | Allowed — script uses the **largest face** |
| Formats | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` |
| Folder name | Use the display name (e.g. `Fatima_Rahman`) |

---

## Setup

From the project root:

```powershell
cd C:\Users\fathi\OneDrive\Desktop\erum\face_recognition_system
```

Install dependencies (if not already):

```powershell
pip install -r requirements.txt
```

---

## Commands

### 1. Validate dataset only (no training)

Use this when photos are added but you want to check folders first:

```powershell
python train_model.py --check-only
```

Stricter check (warn if fewer than 3 photos per person):

```powershell
python train_model.py --check-only --min-photos 3
```

### 2. Train (recommended quality — CNN detector)

```powershell
python train_model.py
```

### 3. Train faster on CPU (HOG detector)

```powershell
python train_model.py --hog-only
```

### 4. Custom tolerance

```powershell
python train_model.py --tolerance 0.60
```

### 5. Skip evaluation (very small test sets)

```powershell
python train_model.py --no-eval
```

---

## Outputs

| File | Description |
|------|-------------|
| `trained_model.pkl` | Trained gallery (encodings + names + tolerance + metrics) |
| `training_report.json` | Human-readable summary for reports |

### Example success output

```text
Total trained faces : 24
People in model     : 6
Recommended tolerance: 0.60
LOO accuracy        : 91.7%
Model saved to      : ...\trained_model.pkl
```

---

## Understanding evaluation

### Leave-one-out (LOO) accuracy

- Each trained photo is tested once while the rest act as the gallery.
- **Higher is better.** Aim for **≥ 85%** on your classmate set.
- Needs **at least 2 photos per person** to run meaningfully.

### Recommended tolerance

- Automatically chosen from `0.50` … `0.75`.
- Stored in the model as `recommended_tolerance`.
- **Lower** = stricter (fewer wrong names, more “Unknown”).
- **Higher** = looser (more matches, more false positives).

Use the recommended value in live recognition when possible.

---

## Using the model in code (camera / app)

`train_model.py` exposes helpers so teammates can integrate without editing the training file:

```python
import face_recognition
from train_model import load_trained_model, predict_with_confidence

model = load_trained_model()  # loads trained_model.pkl

# encoding from a webcam frame or uploaded image
name, distance, confidence = predict_with_confidence(face_encoding, model)

if name == "Unknown":
    print(f"No match (distance={distance:.2f}, confidence={confidence:.2f})")
else:
    print(f"{name} — {confidence:.0%} confident")
```

### Model file contents (v2)

- `encodings` — list of 128-d vectors  
- `names` — label per encoding  
- `unique_names` — list of people  
- `match_tolerance` — value used during training  
- `recommended_tolerance` — value from LOO sweep  
- `metadata.evaluation` — accuracy and per-person stats  

---

## Workflow with your partner

| Step | Who | Action |
|------|-----|--------|
| 1 | Partner | Creates `dataset/<ClassmateName>/` with photos |
| 2 | You | `python train_model.py --check-only` |
| 3 | You | `python train_model.py` |
| 4 | You | Share `trained_model.pkl` (USB / Drive / release) |
| 5 | Partner | Wires `camera.py` to load pkl or uses `predict_with_confidence` |

**Note:** `camera.py` on `main` may still expect a sklearn-style pickle. Coordinate with your partner to use the helpers above or update camera loading together in a separate PR.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Dataset folder not found` | Create `dataset/` in project root |
| `No person folders` | Add subfolders: `dataset/Name/photo.jpg` |
| `No faces encoded` | Use clearer frontal photos |
| `SKIP — no face detected` | Retake photo; improve lighting |
| `WARN — multiple faces` | Normal; largest face was used |
| Training very slow | Use `--hog-only` |
| Low LOO accuracy | Add more photos per person; retake blurry images |
| LFW / `seed_dataset.py` hangs | Not required for this module; use local `dataset/` only |

---

## Git — what to commit

**Commit:**

- `train_model.py`
- `TRAINING.md`

**Do not commit:**

- `trained_model.pkl` (local / shared separately)
- `training_report.json` (optional; regenerate anytime)
- `dataset/` (personal photos — unless team agrees)
- `.venv/`, `myenv/`, `_quick_cache/`

```powershell
git add train_model.py TRAINING.md
git commit -m "Add supervised training documentation"
git push origin supervised
```

---

## Quick reference

```powershell
python train_model.py --check-only --min-photos 3
python train_model.py
streamlit run app.py
```

For questions about the rest of the system (Streamlit UI, SQLite, LFW seeding), see the main `README.md`.
