# Face detection — dataset prep

This repo is set up for **supervised learning dataset preparation**: one folder per person, **one** face photo each (defaults in `prepare_dataset.py`; you can raise limits with `check --min` / `--max`).

**Start here:** open [`DATASET_GUIDE.md`](DATASET_GUIDE.md) for step-by-step beginner instructions.

Quick start:

```powershell
cd $HOME\Desktop\face_detection
python prepare_dataset.py init
pip install -r requirements-dataset.txt
python prepare_dataset.py check
```

Photos go in `dataset/<PersonName>/` (see `people.txt`).
