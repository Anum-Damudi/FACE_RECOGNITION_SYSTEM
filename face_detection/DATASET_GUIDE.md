# Dataset setup for supervised learning (beginner guide)

This project expects **one folder per person**, each folder containing **only** photos of that person. By default it is set up for **exactly one photo per person** (easy to change — see end of this section).

To allow more photos later, run checks with custom limits, for example:

```powershell
python prepare_dataset.py check --min 5 --max 15
```

Or edit `DEFAULT_MIN_PHOTOS` and `DEFAULT_MAX_PHOTOS` near the top of `prepare_dataset.py`.

## 1. Open this project in Cursor / VS Code

1. Open **Cursor** or **Visual Studio Code**.
2. **File → Open Folder…**
3. Choose your project folder: `Desktop\face_detection` (the folder that contains `prepare_dataset.py`).

You should see `prepare_dataset.py`, `people.txt`, and (after the next step) a `dataset` folder in the sidebar.

## 2. Create the folder structure (automatic)

1. Open the **integrated terminal** in Cursor/VS Code: **Terminal → New Terminal**.
2. Make sure the prompt shows your project folder (`face_detection`). If not, run:

   ```powershell
   cd $HOME\Desktop\face_detection
   ```

3. Run:

   ```powershell
   python prepare_dataset.py init
   ```

This creates:

```text
dataset/
  Sabiha/
  Aatufa/
  Nadisah/
  Aysha/
```

(Names come from `people.txt`. Edit that file to add or rename people, then run `init` again.)

## 3. Put photos in the right folders (how to “add” images)

**Goal:** In `dataset\Sabiha\` put **only** pictures of Sabiha. Same for each name.

### Option A — Drag and drop (easiest)

1. In Windows File Explorer, open your photos (e.g. `Pictures` or phone export).
2. In Cursor/VS Code, expand `dataset → Sabiha` in the sidebar.
3. **Drag exactly one** good photo into `dataset\Sabiha\` (or drop into the folder in Explorer). If there are old extra images, **delete** them so only one image file stays.

File names can be `1.jpg`, `2.jpg`, or anything — the script looks for common image types (`.jpg`, `.png`, etc.).

### Option B — Copy from one folder in bulk

If you have all of someone’s photos in, say, `Downloads\Sabiha_pick`:

```powershell
python prepare_dataset.py import-folder "C:\Users\hp\Downloads\Sabiha_pick" Sabiha -y
```

- Replace the path and `Sabiha` with the correct folder and person name.
- `-y` skips the confirmation prompt.

## 4. Rules for good labels (very important)

| Do | Don’t |
|----|--------|
| Clear **front** or mostly front faces | Group photos (multiple people) |
| Same person **only** in that folder | Mixing different people in one folder |
| Sharp, well-lit (different angles OK) | Very blurry or tiny faces |

If you break the “one person per folder” rule, the model learns wrong names — there is no magic fix later.

## 5. Check everything before training

1. (Once) Install check dependencies:

   ```powershell
   pip install -r requirements-dataset.txt
   ```

2. Run:

   ```powershell
   python prepare_dataset.py check
   ```

You will see how many images each person has. By default the script expects **exactly 1** image (too few or too many both fail `check`). Override with `--min` / `--max` if you switch back to a multi-photo dataset.

## 6. Adding another person later

1. Open `people.txt`.
2. Add a new line with the exact folder name you want (e.g. `Zara`).
3. Run:

   ```powershell
   python prepare_dataset.py init
   ```

4. Add **one** photo into `dataset\Zara\` (or use `check --min` / `--max` if you want more).

## 7. Where this dataset is on disk

Everything lives under your project folder, for example:

`C:\Users\hp\Desktop\face_detection\dataset\`

Backup this `dataset` folder if you care about the work — it is your labeled data.

---

## Quick command reference

| Command | What it does |
|---------|----------------|
| `python prepare_dataset.py init` | Create `dataset/<person>/` folders |
| `python prepare_dataset.py list-people` | Print names from `people.txt` |
| `python prepare_dataset.py check` | Count images + basic quality hints |
| `python prepare_dataset.py import-folder <from> <person> -y` | Copy images into one person folder |

If `python` is not found, try `py prepare_dataset.py init` on Windows.
