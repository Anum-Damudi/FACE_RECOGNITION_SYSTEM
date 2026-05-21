#!/usr/bin/env python3
"""
Beginner-friendly helpers for a person-based image dataset.

Folder layout (example):
  dataset/
    Sabiha/
      photo.jpg
    Aatufa/
      ...

Usage (in Terminal inside this project folder):
  python prepare_dataset.py init              # create dataset/<person>/ folders
  python prepare_dataset.py check             # count & validate images
  python prepare_dataset.py list-people       # show names from people.txt

Requires (for `check`): pip install -r requirements-dataset.txt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# How many images `check` expects per person (change here or use --min / --max on the CLI)
DEFAULT_MIN_PHOTOS = 1
DEFAULT_MAX_PHOTOS = 1

# Default names from your plan — edit people.txt to change or add names
DEFAULT_PEOPLE = ["Sabiha", "Aatufa", "Nadisah", "Aysha"]
PEOPLE_FILE = Path("people.txt")


def project_root() -> Path:
    return Path(__file__).resolve().parent


def read_people() -> list[str]:
    root = project_root()
    path = root / PEOPLE_FILE
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    return list(DEFAULT_PEOPLE)


def dataset_dir() -> Path:
    return project_root() / "dataset"


def init_folders() -> None:
    root = dataset_dir()
    root.mkdir(parents=True, exist_ok=True)
    people = read_people()
    if not people:
        print("No names found. Edit people.txt or fix DEFAULT_PEOPLE in prepare_dataset.py.")
        sys.exit(1)
    for name in people:
        folder = root / name
        folder.mkdir(exist_ok=True)
        readme = folder / "PUT_YOUR_PHOTOS_HERE.txt"
        readme.write_text(
            f"Add exactly 1 clear photo of ONLY {name} in this folder.\n"
            "Front face, no groups, not blurry. Remove any extra images so only one file remains.\n",
            encoding="utf-8",
        )
    print(f"Ready: {root}")
    for name in people:
        print(f"  - {root / name}")


def list_image_files(folder: Path) -> list[Path]:
    out: list[Path] = []
    if not folder.is_dir():
        return out
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(p)
    return out


def optional_imports():
    try:
        from PIL import Image  # noqa: WPS433
    except ImportError:
        Image = None  # type: ignore
    try:
        import cv2  # noqa: WPS433
    except ImportError:
        cv2 = None  # type: ignore
    return Image, cv2


def blur_score_bgr(img_bgr) -> float:
    import cv2  # local; only when cv2 present

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_dataset(min_photos: int, max_photos: int, blur_threshold: float) -> int:
    Image, cv2 = optional_imports()
    if Image is None:
        print("Install Pillow for image checks: pip install -r requirements-dataset.txt")
    if cv2 is None:
        print("(Optional) Install OpenCV for blur hints: pip install -r requirements-dataset.txt")

    root = dataset_dir()
    if not root.is_dir():
        print(f"Missing {root}. Run: python prepare_dataset.py init")
        return 1

    people = read_people()
    exit_code = 0
    warnings: list[str] = []

    for name in people:
        folder = root / name
        files = list_image_files(folder)
        n = len(files)
        print(f"\n{name}/  -> {n} image(s)")

        if n < min_photos:
            print(f"  [!] Need at least {min_photos} photos (you have {n}).")
            exit_code = 1
        elif n > max_photos:
            print(
                f"  [!] You have {n} photos; this project expects at most {max_photos}. "
                f"Delete extras so only {max_photos} image(s) remain."
            )
            exit_code = 1

        for fp in files:
            rel = fp.relative_to(root)
            if Image is not None:
                try:
                    with Image.open(fp) as im:
                        im.verify()
                except Exception as e:  # noqa: BLE001
                    print(f"  [x] Corrupt or unreadable: {rel} ({e})")
                    exit_code = 1
                    continue
                try:
                    with Image.open(fp) as im:
                        w, h = im.size
                except Exception as e:  # noqa: BLE001
                    print(f"  [x] Could not read size: {rel} ({e})")
                    exit_code = 1
                    continue
                if min(w, h) < 112:
                    warnings.append(f"{rel}: very small ({w}x{h}); prefer larger faces.")
                if cv2 is not None:
                    bgr = cv2.imread(str(fp))
                    if bgr is None:
                        print(f"  [x] OpenCV could not read: {rel}")
                        exit_code = 1
                    else:
                        score = blur_score_bgr(bgr)
                        if score < blur_threshold:
                            warnings.append(f"{rel}: may be blurry (score={score:.1f}; threshold {blur_threshold}).")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    print("\nDone checking.")
    return exit_code


def copy_from_import(source: Path, person: str, yes: bool) -> int:
    """Copy supported images from a folder (e.g. Downloads batch) into dataset/<person>/."""
    root = dataset_dir()
    dest = root / person
    if not dest.is_dir():
        print(f"Create folder first: python prepare_dataset.py init\nMissing: {dest}")
        return 1
    if not source.is_dir():
        print(f"Not a folder: {source}")
        return 1
    files = [p for p in sorted(source.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        print(f"No images found in {source}")
        return 1
    print(f"Will copy {len(files)} file(s) into {dest}")
    if not yes:
        s = input("Continue? [y/N]: ").strip().lower()
        if s != "y":
            print("Cancelled.")
            return 0
    for p in files:
        target = dest / p.name
        if target.exists():
            stem, suf = p.stem, p.suffix
            k = 1
            while target.exists():
                target = dest / f"{stem}_{k}{suf}"
                k += 1
        shutil.copy2(p, target)
        print(f"  copied -> {target.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Person photo dataset helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create dataset/ and person subfolders")
    p_init.set_defaults(func=lambda _: (init_folders(), 0))

    p_check = sub.add_parser("check", help="Count images and run basic checks")
    p_check.add_argument(
        "--min",
        type=int,
        default=DEFAULT_MIN_PHOTOS,
        dest="min_photos",
        help=f"Minimum images per person (default: {DEFAULT_MIN_PHOTOS})",
    )
    p_check.add_argument(
        "--max",
        type=int,
        default=DEFAULT_MAX_PHOTOS,
        dest="max_photos",
        help=f"Maximum images per person (default: {DEFAULT_MAX_PHOTOS})",
    )
    p_check.add_argument(
        "--blur",
        type=float,
        default=100.0,
        help="Laplacian variance threshold (lower = stricter blur warning). Typical 50-150.",
    )
    p_check.set_defaults(func=lambda a: (None, check_dataset(a.min_photos, a.max_photos, a.blur)))

    p_list = sub.add_parser("list-people", help="Print names from people.txt / defaults")
    p_list.set_defaults(func=lambda _: (print("\n".join(read_people())), 0))

    p_copy = sub.add_parser("import-folder", help="Copy all images from a folder into one person folder")
    p_copy.add_argument("source", type=Path, help="Folder that contains images")
    p_copy.add_argument("person", type=str, help="Person folder name, e.g. Sabiha")
    p_copy.add_argument("-y", action="store_true", help="Do not ask for confirmation")
    p_copy.set_defaults(func=lambda a: (None, copy_from_import(a.source.expanduser().resolve(), a.person, a.y)))

    args = parser.parse_args()
    _, code = args.func(args)
    return int(code) if isinstance(code, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
