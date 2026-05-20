"""
Train a face gallery from local images and save encodings to trained_model.pkl.

Layout:
    dataset/
        Alice/
            photo1.jpg
            photo2.png
        Bob/
            ...

Run:
    python train_model.py
    python train_model.py --dataset dataset --hog-only
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from face_engine import NUM_JITTERS, encode_single_face_rgb

# ---------------------------------------------------------------------------
# Paths (project root = folder containing this file)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_PATH = PROJECT_ROOT / "trained_model.pkl"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_model")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrainingStats:
    """Running counters for the training pass."""

    people: int = 0
    images_scanned: int = 0
    faces_trained: int = 0
    skipped_no_face: int = 0
    skipped_invalid: int = 0
    skipped_errors: int = 0
    per_person: Dict[str, int] = field(default_factory=dict)


@dataclass
class TrainedModelBundle:
    """
    Serialized gallery used by camera.py and app.py.

    confidence is derived at inference from face distance vs match_tolerance.
    """

    version: int
    created_at: str
    match_tolerance: float
    num_jitters: int
    encodings: List[np.ndarray]
    names: List[str]
    class_counts: Dict[str, int]
    metadata: Dict[str, object]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "match_tolerance": self.match_tolerance,
            "num_jitters": self.num_jitters,
            "encodings": self.encodings,
            "names": self.names,
            "class_counts": self.class_counts,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def discover_person_folders(dataset_dir: Path) -> List[Tuple[str, Path]]:
    """Return (label, folder_path) for each subfolder in dataset/."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}\n"
            f"Create it and add one folder per person, e.g. {dataset_dir / 'Alice'}"
        )

    people: List[Tuple[str, Path]] = []
    for entry in sorted(dataset_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            people.append((entry.name, entry))

    if not people:
        raise ValueError(
            f"No person folders in {dataset_dir}. "
            "Add subfolders named after each person (class label)."
        )
    return people


def iter_image_files(person_dir: Path) -> List[Path]:
    """Collect image paths for one person folder."""
    files = [
        p
        for p in sorted(person_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return files


def load_rgb_image(image_path: Path) -> Optional[np.ndarray]:
    """Decode image to RGB; return None if unreadable."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def encode_face_from_path(
    image_path: Path,
    use_cnn: bool,
) -> Optional[np.ndarray]:
    """Load image and return a single 128-d face encoding, or None."""
    rgb = load_rgb_image(image_path)
    if rgb is None:
        return None
    return encode_single_face_rgb(rgb, use_cnn=use_cnn)


def train_from_dataset(
    dataset_dir: Path,
    use_cnn: bool = True,
    match_tolerance: float = 0.65,
) -> TrainedModelBundle:
    """
    Walk dataset/<person>/images, build encodings list aligned with names.
    """
    stats = TrainingStats()
    encodings: List[np.ndarray] = []
    names: List[str] = []

    people = discover_person_folders(dataset_dir)
    stats.people = len(people)

    logger.info("Found %d person folder(s) in %s", stats.people, dataset_dir)

    for person_label, person_dir in people:
        image_files = iter_image_files(person_dir)
        if not image_files:
            logger.warning("  [%s] no images — skipping folder", person_label)
            continue

        person_count = 0
        logger.info("  [%s] %d image(s)", person_label, len(image_files))

        for image_path in image_files:
            stats.images_scanned += 1
            try:
                encoding = encode_face_from_path(image_path, use_cnn=use_cnn)
            except Exception as exc:
                stats.skipped_errors += 1
                logger.warning("    SKIP %s — error: %s", image_path.name, exc)
                continue

            if encoding is None:
                rgb = load_rgb_image(image_path)
                if rgb is None:
                    stats.skipped_invalid += 1
                    logger.warning("    SKIP %s — invalid or unreadable image", image_path.name)
                else:
                    stats.skipped_no_face += 1
                    logger.warning("    SKIP %s — no face detected", image_path.name)
                continue

            encodings.append(np.asarray(encoding, dtype=np.float64))
            names.append(person_label)
            person_count += 1
            stats.faces_trained += 1
            logger.info("    OK   %s", image_path.name)

        stats.per_person[person_label] = person_count

    if not encodings:
        raise RuntimeError(
            "No faces were encoded. Add clear frontal photos under dataset/<name>/."
        )

    bundle = TrainedModelBundle(
        version=1,
        created_at=datetime.now(timezone.utc).isoformat(),
        match_tolerance=match_tolerance,
        num_jitters=NUM_JITTERS,
        encodings=encodings,
        names=names,
        class_counts=dict(stats.per_person),
        metadata={
            "dataset_dir": str(dataset_dir.resolve()),
            "encoder": "face_recognition",
            "detector": "cnn" if use_cnn else "hog",
            "stats": {
                "people": stats.people,
                "images_scanned": stats.images_scanned,
                "faces_trained": stats.faces_trained,
                "skipped_no_face": stats.skipped_no_face,
                "skipped_invalid": stats.skipped_invalid,
                "skipped_errors": stats.skipped_errors,
            },
        },
    )

    _log_training_summary(stats)
    return bundle


def save_trained_model(bundle: TrainedModelBundle, output_path: Path = MODEL_PATH) -> None:
    """Persist bundle to disk as trained_model.pkl."""
    payload = bundle.to_dict()
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved model → %s", output_path.resolve())


def _log_training_summary(stats: TrainingStats) -> None:
    logger.info("-" * 52)
    logger.info("Training summary")
    logger.info("  People folders : %d", stats.people)
    logger.info("  Images scanned : %d", stats.images_scanned)
    logger.info("  Faces trained  : %d", stats.faces_trained)
    logger.info("  No face        : %d", stats.skipped_no_face)
    logger.info("  Invalid image  : %d", stats.skipped_invalid)
    logger.info("  Errors         : %d", stats.skipped_errors)
    for name, count in sorted(stats.per_person.items()):
        logger.info("    %s: %d encoding(s)", name, count)
    logger.info("-" * 52)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train face encodings from dataset/<person>/ images."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Root folder with one subfolder per person (default: {DEFAULT_DATASET_DIR.name}/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_PATH,
        help=f"Output pickle path (default: {MODEL_PATH.name})",
    )
    parser.add_argument(
        "--hog-only",
        action="store_true",
        help="Use HOG detector (faster on CPU, slightly less accurate than CNN).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.65,
        help="Default match tolerance stored in the model (0.45–0.75).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cnn = not args.hog_only

    logger.info("Face recognition training started")
    logger.info("  Dataset : %s", args.dataset.resolve())
    logger.info("  Output  : %s", args.output.resolve())
    logger.info("  Detector: %s", "CNN" if use_cnn else "HOG")

    try:
        bundle = train_from_dataset(
            dataset_dir=args.dataset,
            use_cnn=use_cnn,
            match_tolerance=args.tolerance,
        )
        save_trained_model(bundle, args.output)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    print(f"\nTotal trained faces: {len(bundle.encodings)}")
    print(f"Model saved to: {args.output.resolve()}")
    print("Next: streamlit run app.py  (upload or live camera)")


if __name__ == "__main__":
    main()
