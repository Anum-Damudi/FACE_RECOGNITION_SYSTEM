"""
Supervised face gallery training — optimized for classmate photos (Nadisah dataset).

Default dataset (team):
    face_detection/FACE_DETECTION/<NAME>/*.jpg

Also supports:
    dataset/<NAME>/*.jpg

Outputs:
    trained_model.pkl   — camera-ready (predict / predict_proba) + gallery metadata
    training_report.json

Run:
    python train_model.py
    python train_model.py --dataset face_detection/FACE_DETECTION --hog-only
    python train_model.py --check-only
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import face_recognition
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from face_engine import NUM_JITTERS, face_locations_robust, upscale_if_tiny

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
NADISAH_DATASET = PROJECT_ROOT / "face_detection" / "FACE_DETECTION"
LOCAL_DATASET = PROJECT_ROOT / "dataset"
MODEL_PATH = PROJECT_ROOT / "trained_model.pkl"
REPORT_PATH = PROJECT_ROOT / "training_report.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORE_FILENAMES = {"PUT_YOUR_PHOTOS_HERE.txt", "README.txt", ".gitkeep"}

MODEL_VERSION = 3
DEFAULT_TOLERANCE = 0.60
TOLERANCE_CANDIDATES = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
MIN_JITTERS_FEW_PHOTOS = 10  # more stable vectors when only 1–2 photos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_model")


# ---------------------------------------------------------------------------
# Camera-compatible wrapper (Aysha's camera.py uses .predict / .predict_proba)
# ---------------------------------------------------------------------------


class FaceRecognitionModel:
    """
    Pickled object for camera.py: sklearn KNN + distance fallback metadata.
    """

    def __init__(
        self,
        classifier: KNeighborsClassifier,
        class_names: List[str],
        gallery_encodings: List[np.ndarray],
        gallery_names: List[str],
        match_tolerance: float,
        recommended_tolerance: float,
        metadata: Dict[str, Any],
    ) -> None:
        self.classifier = classifier
        self.class_names = class_names
        self.gallery_encodings = gallery_encodings
        self.gallery_names = gallery_names
        self.match_tolerance = match_tolerance
        self.recommended_tolerance = recommended_tolerance
        self.metadata = metadata

    def predict(self, encodings: Sequence[np.ndarray]) -> np.ndarray:
        labels = []
        tol = self.recommended_tolerance
        for enc in encodings:
            enc = np.asarray(enc, dtype=np.float64)
            knn_label = self.classifier.predict([enc])[0]
            distance = self._min_distance(enc, knn_label)
            labels.append("Unknown" if distance > tol else knn_label)
        return np.array(labels)

    def predict_proba(self, encodings: Sequence[np.ndarray]) -> np.ndarray:
        rows = []
        for enc in encodings:
            enc = np.asarray(enc, dtype=np.float64)
            try:
                proba = self.classifier.predict_proba([enc])[0]
                rows.append(proba)
            except Exception:
                n = len(self.class_names)
                rows.append(np.ones(n) / max(n, 1))
        return np.array(rows)

    def _min_distance(self, query: np.ndarray, label: str) -> float:
        group = [
            e for e, n in zip(self.gallery_encodings, self.gallery_names) if n == label
        ]
        if not group:
            return 1.0
        return float(np.min(face_recognition.face_distance(group, query)))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrainingStats:
    people: int = 0
    images_scanned: int = 0
    faces_trained: int = 0
    augmented_samples: int = 0
    skipped_no_face: int = 0
    skipped_multi_face: int = 0
    skipped_invalid: int = 0
    skipped_errors: int = 0
    per_person: Dict[str, int] = field(default_factory=dict)


@dataclass
class DatasetReport:
    person_folders: int = 0
    total_images: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    per_person_images: Dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def resolve_default_dataset() -> Path:
    """Prefer Nadisah's folder when it exists."""
    if NADISAH_DATASET.is_dir() and any(NADISAH_DATASET.iterdir()):
        return NADISAH_DATASET
    return LOCAL_DATASET


def discover_person_folders(dataset_dir: Path) -> List[Tuple[str, Path]]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_dir}\n"
            f"Expected: {NADISAH_DATASET} or {LOCAL_DATASET}"
        )

    people = [
        (entry.name, entry)
        for entry in sorted(dataset_dir.iterdir())
        if entry.is_dir() and not entry.name.startswith(".")
    ]
    if not people:
        raise ValueError(f"No person folders in {dataset_dir}")
    return people


def iter_image_files(person_dir: Path) -> List[Path]:
    return [
        path
        for path in sorted(person_dir.iterdir())
        if path.is_file()
        and path.name not in IGNORE_FILENAMES
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def load_rgb_image(image_path: Path) -> Optional[np.ndarray]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def validate_dataset(dataset_dir: Path, min_photos: int = 1) -> DatasetReport:
    report = DatasetReport()
    for label, folder in discover_person_folders(dataset_dir):
        images = iter_image_files(folder)
        report.per_person_images[label] = len(images)
        report.total_images += len(images)
        report.person_folders += 1
        if len(images) == 0:
            report.errors.append(f"[{label}] no images (only placeholders?)")
        elif len(images) < min_photos:
            report.warnings.append(
                f"[{label}] {len(images)} photo(s) — recommend {min_photos}+ for best accuracy"
            )
    if report.total_images == 0:
        report.errors.append("No training images found.")
    return report


# ---------------------------------------------------------------------------
# Augmentation (critical when Nadisah has only 1 photo per person)
# ---------------------------------------------------------------------------


def augment_rgb_variants(rgb: np.ndarray) -> List[Tuple[np.ndarray, str]]:
    """Cheap augmentations that help single-photo classes."""
    variants: List[Tuple[np.ndarray, str]] = [("original", rgb.copy())]

    flipped = cv2.flip(rgb, 1)
    variants.append(("mirror", flipped))

    brighter = np.clip(rgb.astype(np.float32) * 1.12, 0, 255).astype(np.uint8)
    variants.append(("bright", brighter))

    darker = np.clip(rgb.astype(np.float32) * 0.88, 0, 255).astype(np.uint8)
    variants.append(("dark", darker))

    return variants


def encode_largest_face(
    rgb: np.ndarray,
    use_cnn: bool,
    num_jitters: int,
) -> Tuple[Optional[np.ndarray], str]:
    rgb = upscale_if_tiny(rgb)
    locations = face_locations_robust(rgb, use_cnn=use_cnn)
    if not locations:
        return None, "no_face"

    if len(locations) > 1:
        areas = [(loc[2] - loc[0]) * (loc[1] - loc[3]) for loc in locations]
        locations = [locations[int(np.argmax(areas))]]
        status = "multi_face"
    else:
        status = "ok"

    encodings = face_recognition.face_encodings(rgb, locations, num_jitters=num_jitters)
    if not encodings:
        return None, "no_face"
    return np.asarray(encodings[0], dtype=np.float64), status


def encode_from_path(
    image_path: Path,
    use_cnn: bool,
    num_jitters: int,
) -> Tuple[Optional[np.ndarray], str]:
    rgb = load_rgb_image(image_path)
    if rgb is None:
        return None, "invalid"
    return encode_largest_face(rgb, use_cnn=use_cnn, num_jitters=num_jitters)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def match_best_person(
    query: np.ndarray,
    encodings: List[np.ndarray],
    names: List[str],
) -> Tuple[str, float]:
    grouped: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
    for enc, name in zip(encodings, names):
        grouped[name].append(np.asarray(enc, dtype=np.float64))

    best_name, best_dist = "Unknown", float("inf")
    for name, group in grouped.items():
        d = float(np.min(face_recognition.face_distance(group, query)))
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name, best_dist


def leave_one_out_accuracy(
    encodings: List[np.ndarray],
    names: List[str],
    tolerance: float,
) -> Tuple[int, int, float]:
    correct = tested = 0
    by_name: DefaultDict[str, List[int]] = defaultdict(list)
    for i, name in enumerate(names):
        by_name[name].append(i)

    for name, indices in by_name.items():
        if len(indices) < 2:
            continue
        for hold in indices:
            query = encodings[hold]
            gallery_enc = [encodings[i] for i in range(len(encodings)) if i != hold]
            gallery_names = [names[i] for i in range(len(names)) if i != hold]
            pred, dist = match_best_person(query, gallery_enc, gallery_names)
            tested += 1
            if pred == name and dist <= tolerance:
                correct += 1

    acc = (correct / tested) if tested else 0.0
    return correct, tested, acc


def pick_best_tolerance(
    encodings: List[np.ndarray],
    names: List[str],
) -> Tuple[float, Dict[str, float]]:
    best_tol, best_acc = DEFAULT_TOLERANCE, -1.0
    sweep: Dict[str, float] = {}
    for tol in TOLERANCE_CANDIDATES:
        _, tested, acc = leave_one_out_accuracy(encodings, names, tol)
        sweep[f"{tol:.2f}"] = round(acc, 4) if tested else 0.0
        if tested and acc > best_acc:
            best_acc, best_tol = acc, tol
    return best_tol, sweep


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_knn_classifier(
    encodings: List[np.ndarray],
    names: List[str],
) -> KNeighborsClassifier:
    """KNN on 128-d embeddings — strong baseline for face recognition."""
    x = np.array(encodings, dtype=np.float64)
    y = np.array(names)
    n_neighbors = min(3, max(1, len(encodings) // max(len(set(names)), 1)))
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean", weights="distance")
    clf.fit(x, y)
    return clf


def train_from_dataset(
    dataset_dir: Path,
    use_cnn: bool = True,
    match_tolerance: float = DEFAULT_TOLERANCE,
    min_photos_warn: int = 3,
    augment_if_few: int = 3,
    run_evaluation: bool = True,
) -> FaceRecognitionModel:
    validation = validate_dataset(dataset_dir, min_photos=1)
    for w in validation.warnings:
        logger.warning(w)
    if not validation.ok:
        raise ValueError("\n".join(validation.errors))

    stats = TrainingStats()
    encodings: List[np.ndarray] = []
    names: List[str] = []
    source_paths: List[str] = []

    people = discover_person_folders(dataset_dir)
    stats.people = len(people)
    logger.info("Training on %d people from %s", stats.people, dataset_dir)

    for person_label, person_dir in people:
        image_files = iter_image_files(person_dir)
        if not image_files:
            logger.warning("  [%s] skip — no images", person_label)
            continue

        raw_count = len(image_files)
        use_augment = raw_count < augment_if_few
        jitters = MIN_JITTERS_FEW_PHOTOS if raw_count <= 2 else NUM_JITTERS

        logger.info(
            "  [%s] %d image(s)%s",
            person_label,
            raw_count,
            " + augmentation" if use_augment else "",
        )

        person_encodings = 0
        for image_path in image_files:
            stats.images_scanned += 1
            try:
                base_rgb = load_rgb_image(image_path)
                if base_rgb is None:
                    stats.skipped_invalid += 1
                    logger.warning("    SKIP %s — invalid", image_path.name)
                    continue

                variants = (
                    augment_rgb_variants(base_rgb) if use_augment else [("original", base_rgb)]
                )

                for tag, rgb in variants:
                    encoding, status = encode_largest_face(rgb, use_cnn, jitters)
                    if encoding is None:
                        if tag == "original":
                            stats.skipped_no_face += 1
                            logger.warning("    SKIP %s — no face", image_path.name)
                        continue

                    if status == "multi_face" and tag == "original":
                        stats.skipped_multi_face += 1
                        logger.info("    WARN %s — multiple faces, used largest", image_path.name)

                    encodings.append(encoding)
                    names.append(person_label)
                    source_paths.append(f"{person_label}/{image_path.name}#{tag}")
                    person_encodings += 1
                    stats.faces_trained += 1
                    if tag != "original":
                        stats.augmented_samples += 1

                logger.info("    OK   %s → %d encoding(s)", image_path.name, person_encodings)

            except Exception as exc:
                stats.skipped_errors += 1
                logger.warning("    SKIP %s — %s", image_path.name, exc)

        stats.per_person[person_label] = person_encodings

    if not encodings:
        raise RuntimeError("No faces encoded. Use clearer frontal photos.")

    recommended_tol, tolerance_sweep = pick_best_tolerance(encodings, names)
    if not run_evaluation:
        recommended_tol = match_tolerance

    _, tested, acc_at_rec = leave_one_out_accuracy(encodings, names, recommended_tol)
    _, _, acc_at_match = leave_one_out_accuracy(encodings, names, match_tolerance)

    classifier = build_knn_classifier(encodings, names)
    unique_names = sorted(set(names))

    metadata: Dict[str, Any] = {
        "version": MODEL_VERSION,
        "dataset_dir": str(dataset_dir.resolve()),
        "encoder": "face_recognition",
        "detector": "cnn" if use_cnn else "hog",
        "class_names": unique_names,
        "stats": {
            "people": stats.people,
            "images_scanned": stats.images_scanned,
            "faces_trained": stats.faces_trained,
            "augmented_samples": stats.augmented_samples,
            "skipped_no_face": stats.skipped_no_face,
            "skipped_multi_face": stats.skipped_multi_face,
            "skipped_invalid": stats.skipped_invalid,
            "skipped_errors": stats.skipped_errors,
            "per_person": stats.per_person,
        },
        "evaluation": {
            "loo_tested_samples": tested,
            "loo_accuracy_recommended": round(acc_at_rec, 4),
            "loo_accuracy_match_tolerance": round(acc_at_match, 4),
            "recommended_tolerance": recommended_tol,
            "tolerance_sweep": tolerance_sweep,
        },
        "source_paths": source_paths,
    }

    _log_summary(stats, unique_names, recommended_tol, acc_at_rec, tested)

    return FaceRecognitionModel(
        classifier=classifier,
        class_names=unique_names,
        gallery_encodings=encodings,
        gallery_names=names,
        match_tolerance=match_tolerance,
        recommended_tolerance=recommended_tol,
        metadata=metadata,
    )


def save_model(model: FaceRecognitionModel, path: Path = MODEL_PATH) -> None:
    with path.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved model → %s", path.resolve())


def save_report(model: FaceRecognitionModel, path: Path = REPORT_PATH) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "people": model.class_names,
        "total_encodings": len(model.gallery_encodings),
        "match_tolerance": model.match_tolerance,
        "recommended_tolerance": model.recommended_tolerance,
        "metadata": model.metadata,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Report saved → %s", path.resolve())


def _log_summary(
    stats: TrainingStats,
    names: List[str],
    tol: float,
    acc: float,
    tested: int,
) -> None:
    logger.info("-" * 58)
    logger.info("  People            : %d", len(names))
    logger.info("  Images scanned    : %d", stats.images_scanned)
    logger.info("  Encodings stored  : %d", stats.faces_trained)
    logger.info("  Augmented extras  : %d", stats.augmented_samples)
    logger.info("  Recommended tol.  : %.2f", tol)
    if tested:
        logger.info("  LOO accuracy      : %.1f%% (%d tests)", acc * 100, tested)
    else:
        logger.info("  LOO accuracy      : n/a (need 2+ samples per person)")
    for name in names:
        logger.info("    %s: %d", name, stats.per_person.get(name, 0))
    logger.info("-" * 58)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    default_ds = resolve_default_dataset()
    parser = argparse.ArgumentParser(description="Train supervised face model from photo folders.")
    parser.add_argument("--dataset", type=Path, default=default_ds)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--hog-only", action="store_true", help="Faster HOG detector")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--min-photos", type=int, default=3, help="Warn if fewer per person")
    parser.add_argument("--check-only", action="store_true", help="Validate dataset only")
    parser.add_argument("--no-augment", action="store_true", help="Disable flip/brightness aug")
    parser.add_argument("--no-eval", action="store_true", help="Skip tolerance tuning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cnn = not args.hog_only
    augment_if_few = 999 if args.no_augment else 3

    logger.info("Supervised training (v%d)", MODEL_VERSION)
    logger.info("  Dataset : %s", args.dataset.resolve())
    logger.info("  Output  : %s", args.output.resolve())

    try:
        if args.check_only:
            report = validate_dataset(args.dataset, min_photos=1)
            print(f"\nFolders: {report.person_folders}  Images: {report.total_images}\n")
            for name, n in sorted(report.per_person_images.items()):
                print(f"  {name}: {n}")
            for w in report.warnings:
                print(f"WARNING: {w}")
            for e in report.errors:
                print(f"ERROR: {e}")
            sys.exit(0 if report.ok else 1)

        model = train_from_dataset(
            dataset_dir=args.dataset,
            use_cnn=use_cnn,
            match_tolerance=args.tolerance,
            min_photos_warn=args.min_photos,
            augment_if_few=augment_if_few,
            run_evaluation=not args.no_eval,
        )
        save_model(model, args.output)
        save_report(model, args.report)

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    ev = model.metadata.get("evaluation", {})
    print(f"\nTotal encodings     : {len(model.gallery_encodings)}")
    print(f"People              : {', '.join(model.class_names)}")
    print(f"Recommended tolerance: {model.recommended_tolerance:.2f}")
    if ev.get("loo_tested_samples"):
        print(f"LOO accuracy        : {ev['loo_accuracy_recommended'] * 100:.1f}%")
    print(f"Model saved         : {args.output.resolve()}")
    print("Camera: Aysha can use trained_model.pkl with camera.py (predict works).")


if __name__ == "__main__":
    main()
