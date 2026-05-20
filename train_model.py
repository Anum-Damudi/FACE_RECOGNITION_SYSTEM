"""
Supervised face gallery training for the face recognition security project.

Dataset layout
--------------
    dataset/
        Alice/
            photo1.jpg
            photo2.png
        Bob/
            ...

Each subfolder name is the class label (person name).

Train
-----
    python train_model.py
    python train_model.py --hog-only
    python train_model.py --check-only          # validate folders only
    python train_model.py --min-photos 3        # require 3+ images per person

Output
------
    trained_model.pkl  — encodings, labels, tolerance, evaluation metrics

Inference (for camera / app integration without editing team files)
-------------------------------------------------------------------
    from train_model import load_trained_model, predict_with_confidence

    model = load_trained_model()
    name, distance, confidence = predict_with_confidence(encoding, model)
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
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np

from face_engine import NUM_JITTERS, face_locations_robust, upscale_if_tiny

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_PATH = PROJECT_ROOT / "trained_model.pkl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "training_report.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_VERSION = 2
DEFAULT_TOLERANCE = 0.65
TOLERANCE_CANDIDATES = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)

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
    people: int = 0
    images_scanned: int = 0
    faces_trained: int = 0
    skipped_no_face: int = 0
    skipped_multi_face: int = 0
    skipped_invalid: int = 0
    skipped_errors: int = 0
    per_person: Dict[str, int] = field(default_factory=dict)


@dataclass
class DatasetReport:
    """Pre-training validation summary."""

    person_folders: int = 0
    total_images: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    per_person_images: Dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class EvaluationReport:
    """Leave-one-out cross-check on the trained gallery."""

    tested_samples: int = 0
    correct: int = 0
    accuracy: float = 0.0
    recommended_tolerance: float = DEFAULT_TOLERANCE
    per_person_accuracy: Dict[str, float] = field(default_factory=dict)
    tolerance_sweep: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainedModelBundle:
    version: int
    created_at: str
    match_tolerance: float
    recommended_tolerance: float
    num_jitters: int
    encodings: List[np.ndarray]
    names: List[str]
    source_paths: List[str]
    class_counts: Dict[str, int]
    unique_names: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "match_tolerance": self.match_tolerance,
            "recommended_tolerance": self.recommended_tolerance,
            "num_jitters": self.num_jitters,
            "encodings": self.encodings,
            "names": self.names,
            "source_paths": self.source_paths,
            "class_counts": self.class_counts,
            "unique_names": self.unique_names,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------


def discover_person_folders(dataset_dir: Path) -> List[Tuple[str, Path]]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}\n"
            f"Create: {dataset_dir / 'PersonName'}/photo.jpg"
        )

    people = [
        (entry.name, entry)
        for entry in sorted(dataset_dir.iterdir())
        if entry.is_dir() and not entry.name.startswith(".")
    ]
    if not people:
        raise ValueError(
            f"No person folders in {dataset_dir}. "
            "Add one subfolder per classmate (label = folder name)."
        )
    return people


def iter_image_files(person_dir: Path) -> List[Path]:
    return [
        path
        for path in sorted(person_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def load_rgb_image(image_path: Path) -> Optional[np.ndarray]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def validate_dataset(
    dataset_dir: Path,
    min_photos_per_person: int = 1,
) -> DatasetReport:
    """Check folder structure before training."""
    report = DatasetReport()
    people = discover_person_folders(dataset_dir)
    report.person_folders = len(people)

    for label, folder in people:
        images = iter_image_files(folder)
        report.per_person_images[label] = len(images)
        report.total_images += len(images)

        if len(images) == 0:
            report.errors.append(f"[{label}] folder has no images.")
        elif len(images) < min_photos_per_person:
            report.warnings.append(
                f"[{label}] only {len(images)} image(s); "
                f"recommend at least {min_photos_per_person} for reliable matching."
            )

    if report.total_images == 0:
        report.errors.append("No images found under dataset/.")

    return report


# ---------------------------------------------------------------------------
# Encoding (largest face = best for group photos)
# ---------------------------------------------------------------------------


def encode_largest_face_from_path(
    image_path: Path,
    use_cnn: bool,
    num_jitters: int = NUM_JITTERS,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Encode the largest detected face. Returns (encoding, status).
    status: ok | no_face | invalid | multi_face
    """
    rgb = load_rgb_image(image_path)
    if rgb is None:
        return None, "invalid"

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


# ---------------------------------------------------------------------------
# Matching & evaluation (used at train time and inference)
# ---------------------------------------------------------------------------


def _group_encodings_by_name(
    encodings: List[np.ndarray],
    names: List[str],
) -> Dict[str, List[np.ndarray]]:
    grouped: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
    for encoding, name in zip(encodings, names):
        grouped[name].append(np.asarray(encoding, dtype=np.float64))
    return dict(grouped)


def match_best_person(
    query_encoding: np.ndarray,
    known_encodings: List[np.ndarray],
    known_names: List[str],
) -> Tuple[str, float]:
    """Return (best_label, distance). Uses min distance per person."""
    if not known_encodings:
        return "Unknown", 1.0

    grouped = _group_encodings_by_name(known_encodings, known_names)
    best_name = "Unknown"
    best_distance = float("inf")

    for name, person_encodings in grouped.items():
        distances = face_recognition.face_distance(person_encodings, query_encoding)
        person_best = float(np.min(distances))
        if person_best < best_distance:
            best_distance = person_best
            best_name = name

    return best_name, best_distance


def distance_to_confidence(distance: float, tolerance: float) -> float:
    """Map face distance to confidence in [0, 1] (higher = more confident)."""
    if distance > tolerance:
        return max(0.0, 1.0 - distance)
    return max(0.0, min(1.0, 1.0 - (distance / tolerance)))


def predict_with_confidence(
    query_encoding: np.ndarray,
    model: Dict[str, Any],
    tolerance: Optional[float] = None,
) -> Tuple[str, float, float]:
    """
    Predict label from a trained_model.pkl dict.

    Returns (name, distance, confidence). name is 'Unknown' if distance > tolerance.
    """
    encodings = model.get("encodings") or []
    names = model.get("names") or []
    tol = tolerance if tolerance is not None else float(
        model.get("recommended_tolerance", model.get("match_tolerance", DEFAULT_TOLERANCE))
    )

    label, distance = match_best_person(query_encoding, encodings, names)
    confidence = distance_to_confidence(distance, tol)
    if distance > tol:
        return "Unknown", distance, confidence
    return label, distance, confidence


def leave_one_out_evaluation(
    encodings: List[np.ndarray],
    names: List[str],
    tolerance: float,
) -> EvaluationReport:
    """Hold out one sample per person and test against the rest."""
    report = EvaluationReport(recommended_tolerance=tolerance)
    grouped_indices: DefaultDict[str, List[int]] = defaultdict(list)
    for index, name in enumerate(names):
        grouped_indices[name].append(index)

    per_person_correct: DefaultDict[str, int] = defaultdict(int)
    per_person_total: DefaultDict[str, int] = defaultdict(int)

    for name, indices in grouped_indices.items():
        if len(indices) < 2:
            continue
        for hold_index in indices:
            query = encodings[hold_index]
            gallery_enc = [encodings[i] for i in range(len(encodings)) if i != hold_index]
            gallery_names = [names[i] for i in range(len(names)) if i != hold_index]
            predicted, distance = match_best_person(query, gallery_enc, gallery_names)
            report.tested_samples += 1
            per_person_total[name] += 1
            if predicted == name and distance <= tolerance:
                report.correct += 1
                per_person_correct[name] += 1

    if report.tested_samples:
        report.accuracy = report.correct / report.tested_samples
        for name in per_person_total:
            total = per_person_total[name]
            report.per_person_accuracy[name] = per_person_correct[name] / total

    return report


def sweep_tolerance(
    encodings: List[np.ndarray],
    names: List[str],
) -> Tuple[float, Dict[str, float]]:
    """Pick tolerance with best leave-one-out accuracy."""
    best_tolerance = DEFAULT_TOLERANCE
    best_accuracy = -1.0
    sweep: Dict[str, float] = {}

    for tolerance in TOLERANCE_CANDIDATES:
        result = leave_one_out_evaluation(encodings, names, tolerance)
        sweep[f"{tolerance:.2f}"] = round(result.accuracy, 4)
        if result.accuracy > best_accuracy:
            best_accuracy = result.accuracy
            best_tolerance = tolerance

    return best_tolerance, sweep


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def train_from_dataset(
    dataset_dir: Path,
    use_cnn: bool = True,
    match_tolerance: float = DEFAULT_TOLERANCE,
    min_photos_per_person: int = 1,
    run_evaluation: bool = True,
) -> TrainedModelBundle:
    validation = validate_dataset(dataset_dir, min_photos_per_person=min_photos_per_person)
    for warning in validation.warnings:
        logger.warning(warning)
    if not validation.ok:
        raise ValueError("\n".join(validation.errors))

    stats = TrainingStats()
    encodings: List[np.ndarray] = []
    names: List[str] = []
    source_paths: List[str] = []

    people = discover_person_folders(dataset_dir)
    stats.people = len(people)
    logger.info("Training on %d person folder(s)", stats.people)

    for person_label, person_dir in people:
        image_files = iter_image_files(person_dir)
        if not image_files:
            continue

        person_count = 0
        logger.info("  [%s] %d image(s)", person_label, len(image_files))

        for image_path in image_files:
            stats.images_scanned += 1
            try:
                encoding, status = encode_largest_face_from_path(image_path, use_cnn=use_cnn)
            except Exception as exc:
                stats.skipped_errors += 1
                logger.warning("    SKIP %s — %s", image_path.name, exc)
                continue

            if status == "invalid" or encoding is None:
                if status == "invalid":
                    stats.skipped_invalid += 1
                    logger.warning("    SKIP %s — invalid image", image_path.name)
                else:
                    stats.skipped_no_face += 1
                    logger.warning("    SKIP %s — no face detected", image_path.name)
                continue

            if status == "multi_face":
                stats.skipped_multi_face += 1
                logger.info("    WARN %s — multiple faces; used largest", image_path.name)

            encodings.append(encoding)
            names.append(person_label)
            source_paths.append(str(image_path.relative_to(dataset_dir)))
            person_count += 1
            stats.faces_trained += 1
            logger.info("    OK   %s", image_path.name)

        stats.per_person[person_label] = person_count

    if not encodings:
        raise RuntimeError("No faces encoded. Use clear frontal photos in dataset/<name>/.")

    recommended_tolerance = match_tolerance
    evaluation_dict: Dict[str, Any] = {}
    if run_evaluation and stats.faces_trained >= 2:
        recommended_tolerance, tolerance_sweep = sweep_tolerance(encodings, names)
        eval_at_match = leave_one_out_evaluation(encodings, names, match_tolerance)
        eval_at_recommended = leave_one_out_evaluation(encodings, names, recommended_tolerance)
        evaluation_dict = {
            "leave_one_out_at_match_tolerance": {
                "tolerance": match_tolerance,
                "accuracy": round(eval_at_match.accuracy, 4),
                "tested_samples": eval_at_match.tested_samples,
                "correct": eval_at_match.correct,
                "per_person_accuracy": eval_at_match.per_person_accuracy,
            },
            "leave_one_out_at_recommended_tolerance": {
                "tolerance": recommended_tolerance,
                "accuracy": round(eval_at_recommended.accuracy, 4),
                "tested_samples": eval_at_recommended.tested_samples,
                "correct": eval_at_recommended.correct,
                "per_person_accuracy": eval_at_recommended.per_person_accuracy,
            },
            "tolerance_sweep": tolerance_sweep,
        }
        logger.info(
            "Evaluation: LOO accuracy %.1f%% @ tolerance %.2f | recommended tolerance %.2f (%.1f%%)",
            eval_at_recommended.accuracy * 100,
            recommended_tolerance,
            recommended_tolerance,
            eval_at_recommended.accuracy * 100,
        )

    unique_names = sorted(set(names))
    bundle = TrainedModelBundle(
        version=MODEL_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        match_tolerance=match_tolerance,
        recommended_tolerance=recommended_tolerance,
        num_jitters=NUM_JITTERS,
        encodings=encodings,
        names=names,
        source_paths=source_paths,
        class_counts=dict(stats.per_person),
        unique_names=unique_names,
        metadata={
            "dataset_dir": str(dataset_dir.resolve()),
            "encoder": "face_recognition",
            "detector": "cnn" if use_cnn else "hog",
            "stats": {
                "people": stats.people,
                "images_scanned": stats.images_scanned,
                "faces_trained": stats.faces_trained,
                "skipped_no_face": stats.skipped_no_face,
                "skipped_multi_face": stats.skipped_multi_face,
                "skipped_invalid": stats.skipped_invalid,
                "skipped_errors": stats.skipped_errors,
                "per_person": stats.per_person,
            },
            "evaluation": evaluation_dict,
        },
    )
    _log_training_summary(stats, bundle)
    return bundle


def save_trained_model(bundle: TrainedModelBundle, output_path: Path = MODEL_PATH) -> None:
    with output_path.open("wb") as handle:
        pickle.dump(bundle.to_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved model → %s", output_path.resolve())


def load_trained_model(model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}. Run: python train_model.py")
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def save_training_report(bundle: TrainedModelBundle, report_path: Path) -> None:
    payload = {
        "created_at": bundle.created_at,
        "people": len(bundle.unique_names),
        "total_encodings": len(bundle.encodings),
        "class_counts": bundle.class_counts,
        "match_tolerance": bundle.match_tolerance,
        "recommended_tolerance": bundle.recommended_tolerance,
        "metadata": bundle.metadata,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Report saved → %s", report_path.resolve())


def _log_training_summary(stats: TrainingStats, bundle: TrainedModelBundle) -> None:
    logger.info("-" * 56)
    logger.info("Training summary")
    logger.info("  People           : %d", len(bundle.unique_names))
    logger.info("  Images scanned   : %d", stats.images_scanned)
    logger.info("  Faces trained    : %d", stats.faces_trained)
    logger.info("  Multi-face picks : %d", stats.skipped_multi_face)
    logger.info("  No face          : %d", stats.skipped_no_face)
    logger.info("  Invalid          : %d", stats.skipped_invalid)
    logger.info("  Errors           : %d", stats.skipped_errors)
    logger.info("  Tolerance        : %.2f (recommended: %.2f)", bundle.match_tolerance, bundle.recommended_tolerance)
    for name in bundle.unique_names:
        logger.info("    %s: %d encoding(s)", name, bundle.class_counts.get(name, 0))
    logger.info("-" * 56)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train supervised face encodings from dataset/<person>/ images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Tips:\n"
            "  • 3–8 clear photos per classmate\n"
            "  • Frontal face, good lighting\n"
            "  • Folder name = display name on camera\n"
        ),
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--report", type=Path, default=None, help="Write JSON training report")
    parser.add_argument("--hog-only", action="store_true", help="Faster HOG detector (CPU)")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--min-photos",
        type=int,
        default=1,
        help="Minimum images per person folder (warn if below)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate dataset/ structure and exit without training",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip leave-one-out evaluation (faster on tiny datasets)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cnn = not args.hog_only

    logger.info("Supervised training pipeline")
    logger.info("  Dataset : %s", args.dataset.resolve())
    logger.info("  Output  : %s", args.output.resolve())
    logger.info("  Detector: %s", "CNN" if use_cnn else "HOG")

    try:
        if args.check_only:
            report = validate_dataset(args.dataset, min_photos_per_person=args.min_photos)
            print(f"\nPerson folders : {report.person_folders}")
            print(f"Total images   : {report.total_images}")
            for name, count in sorted(report.per_person_images.items()):
                print(f"  {name}: {count} image(s)")
            for warning in report.warnings:
                print(f"WARNING: {warning}")
            for error in report.errors:
                print(f"ERROR: {error}")
            sys.exit(0 if report.ok else 1)

        bundle = train_from_dataset(
            dataset_dir=args.dataset,
            use_cnn=use_cnn,
            match_tolerance=args.tolerance,
            min_photos_per_person=args.min_photos,
            run_evaluation=not args.no_eval,
        )
        save_trained_model(bundle, args.output)

        report_path = args.report or DEFAULT_REPORT_PATH
        save_training_report(bundle, report_path)

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    print(f"\nTotal trained faces : {len(bundle.encodings)}")
    print(f"People in model     : {len(bundle.unique_names)}")
    print(f"Recommended tolerance: {bundle.recommended_tolerance:.2f}")
    print(f"Model saved to      : {args.output.resolve()}")
    eval_meta = bundle.metadata.get("evaluation") or {}
    rec = eval_meta.get("leave_one_out_at_recommended_tolerance") or {}
    if rec:
        print(f"LOO accuracy        : {rec.get('accuracy', 0) * 100:.1f}%")
    print("\nNext: python train_model.py after partner adds dataset/, then share trained_model.pkl")


if __name__ == "__main__":
    main()
