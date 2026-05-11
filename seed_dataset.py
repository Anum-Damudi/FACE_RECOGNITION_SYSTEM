import argparse
import random
from collections import defaultdict
from typing import Dict, List

from sklearn.datasets import fetch_lfw_people

from db import clear_all_users, init_db, insert_user, user_exists
from face_engine import encode_single_face_rgb

OCCUPATIONS: List[str] = [
    "Engineer",
    "Teacher",
    "Doctor",
    "Security Analyst",
    "Researcher",
    "Manager",
    "Designer",
]

STATUSES = ["Active", "Verified", "Cleared", "Monitored"]


def build_demo_database(
    max_identities: int = 18,
    samples_per_identity: int = 5,
    fresh: bool = False,
    use_cnn: bool = True,
) -> None:
    print("Downloading/Loading LFW dataset...")
    lfw = fetch_lfw_people(
        min_faces_per_person=max(15, samples_per_identity),
        resize=1.0,
        color=True,
    )
    print(f"Dataset loaded with {len(lfw.images)} images.")

    by_label: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(lfw.images)):
        by_label[int(lfw.target[i])].append(i)

    labels_by_count = sorted(
        by_label.keys(),
        key=lambda lb: len(by_label[lb]),
        reverse=True,
    )
    chosen = [
        lb
        for lb in labels_by_count
        if len(by_label[lb]) >= samples_per_identity
    ][:max_identities]

    if not chosen:
        raise RuntimeError("No LFW identities with enough images; lower samples_per_identity.")

    init_db()
    if fresh:
        clear_all_users()
        print("Cleared existing gallery rows (--fresh).")

    rng = random.Random(42)
    added = 0
    for lbl in chosen:
        name = str(lfw.target_names[lbl])
        slug = name.replace(" ", "_").lower()
        image_indices = by_label[lbl][:samples_per_identity]

        age = rng.randint(22, 58)
        occupation = rng.choice(OCCUPATIONS)
        status = rng.choice(STATUSES)

        for j, i in enumerate(image_indices):
            dataset_id = f"lfw_l{lbl}_i{j}_{slug}"
            if user_exists(dataset_id):
                continue

            rgb_img = lfw.images[i].astype("uint8")
            encoding = encode_single_face_rgb(rgb_img, use_cnn=use_cnn)
            if encoding is None:
                print(f"Skipped LFW index {i} ({name}) — no encoding")
                continue

            insert_user(
                dataset_id=dataset_id,
                name=name,
                age=age,
                occupation=occupation,
                status=status,
                face_encoding=encoding.tolist(),
            )
            added += 1
            print(f"Added sample {j + 1}/{len(image_indices)} for {name} (row {added})")

    print(
        f"Demo database ready: {len(chosen)} identities × up to {samples_per_identity} encodings, "
        f"{added} new rows total."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed LFW gallery with multi-sample identities.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Keep existing DB rows. Default: clear gallery first so old encodings are not mixed in.",
    )
    parser.add_argument(
        "--hog-only",
        action="store_true",
        help="Use HOG only (faster seeding; slightly weaker alignment with CNN live mode).",
    )
    parser.add_argument("--identities", type=int, default=18, help="Max number of people in gallery")
    parser.add_argument("--per-person", type=int, default=5, help="Face encodings per person")
    args = parser.parse_args()
    build_demo_database(
        max_identities=args.identities,
        samples_per_identity=args.per_person,
        fresh=not args.append,
        use_cnn=not args.hog_only,
    )
