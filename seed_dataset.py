import random
from typing import List


from sklearn.datasets import fetch_lfw_people

from db import init_db, insert_user, user_exists


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


def build_demo_database(max_people: int = 35) -> None:
    print("Downloading/Loading LFW dataset...")
    lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.6, color=True)

    init_db()

    added = 0
    for i in range(len(lfw.images)):
        target_idx = int(lfw.target[i])
        name = lfw.target_names[target_idx]
        dataset_id = f"lfw_{i}_{name.replace(' ', '_').lower()}"

        if user_exists(dataset_id):
            continue

        rgb_img = lfw.images[i].astype("uint8")
        encodings = face_recognition.face_encodings(rgb_img)
        if not encodings:
            continue

        age = random.randint(22, 58)
        occupation = random.choice(OCCUPATIONS)
        status = random.choice(STATUSES)

        insert_user(
            dataset_id=dataset_id,
            name=name,
            age=age,
            occupation=occupation,
            status=status,
            face_encoding=encodings[0].tolist(),
        )
        added += 1
        if added >= max_people:
            break

    print(f"Demo database ready. Added {added} records.")


if __name__ == "__main__":
    build_demo_database()
