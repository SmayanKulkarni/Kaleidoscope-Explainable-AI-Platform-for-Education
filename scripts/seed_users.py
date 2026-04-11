"""Seed one student + one instructor account into the running backend."""
import pickle
import requests
from pathlib import Path

BASE = "http://localhost:8000"

# Pick a real learner_id from the training set
ids = pickle.load(open(Path("data/train.pkl"), "rb"))["learner_ids"]
learner_id = str(ids[0])
print(f"Using learner_id: {learner_id}")

USERS = [
    {
        "username":   "student1",
        "email":      "student1@example.com",
        "full_name":  "Test Student",
        "password":   "password123",
        "role":       "student",
        "learner_id": learner_id,
        "course_id":  "AAA_2013J",
    },
    {
        "username":  "instructor1",
        "email":     "instructor1@example.com",
        "full_name": "Test Instructor",
        "password":  "password123",
        "role":      "instructor",
        "department":"Computer Science",
    },
]

for u in USERS:
    r = requests.post(f"{BASE}/auth/register", json=u)
    if r.status_code == 201:
        print(f"  Created {u['username']}  ({u['role']})")
    elif r.status_code == 409:
        print(f"  {u['username']} already exists — skipping")
    else:
        print(f"  ERROR {r.status_code}: {r.text}")

# Enroll student1 under instructor1 so the instructor dashboard has students to show
print("\nEnrolling student1 under instructor1 ...")
token_r = requests.post(f"{BASE}/auth/login", json={"username": "instructor1", "password": "password123"})
if token_r.status_code == 200:
    token = token_r.json()["access_token"]
    enroll_r = requests.post(
        f"{BASE}/auth/enroll",
        params={"learner_id": learner_id, "course_id": "AAA_2013J"},
        headers={"Authorization": f"Bearer {token}"},
    )
    if enroll_r.status_code == 200:
        print(f"  Enrolled {learner_id} under instructor1 (AAA_2013J)")
    else:
        print(f"  Enroll ERROR {enroll_r.status_code}: {enroll_r.text}")
else:
    print(f"  Could not login as instructor1: {token_r.text}")

print("\nTest credentials:")
print("  Student   — username: student1   password: password123")
print("  Instructor — username: instructor1  password: password123")
