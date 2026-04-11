"""Register the three demo accounts directly via the auth API (no train.pkl needed)."""
import requests

BASE = "http://localhost:8000"

accounts = [
    {
        "username":  "demo_student",
        "email":     "demo_student@example.com",
        "full_name": "Demo Student",
        "password":  "DemoStudent2026!",
        "role":      "student",
        "learner_id": "11391",
        "course_id":  "AAA_2013J",
    },
    {
        "username":   "demo_instructor",
        "email":      "demo_instructor@example.com",
        "full_name":  "Demo Instructor",
        "password":   "DemoInstructor2026!",
        "role":       "instructor",
        "department": "Computer Science",
    },
    {
        "username":  "demo_admin",
        "email":     "demo_admin@example.com",
        "full_name": "Demo Admin",
        "password":  "DemoAdmin2026!",
        "role":      "admin",
    },
]

for u in accounts:
    r = requests.post(f"{BASE}/auth/register", json=u, timeout=10)
    if r.status_code == 201:
        print(f"  Created  {u['username']}  ({u['role']})")
    elif r.status_code == 409:
        print(f"  Exists   {u['username']}  — skipping")
    else:
        print(f"  ERROR {r.status_code}: {r.text[:300]}")

print("\nDemo credentials:")
print("  Student    username: demo_student     password: DemoStudent2026!")
print("  Instructor username: demo_instructor  password: DemoInstructor2026!")
print("  Admin      username: demo_admin       password: DemoAdmin2026!")
