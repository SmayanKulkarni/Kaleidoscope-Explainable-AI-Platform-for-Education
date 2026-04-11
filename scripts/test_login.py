import requests
BASE = "http://localhost:8000"
for user, pw in [("student1","password123"), ("instructor1","password123")]:
    r = requests.post(f"{BASE}/auth/login", json={"username": user, "password": pw})
    if r.status_code == 200:
        d = r.json()
        print(f"  OK  {user}  role={d['role']}  learner_id={d.get('learner_id')}")
    else:
        print(f"  FAIL {user}: {r.status_code} {r.text}")
