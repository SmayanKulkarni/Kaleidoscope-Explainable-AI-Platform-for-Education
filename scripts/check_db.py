import sqlite3
conn = sqlite3.connect("data/auth.db")
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", tables)
for (t,) in tables:
    rows = conn.execute(f"SELECT * FROM {t} LIMIT 5").fetchall()
    cols = [d[0] for d in conn.execute(f"PRAGMA table_info({t})").fetchall()]
    print(f"\n{t} columns: {cols}")
    for r in rows:
        print(" ", r)
