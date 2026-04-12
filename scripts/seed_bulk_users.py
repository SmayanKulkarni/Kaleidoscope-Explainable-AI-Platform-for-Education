"""
Seed 50 students + 10 instructors directly into the DB.

- Generates diverse risk profiles (low / medium / high)
- Creates User + StudentSnapshot for each student
- Creates User + InstructorProfile for each instructor
- Enrolls every student under demo_instructor AND round-robins across new instructors
- Idempotent: skips existing rows

Usage:
    python scripts/seed_bulk_users.py
"""
import random
import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from backend.app.auth.database import _SessionFactory, engine
from backend.app.auth.models import (
    Base, User, LearnerProfile, InstructorProfile, StudentSnapshot, CourseEnrollment
)
from backend.app.auth.auth import hash_password

Base.metadata.create_all(bind=engine)

random.seed(42)

MODULES        = ["AAA", "BBB", "CCC", "DDD", "EEE"]
PRESENTATIONS  = ["2013J", "2014J", "2014B", "2013B"]
TRAJECTORIES   = ["improving", "stable", "declining", "at_risk"]
DEPARTMENTS    = [
    "Computer Science", "Mathematics", "Physics", "Engineering",
    "Data Science", "Statistics", "Education Technology",
    "Cognitive Science", "Information Systems", "Humanities",
]

# ── instructor data ───────────────────────────────────────────────────────────
INSTRUCTOR_NAMES = [
    ("Dr. Sarah Chen",       "sarah.chen"),
    ("Prof. Marcus Williams","marcus.williams"),
    ("Dr. Priya Sharma",     "priya.sharma"),
    ("Prof. James Okafor",   "james.okafor"),
    ("Dr. Lisa Tanaka",      "lisa.tanaka"),
    ("Prof. Ahmed Hassan",   "ahmed.hassan"),
    ("Dr. Maria Gonzalez",   "maria.gonzalez"),
    ("Prof. David Park",     "david.park"),
    ("Dr. Emma Johansson",   "emma.johansson"),
    ("Prof. Raj Patel",      "raj.patel"),
]

# ── student data (50 diverse profiles) ───────────────────────────────────────
STUDENT_NAMES = [
    "Alice Chen", "Bob Martinez", "Carla Singh", "David Okafor", "Elena Kowalski",
    "Felix Andersen", "Grace Liu", "Hiro Tanaka", "Isla MacLeod", "Jake Thompson",
    "Kira Patel", "Liam O'Brien", "Maya Rodriguez", "Nadia Volkov", "Oscar Müller",
    "Priya Nair", "Quinn Davis", "Rosa Kim", "Samuel Adeyemi", "Tara Walsh",
    "Umar Sheikh", "Vera Santos", "Will Johnson", "Xiao Yang", "Yuki Nakamura",
    "Zara Ahmed", "Aaron Brooks", "Bella Torres", "Carlos Mendes", "Diana Evans",
    "Ethan Clark", "Fatima Al-Rashid", "Gus Nielsen", "Hannah Johansson", "Ivan Petrov",
    "Julia Costa", "Kenji Yamamoto", "Laura Smithson", "Mikhail Sokolov", "Nour Hassan",
    "Olivia Brown", "Pedro Alves", "Qian Li", "Rebecca Moore", "Stefan Kovacs",
    "Tina Fernandez", "Ulrich Fischer", "Valentina Cruz", "Wesley Osei", "Yvonne Lefebvre",
]


def _risk_feats(risk_tier: str) -> dict:
    """Generate realistic feature values for a given risk tier."""
    rng = random.Random(random.randint(0, 9999))
    if risk_tier == "low":
        return {
            "login_frequency_weekly":     rng.uniform(5, 14),
            "avg_session_duration_min":   rng.uniform(55, 150),
            "forum_posts_count":          rng.randint(5, 20),
            "video_completion_rate":      rng.uniform(0.8, 1.0),
            "quiz_avg_score":             rng.uniform(72, 98),
            "quiz_completion_rate":       rng.uniform(0.85, 1.0),
            "assignment_submission_rate": rng.uniform(0.88, 1.0),
            "days_since_last_activity":   rng.randint(0, 3),
            "prior_course_completions":   rng.randint(2, 5),
            "current_week_in_course":     rng.randint(6, 12),
            "missed_deadlines_count":     rng.randint(0, 1),
            "help_requests_count":        rng.randint(1, 6),
        }
    elif risk_tier == "medium":
        return {
            "login_frequency_weekly":     rng.uniform(2, 6),
            "avg_session_duration_min":   rng.uniform(25, 70),
            "forum_posts_count":          rng.randint(1, 7),
            "video_completion_rate":      rng.uniform(0.45, 0.79),
            "quiz_avg_score":             rng.uniform(50, 72),
            "quiz_completion_rate":       rng.uniform(0.5, 0.84),
            "assignment_submission_rate": rng.uniform(0.5, 0.87),
            "days_since_last_activity":   rng.randint(3, 10),
            "prior_course_completions":   rng.randint(0, 2),
            "current_week_in_course":     rng.randint(3, 9),
            "missed_deadlines_count":     rng.randint(1, 4),
            "help_requests_count":        rng.randint(0, 3),
        }
    else:  # high
        return {
            "login_frequency_weekly":     rng.uniform(0, 2),
            "avg_session_duration_min":   rng.uniform(5, 30),
            "forum_posts_count":          rng.randint(0, 2),
            "video_completion_rate":      rng.uniform(0.05, 0.44),
            "quiz_avg_score":             rng.uniform(20, 52),
            "quiz_completion_rate":       rng.uniform(0.1, 0.49),
            "assignment_submission_rate": rng.uniform(0.05, 0.49),
            "days_since_last_activity":   rng.randint(8, 30),
            "prior_course_completions":   rng.randint(0, 1),
            "current_week_in_course":     rng.randint(1, 6),
            "missed_deadlines_count":     rng.randint(3, 10),
            "help_requests_count":        rng.randint(0, 1),
        }


def _approx_risk(feats: dict) -> float:
    """Heuristic risk score so snapshots have realistic values before GBM runs."""
    low_eng  = max(0.0, 1.0 - feats["video_completion_rate"])
    low_quiz = max(0.0, 1.0 - feats["quiz_avg_score"] / 100)
    low_sub  = max(0.0, 1.0 - feats["assignment_submission_rate"])
    inactive = min(1.0, feats["days_since_last_activity"] / 30)
    missed   = min(1.0, feats["missed_deadlines_count"] / 10)
    return round(0.25*low_eng + 0.25*low_quiz + 0.2*low_sub + 0.15*inactive + 0.15*missed, 3)


def _trajectory(risk_tier: str) -> str:
    if risk_tier == "low":
        return random.choice(["improving", "improving", "stable"])
    if risk_tier == "medium":
        return random.choice(["stable", "stable", "declining"])
    return random.choice(["declining", "at_risk", "at_risk"])


# ── main seeding logic ────────────────────────────────────────────────────────

def main():
    db = _SessionFactory()
    try:
        print("=" * 60)
        print("Bulk seed: 50 students + 10 instructors")
        print("=" * 60)

        # ── 1. Instructors ────────────────────────────────────────────────────
        print("\n── Instructors ──")
        instructor_ids = []
        for i, (full_name, uname_base) in enumerate(INSTRUCTOR_NAMES):
            username = f"instr_{uname_base.replace('.', '_')}"
            email    = f"{uname_base}@learnlens.com"
            existing = db.query(User).filter(User.username == username).first()
            if existing:
                print(f"  ~ Exists  {username}")
                ip = db.query(InstructorProfile).filter(
                    InstructorProfile.user_id == existing.id
                ).first()
                if ip:
                    instructor_ids.append(ip.id)
                continue
            user = User(
                username       = username,
                email          = email,
                full_name      = full_name,
                hashed_password= hash_password("Instructor2026!"),
                role           = "instructor",
                is_active      = True,
            )
            db.add(user)
            db.flush()
            ip = InstructorProfile(
                user_id    = user.id,
                department = DEPARTMENTS[i % len(DEPARTMENTS)],
            )
            db.add(ip)
            db.flush()
            instructor_ids.append(ip.id)
            print(f"  ✓ Created {username}  ({DEPARTMENTS[i % len(DEPARTMENTS)]})")

        db.commit()

        # Grab demo_instructor profile too
        demo_instr_user = db.query(User).filter(User.username == "demo_instructor").first()
        demo_instr_ip   = None
        if demo_instr_user:
            demo_instr_ip = db.query(InstructorProfile).filter(
                InstructorProfile.user_id == demo_instr_user.id
            ).first()

        # ── 2. Students ───────────────────────────────────────────────────────
        print("\n── Students ──")
        # Risk distribution: 16 low, 18 medium, 16 high for 50
        tiers = (["low"] * 16 + ["medium"] * 18 + ["high"] * 16)
        random.shuffle(tiers)

        for idx, (full_name, tier) in enumerate(zip(STUDENT_NAMES, tiers)):
            learner_id = f"learner_s{idx+1:02d}"
            username   = f"stu_{full_name.lower().replace(' ', '_').replace('-', '_')[:18]}"
            email      = f"{username}@learnlens.com"

            existing = db.query(User).filter(User.username == username).first()
            if existing:
                print(f"  ~ Exists  {username}")
                # Ensure snapshot exists
                snap = db.query(StudentSnapshot).filter(
                    StudentSnapshot.learner_id == learner_id
                ).first()
                if not snap:
                    feats = _risk_feats(tier)
                    risk = _approx_risk(feats)
                    snap = StudentSnapshot(
                        learner_id               = learner_id,
                        display_name             = full_name,
                        current_module           = random.choice(MODULES),
                        current_presentation     = random.choice(PRESENTATIONS),
                        current_week_in_course   = int(feats["current_week_in_course"]),
                        login_frequency_weekly   = feats["login_frequency_weekly"],
                        avg_session_duration_min = feats["avg_session_duration_min"],
                        forum_posts_count        = int(feats["forum_posts_count"]),
                        video_completion_rate    = feats["video_completion_rate"],
                        quiz_avg_score           = feats["quiz_avg_score"],
                        quiz_completion_rate     = feats["quiz_completion_rate"],
                        assignment_submission_rate = feats["assignment_submission_rate"],
                        days_since_last_activity   = int(feats["days_since_last_activity"]),
                        prior_course_completions   = int(feats["prior_course_completions"]),
                        missed_deadlines_count     = int(feats["missed_deadlines_count"]),
                        help_requests_count        = int(feats["help_requests_count"]),
                        dropout_risk_score         = risk,
                        risk_trajectory            = _trajectory(tier),
                    )
                    db.add(snap)
                continue

            feats = _risk_feats(tier)
            risk  = _approx_risk(feats)

            user = User(
                username        = username,
                email           = email,
                full_name       = full_name,
                hashed_password = hash_password("Student2026!"),
                role            = "student",
                is_active       = True,
            )
            db.add(user)
            db.flush()

            lp = LearnerProfile(
                user_id    = user.id,
                learner_id = learner_id,
                course_id  = random.choice(MODULES),
            )
            db.add(lp)
            db.flush()

            module = random.choice(MODULES)
            snap = StudentSnapshot(
                learner_id               = learner_id,
                display_name             = full_name,
                current_module           = module,
                current_presentation     = random.choice(PRESENTATIONS),
                current_week_in_course   = int(feats["current_week_in_course"]),
                login_frequency_weekly   = feats["login_frequency_weekly"],
                avg_session_duration_min = feats["avg_session_duration_min"],
                forum_posts_count        = int(feats["forum_posts_count"]),
                video_completion_rate    = feats["video_completion_rate"],
                quiz_avg_score           = feats["quiz_avg_score"],
                quiz_completion_rate     = feats["quiz_completion_rate"],
                assignment_submission_rate = feats["assignment_submission_rate"],
                days_since_last_activity   = int(feats["days_since_last_activity"]),
                prior_course_completions   = int(feats["prior_course_completions"]),
                missed_deadlines_count     = int(feats["missed_deadlines_count"]),
                help_requests_count        = int(feats["help_requests_count"]),
                dropout_risk_score         = risk,
                risk_trajectory            = _trajectory(tier),
            )
            db.add(snap)
            db.flush()

            # Enroll under demo_instructor
            if demo_instr_ip:
                exists = db.query(CourseEnrollment).filter(
                    CourseEnrollment.learner_profile_id   == lp.id,
                    CourseEnrollment.instructor_profile_id == demo_instr_ip.id,
                ).first()
                if not exists:
                    db.add(CourseEnrollment(
                        learner_profile_id    = lp.id,
                        instructor_profile_id = demo_instr_ip.id,
                        course_id             = module,
                    ))

            # Also enroll under one of the new instructors (round-robin)
            if instructor_ids:
                instr_ip_id = instructor_ids[idx % len(instructor_ids)]
                exists2 = db.query(CourseEnrollment).filter(
                    CourseEnrollment.learner_profile_id   == lp.id,
                    CourseEnrollment.instructor_profile_id == instr_ip_id,
                ).first()
                if not exists2:
                    db.add(CourseEnrollment(
                        learner_profile_id    = lp.id,
                        instructor_profile_id = instr_ip_id,
                        course_id             = module,
                    ))

            tier_label = "high" if risk > 0.6 else ("medium" if risk > 0.35 else "low")
            print(f"  ✓ {learner_id:<14} {full_name:<26} {tier_label:<8} risk={risk:.2f}")

        db.commit()

        # ── 3. Summary ────────────────────────────────────────────────────────
        total_students = db.query(StudentSnapshot).count()
        total_instrs   = db.query(InstructorProfile).count()
        total_enroll   = db.query(CourseEnrollment).count()
        print("\n" + "=" * 60)
        print(f"Done.  StudentSnapshots={total_students}  Instructors={total_instrs}  Enrollments={total_enroll}")
        print("Instructor password: Instructor2026!")
        print("Student password:    Student2026!")
        print("=" * 60)

    finally:
        db.close()


if __name__ == "__main__":
    main()
