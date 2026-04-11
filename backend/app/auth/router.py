"""
Auth Router — /auth/* endpoints
================================
POST /auth/register    — create student or instructor account
POST /auth/login       — returns JWT Bearer token
GET  /auth/me          — current user profile (role-dependent view)
GET  /auth/students    — instructor-only: list enrolled students
PUT  /auth/me/week     — student: update current_week_in_course
DELETE /auth/me        — soft-delete own account
"""

from __future__ import annotations

import logging
from datetime import timezone, datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.auth.auth import (
    create_access_token,
    get_current_instructor,
    get_current_user,
    get_current_active_student,
    hash_password,
    verify_password,
    EXPIRE_MINUTES,
)
from backend.app.auth.database import get_db, init_db
from backend.app.auth.models import (
    CourseEnrollment,
    InstructorProfile,
    LearnerProfile,
    User,
)
from backend.app.auth.schemas import (
    EnrolledStudent,
    LoginRequest,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
    UserMeResponse,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


# ──────────────────────────────────────────────────────────────────────────────
# Register
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """
    Create a new user account.

    - **student**: must provide `learner_id` (the XAI system ID) + optionally `course_id`
    - **instructor**: must provide `department`
    - **admin**: no profile sub-table created
    """
    # Duplicate checks
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status.HTTP_409_CONFLICT, f"Username '{req.username}' already taken")
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status.HTTP_409_CONFLICT, f"Email '{req.email}' already registered")

    if req.role == "student":
        if not req.learner_id:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "students must provide learner_id")
        if db.query(LearnerProfile).filter(LearnerProfile.learner_id == req.learner_id).first():
            raise HTTPException(status.HTTP_409_CONFLICT,
                                f"learner_id '{req.learner_id}' already registered")

    user = User(
        username        = req.username,
        email           = req.email,
        full_name       = req.full_name,
        hashed_password = hash_password(req.password),
        role            = req.role,
    )
    db.add(user)
    db.flush()  # get user.id without committing

    learner_id_out: Optional[str] = None

    if req.role == "student":
        lp = LearnerProfile(
            user_id             = user.id,
            learner_id          = req.learner_id,
            course_id           = req.course_id,
            module_presentation = req.module_presentation,
            prior_completions   = req.prior_completions or 0,
        )
        db.add(lp)
        learner_id_out = req.learner_id

    elif req.role == "instructor":
        ip = InstructorProfile(
            user_id    = user.id,
            department = req.department,
        )
        db.add(ip)

    db.commit()
    db.refresh(user)
    log.info("Registered %s  role=%s  learner_id=%s", user.username, user.role, learner_id_out)

    return RegisterResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        learner_id=learner_id_out,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Login
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate and return a JWT Bearer token.
    The token encodes: sub (user_id), role, learner_id (if student).
    """
    user = db.query(User).filter(User.username == req.username, User.is_active == True).first()
    if user is None or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Pull learner_id into token for student-facing self-service
    learner_id: Optional[str] = None
    if user.learner_profile:
        learner_id = user.learner_profile.learner_id

    extra = {}
    if learner_id:
        extra["learner_id"] = learner_id

    token = create_access_token(subject=user.id, role=user.role, extra=extra)

    # Update last_login_at
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()

    log.info("Login  username=%s  role=%s", user.username, user.role)
    return TokenResponse(
        access_token = token,
        token_type   = "bearer",
        expires_in   = EXPIRE_MINUTES * 60,
        role         = user.role,
        user_id      = user.id,
        learner_id   = learner_id,
    )


# ──────────────────────────────────────────────────────────────────────────────
# /auth/me — current user profile
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserMeResponse)
def me(current_user: User = Depends(get_current_user)):
    """Return the full profile for the authenticated user."""
    return current_user


# ──────────────────────────────────────────────────────────────────────────────
# Instructor-only: list enrolled students
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/students", response_model=list[EnrolledStudent])
def list_students(
    course_id: Optional[str] = None,
    current_user: User = Depends(get_current_instructor),
    db: Session = Depends(get_db),
):
    """
    Instructor view: list all students enrolled in their courses.
    Optionally filter by course_id.
    """
    if current_user.instructor_profile is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Instructor profile not found")

    instructor_profile_id = current_user.instructor_profile.id

    query = (
        db.query(LearnerProfile, User, CourseEnrollment)
        .join(CourseEnrollment, CourseEnrollment.learner_profile_id == LearnerProfile.id)
        .join(User, User.id == LearnerProfile.user_id)
        .filter(CourseEnrollment.instructor_profile_id == instructor_profile_id)
    )
    if course_id:
        query = query.filter(CourseEnrollment.course_id == course_id)

    rows = query.all()
    result = []
    for lp, user, enrollment in rows:
        result.append(EnrolledStudent(
            learner_id   = lp.learner_id,
            full_name    = user.full_name,
            username     = user.username,
            course_id    = enrollment.course_id,
            current_week = lp.current_week,
            enrolled_at  = enrollment.enrolled_at,
        ))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Student: update current week (called after weekly sync)
# ──────────────────────────────────────────────────────────────────────────────

@router.put("/me/week")
def update_week(
    week: int,
    current_user: User = Depends(get_current_active_student),
    db: Session = Depends(get_db),
):
    """Student endpoint: update current_week_in_course after a new week starts."""
    if week < 1 or week > 52:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "week must be 1-52")
    lp = current_user.learner_profile
    if lp is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No learner profile found")
    lp.current_week = week
    lp.last_activity_at = datetime.now(timezone.utc)
    db.commit()
    return {"learner_id": lp.learner_id, "current_week": week, "updated": True}


# ──────────────────────────────────────────────────────────────────────────────
# Self-delete (soft deactivation)
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/enroll")
def enroll_student(
    learner_id: str,
    course_id: str,
    current_user: User = Depends(get_current_instructor),
    db: Session = Depends(get_db),
):
    """
    Instructor endpoint: enroll a student into the instructor's course.
    Creates a CourseEnrollment linking the student's LearnerProfile to
    this instructor's InstructorProfile.
    """
    if current_user.instructor_profile is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Instructor profile not found")

    lp = db.query(LearnerProfile).filter(LearnerProfile.learner_id == learner_id).first()
    if lp is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Learner '{learner_id}' not found")

    existing = (
        db.query(CourseEnrollment)
        .filter(
            CourseEnrollment.learner_profile_id == lp.id,
            CourseEnrollment.instructor_profile_id == current_user.instructor_profile.id,
            CourseEnrollment.course_id == course_id,
        )
        .first()
    )
    if existing:
        return {"message": "Already enrolled", "enrollment_id": existing.id}

    enrollment = CourseEnrollment(
        learner_profile_id    = lp.id,
        instructor_profile_id = current_user.instructor_profile.id,
        course_id             = course_id,
    )
    db.add(enrollment)
    db.commit()
    db.refresh(enrollment)
    log.info("Enrolled learner=%s course=%s instructor=%s", learner_id, course_id, current_user.username)
    return {
        "message":       "Student enrolled",
        "enrollment_id": enrollment.id,
        "learner_id":    learner_id,
        "course_id":     course_id,
    }


@router.delete("/me", status_code=status.HTTP_200_OK)
def deactivate_account(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Soft-delete: sets is_active=False. Does not delete data."""
    current_user.is_active = False
    db.commit()
    return {"message": "Account deactivated", "user_id": current_user.id}
