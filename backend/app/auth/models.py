"""
Auth Database Models
====================
SQLAlchemy ORM models for user authentication and profiles.

Tables
------
users               — core auth: id, email, role, hashed_password
learner_profiles    — links a student User to their XAI learner_id + course context
instructor_profiles — instructor department + course assignments
course_enrollments  — many-to-many: which students an instructor oversees
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


# ──────────────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Users — core auth table
# ──────────────────────────────────────────────────────────────────────────────

class User(Base):
    """
    Core user table shared by students and instructors.

    Roles
    -----
    "student"    — can view their own dashboard (learner view)
    "instructor" — can view any enrolled student (instructor view)
    "admin"      — full access including /mlops/* endpoints
    """
    __tablename__ = "users"

    id              = Column(String, primary_key=True, default=_uuid)
    username        = Column(String(64),  unique=True,  nullable=False, index=True)
    email           = Column(String(255), unique=True,  nullable=False, index=True)
    full_name       = Column(String(128), nullable=True)
    hashed_password = Column(String(128), nullable=False)
    role            = Column(String(16),  nullable=False, default="student")
    is_active       = Column(Boolean, default=True, nullable=False)
    created_at      = Column(DateTime(timezone=True), default=_now, nullable=False)
    last_login_at   = Column(DateTime(timezone=True), nullable=True)

    learner_profile    = relationship("LearnerProfile",    back_populates="user", uselist=False, cascade="all, delete-orphan")
    instructor_profile = relationship("InstructorProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User id={self.id!r} username={self.username!r} role={self.role!r}>"


# ──────────────────────────────────────────────────────────────────────────────
# LearnerProfile — student-specific data
# ──────────────────────────────────────────────────────────────────────────────

class LearnerProfile(Base):
    """
    Maps a student User to the XAI system's learner_id and current course context.
    The learner_id is what gets passed to /explain, /history, /predict.
    """
    __tablename__ = "learner_profiles"

    id                  = Column(String, primary_key=True, default=_uuid)
    user_id             = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    learner_id          = Column(String(64), unique=True, nullable=False, index=True)
    course_id           = Column(String(64), nullable=True)
    module_presentation = Column(String(32), nullable=True)
    current_week        = Column(Integer, default=1, nullable=False)
    prior_completions   = Column(Integer, default=0, nullable=False)
    enrolled_at         = Column(DateTime(timezone=True), default=_now, nullable=False)
    last_activity_at    = Column(DateTime(timezone=True), nullable=True)

    user        = relationship("User", back_populates="learner_profile")
    enrollments = relationship("CourseEnrollment", back_populates="learner", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<LearnerProfile learner_id={self.learner_id!r} course={self.course_id!r}>"


# ──────────────────────────────────────────────────────────────────────────────
# InstructorProfile — instructor-specific data
# ──────────────────────────────────────────────────────────────────────────────

class InstructorProfile(Base):
    """Instructor department and course portfolio."""
    __tablename__ = "instructor_profiles"

    id         = Column(String, primary_key=True, default=_uuid)
    user_id    = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    department = Column(String(128), nullable=True)
    bio        = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)

    user        = relationship("User", back_populates="instructor_profile")
    enrollments = relationship("CourseEnrollment", back_populates="instructor", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<InstructorProfile user_id={self.user_id!r} dept={self.department!r}>"


# ──────────────────────────────────────────────────────────────────────────────
# CourseEnrollment — which instructor oversees which student in which course
# ──────────────────────────────────────────────────────────────────────────────

class CourseEnrollment(Base):
    """
    Links a student (LearnerProfile) to an instructor (InstructorProfile)
    within a course. Instructors can only view/explain students enrolled in
    their courses.
    """
    __tablename__ = "course_enrollments"
    __table_args__ = (
        UniqueConstraint("learner_profile_id", "instructor_profile_id", "course_id",
                         name="uq_enrollment"),
    )

    id                   = Column(String, primary_key=True, default=_uuid)
    learner_profile_id   = Column(String, ForeignKey("learner_profiles.id", ondelete="CASCADE"), nullable=False)
    instructor_profile_id = Column(String, ForeignKey("instructor_profiles.id", ondelete="CASCADE"), nullable=False)
    course_id            = Column(String(64), nullable=False, index=True)
    enrolled_at          = Column(DateTime(timezone=True), default=_now, nullable=False)

    learner    = relationship("LearnerProfile",    back_populates="enrollments")
    instructor = relationship("InstructorProfile", back_populates="enrollments")

    def __repr__(self) -> str:
        return (f"<CourseEnrollment course={self.course_id!r} "
                f"learner={self.learner_profile_id!r}>")


# ──────────────────────────────────────────────────────────────────────────────
# StudentSnapshot — cached feature vector + risk score per learner
# ──────────────────────────────────────────────────────────────────────────────

class StudentSnapshot(Base):
    """
    Stores the latest LearnerFeatures vector + GBM risk score for each student.
    Updated whenever a new prediction is made or manually seeded.
    Instructors use this to access enrolled students' current state without
    requiring a live prediction call.
    """
    __tablename__ = "student_snapshots"

    id                          = Column(String, primary_key=True, default=_uuid)
    learner_id                  = Column(String(64), unique=True, nullable=False, index=True)
    display_name                = Column(String(128), nullable=True)

    # Core LearnerFeatures (12 features)
    login_frequency_weekly      = Column(Float, default=3.0)
    avg_session_duration_min    = Column(Float, default=45.0)
    forum_posts_count           = Column(Integer, default=2)
    video_completion_rate       = Column(Float, default=0.6)
    quiz_avg_score              = Column(Float, default=65.0)
    quiz_completion_rate        = Column(Float, default=0.7)
    assignment_submission_rate  = Column(Float, default=0.75)
    days_since_last_activity    = Column(Integer, default=5)
    prior_course_completions    = Column(Integer, default=1)
    current_week_in_course      = Column(Integer, default=6)
    missed_deadlines_count      = Column(Integer, default=2)
    help_requests_count         = Column(Integer, default=1)

    # Computed / model outputs
    dropout_risk_score          = Column(Float, nullable=True)
    risk_trajectory             = Column(String(32), nullable=True)  # improving/stable/deteriorating/at_risk_stable
    current_module              = Column(String(16), nullable=True)
    current_presentation        = Column(String(16), nullable=True)

    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now, nullable=False)

    def to_features_dict(self) -> dict:
        return {
            "login_frequency_weekly":     self.login_frequency_weekly,
            "avg_session_duration_min":   self.avg_session_duration_min,
            "forum_posts_count":          self.forum_posts_count,
            "video_completion_rate":      self.video_completion_rate,
            "quiz_avg_score":             self.quiz_avg_score,
            "quiz_completion_rate":       self.quiz_completion_rate,
            "assignment_submission_rate": self.assignment_submission_rate,
            "days_since_last_activity":   self.days_since_last_activity,
            "prior_course_completions":   self.prior_course_completions,
            "current_week_in_course":     self.current_week_in_course,
            "missed_deadlines_count":     self.missed_deadlines_count,
            "help_requests_count":        self.help_requests_count,
        }

    def __repr__(self) -> str:
        return f"<StudentSnapshot learner_id={self.learner_id!r} risk={self.dropout_risk_score!r}>"
