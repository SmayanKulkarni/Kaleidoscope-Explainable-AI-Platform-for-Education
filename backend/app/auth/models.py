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
    Boolean, Column, DateTime, ForeignKey,
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
