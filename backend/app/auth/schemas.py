"""
Auth Pydantic Schemas — request/response models for auth endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ──────────────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username:  str       = Field(min_length=3, max_length=64)
    email:     EmailStr
    password:  str       = Field(min_length=8, max_length=128)
    full_name: Optional[str] = None
    role:      str       = Field(default="student", pattern="^(student|instructor|admin)$")

    learner_id:          Optional[str] = None
    course_id:           Optional[str] = None
    module_presentation: Optional[str] = None
    prior_completions:   Optional[int] = Field(default=0, ge=0)
    department:          Optional[str] = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"student", "instructor", "admin"}
        if v not in allowed:
            raise ValueError(f"role must be one of {allowed}")
        return v


class RegisterResponse(BaseModel):
    id:         str
    username:   str
    email:      str
    role:       str
    learner_id: Optional[str] = None
    message:    str = "Registration successful"


# ──────────────────────────────────────────────────────────────────────────────
# Login
# ──────────────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int
    role:         str
    user_id:      str
    learner_id:   Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# User profile
# ──────────────────────────────────────────────────────────────────────────────

class LearnerProfileSchema(BaseModel):
    learner_id:          str
    course_id:           Optional[str]
    module_presentation: Optional[str]
    current_week:        int
    prior_completions:   int
    enrolled_at:         datetime

    class Config:
        from_attributes = True


class InstructorProfileSchema(BaseModel):
    department: Optional[str]
    bio:        Optional[str]

    class Config:
        from_attributes = True


class UserMeResponse(BaseModel):
    id:               str
    username:         str
    email:            str
    full_name:        Optional[str]
    role:             str
    is_active:        bool
    created_at:       datetime
    last_login_at:    Optional[datetime]
    learner_profile:  Optional[LearnerProfileSchema]
    instructor_profile: Optional[InstructorProfileSchema]

    class Config:
        from_attributes = True


# ──────────────────────────────────────────────────────────────────────────────
# Instructor: list of enrolled students
# ──────────────────────────────────────────────────────────────────────────────

class EnrolledStudent(BaseModel):
    learner_id:   str
    full_name:    Optional[str]
    username:     str
    course_id:    Optional[str]
    current_week: int
    enrolled_at:  datetime

    class Config:
        from_attributes = True
