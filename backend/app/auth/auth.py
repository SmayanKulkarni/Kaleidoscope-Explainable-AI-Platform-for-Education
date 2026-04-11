"""
Auth Core — JWT Tokens + Password Hashing + FastAPI Dependencies
================================================================

Environment variables required
-------------------------------
JWT_SECRET_KEY  : secret for HS256 signing (generate with: openssl rand -hex 32)
JWT_ALGORITHM   : defaults to "HS256"
JWT_EXPIRE_MIN  : token expiry in minutes, defaults to 1440 (24h)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from backend.app.auth.models import User

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

SECRET_KEY     = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION_use_openssl_rand_hex_32")
ALGORITHM      = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MIN", "1440"))

if SECRET_KEY == "CHANGE_ME_IN_PRODUCTION_use_openssl_rand_hex_32":
    log.warning("JWT_SECRET_KEY is using the default insecure value. Set JWT_SECRET_KEY env var.")

# ──────────────────────────────────────────────────────────────────────────────
# Password hashing
# ──────────────────────────────────────────────────────────────────────────────

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


# ──────────────────────────────────────────────────────────────────────────────
# JWT helpers
# ──────────────────────────────────────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def create_access_token(
    subject: str,
    role: str,
    extra: Optional[dict] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a signed JWT.

    Parameters
    ----------
    subject : user id (UUID string)
    role    : "student" | "instructor" | "admin"
    extra   : additional claims (learner_id, etc.)
    """
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=EXPIRE_MINUTES)
    )
    payload: dict = {
        "sub":  subject,
        "role": role,
        "exp":  expire,
        "iat":  datetime.now(timezone.utc),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and verify a JWT. Raises HTTPException on failure."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI dependency — DB session
# ──────────────────────────────────────────────────────────────────────────────

from backend.app.auth.database import get_db  # noqa: E402 (circular-safe import)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI dependencies — current user + role guards
# ──────────────────────────────────────────────────────────────────────────────

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Decode JWT and return the corresponding User ORM object."""
    payload = decode_token(token)
    user_id: str = payload.get("sub")
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token missing subject")
    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found or deactivated")
    return user


def get_current_active_student(
    current_user: User = Depends(get_current_user),
) -> User:
    if current_user.role not in ("student", "admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Student access required")
    return current_user


def get_current_instructor(
    current_user: User = Depends(get_current_user),
) -> User:
    if current_user.role not in ("instructor", "admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Instructor access required")
    return current_user


def get_current_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    if current_user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
    return current_user


def get_current_user_optional(
    token: Optional[str] = Depends(OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Like get_current_user but returns None instead of raising for unauthenticated requests."""
    if not token:
        return None
    try:
        return get_current_user(token=token, db=db)
    except HTTPException:
        return None
