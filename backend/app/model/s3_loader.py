"""
S3 Model Artifact Loader (Step 4)
===================================
Downloads model artifacts from S3 to the local models/ directory at startup.
Also provides an upload function called after successful retraining.

Environment variables
---------------------
AWS_S3_BUCKET   : S3 bucket name (required in production)
AWS_REGION      : AWS region, default ap-south-1
AWS_S3_PREFIX   : Key prefix inside bucket, default "models/"

If AWS_S3_BUCKET is not set (local dev), all functions are no-ops.

Public API
----------
download_models(models_dir)  -> list[str]  (files downloaded)
upload_models(models_dir)    -> list[str]  (files uploaded)
model_version_on_s3()        -> str | None
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_BUCKET = os.getenv("AWS_S3_BUCKET", "")
_REGION = os.getenv("AWS_REGION", "ap-south-1")
_PREFIX = os.getenv("AWS_S3_PREFIX", "models/").rstrip("/") + "/"

# Artifacts to sync between S3 and local models/
_MODEL_FILES = [
    "gbm.pkl",
    "rf.pkl",
    "lstm.pt",
    "lstm_config.json",
    "training_summary.json",
    "tuning_summary.json",
    "model_version.json",
]

# Engagement model directory (entire dir synced)
_ENG_DIR = "engagement/"


def _boto3_client():
    """Lazy import boto3 so the app doesn't crash if boto3 is not installed in dev."""
    try:
        import boto3
        return boto3.client("s3", region_name=_REGION)
    except ImportError:
        log.warning("boto3 not installed — S3 operations unavailable")
        return None


def download_models(models_dir: Path) -> list[str]:
    """
    Download model artifacts from S3 to models_dir.

    Skips files that already exist locally (cache-friendly: no re-download
    unless file is missing or you force via delete local).

    Returns list of files actually downloaded.
    """
    if not _BUCKET:
        log.info("AWS_S3_BUCKET not set — skipping S3 model download (local dev mode)")
        return []

    client = _boto3_client()
    if client is None:
        return []

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []

    # Download top-level model files
    for fname in _MODEL_FILES:
        local_path = models_dir / fname
        s3_key     = f"{_PREFIX}{fname}"
        if local_path.exists():
            log.debug("S3 skip (exists): %s", fname)
            continue
        try:
            log.info("S3 download  s3://%s/%s → %s", _BUCKET, s3_key, local_path)
            client.download_file(_BUCKET, s3_key, str(local_path))
            downloaded.append(fname)
        except client.exceptions.ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "404":
                log.warning("S3 key not found: s3://%s/%s (skipping)", _BUCKET, s3_key)
            else:
                log.error("S3 download error for %s: %s", fname, e)

    # Download engagement model directory
    eng_dir = models_dir / "engagement"
    try:
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=_BUCKET, Prefix=f"{_PREFIX}{_ENG_DIR}")
        for page in pages:
            for obj in page.get("Contents", []):
                key      = obj["Key"]
                rel_path = key[len(_PREFIX):]           # strip "models/" prefix
                local    = models_dir / rel_path
                local.parent.mkdir(parents=True, exist_ok=True)
                if not local.exists():
                    log.info("S3 download  s3://%s/%s → %s", _BUCKET, key, local)
                    client.download_file(_BUCKET, key, str(local))
                    downloaded.append(rel_path)
    except Exception as e:
        log.warning("S3 engagement model list/download failed: %s", e)

    log.info("S3 download complete  files=%d", len(downloaded))
    return downloaded


def upload_models(models_dir: Path) -> list[str]:
    """
    Upload model artifacts from models_dir to S3.
    Called after a successful retrain to persist new artifacts.

    Returns list of S3 keys uploaded.
    """
    if not _BUCKET:
        log.info("AWS_S3_BUCKET not set — skipping S3 model upload")
        return []

    client = _boto3_client()
    if client is None:
        return []

    models_dir = Path(models_dir)
    uploaded: list[str] = []

    # Upload top-level files
    for fname in _MODEL_FILES:
        local_path = models_dir / fname
        if not local_path.exists():
            continue
        s3_key = f"{_PREFIX}{fname}"
        try:
            log.info("S3 upload  %s → s3://%s/%s", local_path, _BUCKET, s3_key)
            client.upload_file(str(local_path), _BUCKET, s3_key)
            uploaded.append(s3_key)
        except Exception as e:
            log.error("S3 upload error for %s: %s", fname, e)

    # Upload engagement model directory
    eng_dir = models_dir / "engagement"
    if eng_dir.exists():
        for local_path in eng_dir.rglob("*"):
            if local_path.is_file():
                rel = local_path.relative_to(models_dir)
                s3_key = f"{_PREFIX}{rel.as_posix()}"
                try:
                    log.info("S3 upload  %s → s3://%s/%s", local_path, _BUCKET, s3_key)
                    client.upload_file(str(local_path), _BUCKET, s3_key)
                    uploaded.append(s3_key)
                except Exception as e:
                    log.error("S3 upload error for %s: %s", local_path, e)

    log.info("S3 upload complete  keys=%d", len(uploaded))
    return uploaded


def model_version_on_s3() -> Optional[str]:
    """
    Read the model version string from S3 (model_version.json).
    Returns None if not found or S3 unavailable.
    """
    if not _BUCKET:
        return None
    client = _boto3_client()
    if client is None:
        return None
    try:
        import json
        import io
        obj = client.get_object(Bucket=_BUCKET, Key=f"{_PREFIX}model_version.json")
        data = json.loads(obj["Body"].read())
        return f"gbm-v{data.get('version', '?')}"
    except Exception:
        return None
