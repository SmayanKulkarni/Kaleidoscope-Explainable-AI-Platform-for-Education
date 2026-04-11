#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# EC2 User Data — run once on first boot to install Docker and start the API
# Terraform injects: ecr_repo_url, aws_region, s3_bucket, db_url
# Additional secrets (JWT_SECRET_KEY, GROQ_API_KEY) are injected by CD workflow
# ─────────────────────────────────────────────────────────────────────────────

export ECR_REPO_URL="${ecr_repo_url}"
export AWS_REGION="${aws_region}"
export S3_BUCKET="${s3_bucket}"
export DATABASE_URL="${db_url}"

# ── 1. Install Docker ─────────────────────────────────────────────────────────
dnf update -y
dnf install -y docker
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

# ── 2. Install AWS CLI v2 (for ECR login) ────────────────────────────────────
dnf install -y aws-cli

# ── 3. Write environment file (secrets added by CD deploy step) ──────────────
mkdir -p /opt/xai
cat > /opt/xai/.env <<EOF
DATABASE_URL=$DATABASE_URL
AWS_S3_BUCKET=$S3_BUCKET
AWS_REGION=$AWS_REGION
JWT_SECRET_KEY=REPLACE_IN_CD
GROQ_API_KEY=REPLACE_IN_CD
JWT_ALGORITHM=HS256
JWT_EXPIRE_MIN=1440
EOF
chmod 600 /opt/xai/.env

# ── 4. Create systemd service for the API container ──────────────────────────
cat > /etc/systemd/system/xai-api.service <<UNIT
[Unit]
Description=XAI Learning Recommendation API
After=docker.service
Requires=docker.service

[Service]
Restart=always
ExecStartPre=-/usr/bin/docker stop xai-api
ExecStartPre=-/usr/bin/docker rm xai-api
ExecStartPre=/bin/bash -c "aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPO_URL"
ExecStartPre=/usr/bin/docker pull $ECR_REPO_URL:latest
ExecStart=/usr/bin/docker run --name xai-api --rm \
  --env-file /opt/xai/.env \
  -p 8000:8000 \
  $ECR_REPO_URL:latest
ExecStop=/usr/bin/docker stop xai-api

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable xai-api

# ── 5. First login + pull + start (ECR may not have image yet — OK to fail) ──
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REPO_URL" || true
docker pull "$ECR_REPO_URL:latest" && systemctl start xai-api || true

echo "User data complete"
