# XAI Learning Recommendation System

Explainable AI system for student dropout risk prediction with SHAP, DiCE counterfactuals, causal inference, and implicit feedback retraining.

---

## Quick Start (Local Dev)

```bash
# 1. Copy and fill env file
cp .env.example .env

# 2. Start Postgres + API
docker-compose up --build

# 3. API is live at http://localhost:8000
# 4. Swagger docs at http://localhost:8000/docs
```

## Without Docker

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # set JWT_SECRET_KEY at minimum
uvicorn backend.app.main:app --reload
```

---

## Deployment (AWS Free Tier)

See `infra/` for Terraform and `.github/workflows/` for CI/CD.

### Prerequisites
- AWS CLI configured (`aws configure`)
- Terraform >= 1.6 installed
- SSH key pair at `~/.ssh/id_rsa.pub`

### First-time Infrastructure Setup

```bash
# 1. Create Terraform state bucket (one-time bootstrap)
aws s3 mb s3://xai-rec-tf-state --region ap-south-1
aws dynamodb create-table \
  --table-name xai-rec-tf-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region ap-south-1

# 2. Deploy infrastructure
cd infra
terraform init
terraform plan -var="db_password=YOUR_SECURE_PASSWORD"
terraform apply -var="db_password=YOUR_SECURE_PASSWORD"

# 3. Note outputs: ec2_public_ip, ecr_repo_url, s3_bucket_name, database_url
terraform output
```

### GitHub Secrets Required

| Secret | How to get |
|--------|-----------|
| `AWS_ACCESS_KEY_ID` | IAM → Create access key |
| `AWS_SECRET_ACCESS_KEY` | Same |
| `AWS_REGION` | e.g. `ap-south-1` |
| `ECR_REPO_URL` | `terraform output ecr_repo_url` |
| `EC2_HOST` | `terraform output ec2_public_ip` |
| `EC2_SSH_KEY` | Contents of `~/.ssh/id_rsa` (private key) |
| `DATABASE_URL` | `terraform output database_url` |
| `JWT_SECRET_KEY` | `python -c "import secrets; print(secrets.token_hex(32))"` |
| `GROQ_API_KEY` | https://console.groq.com |
| `AWS_S3_BUCKET` | `terraform output s3_bucket_name` |
| `DB_PASSWORD` | Same password used in `terraform.tfvars` (for Terraform workflow) |

### Terraform Workflow

| Workflow | Trigger | Action |
|----------|---------|--------|
| `terraform.yml` | Changes to `infra/**` | Plan on PR (posts diff as comment) → Apply on merge to `main` |

### Database Migrations

```bash
# First time — create all tables
alembic upgrade head

# After a model change — autogenerate a new migration
alembic revision --autogenerate -m "add new column"
alembic upgrade head

# Check current revision
alembic current
```

In the CD workflow, migrations run automatically on deploy (added to `deploy.yml` SSH step).

### Upload Initial Models to S3

```bash
aws s3 cp models/ s3://$(terraform -chdir=infra output -raw s3_bucket_name)/models/ --recursive
```

### CI/CD Workflows

| Workflow | Trigger | Action |
|----------|---------|--------|
| `ci.yml` | Every push / PR | Lint (ruff) + pytest |
| `deploy.yml` | Push to `main` | Build Docker → push ECR → SSH deploy → health check |
| `retrain.yml` | Manual / weekly Sunday 02:00 UTC | `POST /mlops/retrain` → `POST /mlops/reload` |
| `terraform.yml` | Changes to `infra/**` | Plan on PR (posts diff as comment) → Apply on merge to `main` |

---

## Architecture

```
FastAPI (uvicorn)
  ├── /predict          GBM risk score
  ├── /explain          SHAP + DiCE + Anchors + Prototypes + Causal
  ├── /whatif           What-If counterfactual
  ├── /simulate         LSTM temporal sequence
  ├── /events           Implicit feedback ingestion
  ├── /feedback         Explicit feedback (ratings, corrections)
  ├── /mlops/retrain    Trigger retraining pipeline
  ├── /mlops/reload     Hot-reload new model into live process
  └── /auth/*           JWT auth (student / instructor roles)

Databases (PostgreSQL / SQLite fallback)
  auth        Users, roles, learner/instructor profiles
  explanations  SHAP + explanation history per learner
  feedback    Explicit ratings, follow-rate, corrections
  events      Implicit interaction telemetry (clicks, time-on-page)

ML Pipeline
  OULAD 12 features → GBM / RF / LSTM
  Engagement Autoencoder (20 signals → 3 latent scores)
  Retrain: 12 + 3 = 15 features → new GBM/RF
  Model artifacts stored in S3 (versioned)
```

---

## Key Environment Variables

See `.env.example` for the full list. Minimum required for local dev:
- `JWT_SECRET_KEY` — any 32+ char random string