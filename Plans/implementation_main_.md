# XAI Learning Recommendation System — Master Plan
  
> Merged from `expanded-mlops-system-design` + `aws-mlops-system-design`.  
> Single source of truth covering architecture, gaps, implementation phases, AWS provisioning, and operational runbook.
  
---
  
## 1. Architecture Overview
  
### 1.1 ML Pipeline (Two-Stage)
  
```
Stage 1 — Engagement Encoding
  ImplicitAggregator → 20 signals (15 implicit + 5 explicit)
  → EngagementAutoencoder → 3 latent features
  
Stage 2 — Risk Prediction
  12 OULAD features + 3 latent = 15 features → GBM (primary) / RF / LSTM
```
  
### 1.2 Recommendation Engine
  
```
Student path:   POST /recommend/student
                → LambdaMART student_ranker (LGBMRanker)
                → RankerExplainer (TreeSHAP + Anchors + Interactions + Causal + Stability)
                → FairnessAuditor (demographic parity across gender/age/disability)
  
Instructor path: POST /instructor/recommend/{learner_id}
                → LambdaMART instructor_ranker (34-feature schema)
                → Full RankerExplainer stack
```
  
### 1.3 XAI Stack (Parallel at /explain time)
  
| Explainer | Output |
|-----------|--------|
| **TreeSHAP / DeepSHAP / Captum IG** | Feature attribution |
| **Archipelago** | Feature interaction pairs |
| **Anchors** | IF-THEN rule with precision/coverage |
| **DiCE** | Counterfactuals (immutable feature locks) |
| **Prototypes** | k-NN similar learners |
| **MAPIE** | Conformal prediction uncertainty |
| **CausalAnnotator** | DoWhy DAG — causal vs correlational |
| **TrustScorer** | 0.40×fidelity + 0.35×stability + 0.25×completeness |
| **LLM Narrator** | Groq — narrates pre-computed XAI payloads only |
  
### 1.4 MLOps Feedback Loop
  
```
Users → /events (implicit) + /feedback (explicit) → PostgreSQL
  ↓
RetrainPipeline (weekly cron or manual):
  1. ImplicitAggregator: EventStore → 20 per-learner engagement signals
  2. EngagementAutoencoder: 20 → 3 latent features
  3. Augment: OULAD_12 + latent_3 = 15 features
  4. Retrain GBM + RF, validate: AUC drop ≤ 0.02, Brier rise ≤ 0.03, SHAP fidelity ≤ 0.05
  5. Save to disk → upload to S3
  
RecommendationRetrainPipeline (same trigger):
  1. Same ImplicitAggregator signals
  2. Enrich recommendation training data with learned_implicit_* columns
  3. Retrain student_ranker + instructor_ranker (LambdaMART)
  4. Regenerate precomputed top-K CSVs
  5. Save to disk → upload to S3
  
Hot-Reload → POST /mlops/reload-all:
  → S3 download → atomic swap of ALL 12+ components
  → Canary mode supported (fractional traffic routing)
```
  
### 1.5 Drift Monitoring
  
- **Evidently DriftMonitor**: recent prediction inputs vs training distribution
- **ExplanationDriftDetector**: JSD on |SHAP| softmax distributions
- **ConceptDriftProxy**: calibration gap between risk scores and follow rates
- **Slack webhook** alert when drift_share ≥ 20%
- **Trend tracking**: last 5 drift reports
  
### 1.6 CI/CD Pipelines
  
| Workflow | Trigger | What It Does |
|----------|---------|--------------|
| `ci.yml` | Push/PR to main | Ruff lint + pytest (Postgres service container) + frontend build check + smoke tests |
| `deploy.yml` | Push to main | Docker → ECR → SSH EC2 → Alembic migrate → systemd restart → health check → rollback on failure |
| `retrain.yml` | Weekly Sun 02:00 UTC + manual | POST /mlops/retrain-all → POST /mlops/reload-all → health check |
| `terraform.yml` | Changes to `infra/**` | Plan on PR (post as comment) → Apply on merge |
  
---
  
## 2. Full Data Flow Diagram
  
```
┌────────────────────────────────────────────────────────────────────────┐
│                           USER INTERACTIONS                            │
│                                                                        │
│   Student Portal          Instructor Portal          Admin Portal      │
│   ┌────────────────┐     ┌──────────────────┐     ┌────────────────┐  │
│   │ Dashboard      │     │ Student List      │     │ System Health  │  │
│   │ Explain View   │     │ Per-Student Reco  │     │ MLOps Panel    │  │
│   │ What-If        │     │ What-If           │     │ User Mgmt      │  │
│   │ Action Plan    │     │ Intervention Plan │     │ Enrollments    │  │
│   │ History        │     │ Compare Students  │     │ OULAD Sync     │  │
│   │ Simulate       │     │                   │     │ Drift Report   │  │
│   └───────┬────────┘     └────────┬──────────┘     │ Retrain/Reload │  │
│           │                       │                 │ Fairness Audit │  │
│           │  useEventTracker()    │                 └───────┬────────┘  │
│           └──────────┬────────────┘                        │            │
│                      │                                     │            │
│   ┌──────────────────▼─────────────────────────────────────▼────────┐  │
│   │           eventTracker.js (buffer → batch POST /events)         │  │
│   │   page_view, explain_requested, recommendation_viewed/clicked,  │  │
│   │   whatif_run, action_viewed/taken, history_viewed,              │  │
│   │   compare_run, simulate_run, session_start/end                  │  │
│   └──────────────────────────────┬───────────────────────────────────┘  │
└─────────────────────────────────┼──────────────────────────────────────┘
                                  │ HTTP :8000
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  EC2 t3.micro — Docker Container (xai-api)                               │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                        FastAPI Application                         │  │
│  │                                                                    │  │
│  │  /auth/*        ── JWT (register, login, me, enroll)              │  │
│  │  /predict       ── GBM/RF risk scoring → PredictionLogger         │  │
│  │  /explain       ── Full XAI stack (SHAP, DiCE, Anchors, etc.)    │  │
│  │  /recommend/*   ── Student + Instructor LambdaMART rankers + XAI  │  │
│  │  /instructor/*  ── Instructor-specific endpoints                  │  │
│  │  /events        ── Implicit data collection → EventStore          │  │
│  │  /feedback      ── Explicit feedback → FeedbackStore              │  │
│  │  /admin/*       ── User mgmt, enrollments, OULAD sync             │  │
│  │  /mlops/*       ── Retrain, reload, drift, health                 │  │
│  │  /causal/*      ── DAG visualization + narration                  │  │
│  │  /fairness/*    ── Algorithmic fairness audit                     │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────────────────┐  ┌──────────────────────────────────────────┐  │
│  │ Dropout MLOps Engine │  │ Recommendation Engine                    │  │
│  │  RetrainPipeline     │  │  student_ranker (LGBMRanker)             │  │
│  │  EngagementAE        │  │  instructor_ranker (LGBMRanker)          │  │
│  │  Hot-Reload (12+ cmp)│  │  RecommendationRetrainPipeline          │  │
│  │  Canary support      │  │  FairnessAuditor                        │  │
│  └──────────────────────┘  └──────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ Monitoring                                                         │  │
│  │  Evidently DriftMonitor | ExplanationDriftDetector (JSD)           │  │
│  │  ConceptDriftProxy | Slack webhook | MLflow experiment tracking    │  │
│  │  PredictionLogger (rolling 10K) | TrustScorer                     │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  IAM Role: ECR Pull + S3 Read/Write                                      │
│  Security Group: SSH(22) + HTTP(80, 8000)                                │
└──────────────────────────────────────────────────────────────────────────┘
                      │
         ┌────────────┴────────────────────────┐
         ▼                                      ▼
┌──────────────────────┐         ┌──────────────────────────────────┐
│  RDS db.t3.micro     │         │  S3 (versioned + AES256)          │
│  PostgreSQL 16       │         │                                    │
│  ┌────────────────┐  │         │  Dropout models:                   │
│  │ users          │  │         │    gbm.pkl, rf.pkl, lstm.pt        │
│  │ learner_       │  │         │    engagement/ (autoencoder)       │
│  │  profiles      │  │         │                                    │
│  │ instructor_    │  │         │  Recommender models:               │
│  │  profiles      │  │         │    recommenders/student_ranker.pkl │
│  │ course_        │  │         │    recommenders/instructor_ranker  │
│  │  enrollments   │  │         │                                    │
│  │ student_       │  │         │  Meta:                             │
│  │  snapshots     │  │         │    model_version.json              │
│  │ interaction_   │  │         │    model_manifest.json             │
│  │  events        │  │         │    recommendation_training_        │
│  │ feedback_      │  │         │    summary.json                    │
│  │  records       │  │         └──────────────────────────────────┘
│  │ prediction_log │  │
│  │ explanation_   │  │         ┌──────────────────────────────────┐
│  │  records       │  │         │  ECR                              │
│  └────────────────┘  │         │  Docker images (keep 5 + scan)    │
│  SG: EC2 only        │         └──────────────────────────────────┘
│  7-day backups       │
└──────────────────────┘
```
  
---
  
## 3. Security Model
  
| Layer | Mechanism |
|-------|-----------|
| **Network** | RDS private (SG allows only EC2 SG), SSH restricted to admin IP |
| **Auth** | JWT HS256, 24h expiry — student / instructor / admin roles |
| **MLOps protection** | `/mlops/retrain` + `/mlops/reload` require admin JWT OR `X-MLOPS-Token` header |
| **Secrets** | GitHub Secrets → EC2 `/opt/xai/.env` (chmod 600) — never in Terraform state |
| **S3** | Private, blocked public access, AES256 SSE |
| **ECR** | Scan-on-push for vulnerabilities |
| **Docker** | Non-root user (`xaiuser`), health checks |
| **DB** | Alembic-managed migrations, connection string via env only |
  
### Cross-Portal Visibility
  
| Actor | Can See | Mechanism |
|-------|---------|-----------|
| **Student** | Own dashboard, risk, explanations, recommendations, history | JWT learner_id → profile filter |
| **Instructor** | Enrolled students' risk, per-student recommendations, what-if | CourseEnrollment join, InstructorProfile filter |
| **Admin** | All users, all enrollments, system health, MLOps, drift, fairness | role="admin", no enrollment filter |
  
---
  
## 4. Implementation Gap Analysis & Status
  
### ✅ Complete (No Changes Needed)
  
| Component | Files |
|-----------|-------|
| Auth (JWT, bcrypt, 24h) | `auth/auth.py`, `auth/router.py`, `auth/models.py`, `auth/schemas.py` |
| Role guards | `get_current_user`, `get_current_active_student`, `get_current_instructor`, `get_current_admin` |
| Student recommendation + XAI | `POST /recommend/student` + `/recommend/student/explain` |
| Instructor recommendation + XAI | `POST /instructor/recommend/{learner_id}` + explain |
| Recommendation retraining pipeline | `RecommendationRetrainPipeline` |
| Event tracking backend | `POST /events` → EventStore (PostgreSQL) |
| Event tracker frontend service | `eventTracker.js` (buffer, 10s flush, beforeunload), `useEventTracker.js` |
| Feedback collection | `POST /feedback` → FeedbackStore |
| All frontend pages | Login, Register, Admin/Instructor/Student Dashboards, WhatIf, History, ActionPlan, Compare, Simulate |
| Dropout model pipeline | GBM/RF/LSTM + EngagementAutoencoder + RetrainPipeline + validation gates |
| Drift monitoring | Evidently + JSD + ConceptDriftProxy + Slack alerts |
| MLOps endpoints | `/mlops/retrain`, `/mlops/retrain-all`, `/mlops/reload`, `/mlops/reload-all`, `/mlops/drift-report`, `/mlops/health` |
| Demo seeder | `_seed_demo_users()` seeds 3 core + 6 roster students + enrollments + snapshots |
| Admin enrollment endpoints | `GET/POST/DELETE /admin/enrollments`, `GET /admin/instructors` |
| Admin user sync endpoints | `POST /admin/sync-oulad`, `GET /admin/students`, `POST /admin/users/{id}/activate`, `POST /admin/users/{id}/deactivate` |
| Admin bulk enrollment | `POST /admin/enrollments/bulk` |
| S3 recommender sync | `s3_loader.py` — `recommenders/` dir upload/download, `recommendation_training_summary.json` |
| Navbar admin routing | Admin role sees Instructor View + Admin Dashboard links |
| Sidebar | Sign Out button, expanded admin links (What-If, Simulate, Compare, XAI Tools) |
| `/admin` route | Redirects to `/dashboard/admin` |
| Admin Dashboard UI | OULAD Sync button, User Management table with risk badges + activate/deactivate |
| Event type constants | `frontend/src/constants/eventTypes.js` |
| Page event tracking | `useEventTracker` + `PAGE_VIEW` wired in all 11 pages; `WHATIF_RUN` on mutation |
| learner_id propagation | `setLearnerId` called on login/logout/signup in AuthContext |
| Cold-start StudentSnapshot | Created automatically on student registration |
| learner_id auto-generation | Backend generates `learner_{uuid8}` if student registers without providing one |
  
### ❌ Remaining Gaps
  
| # | Gap | Severity |
|---|-----|----------|
| **G1** | Precomputed CSVs (`student_topk.csv`, `instructor_topk.csv`) not yet included in S3 upload/download | MEDIUM |
| **G2** | AWS infrastructure not yet provisioned (Terraform never applied) | BLOCKER for deploy |
| **G3** | GitHub Secrets not yet configured | BLOCKER for CI/CD |
| **G4** | Initial model upload to S3 not done | BLOCKER for deploy |
  
---
  
## 5. Implementation Phases
  
### Phase A — S3 Precomputed CSV Sync (G1) ⏱ ~15 min
  
Add `data/recommendations/precomputed/student_topk.csv` and `instructor_topk.csv` to `s3_loader.py` upload/download.
  
**File**: `backend/app/model/s3_loader.py`
  
```python
# In _DATA_FILES (new list) or alongside _MODEL_FILES:
_PRECOMPUTED_FILES = [
    ("data/recommendations/precomputed/student_topk.csv",    "precomputed/student_topk.csv"),
    ("data/recommendations/precomputed/instructor_topk.csv", "precomputed/instructor_topk.csv"),
]
```
  
---
  
### Phase B — AWS Bootstrap (G2, G3, G4) ⏱ ~30 min total
  
#### B.1 One-Time Manual Setup (~15 min)
  
```bash
# 1. Create IAM user xai-ci-cd with these policies:
#    AmazonEC2FullAccess, AmazonRDSFullAccess, AmazonS3FullAccess,
#    AmazonEC2ContainerRegistryFullAccess, AmazonDynamoDBFullAccess
# Save: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
  
# 2. Create Terraform state backend
aws s3 mb s3://xai-rec-tf-state --region ap-south-1
aws dynamodb create-table \
  --table-name xai-rec-tf-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region ap-south-1
  
# 3. Generate SSH key pair
ssh-keygen -t rsa -b 4096 -f ~/.ssh/xai-rec-key -N ""
  
# 4. Create tfvars
cp infra/terraform.tfvars.example infra/terraform.tfvars
# Set: db_password, ssh_public_key_path, allowed_ssh_cidr (your IP)
```
  
#### B.2 Provision Infrastructure (~10 min)
  
```bash
cd infra
terraform init      # Connects to S3 state backend
terraform validate
terraform plan
terraform apply     # Creates: EC2 + RDS + ECR + S3 + VPC/SGs
```
  
**Save these outputs for GitHub Secrets:**
  
| Output | Secret Name |
|--------|-------------|
| `ec2_public_ip` | `EC2_HOST` |
| `ecr_repo_url` | `ECR_REPO_URL` |
| `database_url` | `DATABASE_URL` |
| `s3_bucket_name` | `AWS_S3_BUCKET` |
  
#### B.3 Configure GitHub Secrets (~5 min)
  
GitHub → Repo Settings → Secrets and variables → Actions:
  
| Secret | Value |
|--------|-------|
| `AWS_ACCESS_KEY_ID` | From B.1 |
| `AWS_SECRET_ACCESS_KEY` | From B.1 |
| `AWS_REGION` | `ap-south-1` |
| `ECR_REPO_URL` | From `terraform output ecr_repo_url` |
| `EC2_HOST` | From `terraform output ec2_public_ip` |
| `EC2_SSH_KEY` | Contents of `~/.ssh/xai-rec-key` (private key) |
| `DATABASE_URL` | From `terraform output -raw database_url` |
| `JWT_SECRET_KEY` | `python -c "import secrets; print(secrets.token_hex(32))"` |
| `GROQ_API_KEY` | From https://console.groq.com |
| `AWS_S3_BUCKET` | From `terraform output s3_bucket_name` |
| `DB_PASSWORD` | Same as in terraform.tfvars |
| `MLOPS_AUTOMATION_TOKEN` | `python -c "import secrets; print(secrets.token_hex(32))"` |
  
Also create a GitHub Environment named `production` (Settings → Environments).
  
#### B.4 Initial Model Upload to S3 (~2 min)
  
```bash
export AWS_S3_BUCKET=$(terraform -chdir=infra output -raw s3_bucket_name)
make models-upload
# Uploads: gbm.pkl, rf.pkl, lstm.pt, engagement/, recommenders/,
#          model_version.json, model_manifest.json, recommendation_training_summary.json
```
  
#### B.5 First Deployment (~10 min)
  
```bash
git add . && git commit -m "deploy: initial production release"
git push origin main
```
  
`deploy.yml` runs automatically:
1. Multi-stage Docker build (React frontend + Python backend)
2. Push to ECR (tagged with commit SHA + timestamp)
3. SSH to EC2 → write secrets to `/opt/xai/.env`
4. Pull image by digest (immutable)
5. `alembic upgrade head` (creates all DB tables)
6. Restart `xai-api` systemd service
7. Health check loop (up to 2 min)
8. On failure: automatic rollback to previous image
  
#### B.6 Verify Deployment (~5 min)
  
```bash
EC2_IP=$(terraform -chdir=infra output -raw ec2_public_ip)
  
curl -sf http://$EC2_IP:8000/health | python -m json.tool
  
curl -X POST http://$EC2_IP:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"login_frequency_weekly":3,"avg_session_duration_min":45,"forum_posts_count":2,
       "video_completion_rate":0.6,"quiz_avg_score":55,"quiz_completion_rate":0.7,
       "assignment_submission_rate":0.65,"days_since_last_activity":7,
       "prior_course_completions":1,"current_week_in_course":6,
       "missed_deadlines_count":2,"help_requests_count":1,
       "engagement_latent_1":0,"engagement_latent_2":0,"engagement_latent_3":0}'
  
echo "Swagger: http://$EC2_IP:8000/docs"
echo "Frontend: http://$EC2_IP:8000/login"
```
  
---
  
## 6. Operational Runbook
  
### 6.1 Weekly Retrain (Automatic)
  
Configured in `retrain.yml` — runs every **Sunday 02:00 UTC** or manually via GitHub Actions UI:
1. `POST /mlops/retrain-all` (with `X-MLOPS-Token`)
2. On success: `POST /mlops/reload-all`
3. Health check confirms live
  
### 6.2 Rollback Scenarios
  
| Scenario | Mechanism |
|----------|-----------|
| Deploy fails health check | `deploy.yml` reverts to previous Docker image digest |
| Retrain fails validation gates | `RetrainPipeline` returns `success: false`, model unchanged |
| Hot-reload fails | Staging dict discarded, `AppState` untouched |
| Canary shows issues | `POST /mlops/reload?canary_fraction=0.0` aborts canary |
| Infra drift | Terraform state in S3; `terraform plan` surfaces drift |
  
### 6.3 Admin Workflows
  
| Task | Action |
|------|--------|
| Bulk-create users from OULAD | Admin Dashboard → **Sync OULAD Users** button |
| View / activate / deactivate users | Admin Dashboard → **User Management** table |
| Add enrollment | Admin Dashboard → **Enrollment Management** → Add Enrollment form |
| Bulk enroll | `POST /admin/enrollments/bulk` (API or future bulk UI) |
| Trigger retrain | Admin Dashboard → **MLOps Controls** → Run Retrain |
| Hot-reload model | Admin Dashboard → **MLOps Controls** → Hot Reload |
| Check drift | Admin Dashboard → **Drift Report** section |
| Fairness audit | Admin Dashboard → **Fairness Monitor** → Run Fairness Check |
  
### 6.4 Auth Flows
  
**Login:**
```
POST /auth/login { username, password }
→ bcrypt verify → JWT { sub: user_id, role, learner_id, exp: 24h }
→ Frontend: localStorage(ll_token, ll_user) + setLearnerId(learner_id)
→ Redirect: student→/student, instructor→/instructor, admin→/dashboard/admin
```
  
**Registration (Student):**
```
POST /auth/register { username, email, password, role="student", ... }
→ Creates User + LearnerProfile
→ Auto-generates learner_id if not provided (learner_{uuid8})
→ Creates cold-start StudentSnapshot (enables dashboard immediately)
→ Returns { id, username, email, role, learner_id }
```
  
**Admin OULAD Sync:**
```
POST /admin/sync-oulad  (admin JWT required)
→ Reads data/oulad_seed_credentials.json
→ For each entry: create User + Profile (skip if username exists)
→ For students: create StudentSnapshot with OULAD features
→ Returns { created_users, skipped_existing, snapshots_created, errors }
```
  
---
  
## 7. Cost Estimate (AWS Free Tier)
  
| Resource | Free Tier | Our Usage | Notes |
|----------|-----------|-----------|-------|
| EC2 t3.micro | 750 hrs/month (12 mo) | 1 instance 24/7 | Within free tier |
| RDS db.t3.micro | 750 hrs/month (12 mo) | 1 instance, 20GB | Within free tier |
| S3 | 5GB, 20K GET, 2K PUT | ~300MB models + CSVs | Within free tier |
| ECR | 500MB storage | ~1.2GB (5 images) | ~<img src="https://latex.codecogs.com/gif.latex?0.05/month%20overage%20||%20DynamoDB%20|%2025GB%20+%2025%20WCU/RCU%20|%20TF%20state%20lock%20only%20|%20Within%20free%20tier%20||%20Data%20transfer%20|%20100GB%20out/month%20|%20Minimal%20|%20Within%20free%20tier%20|**Estimated%20monthly%20cost:%20~"/>0.05** (ECR overage only).
  
---
  
## 8. File Reference
  
### Backend
  
| File | Purpose |
|------|---------|
| `backend/app/main.py` | All FastAPI endpoints (3500+ lines) |
| `backend/app/auth/router.py` | register, login, me, enroll — auto learner_id + cold-start snapshot |
| `backend/app/auth/models.py` | User, LearnerProfile, InstructorProfile, CourseEnrollment, StudentSnapshot |
| `backend/app/model/s3_loader.py` | S3 upload/download — dropout models + engagement/ + recommenders/ |
| `backend/app/model/retrain_pipeline.py` | Dropout model retraining |
| `backend/app/recommender/train_recommenders.py` | LambdaMART ranker training |
| `backend/app/recommender/retrain_pipeline.py` | Recommendation retraining pipeline |
| `backend/app/tracker/event_store.py` | EventStore (PostgreSQL) |
| `backend/app/tracker/drift_detector.py` | JSD-based explanation drift |
| `backend/app/mlops/drift_monitor.py` | Evidently drift monitor |
| `backend/app/narrator/llm_narrator.py` | Groq narration + rule-based fallback |
| `alembic/versions/` | DB schema migrations |
| `infra/` | Terraform (EC2, RDS, S3, ECR, VPC) |
| `.github/workflows/` | ci.yml, deploy.yml, retrain.yml, terraform.yml |
  
### Frontend
  
| File | Purpose |
|------|---------|
| `frontend/src/constants/eventTypes.js` | Canonical event type constants |
| `frontend/src/context/AuthContext.jsx` | Auth state + setLearnerId on login/logout |
| `frontend/src/api/recommend.js` | API client — recommend, admin, sync-oulad, bulk-enroll |
| `frontend/src/api/dropout.js` | API client — predict, MLOps, health |
| `frontend/src/components/Navbar.jsx` | Role-aware nav (student/instructor/admin links) |
| `frontend/src/components/Sidebar.jsx` | Role-aware sidebar, Sign Out button |
| `frontend/src/pages/AdminDashboard.jsx` | System health, MLOps, enrollment, user management, OULAD sync |
| `frontend/src/pages/StudentDashboard.jsx` | Risk score, recommendations, page_view + reco events |
| `frontend/src/pages/InstructorDashboard.jsx` | Student roster, recommendations |
| `frontend/src/pages/WhatIfExplorer.jsx` | What-if analysis + whatif_run event |
| `frontend/src/pages/SimulatePage.jsx` | Monte Carlo simulation + simulate_run event |
| `frontend/src/pages/ComparePage.jsx` | Model comparison + compare_run event |
| `frontend/src/pages/HistoryPage.jsx` | Explanation history + history_viewed event |
| `frontend/src/pages/ActionPlan.jsx` | Prioritised actions + action_viewed event |
| `frontend/src/App.jsx` | Routes — /admin → /dashboard/admin redirect |
  
---
  
## 9. Implicit Data → MLOps: End-to-End Trace
  
```
1. Student clicks "View Explanation" on StudentView
   → useEventTracker.track(EXPLAIN_REQUESTED, { target: learner_id })
   → Buffered in eventTracker.js
  
2. Buffer hits 50 events OR 10s timer fires (or beforeunload)
   → POST /events { events: [...50 events] }
   → EventStore writes to PostgreSQL interaction_events
  
3. Sunday 02:00 UTC — retrain.yml fires
   → POST /mlops/retrain-all (X-MLOPS-Token)
   → RetrainPipeline:
       a. ImplicitAggregator: EventStore → 20 engagement signals per learner
       b. EngagementAutoencoder: 20 → 3 latent features
       c. Augment: OULAD_12 + latent_3 = 15 features
       d. Retrain GBM + RF, validate gates (AUC/Brier/SHAP)
   → RecommendationRetrainPipeline:
       a. Same signals → enrich recommendation training data
       b. Retrain student_ranker + instructor_ranker (LambdaMART)
       c. Regenerate precomputed top-K CSVs
   → Both save to disk → upload to S3
  
4. POST /mlops/reload-all
   → S3 download → atomic swap of ALL 12+ components
   → Health check confirms live
   → Zero downtime (canary mode optional)
```
  
---
  
*Last updated: 2025-07-17. All Phase 1–3 implementation tasks complete. Remaining blocker: AWS provisioning (Phase B).*
  