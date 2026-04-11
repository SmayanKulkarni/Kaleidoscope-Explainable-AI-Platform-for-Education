# Operations Runbook — XAI Learning Recommendation System

> **Audience:** On-call engineers, MLOps operators
> **Last updated:** see git log

---

## 1. Rollback (Deployment)

### Trigger
Health check failure after deploy, or error rate spike observed post-deploy.

### Steps
```bash
# 1. SSH into EC2
ssh ec2-user@$EC2_HOST

# 2. Check current and previous image
cat /opt/xai/deploy-history.json
# {"deployed_at": "...", "image": "...", "tag": "...", "previous": "...", "commit": "..."}

# 3. Rollback to previous pinned image
PREV=$(cat /opt/xai/deploy-history.json | python3 -c "import json,sys; print(json.load(sys.stdin)['previous'])")
sudo sed -i "s|XAI_IMAGE=.*|XAI_IMAGE=${PREV}|" /opt/xai/.env
sudo systemctl restart xai-api

# 4. Verify
sleep 15
curl -sf http://localhost:8000/health
```

**Automated rollback:** The deploy.yml script will automatically restart with `PREV_IMAGE` if health checks fail. Check GitHub Actions summary for details.

---

## 2. Rollback (Model — MLflow)

### Trigger
Drift report shows severe distribution shift after a retrain + reload, or production AUC degrades.

### Steps
```bash
# Option A: via API (requires X-MLOPS-Token)
curl -X POST http://$EC2_HOST:8000/mlops/reload \
  -H "X-MLOPS-Token: $MLOPS_AUTOMATION_TOKEN"
# This loads whatever is in models/ — so you must first restore models/ from git or S3

# Option B: restore previous model artifacts from git LFS
git log models/gbm.pkl --oneline | head -5   # find previous commit
git checkout <prev_commit_sha> -- models/gbm.pkl models/rf.pkl models/model_version.json models/training_summary.json models/model_manifest.json
# Then POST /mlops/reload

# Option C: MLflow registry rollback (demote current version, promote previous)
python3 -c "
import mlflow
from backend.app.mlops.mlflow_config import TRACKING_URI
mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions('name=\"gbm-dropout-risk\"')
# Find previous Production version and promote it back
"
```

**Promotion criteria for Production stage:**
- `test_auc_roc` ≥ 0.90
- `test_brier_score` ≤ 0.12
- `test_f1` ≥ 0.75
- Cold-start fraction < 80%
- Feature schema hash matches current manifest
- All retrain gates pass (no rejection reasons)

---

## 3. Retrain Failure Triage

### Possible `rejection_reason` values and fixes

| Rejection message | Root cause | Fix |
|---|---|---|
| `Insufficient training data: N rows < 1000` | Too few events accumulated | Wait for more interactions, or lower `min_events` guard |
| `Class imbalance: positive fraction X < 0.05` | Almost no dropout cases in data | Check event aggregation, possible data pipeline bug |
| `Cold-start fraction X% > 95%` | No real engagement events recorded | Check `/events` ingestion, ensure implicit events flow |
| `Feature schema changed: prev_hash != curr_hash` | Feature list changed between retrains | Run schema migration; update manifest manually if intentional |
| `AUC regression: new < prev - 0.02` | New model worse than production | Check data quality, event aggregation logic |
| `Brier regression: new > prev + 0.03` | Calibration degraded | Recalibrate with `CalibratedClassifierCV`; check class balance |

### Enabling strict mode
```bash
export RETRAIN_STRICT_MODE=true
# Adds cold-start fraction as a hard rejection gate
```

### Manual retrain with override
```bash
curl -X POST "http://$EC2_HOST:8000/mlops/retrain?min_events=0&operator=manual-override" \
  -H "X-MLOPS-Token: $MLOPS_AUTOMATION_TOKEN"
```

---

## 4. Drift Response SOP

### Severity levels

| `drift_share` | `trend_direction` | Action |
|---|---|---|
| < 20% | any | Monitor, no action |
| 20–40% | stable | Investigate top drifting features, schedule retrain |
| 20–40% | increasing | **Trigger retrain this week** |
| > 40% | any | **Immediate retrain + reload** |
| `alert_fired: true` + `concept_drift_proxy_flag: true` | any | **Escalate to ML lead** |

### Check drift status
```bash
curl -sf http://$EC2_HOST:8000/mlops/drift-report | python3 -m json.tool
```

### Configuring thresholds (env vars)
```bash
DRIFT_CACHE_TTL_SECONDS=1800   # check every 30m instead of 1h
DRIFT_SHARE_THRESHOLD=0.15     # alert at 15% instead of 20%
DRIFT_TREND_WINDOW=10          # use 10-report rolling window
SLACK_DRIFT_WEBHOOK_URL=https://hooks.slack.com/services/xxx  # Slack alerts
```

---

## 5. Canary Reload

To gradually roll out a new model without full cutover, use `canary_fraction`:

```bash
# Serve new model for 20% of /predict requests
curl -X POST "http://$EC2_HOST:8000/mlops/reload?canary_fraction=0.2" \
  -H "X-MLOPS-Token: $MLOPS_AUTOMATION_TOKEN"

# Promote to 100% after validation
curl -X POST "http://$EC2_HOST:8000/mlops/reload?canary_fraction=1.0" \
  -H "X-MLOPS-Token: $MLOPS_AUTOMATION_TOKEN"

# Abort canary (revert to previous model)
curl -X POST "http://$EC2_HOST:8000/mlops/reload?canary_fraction=0.0" \
  -H "X-MLOPS-Token: $MLOPS_AUTOMATION_TOKEN"
```

---

## 6. CI Failures

### pytest artifact missing
The test run always emits `pytest-results-<run_id>.xml`. If the artifact upload step shows `warn`, the test process itself likely crashed before writing the XML. Check the raw pytest stdout in the Actions log.

### Smoke test failures
Each smoke test step is independent. A failure means:
- `/health` fail → server not running; check `systemctl status xai-api`
- `/predict` fail → model not loaded; check startup logs
- `/mlops/retrain` auth fail → JWT or `X-MLOPS-Token` auth guard broken; regression in `main.py`

---

## 7. Useful Commands

```bash
# Server logs
sudo journalctl -u xai-api -f

# Current model version
curl -sf http://localhost:8000/mlops/health | python3 -m json.tool

# Check manifest
cat /opt/xai/models/model_manifest.json

# MLflow UI (local)
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# SDK quick test
python3 -c "
from backend.sdk.xai_sdk import XAIClient
c = XAIClient()
print(c.health())
"
```
