## Plan: MLOps Pipeline Hardening

Strengthen security, deployment determinism, retrain safety, and observability with a phased rollout that reduces production risk first, then improves model governance and monitoring depth.

**Steps**
1. Phase 0: Immediate risk controls. Add authentication and authorization guards for retrain and reload endpoints in [backend/app/main.py](backend/app/main.py). Add signed caller validation for automation-triggered operations and enforce HTTPS-only production calls in workflows. This phase blocks unauthorized control-plane actions and insecure transport.
2. Phase 0: Concurrency safety for live swaps. Add a deployment lock around reload paths in [backend/app/model/hot_reload.py](backend/app/model/hot_reload.py) and integrate lock acquisition and lock-fail responses in [backend/app/main.py](backend/app/main.py). This prevents overlapping reload/retrain requests.
3. Phase 1: Deterministic deployment and rollback. Update deploy workflow to use immutable image tags or image digests, not mutable latest, in [.github/workflows/deploy.yml](.github/workflows/deploy.yml). Record deployed artifact identity and previous artifact identity for rollback.
4. Phase 1: Artifact manifest governance. Add model manifest generation and validation for S3 sync in [backend/app/model/s3_loader.py](backend/app/model/s3_loader.py) and retrain pipeline output in [backend/app/model/retrain_pipeline.py](backend/app/model/retrain_pipeline.py). Include model version, training timestamp, metrics snapshot, feature schema hash, and data fingerprint.
5. Phase 2: Centralize MLflow config. Remove tracking URI divergence by routing all training, tuning, and retrain flows through one config source used by [backend/app/mlops/experiment_tracker.py](backend/app/mlops/experiment_tracker.py), [backend/app/model/trainer.py](backend/app/model/trainer.py), [backend/app/model/tune.py](backend/app/model/tune.py), and [backend/app/model/retrain_pipeline.py](backend/app/model/retrain_pipeline.py). *Parallel with step 4.*
6. Phase 2: Retrain gate hardening. Expand validation in [backend/app/model/retrain_pipeline.py](backend/app/model/retrain_pipeline.py) beyond AUC and Brier drift: enforce minimum data volume, class balance checks, cold-start fraction guardrails, feature-schema consistency, and explicit rejection reasoning payloads. Add strict mode configurable via env.
7. Phase 2: Async operational performance improvements. Move prediction log pruning out of request path in [backend/app/mlops/prediction_logger.py](backend/app/mlops/prediction_logger.py), using periodic cleanup or amortized pruning windows. Add counters and timing for prune jobs.
8. Phase 3: Monitoring upgrade. Extend [backend/app/mlops/drift_monitor.py](backend/app/mlops/drift_monitor.py) with configurable TTL and thresholds, drift trend windows, and alert hooks. Add concept-drift proxies using feedback outcomes and prediction calibration movement where labels become available. *Depends on step 7 for stable telemetry.*
9. Phase 3: CI/CD observability hardening. Update [.github/workflows/ci.yml](.github/workflows/ci.yml) to emit junit XML and upload artifacts reliably. Add deploy/redeploy smoke checks in workflow gates for [backend/app/main.py](backend/app/main.py) endpoints: health, predict, retrain auth, reload auth, drift-report.
10. Phase 4: Operational runbooks and rollout controls. Document rollback, retrain failure triage, and drift-response SOP in project docs. Add canary-style rollout option for reload path and define promotion criteria for Production stage in model registry.

**Dependencies and Parallelization**
1. Security and transport changes in phase 0 must complete before any redeploy automation hardening is considered done.
2. Immutable deploy identity and S3 manifest validation should ship before expanded rollback automation.
3. MLflow centralization and manifest governance can run in parallel, but both must be complete before tightening retrain promotion gates.
4. Monitoring alert logic depends on stable log retention and predictable feature schema from earlier phases.

**Relevant files**
- [backend/app/main.py](backend/app/main.py) — MLOps endpoints, auth hooks, lock integration, runtime checks
- [backend/app/model/hot_reload.py](backend/app/model/hot_reload.py) — atomic swap safety and lock behavior
- [backend/app/model/retrain_pipeline.py](backend/app/model/retrain_pipeline.py) — validation gates, retrain acceptance logic, artifact metadata
- [backend/app/model/s3_loader.py](backend/app/model/s3_loader.py) — artifact freshness checks and remote/local reconciliation
- [backend/app/mlops/experiment_tracker.py](backend/app/mlops/experiment_tracker.py) — unified experiment tracking behavior
- [backend/app/mlops/model_registry.py](backend/app/mlops/model_registry.py) — promotion and rollback guardrails
- [backend/app/mlops/prediction_logger.py](backend/app/mlops/prediction_logger.py) — request-path overhead and retention strategy
- [backend/app/mlops/drift_monitor.py](backend/app/mlops/drift_monitor.py) — drift report lifecycle, thresholds, alerts
- [.github/workflows/deploy.yml](.github/workflows/deploy.yml) — immutable artifact deployment and post-deploy verification
- [.github/workflows/retrain.yml](.github/workflows/retrain.yml) — secure retrain trigger and reload sequencing
- [.github/workflows/ci.yml](.github/workflows/ci.yml) — test artifact reliability and trend visibility
- [Dockerfile](Dockerfile) — runtime env, health semantics, deploy runtime assumptions
- [docker-compose.yml](docker-compose.yml) — local parity and operational baseline checks

**Verification**
1. Security verification: unauthenticated and non-admin calls to retrain/reload must fail with expected status codes and audit logs.
2. Deployment verification: deploy workflow must pin a specific artifact identity and expose it in release summary.
3. Reload safety verification: concurrent reload attempts should produce one success and deterministic lock-based rejections.
4. Retrain quality verification: gate-fail scenarios must reject promotion with explicit reasons and preserve prior production model.
5. Drift monitoring verification: simulated distribution shift should trigger drift signals and configured alerts.
6. CI verification: junit test artifact should always be generated and uploaded; smoke tests should gate deploy success.

**Decisions**
- Include now: backend security, deployment determinism, retrain validation, drift and monitoring reliability, CI observability.
- Exclude now: frontend changes, broad architecture rewrites, replacing current model stack.
- Keep current serving architecture: improve guardrails and controls around existing FastAPI plus reload flow.

**Further Considerations**
1. Alerting target selection: Option A use Slack webhook first for fast rollout, Option B use CloudWatch alarms for tighter AWS-native ops, Option C dual channel for redundancy.
2. Promotion policy strictness: Option A automatic promotion when gates pass, Option B manual approval for production stage, Option C auto to staging then human production approval.
3. Drift strategy depth: Option A input drift only for low overhead, Option B input plus calibration drift, Option C full concept drift when outcome labels mature.
