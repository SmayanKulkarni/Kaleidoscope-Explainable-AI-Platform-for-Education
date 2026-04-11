# Frontend ↔ Backend Integration Plan

Comprehensive guide for the frontend engineer to connect every backend endpoint, with role-based views, sample data, and a seed script that populates demo users via Groq.

---

## 1. Backend API — Complete Endpoint Catalogue

Base URL: `VITE_API_URL` (default `http://localhost:8000`)

### Auth Endpoints (prefix `/auth`)

| Method | Path | Auth | Request | Response | Used by |
|--------|------|------|---------|----------|---------|
| POST | `/auth/register` | None | `{username, email, password, full_name?, role: "student"\|"instructor"\|"admin", learner_id?, course_id?, module_presentation?, department?}` | `{id, username, email, role, learner_id?, message}` | Registration page |
| POST | `/auth/login` | None | `{username, password}` | `{access_token, token_type:"bearer", expires_in, role, user_id, learner_id?}` | Login page |
| GET | `/auth/me` | Bearer | — | `{id, username, email, full_name, role, is_active, created_at, last_login_at, learner_profile?, instructor_profile?}` | Profile panel, role routing |
| GET | `/auth/students` | Instructor | `?course_id=` (optional) | `[{learner_id, full_name, username, course_id, current_week, enrolled_at}]` | Instructor student list |
| POST | `/auth/enroll` | Instructor | `?learner_id=&course_id=` | `{message, enrollment_id, learner_id, course_id}` | Instructor enroll dialog |
| PUT | `/auth/me/week` | Student | `?week=N` (1–52) | `{learner_id, current_week, updated}` | Week sync |
| DELETE | `/auth/me` | Bearer | — | `{message, user_id}` | Account settings |

**JWT token** goes in `Authorization: Bearer <token>` header for all authenticated calls.

### Dropout Risk Engine

| Method | Path | Auth | Body | Response highlights |
|--------|------|------|------|---------------------|
| GET | `/health` | None | — | `{status, model_version, gbm_loaded, lstm_loaded, device, gpu, timestamp}` |
| POST | `/predict` | None | `LearnerFeatures` + `?model=gbm\|lstm` | `{risk_score, risk_label, model_used, uncertainty?}` |
| POST | `/explain` | None | `{features: LearnerFeatures, learner_id, model, audience, history?}` | `{risk_score, risk_label, shap_values, base_value, top_features, stability, interactions?, anchor_rule?, prototypes?, counterfactual?, ranked_actions?, causal_annotations?, trust_score?, uncertainty?, temporal_attributions?, explanation_drift?, narratives?}` |
| POST | `/whatif` | None | `{features: LearnerFeatures, overrides: {feature: value}}` | `{shap_values, base_value, risk_score, risk_delta, top_features}` |
| POST | `/compare` | None | `{features: LearnerFeatures, history?, learner_id}` | `{gbm_score, gbm_label, gbm_top3, lstm_score?, lstm_label?, lstm_top3_temporal?, disagreement_flag, …}` |
| POST | `/counterfactual` | None | `LearnerFeatures` | `{actions, counterfactuals, best_cf, changed_features, ranked_actions?}` |
| POST | `/simulate` | None | `{features: LearnerFeatures, current_week, target_week, n_simulations}` | `{feature_distributions, outcome_distribution, risk_percentiles}` |
| GET | `/history/{learner_id}` | None | — | `{learner_id, timeline, history, drift_flags}` |

### Authenticated Self-Service (Student)

| Method | Path | Auth | Body | Response |
|--------|------|------|------|----------|
| POST | `/explain/me` | Student | `{audience?}` | `{learner_id, current_week, course_id, latest_risk, top3_features, trust_score, history_url}` |
| GET | `/explain/me/history` | Student | — | `{learner_id, timeline, history, drift_flags}` |

### Recommendation Engine

| Method | Path | Auth | Body | Response |
|--------|------|------|------|----------|
| GET | `/recommend/health` | None | — | `{student_ranker: {loaded, n_features, …}, instructor_ranker: {…}}` |
| POST | `/recommend/student` | None | `{learner_id, items: [{item_id, features}], top_k, include_shap}` | `{learner_id, top_k, recommendations: [{item_id, score, rank, top_features, shap_values}]}` |
| POST | `/recommend/student/explain` | None | `{learner_id, features, item_id}` | `{learner_id, item_id, score, shap_values, top_features, anchor_rule, feature_interactions, causal_annotations, shap_stability, plain_language}` |
| POST | `/recommend/student/whatif` | None | `{learner_id, features, overrides}` | `{learner_id, original_score, modified_score, score_delta, direction, changed_features, shap_delta}` |
| POST | `/recommend/instructor` | None | `{instructor_id, items, top_k, include_shap}` | Same shape as student |
| POST | `/recommend/instructor/explain` | None | `{instructor_id, features, item_id}` | Same shape as student explain |
| POST | `/recommend/instructor/whatif` | None | `{instructor_id, features, overrides}` | Same shape as student whatif |

### Feedback + Events

| Method | Path | Auth | Body | Response |
|--------|------|------|------|----------|
| POST | `/feedback` | None | `{learner_id, rating? (1–5), followed_recommendation?, explanation_id?, correction_feature?, correction_comment?, audience?}` | `{recorded, feedback_id, learner_id}` |
| GET | `/feedback/stats` | None | — | `{avg_rating, n_ratings, follow_rate, top_corrections, …}` |
| GET | `/feedback/{learner_id}` | None | — | `{learner_id, records: [...]}` |
| POST | `/events` | None | `{events: [{learner_id, session_id?, event_type, event_target?, event_value?, page?, extra?, client_ts?}]}` | `{recorded, message}` |
| GET | `/events/{learner_id}` | None | `?limit=100` | `{learner_id, count, events: [...]}` |

**Valid `event_type` values:** `page_view`, `click`, `scroll`, `whatif_slider`, `action_viewed`, `action_dismissed`, `explanation_revisit`, `resource_download`, `time_to_first_action`, `prototype_click`, `session_start`, `session_end`

### MLOps (Admin only)

| Method | Path | Auth | Response |
|--------|------|------|----------|
| GET | `/mlops/health` | None | `{status, model_version, drift_status, prediction_count, timestamp}` |
| GET | `/mlops/drift-report` | None | `{dataset_drift, n_drifted_features, …}` |
| GET | `/mlops/metrics` | None | `{training: {auc, f1, brier, …}, model_version}` |
| POST | `/mlops/retrain` | Admin/Token | `{queued, reason?, success?, …}` |
| POST | `/mlops/reload` | Admin/Token | `{success, model_version, …}` |

---

## 2. LearnerFeatures Schema (used in `/predict`, `/explain`, `/whatif`, `/compare`, `/simulate`)

```typescript
interface LearnerFeatures {
  login_frequency_weekly: number;      // 0–14
  avg_session_duration_min: number;    // 0–300
  forum_posts_count: number;           // ≥0
  video_completion_rate: number;       // 0.0–1.0
  quiz_avg_score: number;              // 0.0–100.0
  quiz_completion_rate: number;        // 0.0–1.0
  assignment_submission_rate: number;  // 0.0–1.0
  days_since_last_activity: number;    // ≥0
  prior_course_completions: number;    // ≥0
  current_week_in_course: number;      // 1–52
  missed_deadlines_count: number;      // ≥0
  help_requests_count: number;         // ≥0
  engagement_latent_1?: number;        // default 0.0
  engagement_latent_2?: number;        // default 0.0
  engagement_latent_3?: number;        // default 0.0
}
```

---

## 3. Role-Based View Routing

```
/login                → LoginPage
/register             → RegisterPage

After login, route by `role` from TokenResponse:

role=student    → /dashboard/student
role=instructor → /dashboard/instructor  
role=admin      → /dashboard/admin
```

### Student Dashboard Pages
| Page | Endpoints used |
|------|---------------|
| **My Risk** | `POST /predict`, `POST /explain/me`, `GET /explain/me/history` |
| **Full Explanation** | `POST /explain` (with features + history) |
| **What-If** | `POST /whatif` |
| **Recommendations** | `POST /recommend/student`, `POST /recommend/student/explain` |
| **Simulation** | `POST /simulate` |
| **Feedback** | `POST /feedback`, `POST /events` |

### Instructor Dashboard Pages
| Page | Endpoints used |
|------|---------------|
| **My Students** | `GET /auth/students` |
| **Student Detail** | `POST /explain` (with student's learner_id), `GET /history/{learner_id}` |
| **Student Compare** | `POST /compare` |
| **Interventions** | `POST /recommend/instructor`, `POST /recommend/instructor/explain` |
| **Enroll** | `POST /auth/enroll` |
| **Feedback Review** | `GET /feedback/{learner_id}`, `GET /feedback/stats` |

### Admin Dashboard Pages
| Page | Endpoints used |
|------|---------------|
| **System Health** | `GET /health`, `GET /mlops/health`, `GET /recommend/health` |
| **Model Metrics** | `GET /mlops/metrics` |
| **Drift Report** | `GET /mlops/drift-report` |
| **Retrain** | `POST /mlops/retrain` |
| **Hot Reload** | `POST /mlops/reload` (with canary_fraction slider) |
| **Feedback Analytics** | `GET /feedback/stats` |
| **All student views** | (admin has access to everything) |

---

## 4. Sample Fixture Data (already in repo)

These files in `data/fixtures/` are ready-to-use request bodies:

| File | Risk level | Use with |
|------|-----------|----------|
| `high_risk.json` | High (0.7+) | `POST /explain` |
| `medium_risk.json` | Medium (0.4–0.7) | `POST /explain` |
| `low_risk.json` | Low (<0.4) | `POST /explain` |
| `temporal_sequence.json` | Declining over weeks 2–12 | `POST /compare`, `POST /simulate` |

---

## 5. Seed Script: Generate Demo Users + Data via Groq

**What the script does:** Creates 3 demo accounts (student, instructor, admin), enrolls the student under the instructor, and generates realistic learner feature profiles using Groq LLM for varied natural-language bios/names.

**File:** `scripts/seed_demo_data.py`

```
Usage:  python scripts/seed_demo_data.py --base-url http://localhost:8000
Env:    GROQ_API_KEY (required for name/bio generation; falls back to hardcoded defaults)
```

**Demo accounts created:**

| Username | Password | Role | Extra |
|----------|----------|------|-------|
| `demo_student` | `DemoStudent2026!` | student | learner_id=`demo-learner-001`, course_id=`AAA`, module=`2014J` |
| `demo_instructor` | `DemoInstructor2026!` | instructor | department=`Computer Science` |
| `demo_admin` | `DemoAdmin2026!` | admin | — |

**The script also:**
1. Calls Groq (`llama-3.3-70b-versatile`) to generate 5 additional synthetic student profiles with realistic names, bios, and varied feature distributions
2. Registers all students and enrolls them under the instructor
3. Calls `POST /explain` for each student to populate explanation history
4. Calls `POST /feedback` with sample ratings to populate feedback store
5. Calls `POST /events` with sample interaction events
6. Prints a summary table of all created accounts + JWT tokens

**Groq usage is narration-only:** The LLM generates names/bios/varied feature values — it does NOT generate explanations (those come from the XAI algorithms via the API calls).

---

## 6. Auth Flow — Frontend Implementation

### Login → Store Token → Route by Role

```typescript
// api/auth.ts
async function login(username: string, password: string): Promise<TokenResponse> {
  const res = await axios.post(`${API_URL}/auth/login`, { username, password });
  // Store in localStorage or httpOnly cookie
  localStorage.setItem('token', res.data.access_token);
  localStorage.setItem('role', res.data.role);
  localStorage.setItem('user_id', res.data.user_id);
  if (res.data.learner_id) localStorage.setItem('learner_id', res.data.learner_id);
  return res.data;
}

// Axios interceptor
axios.interceptors.request.use(config => {
  const token = localStorage.getItem('token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});
```

### Token Expiry
- Default: 24 hours (`JWT_EXPIRE_MIN=1440`)
- On 401 response → redirect to `/login`

---

## 7. Event Tracking — Frontend Hooks

The backend expects interaction events for implicit feedback. The frontend should emit these automatically:

```typescript
// hooks/useEventTracker.ts
function trackEvent(type: EventType, target?: string, value?: number) {
  const event = {
    learner_id: localStorage.getItem('learner_id') || 'anonymous',
    session_id: sessionStorage.getItem('session_id'),
    event_type: type,
    event_target: target,
    event_value: value,
    page: window.location.pathname,
    client_ts: new Date().toISOString(),
  };
  eventBuffer.push(event);
  // Flush every 10 events or 30s
  if (eventBuffer.length >= 10) flushEvents();
}
```

**Required event hooks:**
| UI Action | Event Type | event_target | event_value |
|-----------|-----------|-------------|-------------|
| Page loads | `page_view` | page name | — |
| Click on explanation | `click` | element id | — |
| Move whatif slider | `whatif_slider` | feature name | slider value |
| View an action card | `action_viewed` | action feature | — |
| Dismiss an action | `action_dismissed` | action feature | — |
| Revisit explanation | `explanation_revisit` | learner_id | — |
| Download resource | `resource_download` | resource name | — |
| Click on prototype | `prototype_click` | prototype id | — |
| Tab/window focus | `session_start` | — | — |
| Tab/window blur | `session_end` | — | elapsed_ms |

---

## 8. Implementation Checklist

### Phase 1: Scaffold + Auth (Days 1–2)
- [ ] Vite + React 18 + TailwindCSS + shadcn/ui setup
- [ ] `api/client.ts` with axios + interceptor
- [ ] Login page, Register page
- [ ] Token storage + role-based route guards
- [ ] `GET /auth/me` → profile sidebar
- [ ] Run `scripts/seed_demo_data.py` → verify login works

### Phase 2: Student Dashboard (Days 3–5)
- [ ] Risk score card (`POST /predict`)
- [ ] Full explanation view (`POST /explain`)
  - SHAP bar chart
  - Anchor rule chips
  - Trust score meter
  - Counterfactual actions list
  - Causal annotation badges
  - LLM narrative card
- [ ] What-If form (`POST /whatif`)
- [ ] Risk timeline (`GET /explain/me/history`)
- [ ] Simulation view (`POST /simulate`)
- [ ] Feedback widget (`POST /feedback`)

### Phase 3: 3D Graphs (Days 5–7)
- [ ] SHAP interaction network graph
- [ ] Causal DAG graph
- [ ] GBM vs LSTM comparison graph (`POST /compare`)
- [ ] Recommendation path graph

### Phase 4: Recommendation Views (Days 7–8)
- [ ] Student recommendations list (`POST /recommend/student`)
- [ ] Recommendation explain panel (`POST /recommend/student/explain`)
- [ ] Recommendation what-if (`POST /recommend/student/whatif`)

### Phase 5: Instructor Dashboard (Days 8–10)
- [ ] Student list (`GET /auth/students`)
- [ ] Student detail → reuse student panels with instructor audience
- [ ] Intervention recommendations (`POST /recommend/instructor`)
- [ ] Enroll dialog (`POST /auth/enroll`)

### Phase 6: Admin Dashboard (Days 10–11)
- [ ] System health panel (3 health endpoints)
- [ ] Model metrics cards
- [ ] Drift report view
- [ ] Retrain + Reload buttons (with canary_fraction slider)

### Phase 7: Event Tracking + Polish (Days 11–12)
- [ ] `useEventTracker` hook wired to all interactive elements
- [ ] Event buffer + batch flush to `POST /events`
- [ ] Loading skeletons, error boundaries
- [ ] Responsive layout

---

## 9. Monte Carlo Simulation — Frontend Deep Dive

### Endpoint

```
POST /simulate
Body:
{
  features:      LearnerFeatures,   // student's current state
  current_week:  int (1–52),        // default 6
  target_week:   int (2–52),        // default 12, must be > current_week
  n_simulations: int (10–10000)     // default 1000
}
```

**Constraint:** `target_week > current_week` — the server returns 422 otherwise.

**Requires:** `data/temporal/transitions.pkl` must exist (built by `temporal_builder.py`). If missing, the endpoint returns a graceful message body instead of 503 — check for `"message"` key in response.

---

### Full Response Shape

```typescript
interface SimulateResponse {
  current_week:  number;
  target_week:   number;
  n_simulations: number;
  message?:      string;   // only present when transitions.pkl is missing

  feature_distributions: {
    [feature: string]: {
      mean: number;   // projected mean value at target_week
      std:  number;   // spread across simulations
      q10:  number;   // 10th percentile (pessimistic path)
      q50:  number;   // median projection
      q90:  number;   // 90th percentile (optimistic path)
    };
  };

  outcome_distribution: {
    dropout_prob_mean: number;   // mean predicted dropout probability
    dropout_prob_std:  number;
    dropout_prob_q10:  number;   // best-case risk (10th pct)
    dropout_prob_q50:  number;   // median risk
    dropout_prob_q90:  number;   // worst-case risk (90th pct)
    dropout_rate:      number;   // fraction of simulations predicting dropout (prob > 0.5)
  };
}
```

Note: Individual trajectories are stripped server-side before returning — only the statistical summaries are sent.

---

### How the Simulator Works (for UI copy/tooltips)

The simulator samples from empirical feature-change distributions learned from real student data between consecutive course weeks (2→4→6→8→10→12). Each of the N simulations walks the student's current feature vector forward through 2-week steps, applying randomly sampled deltas from students who were at similar points in their course. The GBM model then scores each projected state to produce the outcome distribution.

---

### Recommended UI Components

#### 1. `SimulationSetupPanel`
- **Week range slider** — `current_week` (fixed at learner's actual week from `/auth/me`) → `target_week` (slider: 2–52, snaps to even numbers since steps are 2-week)
- **Simulations count selector** — Radio/select: 100 (fast), 500, 1000 (default), 5000 (precise)
- **Run button** → calls `POST /simulate`, shows spinner

#### 2. `OutcomeDistributionCard`
Primary result card. Shows projected dropout risk range.

```
Visualization: horizontal confidence band (like a weather forecast)

[Current risk: 0.42]  →  [Q10: 0.28 | Q50: 0.51 | Q90: 0.74]  at week 12

Band chart:
  ████░░░░░░░░░░░  Q10  (optimistic)
  ████████░░░░░░░  Q50  (median)
  ████████████░░░  Q90  (pessimistic)
```

Fields to display:
- `dropout_prob_mean` → large number with `risk_label` colour (green/yellow/red)
- `dropout_prob_q10` / `dropout_prob_q90` → "Best case" / "Worst case" labels
- `dropout_rate` → "X% of simulated futures predict dropout"
- `dropout_prob_std` → "Confidence: low/medium/high" band

#### 3. `FeatureProjectionGrid`
Grid of small sparkline/range cards — one per feature. Shows current vs projected range.

```
For each feature in feature_distributions:
  Feature name
  Current value ──●─────────── [Q10 ░░░ Q50 ▓▓▓ Q90]
                  ↑ now        ← projected range at target_week →
```

Prioritise the top features from the last `/explain` call (show those first, collapse the rest).

#### 4. `SimulationInterpretationCard` (LLM Narration)
Feed `outcome_distribution` into the existing `/explain` → `narratives.learner` pattern.
Or display a static template:
- If `dropout_prob_q90 > 0.7` → "Warning: in the most pessimistic scenarios your risk climbs above 70%"
- If `dropout_prob_q10 < 0.3` && `dropout_prob_q90 < 0.6` → "Your risk stays manageable across most simulated paths"

---

### Week Picker Logic

Valid projection targets depend on the student's current week. Use snapping:

```typescript
const WEEK_SLOTS = [2, 4, 6, 8, 10, 12];

function getValidTargetWeeks(currentWeek: number): number[] {
  return WEEK_SLOTS.filter(w => w > currentWeek);
}
// If student is at week 4, they can project to: [6, 8, 10, 12]
// If student is at week 10, they can project to: [12]
// If student is at week 12: simulation not available (show message)
```

---

### Fixture Data for Testing

`data/fixtures/temporal_sequence.json` has a student who declines from week 2 → 12:
- **Week 2:** login=5.5, quiz=74 → low risk
- **Week 6:** login=2.5, quiz=52 → medium risk
- **Week 10:** login=0.5, quiz=31 → high risk

Use the week-8 snapshot as `features` with `current_week=8`, `target_week=12` to demo a declining trajectory with visible risk escalation.

---

### Page Placement

**Student Dashboard → "My Future Risk" tab**

```
[Week slider: ── current_week=8 ──●──────── target_week=12]
[Simulations: ○100  ●1000  ○5000]             [Run Simulation ▶]

┌─────────────────────────────────────────────────────────┐
│ Outcome at Week 12          OutcomeDistributionCard      │
│ Mean: 0.61  ⚠ Medium-High                               │
│ Best case: 0.38   Median: 0.61   Worst case: 0.81        │
│ 58% of simulated futures predict dropout                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Feature Projections          FeatureProjectionGrid       │
│ quiz_avg_score:   42 ──● [28░░░░░38▓▓▓▓▓▓52]           │
│ login_frequency:  1.5──● [0.2░░░0.8▓▓▓▓▓2.1]           │
│ …                                                       │
└─────────────────────────────────────────────────────────┘
```

**Instructor Dashboard → Student Detail → "Simulate" tab**
Same view but instructor controls both `current_week` and `target_week` manually (not locked to the student's actual week).

---

### Checklist Addition (Phase 2 — Student Dashboard)

Add to the existing Phase 2 checklist:
- [ ] `SimulationSetupPanel` — week slider + simulations selector
- [ ] `POST /simulate` wired via `useSimulate` hook
- [ ] `OutcomeDistributionCard` — fan/band chart (recharts AreaChart with Q10/Q50/Q90 series)
- [ ] `FeatureProjectionGrid` — range bars per feature, sorted by last SHAP magnitude
- [ ] `SimulationInterpretationCard` — template text based on Q90 threshold
- [ ] Week picker snapping logic (`getValidTargetWeeks`)
- [ ] Graceful empty state when `transitions.pkl` missing (`message` key check)

---

## 10. Environment Variables

```env
# Frontend .env.local
VITE_API_URL=http://localhost:8000
VITE_APP_TITLE=XAI Learning System

# Backend .env (already exists)
JWT_SECRET_KEY=<your-secret>
GROQ_API_KEY=<for-narration-and-seed-script>
MLOPS_AUTOMATION_TOKEN=<for-admin-retrain-reload>
```

---

## 10. CORS

Already configured in the backend:
```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```
No proxy needed during development.
