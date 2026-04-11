# Frontend → Backend Integration Plan

> **Rule**: Backend is frozen. All changes are frontend-only.
> **Backend base URL**: `http://localhost:8000` (FastAPI, CORS `allow_origins=["*"]`)
> **Frontend stack**: React 18 + Vite + TailwindCSS + Material Design 3 tokens

---

## 0. Executive Summary

The frontend is a fully mocked prototype — every service function returns hardcoded data via `setTimeout`. Auth uses fake `loginMock`/`signupMock` that never hit the network. No HTTP client exists. Every page consumes mock shapes that **do not match** the real backend response schemas.

This plan rewires the frontend to make real HTTP calls to the FastAPI backend, adapts every page to consume the real response shapes, and adds the missing implicit-event telemetry layer the backend expects.

---

## 1. Infra / Plumbing (files to create or modify)

### 1.1 Create `frontend/src/services/api.js` — Axios instance

```
Purpose: Centralized HTTP client with baseURL, JWT interceptor, error handling.
```

- Install `axios` (`npm i axios`).
- Create shared instance:
  - `baseURL` from `VITE_API_URL` env var (default `http://localhost:8000`).
  - Request interceptor: attach `Authorization: Bearer <token>` from `localStorage('ll_token')`.
  - Response interceptor: on 401 → clear token + redirect to `/login`.
- Export `api` instance.

### 1.2 Create `frontend/.env.development`

```
VITE_API_URL=http://localhost:8000
```

### 1.3 Add Vite proxy (optional, avoids CORS in dev)

In `vite.config.js`, add:
```js
server: {
  proxy: {
    '/api': { target: 'http://localhost:8000', changeOrigin: true, rewrite: p => p.replace(/^\/api/, '') }
  }
}
```
If proxy is used, `baseURL` becomes `/api`. Otherwise, direct calls work because backend has `allow_origins=["*"]`.

---

## 2. Auth Layer — Replace Mock with Real JWT

### 2.1 Rewrite `authService.js`

| Current (mock) | Target (real) |
|---|---|
| `loginMock(email, password)` → hardcoded user object | `POST /auth/login` with `{ username, password }` → `TokenResponse` |
| `signupMock(name, email, password, role)` → fake user | `POST /auth/register` with `RegisterRequest` → `RegisterResponse` |

**Backend contract**:
- `POST /auth/login` expects `{ username: str, password: str }` — **NOT email**. The frontend Login form uses `email` as the field name but the backend `LoginRequest` uses `username`. Either:
  - (a) Change the frontend field label from "Email" to "Username", or
  - (b) Send `form.email` as the `username` field (since the backend treats it as a lookup key).
  - **Decision**: Option (b) — keep the UI label as "Email" but map `form.email → username` in the request body. The backend `User` model stores both username and email; login is by username.

- `POST /auth/login` returns:
  ```json
  {
    "access_token": "eyJ...",
    "token_type": "bearer",
    "expires_in": 3600,
    "role": "student",
    "user_id": "uuid",
    "learner_id": "L12345"   // null for instructors
  }
  ```

New `authService.js` exports:
```js
login(username, password)  → POST /auth/login → store token + return user info
register(payload)          → POST /auth/register
getMe()                    → GET /auth/me (uses JWT)
getStudents(courseId?)      → GET /auth/students?course_id=... (instructor-only)
```

### 2.2 Rewrite `AuthContext.jsx`

| Current | Target |
|---|---|
| Stores `ll_mock_user` in localStorage (full user object) | Store `ll_token` (JWT string) + `ll_user` (parsed from login response) |
| `login()` calls `loginMock()` | `login()` calls `authService.login()`, stores `access_token` |
| `signup()` calls `signupMock()` | `signup()` calls `authService.register()` + auto-login |
| `logout()` removes `ll_mock_user` | `logout()` removes `ll_token` + `ll_user` |
| No token refresh | Add: on mount, if token exists, call `GET /auth/me` to validate + hydrate user |

User shape stored in context:
```js
{
  id: "uuid",
  username: "alex_j",
  email: "alex@example.com",
  full_name: "Alex Johnson",
  role: "student",
  learner_id: "L12345",       // from login response or /auth/me
  learner_profile: { ... },   // from /auth/me (optional hydration)
}
```

### 2.3 Update `Login.jsx`

- Field mapping: send `{ username: form.email, password: form.password }` to `POST /auth/login`.
- Add error handling (show toast/alert on 401).
- Dev quick-fill buttons: change to use valid test usernames/passwords that match seeded data.
- Consider adding a Registration link/form (currently missing).

---

## 3. XAI Service Layer — Replace Mocks with Real Endpoints

### 3.1 Rewrite `xaiService.js`

Every function must be rewritten. Below is the full mapping:

#### 3.1.1 `getStudentData()` → Composite of `/explain` + `/predict`

**Current mock returns**:
```js
{ student, riskScore, riskLevel, trustScore, trustLevel, topAction,
  shapFeatures, featureContributions, conceptAnalysis, lookalikes, aiNarrative }
```

**Real backend**: Call `POST /explain` with:
```json
{
  "features": { <LearnerFeatures — 15 fields> },
  "learner_id": "<from auth context>",
  "model": "gbm",
  "audience": "learner"
}
```

Response contains all necessary data but in **different shapes**:

| Frontend mock field | Backend /explain field | Transform needed |
|---|---|---|
| `riskScore` (0-100 int) | `risk_score` (0.0-1.0 float) | `Math.round(risk_score * 100)` |
| `riskLevel` ("HIGH") | `risk_label` ("high") | `.toUpperCase()` |
| `trustScore` (0.92) | `trust_score.trust_score` (0.0-1.0) | Direct |
| `trustLevel` ("HIGH") | `trust_score.label` ("High") | `.toUpperCase()` |
| `topAction` | `ranked_actions[0]` | Map `{ label: plain_language, riskReduction: estimated_impact * 100 }` |
| `shapFeatures[]` | `top_features[]` | Map: `{ name: tf.name, value: abs(tf.shap), direction: tf.direction === 'risk' ? 'increases' : 'decreases' }` |
| `featureContributions[]` | `causal_annotations[]` (where `causal_type === 'causal'`) | Map similarly |
| `conceptAnalysis[]` | `interactions[]` (Archipelago) | Map: `{ name: ix.features.join(' × '), value: abs(ix.interaction_value), direction }` |
| `lookalikes[]` | `prototypes.matches[]` | Map: `{ id: m.learner_id, match: Math.round(m.similarity * 100), outcome: m.outcome.toUpperCase().replace('_', ' ') }` |
| `aiNarrative` | `narratives.learner` or `narratives.instructor` | Direct string |

**Key issue**: The frontend currently doesn't know the learner's features. Options:
1. Use `POST /explain/me` (JWT-based, pulls features from profile) — simplest for students.
2. Fetch learner profile via `GET /auth/me`, construct `LearnerFeatures` from profile + defaults, then call `POST /explain`.
3. **Decision**: For students, call `POST /explain/me` first to get the latest snapshot, then call `POST /explain` with full features for a richer response. If no prior explanation exists, show an onboarding state.

#### 3.1.2 `getBaselineFeatures()` → Static feature metadata + learner data

**Current mock**: Returns 5 simplified features with sliders.

**Real backend**: The `LearnerFeatures` schema has 12 mutable + 3 latent = 15 fields. For What-If, only the **8 actionable features** should be surfaced as sliders:

```
login_frequency_weekly      (0-14)
avg_session_duration_min    (0-300)
forum_posts_count           (0-∞, int)
video_completion_rate       (0-1)
quiz_completion_rate        (0-1)
assignment_submission_rate  (0-1)
days_since_last_activity    (0-∞, int)
help_requests_count         (0-∞, int)
```

The immutable features (`prior_course_completions`, `current_week_in_course`, `quiz_avg_score`, `engagement_latent_1/2/3`) are locked and should be displayed as read-only context.

**Source of current values**: The learner's last `/explain` response contains `shap_values` dict which has all 15 feature keys. Or call `GET /auth/me` → `learner_profile` for basic profile data + use last explanation features.

**Transform**: Build a feature metadata array client-side with labels, min/max, units, and populate `value` from the learner's current features.

#### 3.1.3 `simulateWhatIf(sliders)` → `POST /whatif`

**Current mock**: Client-side math with hardcoded coefficients.

**Real backend** `POST /whatif`:
```json
{
  "features": { <full LearnerFeatures — all 15 fields> },
  "overrides": { "forum_posts_count": 5, "days_since_last_activity": 3 }
}
```

Response:
```json
{
  "shap_values": { "feature": 0.123, ... },
  "base_value": 0.45,
  "risk_score": 0.62,
  "risk_delta": -0.08,
  "top_features": [{ "name": "...", "shap": 0.1, "direction": "risk" }]
}
```

**Transform**: 
- `newRisk` = `Math.round(risk_score * 100)`
- `riskDelta` = `risk_delta` (already a float delta, convert to percentage)
- `impactFactors` = derive from `shap_values` delta vs baseline SHAP

**Requirement**: The frontend must hold the learner's full `LearnerFeatures` state to send as `features`, with only the slider-changed fields in `overrides`.

#### 3.1.4 `getActionPlan()` → from `/explain` response

**Current mock**: Returns `{ actions[], combinedRiskReduction, finalSuccessProbability }`.

**Real backend**: The `/explain` response already contains `ranked_actions[]` and `counterfactual`.

Map:
```js
ranked_actions → actions.map(ra => ({
  id:              ra.feature,
  priority:        `P${ra.priority_rank}`,
  type:            causal_annotations.find(ca => ca.name === ra.feature)?.causal_type?.toUpperCase() || 'CORRELATED',
  feature:         ra.feature,
  current:         ra.current_value,
  target:          ra.target_value,
  recommendation:  ra.plain_language,
  riskReduction:   Math.round(ra.estimated_impact * 100),
}))
```

`combinedRiskReduction` = sum of all `estimated_impact` values (capped at risk_score).
`finalSuccessProbability` = `Math.round((1 - risk_score + combinedRiskReduction) * 100)`.

#### 3.1.5 `getInstructorData()` → Composite of multiple endpoints

**Current mock returns**: `{ classStats, shapFeatures, anchorRule, modelComparison, trustMetrics, interventions }`.

**Real backend mapping**:

| Mock field | Backend source | Endpoint |
|---|---|---|
| `classStats.riskScore` | Average risk across enrolled students | `GET /auth/students` → iterate + `POST /predict` per student, or single aggregate call |
| `classStats.modelConfidence` | `GET /mlops/metrics` → `training.test_auc_roc` |
| `classStats.stabilityScore` | From a representative `/explain` call → `stability` |
| `classStats.fidelityScore` | From a representative `/explain` call → `trust_score.fidelity` |
| `shapFeatures[]` | `/explain` response → `top_features[]` + `causal_annotations[]` | Map: add `type: causal_annotation.causal_type` |
| `anchorRule` | `/explain` response → `anchor_rule.human_readable` |
| `modelComparison` | `GET /mlops/metrics` → `training.gbm` vs `training.rf` metrics |
| `trustMetrics` | `/explain` response → `trust_score` object |
| `interventions[]` | `/explain` response → `ranked_actions[]` | Same mapping as action plan |

**Design decision**: For the instructor dashboard, we need a "representative" student. Options:
- (a) Let instructor select a student from `GET /auth/students` list, then call `/explain` for that student.
- (b) Show aggregated class-level stats from `/mlops/metrics` + `/mlops/health`, and per-student drill-down.
- **Decision**: (b) for stats cards + (a) for XAI details. Add a student selector dropdown.

#### 3.1.6 New: `submitFeedback()` → `POST /feedback`

Not in the mock but the backend supports it. Add:
```js
submitFeedback({ learner_id, rating, followed_recommendation, correction_feature, correction_comment })
```

#### 3.1.7 New: `sendEvents(events)` → `POST /events`

The backend expects implicit event telemetry. Add:
```js
sendEvents(events: EventPayload[])  → POST /events { events: [...] }
```

Event types: `page_view`, `click`, `scroll`, `whatif_slider`, `action_viewed`, `action_dismissed`, `explanation_revisit`, `resource_download`, `time_to_first_action`, `prototype_click`, `session_start`, `session_end`.

#### 3.1.8 New: `getHistory(learnerId)` → `GET /history/{learner_id}`

Returns explanation timeline + drift flags.

#### 3.1.9 New: `getMlopsHealth()` → `GET /mlops/health`

For instructor dashboard status panel.

---

## 4. Page-by-Page Changes

### 4.1 `Login.jsx`

| # | Change | Priority |
|---|---|---|
| 1 | Replace `loginMock` call with `authService.login(username, password)` | P0 |
| 2 | Map form field: `form.email` → sent as `username` in request body | P0 |
| 3 | Store `access_token` in `localStorage('ll_token')` | P0 |
| 4 | Add error state + display for 401/409 errors | P0 |
| 5 | Update dev quick-fill buttons to use seeded test credentials | P1 |
| 6 | Add optional "Register" link/flow | P2 |

### 4.2 `StudentDashboard.jsx`

| # | Change | Priority |
|---|---|---|
| 1 | Replace `getStudentData()` with real `/explain` call | P0 |
| 2 | Build `LearnerFeatures` from auth profile or prompt user to input | P0 |
| 3 | Map `risk_score` (0-1) → percentage display | P0 |
| 4 | Map `top_features[]` → `shapFeatures[]` shape expected by tab renderer | P0 |
| 5 | Map `causal_annotations[]` → `featureContributions[]` for "Feature" tab | P0 |
| 6 | Map `interactions[]` → `conceptAnalysis[]` for "Concept" tab | P0 |
| 7 | Map `prototypes.matches[]` → `lookalikes[]` | P0 |
| 8 | Map `narratives.learner` → `aiNarrative` | P1 |
| 9 | Replace "Narrated by Claude AI" text → "Narrated by Groq AI" (backend uses Groq) | P1 |
| 10 | Handle loading/error states for API calls | P0 |
| 11 | Add feedback widget (star rating + follow toggle) → `POST /feedback` | P2 |
| 12 | Fire `page_view` event on mount, `session_start`/`session_end` | P2 |

### 4.3 `InstructorDashboard.jsx`

| # | Change | Priority |
|---|---|---|
| 1 | Replace `getInstructorData()` with real API calls | P0 |
| 2 | Add student selector: `GET /auth/students` → dropdown | P0 |
| 3 | For selected student: `POST /explain` with `audience: "instructor"` | P0 |
| 4 | Stats cards: pull from `/mlops/metrics` + `/mlops/health` | P0 |
| 5 | Map `top_features[]` + `causal_annotations[]` → SHAP feature chart with causal/correlated badges | P0 |
| 6 | Map `anchor_rule.human_readable` → anchor rule display | P0 |
| 7 | Map `trust_score` → trust metrics breakdown (values are 0-1 floats, not percentages — multiply by 100) | P0 |
| 8 | Map `ranked_actions[]` → interventions table | P0 |
| 9 | Model comparison: pull from `/mlops/metrics` → `training.gbm` vs `training.rf` | P1 |
| 10 | Add drift status badge from `/mlops/health` → `drift_status` | P2 |

### 4.4 `WhatIfExplorer.jsx`

| # | Change | Priority |
|---|---|---|
| 1 | Replace mock `simulateWhatIf()` with `POST /whatif` | P0 |
| 2 | Build slider set from real feature names (8 actionable features) | P0 |
| 3 | Hold full `LearnerFeatures` state; send as `features` with slider values as `overrides` | P0 |
| 4 | Map response: `risk_score` (0-1) → percentage, `risk_delta` → delta display | P0 |
| 5 | Map `top_features[]` → impact trajectory chart | P0 |
| 6 | Replace hardcoded "Model Confidence 0.94" with real `trust_score` from last `/explain` | P1 |
| 7 | Fire `whatif_slider` events to `POST /events` on each slider change (debounced) | P2 |
| 8 | Remove hardcoded "STU-8492" → show actual selected learner ID | P0 |
| 9 | Adjust slider min/max to match `LearnerFeatures` Field constraints | P0 |

### 4.5 `ActionPlan.jsx`

| # | Change | Priority |
|---|---|---|
| 1 | Replace `getStudentData()` with data from `/explain` response (reuse cached) | P0 |
| 2 | Render `ranked_actions[]` dynamically instead of hardcoded cards | P0 |
| 3 | Map each `RankedAction` → card with `plain_language`, `current_value`, `target_value`, `estimated_impact` | P0 |
| 4 | Top priority card = `ranked_actions[0]`, secondary = rest | P0 |
| 5 | Fire `action_viewed` event when user views an action card | P2 |
| 6 | Add "I followed this" toggle → `POST /feedback` with `followed_recommendation: true` | P2 |

### 4.6 `Navbar.jsx`

| # | Change | Priority |
|---|---|---|
| 1 | Display `user.full_name` instead of `user.name` (backend field name) | P0 |
| 2 | Display `user.username` or `user.email` as subtitle | P1 |

### 4.7 `Sidebar.jsx`

| # | Change | Priority |
|---|---|---|
| 1 | Wire `#history` link to a real `/history` route (or modal) using `GET /history/{learner_id}` | P1 |
| 2 | Display `user.full_name` instead of `user.name` | P0 |

---

## 5. New Files to Create

| File | Purpose |
|---|---|
| `frontend/src/services/api.js` | Axios instance with JWT interceptor |
| `frontend/src/services/eventTracker.js` | Implicit event batching + `POST /events` (debounced, max 50 events per batch) |
| `frontend/src/hooks/useExplainData.js` | Custom hook: calls `/explain`, transforms response, caches per learner_id |
| `frontend/src/hooks/useMlopsHealth.js` | Custom hook: calls `/mlops/health` + `/mlops/metrics`, returns combined stats |
| `frontend/.env.development` | `VITE_API_URL=http://localhost:8000` |

---

## 6. Data Shape Compatibility Matrix

### 6.1 `LearnerFeatures` — The Critical Contract

The backend `POST /predict`, `POST /explain`, `POST /whatif`, `POST /simulate` ALL require a `LearnerFeatures` object. The frontend must be able to construct this.

```
LearnerFeatures {
  login_frequency_weekly:     float (0-14)
  avg_session_duration_min:   float (0-300)
  forum_posts_count:          int   (≥0)
  video_completion_rate:      float (0-1)
  quiz_avg_score:             float (0-100)
  quiz_completion_rate:       float (0-1)
  assignment_submission_rate: float (0-1)
  days_since_last_activity:   int   (≥0)
  prior_course_completions:   int   (≥0)
  current_week_in_course:     int   (1-52)
  missed_deadlines_count:     int   (≥0)
  help_requests_count:        int   (≥0)
  engagement_latent_1:        float (default 0.0)
  engagement_latent_2:        float (default 0.0)
  engagement_latent_3:        float (default 0.0)
}
```

**Problem**: The frontend currently has NO mechanism to obtain these features for a student.
**Solution**: Two approaches:
1. **For students**: Use `POST /explain/me` which auto-pulls from the last explanation record. For first-time users, show an onboarding form to enter initial features, or pull from a seeded profile.
2. **For instructors**: Select a student → use `GET /history/{learner_id}` to get last features from stored explanation, or provide a feature input form.

### 6.2 Backend Response → Frontend Display Transforms

| Backend field | Type | Frontend display | Transform |
|---|---|---|---|
| `risk_score` | 0.0-1.0 | "78%" | `Math.round(v * 100)` |
| `risk_label` | "high"/"medium"/"low" | "HIGH" | `.toUpperCase()` |
| `trust_score.trust_score` | 0.0-1.0 | "0.92" | Direct or `(v * 100).toFixed(0) + '%'` |
| `trust_score.label` | "High"/"Medium"/"Low" | "HIGH" | `.toUpperCase()` |
| `trust_score.fidelity` | 0.0-1.0 | "91%" | `Math.round(v * 100)` |
| `trust_score.stability` | 0.0-1.0 | "94%" | `Math.round(v * 100)` |
| `trust_score.completeness` | 0.0-1.0 | "88%" | `Math.round(v * 100)` |
| `top_features[].shap` | float | bar width | `Math.abs(v)` scaled to max |
| `top_features[].direction` | "risk"/"protective" | "increases"/"decreases" | Map |
| `prototypes.matches[].similarity` | 0.0-1.0 | "95%" | `Math.round(v * 100)` |
| `prototypes.matches[].outcome` | "dropped_out"/"completed" | "DROPPED OUT" | `.toUpperCase().replace('_', ' ')` |
| `ranked_actions[].estimated_impact` | 0.0-1.0 float | "-12% Risk Drop" | `Math.round(v * 100)` |
| `anchor_rule.human_readable` | string | Displayed as-is | Direct |

---

## 7. Event Telemetry Layer (New)

### 7.1 `eventTracker.js` Design

```
- Buffer events in memory (max 50).
- Flush every 10 seconds OR when buffer hits 50 OR on page unload.
- Each event: { learner_id, session_id, event_type, event_target, event_value, page, client_ts }
- session_id: generated once per browser session (sessionStorage).
- learner_id: from AuthContext.
```

### 7.2 Events to Fire

| Page | Event | Trigger |
|---|---|---|
| All | `session_start` | App mount / tab focus |
| All | `session_end` | App unmount / tab blur |
| All | `page_view` | Route change |
| StudentDashboard | `explanation_revisit` | Component re-mount |
| StudentDashboard | `scroll` | Scroll depth checkpoint (25%, 50%, 75%, 100%) |
| WhatIfExplorer | `whatif_slider` | Slider change (debounced 300ms) |
| ActionPlan | `action_viewed` | Action card enters viewport |
| ActionPlan | `action_dismissed` | User clicks "dismiss" (if added) |
| Prototypes | `prototype_click` | User clicks a prototype match |

---

## 8. Implementation Order (Recommended)

### Phase A — Auth + Infra (Must do first)
1. Create `api.js` (Axios instance + JWT interceptor)
2. Create `.env.development`
3. Rewrite `authService.js` → real `POST /auth/login`, `POST /auth/register`, `GET /auth/me`
4. Rewrite `AuthContext.jsx` → JWT-based auth, token storage, user hydration
5. Update `Login.jsx` → real auth flow + error handling
6. Update `Navbar.jsx` + `Sidebar.jsx` → use `user.full_name` / `user.username`

### Phase B — Core XAI Integration (Main value)
7. Create `useExplainData.js` hook — calls `POST /explain`, transforms response
8. Rewrite `xaiService.js` — all real API calls + response transforms
9. Rewire `StudentDashboard.jsx` — consume transformed `/explain` data
10. Rewire `ActionPlan.jsx` — consume `ranked_actions` from `/explain`
11. Create `useMlopsHealth.js` hook — calls `/mlops/health` + `/mlops/metrics`
12. Rewire `InstructorDashboard.jsx` — student selector + real API calls
13. Rewire `WhatIfExplorer.jsx` — real `POST /whatif` with full `LearnerFeatures`

### Phase C — Telemetry + Polish
14. Create `eventTracker.js` — implicit event batching
15. Wire events into all pages
16. Add feedback widget (star rating) → `POST /feedback`
17. Wire History sidebar link → `GET /history/{learner_id}`
18. Add error boundaries + loading skeletons
19. Add a registration page/modal

---

## 9. Risk & Dependency Notes

| Risk | Mitigation |
|---|---|
| Backend not running during frontend dev | Keep mock fallback mode: if API call fails, fall back to mock data with a banner "Using demo data" |
| First-time student has no features | Use `POST /explain/me` → if 404, show onboarding form with sensible defaults |
| Instructor has no enrolled students | Show empty state with "Enroll students via /auth/enroll" instructions |
| Event telemetry volume | Batch + debounce; 50 max per flush; backend accepts max 500 per call |
| LearnerFeatures construction | Persist last-known features in `sessionStorage` after first `/explain` call; re-use for `/whatif` |
| Token expiry | Interceptor catches 401 → redirect to login. Token TTL from backend = `EXPIRE_MINUTES * 60` seconds |

---

## 10. Files Changed Summary

| File | Action | Phase |
|---|---|---|
| `frontend/src/services/api.js` | **CREATE** | A |
| `frontend/.env.development` | **CREATE** | A |
| `frontend/src/services/authService.js` | **REWRITE** | A |
| `frontend/src/context/AuthContext.jsx` | **REWRITE** | A |
| `frontend/src/pages/Login.jsx` | **MODIFY** | A |
| `frontend/src/components/Navbar.jsx` | **MODIFY** | A |
| `frontend/src/components/Sidebar.jsx` | **MODIFY** | A |
| `frontend/src/services/xaiService.js` | **REWRITE** | B |
| `frontend/src/hooks/useExplainData.js` | **CREATE** | B |
| `frontend/src/hooks/useMlopsHealth.js` | **CREATE** | B |
| `frontend/src/pages/StudentDashboard.jsx` | **MODIFY** | B |
| `frontend/src/pages/ActionPlan.jsx` | **MODIFY** | B |
| `frontend/src/pages/InstructorDashboard.jsx` | **MODIFY** | B |
| `frontend/src/pages/WhatIfExplorer.jsx` | **MODIFY** | B |
| `frontend/src/services/eventTracker.js` | **CREATE** | C |
| `frontend/vite.config.js` | **MODIFY** (proxy) | A |
| `frontend/package.json` | **MODIFY** (add axios) | A |

**Total: 6 new files, 11 modified files, 0 backend changes.**
