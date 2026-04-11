# XAI Learning System — Frontend Integration & New Components Plan

> **For the IDE agent implementing this plan:**
>
> Before writing any code, run these discovery steps IN ORDER:
> 1. List the frontend root directory structure (`ls` / `tree`)
> 2. Read `package.json` to confirm the tech stack (React, Vite, Tailwind, shadcn, recharts, etc.)
> 3. Read `src/App.tsx` (or `App.jsx`) to understand routing
> 4. Read `src/api/` directory to understand the existing API layer
> 5. Read `src/pages/InstructorView.tsx` (or `InstructorDashboard`) to see the current instructor page
> 6. Read `src/pages/StudentView.tsx` to see current student page
> 7. Read `src/components/` tree to inventory existing components
> 8. Cross-reference all component names with this plan before creating new files
>
> **Only then** begin implementing changes in the order defined in each phase.

---

## Part 1 — Backend Updates Summary (Already Implemented)

These changes are DONE in the backend. The frontend must consume the new fields.

### 1.1 Instructor Recommendation Engine (Breaking Change: 9 → 34 features)

The instructor ranker was retrained on a richer dataset. Items sent to
`POST /recommend/instructor` now require 34 features (see full schema in
`docs/FRONTEND_BACKEND_INTEGRATION.md § 1b`).

**Key new instructor-side fields:**
- `instructor_department`, `instructor_teaching_style`, `instructor_experience_years`
- `instructor_intervention_intensity`, `instructor_past_success_rate`, `instructor_confidence_score`

**Key new student-side fields (per item):**
- `student_dropout_risk_score`, `student_risk_trajectory`, `student_learning_momentum`
- `student_quiz_avg_score`, `student_assignment_submission_rate`
- `student_missed_deadlines`, `student_days_inactive`, `student_week_in_course`
- `student_vs_cohort_quiz_delta`, `student_peer_collab_readiness`

**Key new cohort fields (per item):**
- `cohort_avg_quiz_score`, `cohort_avg_dropout_rate`, `cohort_size`, `cohort_engagement_percentile`

**Key new intervention fields (per item):**
- `intervention_type` (`remediation` | `enrichment` | `peer_support` | ...)
- `intervention_urgency` (`high` | `medium` | `low`)
- `recommended_content_type` (`video` | `quiz` | `reflection_questions` | ...)
- `estimated_effort_hours`

### 1.2 New Response Fields on All Recommendation Endpoints

| Endpoint | New Fields |
|----------|-----------|
| `POST /recommend/student` | `diversity_score` (0–1), `diversity_warning` (string\|null), `fairness_audit` |
| `POST /recommend/instructor` | `diversity_score`, `diversity_warning`, `fairness_audit` |
| `POST /recommend/student/explain` | `trust_score`, `anchor_precision`, `prototypes`, `narratives`, `explanation_drift` |
| `POST /recommend/instructor/explain` | All above + `intervention_type`, `intervention_urgency`, `recommended_content_type`, `estimated_effort_hours`, `student_dropout_risk_score`, `student_risk_trajectory`, `instructor_archetype`, `instructor_teaching_style`, `cohort_avg_dropout_rate` |
| `GET /causal/graph` | **NEW endpoint** — `{nodes, edges}` for DAG visualization |

### 1.3 New Endpoint: `GET /causal/graph`

```
GET /causal/graph
Response:
{
  nodes: [
    { id: string, label: string, group: "confounder"|"engagement"|"performance"|"risk_signal"|"outcome",
      is_causal: boolean, ate: number, effect_direction: "positive"|"negative"|"neutral" }
  ],
  edges: [ { from: string, to: string } ]
}
```

---

## Part 2 — API Layer Updates

> **Agent**: Read `src/api/recommend.ts` (or equivalent) before editing.
> Add/update the following typed functions. Do NOT delete existing functions.

### 2.1 Update `recommend.ts`

```typescript
// Update student recommend response type
interface StudentRecoResponse {
  learner_id:        string;
  top_k:             number;
  recommendations:   ScoredItem[];
  diversity_score:   number;
  diversity_warning: string | null;
  fairness_audit?:   FairnessReport;
}

// Update instructor recommend response type
interface InstructorRecoResponse {
  instructor_id:     string;
  top_k:             number;
  recommendations:   ScoredItem[];
  diversity_score:   number;
  diversity_warning: string | null;
  fairness_audit?:   FairnessReport;
}

interface FairnessReport {
  overall_fair:       boolean;
  group_scores:       Record<string, Record<string, number>>;
  flagged_disparities: FlaggedDisparity[];
}

interface FlaggedDisparity {
  feature:       string;
  group:         string;
  score:         number;
  deviation_pct: number;
  direction:     "above" | "below";
}

// Update student explain response type
interface StudentExplainResponse {
  learner_id:           string;
  item_id:              string;
  score:                number;
  shap_values:          Record<string, number>;
  top_features:         string[];
  anchor_rule:          string;
  anchor_precision:     number;
  feature_interactions: FeatureInteraction[];
  causal_annotations:   Record<string, CausalAnnotation>;
  shap_stability:       number;
  plain_language:       string;
  trust_score:          TrustScore | null;
  prototypes:           Prototype[] | null;
  narratives:           Narratives | null;
  explanation_drift:    object | null;
}

// Update instructor explain response type
interface InstructorExplainResponse extends StudentExplainResponse {
  instructor_id:              string;
  intervention_type?:         string;
  intervention_urgency?:      string;
  recommended_content_type?:  string;
  estimated_effort_hours?:    number;
  student_dropout_risk_score?: number;
  student_risk_trajectory?:   string;
  instructor_archetype?:      string;
  instructor_teaching_style?: string;
  cohort_avg_dropout_rate?:   number;
}

interface TrustScore  { trust_score: number; fidelity: number; stability: number; completeness: number; }
interface Prototype   { item_id: string; score: number; similarity: number; outcome?: string; }
interface Narratives  { learner_text: string | null; instructor_text: string | null; tokens_used?: number; }
interface FeatureInteraction { features: [string, string]; strength: number; direction: string; }
interface CausalAnnotation   { type: "causal" | "correlational" | "confounder"; ate: number; }
interface ScoredItem         { item_id: string; score: number; rank: number; top_features: string[]; shap_values?: Record<string, number>; }
```

### 2.2 Add `causal.ts`

```typescript
// src/api/causal.ts
import { apiClient } from "./client";

export interface CausalNode {
  id:               string;
  label:            string;
  group:            "confounder" | "engagement" | "performance" | "risk_signal" | "outcome";
  is_causal:        boolean;
  ate:              number;
  effect_direction: "positive" | "negative" | "neutral";
}

export interface CausalEdge { from: string; to: string; }

export interface CausalGraph { nodes: CausalNode[]; edges: CausalEdge[]; }

export async function getCausalGraph(): Promise<CausalGraph> {
  const { data } = await apiClient.get<CausalGraph>("/causal/graph");
  return data;
}
```

---

## Part 3 — Updated Existing Components

> **Agent**: Read each component file before editing. Add only the new fields.
> Do NOT remove existing functionality.

### 3.1 `RecommendationList` (student + instructor)

**What to add:**
1. Below the list header, add a `DiversityBadge` row:
   - Show `diversity_score` as a percentage pill (e.g. "67% diverse")
   - If `diversity_warning` is not null, show it as a yellow `AlertBanner`
2. If `fairness_audit` is present and `overall_fair === false`:
   - Show a red `FairnessBanner`: "⚠ Fairness issue detected — some groups are scored differently"
   - Show a collapsible `FairnessDetail` list of `flagged_disparities`

**Props delta (add to existing):**
```typescript
diversity_score?:   number;
diversity_warning?: string | null;
fairness_audit?:    FairnessReport;
```

### 3.2 `TrustScoreCard`

**What to add:** Accept full `trust_score` dict, not just scalar stability.
Show a 4-field breakdown:
```
Trust Score: 0.82  ████████░░
  Fidelity:     0.91  ████████████░
  Stability:    0.76  ███████████░░
  Completeness: 0.88  ████████████░
```
Use a compact horizontal bar for each sub-metric.

**Fallback:** If `trust_score` is null or only `shap_stability` is available, show the old single-bar display.

### 3.3 `AnchorRuleCard`

**What to add:** Display `anchor_precision` as a badge next to the rule:
```
IF quiz_avg_score ≤ 55 AND days_inactive > 7 THEN risk = HIGH
                                                  [Precision: 94%]
```

---

## Part 4 — New Components

> **Agent**: Check if a component with a similar name already exists before creating.
> Place all new components in the folder structure that already exists (likely `src/components/`).
> If a `panels/` subdirectory exists, place panel components there.
> If a `graphs/` subdirectory exists, place graph components there.

### 4.1 `NarrativeCard` (Panel)

**Purpose:** Display LLM-generated narration (learner or instructor view).

**File:** `src/components/panels/NarrativeCard.tsx`

**Props:**
```typescript
interface NarrativeCardProps {
  learnerText?:     string | null;
  instructorText?:  string | null;
  audience:         "learner" | "instructor" | "both";
  isLoading?:       boolean;
}
```

**UI:**
- Card with title "AI Explanation" (learner) or "AI Assessment" (instructor)
- Quote icon + italic text body
- If null: show greyed placeholder "Narration not available (LLM offline)"
- If loading: skeleton pulse placeholder

### 4.2 `PrototypesCard` (Panel)

**Purpose:** Show K=3 nearest-neighbour similar learners/items.

**File:** `src/components/panels/PrototypesCard.tsx`

**Props:**
```typescript
interface PrototypesCardProps {
  prototypes: Array<{
    item_id:    string;
    score:      number;
    similarity: number;
    outcome?:   string;
  }>;
}
```

**UI:**
- Title: "Similar Learners" (student context) or "Similar Assignments" (instructor context)
- 3 rows, each showing:
  - Avatar placeholder + `item_id`
  - Similarity bar (0–100%)
  - Score badge
  - `outcome` chip (green="completed" / red="dropped out" / grey=unknown)
- If empty list: "No similar profiles found"

### 4.3 `ExplanationDriftBanner` (Panel)

**Purpose:** Surface drift alerts when a learner's top features shift between sessions.

**File:** `src/components/panels/ExplanationDriftBanner.tsx`

**Props:**
```typescript
interface DriftBannerProps {
  drift: {
    drift_detected:     boolean;
    jsd_score?:         number;
    top3_changed?:      boolean;
    previous_top3?:     string[];
    current_top3?:      string[];
  } | null;
}
```

**UI:**
- If `drift === null` or `drift_detected === false`: render nothing (null)
- If `drift_detected === true`:
  ```
  ⚡ Explanation changed since last session
  Previous drivers: [quiz_avg_score] [days_inactive] [assignment_rate]
  Current drivers:  [days_inactive] [login_freq] [forum_posts]
  ```
  Yellow alert card.

### 4.4 `FairnessAuditPanel` (Panel)

**Purpose:** Show fairness audit results per protected feature.

**File:** `src/components/panels/FairnessAuditPanel.tsx`

**Props:**
```typescript
interface FairnessAuditPanelProps {
  report: FairnessReport;
}
```

**UI:**
- Overall status badge: green "✓ Fair" or red "⚠ Disparities Detected"
- Collapsible sections per protected feature (e.g., `explicit_gender`, `instructor_department`):
  ```
  explicit_gender
    Male:     avg score 0.72  ████████░░
    Female:   avg score 0.61  ███████░░░  ← flagged (-15%)
    Unknown:  avg score 0.69  ████████░░
  ```
- Flag icon next to flagged groups with `deviation_pct`

### 4.5 `InterventionMetaCard` (Panel — Instructor only)

**Purpose:** Display intervention context returned by `/recommend/instructor/explain`.

**File:** `src/components/panels/InterventionMetaCard.tsx`

**Props:**
```typescript
interface InterventionMetaCardProps {
  interventionType?:     string;
  interventionUrgency?:  string;
  contentType?:          string;
  effortHours?:          number;
  studentRiskScore?:     number;
  studentTrajectory?:    string;
  cohortDropoutRate?:    number;
  instructorArchetype?:  string;
  teachingStyle?:        string;
}
```

**UI:**
```
┌─────────────────────────────────────────────┐
│  Intervention Context                        │
│  Type: Remediation  [HIGH URGENCY]           │
│  Content: Reflection Questions  · 2.5h       │
│  ─────────────────────────────────────────── │
│  Student Risk: 0.87  ▓▓▓▓▓▓▓▓▓░  declining  │
│  Cohort Dropout: 30%                         │
│  Instructor: Socratic Guide (socratic)        │
└─────────────────────────────────────────────┘
```
- Urgency badge: red="high", yellow="medium", green="low"
- Risk bar coloured: red > 0.6, yellow 0.3–0.6, green < 0.3
- Trajectory chip with directional arrow

### 4.6 `CausalDagGraph` (Graph — update existing or create)

**Purpose:** Render the causal DAG from `GET /causal/graph` using `ForceGraph3D`.

**File:** `src/components/graphs/CausalDagGraph.tsx`

> **Agent**: Check if `CausalDagGraph.tsx` already exists.
> If yes: update it to use the `/causal/graph` API endpoint instead of
> constructing nodes manually. If no: create it.

**Data fetch:** Call `getCausalGraph()` from `src/api/causal.ts` on mount.

**Node styling:**
```typescript
const GROUP_COLORS = {
  confounder:  "#94a3b8",  // slate
  engagement:  "#60a5fa",  // blue
  performance: "#34d399",  // green
  risk_signal: "#f87171",  // red
  outcome:     "#fbbf24",  // amber
};

nodeVal  = node.is_causal ? Math.abs(node.ate) * 300 + 10 : 6
nodeColor = GROUP_COLORS[node.group]
```

**Edge styling:**
```typescript
linkColor     = "#6b7280"
linkDirectionalArrowLength = 4
linkLineDash  = (link) => !isCausalEdge(link) ? [2, 2] : []
```

**Legend:** Small overlay box with group colour key.

### 4.7 `InstructorStudentCard` (Panel — Instructor only)

**Purpose:** Per-student card in instructor intervention list, replacing the old minimal
`learner_id + score` row with rich student data.

**File:** `src/components/panels/InstructorStudentCard.tsx`

**Props:**
```typescript
interface InstructorStudentCardProps {
  item:        ScoredItem;
  features:    InstructorRecoItem["features"];
  onExplain:   () => void;
  onWhatIf:    () => void;
}
```

**UI:**
```
┌──────────────────────────────────────────────────────────┐
│  [learner_123]        Priority: ████████░░ 0.81           │
│  Risk: 0.87 ▓ HIGH  →  declining                         │
│  Quiz: 55.2 (vs cohort: +12.3)  · Inactive: 11d          │
│  Intervention: remediation [HIGH]  · Video · 2.5h         │
│  [Explain ↗]  [What-If ↗]                                │
└──────────────────────────────────────────────────────────┘
```

---

## Part 5 — Page Integration

### 5.1 Student Dashboard — Recommendation Tab

**File:** Read the existing student recommendations page/component.

**Changes to make:**
1. After fetching `/recommend/student`:
   - Show `DiversityBadge` with `diversity_score`
   - If `diversity_warning`: show `AlertBanner`
   - If `fairness_audit?.overall_fair === false`: show `FairnessAuditPanel`
2. When a student clicks "Explain" on a recommendation item:
   - Call `/recommend/student/explain` with `audience: "learner"`
   - Show new panels below existing explanation:
     - `TrustScoreCard` (full 4-metric breakdown)
     - `NarrativeCard` with `narratives.learner_text`
     - `PrototypesCard` with `prototypes`
     - `ExplanationDriftBanner` with `explanation_drift`
     - `AnchorRuleCard` updated with `anchor_precision`

### 5.2 Instructor Dashboard — Interventions Page

**File:** Read the existing instructor interventions page/component.

**Changes to make:**
1. Replace plain item rows with `InstructorStudentCard` components
2. After fetching `/recommend/instructor`:
   - Show `DiversityBadge` (by intervention_type diversity)
   - If `diversity_warning`: show `AlertBanner`
   - If `fairness_audit?.overall_fair === false`: show `FairnessAuditPanel`
3. When instructor clicks "Explain" on an item:
   - Call `/recommend/instructor/explain` with `audience: "instructor"`
   - Show below existing explanation:
     - `InterventionMetaCard` with all intervention fields
     - `TrustScoreCard` (full breakdown)
     - `NarrativeCard` with `narratives.instructor_text`
     - `PrototypesCard`
     - `AnchorRuleCard` with precision
4. Add a "Causal Graph" tab to the instructor sidebar/nav that renders `CausalDagGraph`

### 5.3 Admin Dashboard — New Sections

Add to the existing Admin page:
1. **Fairness Monitor** section: show aggregate `fairness_audit` from a sample
   `/recommend/student` call (e.g., using the last 10 scored recommendations)
2. **Causal DAG** section: embed `CausalDagGraph`

---

## Part 6 — Request Body Builders (Instructor)

Because the instructor ranker now needs 34 features per item, the frontend
must construct these from whatever data it has about the student and
instructor context. Add these builder utilities:

```typescript
// src/lib/requestBuilders.ts

export function buildInstructorRecoItem(
  studentId:  string,
  learnerId:  string,
  instructor: { archetype: string; department: string; teaching_style: string;
                experience_years: number; avg_cohort_size: number;
                intervention_intensity: number; past_success_rate: number;
                confidence_score: number; },
  student:    { current_module: string; current_presentation: string;
                dropout_risk_score: number; risk_trajectory: string;
                learning_momentum: number; quiz_avg_score: number;
                assignment_submission_rate: number; missed_deadlines: number;
                days_inactive: number; week_in_course: number;
                vs_cohort_quiz_delta: number; peer_collab_readiness: number; },
  target:     { recommended_module: string; recommended_presentation: string;
                affinity_score: number; intervention_type: string;
                intervention_urgency: string; content_type: string;
                effort_hours: number; },
  cohort:     { signal_score: number; avg_quiz_score: number;
                avg_dropout_rate: number; size: number;
                engagement_percentile: number; },
): InstructorRecoItem {
  return {
    item_id: `${learnerId}-${target.recommended_module}`,
    features: {
      instructor_archetype:               instructor.archetype,
      instructor_department:              instructor.department,
      instructor_teaching_style:          instructor.teaching_style,
      instructor_experience_years:        instructor.experience_years,
      instructor_avg_cohort_size:         instructor.avg_cohort_size,
      instructor_intervention_intensity:  instructor.intervention_intensity,
      instructor_past_success_rate:       instructor.past_success_rate,
      instructor_confidence_score:        instructor.confidence_score,
      student_id:                         parseInt(studentId, 10) || 0,
      learner_id:                         learnerId,
      student_current_module:             student.current_module,
      student_current_presentation:       student.current_presentation,
      student_dropout_risk_score:         student.dropout_risk_score,
      student_risk_trajectory:            student.risk_trajectory,
      student_learning_momentum:          student.learning_momentum,
      student_quiz_avg_score:             student.quiz_avg_score,
      student_assignment_submission_rate: student.assignment_submission_rate,
      student_missed_deadlines:           student.missed_deadlines,
      student_days_inactive:              student.days_inactive,
      student_week_in_course:             student.week_in_course,
      student_vs_cohort_quiz_delta:       student.vs_cohort_quiz_delta,
      student_peer_collab_readiness:      student.peer_collab_readiness,
      recommended_module:                 target.recommended_module,
      recommended_presentation:           target.recommended_presentation,
      student_affinity_score:             target.affinity_score,
      cohort_signal_score:                cohort.signal_score,
      cohort_avg_quiz_score:              cohort.avg_quiz_score,
      cohort_avg_dropout_rate:            cohort.avg_dropout_rate,
      cohort_size:                        cohort.size,
      cohort_engagement_percentile:       cohort.engagement_percentile,
      intervention_type:                  target.intervention_type,
      intervention_urgency:               target.intervention_urgency,
      recommended_content_type:           target.content_type,
      estimated_effort_hours:             target.effort_hours,
    },
  };
}
```

---

## Part 7 — Hooks to Add / Update

> **Agent**: Check `src/hooks/` for existing hooks. Update if present, create if absent.

### 7.1 `useInstructorRecommend.ts`

```typescript
// src/hooks/useInstructorRecommend.ts
import { useMutation } from "@tanstack/react-query";
import { postInstructorRecommend, postInstructorExplain } from "../api/recommend";

export function useInstructorRecommend() {
  return useMutation({ mutationFn: postInstructorRecommend });
}

export function useInstructorExplain() {
  return useMutation({ mutationFn: postInstructorExplain });
}
```

### 7.2 `useCausalGraph.ts`

```typescript
// src/hooks/useCausalGraph.ts
import { useQuery } from "@tanstack/react-query";
import { getCausalGraph } from "../api/causal";

export function useCausalGraph() {
  return useQuery({
    queryKey: ["causal-graph"],
    queryFn: getCausalGraph,
    staleTime: Infinity,  // DAG doesn't change at runtime
  });
}
```

### 7.3 Update `useRecommend.ts` (student)

Add `diversity_score`, `diversity_warning`, `fairness_audit` destructuring to wherever
the hook returns the student recommend response.

---

## Part 8 — Implementation Order

Implement phases in this order to minimise broken intermediate states:

```
Phase 1 — Type updates (no UI change, safe first)
  □ Update type definitions in recommend.ts
  □ Create src/api/causal.ts
  □ Create src/lib/requestBuilders.ts with buildInstructorRecoItem

Phase 2 — New small panels (standalone, no page wiring)
  □ NarrativeCard
  □ PrototypesCard
  □ ExplanationDriftBanner
  □ InterventionMetaCard
  □ FairnessAuditPanel
  □ Update TrustScoreCard (full breakdown)
  □ Update AnchorRuleCard (precision badge)

Phase 3 — Hooks
  □ useInstructorRecommend + useInstructorExplain
  □ useCausalGraph
  □ Update useRecommend (diversity/fairness fields)

Phase 4 — Graph
  □ CausalDagGraph (create or update)

Phase 5 — Page wiring
  □ Student recommendation tab — add new panels to explain flow
  □ Instructor interventions page — InstructorStudentCard + new panels
  □ Admin page — Causal DAG tab + Fairness monitor

Phase 6 — Request body migration
  □ Update instructor recommend call site to use buildInstructorRecoItem
  □ Verify all 34 features are present in requests (check browser network tab)

Phase 7 — Polish
  □ Loading skeletons for all new async panels
  □ Error boundary around CausalDagGraph (heavy 3D component)
  □ Mobile responsiveness for new cards
```

---

## Part 9 — Fixture / Mock Data for Development

If the backend is unavailable during frontend development, use these mock
values to develop UI in isolation.

### Mock `FairnessReport`
```json
{
  "overall_fair": false,
  "group_scores": {
    "explicit_gender": {"Male": 0.71, "Female": 0.58, "Unknown": 0.68},
    "instructor_department": {"STEM": 0.76, "Humanities": 0.62, "Social Sciences": 0.70}
  },
  "flagged_disparities": [
    {"feature": "explicit_gender", "group": "Female", "score": 0.58, "deviation_pct": 18.3, "direction": "below"},
    {"feature": "instructor_department", "group": "Humanities", "score": 0.62, "deviation_pct": 16.4, "direction": "below"}
  ]
}
```

### Mock `TrustScore`
```json
{ "trust_score": 0.82, "fidelity": 0.91, "stability": 0.76, "completeness": 0.88 }
```

### Mock `Narratives`
```json
{
  "learner_text": "Based on your recent activity, this module has been selected because your quiz scores and engagement patterns align well with learners who found it helpful. Your next step: complete the first section of the recommended module this week.",
  "instructor_text": "Student learner_142 shows a declining risk trajectory (score 0.87) driven primarily by low assignment submission and extended inactivity. A high-urgency remediation intervention via reflection questions (2.5h effort) is recommended to re-engage before week 12.",
  "tokens_used": 284
}
```

### Mock `Prototypes`
```json
[
  {"item_id": "learner_042", "score": 0.79, "similarity": 0.91, "outcome": "completed"},
  {"item_id": "learner_217", "score": 0.61, "similarity": 0.84, "outcome": "dropped_out"},
  {"item_id": "learner_389", "score": 0.83, "similarity": 0.77, "outcome": "completed"}
]
```

### Mock `ExplanationDrift`
```json
{
  "drift_detected": true,
  "jsd_score": 0.22,
  "top3_changed": true,
  "previous_top3": ["quiz_avg_score", "days_inactive", "assignment_submission_rate"],
  "current_top3": ["days_inactive", "login_frequency_weekly", "forum_posts_count"]
}
```

### Mock Causal Graph (for `CausalDagGraph`)
```json
{
  "nodes": [
    {"id": "prior_course_completions", "label": "Prior Course Completions", "group": "confounder", "is_causal": true, "ate": -0.12, "effect_direction": "negative"},
    {"id": "login_frequency_weekly",  "label": "Login Frequency Weekly",   "group": "engagement", "is_causal": true,  "ate": -0.19, "effect_direction": "negative"},
    {"id": "quiz_avg_score",           "label": "Quiz Avg Score",           "group": "performance","is_causal": false, "ate": -0.08, "effect_direction": "negative"},
    {"id": "days_since_last_activity", "label": "Days Since Last Activity", "group": "risk_signal","is_causal": true,  "ate":  0.24, "effect_direction": "positive"},
    {"id": "dropout_risk",             "label": "Dropout Risk",             "group": "outcome",    "is_causal": true,  "ate":  0.0,  "effect_direction": "neutral"}
  ],
  "edges": [
    {"from": "prior_course_completions", "to": "login_frequency_weekly"},
    {"from": "login_frequency_weekly",   "to": "quiz_avg_score"},
    {"from": "login_frequency_weekly",   "to": "days_since_last_activity"},
    {"from": "days_since_last_activity", "to": "dropout_risk"},
    {"from": "quiz_avg_score",           "to": "dropout_risk"},
    {"from": "login_frequency_weekly",   "to": "dropout_risk"}
  ]
}
```

---

## Part 10 — Environment & CORS

No changes needed. CORS is already `allow_origins=["*"]` in the backend.

```env
# frontend/.env.local (unchanged)
VITE_API_URL=http://localhost:8000
```

---

## Part 11 — Testing Checklist (Verify After Each Phase)

```
□ POST /recommend/student  → response has diversity_score (float), fairness_audit (object|undefined)
□ POST /recommend/instructor → response has diversity_score, fairness_audit, recommendations[0] has score
□ POST /recommend/student/explain → response has trust_score.fidelity, narratives.learner_text, prototypes[0].similarity
□ POST /recommend/instructor/explain → response has intervention_type, intervention_urgency, narratives.instructor_text
□ GET /causal/graph → response has nodes[0].ate (number), edges[0].from (string)
□ NarrativeCard renders LLM text or "not available" fallback
□ PrototypesCard renders 3 rows (or empty state)
□ FairnessAuditPanel shows flagged groups in red
□ TrustScoreCard shows 4 sub-metrics
□ CausalDagGraph loads without crashing (error boundary test: set VITE_API_URL to invalid URL)
□ InstructorStudentCard shows risk badge coloured correctly
□ InterventionMetaCard shows urgency badge (red=high, yellow=medium, green=low)
```
