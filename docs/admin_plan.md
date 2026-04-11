# Admin Dashboard Improvements — Implementation Plan

## Problem Summary

Three bugs were reported, plus a general request for dashboard quality improvements:

1. **Admin Overview redirects to `/student`** — clicking "Overview" in the sidebar takes an admin user to the student dashboard instead of `/dashboard/admin`.
2. **SHAP Fidelity shows "—"** — the Model Metrics card for SHAP Fidelity always displays a dash.
3. **Causal DAG is too complex** — the raw 3D force graph is incomprehensible without context. User requests an LLM explainer (Groq) to narrate the DAG.

Additionally, [AGENT_LEDGER.md](file:///media/smayan/500GB%20SSD/Datahack%204.0/AGENT_LEDGER.md) has unresolved git merge conflict markers that should be cleaned up.

---

## Root Cause Analysis

### Bug 1: Admin sidebar navigates to `/student`

[Sidebar.jsx](file:///media/smayan/500GB%20SSD/Datahack%204.0/frontend/src/components/Sidebar.jsx#L38) line 38:
```js
const links = user.role === 'instructor' ? instructorLinks : studentLinks;
```
This is a binary check: only [instructor](file:///media/smayan/500GB%20SSD/Datahack%204.0/backend/app/main.py#1982-2038) gets its own links; **`admin` falls through to `studentLinks`**. There is no `adminLinks` array defined, and the `studentLinks` array has `{ to: '/student', icon: 'dashboard', label: 'Overview' }` — hence the redirect to `/student`.

Similarly, [Navbar.jsx](file:///media/smayan/500GB%20SSD/Datahack%204.0/frontend/src/components/Navbar.jsx#L34) line 34 only shows "Student View" or "Instructor View" — no admin nav entry exists.

### Bug 2: SHAP Fidelity is null

[main.py](file:///media/smayan/500GB%20SSD/Datahack%204.0/backend/app/main.py#L1253) line 1253:
```python
"shap_fidelity": raw.get("shap_fidelity"),  # optional; None if absent
```
The `training_summary.json` does not contain a top-level `shap_fidelity` key. The frontend's [StatCard](file:///media/smayan/500GB%20SSD/Datahack%204.0/frontend/src/pages/AdminDashboard.jsx#27-39) renders `value={mlopsMetrics?.training?.shap_fidelity?.toFixed(4)}` which evaluates to `undefined`, displaying "—".

### Bug 3: Causal DAG lacks explanation

[CausalDagGraph.jsx](file:///media/smayan/500GB%20SSD/Datahack%204.0/frontend/src/components/graphs/CausalDagGraph.jsx) renders only a raw `ForceGraph3D` with a small color legend. There is no textual summary of what the DAG means or what its key insights are.

---

## Proposed Changes

### 1. Frontend — Admin sidebar & navbar

#### [MODIFY] [Sidebar.jsx](file:///media/smayan/500GB%20SSD/Datahack%204.0/frontend/src/components/Sidebar.jsx)
- Add an `adminLinks` array with admin-specific entries:
  - Overview → `/dashboard/admin`
  - Student View → `/student`
  - Instructor View → `/instructor`
  - What-If Explorer → `/what-if`
  - Model Compare → `/xai/compare`
- Update the ternary on line 38 to a 3-way check: `admin` → `adminLinks`, [instructor](file:///media/smayan/500GB%20SSD/Datahack%204.0/backend/app/main.py#1982-2038) → `instructorLinks`, else `studentLinks`.

#### [MODIFY] [Navbar.jsx](file:///media/smayan/500GB%20SSD/Datahack%204.0/frontend/src/components/Navbar.jsx)
- Add an `admin` branch (line 34) that shows a "Admin Dashboard" nav button pointing to `/dashboard/admin`, plus quick links to Student/Instructor views.

---

### 2. Backend — Compute SHAP fidelity at training time

#### [MODIFY] [main.py](file:///media/smayan/500GB%20SSD/Datahack%204.0/backend/app/main.py)
- In the `/mlops/metrics` endpoint, if `shap_fidelity` is `None`, compute it on-the-fly from the loaded GBM model + a sample of training data:
  ```
  fidelity = 1 - mean(abs(sum(shap_values, axis=1) + base_value - model.predict_proba(X)[:, 1]))
  ```
- Cache the result in `state` so it's only computed once per model load.

---

### 3. Frontend + Backend — LLM-narrated Causal DAG explanation

#### [NEW] Backend endpoint `POST /causal/explain`
- In [main.py](file:///media/smayan/500GB%20SSD/Datahack%204.0/backend/app/main.py), add a new endpoint that:
  1. Fetches the current causal graph data (reuses the `/causal/graph` logic).
  2. Sends the graph structure (nodes with ATE values, groups, edges) to the Groq LLM via the existing `LLMNarrator`.
  3. Returns a structured JSON with: `summary` (2-3 sentence overview), `key_insights` (list of bullet points), `strongest_drivers` (top features by ATE).

#### [MODIFY] [CausalDagGraph.jsx](file:///media/smayan/500GB%20SSD/Datahack%204.0/frontend/src/components/graphs/CausalDagGraph.jsx)
- Add a "✨ Explain This Graph" button below the graph.
- On click, call `POST /causal/explain`.
- Render the LLM response in a styled card below the graph with the summary, key insights as bullet points, and strongest driver highlights.

#### [MODIFY] [causal.js](file:///media/smayan/500GB%20SSD/Datahack%204.0/frontend/src/api/causal.js)
- Add `explainCausalGraph()` API function calling `POST /causal/explain`.

---

### 4. Cleanup — Merge conflict markers

#### [MODIFY] [AGENT_LEDGER.md](file:///media/smayan/500GB%20SSD/Datahack%204.0/.agents/logs/AGENT_LEDGER.md)
- Remove the `<<<<<<< HEAD`, `=======`, `>>>>>>> 741a24ed...` conflict markers on lines 22-29.

---

## Verification Plan

### Automated
- Start the dev server (`npm run dev`) and verify:
  - Login as `demo_admin` → lands on `/dashboard/admin` ✓
  - Sidebar shows admin-specific links ✓
  - SHAP Fidelity card shows a numeric value instead of "—" ✓
  - Causal DAG "Explain" button returns and renders an LLM narrative ✓

### Manual
- Visual review of the admin sidebar and navbar layout.
- Confirm the LLM explanation is coherent and correctly references the DAG's strongest causal drivers.
