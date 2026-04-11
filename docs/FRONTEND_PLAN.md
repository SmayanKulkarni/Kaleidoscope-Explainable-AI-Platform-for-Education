# XAI Frontend — 3D Explainability Dashboard

React + Vite frontend with 3D force-graph explainability views and core XAI panels, wired to the existing FastAPI backend.

---

## Stack

| Tool | Purpose |
|------|---------|
| React 18 + Vite | App framework + dev server |
| TailwindCSS + shadcn/ui | Styling + UI components |
| react-force-graph (`ForceGraph3D`) | 3D WebGL graph rendering (Three.js) |
| react-query (`@tanstack/react-query`) | API data fetching + caching |
| recharts | 2D charts (SHAP bars, risk timeline) |
| react-router-dom | Client-side routing |

---

## Folder Structure

```
frontend/
├── index.html
├── vite.config.ts
├── tailwind.config.ts
├── package.json
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── api/
│   │   ├── client.ts          # axios instance pointing at VITE_API_URL
│   │   ├── dropout.ts         # /predict  /explain  /compare  /whatif
│   │   └── recommend.ts       # /recommend/student  /recommend/instructor
│   ├── components/
│   │   ├── graphs/
│   │   │   ├── ShapInteractionGraph.tsx    # 3D — SHAP feature interaction network
│   │   │   ├── CausalDagGraph.tsx          # 3D — causal DAG
│   │   │   ├── ModelCompareGraph.tsx       # 3D — GBM vs LSTM bipartite graph
│   │   │   └── RecommendPathGraph.tsx      # 3D — student → features → resources
│   │   ├── panels/
│   │   │   ├── RiskScoreCard.tsx           # risk score + label + gauge
│   │   │   ├── TopFeaturesBar.tsx          # horizontal SHAP bar chart
│   │   │   ├── AnchorRuleCard.tsx          # IF-THEN rule display
│   │   │   ├── WhatIfForm.tsx              # feature override form + delta display
│   │   │   ├── TrustScoreCard.tsx          # stability + trust band
│   │   │   └── RecommendationList.tsx      # scored items with top-3 features
│   │   └── layout/
│   │       ├── Sidebar.tsx
│   │       ├── PageHeader.tsx
│   │       └── GraphCard.tsx              # shared wrapper: title + 3D graph + legend
│   ├── pages/
│   │   ├── StudentView.tsx                # dropout risk + recommendation for a student
│   │   ├── InstructorView.tsx             # intervention recommendations for instructors
│   │   └── ComparePage.tsx                # GBM vs LSTM comparison
│   ├── hooks/
│   │   ├── useExplain.ts                  # calls POST /explain
│   │   ├── useCompare.ts                  # calls POST /compare
│   │   └── useRecommend.ts                # calls POST /recommend/student or /instructor
│   └── lib/
│       ├── graphTransforms.ts             # API response → {nodes, links} for ForceGraph3D
│       └── colors.ts                      # shared color palette (causal/correlational/etc.)
```

---

## API Endpoints the Frontend Consumes

Base URL: `VITE_API_URL` (env var, e.g. `http://localhost:8000`)

| Endpoint | Used by |
|----------|---------|
| `POST /predict` | RiskScoreCard |
| `POST /explain` | TopFeaturesBar, AnchorRuleCard, TrustScoreCard, ShapInteractionGraph, CausalDagGraph |
| `POST /whatif` | WhatIfForm |
| `POST /compare` | ModelCompareGraph |
| `POST /recommend/student` | RecommendationList, RecommendPathGraph |
| `POST /recommend/student/explain` | ShapInteractionGraph (recommendation context) |
| `POST /recommend/student/whatif` | WhatIfForm (recommendation context) |
| `POST /recommend/instructor` | InstructorView → RecommendationList |
| `GET /recommend/health` | Sidebar model status badge |

---

## The 4 Three.js Graph Components

### 1. `ShapInteractionGraph` — Feature Interaction Network
**API:** `POST /explain` or `POST /recommend/student/explain`

```
Nodes  = features
  size   = |shap_value| × 100
  color  = causal_annotations: green=causal / yellow=correlational / grey=unknown

Links  = feature_interactions[]
  width  = interaction.strength × 500
  color  = synergistic → #34d399  |  opposing → #f87171
  arrows = directional
```

### 2. `CausalDagGraph` — Causal DAG
**API:** `POST /explain` → `causal_annotations` + `shap_values`

```
Nodes  = features + 1 outcome node ("Dropout Risk")
  color  = same causal palette
  size   = |shap_value|

Links  = feature → outcome (all), styled by causal label
  solid   = causal
  dashed  = correlational (use linkLineDash)
```

### 3. `ModelCompareGraph` — GBM vs LSTM Bipartite
**API:** `POST /compare`

```
Left cluster  = GBM top features (blue nodes)
Right cluster = LSTM temporal features (purple nodes)
Centre        = Overlap nodes (teal — appear in both)

Links  = feature → model node
  color = agreement flag (#22c55e if agree, #ef4444 if disagree)
```

### 4. `RecommendPathGraph` — Recommendation Path
**API:** `POST /recommend/student` (one item's shap_values)

```
Student node (centre, large, indigo)
  → feature nodes (sized by |shap|)
  → resource/item node (gold)

Layout: 3 depth layers via custom nodeVal + forceZ
```

---

## Core XAI Panels

### `RiskScoreCard`
- Circular gauge (recharts RadialBarChart)
- Color bands: green < 0.3 / yellow 0.3–0.6 / red > 0.6
- Shows `risk_score`, `risk_label`, `model_version`

### `TopFeaturesBar`
- Horizontal bar chart from `shap_values` (top 8)
- Bars colored by sign: positive = red (increases risk), negative = green

### `AnchorRuleCard`
- Displays `anchor_rule` string in styled pill/chip format
- Parses `IF … AND … THEN` into readable condition chips

### `WhatIfForm`
- Dynamic form built from `feature_names` (fetch from `/health`)
- On submit → `POST /whatif` → show `score_delta` + `direction` badge
- Or for recommend: `POST /recommend/student/whatif`

### `TrustScoreCard`
- Displays `shap_stability` (0–1) as a meter
- For dropout: `trust_score.band` label + `stability` from `/explain`

### `RecommendationList`
- Ranked list of scored items
- Each row: rank badge + item_id + score bar + top-3 feature chips
- Click → opens `ShapInteractionGraph` + full explain panel

---

## Pages

### `StudentView`
Layout (two columns):
- **Left:** RiskScoreCard → TopFeaturesBar → AnchorRuleCard → WhatIfForm
- **Right:** ShapInteractionGraph (tabbed with CausalDagGraph)

Below: RecommendationList (top-5) → click item → RecommendPathGraph

### `InstructorView`
- RecommendationList of interventions
- Click item → ShapInteractionGraph for that assignment
- WhatIfForm for counterfactual adjustment

### `ComparePage`
- Two RiskScoreCards side by side (GBM + LSTM)
- ModelCompareGraph (3D bipartite)
- Disagreement badge + interpretation string

---

## Implementation Phases (suggested order)

### Phase 1 — Scaffold
- [ ] `npm create vite@latest frontend -- --template react-ts`
- [ ] Install: `tailwindcss`, `@shadcn/ui`, `react-router-dom`, `@tanstack/react-query`, `axios`, `recharts`, `three`, `react-force-graph-3d`
- [ ] Set up `api/client.ts` with base URL + auth header stub
- [ ] Create `api/dropout.ts` and `api/recommend.ts` with typed fetch functions
- [ ] Scaffold `graphTransforms.ts` with all 4 transform functions

### Phase 2 — Core XAI Panels
- [ ] RiskScoreCard
- [ ] TopFeaturesBar
- [ ] AnchorRuleCard
- [ ] TrustScoreCard
- [ ] RecommendationList

### Phase 3 — 3D Graphs
- [ ] ShapInteractionGraph (start here — most reusable)
- [ ] CausalDagGraph
- [ ] RecommendPathGraph
- [ ] ModelCompareGraph

### Phase 4 — Pages + WhatIf
- [ ] StudentView layout
- [ ] WhatIfForm + delta display
- [ ] InstructorView layout
- [ ] ComparePage

### Phase 5 — Polish
- [ ] GraphCard wrapper (title, fullscreen toggle, legend)
- [ ] Sidebar model status badges from `/recommend/health` + `/health`
- [ ] Loading skeletons for all async panels
- [ ] Error boundaries around each 3D graph

---

## Key Graph Data Transform (copy-paste ready)

```typescript
// src/lib/graphTransforms.ts

export function shapInteractionToGraph(explanation: ExplainResponse) {
  const nodes = explanation.top_features.map(f => ({
    id: f,
    val: Math.abs(explanation.shap_values[f] ?? 0) * 100,
    type: explanation.causal_annotations?.[f] ?? 'unknown',
  }));
  const links = (explanation.feature_interactions ?? []).map(ix => ({
    source: ix.features[0],
    target: ix.features[1],
    value:  ix.strength,
    direction: ix.direction,
  }));
  return { nodes, links };
}

export function compareToGraph(compare: CompareResponse) {
  const gbmNodes  = compare.gbm_top3.map(f => ({ id: f, group: 'gbm' }));
  const lstmNodes = compare.lstm_top3_temporal?.map(f => ({ id: f, group: 'lstm' })) ?? [];
  const shared    = new Set(compare.gbm_top3.filter(f => compare.lstm_top3_temporal?.includes(f)));
  const allNodes  = [
    ...gbmNodes.map(n => ({ ...n, group: shared.has(n.id) ? 'shared' : 'gbm' })),
    ...lstmNodes.filter(n => !shared.has(n.id)),
    { id: 'GBM Model',  group: 'model-gbm' },
    { id: 'LSTM Model', group: 'model-lstm' },
  ];
  const links = [
    ...compare.gbm_top3.map(f => ({ source: f, target: 'GBM Model' })),
    ...(compare.lstm_top3_temporal ?? []).map(f => ({ source: f, target: 'LSTM Model' })),
  ];
  return { nodes: allNodes, links };
}
```

---

## Environment Variables

```env
# frontend/.env.local
VITE_API_URL=http://localhost:8000
VITE_APP_TITLE=XAI Learning System
```

---

## Notes for Frontend Developer

- All explainability endpoints return gracefully even when models aren't loaded — check `loaded: false` from `/recommend/health` to show disabled state
- `shap_values` keys match `feature_columns` from the ranker; use these as node IDs in the graph
- `ForceGraph3D` is heavy — lazy-load it with `React.lazy()` + `Suspense`
- The `anchor_rule` string always follows the pattern `IF cond AND cond THEN recommended` — safe to parse on spaces around `AND`
- Auth: the `/recommend/*` and `/explain` endpoints are public; MLOps endpoints (`/mlops/*`) require Bearer token
