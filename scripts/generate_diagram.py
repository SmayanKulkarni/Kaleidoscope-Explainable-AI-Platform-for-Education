#!/usr/bin/env python3
"""Generate Excalidraw architecture diagram for XAI Learning Recommendation System."""

import json
import random

random.seed(42)

ELEMENTS = []
_id_counter = 0

# Colors
C_DATA = "#a5d8ff"
C_MODEL = "#b2f2bb"
C_XAI = "#d0bfff"
C_EVAL = "#ffd8a8"
C_MLOPS = "#fcc2d7"
C_NARR = "#ffec99"
C_API = "#dee2e6"
C_FRONT = "#99e9f2"
C_RUST = "#ffa8a8"
C_TITLE = "#1971c2"
C_SECTION = "#f8f9fa"


def _uid():
    global _id_counter
    _id_counter += 1
    return f"el_{_id_counter:04d}"


def _seed():
    return random.randint(100000, 999999)


def add_rect(x, y, w, h, bg, stroke="#1e1e1e", sw=2, group=None, label=None, font_size=16, font_color="#1e1e1e"):
    rid = _uid()
    tid = _uid() if label else None
    rect = {
        "id": rid, "type": "rectangle",
        "x": x, "y": y, "width": w, "height": h,
        "angle": 0, "strokeColor": stroke, "backgroundColor": bg,
        "fillStyle": "solid", "strokeWidth": sw, "roughness": 1, "opacity": 100,
        "groupIds": [group] if group else [],
        "frameId": None, "roundness": {"type": 3},
        "seed": _seed(), "version": 1, "versionNonce": _seed(),
        "isDeleted": False,
        "boundElements": [{"id": tid, "type": "text"}] if tid else [],
        "updated": 1700000000000, "link": None, "locked": False,
    }
    ELEMENTS.append(rect)
    if label and tid:
        txt = {
            "id": tid, "type": "text",
            "x": x + 5, "y": y + (h / 2) - (font_size * 1.25 * label.count('\n') + font_size * 1.25) / 2,
            "width": w - 10, "height": font_size * 1.25 * (label.count('\n') + 1),
            "angle": 0, "strokeColor": font_color, "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 1, "roughness": 1, "opacity": 100,
            "groupIds": [group] if group else [],
            "frameId": None, "roundness": None,
            "seed": _seed(), "version": 1, "versionNonce": _seed(),
            "isDeleted": False, "boundElements": None,
            "updated": 1700000000000, "link": None, "locked": False,
            "text": label, "fontSize": font_size, "fontFamily": 1,
            "textAlign": "center", "verticalAlign": "middle",
            "containerId": rid, "originalText": label, "autoResize": True,
            "lineHeight": 1.25,
        }
        ELEMENTS.append(txt)
    return rid


def add_arrow(x1, y1, x2, y2, start_id=None, end_id=None, color="#1e1e1e", sw=2, dashed=False):
    aid = _uid()
    dx = x2 - x1
    dy = y2 - y1
    arrow = {
        "id": aid, "type": "arrow",
        "x": x1, "y": y1,
        "width": abs(dx), "height": abs(dy),
        "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": sw, "roughness": 1, "opacity": 100,
        "groupIds": [], "frameId": None,
        "roundness": {"type": 2},
        "seed": _seed(), "version": 1, "versionNonce": _seed(),
        "isDeleted": False, "boundElements": None,
        "updated": 1700000000000, "link": None, "locked": False,
        "points": [[0, 0], [dx, dy]],
        "lastCommittedPoint": None,
        "startBinding": {"elementId": start_id, "focus": 0, "gap": 8} if start_id else None,
        "endBinding": {"elementId": end_id, "focus": 0, "gap": 8} if end_id else None,
        "startArrowhead": None, "endArrowhead": "arrow",
    }
    if dashed:
        arrow["strokeStyle"] = "dashed"
    ELEMENTS.append(arrow)
    return aid


def add_section(x, y, w, h, title, bg=C_SECTION):
    gid = _uid()
    add_rect(x, y, w, h, bg, stroke="#868e96", sw=1, group=gid)
    add_rect(x, y, w, 32, bg="#343a40", stroke="#343a40", sw=1, group=gid,
             label=title, font_size=14, font_color="#ffffff")
    return y + 40  # content start Y


# ============================================================
# LAYOUT CONSTANTS
# ============================================================
W = 1900  # total diagram width
LM = 50   # left margin

# ============================================================
# TITLE
# ============================================================
add_rect(LM, 20, W, 65, "#1971c2", stroke="#1864ab", sw=3,
         label="XAI Learning Recommendation System — Architecture", font_size=24, font_color="#ffffff")

add_rect(LM, 90, W, 30, "transparent", stroke="transparent", sw=0,
         label="Dual-Model (GBM + LSTM) · 8 XAI Modules · MLflow MLOps · Rust MC Simulator · React Dashboard",
         font_size=13, font_color="#868e96")

# ============================================================
# LAYER 1: DATA
# ============================================================
cy = add_section(LM, 140, W, 200, "📊  DATA LAYER — OULAD Feature Engineering")

oulad = add_rect(80, cy + 10, 200, 70, C_DATA,
                 label="OULAD Raw Data\n7 CSVs · 10.6M rows")
dl = add_rect(340, cy + 10, 220, 70, C_DATA,
              label="data_loader.py\nStatic Pipeline")
tb = add_rect(340, cy + 90, 220, 70, C_DATA,
              label="temporal_builder.py\nTemporal Pipeline")
static_out = add_rect(620, cy + 10, 280, 70, "#d0ebff",
                       label="learners.csv (32,593)\ntrain.pkl · test.pkl")
temp_out = add_rect(620, cy + 90, 280, 70, "#d0ebff",
                     label="snapshots.pkl (195,558)\ntransitions.pkl (162,965)")

# Data arrows
add_arrow(280, cy + 45, 340, cy + 45, oulad, dl)
add_arrow(280, cy + 125, 340, cy + 125, oulad, tb)
add_arrow(560, cy + 45, 620, cy + 45, dl, static_out)
add_arrow(560, cy + 125, 620, cy + 125, tb, temp_out)

# Feature list box
add_rect(960, cy + 10, 480, 150, "#e7f5ff", stroke="#74c0fc", sw=1,
         label="12 LEARNER_FEATURES\n────────────────────\nlogin_frequency_weekly · avg_session_duration_min\nforum_posts_count · video_completion_rate\nquiz_avg_score · quiz_completion_rate\nassignment_submission_rate · days_since_last_activity\nprior_course_completions · current_week_in_course\nmissed_deadlines_count · help_requests_count",
         font_size=12)

# Target box
add_rect(1500, cy + 10, 200, 70, "#fff3bf", stroke="#fab005", sw=2,
         label="Target\ndropout_risk\n31.2% positive", font_size=13)
add_rect(1500, cy + 90, 200, 70, "#fff3bf", stroke="#fab005", sw=2,
         label="Temporal\n6 snapshots/student\nWeeks 2,4,6,8,10,12", font_size=12)

# ============================================================
# LAYER 2: MODELS
# ============================================================
cy = add_section(LM, 360, 920, 170, "🤖  MODELS — Dual Architecture")

gbm = add_rect(80, cy + 10, 240, 55, C_MODEL,
               label="GradientBoosting (GBM)\nPrimary · TreeSHAP", font_size=13)
rf = add_rect(80, cy + 75, 240, 55, C_MODEL,
              label="RandomForest (RF)\nComparison · TreeSHAP", font_size=13)
lstm = add_rect(380, cy + 10, 240, 55, "#69db7c",
                label="LSTM (2-layer)\n6×12 sequence · DeepSHAP", font_size=13)
lstm_detail = add_rect(380, cy + 75, 240, 55, "#69db7c",
                       label="BCELoss · AdamW\nEarly stop on AUC", font_size=12)

# Calibration box
add_rect(680, cy + 10, 220, 55, "#d3f9d8", stroke="#51cf66", sw=1,
         label="CalibratedClassifierCV\nReliable predict_proba", font_size=12)
add_rect(680, cy + 75, 220, 55, "#d3f9d8", stroke="#51cf66", sw=1,
         label="models/gbm.pkl\nmodels/rf.pkl\nmodels/lstm.pt", font_size=11)

# ============================================================
# LAYER 2A: MLOps (right side, same Y as models)
# ============================================================
cy_mlops = add_section(1000, 360, 950, 170, "⚙️  MLOps — MLflow + Evidently")

mlflow_track = add_rect(1030, cy_mlops + 10, 220, 55, C_MLOPS,
                        label="MLflow Tracking\nParams · Metrics · Artifacts", font_size=12)
mlflow_reg = add_rect(1030, cy_mlops + 75, 220, 55, C_MLOPS,
                      label="Model Registry\nStaging → Production", font_size=12)
evidently = add_rect(1290, cy_mlops + 10, 220, 55, C_MLOPS,
                     label="Evidently AI\nData + Prediction Drift", font_size=12)
pred_log = add_rect(1290, cy_mlops + 75, 220, 55, C_MLOPS,
                    label="Prediction Logger\nSQLite rolling log", font_size=12)
model_card = add_rect(1550, cy_mlops + 10, 220, 55, C_MLOPS,
                      label="Model Card\nAuto-generated MD", font_size=12)
mlops_api = add_rect(1550, cy_mlops + 75, 220, 55, C_MLOPS,
                     label="/mlops/health\n/mlops/drift-report", font_size=12)

# ============================================================
# LAYER 3: XAI ENGINE
# ============================================================
cy = add_section(LM, 550, W, 200, "🔍  XAI ENGINE — 8 Explanation Modules")

# Row 1: Attribution methods
bw = 210
treeshap = add_rect(80, cy + 10, bw, 55, C_XAI,
                    label="TreeSHAP\nGBM feature attribution", font_size=12)
deepshap = add_rect(310, cy + 10, bw, 55, "#b197fc",
                    label="DeepSHAP\nLSTM attribution", font_size=12)
ig = add_rect(540, cy + 10, bw, 55, "#b197fc",
              label="Integrated Gradients\nCaptum · LSTM temporal", font_size=12)
archipelago = add_rect(770, cy + 10, bw, 55, C_XAI,
                       label="Archipelago\nPairwise interactions", font_size=12)
anchors = add_rect(1000, cy + 10, bw, 55, C_XAI,
                   label="Anchors\nIF-THEN rules ≥90%", font_size=12)

# Row 2: Counterfactual, prototype, uncertainty, causal
dice = add_rect(80, cy + 80, bw, 55, C_XAI,
                label="DiCE\nCounterfactual actions", font_size=12)
proto = add_rect(310, cy + 80, bw, 55, C_XAI,
                 label="Prototypes (k-NN)\nSimilar learner stories", font_size=12)
mapie = add_rect(540, cy + 80, bw, 55, C_XAI,
                 label="MAPIE\nConformal uncertainty", font_size=12)
causal = add_rect(770, cy + 80, bw, 55, C_XAI,
                  label="Causal DAG\nCausal vs correlational", font_size=12)
action_rank = add_rect(1000, cy + 80, bw, 55, C_XAI,
                       label="ActionRanker\nSHAP × causal weight", font_size=12)

# XAI Signal box
add_rect(1250, cy + 10, 440, 125, "#f3f0ff", stroke="#9775fa", sw=1,
         label="Unique Signals for LLM Narrator\n─────────────────────\nTreeSHAP: signed feature importance\nDeepSHAP: temporal attribution (6×12)\nArchipelago: pairwise risk amplifiers\nAnchors: human-readable IF-THEN rules\nDiCE: concrete actionable targets\nPrototypes: relatable student stories",
         font_size=11)

# ============================================================
# LAYER 4: EVALUATION
# ============================================================
cy = add_section(LM, 770, 920, 160, "✅  EVALUATION — Trust & Consistency")

trust = add_rect(80, cy + 10, 250, 55, C_EVAL,
                 label="TrustScorer\n0.4×Fidelity + 0.35×Stability + 0.25×Completeness", font_size=11)
drift_jsd = add_rect(360, cy + 10, 250, 55, C_EVAL,
                     label="Drift Detector (JSD)\nSHAP shift threshold >0.15", font_size=11)
expl_store = add_rect(640, cy + 10, 250, 55, C_EVAL,
                      label="ExplanationStore\nSQLite timeline", font_size=12)
trust_formula = add_rect(80, cy + 75, 810, 45, "#fff4e6", stroke="#fd7e14", sw=1,
                         label="Gate: trust_score < 0.5 → suppress explanation display  |  JSD > 0.15 → drift flag  |  Every /explain → TrustScorer mandatory",
                         font_size=11)

# ============================================================
# LAYER 4A: SIMULATION (right side)
# ============================================================
cy_sim = add_section(1000, 770, 950, 160, "🚀  SIMULATION — Rust Monte Carlo")

mc_rust = add_rect(1030, cy_sim + 10, 280, 55, C_RUST,
                   label="Rust MC Simulator (PyO3)\nRayon parallel · 10K trajectories", font_size=12)
mc_detail = add_rect(1030, cy_sim + 75, 280, 55, C_RUST,
                     label="mc_simulator/src/lib.rs\nmaturin develop --release", font_size=11)
mc_out = add_rect(1350, cy_sim + 10, 300, 55, "#ffe3e3", stroke="#fa5252", sw=1,
                  label="Output: feature_distributions\noutcome_distribution (dropout_prob)", font_size=11)
mc_narrator = add_rect(1350, cy_sim + 75, 300, 55, "#ffe3e3", stroke="#fa5252", sw=1,
                       label="→ Narrator: \"If patterns continue,\n72% dropout by week 16\"", font_size=11)

add_arrow(1310, cy_sim + 37, 1350, cy_sim + 37, mc_rust, mc_out)

# ============================================================
# LAYER 5: LLM NARRATION
# ============================================================
cy = add_section(LM, 950, W, 160, "💬  LLM NARRATION — Groq (Dual Audience)")

narrator = add_rect(80, cy + 10, 300, 55, C_NARR,
                    label="LLMNarrator (Groq SDK)\nCRITICAL: Narration only, never generates", font_size=12)
learner_view = add_rect(420, cy + 10, 280, 55, "#fff9db", stroke="#fcc419", sw=1,
                        label="🎓 Learner View\nMotivational · ≤100 words · 1 action", font_size=12)
instructor_view = add_rect(740, cy + 10, 280, 55, "#fff9db", stroke="#fcc419", sw=1,
                           label="👨‍🏫 Instructor View\nTechnical · ≤150 words · interventions", font_size=12)

add_arrow(380, cy + 37, 420, cy + 37, narrator, learner_view)
add_arrow(380, cy + 37, 740, cy + 37, narrator, instructor_view)

# Payload box
payload = add_rect(1060, cy + 10, 600, 110, "#fffbeb", stroke="#f59f00", sw=1,
                   label="LLM Payload (pre-computed XAI data only)\n──────────────────────────────────\n{shap_top3, interactions, anchor_rule, actions,\nprototypes, risk_score, uncertainty,\ncausal_annotations, mc_projection, temporal_attributions}",
                   font_size=12)

# ============================================================
# LAYER 6: API
# ============================================================
cy = add_section(LM, 1130, W, 140, "🌐  API — FastAPI (localhost:8000)")

ew = 170
predict = add_rect(80, cy + 10, ew, 55, C_API,
                   label="POST /predict\nrisk + uncertainty", font_size=11)
explain = add_rect(270, cy + 10, ew, 55, C_API,
                   label="POST /explain\nfull XAI suite", font_size=11)
whatif = add_rect(460, cy + 10, ew, 55, C_API,
                  label="POST /whatif\nTreeSHAP <50ms", font_size=11)
counterfactual = add_rect(650, cy + 10, ew, 55, C_API,
                          label="POST /counter..\nDiCE actions", font_size=11)
simulate = add_rect(840, cy + 10, ew, 55, C_API,
                    label="POST /simulate\nRust MC proj.", font_size=11)
history = add_rect(1030, cy + 10, ew, 55, C_API,
                   label="GET /history/{id}\ntimeline + drift", font_size=11)
mlops_ep = add_rect(1220, cy + 10, ew, 55, C_API,
                    label="GET /mlops/*\ndrift + metrics", font_size=11)
health = add_rect(1410, cy + 10, ew, 55, C_API,
                  label="GET /health\nmodel version", font_size=11)
feedback_ep = add_rect(1600, cy + 10, ew, 55, C_API,
                       label="POST /feedback\nhuman-in-loop", font_size=11)

# ============================================================
# LAYER 7: FRONTEND
# ============================================================
cy = add_section(LM, 1290, W, 170, "🖥️  FRONTEND — React + Tailwind + Recharts")

fw = 230
whatif_panel = add_rect(80, cy + 10, fw, 55, C_FRONT,
                        label="WhatIfPanel\n12 sliders · debounce 300ms", font_size=12)
expl_card = add_rect(330, cy + 10, fw, 55, C_FRONT,
                     label="ExplanationCard\nSHAP bars + trust + anchors", font_size=12)
cf_view = add_rect(580, cy + 10, fw, 55, C_FRONT,
                   label="CounterfactualView\nDiCE action cards", font_size=12)
timeline = add_rect(830, cy + 10, fw, 55, C_FRONT,
                    label="ConsistencyTimeline\nSHAP over sessions", font_size=12)
model_cmp = add_rect(1080, cy + 10, fw, 55, C_FRONT,
                     label="ModelComparison\nGBM vs LSTM side-by-side", font_size=12)
temp_traj = add_rect(1330, cy + 10, fw, 55, C_FRONT,
                     label="TemporalTrajectory\nMC confidence band", font_size=12)
audience = add_rect(1580, cy + 10, fw, 55, C_FRONT,
                    label="AudienceToggle\nLearner ↔ Instructor", font_size=12)

# Second row: additional frontend
causal_dag = add_rect(80, cy + 75, fw, 55, C_FRONT,
                      label="CausalDAG\nSVG react-flow graph", font_size=12)
add_rect(330, cy + 75, fw, 55, "#c3fae8", stroke="#20c997", sw=1,
         label="Frontend Stack\nReact · TypeScript · Tailwind\nRecharts · Headless UI", font_size=11)

# ============================================================
# CONNECTING ARROWS (layer-to-layer)
# ============================================================
# Data → Models (vertical)
add_arrow(500, 340, 200, 400, static_out, gbm, color="#339af0", sw=2)
add_arrow(760, 340, 500, 400, temp_out, lstm, color="#339af0", sw=2)

# Models → XAI (vertical)
add_arrow(200, 530, 185, 600, gbm, treeshap, color="#7950f2", sw=2)
add_arrow(500, 530, 415, 600, lstm, deepshap, color="#7950f2", sw=2)
add_arrow(500, 530, 645, 600, lstm, ig, color="#7950f2", sw=2)

# XAI → Evaluation (vertical)
add_arrow(205, 750, 205, 810, treeshap, trust, color="#e67700", sw=2)
add_arrow(485, 750, 485, 810, dice, drift_jsd, color="#e67700", sw=2)

# Evaluation → Narrator (vertical)
add_arrow(205, 930, 230, 990, trust, narrator, color="#f59f00", sw=2)

# Narrator → API (vertical)
add_arrow(230, 1110, 355, 1170, narrator, explain, color="#495057", sw=2)

# API → Frontend (vertical)
add_arrow(355, 1270, 445, 1330, explain, expl_card, color="#0c8599", sw=2)
add_arrow(545, 1270, 195, 1330, whatif, whatif_panel, color="#0c8599", sw=2)
add_arrow(925, 1270, 945, 1330, simulate, temp_traj, color="#0c8599", sw=2)

# MLOps connections (dashed)
add_arrow(200, 460, 1030, 400, gbm, mlflow_track, color="#e64980", sw=1, dashed=True)
add_arrow(500, 460, 1030, 430, lstm, mlflow_track, color="#e64980", sw=1, dashed=True)


# ============================================================
# LEGEND
# ============================================================
ly = 1480
add_rect(LM, ly, W, 80, "#f8f9fa", stroke="#dee2e6", sw=1)
add_rect(LM, ly, W, 28, "#343a40", stroke="#343a40", sw=1,
         label="LEGEND", font_size=12, font_color="#ffffff")

legend_items = [
    (80, ly + 35, 100, 30, C_DATA, "Data"),
    (200, ly + 35, 100, 30, C_MODEL, "Models"),
    (320, ly + 35, 100, 30, C_XAI, "XAI"),
    (440, ly + 35, 100, 30, C_EVAL, "Evaluation"),
    (560, ly + 35, 100, 30, C_MLOPS, "MLOps"),
    (680, ly + 35, 100, 30, C_NARR, "Narration"),
    (800, ly + 35, 100, 30, C_API, "API"),
    (920, ly + 35, 100, 30, C_FRONT, "Frontend"),
    (1040, ly + 35, 100, 30, C_RUST, "Rust MC"),
]
for lx, lly, lw, lh, lc, ll in legend_items:
    add_rect(lx, lly, lw, lh, lc, label=ll, font_size=11)

# Status indicators
add_rect(1200, ly + 35, 180, 30, "#d3f9d8", stroke="#40c057", sw=2,
         label="✅ DONE", font_size=12)
add_rect(1400, ly + 35, 180, 30, "#fff3bf", stroke="#fab005", sw=2,
         label="⏳ PENDING", font_size=12)
add_rect(1600, ly + 35, 180, 30, "#ffe3e3", stroke="#fa5252", sw=2,
         label="🔜 PLANNED", font_size=12)


# ============================================================
# BUILD & SAVE
# ============================================================
diagram = {
    "type": "excalidraw",
    "version": 2,
    "source": "xai-learning-rec-system",
    "elements": ELEMENTS,
    "appState": {
        "gridSize": None,
        "viewBackgroundColor": "#ffffff",
    },
    "files": {},
}

out_path = "/media/smayan/500GB SSD/Datahack 4.0/files/architecture.excalidraw"
with open(out_path, "w") as f:
    json.dump(diagram, f, indent=2)

print(f"Generated {len(ELEMENTS)} elements → {out_path}")
print(f"File size: {len(json.dumps(diagram)):,} bytes")
