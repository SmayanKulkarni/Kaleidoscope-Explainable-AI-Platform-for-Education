"""
Model Card Generator
=====================
Auto-generates a Markdown model card from training artifacts and MLflow metadata.

Public API
----------
ModelCardGenerator(models_dir, feature_names)
    .generate() -> str  (markdown text)
    .save(output_path) -> Path
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR   = PROJECT_ROOT / "models"
FILES_DIR    = PROJECT_ROOT / "files"


class ModelCardGenerator:
    def __init__(
        self,
        models_dir: Path | None = None,
        feature_names: list[str] | None = None,
    ):
        self.models_dir    = models_dir or MODELS_DIR
        self.feature_names = feature_names or []

    def _load_json(self, name: str) -> dict:
        path = self.models_dir / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}

    def generate(self) -> str:
        training_summary = self._load_json("training_summary.json")
        tuning_summary   = self._load_json("tuning_summary.json")
        lstm_config      = self._load_json("lstm_config.json")

        feature_names = self.feature_names or training_summary.get("feature_names", [])

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        sections = []

        # Title
        sections.append("# Model Card — XAI Dropout Risk Prediction System")
        sections.append(f"\n**Generated:** {now}  ")
        sections.append(f"**Dataset:** OULAD (Open University Learning Analytics Dataset)  ")
        sections.append(f"**Task:** Binary classification — predict student dropout risk  ")

        # Overview
        sections.append("\n## Overview\n")
        sections.append("This system uses a dual-model architecture for dropout risk prediction:")
        sections.append("- **GBM (GradientBoostingClassifier)** — primary static model, TreeSHAP-compatible")
        sections.append("- **RF (RandomForestClassifier)** — comparison / ensemble candidate")
        sections.append("- **LSTM (2-layer PyTorch)** — temporal trajectory model, DeepSHAP-compatible")

        # Features
        sections.append("\n## Input Features\n")
        sections.append(f"**Number of features:** {len(feature_names)}\n")
        sections.append("| # | Feature | Type |")
        sections.append("|---|---------|------|")
        for i, f in enumerate(feature_names, 1):
            ftype = "integer" if f in (
                "forum_posts_count", "days_since_last_activity",
                "missed_deadlines_count", "help_requests_count",
                "prior_course_completions", "current_week_in_course"
            ) else "float"
            sections.append(f"| {i} | `{f}` | {ftype} |")

        sections.append("\n**Synthetic approximations (clearly documented):**")
        sections.append("- `avg_session_duration_min`: sum_click/day × 2.5 min/click, capped 180 min/day")
        sections.append("- `forum_posts_count`: sum_click on forumng/ouwiki/forum activity types")
        sections.append("- `video_completion_rate`: distinct 'oucontent' sites / total available")
        sections.append("- `help_requests_count`: sum_click on questionnaire/resource; zero-rows Poisson-filled")

        # Training Data
        sections.append("\n## Training Data\n")
        gbm_metrics = training_summary.get("gbm", {})
        rf_metrics  = training_summary.get("rf", {})
        sections.append("| Split | Samples | Dropout Rate |")
        sections.append("|-------|---------|-------------|")
        sections.append("| Train | 26,074 | ~31.2% |")
        sections.append("| Test  | 6,519  | ~31.2% |")
        sections.append("| Temporal (LSTM) | 32,593 sequences (6 × 12) | ~31.2% |")

        # Performance
        sections.append("\n## Performance Metrics\n")
        sections.append("### Baseline (before tuning)\n")
        sections.append("| Metric | GBM | RF | LSTM |")
        sections.append("|--------|:---:|:--:|:----:|")

        def _m(d, k):
            v = d.get(k)
            return f"{v}" if v is not None else "—"

        sections.append(f"| AUC-ROC | {_m(gbm_metrics, 'test_auc_roc')} | {_m(rf_metrics, 'test_auc_roc')} | {_m(lstm_config, 'best_val_auc')} |")
        sections.append(f"| F1 | {_m(gbm_metrics, 'test_f1')} | {_m(rf_metrics, 'test_f1')} | — |")
        sections.append(f"| Brier Score | {_m(gbm_metrics, 'test_brier_score')} | {_m(rf_metrics, 'test_brier_score')} | — |")

        if tuning_summary:
            sections.append("\n### After Optuna tuning (20 trials per model)\n")
            sections.append("| Model | Tuned AUC | Improved? |")
            sections.append("|-------|:---------:|:---------:|")
            for name in ("gbm", "rf", "lstm"):
                ts = tuning_summary.get(name, {})
                auc = ts.get("test_auc") or ts.get("best_val_auc", "—")
                imp = "✅" if ts.get("improved") else "❌"
                sections.append(f"| {name.upper()} | {auc} | {imp} |")

        # Hyperparameters
        sections.append("\n## Hyperparameters\n")
        if tuning_summary.get("gbm", {}).get("best_params"):
            sections.append("### GBM (Tuned)\n")
            for k, v in tuning_summary["gbm"]["best_params"].items():
                if k != "random_state":
                    sections.append(f"- `{k}`: {v}")

        if lstm_config:
            sections.append("\n### LSTM\n")
            for k in ("hidden_dim", "n_layers", "dropout", "n_features", "n_timesteps"):
                if k in lstm_config:
                    sections.append(f"- `{k}`: {lstm_config[k]}")

        # XAI Methods
        sections.append("\n## Explainability Methods\n")
        sections.append("| Method | Module | Purpose |")
        sections.append("|--------|--------|---------|")
        sections.append("| TreeSHAP | `shap_explainer.py` | Per-feature attribution (GBM) |")
        sections.append("| DeepSHAP | `shap_explainer.py` | Per-timestep attribution (LSTM) |")
        sections.append("| SHAP Interaction | `archipelago.py` | Feature pair interactions |")
        sections.append("| DiCE | `dice_explainer.py` | Diverse counterfactual explanations |")
        sections.append("| Anchors (alibi) | `anchors_explainer.py` | IF-THEN rule extraction |")
        sections.append("| Prototypes (k-NN) | `prototype_explainer.py` | Similar past learner matching |")
        sections.append("| DoWhy | `causal_annotator.py` | Causal vs correlational annotation |")
        sections.append("| MAPIE | `uncertainty_estimator.py` | Conformal prediction sets |")
        sections.append("| Trust Score | `trust_scorer.py` | Composite fidelity/stability/completeness |")
        sections.append("| Explanation Drift | `drift_detector.py` | JSD-based explanation change detection |")

        # Limitations
        sections.append("\n## Known Limitations\n")
        sections.append("- 4 of 12 features are **synthetic approximations** from OULAD click data")
        sections.append("- LSTM temporal sequences are reconstructed from biweekly snapshots, not real-time")
        sections.append("- Causal effects estimated via DoWhy with linear regression — may miss non-linear effects")
        sections.append("- No demographic features used (age, gender, disability) — fairness by design")
        sections.append("- Conformal intervals assume exchangeability of calibration data")

        # Ethical Considerations
        sections.append("\n## Ethical Considerations\n")
        sections.append("- **No demographic features** used to prevent proxy discrimination")
        sections.append("- All explanations are **algorithmically derived**, not LLM-generated")
        sections.append("- LLM narrator receives only pre-computed data — it cannot invent explanations")
        sections.append("- Causal annotations distinguish actionable vs correlational features")
        sections.append("- System is designed as a **support tool**, not an autonomous decision-maker")

        return "\n".join(sections)

    def save(self, output_path: Path | None = None) -> Path:
        output_path = output_path or FILES_DIR / "MODEL_CARD.md"
        text = self.generate()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(text)
        log.info("Model card saved → %s", output_path)
        return output_path
