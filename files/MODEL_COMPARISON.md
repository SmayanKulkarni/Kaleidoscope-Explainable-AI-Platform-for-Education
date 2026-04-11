# Model Comparison Report

**Project:** XAI Learning Recommendation System — Dropout Risk Prediction  
**Dataset:** OULAD (32,593 students · 31.2% dropout rate)  
**Evaluated:** 2026-04-11  
**Train / Test split:** 80 / 20 stratified (26,074 train · 6,519 test)

---

## Overview

Three models are trained in this system serving different purposes:

| Model | Type | Input | Primary Role |
|---|---|---|---|
| **GBM** | GradientBoostingClassifier (calibrated) | 12 static features | Primary predictor + TreeSHAP |
| **RF** | RandomForestClassifier (calibrated) | 12 static features | Comparison / ensemble candidate |
| **LSTM** | 2-layer LSTM (PyTorch, GPU) | 6 × 12 temporal sequence | Trajectory-aware prediction + DeepSHAP |

---

## Test Set Metrics

| Metric | GBM | RF | LSTM |
|---|:---:|:---:|:---:|
| **AUC-ROC** ↑ | **0.9427** | 0.9425 | 0.8771 |
| **Avg Precision (AP)** ↑ | **0.8439** | **0.8442** | 0.7856 |
| **F1 Score** ↑ | **0.8035** | 0.7851 | 0.7275 |
| **Brier Score** ↓ | **0.0873** | 0.0902 | 0.1189 |
| **Accuracy** ↑ | **0.8700** | — | 0.8406 |
| **Training Time** | 22.3s (CPU) | 1.7s (CPU) | ~90s (RTX 4070S) |

> RF accuracy not separately recorded — class report logged to MLflow.  
> LSTM metrics evaluated on 20% hold-out (stratified, `random_state=42`).

---

## GBM — Detailed Classification Report

```
              precision    recall  f1-score   support

  No Dropout       0.93      0.88      0.90      4488
     Dropout       0.76      0.85      0.80      2031

    accuracy                           0.87      6519
   macro avg       0.85      0.86      0.85      6519
weighted avg       0.88      0.87      0.87      6519
```

---

## Model Hyperparameters

### GBM
| Param | Value |
|---|---|
| `n_estimators` | 200 |
| `learning_rate` | 0.05 |
| `max_depth` | 4 |
| `subsample` | 0.8 |
| `min_samples_leaf` | 20 |
| Calibration | `CalibratedClassifierCV` isotonic, cv=5 |

### RF
| Param | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | None (full trees) |
| `min_samples_leaf` | 10 |
| Calibration | `CalibratedClassifierCV` sigmoid, cv=5 |

### LSTM
| Param | Value |
|---|---|
| Architecture | Input(6,12) → LSTM(64) → LSTM(32) → Dropout(0.3) → Dense(1) |
| `hidden_dim` | 64 |
| `n_layers` | 2 |
| `dropout` | 0.3 |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4 |
| Loss | BCEWithLogitsLoss |
| Training | AMP (FP16), torch.compile (reduce-overhead), TF32 |
| Early stop | val AUC, patience=5 |
| Best val AUC | **0.8771** |
| Device | NVIDIA RTX 4070 SUPER (CUDA 12.6) |

---

## Post-Tuning Results (Optuna TPE, 20 trials each)

### Tuned vs Baseline — AUC-ROC

| Model | Baseline AUC | Tuned AUC | Δ | Model Updated? |
|---|:---:|:---:|:---:|:---:|
| **GBM** | 0.9427 | **0.9431** | +0.0004 | ✅ Yes |
| **RF** | 0.9425 | 0.9422 | −0.0003 | ❌ No (baseline kept) |
| **LSTM** | 0.8771 | **0.8780** | +0.0009 | ✅ Yes |

> RF tuned AUC did not beat baseline — original `rf.pkl` retained.

### Best Hyperparameters Found

#### GBM (Tuned)
| Param | Baseline | Tuned |
|---|---|---|
| `n_estimators` | 200 | **300** |
| `learning_rate` | 0.05 | **0.0349** |
| `max_depth` | 4 | 4 |
| `subsample` | 0.8 | **0.774** |
| `min_samples_leaf` | 20 | **29** |
| `max_features` | — | **sqrt** |

#### RF (Best found, not saved)
| Param | Baseline | Best Trial |
|---|---|---|
| `n_estimators` | 200 | 500 |
| `max_depth` | None | 13 |
| `min_samples_leaf` | 10 | 19 |
| `max_features` | — | log2 |
| `min_samples_split` | — | 15 |

#### LSTM (Tuned)
| Param | Baseline | Tuned |
|---|---|---|
| `hidden_dim` | 64 | **32** |
| `n_layers` | 2 | **1** |
| `dropout` | 0.30 | **0.293** |
| `lr` | 1e-3 | **3.54e-3** |
| `weight_decay` | 1e-4 | **2.61e-5** |
| `batch_size` | 512 | **1024** |

> LSTM converged to a simpler architecture (1-layer, hidden=32) — suggests the temporal patterns in OULAD data are relatively shallow.

---

## Analysis

### GBM vs RF
- Near-identical AUC-ROC (0.9427 vs 0.9425) and Avg Precision (0.8439 vs 0.8442).
- GBM wins on F1 (+0.018) and Brier score (+0.003) — better calibrated probabilities.
- RF trains in **1.7s** vs GBM's **22.3s** — useful for rapid experimentation.
- **GBM selected as primary** for TreeSHAP compatibility and marginally better calibration.

### GBM vs LSTM
- GBM outperforms LSTM on all metrics with static features alone.
- LSTM gap (AUC −0.066, F1 −0.076) is expected: the temporal model sees a **6-step trajectory** reconstructed from static snapshots, not ground-truth week-by-week records at test time.
- LSTM's value is **explainability depth** — DeepSHAP provides per-timestep attributions showing *when* in the course the risk signal emerged, which GBM cannot provide.
- In production the two models are **complementary**: GBM drives the primary risk score; LSTM drives the temporal explanation view.

### Calibration (Brier Score)
All three models achieve Brier < 0.12 (random = 0.215 for 31.2% base rate), confirming reliable `predict_proba` outputs suitable for MAPIE conformal uncertainty estimation.

---

## MLflow Registry

| Model | Registered Name | Version |
|---|---|---|
| GBM | `gbm-dropout-risk` | 1 |
| RF | `rf-dropout-risk` | 1 |
| LSTM | `lstm-dropout-risk` | 1 |

View runs: `mlflow ui --backend-store-uri mlruns/`

---

## Files

| Artifact | Path |
|---|---|
| GBM weights | `models/gbm.pkl` |
| RF weights | `models/rf.pkl` |
| LSTM weights | `models/lstm.pt` |
| LSTM config | `models/lstm_config.json` |
| Training summary | `models/training_summary.json` |
| MLflow runs | `mlruns/` |
