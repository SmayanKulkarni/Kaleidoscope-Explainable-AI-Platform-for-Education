# XAI Model Audit & Architecture Review
## Explainable Learning Recommendation System — OULAD Dataset

> **Scope:** Full study of every XAI technique in the stack, appropriateness for the problem statement, speed analysis, alternatives, layer-by-layer integration audit, and recommendations.

---

# PART 1 — XAI MODEL DEEP DIVE

---

## 1. TreeSHAP (Primary Feature Attribution)

### What It Is
TreeSHAP is a specialization of SHAP (SHapley Additive exPlanations) optimized for tree-ensemble models. It computes **exact** Shapley values by exploiting the tree structure, avoiding the exponential cost of brute-force coalitional game theory. Each feature gets a signed attribution: positive = pushes risk up, negative = pushes risk down.

**Complexity:** `O(TLD²)` where T = trees, L = leaves, D = depth.

### Why We Use It
- **Mandatory MVP requirement** — "You must implement at least one of SHAP, LIME, Integrated Gradients, or Attention Visualization."
- **Feature-level explanation** (mandatory) — TreeSHAP provides per-feature signed attribution for every individual prediction.
- **Stability indicator** — by re-running SHAP under Gaussian noise (n=20), we derive a stability metric quantifying how robust the explanation is.
- **Fidelity guarantee** — Shapley values satisfy the *efficiency* axiom: `sum(SHAP_values) + base_value ≈ predict_proba(x)`. This is the strongest faithfulness guarantee available.

### Appropriateness for OULAD + Dropout Prediction
**Highly appropriate.** Reasons:

| Factor | Assessment |
|--------|------------|
| Model type | GBM is a tree ensemble → TreeSHAP is the *native* and fastest explainer |
| Feature count | 12 features → SHAP computation is very fast (~10-50ms) |
| Feature types | Mix of continuous (rates, scores) and discrete (counts, weeks) — SHAP handles both |
| Educational interpretability | SHAP values directly answer "which features drove this student's risk prediction" |
| Additivity | SHAP values decompose the prediction exactly — perfect for the Trust Score fidelity check |

### Speed Assessment
- **12 features × GBM with ~100 trees:** ~10-50ms per explanation. **Fast enough for stored explanations.**
- **Not fast enough** for real-time What-If slider updates (12 sliders × rapid changes = needs <5ms). This is why FastSHAP exists as a companion.

### Could We Improve or Replace?
| Alternative | Verdict |
|-------------|---------|
| **KernelSHAP** | Model-agnostic but 10-100× slower. Only use as fallback if model changes to non-tree. |
| **Integrated Gradients** | Requires differentiable model. Not applicable to GBM directly. Would need a neural surrogate, adding complexity with no benefit over TreeSHAP. |
| **LIME** | Faster for a single explanation but lacks SHAP's theoretical guarantees (efficiency, symmetry). We use LIME separately at the concept level. |
| **Attention Visualization** | Requires transformer/attention architecture. Not applicable. |

**Verdict: TreeSHAP is the optimal choice for this model + dataset. No change needed.**

### Where It's Used (Layers)
1. **Feature Level (Explanation Layer)** — per-feature signed attributions
2. **Prediction Level** — global SHAP summary (mean |SHAP| across all training data)
3. **Evaluation Layer** — fidelity component of Trust Score
4. **Evaluation Layer** — stability score via perturbation analysis
5. **Tracking Layer** — SHAP distributions stored for drift detection (JSD)
6. **Prescription Layer** — SHAP magnitude feeds into Action Ranker priority formula

### Relation to Problem Statement
The PS demands: *"Explanations must be derived directly from model internals."* TreeSHAP uses the actual tree splits/paths of the GBM model to compute attributions — this is as "model internal" as it gets. It directly maps to the **"Feature-level explanation"** in the Multi-Layer Explanation Engine requirement.

---

## 2. FastSHAP (Real-Time Amortized Attribution)

### What It Is
FastSHAP (Jethani et al., 2021) trains a small neural network surrogate that learns to **predict SHAP values directly** from input features. Instead of computing SHAP from scratch each time, the surrogate does a single forward pass:

```
Input: X (12-dim feature vector)
Output: Ŝ (12-dim approximate SHAP values)
Latency: ~3-5ms (vs 10-50ms for TreeSHAP)
```

The surrogate is trained once on (X_train → TreeSHAP(X_train)) pairs with an MSE loss.

### Why We Use It
- **What-If Panel** — The PS requires *"Allow users to modify input features via UI controls to see real-time updates to prediction scores, recommendations, and risk levels."*
- When a user drags 12 sliders with 300ms debounce, we need SHAP values in <5ms. TreeSHAP's 10-50ms creates perceptible lag.
- FastSHAP sacrifices ~5% accuracy for ~10× speed, which is an excellent trade-off for interactive use.

### Appropriateness for OULAD
**Appropriate with caveats.**

| Factor | Assessment |
|--------|------------|
| Training cost | 2-3 minutes to train the MLP surrogate on X_train SHAP values — acceptable at startup |
| Accuracy loss | ~5% deviation from exact SHAP — fine for interactive previews, NOT for stored explanations |
| Feature count | 12 features → small MLP (64-64-12) is more than sufficient |
| Risk | If training data distribution shifts, the surrogate may produce bad SHAP estimates silently |

### Speed Assessment
- **~3-5ms per call** — ✅ Real-time slider interaction.
- **Training:** ~2-3 min at startup — acceptable if precomputed/cached.

### Could We Improve or Replace?
| Alternative | Verdict |
|-------------|---------|
| **Cached TreeSHAP** | Pre-compute SHAP for a grid of feature values and interpolate. Works for <5 features but 12-dim grid is impractical. |
| **Linear SHAP** | Instant but requires a linear model. Too lossy for GBM. |
| **No FastSHAP (just TreeSHAP)** | 10-50ms might be acceptable with aggressive UI debounce (500ms+). Worth testing before adding the complexity of a surrogate. |
| **Gradient-based approximation** | Build a simple NN surrogate of the GBM, then use Integrated Gradients. Similar concept to FastSHAP but with different theoretical basis. More complex, no clear benefit. |

**Recommendation:** FastSHAP is a strong choice. **However**, test whether TreeSHAP at ~50ms with 300ms debounce gives acceptable UX first. If so, you can simplify by dropping FastSHAP entirely and using TreeSHAP everywhere. This eliminates the surrogate training step and the accuracy gap. **Only add FastSHAP if TreeSHAP is measurably too slow.**

### Where It's Used (Layers)
1. **What-If Panel** — real-time SHAP bar chart updates when sliders move

### Relation to Problem Statement
Maps to: *"Changes made in the What-If interface must feed through the actual model."* FastSHAP calls the actual GBM's `predict_proba` during training, so its approximations are grounded in the real model. The `predict_proba` call itself for the risk score still goes directly through GBM — FastSHAP only approximates the SHAP decomposition.

---

## 3. Archipelago / SHAP Interaction Values (Feature Interactions)

### What It Is
Not the standalone "Archipelago" library (Tsang et al., 2020), but rather the **SHAP interaction values** available from `shap.TreeExplainer.shap_interaction_values()`. This produces an `(n_features × n_features)` matrix where off-diagonal entries `[i,j]` quantify the *synergistic* contribution of feature pair `(i,j)` — i.e., the excess attribution beyond their individual SHAP values.

### Why We Use It
- **Concept-level explanation** (mandatory) — interactions reveal *synergistic risk patterns* like "low quiz scores + high inactivity amplify each other."
- In education, single-feature explanations miss the point. A student with 50% quiz completion might be fine if they're active. But 50% quiz completion + 10 days inactive = catastrophic. Interaction values catch this.
- Directly supports the **Diagnosis layer** — understanding *why combinations of features* produce a prediction.

### Appropriateness for OULAD
**Highly appropriate.**

| Factor | Assessment |
|--------|------------|
| Educational relevance | Dropout is inherently an interaction phenomenon — engagement × performance × consistency |
| Interpretability | "Your quiz scores AND your inactivity together amplify risk" is far more actionable than individual features |
| Feature count | 12 features → C(12,2) = 66 pairs. Top-5 is very manageable. |

### Speed Assessment
- `shap_interaction_values()` computes the full interaction matrix. For GBM with 100 trees: **~100-500ms**.
- This is acceptable for the `/explain` endpoint (not used in What-If).

### Could We Improve or Replace?
| Alternative | Verdict |
|-------------|---------|
| **Standalone Archipelago library** | Would require separate installation and doesn't integrate as cleanly with TreeSHAP. The `shap_interaction_values()` method is native and simpler. |
| **H-statistic (Friedman)** | Measures interaction strength but requires many model evaluations. Slower and less granular. |
| **LIME with interaction terms** | LIME can fit interaction terms in its local linear model, but less theoretically grounded than SHAP interactions. |
| **Attention weights** | Not applicable (no attention model). |

**Verdict: Using SHAP interaction values from TreeExplainer is the best approach — native, exact, and requires no additional library.**

### Where It's Used (Layers)
1. **Concept Level (Explanation Layer)** — top pairwise interactions with "amplifying/dampening" labels
2. **Narration Layer** — interaction narrative fed to LLM for human-readable description

### Relation to Problem Statement
Maps to: *"Generate explanations at the concept-level. Simply showing raw feature names is not acceptable."* Interaction values go beyond raw feature names by showing how features *combine* to produce risk — this is concept-level reasoning.

---

## 4. DiCE (Diverse Counterfactual Explanations)

### What It Is
DiCE (Mothilal et al., 2020) generates **diverse counterfactual examples** — minimal changes to input features that would flip the model's prediction. It uses optimization (gradient-based, genetic, or random search) to find counterfactuals that are (a) close to the original, (b) achieve the desired outcome, and (c) diverse from each other.

Key constraint: **actionability**. Immutable features (`prior_course_completions`, `current_week_in_course`) are locked via `features_to_vary`.

### Why We Use It
- **Mandatory MVP** — "Counterfactual Explanation Engine: Automatically compute and display human-readable, minimal input changes required to flip a prediction."
- **Mandatory MVP** — "Action recommendations must not be hard-coded; they must be derived from the model output and counterfactual analysis."
- DiCE is the **only** algorithm in the stack that directly generates actionable prescriptions grounded in the model.

### Appropriateness for OULAD
**Extremely appropriate — this is the single most important XAI technique for the problem.**

| Factor | Assessment |
|--------|------------|
| Actionability | Education is inherently prescriptive — "what should this student DO?" DiCE answers this directly. |
| Feature constraints | Students can't change their course history — DiCE respects immutability constraints natively. |
| Diversity | Multiple counterfactuals give instructors options: "Either improve quizzes OR increase login frequency" |
| Realism | `permitted_range` ensures counterfactuals stay in realistic bounds (e.g., can't log in 100x/week) |

### Speed Assessment
- **`method="random"`:** ~200-500ms for 3 counterfactuals. Acceptable for `/counterfactual` endpoint.
- **`method="genetic"`:** ~1-3 seconds. Better diversity but slower.
- **`method="gradient"`:** Requires differentiable model. Not applicable to GBM directly.

### Could We Improve or Replace?
| Alternative | Verdict |
|-------------|---------|
| **Alibi Counterfactual** | Similar capability but less control over actionability constraints. DiCE is more mature for tabular data. |
| **CARLA framework** | Unified counterfactual benchmarking framework. Good for research comparison but overkill for hackathon. |
| **CEM Pertinent Negatives** | Already in the stack. CEM PN gives "what to change" but less control over diversity and actionability. DiCE is superior for prescriptive actions. |
| **Manual rules** | Violates PS: "Action recommendations must not be hard-coded." |
| **`method="genetic"` over `method="random"`** | **Recommended upgrade.** Genetic method produces more diverse, realistic counterfactuals. The 1-3s latency is acceptable since `/counterfactual` is not a real-time endpoint. |

**Recommendation:** Switch from `method="random"` to `method="genetic"` for better counterfactual quality. The latency is acceptable.

### Where It's Used (Layers)
1. **Prescription Layer** — generates counterfactual "target states"
2. **Action Ranker** — DiCE outputs feed into `priority_score = shap_magnitude × actionability × causality`
3. **Counterfactual View (Frontend)** — displayed as action cards with "current → target"
4. **What-If Panel** — "Apply This Change" button pre-fills sliders from counterfactual values

### Relation to Problem Statement
Directly maps to: *"Every explanation must conclude with at least one specific, prioritized action and an estimated impact"* and *"The core flow should be: Prediction, Explanation, Diagnosis, and Prescription."* DiCE IS the Prescription step.

---

## 5. CEM (Contrastive Explanation Method)

### What It Is
CEM (Dhurandhar et al., 2018) produces two types of contrastive explanations:
- **Pertinent Positive (PP):** The minimal set of features that are *sufficient* for the current prediction. "Your risk is HIGH **because of** low assignment submission + high inactivity."
- **Pertinent Negative (PN):** The minimal feature changes that would *flip* the prediction. "You would NOT be flagged if you logged in 4x/week."

CEM requires a **differentiable model** (uses gradient-based optimization), hence the Keras surrogate.

### Why We Use It
- **Diagnosis Layer** — CEM answers "why this prediction AND why not the opposite?" which SHAP alone cannot.
- SHAP tells you *how much* each feature contributed (quantitative). CEM tells you *which features are sufficient/necessary* (qualitative). These are complementary perspectives.
- PP → core diagnosis ("here's what's driving the problem")
- PN → prescription preview ("here's what would fix it")

### Appropriateness for OULAD
**Appropriate but with significant practical concerns.**

| Factor | Assessment |
|--------|------------|
| Diagnostic value | PP/PN is an excellent framing for educators: "why flagged" + "how to unflag" |
| Keras surrogate | Adds complexity — must train an NN to mimic GBM. Fidelity of surrogate ≠ fidelity of original model. |
| CEM speed | Slow: ~500ms-2s per explanation due to iterative optimization. |
| `alibi` CEM | The `alibi` library's CEM implementation can be finicky with hyperparameters (kappa, beta, gamma, theta). |
| Redundancy with DiCE | PN (minimal flip) overlaps significantly with DiCE counterfactuals. |

### Speed Assessment
- **~500ms-2s per call** — run synchronously after the parallel SHAP/Anchors/Prototypes batch.
- This is the **slowest component** in the `/explain` pipeline.

### Could We Improve or Replace?
| Alternative | Verdict |
|-------------|---------|
| **Drop CEM, use DiCE for PN** | DiCE already finds "minimal changes to flip prediction" — this IS the Pertinent Negative. Dropping CEM saves ~1-2s per explanation and eliminates the Keras surrogate. **Strong recommendation.** |
| **CEM PP only (skip PN)** | Keep only the "sufficient features" analysis. But this can be approximated by taking the top-k SHAP features that account for >80% of attribution (Completeness metric). |
| **Approximate PP via SHAP** | Top-k features by |SHAP| that cover ≥ 80% of total attribution ≈ pertinent positives. Less rigorous but much faster and no surrogate needed. |
| **Anchors as PP proxy** | Anchors already produce "IF condition THEN prediction" with high precision — this is essentially a pertinent positive in rule form. |
| **Keep CEM but pre-compute** | Pre-compute CEM for common feature profiles at startup, cache results. Reduces per-request latency but adds startup time. |

**⚠️ CRITICAL RECOMMENDATION:** CEM is the **weakest link** in the architecture. It:
1. Requires a Keras surrogate (extra training, fidelity risk)
2. Is the slowest component (~2s)
3. Overlaps with DiCE (PN) and Anchors/SHAP (PP)

**Proposed simplification:**
- **PP →** Approximate using top-k SHAP features covering ≥80% attribution + anchor rules
- **PN →** Already provided by DiCE counterfactuals
- **Result:** Eliminate CEM, Keras surrogate, and TensorFlow dependency. Save ~2s per explanation. No loss of diagnostic capability.

If judges specifically value CEM as a "novel technique," keep it but make it async/optional.

### Where It's Used (Layers)
1. **Diagnosis Layer** — PP and PN outputs
2. **Narration Layer** — PP/PN narratives for LLM

### Relation to Problem Statement
Maps to the "Diagnosis" step in *"Prediction, Explanation, Diagnosis, and Prescription."* However, the same diagnostic insight can be achieved through SHAP (top features = PP) + DiCE (counterfactuals = PN) without CEM.

---

## 6. Anchors (Rule-Based Explanations)

### What It Is
Anchors (Ribeiro et al., 2018) produce IF-THEN rules that are **guaranteed to hold with ≥ threshold precision** in the local neighborhood. Example:

> IF assignment_submission_rate ≤ 0.4 AND days_since_last_activity > 7 THEN dropout_risk = HIGH (precision: 94%, coverage: 21%)

Uses beam search + bandit sampling to find the shortest anchor with sufficient precision.

### Why We Use It
- **Dual-Audience (Instructor View)** — Rules are the **most natural explanation format for instructors**. Instructors think in policies: "students who do X and Y tend to drop out."
- **Prediction-level explanation** (mandatory) — Anchors summarize the decision boundary around a prediction.
- **Precision guarantee** — Unlike SHAP (which gives magnitudes), anchors give *confidence bounds*.

### Appropriateness for OULAD
**Highly appropriate.**

| Factor | Assessment |
|--------|------------|
| Instructor readability | IF-THEN rules are immediately actionable for course design ("if < 40% submission, flag early") |
| Precision threshold | 90% precision means the rule generalizes well in the local neighborhood |
| Coverage tradeoff | High-precision rules often have low coverage (apply to few students). Surface both metrics. |
| Discretizer | Quartile discretization works well for educational features (naturally binned into performance quartiles) |

### Speed Assessment
- **~200ms-1s** depending on beam width and feature count.
- Can be parallelized with SHAP/Prototypes in the `/explain` pipeline.

### Could We Improve or Replace?
| Alternative | Verdict |
|-------------|---------|
| **Decision rules from GBM directly** | Extract rules from the tree paths. Faster but less rigorous (no precision guarantee). |
| **LORE (Local Rule-based Explanations)** | Generates rules + counterfactual rules. More complex, less mature library. |
| **BRL (Bayesian Rule Lists)** | Global rule lists. More interpretable but requires model retraining. |
| **Reduce threshold to 0.85** | Produces shorter, higher-coverage rules at slight precision cost. Worth testing. |

**Verdict: Anchors is the right choice. No change needed.**

### Where It's Used (Layers)
1. **Prediction Level (Explanation Layer)** — anchor rule + precision + coverage
2. **Instructor View (Frontend)** — displayed in monospace box
3. **Narration Layer** — anchor rule text fed to LLM for instructor narrative

### Relation to Problem Statement
Maps to: *"Build distinct, instantly switchable explanation views for instructors (feature breakdowns, model confidence)."* Anchors provide the rule-based view that complements SHAP's numerical view.

---

## 7. ProtoDash / NearestNeighbors (Prototype Explanations)

### What It Is
Example-based explanations that find **similar historical learners** and present their outcomes. The architecture plans to use ProtoDash (IBM AIX360) but falls back to `sklearn.NearestNeighbors` with cosine similarity.

Output: "Your profile is 89% similar to Learner 218 who dropped out, and 76% similar to Learner 455 who completed after improving quiz completion."

### Why We Use It
- **Concept-level explanation** (mandatory) — Prototypes give the most human-comprehensible explanation type.
- **Dual-Audience (Learner View)** — Numbers alienate learners. "Similar to students who succeeded" is motivational and relatable.
- **Narrative anchor** — Provides a concrete, story-based explanation that the LLM narrator can expand.

### Appropriateness for OULAD
**Very appropriate, with one consideration.**

| Factor | Assessment |
|--------|------------|
| OULAD structure | OULAD has ~30,000 learners with known outcomes — excellent prototype pool |
| Motivational value | Learners respond better to peer comparison than statistical metrics |
| Privacy | Must anonymize learner IDs. Use synthetic IDs or aggregated profiles. |
| Similarity metric | Cosine similarity on standardized features works well for 12 continuous/discrete features |

### Speed Assessment
- **NearestNeighbors:** ~1-5ms after fitting. **Extremely fast.**
- **ProtoDash:** ~50-100ms. Slightly slower but produces weighted prototypes.

### Could We Improve or Replace?
| Alternative | Verdict |
|-------------|---------|
| **Use NearestNeighbors over ProtoDash** | **Recommended.** ProtoDash (aix360) has heavy dependencies and installation issues. k-NN with cosine similarity gives 90% of the value with zero friction. |
| **MMD-critic** | Produces prototypes AND criticisms (unusual examples). Interesting for instructor view but adds complexity. |
| **Cluster-based prototypes** | Pre-cluster learners, assign to nearest cluster, show cluster centroid. Simpler but less personalized. |
| **Add outcome-aware filtering** | Find similar learners who *succeeded despite similar starting conditions*. More motivational. **Recommended enhancement.** |

**Recommendation:** Use `NearestNeighbors` (cosine). Filter prototypes to show at least one success story. Add a "what they did differently" narrative.

### Where It's Used (Layers)
1. **Concept Level (Explanation Layer)** — similar learner matches + narrative
2. **Learner View (Frontend)** — motivational narrative
3. **Narration Layer** — prototype narrative fed to LLM

### Relation to Problem Statement
Maps to: *"Explanations must be personalized to the learner"* and *"Dual-Audience: learners (motivational, plain language)."* Prototypes are the primary personalization + motivation mechanism.

---

## 8. MAPIE (Uncertainty Quantification via Conformal Prediction)

### What It Is
MAPIE wraps any sklearn classifier to produce **conformal prediction intervals** — set-valued predictions with coverage guarantees. At alpha=0.1, the prediction set covers the true label with ≥90% probability.

For binary classification: uncertainty ∝ prediction set size. Set = {0,1} → uncertain. Set = {1} → confident.

### Why We Use It
- **Trust Score** — uncertainty feeds into the "model confidence" display.
- **PS requirement:** *"Surface a stability indicator for each explanation."*
- Conformal prediction is more rigorous than raw `predict_proba` confidence (which is often poorly calibrated).

### Appropriateness for OULAD
**Appropriate.**

| Factor | Assessment |
|--------|------------|
| Calibration | GBM probabilities + CalibratedClassifierCV + MAPIE = well-calibrated uncertainty |
| Binary classification | MAPIE's "set size" heuristic works cleanly for 2-class problems |
| Computational cost | Negligible — single `predict()` call |

### Could We Improve or Replace?
| Alternative | Verdict |
|-------------|---------|
| **MC Dropout** | Requires neural network with dropout. Not applicable to GBM. |
| **Ensemble variance** | Use variance across GBM's internal trees. Simpler but less principled than conformal. |
| **Raw predict_proba entropy** | Too naive — GBM probabilities are often overconfident. |
| **Venn-ABERS** | Another conformal method. Similar guarantees. MAPIE is better supported. |

**Verdict: MAPIE is the correct choice for sklearn models. No change needed.**

### Where It's Used (Layers)
1. **Prediction Layer** — uncertainty + confidence_width in `/predict` response
2. **Evaluation Layer** — feeds into displayed model confidence

---

## 9. DoWhy (Causal Inference Layer)

### What It Is
DoWhy provides a formal causal inference framework: define a causal DAG (Directed Acyclic Graph), identify estimands via do-calculus, and estimate causal effects via backdoor/frontdoor adjustment. In this system, it's used to **annotate** SHAP features as "causal" vs "correlational."

### Why We Use It
- **Brownie Points** — "Build a causal DAG over learner features using do-calculus or SEM, annotating elements as causal or correlational."
- **Action prioritization** — Causal features get 1.0× weight in the Action Ranker; correlational features get 0.7×. This prevents recommending actions on spuriously correlated features.
- **Educational relevance** — High `quiz_avg_score` *correlates* with low dropout but doesn't *cause* it. `assignment_submission_rate` has a more direct causal link.

### Appropriateness for OULAD
**Appropriate with important caveats.**

| Factor | Assessment |
|--------|------------|
| DAG validity | The causal DAG is **hand-specified** based on domain knowledge. It's plausible but not validated from data. This is standard practice. |
| Identifiability | Backdoor adjustment works if no unmeasured confounders exist. In OULAD, this is likely violated (e.g., student motivation is unmeasured). |
| Causal vs correlational | The binary annotation is a simplification — in reality, all features have some mediated causal effect. |
| Practical value | Even approximate causal annotations dramatically improve action quality vs. naive SHAP ranking. |

### Speed Assessment
- **DAG + annotation:** <1ms (pre-computed dictionary lookup).
- **Causal effect estimation (DoWhy):** ~100-500ms. Only needed for the brownie point visualization, not per-request.

### Could We Improve or Replace?
| Alternative | Verdict |
|-------------|---------|
| **PC/FCI algorithm** | Learn the DAG from data instead of hand-specifying. More principled but needs large N and assumes faithfulness. With OULAD's ~30K samples, feasible. **Worth exploring as enhancement.** |
| **SEM (Structural Equation Modeling)** | Simultaneous estimation of all paths. More comprehensive but harder to implement in 24h. |
| **No causal layer** | Falls back to naive SHAP ranking for actions. Loses the causal vs correlational distinction. Significant quality loss. |
| **Granger causality on time series** | OULAD has temporal data (VLE interactions over weeks). Could detect temporal causal patterns. **Interesting enhancement** but complex. |

**Recommendation:** The current approach (hand-specified DAG + annotation) is pragmatic and correct for a hackathon. For a production system, learn the DAG from data.

### Where It's Used (Layers)
1. **Diagnosis Layer** — causal vs correlational annotations on SHAP features
2. **Prescription Layer** — Action Ranker weights causal actions higher (1.0 vs 0.7)
3. **Frontend** — Causal DAG visualization (brownie), causal badges on action cards

### Relation to Problem Statement
Maps to brownie: *"Causal Inference Layer: Build a causal DAG over learner features using do-calculus or SEM."* Also strengthens the core prescription: causal actions → higher confidence in estimated impact.

---

## 10. LIME (Concept-Level Explanations)

### What It Is
LIME (Ribeiro et al., 2016) fits a **local linear model** around a prediction by sampling perturbed instances and weighting them by proximity. The linear coefficients serve as local feature attributions.

In this system, LIME is used **over concept groups** (not individual features) to produce concept-level explanations:
- **Engagement** = f(login_frequency, avg_session_duration, forum_posts, video_completion)
- **Assessment Performance** = f(quiz_avg_score, quiz_completion, assignment_submission)
- **Consistency** = f(days_since_last_activity, missed_deadlines)

### Why We Use It
- **Concept-level explanation** (mandatory) — "Simply showing raw feature names is not acceptable."
- Grouping 12 features into 3 concepts makes explanations digestible for learners.
- LIME over concept groups tells you "Engagement is 31% of your risk" rather than 4 separate numbers.

### Appropriateness for OULAD
**Appropriate for concept grouping, but there's a concern about redundancy with SHAP.**

| Factor | Assessment |
|--------|------------|
| Concept grouping | Excellent — maps naturally to educational domains |
| Faithfulness | LIME is less faithful than SHAP (local linear approximation). The groups add another approximation layer. |
| Speed | ~100-300ms per explanation |
| Redundancy | SHAP values can be grouped post-hoc: `engagement_shap = sum(SHAP[login, session, forum, video])`. This is exact and free. |

### Could We Improve or Replace?
| Alternative | Verdict |
|-------------|---------|
| **Grouped SHAP** | Sum SHAP values by concept group. Exact, zero-cost, theoretically grounded. **Strongly recommended as replacement.** |
| **TCAV (Testing with Concept Activation Vectors)** | Mentioned in the architecture doc. Requires a neural model + concept datasets. Overkill for this setting. |
| **LIME with grouped perturbation** | Perturb at the concept level (all engagement features together). Better than feature-level LIME but still approximate. |

**⚠️ RECOMMENDATION: Replace LIME concept groups with grouped SHAP sums.** You already compute TreeSHAP for every feature. Simply aggregate:
```python
concept_scores = {
    "engagement": sum(shap[f] for f in ["login_frequency_weekly", "avg_session_duration_min", "forum_posts_count", "video_completion_rate"]),
    "assessment": sum(shap[f] for f in ["quiz_avg_score", "quiz_completion_rate", "assignment_submission_rate"]),
    "consistency": sum(shap[f] for f in ["days_since_last_activity", "missed_deadlines_count"]),
}
```
This is **exact** (no approximation), **free** (no extra computation), and **faithful** (satisfies SHAP efficiency axiom at concept level too).

### Where It's Used (Layers)
1. **Concept Level (Explanation Layer)** — concept group scores
2. **Learner View** — simplified concept bars

### Relation to Problem Statement
Maps to: *"Generate explanations at the concept-level. Simply showing raw feature names is not acceptable."* Grouped SHAP achieves this better than LIME.

---

## 11. JSD Drift Detector (Explanation Consistency)

### What It Is
Tracks SHAP attribution distributions over time using **Jensen-Shannon Divergence** between consecutive explanations for the same learner. Also monitors top-3 feature rank shifts.

- **JSD > 0.15** or **top-3 rank shift > 1 feature** → drift alert.

### Why We Use It
- **Mandatory MVP** — "Store the top three contributing features per explanation across sessions and surface a timeline. Visibly flag any inconsistencies."
- Builds trust — if explanations wildly change between sessions, users lose confidence.

### Appropriateness for OULAD
**Very appropriate.**

| Factor | Assessment |
|--------|------------|
| Temporal data | OULAD has multi-session learner data — perfect for tracking explanation evolution |
| JSD threshold | 0.15 is reasonable but should be calibrated on actual SHAP distributions |
| Rank shift | Top-3 is appropriate for 12 features |

### Could We Improve?
| Enhancement | Verdict |
|-------------|---------|
| **Adaptive thresholds** | Compute JSD baseline from training data; set threshold at mean + 2σ. More principled than fixed 0.15. |
| **PSI (Population Stability Index)** | Alternative to JSD. More commonly used in production ML monitoring. Similar concept. |
| **Track at concept level too** | Flag drift in concept-level explanations, not just feature-level. Easy addition. |

**Verdict: Current approach is sound. Minor calibration recommended.**

---

## 12. Trust Score Composite (Explanation Quality)

### What It Is
A weighted composite metric:
```
Trust = 0.40 × Fidelity + 0.35 × Stability + 0.25 × Completeness
```

- **Fidelity:** `1 - |predict_proba - (SHAP_sum + base_value)|`
- **Stability:** `1 - normalized_std(SHAP ranks under noise, n=20)`
- **Completeness:** `|top5_SHAP_sum| / |total_SHAP_sum|`

### Why We Use It
- **Mandatory MVP** — "Surface a stability indicator for each explanation."
- **Brownie** — "Explanation Quality Evaluation: Compute a trust score combining fidelity, stability, and completeness."

### Appropriateness
**Well-designed.** The three components cover the key failure modes:
1. Explanation doesn't match model (fidelity)
2. Explanation is fragile to noise (stability)
3. Explanation misses important factors (completeness)

### Could We Improve?
| Enhancement | Verdict |
|-------------|---------|
| **Weight tuning** | Current weights (0.4/0.35/0.25) are reasonable defaults. Could be tuned based on user feedback data. |
| **Add consistency** | 4th component: how much the explanation changed from the previous session (from drift detector). |
| **Stability sampling** | n=20 perturbation runs adds ~400ms. Could reduce to n=10 with minimal quality loss. **Recommended for speed.** |

---

## 13. Monte Carlo Dropout / MAPIE — Epistemic Uncertainty (Detailed)

### What It Is
The architecture documents reference **MC Dropout** (running N forward passes with dropout active to get a distribution over predictions) as a novel uncertainty technique. However, MC Dropout requires a **neural network with dropout layers** — it is **not applicable to GBM/RF models**.

The actual implementation uses **MAPIE (Model Agnostic Prediction Interval Estimator)** as a drop-in replacement. MAPIE uses **conformal prediction** to produce set-valued predictions with formal coverage guarantees. For binary classification:
- Prediction set = `{1}` → model is confident it's high-risk
- Prediction set = `{0, 1}` → model is uncertain
- Prediction set = `{0}` → model is confident it's low-risk

The "uncertainty" metric = prediction set size. Set size 2 = maximally uncertain, set size 1 = confident.

### Why It Matters for This System
Uncertainty quantification serves three roles:

1. **User-facing trust** — "This prediction has HIGH confidence" vs "⚠️ This prediction is uncertain — use with caution." The PS requires *"surface a stability indicator for each explanation."*
2. **Trust Score input** — Model confidence feeds into the displayed reliability metrics.
3. **Instructor decision support** — An instructor should intervene differently for a HIGH-risk prediction with HIGH confidence vs HIGH-risk with LOW confidence.

### Appropriateness for OULAD + GBM

| Factor | MC Dropout | MAPIE (Conformal) |
|--------|-----------|-------------------|
| **Model compatibility** | ❌ Requires neural net with dropout | ✅ Wraps any sklearn estimator |
| **Theoretical guarantee** | Approximate Bayesian inference | ✅ Formal coverage guarantee (frequentist) |
| **Calibration** | Depends on dropout rate tuning | ✅ Distribution-free — works regardless of model calibration |
| **Speed** | N forward passes (~N×5ms for NN) | Single `predict()` call (<1ms) |
| **Implementation** | Need to build NN surrogate + dropout | `MapieClassifier(estimator=gbm, method="score", cv=5)` — 3 lines |

**MAPIE is the correct choice for GBM.** MC Dropout is mentioned in the algorithms doc as a conceptual reference but should NOT be implemented — it would require an unnecessary neural network surrogate just for uncertainty, adding complexity with no benefit over MAPIE's conformal approach.

### Could We Improve?
| Alternative | Verdict |
|-------------|---------|
| **Ensemble variance** | Use variance across GBM's internal estimators (`model.estimators_`). Simpler but no formal coverage guarantee. |
| **Platt scaling + entropy** | Calibrate probabilities, then use predictive entropy. Less rigorous than conformal. |
| **Venn-ABERS predictors** | Another conformal method with multi-probability outputs. Similar to MAPIE, slightly more complex. |
| **Combine MAPIE + SHAP stability** | Use MAPIE for *prediction* uncertainty and SHAP perturbation for *explanation* uncertainty. Display both. **Already in the architecture — this is the right approach.** |

### Where It's Used (Layers)
1. **Prediction Layer** — `uncertainty` and `confidence_width` in `/predict` response
2. **Evaluation Layer** — contributes to displayed model confidence
3. **Frontend** — uncertainty badge on prediction card ("High/Medium/Low confidence")
4. **Instructor View** — precise confidence intervals for decision-making

### Relation to Problem Statement
Maps to: *"Surface a stability indicator for each explanation"* and *"model confidence"* in the `/predict` response. Also strengthens the brownie point: *"Explanation Quality Evaluation: Compute a trust score combining fidelity, stability, and completeness."*

---

## 13b. Monte Carlo Simulation for Temporal Trajectory Forecasting ⭐ POTENTIAL ADDITION

### What It Is (Different from MC Dropout)
**Monte Carlo Simulation** here means: given a learner's current feature state, simulate thousands of possible *future trajectories* by sampling from learned transition distributions, then run the GBM on each trajectory to get a **distribution over future risk scores**.

This is fundamentally different from MC Dropout:
- **MC Dropout** → uncertainty about the model's current prediction (epistemic)
- **MC Simulation** → uncertainty about the learner's future behavior and its impact on predictions (aleatoric + temporal)

### How It Would Work

```
Week 5: Learner has quiz_completion_rate = 0.35, login_freq = 2.0, ...
           │
           ▼
Step 1: Learn transition model from OULAD historical data
        P(features_week_6 | features_week_5) — per feature
        e.g., login_freq follows N(μ=current ± drift, σ=learned_std)
           │
           ▼
Step 2: Sample N=1000 future trajectories (weeks 6-12)
        Each trajectory: [features_w6, features_w7, ..., features_w12]
        Features evolve stochastically based on transition distributions
           │
           ▼
Step 3: Run GBM.predict_proba() on each sampled future state
        → 1000 risk scores at week 8, week 10, week 12
           │
           ▼
Step 4: Aggregate
        → "By week 10, there's a 73% chance risk exceeds 0.7"
        → "If current trend continues: risk = 0.82 ± 0.14 by week 12"
        → Confidence fan chart (like weather forecasts)
```

### Why This Is Relevant to OULAD

OULAD has a critical property the current architecture doesn't exploit: **temporal progression**. Students are at week 1, 2, ..., up to ~36 weeks. Their features evolve over time. The current system only explains the *current* snapshot — it doesn't answer:

- "If this student continues on this trajectory, what happens by exam week?"
- "How likely is dropout by week 10 vs week 15?"
- "What's the window of opportunity to intervene?"

MC simulation answers all three.

| OULAD Property | MC Simulation Benefit |
|---------------|----------------------|
| **VLE logs per week** | Learn per-week feature transition distributions from historical cohorts |
| **Multiple course presentations** | Same course taught multiple times = multiple cohorts to learn transitions from |
| **Known final outcomes** | Can validate simulated trajectories against actual historical trajectories |
| **Early warning** | A week-5 student can see projected risk at week 10 — much more actionable than just "current risk is 0.6" |

### How It Integrates with XAI

This is where it gets powerful — MC simulation isn't just prediction, it's **explainable forecasting**:

1. **SHAP on simulated futures** — Run TreeSHAP on the median projected week-10 state. "By week 10, your biggest risk factor is projected to be missed_deadlines (currently 2 → projected 5)."

2. **Counterfactual trajectories** — Compare simulated trajectory with and without an intervention. "If you increase login frequency from 2→4 starting now, projected week-10 risk drops from 0.82→0.51." This makes DiCE actions **time-aware**.

3. **Confidence fan chart** — Show a band of possible futures on a timeline. Wide band = uncertain trajectory, narrow band = predictable path. Directly maps to the PS stability indicator.

4. **Temporal prescriptions** — "You need to act within the next 2 weeks — after week 8, the projected risk becomes very hard to reverse." This is something no current technique in the stack provides.

### Implementation Sketch

```python
class TrajectorySimulator:
    def __init__(self, historical_data: pd.DataFrame, model):
        self.model = model
        # Learn per-feature transition distributions from historical cohorts
        # P(feature_t+1 | feature_t) per week
        self.transitions = self._learn_transitions(historical_data)
    
    def _learn_transitions(self, data: pd.DataFrame) -> dict:
        """From OULAD: group by student, compute week-over-week deltas per feature."""
        transitions = {}
        for feature in LEARNER_FEATURES:
            deltas = []  # week-over-week changes across all historical students
            # ... compute from temporal OULAD data
            transitions[feature] = {"mean_delta": np.mean(deltas), "std_delta": np.std(deltas)}
        return transitions
    
    def simulate(self, current_features: dict, current_week: int, 
                 target_week: int, n_simulations: int = 1000) -> dict:
        trajectories = []
        for _ in range(n_simulations):
            features = current_features.copy()
            weekly_risk = []
            for week in range(current_week + 1, target_week + 1):
                # Evolve each feature stochastically
                for f in ACTIONABLE_FEATURES:
                    delta = np.random.normal(
                        self.transitions[f]["mean_delta"],
                        self.transitions[f]["std_delta"]
                    )
                    features[f] = np.clip(features[f] + delta, *FEATURE_RANGES[f])
                # Predict risk at this future week
                risk = self.model.predict_proba([list(features.values())])[0][1]
                weekly_risk.append(risk)
            trajectories.append(weekly_risk)
        
        trajectories = np.array(trajectories)  # (1000, n_weeks)
        return {
            "median_trajectory": np.median(trajectories, axis=0).tolist(),
            "p10": np.percentile(trajectories, 10, axis=0).tolist(),
            "p90": np.percentile(trajectories, 90, axis=0).tolist(),
            "risk_at_target": {
                "median": float(np.median(trajectories[:, -1])),
                "std": float(np.std(trajectories[:, -1])),
                "prob_high_risk": float((trajectories[:, -1] > 0.7).mean()),
            },
            "weeks": list(range(current_week + 1, target_week + 1)),
        }
```

### Speed Assessment
- **1000 simulations × 7 weeks × GBM predict:** ~1000 × 7 × 0.1ms ≈ **700ms**
- Acceptable for a dedicated `/forecast` endpoint (not real-time)
- Can reduce to 200 simulations for ~140ms if needed

### Should We Add It?

| Argument For | Argument Against |
|-------------|-----------------|
| Unique differentiator — no other team will have temporal XAI | Adds implementation complexity (~2-3 hours) |
| Directly exploits OULAD's temporal structure | Requires feature engineering of week-over-week transitions |
| "When to act" is more actionable than "what to change" | Not in the PS MVP checklist — judges may not weight it |
| Fan charts are visually stunning in demos | Another endpoint + frontend component |

**Verdict: Strong candidate for brownie points if time permits.** It would be a new endpoint (`POST /forecast`) and a single frontend component (confidence fan chart). The OULAD temporal data makes it uniquely feasible. It directly strengthens the "Prescription" story: not just *what* to change but *when* to act.

---

## 14. Human-in-the-Loop Feedback System

### What It Is
A feedback collection and tracking system that allows users to:
- **Rate** explanations (1-5 stars)
- **Report** whether they followed the recommended action
- **Correct** feature values or interpretations ("I was on a planned break, inactivity shouldn't count")

### Architecture & Data Flow

```
User views explanation
        │
        ▼
Rate explanation (1-5) + "Did you follow the recommendation?" (yes/no)
        │
        ├── Optional: Submit correction
        │     {feature: "days_since_last_activity", comment: "Planned break"}
        │
        ▼
POST /feedback
        │
        ▼
FeedbackStore (SQLite)
        │
        ├── FeedbackRecord(id, learner_id, explanation_id, rating,
        │                   followed_recommendation, correction_feature,
        │                   correction_comment, timestamp)
        │
        ▼
GET /feedback/stats → aggregate metrics
        │
        ├── avg_rating
        ├── recommendation_follow_rate
        ├── high_trust_follow_rate vs low_trust_follow_rate
        ├── top_corrected_features
        └── per-feature follow rates
```

### Where Feedback Is Managed

| Component | Location | Role |
|-----------|----------|------|
| **Backend endpoint** | `POST /feedback` in `main.py` | Receives feedback submissions |
| **Backend endpoint** | `GET /feedback/stats` in `main.py` | Returns aggregate statistics |
| **Storage** | `backend/app/tracker/feedback_store.py` | SQLite table `FeedbackRecord` via SQLAlchemy |
| **Frontend** | Feedback widget on ExplanationCard | Star rating + toggle + correction form |
| **Analytics** | `/feedback/stats` response | Tracks whether high-trust explanations lead to higher action follow rates |

### Why It Matters

1. **Brownie point** — *"Human-in-the-Loop Feedback: Allow users to rate explanations, submit corrections, and feed signals back for model refinement."*
2. **Trust calibration** — If users rate explanations with high trust scores higher than low trust scores, it validates the Trust Score formula.
3. **Feature correction** — Corrections like "I was on a planned break" reveal contextual information the model doesn't have. These can feed into future feature engineering.
4. **Follow-rate tracking** — `recommendation_follow_rate` directly measures whether the system is *actionable* — the core PS requirement.

### How Feedback Feeds Back

Currently, feedback is **stored and reported** but doesn't close the loop back to the model. For a full human-in-the-loop cycle:

| Feedback Signal | How It Could Feed Back |
|----------------|----------------------|
| **Low ratings on specific features** | Down-weight those features in the Action Ranker's `FEATURE_ACTIONABILITY_WEIGHTS` |
| **Corrections** | Add corrected values as ground truth for re-training; flag the feature as "contextual" |
| **Low follow rate for an action** | Reduce that action's priority in future recommendations |
| **High follow rate + outcome improvement** | Increase confidence in DiCE's counterfactual for that feature |

For the hackathon MVP, storing + reporting is sufficient. The feedback stats endpoint gives judges proof that the system is designed for iterative improvement.

### Relation to Problem Statement
Maps to brownie: *"Allow users to rate explanations, submit corrections, and feed signals back for model refinement."* Also supports: *"track if users change behavior based on recommendations"* via `followed_recommendation` + `recommendation_follow_rate`.

---

# PART 2 — ARCHITECTURE AUDIT

---

## Layer-by-Layer XAI Integration Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION LAYER                              │
│                                                                  │
│  GBM.predict_proba(X) → risk_score (0-1)                       │
│  RF.predict_proba(X) → comparison_score (brownie)               │
│  MAPIE.predict(X) → uncertainty interval                        │
│                                                                  │
│  XAI here: NONE — pure prediction                               │
│  OULAD mapping: 12 derived features → dropout probability       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    EXPLANATION LAYER                              │
│                                                                  │
│  Feature Level:                                                  │
│    TreeSHAP → per-feature signed attributions                   │
│    FastSHAP → real-time approximations (What-If only)           │
│    Archipelago → pairwise interaction scores                    │
│                                                                  │
│  Concept Level:                                                  │
│    LIME groups → engagement/assessment/consistency scores        │
│    ★ RECOMMEND: Replace with grouped SHAP sums                  │
│    Prototypes → similar learner matches                         │
│                                                                  │
│  Prediction Level:                                               │
│    Global SHAP → mean |SHAP| across dataset                    │
│    Anchors → IF-THEN rules with precision guarantees            │
│                                                                  │
│  XAI techniques: 6 (SHAP, FastSHAP, Archipelago, LIME,         │
│                     Prototypes, Anchors)                         │
│  Parallel execution: SHAP + Anchors + Prototypes + Archipelago  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    DIAGNOSIS LAYER                                │
│                                                                  │
│  CEM PP → minimal sufficient features for prediction            │
│  CEM PN → minimal changes to flip prediction                    │
│  ★ CONCERN: Redundant with DiCE (PN) + SHAP top-k (PP)         │
│  ★ CONCERN: Requires Keras surrogate + TensorFlow dependency     │
│                                                                  │
│  Causal Annotator → tags features as causal/correlational       │
│  DoWhy SCM → estimates causal effects (optional)                │
│                                                                  │
│  XAI techniques: 2 (CEM, DoWhy)                                │
│  Synchronous execution (CEM is slow ~1-2s)                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    PRESCRIPTION LAYER                             │
│                                                                  │
│  DiCE → diverse counterfactuals with actionability constraints  │
│  ActionRanker → priority = SHAP × actionability × causality    │
│  Output: ranked PrescriptiveAction list with estimated impact   │
│                                                                  │
│  XAI techniques: 1 (DiCE)                                       │
│  This is the MOST CRITICAL layer for the PS                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    EVALUATION + TRACKING                          │
│                                                                  │
│  TrustScorer → fidelity + stability + completeness composite    │
│  DriftDetector → JSD on SHAP distributions + rank shift         │
│  ExplanationStore → SQLite history of top-3 features per session│
│                                                                  │
│  XAI techniques: 0 (meta-evaluation of XAI outputs)             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    NARRATION LAYER                                │
│                                                                  │
│  Claude Sonnet → narrates pre-computed data ONLY                │
│  Learner view: motivational, plain language, ≤100 words         │
│  Instructor view: technical, data-driven, ≤150 words            │
│                                                                  │
│  ⚠️ LLM NEVER generates explanations — only narrates            │
└─────────────────────────────────────────────────────────────────┘
```

---

## PS Requirement → XAI Technique Mapping

| # | MVP Requirement | Techniques Used | Sufficient? |
|---|----------------|-----------------|-------------|
| 1 | **Multi-Layer Explanation Engine** (prediction + feature + concept) | TreeSHAP (feature), LIME/Grouped SHAP (concept), Global SHAP + Anchors (prediction) | ✅ Yes — all 3 levels covered |
| 2 | **Faithful XAI Integration** (link to model internals + stability) | TreeSHAP (exact from tree internals), SHAP stability score, Trust Score | ✅ Yes — TreeSHAP is the gold standard for faithfulness |
| 3 | **Interactive What-If** (real-time model feedback) | FastSHAP + GBM.predict_proba() | ✅ Yes — but test if TreeSHAP alone suffices |
| 4 | **Counterfactual Engine** (minimal changes to flip) | DiCE with actionability constraints | ✅ Yes — directly addresses requirement |
| 5 | **Dual-Audience Adaptive** (learner vs instructor) | Prototypes + LIME (learner), SHAP + Anchors (instructor), LLM narrator | ✅ Yes — each audience has distinct explainers |
| 6 | **Consistency & Tracking** (top-3 timeline + drift flags) | ExplanationStore + JSD Drift Detector | ✅ Yes — stores top-3, computes JSD, flags rank shifts |
| 7 | **Action Recommendation** (prescriptive AI) | DiCE + ActionRanker + CausalAnnotator | ✅ Yes — actions derived from model, not hardcoded |

| # | Brownie Point | Techniques Used | Status |
|---|--------------|-----------------|--------|
| B1 | Causal Inference Layer | DoWhy SCM + causal annotation | ✅ Planned |
| B2 | Explanation Quality Evaluation | TrustScorer composite | ✅ Planned |
| B3 | Multi-Model Comparison | GBM vs RF parallel predictions | ✅ Planned |
| B4 | Drift-Aware Monitoring | JSD Drift Detector | ✅ Planned |
| B5 | Human-in-the-Loop Feedback | POST /feedback + FeedbackStore | ✅ Planned |
| B6 | Deployment | Docker Compose | ✅ Planned |
| B7 | XAI SDK | xai-learner-sdk | ✅ Planned |

---

## Critical Findings & Recommendations

### 🔴 HIGH PRIORITY

1. **Drop or defer CEM** — It's the slowest component (~2s), requires TensorFlow + Keras surrogate, and its PP/PN outputs overlap with existing techniques (SHAP top-k ≈ PP, DiCE ≈ PN). This is the single biggest simplification you can make.

2. **Replace LIME concept groups with grouped SHAP sums** — You already compute TreeSHAP. Sum by concept group for free. Eliminates LIME dependency, is more faithful, and is instantaneous.

3. **Test TreeSHAP vs FastSHAP latency** — If TreeSHAP at 50ms + 300ms debounce = 350ms total round-trip is acceptable UX, skip FastSHAP entirely. Saves surrogate training time and eliminates accuracy gap.

### 🟡 MEDIUM PRIORITY

4. **Switch DiCE to `method="genetic"`** — Better counterfactual diversity, 1-3s latency is acceptable for non-real-time endpoint.

5. **Reduce stability perturbation runs from n=20 to n=10** — Saves ~200ms per explanation with negligible quality loss.

6. **Calibrate JSD drift threshold** — Compute baseline JSD distribution from training data rather than hardcoding 0.15.

7. **OULAD feature engineering** — The 12 features are synthetic in the current plan. When switching to real OULAD, you need to derive them from VLE interaction logs + assessment data. Map:
   - `login_frequency_weekly` ← VLE log counts per week
   - `days_since_last_activity` ← max(current_date - last_VLE_date)
   - `assignment_submission_rate` ← submitted_TMAs / total_TMAs
   - `quiz_avg_score` ← mean assessment scores
   - etc.

### 🟢 NICE TO HAVE

8. **Outcome-aware prototype filtering** — Show at least one similar learner who succeeded to maximize motivational impact.

9. **Learn causal DAG from data** — Use PC algorithm on OULAD to discover the DAG instead of hand-specifying.

10. **Pre-compute global explanations at startup** — Global SHAP summary, PDP plots, and prototype clusters can be computed once and cached.

---

## Revised Minimal Architecture (Recommended)

If you apply the high-priority recommendations, the stack simplifies to:

```
PREDICTION:   GBM + MAPIE
EXPLANATION:  TreeSHAP (feature + concept via grouping + prediction via global)
              Anchors (rules for instructors)
              Prototypes (k-NN for learners)
              Archipelago (interactions)
DIAGNOSIS:    Causal Annotator (DoWhy DAG annotation)
              [CEM dropped — SHAP top-k + DiCE cover PP/PN]
PRESCRIPTION: DiCE (counterfactuals → actions)
              ActionRanker (SHAP × actionability × causality)
EVALUATION:   TrustScorer + DriftDetector + ExplanationStore
NARRATION:    Claude Sonnet (narration only)
```

**Dependencies eliminated:** TensorFlow, Keras, LIME, (optionally) FastSHAP, aix360
**Time saved per explanation:** ~2-3 seconds (CEM removal + reduced perturbation runs)
**Complexity reduction:** 3 fewer explainer classes, no surrogate training

---

## OULAD Dataset Considerations

The OULAD (Open University Learning Analytics Dataset) has specific characteristics that affect XAI choices:

| OULAD Property | Implication for XAI |
|---------------|---------------------|
| **~30,000 learners** | Sufficient for SHAP stability, good prototype pool |
| **7 courses, multiple presentations** | Can compute per-course explanations (feature importance varies by course) |
| **VLE interaction logs** | Rich temporal data → derive engagement features at granular level |
| **Assessment scores** | Direct feature → clean signal for quiz/assignment rates |
| **Binary outcome** (pass/fail + withdrawn) | Clean binary classification target. Consider also treating "distinction/pass/fail/withdrawn" as ordinal for richer prediction. |
| **Temporal structure** | Students progress through weeks → `current_week_in_course` is a natural immutable feature. Explanations should evolve as more data accumulates per student. |
| **Class imbalance** | ~30-40% withdrawal rate in OULAD. Use SMOTE or class weights in GBM. This affects SHAP base_value. |

---

## Summary

The architecture is **well-designed** and covers all 7 MVP requirements + 7 brownie points. The XAI technique selection is broadly excellent — TreeSHAP, DiCE, Anchors, Prototypes, and MAPIE are all optimal choices for a tree-based model on tabular educational data.

The three actionable improvements are:
1. **Drop CEM** (saves 2s, removes TF dependency, no capability loss)
2. **Replace LIME with grouped SHAP** (free, more faithful)
3. **Test TreeSHAP before committing to FastSHAP** (may be unnecessary complexity)

Everything else is solid and directly maps to the problem statement requirements.
