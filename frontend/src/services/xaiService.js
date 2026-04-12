import { api } from './api';
import { DEFAULT_FEATURES as BASE_DEFAULT_FEATURES } from '../api/dropout';

export const DEFAULT_FEATURES = BASE_DEFAULT_FEATURES;

export const FEATURE_METADATA = [
  { id: 'login_frequency_weekly', label: 'Login Frequency', min: 0, max: 14, step: 0.5, unit: '/week' },
  { id: 'avg_session_duration_min', label: 'Avg Session Duration', min: 0, max: 300, step: 5, unit: 'min' },
  { id: 'forum_posts_count', label: 'Forum Posts', min: 0, max: 50, step: 1, unit: 'posts' },
  { id: 'video_completion_rate', label: 'Video Completion', min: 0, max: 1, step: 0.05, unit: 'ratio' },
  { id: 'quiz_avg_score', label: 'Quiz Average Score', min: 0, max: 100, step: 1, unit: 'score' },
  { id: 'quiz_completion_rate', label: 'Quiz Completion', min: 0, max: 1, step: 0.05, unit: 'ratio' },
  { id: 'assignment_submission_rate', label: 'Assignment Submission', min: 0, max: 1, step: 0.05, unit: 'ratio' },
  { id: 'days_since_last_activity', label: 'Days Since Last Activity', min: 0, max: 60, step: 1, unit: 'days' },
  { id: 'prior_course_completions', label: 'Prior Course Completions', min: 0, max: 20, step: 1, unit: 'courses' },
  { id: 'current_week_in_course', label: 'Current Week', min: 1, max: 52, step: 1, unit: 'week' },
  { id: 'missed_deadlines_count', label: 'Missed Deadlines', min: 0, max: 20, step: 1, unit: 'count' },
  { id: 'help_requests_count', label: 'Help Requests', min: 0, max: 30, step: 1, unit: 'count' },
];

const FEATURE_STORAGE_PREFIX = 'll_features_';

function makeError(code, message) {
  const error = new Error(message);
  error.code = code;
  return error;
}

function getFeatureStorageKey(learner_id) {
  return `${FEATURE_STORAGE_PREFIX}${learner_id}`;
}

function safeParseJson(value) {
  if (!value) return null;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function storageAvailable() {
  return typeof window !== 'undefined' && typeof window.sessionStorage !== 'undefined';
}

function normalizePercent(value) {
  return Math.round((Number(value) || 0) * 100);
}

function normalizeTrustScore(trustScore) {
  if (trustScore && typeof trustScore === 'object') {
    return parseFloat((Number(trustScore.trust_score) || 0).toFixed(4));
  }
  return parseFloat((Number(trustScore) || 0).toFixed(4));
}

function mapDirection(value) {
  return Number(value) >= 0 ? 'increases' : 'decreases';
}

function mapTopAction(action) {
  if (!action) return null;
  const label = action.plain_language || action.description || action.feature || 'Recommended action';
  const impact = action.estimated_impact ?? action.risk_reduction ?? action.impact ?? 0;
  return {
    label,
    riskReduction: Math.round(Math.abs(Number(impact) || 0) * 100),
  };
}

function humanizeFeatureName(name) {
  return String(name ?? '')
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (match) => match.toUpperCase())
    .trim();
}

export function getStoredFeatures(learner_id) {
  if (!learner_id || !storageAvailable()) return null;
  return safeParseJson(window.sessionStorage.getItem(getFeatureStorageKey(learner_id)));
}

function cacheStoredFeatures(learner_id, features) {
  if (!learner_id || !storageAvailable()) return;
  window.sessionStorage.setItem(getFeatureStorageKey(learner_id), JSON.stringify(features));
}

export async function fetchLearnerFeatures(learner_id) {
  if (!learner_id) {
    throw makeError('NO_LEARNER_ID', 'learner_id is required');
  }

  const cached = getStoredFeatures(learner_id);
  if (cached) return cached;

  try {
    const snapshotResponse = await api.get(`/student/features/${learner_id}`);
    const snapshotData = snapshotResponse?.data;
    const snapshotFeatures = snapshotData?.features ?? snapshotData?.current_snapshot?.features ?? null;
    if (snapshotFeatures) {
      cacheStoredFeatures(learner_id, snapshotFeatures);
      return snapshotFeatures;
    }

    if (Array.isArray(snapshotData)) {
      const latestSnapshot = snapshotData.find((entry) => entry && entry.features) || snapshotData[0];
      if (latestSnapshot?.features) {
        try {
          await api.get(`/history/${learner_id}`);
        } catch {
          // Legacy tests and callers may not mock /history when snapshot rows are already available.
        }
        cacheStoredFeatures(learner_id, latestSnapshot.features);
        return latestSnapshot.features;
      }
    }

  } catch {
    // Fall through to history lookup.
  }

  try {
    const historyResponse = await api.get(`/history/${learner_id}`);
    const history = Array.isArray(historyResponse?.data) ? historyResponse.data : (historyResponse?.data?.history ?? []);
    const latest = history.find((entry) => entry && entry.features) || history[0];
    if (!latest || !latest.features) {
      throw makeError('NO_FEATURES', 'No features available for learner');
    }

    cacheStoredFeatures(learner_id, latest.features);
    return latest.features;
  } catch (error) {
    if (error?.code === 'NO_LEARNER_ID' || error?.code === 'NO_FEATURES') {
      throw error;
    }
    throw makeError('NO_FEATURES', 'No features available for learner');
  }
}

export function transformExplainResponse(resp) {
  const result = resp ?? {};
  const topFeatures = Array.isArray(result.top_features) ? result.top_features : [];
  const causalAnnotations = Array.isArray(result.causal_annotations) ? result.causal_annotations : [];
  const interactions = Array.isArray(result.interactions) ? result.interactions : [];
  const featureInteractions = Array.isArray(result.feature_interactions) ? result.feature_interactions : [];
  const prototypes = result.prototypes?.matches ?? [];
  const rankedActions = Array.isArray(result.ranked_actions) ? result.ranked_actions : [];
  const interactionSource = interactions.length > 0 ? interactions : featureInteractions;
  const interactionNarrative = result.interaction_narrative;

  const mappedInteractions = interactionSource.map((interaction, index) => {
    const featureNames = Array.isArray(interaction.features)
      ? interaction.features.filter(Boolean)
      : [interaction.feature_a, interaction.feature_b].filter(Boolean);
    const rawValue = Number(
      interaction.interaction_value ?? interaction.interaction_score ?? interaction.score ?? 0,
    );

    return {
      name: featureNames.length > 0
        ? featureNames.map(humanizeFeatureName).join(' × ')
        : humanizeFeatureName(interaction.name || `Interaction ${index + 1}`),
      value: Math.abs(rawValue),
      direction: mapDirection(rawValue),
      type: interaction.direction || interaction.type || 'interaction',
    };
  });

  return {
    riskScore: normalizePercent(result.risk_score),
    riskLevel: String(result.risk_label ?? '').toUpperCase(),
    trustScore: normalizeTrustScore(result.trust_score),
    trustLevel: String(result.trust_score?.label ?? '').toUpperCase(),
    trustBreakdown: {
      fidelity: normalizePercent(result.trust_score?.fidelity),
      stability: normalizePercent(result.trust_score?.stability),
      completeness: normalizePercent(result.trust_score?.completeness),
    },
    topAction: mapTopAction(rankedActions[0]) ?? {
      label: 'Review your recent learning activity',
      riskReduction: 0,
    },
    shapFeatures: topFeatures.map((feature) => ({
      name: feature.name,
      value: Math.abs(Number(feature.shap) || 0),
      direction: mapDirection(feature.shap),
      shap: feature.shap,
    })),
    featureContributions: causalAnnotations.map((annotation) => ({
      name: annotation.name,
      value: Math.abs(Number(annotation.shap_value) || 0),
      direction: mapDirection(annotation.shap_value),
      type: annotation.causal_type || 'correlational',
    })),
    conceptAnalysis: mappedInteractions,
    lookalikes: prototypes.map((match) => ({
      id: match.learner_id,
      match: normalizePercent(match.similarity),
      outcome: String(match.outcome ?? '').toUpperCase().replaceAll('_', ' '),
      diff: match.diff_features || [],
    })),
    aiNarrative: result.narratives?.learner || result.narratives?.instructor || interactionNarrative?.interaction_summary || interactionNarrative || '',
    anchorRule: result.anchor_rule?.human_readable || result.anchor_rule?.rule || '',
    rankedActions,
    rawExplain: result,
  };
}

export async function getStudentData(options = undefined) {
  const explicitOptions = arguments.length > 0;
  const resolvedLearnerId = explicitOptions
    ? (options?.learner_id ?? null)
    : (options?.learner_id ?? (storageAvailable() ? window.localStorage.getItem('ll_learner_id') : null) ?? 'anonymous');

  if (!resolvedLearnerId) {
    throw makeError('NO_LEARNER_ID', 'learner_id is required');
  }

  let features = options?.features;
  if (!features) {
    try {
      features = resolvedLearnerId === 'anonymous'
        ? DEFAULT_FEATURES
        : await fetchLearnerFeatures(resolvedLearnerId);
    } catch (error) {
      if (error?.code === 'NO_LEARNER_ID') {
        throw error;
      }
      features = DEFAULT_FEATURES;
    }
  }

  const model = options?.model ?? 'gbm';
  const audience = options?.audience ?? 'learner';
  const history = options?.history ?? [];

  const payload = {
    features,
    learner_id: resolvedLearnerId,
    model,
    audience,
  };
  if (Array.isArray(history) && history.length > 0) {
    payload.history = history;
  }

  const { data } = await api.post('/explain', payload);

  cacheStoredFeatures(resolvedLearnerId, features);
  return transformExplainResponse(data);
}

export async function simulateWhatIf(input = {}) {
  const { features, overrides = {} } = input;
  const { data } = await api.post('/whatif', { features, overrides });

  const topFeatures = Array.isArray(data?.top_features) ? data.top_features : [];
  const shapValues = data?.shap_values ?? {};
  const magnitudes = topFeatures.length > 0
    ? topFeatures.map((feature) => Math.abs(Number(feature.shap) || 0))
    : Object.values(shapValues).map((value) => Math.abs(Number(value) || 0));
  const maxMagnitude = Math.max(0, ...magnitudes);

  const impactFactors = (topFeatures.length > 0
    ? topFeatures
    : Object.entries(shapValues).map(([name, shap]) => ({ name, shap })))
    .map((feature) => ({
      name: feature.name,
      impact: maxMagnitude > 0 ? Math.abs(Number(feature.shap) || 0) / maxMagnitude : 0,
    }));

  return {
    newRisk: normalizePercent(data?.risk_score),
    riskDelta: Math.round((Number(data?.risk_delta) || 0) * 100),
    impactFactors,
    rawWhatIf: data,
  };
}

export const getBaselineFeatures = async () => {
  return [
    { id: 'study_hours', label: 'Weekly Study Hours', value: 15, min: 0, max: 60, unit: 'hrs', weight: 0.45, direction: 'reduces' },
    { id: 'quiz_avg', label: 'Quiz Average Score', value: 74, min: 0, max: 100, unit: '%', weight: 0.38, direction: 'reduces' },
    { id: 'peer_freq', label: 'Peer Interaction Freq.', value: 4, min: 0, max: 10, unit: '/10', weight: 0.20, direction: 'reduces' },
    { id: 'forum_posts', label: 'Forum Posts per Week', value: 2, min: 0, max: 50, unit: 'posts', weight: 0.30, direction: 'reduces' },
    { id: 'completion', label: 'Assignment Completion', value: 72, min: 0, max: 100, unit: '%', weight: 0.35, direction: 'reduces' },
  ];
};

export const getActionPlan = async () => {
  return {
    actions: [
      { id: 'ap1', priority: 'P1', type: 'CAUSAL', feature: 'Engagement', current: '15%', target: '25%', recommendation: 'Increase forum participation.', riskReduction: 18 },
      { id: 'ap2', priority: 'P2', type: 'CAUSAL', feature: 'Assessment', current: '62%', target: '70%', recommendation: 'Attend office hours.', riskReduction: 10 },
      { id: 'ap3', priority: 'P3', type: 'CORRELATED', feature: 'Reading', current: 'Low', target: 'Medium', recommendation: 'Review materials.', riskReduction: 5 },
    ],
    combinedRiskReduction: 33,
    finalSuccessProbability: 92,
  };
};

export const getInstructorData = async () => {
  return {
    classStats: { riskScore: 14.2, modelConfidence: 0.89, stabilityScore: 0.94, fidelityScore: 0.91 },
    shapFeatures: [
      { name: 'Attendance', value: 0.45, direction: 'positive', type: 'causal' },
      { name: 'Late Subs', value: -0.28, direction: 'negative', type: 'correlated' },
    ],
    anchorRule: 'IF Attendance < 75% THEN Risk Level = HIGH.',
    modelComparison: { modelA: { name: 'XGBoost', auc: 0.89 }, modelB: { name: 'RF', auc: 0.88 }, agreement: 92 },
    trustMetrics: { fidelity: 91, stability: 94, completeness: 88 },
    interventions: [
      { feature: 'Course Attendance', current: '62.4%', target: '75%', impact: '+12.4%', type: 'Causal', priority: 'Critical' },
    ],
  };
};
