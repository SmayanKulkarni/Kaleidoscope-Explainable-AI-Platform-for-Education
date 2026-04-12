import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../services/api', () => ({
  api: { post: vi.fn(), get: vi.fn() },
}));

import { api } from '../services/api';
import {
  fetchLearnerFeatures,
  getStudentData,
  simulateWhatIf,
  getStoredFeatures,
  transformExplainResponse,
} from '../services/xaiService';

const MOCK_FEATURES = {
  login_frequency_weekly:     5,
  avg_session_duration_min:   60,
  forum_posts_count:          3,
  video_completion_rate:      0.8,
  quiz_avg_score:             75,
  quiz_completion_rate:       0.85,
  assignment_submission_rate: 0.9,
  days_since_last_activity:   2,
  prior_course_completions:   2,
  current_week_in_course:     8,
  missed_deadlines_count:     0,
  help_requests_count:        2,
  engagement_latent_1:        0.0,
  engagement_latent_2:        0.0,
  engagement_latent_3:        0.0,
};

const MOCK_EXPLAIN_RESP = {
  risk_score:  0.74,
  risk_label:  'high',
  trust_score: { trust_score: 0.91, fidelity: 0.92, stability: 0.88, completeness: 0.85, label: 'high' },
  top_features:        [],
  causal_annotations:  [],
  prototypes:          { matches: [] },
  ranked_actions:      [],
  anchor_rule:         { human_readable: '' },
  narratives:          { learner: 'Keep it up.', instructor: '' },
  interactions:        [],
};

beforeEach(() => {
  sessionStorage.clear();
  vi.clearAllMocks();
});

describe('fetchLearnerFeatures', () => {
  it('throws NO_LEARNER_ID when learner_id is null', async () => {
    await expect(fetchLearnerFeatures(null)).rejects.toMatchObject({ code: 'NO_LEARNER_ID' });
  });

  it('throws NO_LEARNER_ID when learner_id is empty string', async () => {
    await expect(fetchLearnerFeatures('')).rejects.toMatchObject({ code: 'NO_LEARNER_ID' });
  });

  it('returns cached features from sessionStorage without hitting API', async () => {
    sessionStorage.setItem('ll_features_L001', JSON.stringify(MOCK_FEATURES));
    const result = await fetchLearnerFeatures('L001');
    expect(result).toEqual(MOCK_FEATURES);
    expect(api.get).not.toHaveBeenCalled();
  });

  it('calls GET /history/{learner_id} when no sessionStorage cache', async () => {
    api.get.mockResolvedValueOnce({ data: [{ features: MOCK_FEATURES, risk_score: 0.5 }] });
    const result = await fetchLearnerFeatures('L002');
    expect(api.get).toHaveBeenCalledWith('/history/L002');
    expect(result).toEqual(MOCK_FEATURES);
  });

  it('caches history features in sessionStorage after retrieval', async () => {
    api.get.mockResolvedValueOnce({ data: [{ features: MOCK_FEATURES }] });
    await fetchLearnerFeatures('L003');
    const cached = JSON.parse(sessionStorage.getItem('ll_features_L003'));
    expect(cached).toEqual(MOCK_FEATURES);
  });

  it('throws NO_FEATURES when history returns empty array', async () => {
    api.get.mockResolvedValueOnce({ data: [] });
    await expect(fetchLearnerFeatures('L004')).rejects.toMatchObject({ code: 'NO_FEATURES' });
  });

  it('throws NO_FEATURES when history returns entry without features field', async () => {
    api.get.mockResolvedValueOnce({ data: [{ risk_score: 0.5 }] });
    await expect(fetchLearnerFeatures('L005')).rejects.toMatchObject({ code: 'NO_FEATURES' });
  });

  it('throws NO_FEATURES when history API call throws', async () => {
    api.get.mockRejectedValueOnce(new Error('Network error'));
    await expect(fetchLearnerFeatures('L006')).rejects.toMatchObject({ code: 'NO_FEATURES' });
  });
});

describe('getStudentData', () => {
  it('throws NO_LEARNER_ID when called without learner_id', async () => {
    await expect(getStudentData({ learner_id: null })).rejects.toMatchObject({ code: 'NO_LEARNER_ID' });
  });

  it('calls POST /explain with correct payload when features provided', async () => {
    api.post.mockResolvedValueOnce({ data: MOCK_EXPLAIN_RESP });
    await getStudentData({ features: MOCK_FEATURES, learner_id: 'L007' });
    expect(api.post).toHaveBeenCalledWith('/explain', {
      features:   MOCK_FEATURES,
      learner_id: 'L007',
      model:      'gbm',
      audience:   'learner',
    });
  });

  it('stores features in sessionStorage after successful explain call', async () => {
    api.post.mockResolvedValueOnce({ data: MOCK_EXPLAIN_RESP });
    await getStudentData({ features: MOCK_FEATURES, learner_id: 'L008' });
    const cached = JSON.parse(sessionStorage.getItem('ll_features_L008'));
    expect(cached).toEqual(MOCK_FEATURES);
  });

  it('returns riskScore as integer percentage from explain response', async () => {
    api.post.mockResolvedValueOnce({ data: { ...MOCK_EXPLAIN_RESP, risk_score: 0.63 } });
    const result = await getStudentData({ features: MOCK_FEATURES, learner_id: 'L009' });
    expect(result.riskScore).toBe(63);
  });
});

describe('transformExplainResponse', () => {
  it('maps backend interaction pairs into concept analysis entries', () => {
    const result = transformExplainResponse({
      ...MOCK_EXPLAIN_RESP,
      interactions: [
        {
          feature_a: 'days_since_last_activity',
          feature_b: 'assignment_submission_rate',
          interaction_score: 0.24,
          direction: 'amplifying',
        },
      ],
    });

    expect(result.conceptAnalysis).toEqual([
      {
        name: 'Days Since Last Activity × Assignment Submission Rate',
        value: 0.24,
        direction: 'increases',
        type: 'amplifying',
      },
    ]);
  });
});

describe('simulateWhatIf', () => {
  it('maps risk_score (0–1) → newRisk integer percentage', async () => {
    api.post.mockResolvedValueOnce({
      data: { risk_score: 0.57, risk_delta: 0, shap_values: { x: 0.1 }, top_features: [] },
    });
    const result = await simulateWhatIf({ features: MOCK_FEATURES, overrides: {} });
    expect(result.newRisk).toBe(57);
  });

  it('maps risk_delta (0–1) → riskDelta integer percentage', async () => {
    api.post.mockResolvedValueOnce({
      data: { risk_score: 0.4, risk_delta: -0.15, shap_values: { x: 0.1 }, top_features: [] },
    });
    const result = await simulateWhatIf({ features: MOCK_FEATURES, overrides: {} });
    expect(result.riskDelta).toBe(-15);
  });

  it('normalises impactFactors relative to the largest SHAP magnitude', async () => {
    api.post.mockResolvedValueOnce({
      data: {
        risk_score: 0.4,
        risk_delta: 0,
        shap_values: { a: 0.4, b: 0.2 },
        top_features: [{ name: 'a', shap: 0.4 }, { name: 'b', shap: 0.2 }],
      },
    });
    const result = await simulateWhatIf({ features: MOCK_FEATURES, overrides: {} });
    expect(result.impactFactors[0].impact).toBeCloseTo(1.0, 4);
    expect(result.impactFactors[1].impact).toBeCloseTo(0.5, 4);
  });

  it('handles empty shap_values without throwing', async () => {
    api.post.mockResolvedValueOnce({
      data: { risk_score: 0.3, risk_delta: 0, shap_values: {}, top_features: [] },
    });
    const result = await simulateWhatIf({ features: MOCK_FEATURES, overrides: {} });
    expect(result.newRisk).toBe(30);
    expect(result.impactFactors).toEqual([]);
  });
});
