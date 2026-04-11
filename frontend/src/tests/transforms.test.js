import { describe, it, expect, vi } from 'vitest';

vi.mock('../services/api', () => ({ api: { post: vi.fn(), get: vi.fn() } }));

import { transformExplainResponse } from '../hooks/useExplainData';

const MOCK_EXPLAIN = {
  risk_score: 0.74,
  risk_label: 'high',
  trust_score: {
    trust_score:  0.91,
    fidelity:     0.92,
    stability:    0.88,
    completeness: 0.85,
    label:        'high',
  },
  top_features: [
    { name: 'days_since_last_activity', shap:  0.32, direction: 'risk'       },
    { name: 'forum_posts_count',        shap: -0.18, direction: 'protective' },
  ],
  causal_annotations: [
    { name: 'days_since_last_activity', shap_value:  0.32, shap_direction: 'risk',       causal_type: 'causal'        },
    { name: 'forum_posts_count',        shap_value: -0.18, shap_direction: 'protective', causal_type: 'correlational' },
  ],
  prototypes: {
    matches: [
      { learner_id: 'L001', similarity: 0.95, outcome: 'dropped_out', diff_features: [] },
      { learner_id: 'L002', similarity: 0.88, outcome: 'completed',   diff_features: [] },
    ],
  },
  ranked_actions: [
    {
      feature:          'days_since_last_activity',
      plain_language:   'Log in more frequently',
      estimated_impact: 0.12,
      priority_rank:    1,
      direction:        'decrease',
      current_value:    14,
      target_value:     3,
    },
  ],
  anchor_rule: { human_readable: 'IF days_since_last_activity > 10 THEN HIGH RISK' },
  narratives:  { learner: 'You have been inactive for too long.', instructor: null },
  interactions: [],
};

describe('transformExplainResponse — risk + trust mapping', () => {
  it('maps risk_score float (0–1) → riskScore integer percentage', () => {
    expect(transformExplainResponse(MOCK_EXPLAIN).riskScore).toBe(74);
  });

  it('uppercases risk_label', () => {
    expect(transformExplainResponse(MOCK_EXPLAIN).riskLevel).toBe('HIGH');
  });

  it('keeps trust_score.trust_score as float', () => {
    expect(transformExplainResponse(MOCK_EXPLAIN).trustScore).toBeCloseTo(0.91, 4);
  });

  it('converts trust breakdown sub-scores to integer percentages', () => {
    const { trustBreakdown } = transformExplainResponse(MOCK_EXPLAIN);
    expect(trustBreakdown.fidelity).toBe(92);
    expect(trustBreakdown.stability).toBe(88);
    expect(trustBreakdown.completeness).toBe(85);
  });
});

describe('transformExplainResponse — direction mapping', () => {
  it('maps shap direction "risk" → "positive"', () => {
    expect(transformExplainResponse(MOCK_EXPLAIN).shapFeatures[0].direction).toBe('positive');
  });

  it('maps shap direction "protective" → "negative"', () => {
    expect(transformExplainResponse(MOCK_EXPLAIN).shapFeatures[1].direction).toBe('negative');
  });

  it('maps causal_annotations shap_direction "risk" → "positive"', () => {
    expect(transformExplainResponse(MOCK_EXPLAIN).featureContributions[0].direction).toBe('positive');
  });
});

describe('transformExplainResponse — prototype outcomes', () => {
  it('normalises "dropped_out" → "DROPPED OUT"', () => {
    expect(transformExplainResponse(MOCK_EXPLAIN).lookalikes[0].outcome).toBe('DROPPED OUT');
  });

  it('normalises "completed" → "COMPLETED"', () => {
    expect(transformExplainResponse(MOCK_EXPLAIN).lookalikes[1].outcome).toBe('COMPLETED');
  });

  it('maps prototype similarity float → integer percentage', () => {
    expect(transformExplainResponse(MOCK_EXPLAIN).lookalikes[0].match).toBe(95);
  });
});

describe('transformExplainResponse — actions + narrative', () => {
  it('extracts topAction from first ranked_action', () => {
    const { topAction } = transformExplainResponse(MOCK_EXPLAIN);
    expect(topAction.feature).toBe('days_since_last_activity');
    expect(topAction.riskReduction).toBe(12);
  });

  it('uses learner narrative as aiNarrative', () => {
    expect(transformExplainResponse(MOCK_EXPLAIN).aiNarrative).toBe('You have been inactive for too long.');
  });

  it('extracts anchor_rule.human_readable', () => {
    expect(transformExplainResponse(MOCK_EXPLAIN).anchorRule).toContain('days_since_last_activity');
  });
});

describe('transformExplainResponse — empty/missing field resilience', () => {
  it('returns empty arrays when top_features, prototypes, ranked_actions are absent', () => {
    const result = transformExplainResponse({ risk_score: 0.1, risk_label: 'low' });
    expect(result.shapFeatures).toEqual([]);
    expect(result.lookalikes).toEqual([]);
    expect(result.rankedActions).toEqual([]);
  });

  it('returns null topAction when ranked_actions is empty', () => {
    expect(transformExplainResponse({ risk_score: 0.1, risk_label: 'low' }).topAction).toBeNull();
  });

  it('returns empty string anchorRule when anchor_rule absent', () => {
    expect(transformExplainResponse({ risk_score: 0.1, risk_label: 'low' }).anchorRule).toBe('');
  });
});
