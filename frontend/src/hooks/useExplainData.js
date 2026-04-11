import { useState, useCallback, useRef } from 'react';
import { api } from '../services/api';

const _riskPct     = (v) => Math.round((v || 0) * 100);
const _pct         = (v) => Math.round((v || 0) * 100);
const _impactPct   = (v) => Math.round(Math.abs(v || 0) * 100);

export const transformExplainResponse = (resp) => {
  const r = resp;

  const shapFeatures = (r.top_features || []).map((f) => ({
    name:      f.name,
    value:     parseFloat(Math.abs(f.shap).toFixed(4)),
    direction: f.direction === 'risk' ? 'positive' : 'negative',
    shap:      f.shap,
  }));

  const featureContributions = (r.causal_annotations || []).map((ca) => ({
    name:      ca.name,
    value:     parseFloat(Math.abs(ca.shap_value).toFixed(4)),
    direction: ca.shap_direction === 'risk' ? 'positive' : 'negative',
    type:      ca.causal_type || 'correlational',
  }));

  const conceptAnalysis = (r.interactions || []).map((ix) => ({
    name:      ix.features ? ix.features.join(' × ') : ix.name || '',
    value:     parseFloat(Math.abs(ix.interaction_value || 0).toFixed(4)),
    direction: (ix.interaction_value || 0) > 0 ? 'positive' : 'negative',
  }));

  const lookalikes = ((r.prototypes || {}).matches || []).map((m) => ({
    id:       m.learner_id,
    match:    _pct(m.similarity),
    outcome:  m.outcome.toUpperCase().replace('_', ' '),
    diff:     m.diff_features || [],
  }));

  const topAction = (r.ranked_actions || [])[0]
    ? {
        label:        r.ranked_actions[0].plain_language,
        riskReduction: _impactPct(r.ranked_actions[0].estimated_impact),
        feature:       r.ranked_actions[0].feature,
      }
    : null;

  return {
    riskScore:            _riskPct(r.risk_score),
    riskLevel:            (r.risk_label || '').toUpperCase(),
    trustScore:           parseFloat((r.trust_score?.trust_score || 0).toFixed(4)),
    trustLevel:           (r.trust_score?.label || '').toUpperCase(),
    trustBreakdown: {
      fidelity:     _pct(r.trust_score?.fidelity),
      stability:    _pct(r.trust_score?.stability),
      completeness: _pct(r.trust_score?.completeness),
    },
    topAction,
    shapFeatures,
    featureContributions,
    conceptAnalysis,
    lookalikes,
    aiNarrative:          r.narratives?.learner || r.narratives?.instructor || '',
    anchorRule:           r.anchor_rule?.human_readable || '',
    rankedActions:        r.ranked_actions || [],
    rawExplain:           r,
  };
};

export default function useExplainData() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const cacheRef = useRef({});

  const fetchExplain = useCallback(async ({ features, learner_id, model = 'gbm', audience = 'learner' }) => {
    const cacheKey = `${learner_id}-${model}-${audience}`;
    if (cacheRef.current[cacheKey]) {
      setData(cacheRef.current[cacheKey]);
      return cacheRef.current[cacheKey];
    }
    setLoading(true);
    setError(null);
    try {
      const { data: resp } = await api.post('/explain', { features, learner_id, model, audience });
      const transformed = transformExplainResponse(resp);
      cacheRef.current[cacheKey] = transformed;
      setData(transformed);
      sessionStorage.setItem(`ll_features_${learner_id}`, JSON.stringify(features));
      return transformed;
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load explanation.');
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  const invalidate = useCallback(() => {
    cacheRef.current = {};
  }, []);

  return { data, loading, error, fetchExplain, invalidate };
}
