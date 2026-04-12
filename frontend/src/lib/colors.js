export const CAUSAL_COLORS = {
  causal:         '#34d399',
  correlational:  '#fbbf24',
  confounder:     '#a78bfa',
  unknown:        '#94a3b8',
};

export const SHAP_COLORS = {
  positive: '#f87171',
  negative: '#34d399',
};

export const MODEL_COLORS = {
  gbm:         '#3b82f6',
  lstm:        '#8b5cf6',
  shared:      '#14b8a6',
  'model-gbm':  '#1d4ed8',
  'model-lstm': '#6d28d9',
};

export const GRAPH_LINK_COLORS = {
  synergistic: '#34d399',
  opposing:    '#f87171',
  agreement:   '#22c55e',
  disagreement:'#ef4444',
};

export const RISK_COLORS = {
  low:    '#22c55e',
  medium: '#f59e0b',
  high:   '#ef4444',
};

export const riskColor = (score) => {
  if (score < 0.3) return RISK_COLORS.low;
  if (score < 0.6) return RISK_COLORS.medium;
  return RISK_COLORS.high;
};

// Alias used by OutcomeDistributionCard — returns a Tailwind class string
export const getRiskColor = (score) => {
  if (score < 0.3) return 'text-green-600';
  if (score < 0.6) return 'text-amber-500';
  return 'text-red-600';
};

