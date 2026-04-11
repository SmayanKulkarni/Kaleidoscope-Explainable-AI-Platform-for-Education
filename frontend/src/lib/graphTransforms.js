import { CAUSAL_COLORS, MODEL_COLORS, GRAPH_LINK_COLORS } from './colors';

export function shapInteractionToGraph(explanation) {
  if (!explanation) return { nodes: [], links: [] };

  const shapValues   = explanation.shap_values   ?? {};
  const topFeatures  = explanation.top_features  ?? [];
  const causalAnns   = explanation.causal_annotations ?? [];
  const interactions = explanation.feature_interactions ?? explanation.interactions ?? [];

  const causalMap = Object.fromEntries(
    causalAnns.map((ca) => [ca.name, ca.causal_type ?? 'unknown'])
  );

  const nodes = topFeatures.map((f) => ({
    id:    f.name ?? f,
    val:   Math.abs(shapValues[f.name ?? f] ?? f.shap ?? 0) * 100,
    type:  causalMap[f.name ?? f] ?? 'unknown',
    color: CAUSAL_COLORS[causalMap[f.name ?? f]] ?? CAUSAL_COLORS.unknown,
    name:  f.name ?? f,
  }));

  const links = (interactions).map((ix) => {
    const feats = ix.features ?? [];
    return {
      source:    feats[0] ?? '',
      target:    feats[1] ?? '',
      value:     Math.abs(ix.strength ?? ix.interaction_value ?? 0),
      direction: ix.direction ?? 'synergistic',
      color:     ix.direction === 'opposing'
        ? GRAPH_LINK_COLORS.opposing
        : GRAPH_LINK_COLORS.synergistic,
    };
  }).filter((l) => l.source && l.target);

  return { nodes, links };
}

export function causalDagToGraph(explanation) {
  if (!explanation) return { nodes: [], links: [] };

  const shapValues = explanation.shap_values  ?? {};
  const causalAnns = explanation.causal_annotations ?? [];
  const topFeatures = explanation.top_features ?? [];

  const featureNames = topFeatures.length
    ? topFeatures.map((f) => f.name ?? f)
    : Object.keys(shapValues);

  const causalMap = Object.fromEntries(
    causalAnns.map((ca) => [ca.name, ca.causal_type ?? 'unknown'])
  );

  const nodes = [
    ...featureNames.map((name) => ({
      id:    name,
      val:   Math.abs(shapValues[name] ?? 0) * 80 + 4,
      type:  causalMap[name] ?? 'unknown',
      color: CAUSAL_COLORS[causalMap[name]] ?? CAUSAL_COLORS.unknown,
      name,
    })),
    { id: 'Dropout Risk', val: 20, color: '#ef4444', type: 'outcome', name: 'Dropout Risk' },
  ];

  const links = featureNames.map((name) => ({
    source:  name,
    target:  'Dropout Risk',
    value:   Math.abs(shapValues[name] ?? 0),
    dashed:  (causalMap[name] ?? 'unknown') !== 'causal',
    color:   CAUSAL_COLORS[causalMap[name]] ?? CAUSAL_COLORS.unknown,
  }));

  return { nodes, links };
}

export function compareToGraph(compareResp) {
  if (!compareResp) return { nodes: [], links: [] };

  const gbmTop3  = compareResp.gbm_top3  ?? [];
  const lstmTop3 = compareResp.lstm_top3_temporal ?? [];
  const shared   = new Set(gbmTop3.filter((f) => lstmTop3.includes(f)));

  const allNodes = [
    ...gbmTop3.map((f) => ({
      id:    f,
      group: shared.has(f) ? 'shared' : 'gbm',
      color: shared.has(f) ? MODEL_COLORS.shared : MODEL_COLORS.gbm,
      val:   8,
      name:  f,
    })),
    ...lstmTop3.filter((f) => !shared.has(f)).map((f) => ({
      id:    f,
      group: 'lstm',
      color: MODEL_COLORS.lstm,
      val:   8,
      name:  f,
    })),
    { id: 'GBM Model',  group: 'model-gbm',  color: MODEL_COLORS['model-gbm'],  val: 20, name: 'GBM Model'  },
    { id: 'LSTM Model', group: 'model-lstm', color: MODEL_COLORS['model-lstm'], val: 20, name: 'LSTM Model' },
  ];

  const agree = !compareResp.disagreement_flag;
  const links = [
    ...gbmTop3.map((f)  => ({ source: f, target: 'GBM Model',  color: agree ? GRAPH_LINK_COLORS.agreement : GRAPH_LINK_COLORS.disagreement })),
    ...lstmTop3.map((f) => ({ source: f, target: 'LSTM Model', color: agree ? GRAPH_LINK_COLORS.agreement : GRAPH_LINK_COLORS.disagreement })),
  ];

  return { nodes: allNodes, links };
}

export function recommendPathToGraph(recommendation, learnerId = 'You') {
  if (!recommendation) return { nodes: [], links: [] };

  const shap = recommendation.shap_values ?? {};
  const itemId = recommendation.item_id ?? 'Resource';

  const featureNodes = Object.entries(shap).map(([name, val]) => ({
    id:    name,
    val:   Math.abs(val) * 60 + 4,
    color: '#6366f1',
    name,
  }));

  const nodes = [
    { id: learnerId,  val: 24, color: '#6366f1', name: learnerId, isStudent: true },
    ...featureNodes,
    { id: itemId, val: 18, color: '#f59e0b', name: itemId, isResource: true },
  ];

  const links = [
    ...featureNodes.map((f) => ({ source: learnerId,  target: f.id })),
    ...featureNodes.map((f) => ({ source: f.id, target: itemId })),
  ];

  return { nodes, links };
}
