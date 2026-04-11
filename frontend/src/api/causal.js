import { client } from './client';

export function getCausalGraph() {
  return client.get('/causal/graph').then((r) => r.data);
}

export function explainCausalGraph() {
  return client.post('/causal/explain').then((r) => r.data);
}
