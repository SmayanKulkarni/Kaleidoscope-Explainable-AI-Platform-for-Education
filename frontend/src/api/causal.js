import { client } from './client';

export function getCausalGraph() {
  return client.get('/causal/graph').then((r) => r.data);
}
