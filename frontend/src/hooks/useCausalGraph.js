import { useQuery } from '@tanstack/react-query';
import { getCausalGraph } from '../api/causal';

export function useCausalGraph() {
  return useQuery({
    queryKey: ['causal-graph'],
    queryFn: getCausalGraph,
    staleTime: Infinity,
  });
}
