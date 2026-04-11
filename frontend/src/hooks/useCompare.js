import { useQuery } from '@tanstack/react-query';
import { compare } from '../api/dropout';

export function useCompare({ features, learner_id, history = [] }, options = {}) {
  return useQuery({
    queryKey:  ['compare', learner_id],
    queryFn:   () => compare(features, learner_id, history),
    enabled:   !!features && !!learner_id,
    staleTime: 5 * 60 * 1000,
    ...options,
  });
}
