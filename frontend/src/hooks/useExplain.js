import { useQuery } from '@tanstack/react-query';
import { explain } from '../api/dropout';

export function useExplain({ features, learner_id, model = 'gbm', audience = 'learner', history = [] }, options = {}) {
  return useQuery({
    queryKey:  ['explain', learner_id, model, audience],
    queryFn:   () => explain(features, learner_id, model, audience, history),
    enabled:   !!features && !!learner_id,
    staleTime: 5 * 60 * 1000,
    ...options,
  });
}
