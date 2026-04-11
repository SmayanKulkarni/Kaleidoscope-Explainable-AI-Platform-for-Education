import { useQuery } from '@tanstack/react-query';
import { recommendStudent, recommendInstructor, recommendHealth } from '../api/recommend';

export function useRecommendStudent({ learner_id, items, top_k = 5 }, options = {}) {
  return useQuery({
    queryKey:  ['recommend-student', learner_id],
    queryFn:   () => recommendStudent(learner_id, items, top_k),
    enabled:   !!learner_id && Array.isArray(items) && items.length > 0,
    staleTime: 5 * 60 * 1000,
    ...options,
  });
}

export function useRecommendInstructor({ instructor_id, items, top_k = 5 }, options = {}) {
  return useQuery({
    queryKey:  ['recommend-instructor', instructor_id],
    queryFn:   () => recommendInstructor(instructor_id, items, top_k),
    enabled:   !!instructor_id && Array.isArray(items) && items.length > 0,
    staleTime: 5 * 60 * 1000,
    ...options,
  });
}

export function useRecommendHealth(options = {}) {
  return useQuery({
    queryKey:  ['recommend-health'],
    queryFn:   recommendHealth,
    staleTime: 60 * 1000,
    ...options,
  });
}
