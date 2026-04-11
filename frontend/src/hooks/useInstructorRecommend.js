import { useMutation } from '@tanstack/react-query';
import { recommendInstructor, recommendInstructorExplain } from '../api/recommend';

export function useInstructorRecommend() {
  return useMutation({
    mutationFn: ({ instructor_id, items, top_k = 5, include_shap = true }) =>
      recommendInstructor(instructor_id, items, top_k, include_shap),
  });
}

export function useInstructorExplain() {
  return useMutation({
    mutationFn: ({ instructor_id, features, item_id = '' }) =>
      recommendInstructorExplain(instructor_id, features, item_id),
  });
}
