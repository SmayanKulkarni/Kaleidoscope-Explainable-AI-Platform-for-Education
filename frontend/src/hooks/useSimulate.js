import { useMutation } from '@tanstack/react-query';
import { simulate } from '../api/simulate';

export function useSimulate() {
  return useMutation({
    mutationFn: simulate,
  });
}
