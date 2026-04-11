import { useState, useEffect } from 'react';
import { api } from '../services/api';

export default function useMlopsHealth() {
  const [health, setHealth]   = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    let cancelled = false;
    const fetchAll = async () => {
      try {
        const [hRes, mRes] = await Promise.allSettled([
          api.get('/mlops/health'),
          api.get('/mlops/metrics'),
        ]);
        if (cancelled) return;
        if (hRes.status === 'fulfilled') setHealth(hRes.value.data);
        if (mRes.status === 'fulfilled') setMetrics(mRes.value.data);
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchAll();
    return () => { cancelled = true; };
  }, []);

  const modelConfidence = metrics?.training?.gbm?.test_auc_roc
    ?? metrics?.training?.test_auc_roc
    ?? null;

  return { health, metrics, loading, error, modelConfidence };
}
