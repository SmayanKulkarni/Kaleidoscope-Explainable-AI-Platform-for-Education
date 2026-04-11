import { useRef, useCallback, useEffect } from 'react';
import { postEvents } from '../api/feedback';

const SESSION_KEY = 'xai_session_id';

function getSessionId() {
  let id = sessionStorage.getItem(SESSION_KEY);
  if (!id) {
    id = `sess_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    sessionStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

export function useEventTracker() {
  const buffer = useRef([]);
  const flushTimer = useRef(null);

  const flush = useCallback(async () => {
    if (buffer.current.length === 0) return;
    const toSend = buffer.current.splice(0);
    try {
      await postEvents(toSend);
    } catch (_) {
      // silently drop — telemetry must never crash UI
    }
  }, []);

  const track = useCallback(
    (eventType, eventTarget = null, eventValue = null) => {
      const learnerId =
        localStorage.getItem('ll_learner_id') ||
        localStorage.getItem('ll_user')
          ? JSON.parse(localStorage.getItem('ll_user') || '{}')?.learner_id
          : 'anonymous';

      buffer.current.push({
        learner_id: learnerId ?? 'anonymous',
        session_id: getSessionId(),
        event_type: eventType,
        event_target: eventTarget,
        event_value: eventValue,
        page: window.location.pathname,
        client_ts: new Date().toISOString(),
      });

      if (buffer.current.length >= 10) {
        flush();
      } else {
        clearTimeout(flushTimer.current);
        flushTimer.current = setTimeout(flush, 30_000);
      }
    },
    [flush],
  );

  useEffect(() => {
    track('session_start');
    const handleBlur = () => {
      const elapsed = Date.now() - performance.timeOrigin;
      track('session_end', null, Math.round(elapsed));
      flush();
    };
    const handleFocus = () => track('session_start');
    window.addEventListener('blur', handleBlur);
    window.addEventListener('focus', handleFocus);
    return () => {
      window.removeEventListener('blur', handleBlur);
      window.removeEventListener('focus', handleFocus);
      clearTimeout(flushTimer.current);
    };
  }, [track, flush]);

  return { track };
}
