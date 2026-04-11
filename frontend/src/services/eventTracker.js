import { api } from './api';

const SESSION_ID_KEY = 'll_session_id';

const _getSessionId = () => {
  let id = sessionStorage.getItem(SESSION_ID_KEY);
  if (!id) {
    id = `sess_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    sessionStorage.setItem(SESSION_ID_KEY, id);
  }
  return id;
};

let _buffer = [];
let _flushTimer = null;
let _learnerId = null;

export const setLearnerId = (id) => { _learnerId = id; };

const _flush = async () => {
  if (_buffer.length === 0) return;
  const batch = _buffer.splice(0, 50);
  try {
    await api.post('/events', { events: batch });
  } catch (_) {}
};

const _scheduleFlush = () => {
  if (_flushTimer) return;
  _flushTimer = setTimeout(() => {
    _flushTimer = null;
    _flush();
  }, 10_000);
};

export const track = (event_type, opts = {}) => {
  if (!_learnerId) return;
  const event = {
    learner_id:   _learnerId,
    session_id:   _getSessionId(),
    event_type,
    event_target: opts.target  || null,
    event_value:  opts.value   ?? null,
    page:         opts.page    || window.location.pathname,
    extra:        opts.extra   || null,
    client_ts:    new Date().toISOString(),
  };
  _buffer.push(event);
  if (_buffer.length >= 50) {
    if (_flushTimer) { clearTimeout(_flushTimer); _flushTimer = null; }
    _flush();
  } else {
    _scheduleFlush();
  }
};

export const _resetForTesting = () => {
  _buffer.splice(0);
  if (_flushTimer) { clearTimeout(_flushTimer); _flushTimer = null; }
  _learnerId = null;
};

if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => _flush());
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') _flush();
  });
}
