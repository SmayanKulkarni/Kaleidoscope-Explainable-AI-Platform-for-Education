import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../services/api', () => ({
  api: { post: vi.fn() },
}));

import { api } from '../services/api';
import { track, setLearnerId, _resetForTesting } from '../services/eventTracker';

beforeEach(() => {
  _resetForTesting();
  vi.clearAllMocks();
  sessionStorage.clear();
});

describe('track() — no-op guard', () => {
  it('does nothing when learner_id has not been set', () => {
    track('page_view', { page: '/student' });
    expect(api.post).not.toHaveBeenCalled();
  });

  it('does nothing after learner_id is cleared via setLearnerId(null)', () => {
    setLearnerId('L001');
    setLearnerId(null);
    track('page_view');
    expect(api.post).not.toHaveBeenCalled();
  });
});

describe('track() — event shape', () => {
  it('each event carries required fields', () => {
    api.post.mockResolvedValue({ data: {} });
    setLearnerId('L042');
    for (let i = 0; i < 50; i++) {
      track('action_viewed', { target: 'btn_take', value: 1 });
    }
    const events = api.post.mock.calls[0][1].events;
    const ev = events[0];
    expect(ev.learner_id).toBe('L042');
    expect(ev.event_type).toBe('action_viewed');
    expect(ev.event_target).toBe('btn_take');
    expect(ev.event_value).toBe(1);
    expect(typeof ev.client_ts).toBe('string');
    expect(typeof ev.session_id).toBe('string');
  });

  it('all events in one flush share the same session_id', () => {
    api.post.mockResolvedValue({ data: {} });
    setLearnerId('L043');
    for (let i = 0; i < 50; i++) {
      track('page_view');
    }
    const events = api.post.mock.calls[0][1].events;
    const sessionIds = new Set(events.map((e) => e.session_id));
    expect(sessionIds.size).toBe(1);
  });

  it('event without opts produces null target and null value', () => {
    api.post.mockResolvedValue({ data: {} });
    setLearnerId('L044');
    for (let i = 0; i < 50; i++) {
      track('page_view');
    }
    const ev = api.post.mock.calls[0][1].events[0];
    expect(ev.event_target).toBeNull();
    expect(ev.event_value).toBeNull();
  });
});

describe('track() — flush behaviour', () => {
  it('flushes immediately via POST /events when buffer reaches 50 events', () => {
    api.post.mockResolvedValue({ data: {} });
    setLearnerId('L045');
    for (let i = 0; i < 50; i++) {
      track('page_view');
    }
    expect(api.post).toHaveBeenCalledOnce();
    expect(api.post).toHaveBeenCalledWith('/events', expect.objectContaining({ events: expect.any(Array) }));
  });

  it('sends exactly 50 events in the flushed batch', () => {
    api.post.mockResolvedValue({ data: {} });
    setLearnerId('L046');
    for (let i = 0; i < 50; i++) {
      track('whatif_slider');
    }
    expect(api.post.mock.calls[0][1].events).toHaveLength(50);
  });

  it('does not flush before buffer reaches 50 events', () => {
    api.post.mockResolvedValue({ data: {} });
    setLearnerId('L047');
    for (let i = 0; i < 49; i++) {
      track('page_view');
    }
    expect(api.post).not.toHaveBeenCalled();
  });
});
