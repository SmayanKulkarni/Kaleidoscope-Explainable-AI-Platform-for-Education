import { describe, it, expect, vi } from 'vitest';

vi.mock('../services/api', () => ({ api: { post: vi.fn(), get: vi.fn() } }));

import { FEATURE_METADATA, DEFAULT_FEATURES } from '../services/xaiService';

const BACKEND_OULAD_FIELDS = [
  'login_frequency_weekly',
  'avg_session_duration_min',
  'forum_posts_count',
  'video_completion_rate',
  'quiz_avg_score',
  'quiz_completion_rate',
  'assignment_submission_rate',
  'days_since_last_activity',
  'prior_course_completions',
  'current_week_in_course',
  'missed_deadlines_count',
  'help_requests_count',
];
const BACKEND_LATENT_FIELDS = [
  'engagement_latent_1',
  'engagement_latent_2',
  'engagement_latent_3',
];
const ALL_BACKEND_FIELDS = [...BACKEND_OULAD_FIELDS, ...BACKEND_LATENT_FIELDS];

describe('FEATURE_METADATA ↔ backend LearnerFeatures contract', () => {
  it('has exactly 12 entries (OULAD features; latent are excluded from UI)', () => {
    expect(FEATURE_METADATA).toHaveLength(12);
  });

  it('covers every OULAD backend field', () => {
    const ids = FEATURE_METADATA.map((f) => f.id);
    BACKEND_OULAD_FIELDS.forEach((field) => {
      expect(ids, `${field} missing from FEATURE_METADATA`).toContain(field);
    });
  });

  it('contains no unknown field IDs beyond the 15-field schema', () => {
    FEATURE_METADATA.forEach((f) => {
      expect(ALL_BACKEND_FIELDS, `${f.id} is not a recognised backend field`).toContain(f.id);
    });
  });

  it('has no duplicate IDs', () => {
    const ids = FEATURE_METADATA.map((f) => f.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('every entry has min strictly less than max', () => {
    FEATURE_METADATA.forEach((f) => {
      expect(f.min, `${f.id}: min must be < max`).toBeLessThan(f.max);
    });
  });

  it('every entry has a positive step value', () => {
    FEATURE_METADATA.forEach((f) => {
      expect(f.step, `${f.id}: step must be > 0`).toBeGreaterThan(0);
    });
  });

  it('every entry has a non-empty label string', () => {
    FEATURE_METADATA.forEach((f) => {
      expect(typeof f.label).toBe('string');
      expect(f.label.length).toBeGreaterThan(0);
    });
  });
});

describe('DEFAULT_FEATURES ↔ backend LearnerFeatures contract', () => {
  it('contains all 15 backend fields (12 OULAD + 3 latent)', () => {
    ALL_BACKEND_FIELDS.forEach((field) => {
      expect(DEFAULT_FEATURES, `${field} missing from DEFAULT_FEATURES`).toHaveProperty(field);
    });
  });

  it('contains no extra fields beyond the 15-field schema', () => {
    const allowed = new Set(ALL_BACKEND_FIELDS);
    Object.keys(DEFAULT_FEATURES).forEach((key) => {
      expect(allowed.has(key), `${key} is not a recognised backend field`).toBe(true);
    });
  });

  it('all values are finite numbers', () => {
    Object.entries(DEFAULT_FEATURES).forEach(([k, v]) => {
      expect(typeof v, `${k} is not a number`).toBe('number');
      expect(Number.isFinite(v), `${k} is not finite`).toBe(true);
    });
  });

  it('OULAD field values stay within FEATURE_METADATA bounds', () => {
    FEATURE_METADATA.forEach((meta) => {
      const val = DEFAULT_FEATURES[meta.id];
      expect(val, `${meta.id} default below min`).toBeGreaterThanOrEqual(meta.min);
      expect(val, `${meta.id} default above max`).toBeLessThanOrEqual(meta.max);
    });
  });

  it('latent features default to 0.0', () => {
    BACKEND_LATENT_FIELDS.forEach((field) => {
      expect(DEFAULT_FEATURES[field]).toBe(0.0);
    });
  });
});
