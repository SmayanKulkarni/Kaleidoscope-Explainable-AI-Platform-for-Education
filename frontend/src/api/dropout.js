import { client } from './client';

export const predict = (features, model = 'gbm') =>
  client.post('/predict', { features, model }).then((r) => r.data);

export const explain = (features, learner_id, model = 'gbm', audience = 'learner', history = []) =>
  client.post('/explain', { features, learner_id, model, audience, history }).then((r) => r.data);

export const whatif = (features, overrides) =>
  client.post('/whatif', { features, overrides }).then((r) => r.data);

export const compare = (features, learner_id, history = []) =>
  client.post('/compare', { features, learner_id, history }).then((r) => r.data);

export const getHealth = () =>
  client.get('/health').then((r) => r.data);

export const getMlopsHealth = () =>
  client.get('/mlops/health').then((r) => r.data);

export const getMlopsMetrics = () =>
  client.get('/mlops/metrics').then((r) => r.data);

export const getHistory = (learner_id) =>
  client.get(`/history/${learner_id}`).then((r) => r.data);

export const getMlopsDriftReport = () =>
  client.get('/mlops/drift-report').then((r) => r.data);

export const triggerRetrain = () =>
  client.post('/mlops/retrain').then((r) => r.data);

export const triggerReload = (canary_fraction = 0.1) =>
  client.post('/mlops/reload', { canary_fraction }).then((r) => r.data);

export const counterfactual = (features) =>
  client.post('/counterfactual', features).then((r) => r.data);

export const explainMe = (audience = 'learner') =>
  client.post('/explain/me', { audience }).then((r) => r.data);

export const getExplainMeHistory = () =>
  client.get('/explain/me/history').then((r) => r.data);

export const DEFAULT_FEATURES = {
  login_frequency_weekly:     3,
  avg_session_duration_min:   45,
  forum_posts_count:          2,
  video_completion_rate:      0.6,
  quiz_avg_score:             65,
  quiz_completion_rate:       0.7,
  assignment_submission_rate: 0.75,
  days_since_last_activity:   5,
  prior_course_completions:   1,
  current_week_in_course:     6,
  missed_deadlines_count:     2,
  help_requests_count:        1,
  engagement_latent_1:        0.0,
  engagement_latent_2:        0.0,
  engagement_latent_3:        0.0,
};

export const MUTABLE_FEATURES = [
  { id: 'login_frequency_weekly',     label: 'Login Frequency',          min: 0, max: 14,  step: 0.5, unit: '/week' },
  { id: 'avg_session_duration_min',   label: 'Avg Session Duration',     min: 0, max: 300, step: 5,   unit: 'min'   },
  { id: 'forum_posts_count',          label: 'Forum Posts',              min: 0, max: 50,  step: 1,   unit: 'posts' },
  { id: 'video_completion_rate',      label: 'Video Completion',         min: 0, max: 1,   step: 0.05,unit: '%×100' },
  { id: 'quiz_completion_rate',       label: 'Quiz Completion',          min: 0, max: 1,   step: 0.05,unit: '%×100' },
  { id: 'assignment_submission_rate', label: 'Assignment Submission',    min: 0, max: 1,   step: 0.05,unit: '%×100' },
  { id: 'days_since_last_activity',   label: 'Days Since Last Activity', min: 0, max: 60,  step: 1,   unit: 'days'  },
  { id: 'help_requests_count',        label: 'Help Requests',            min: 0, max: 30,  step: 1,   unit: ''      },
];
