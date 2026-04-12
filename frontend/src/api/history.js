import { client } from './client';

/** Student: fetch own explanation history (authenticated via JWT) */
export const getMyHistory = () =>
  client.get('/explain/me/history').then((r) => r.data);

/** Instructor/Admin: fetch history for a specific learner */
export const getLearnerHistory = (learner_id) =>
  client.get(`/history/${learner_id}`).then((r) => r.data);
