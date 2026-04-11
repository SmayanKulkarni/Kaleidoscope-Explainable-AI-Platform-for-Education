import { client } from './client';

export const postFeedback = (body) =>
  client.post('/feedback', body).then((r) => r.data);

export const getFeedbackStats = () =>
  client.get('/feedback/stats').then((r) => r.data);

export const getFeedbackByLearner = (learnerId) =>
  client.get(`/feedback/${learnerId}`).then((r) => r.data);

export const postEvents = (events) =>
  client.post('/events', { events }).then((r) => r.data);

export const getEvents = (learnerId, limit = 100) =>
  client
    .get(`/events/${learnerId}`, { params: { limit } })
    .then((r) => r.data);
