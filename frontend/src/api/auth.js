import { client } from './client';

export const register = (body) =>
  client.post('/auth/register', body).then((r) => r.data);

export const login = (username, password) =>
  client.post('/auth/login', { username, password }).then((r) => r.data);

export const getMe = () =>
  client.get('/auth/me').then((r) => r.data);

export const getStudents = (courseId) =>
  client
    .get('/auth/students', { params: courseId ? { course_id: courseId } : {} })
    .then((r) => r.data);

export const enrollStudent = (learnerId, courseId) =>
  client
    .post('/auth/enroll', null, {
      params: { learner_id: learnerId, course_id: courseId },
    })
    .then((r) => r.data);

export const updateWeek = (week) =>
  client
    .put('/auth/me/week', null, { params: { week } })
    .then((r) => r.data);

export const deleteAccount = () =>
  client.delete('/auth/me').then((r) => r.data);
