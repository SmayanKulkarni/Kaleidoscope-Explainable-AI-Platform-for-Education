import { client } from './client';

export const explainFairness = (report) =>
  client.post('/fairness/explain', { report }).then((r) => r.data);

export const recommendStudent = (learner_id, items, top_k = 5, include_shap = true) =>
  client.post('/recommend/student', { learner_id, items, top_k, include_shap }).then((r) => r.data);

export const recommendStudentExplain = (learner_id, features, item_id = '') =>
  client.post('/recommend/student/explain', { learner_id, features, item_id }).then((r) => r.data);

export const recommendStudentWhatif = (learner_id, features, overrides) =>
  client.post('/recommend/student/whatif', { learner_id, features, overrides }).then((r) => r.data);

export const recommendInstructor = (instructor_id, items, top_k = 5, include_shap = true) =>
  client.post('/recommend/instructor', { instructor_id, items, top_k, include_shap }).then((r) => r.data);

export const recommendInstructorExplain = (instructor_id, features, item_id = '') =>
  client.post('/recommend/instructor/explain', { instructor_id, features, item_id }).then((r) => r.data);

export const recommendHealth = () =>
  client.get('/recommend/health').then((r) => r.data);

// Object-form aliases used by useMutation hooks
export const postStudentRecommend = ({ learner_id, items, top_k = 5, include_shap = true }) =>
  recommendStudent(learner_id, items, top_k, include_shap);

export const postStudentExplain = ({ learner_id, features, item_id = '' }) =>
  recommendStudentExplain(learner_id, features, item_id);

export const postInstructorRecommend = ({ instructor_id, items, top_k = 5, include_shap = true }) =>
  recommendInstructor(instructor_id, items, top_k, include_shap);

export const postInstructorExplain = ({ instructor_id, features, item_id = '' }) =>
  recommendInstructorExplain(instructor_id, features, item_id);

// ── Instructor roster + per-student endpoints ──────────────────────────────

export const getInstructorStudents = () =>
  client.get('/instructor/students').then((r) => r.data);

export const getAdminStudents = () =>
  client.get('/admin/students').then((r) => r.data);

export const getInstructorRecoForStudent = (learner_id) =>
  client.post(`/instructor/recommend/${learner_id}`, {}).then((r) => r.data);

export const explainInstructorRecoForStudent = (learner_id, features, item_id) =>
  client.post(`/instructor/recommend/${learner_id}/explain`, { features, item_id }).then((r) => r.data);

export const instructorWhatIfForStudent = (learner_id, overrides) =>
  client.post(`/instructor/whatif/${learner_id}`, { overrides }).then((r) => r.data);

// ── Student study-tips + compare narrate ──────────────────────────────────

export const getStudyTips = (payload) =>
  client.post('/student/study-tips', payload).then((r) => r.data);

export const compareNarrate = (payload) =>
  client.post('/compare/narrate', payload).then((r) => r.data);

// ── Admin enrollment endpoints ─────────────────────────────────────────────

export const getAdminInstructors = () =>
  client.get('/admin/instructors').then((r) => r.data);

export const getAdminEnrollments = () =>
  client.get('/admin/enrollments').then((r) => r.data);

export const addAdminEnrollment = (payload) =>
  client.post('/admin/enrollments', payload).then((r) => r.data);

export const deleteAdminEnrollment = (id) =>
  client.delete(`/admin/enrollments/${id}`).then((r) => r.data);

export const syncOuladUsers = () =>
  client.post('/admin/sync-oulad').then((r) => r.data);

export const bulkEnrollStudents = (payload) =>
  client.post('/admin/enrollments/bulk', payload).then((r) => r.data);

export const deactivateUser = (userId) =>
  client.post(`/admin/users/${userId}/deactivate`).then((r) => r.data);

export const activateUser = (userId) =>
  client.post(`/admin/users/${userId}/activate`).then((r) => r.data);
