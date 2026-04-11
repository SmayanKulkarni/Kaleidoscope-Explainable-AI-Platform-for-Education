import { client } from './client';

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
