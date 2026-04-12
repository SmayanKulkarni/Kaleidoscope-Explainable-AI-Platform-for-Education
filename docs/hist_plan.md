# History Tab Implementation Plan

Background: The repository already features robust history storage and drift detection on the backend. The backend `consistency_store` handles history retention, meaning we just need to stitch the frontend to these existing endpoints.

## 1. Backend Assessment (Already Implemented)
The following backend endpoints are available for history fetching in `backend/app/main.py`:
- `GET /history/{learner_id}`: Used by instructors or admins to get history for a specific learner. Returns `{"history": [...]}`.
- `GET /explain/me/history`: Used by authenticated students to fetch their own explanation timeline. Returns `{"history": [...]}`.

## 2. Frontend API Additions
In `frontend/src/api/recommend.js` (or a dedicated `history.js`), we need to add bindings for these endpoints:
```javascript
export const getMyHistory = () => 
  client.get('/explain/me/history').then((r) => r.data);

export const getLearnerHistory = (learner_id) => 
  client.get(`/history/${learner_id}`).then((r) => r.data);
```

## 3. Sidebar & Routing Updates
1. **Update `Sidebar.jsx`**: Change the stub `#history` link.
   - Student: `{ to: '/history', icon: 'history', label: 'History' }`
   - Instructor: `{ to: '/instructor/history', icon: 'history', label: 'History' }`
2. **Update `App.jsx`**:
   - Add `<Route path="/history" element={<HistoryPage />} />` under student routes.
   - Add `<Route path="/instructor/history" element={<HistoryPage />} />` under instructor routes.

## 4. `HistoryPage.jsx` Implementation
Create a new generic or semi-generic React page (`frontend/src/pages/HistoryPage.jsx`) that handles both roles:
- **Role Detection**: Use `const { user } = useAuth();` to determine if the viewer is a `student` or `instructor`.
- **Query Hook**: Use React Query (`useQuery`) to fetch history. If `student`, call `getMyHistory()`. If `instructor`, provide a dropdown to select a student (`getInstructorStudents()`) and then call `getLearnerHistory(selected_learner_id)`.
- **UI Presentation**:
  - Render a vertical timeline of explanation events (ordered chronologically or newest-first).
  - Show the features/shap values for each historical point. 
  - Add visual flags if `drift_detector` triggered any changes between snapshots (if SHAP drift was returned).

## 5. Execution Steps
- [ ] Add API hooks in `frontend/src/api/recommend.js`.
- [ ] Implement `frontend/src/pages/HistoryPage.jsx`.
- [ ] Wire the route into `App.jsx`.
- [ ] Update `Sidebar.jsx` exact paths.
- [ ] Test the integration to ensure the History view accurately populates with data for both student and instructor roles.
