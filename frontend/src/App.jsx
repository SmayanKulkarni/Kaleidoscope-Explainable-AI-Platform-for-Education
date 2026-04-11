import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import Login from './pages/Login';
import StudentDashboard from './pages/StudentDashboard';
import InstructorDashboard from './pages/InstructorDashboard';
import WhatIfExplorer from './pages/WhatIfExplorer';
import ActionPlan from './pages/ActionPlan';

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          
          <Route element={<ProtectedRoute allowedRoles={['student', 'instructor']} />}>
            <Route path="/student" element={<StudentDashboard />} />
            <Route path="/action-plan" element={<ActionPlan />} />
          </Route>

          <Route element={<ProtectedRoute allowedRoles={['instructor']} />}>
            <Route path="/instructor" element={<InstructorDashboard />} />
            <Route path="/what-if" element={<WhatIfExplorer />} />
          </Route>

          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
