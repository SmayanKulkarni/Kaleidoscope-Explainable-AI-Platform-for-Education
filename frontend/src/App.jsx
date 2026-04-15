import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import Login from './pages/Login';
import StudentDashboard from './pages/StudentDashboard';
import InstructorDashboard from './pages/InstructorDashboard';
import WhatIfExplorer from './pages/WhatIfExplorer';
import ActionPlan from './pages/ActionPlan';
import StudentView from './pages/StudentView';
import InstructorView from './pages/InstructorView';
import ComparePage from './pages/ComparePage';
import RegisterPage from './pages/RegisterPage';
import AdminDashboard from './pages/AdminDashboard';
import SimulatePage from './pages/SimulatePage';
import HistoryPage from './pages/HistoryPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<RegisterPage />} />

            <Route element={<ProtectedRoute allowedRoles={['student', 'instructor', 'admin']} />}>
              <Route path="/student" element={<StudentDashboard />} />
              <Route path="/dashboard/student" element={<StudentDashboard />} />
              <Route path="/action-plan" element={<ActionPlan />} />
              <Route path="/xai/student" element={<StudentView />} />
              <Route path="/xai/compare" element={<ComparePage />} />
              <Route path="/simulate" element={<SimulatePage />} />
              <Route path="/history" element={<HistoryPage />} />
            </Route>

            <Route element={<ProtectedRoute allowedRoles={['instructor', 'admin']} />}>
              <Route path="/instructor" element={<InstructorDashboard />} />
              <Route path="/dashboard/instructor" element={<InstructorDashboard />} />
              <Route path="/what-if" element={<WhatIfExplorer />} />
              <Route path="/xai/instructor" element={<InstructorView />} />
              <Route path="/instructor/history" element={<HistoryPage />} />
            </Route>

            <Route element={<ProtectedRoute allowedRoles={['admin']} />}>
              <Route path="/dashboard/admin" element={<AdminDashboard />} />
              <Route path="/admin" element={<Navigate to="/dashboard/admin" replace />} />
            </Route>

            <Route path="*" element={<Navigate to="/login" replace />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}
