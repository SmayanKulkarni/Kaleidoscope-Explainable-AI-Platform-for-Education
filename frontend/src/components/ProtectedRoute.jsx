import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function ProtectedRoute({ allowedRoles }) {
  const { user, loading } = useAuth();

  if (loading) return null;
  if (!user) return <Navigate to="/login" replace />;
  
  if (!allowedRoles.includes(user.role)) {
    return (
      <div className="min-h-screen flex items-center justify-center p-8 bg-surface">
        <div className="text-center">
          <h1 className="text-3xl font-bold font-headline text-error mb-4">Access Denied</h1>
          <p className="text-on-surface-variant">Your role ({user.role}) does not have permission.</p>
        </div>
      </div>
    );
  }

  return <Outlet />;
}
