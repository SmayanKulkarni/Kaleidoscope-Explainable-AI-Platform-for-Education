import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const ROLE_ROUTES = {
  student: '/dashboard/student',
  instructor: '/dashboard/instructor',
  admin: '/dashboard/admin',
};

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();

  const [form, setForm] = useState({ username: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const u = await login(form.username, form.password);
      navigate(ROLE_ROUTES[u.role] ?? '/dashboard/student', { replace: true });
    } catch (err) {
      setError(err.response?.data?.detail ?? 'Login failed. Check credentials.');
    } finally {
      setLoading(false);
    }
  };

  const fillDemo = (username, password) => setForm({ username, password });

  return (
    <div className="min-h-screen bg-surface flex flex-col items-center justify-center relative">
      <div className="max-w-md w-full bg-surface-container-low p-8 rounded-2xl border border-outline-variant/20 shadow-2xl z-10 text-center">
        <h1 className="text-4xl font-headline font-black text-primary mb-8">Kaliedoscope</h1>

        <h2 className="text-2xl font-bold mb-6">Sign In</h2>

        <div className="flex gap-2 mb-6">
          <button type="button" onClick={() => fillDemo('demo_student', 'student123')}
            className="flex-1 py-2 rounded-lg bg-surface-container border border-outline-variant/30 text-xs font-bold text-on-surface-variant hover:text-on-surface hover:border-outline-variant/60 transition-colors">
            Demo Student
          </button>
          <button type="button" onClick={() => fillDemo('demo_instructor', 'instructor123')}
            className="flex-1 py-2 rounded-lg bg-surface-container border border-outline-variant/30 text-xs font-bold text-on-surface-variant hover:text-on-surface hover:border-outline-variant/60 transition-colors">
            Demo Instructor
          </button>
          <button type="button" onClick={() => fillDemo('demo_admin', 'admin123')}
            className="flex-1 py-2 rounded-lg bg-surface-container border border-outline-variant/30 text-xs font-bold text-on-surface-variant hover:text-on-surface hover:border-outline-variant/60 transition-colors">
            Demo Admin
          </button>
        </div>

        {error && (
          <div className="mb-4 px-4 py-3 rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm text-left">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="text"
            placeholder="Username"
            value={form.username}
            onChange={(e) => setForm({ ...form, username: e.target.value })}
            className="w-full bg-surface-container px-4 py-3 rounded-xl border border-outline-variant/20 focus:border-primary/50 focus:outline-none transition-colors"
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={form.password}
            onChange={(e) => setForm({ ...form, password: e.target.value })}
            className="w-full bg-surface-container px-4 py-3 rounded-xl border border-outline-variant/20 focus:border-primary/50 focus:outline-none transition-colors"
            required
          />

          <button type="submit" disabled={loading} className="btn-primary w-full mt-4">
            {loading ? 'Authenticating...' : 'Sign In'}
          </button>
        </form>

        <p className="mt-6 text-sm text-slate-500">
          Don't have an account?{' '}
          <Link to="/register" className="text-primary font-bold hover:underline">
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}
