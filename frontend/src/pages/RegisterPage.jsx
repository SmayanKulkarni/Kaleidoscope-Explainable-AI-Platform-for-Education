import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const ROLE_ROUTES = {
  student: '/dashboard/student',
  instructor: '/dashboard/instructor',
  admin: '/dashboard/admin',
};

export default function RegisterPage() {
  const { signup } = useAuth();
  const navigate = useNavigate();

  const [form, setForm] = useState({
    username: '',
    email: '',
    password: '',
    full_name: '',
    role: 'student',
    learner_id: '',
    course_id: '',
    module_presentation: '',
    department: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const body = {
        username: form.username,
        email: form.email,
        password: form.password,
        full_name: form.full_name || undefined,
        role: form.role,
        ...(form.role === 'student' && {
          learner_id: form.learner_id || undefined,
          course_id: form.course_id || undefined,
          module_presentation: form.module_presentation || undefined,
        }),
        ...(form.role === 'instructor' && {
          department: form.department || undefined,
        }),
      };
      const u = await signup(body);
      navigate(ROLE_ROUTES[u.role] ?? '/dashboard/student', { replace: true });
    } catch (err) {
      setError(err.response?.data?.detail ?? 'Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-surface flex flex-col items-center justify-center px-4 py-12">
      <div className="max-w-lg w-full bg-surface-container-low p-8 rounded-2xl border border-outline-variant/20 shadow-2xl">
        <h1 className="text-4xl font-headline font-black text-primary mb-2 text-center">LearnLens</h1>
        <h2 className="text-2xl font-bold mb-6 text-center">Create Account</h2>

        {error && (
          <div className="mb-4 px-4 py-3 rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-bold text-slate-500 mb-1 uppercase tracking-wider">Username *</label>
              <input type="text" value={form.username} onChange={(e) => set('username', e.target.value)}
                className="w-full bg-surface-container px-4 py-3 rounded-xl border border-outline-variant/20 focus:border-primary/50 focus:outline-none transition-colors text-sm"
                required />
            </div>
            <div>
              <label className="block text-xs font-bold text-slate-500 mb-1 uppercase tracking-wider">Full Name</label>
              <input type="text" value={form.full_name} onChange={(e) => set('full_name', e.target.value)}
                className="w-full bg-surface-container px-4 py-3 rounded-xl border border-outline-variant/20 focus:border-primary/50 focus:outline-none transition-colors text-sm" />
            </div>
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-500 mb-1 uppercase tracking-wider">Email *</label>
            <input type="email" value={form.email} onChange={(e) => set('email', e.target.value)}
              className="w-full bg-surface-container px-4 py-3 rounded-xl border border-outline-variant/20 focus:border-primary/50 focus:outline-none transition-colors text-sm"
              required />
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-500 mb-1 uppercase tracking-wider">Password *</label>
            <input type="password" value={form.password} onChange={(e) => set('password', e.target.value)}
              className="w-full bg-surface-container px-4 py-3 rounded-xl border border-outline-variant/20 focus:border-primary/50 focus:outline-none transition-colors text-sm"
              required minLength={8} />
          </div>

          <div>
            <label className="block text-xs font-bold text-slate-500 mb-1 uppercase tracking-wider">Role *</label>
            <div className="flex gap-3">
              {['student', 'instructor'].map((r) => (
                <button key={r} type="button"
                  onClick={() => set('role', r)}
                  className={`flex-1 py-2.5 rounded-xl border text-sm font-bold capitalize transition-colors ${
                    form.role === r
                      ? 'bg-primary text-white border-primary'
                      : 'bg-surface-container border-outline-variant/30 text-on-surface-variant hover:border-primary/40'
                  }`}>
                  {r}
                </button>
              ))}
            </div>
          </div>

          {form.role === 'student' && (
            <div className="grid grid-cols-2 gap-4 p-4 rounded-xl bg-surface-container border border-outline-variant/10">
              <div>
                <label className="block text-xs font-bold text-slate-500 mb-1 uppercase tracking-wider">Learner ID</label>
                <input type="text" value={form.learner_id} onChange={(e) => set('learner_id', e.target.value)}
                  placeholder="e.g. 28400"
                  className="w-full bg-surface px-3 py-2 rounded-lg border border-outline-variant/20 focus:border-primary/50 focus:outline-none transition-colors text-sm" />
              </div>
              <div>
                <label className="block text-xs font-bold text-slate-500 mb-1 uppercase tracking-wider">Course ID</label>
                <input type="text" value={form.course_id} onChange={(e) => set('course_id', e.target.value)}
                  placeholder="e.g. AAA"
                  className="w-full bg-surface px-3 py-2 rounded-lg border border-outline-variant/20 focus:border-primary/50 focus:outline-none transition-colors text-sm" />
              </div>
              <div className="col-span-2">
                <label className="block text-xs font-bold text-slate-500 mb-1 uppercase tracking-wider">Module Presentation</label>
                <input type="text" value={form.module_presentation} onChange={(e) => set('module_presentation', e.target.value)}
                  placeholder="e.g. 2014J"
                  className="w-full bg-surface px-3 py-2 rounded-lg border border-outline-variant/20 focus:border-primary/50 focus:outline-none transition-colors text-sm" />
              </div>
            </div>
          )}

          {form.role === 'instructor' && (
            <div className="p-4 rounded-xl bg-surface-container border border-outline-variant/10">
              <label className="block text-xs font-bold text-slate-500 mb-1 uppercase tracking-wider">Department</label>
              <input type="text" value={form.department} onChange={(e) => set('department', e.target.value)}
                placeholder="e.g. Computer Science"
                className="w-full bg-surface px-3 py-2 rounded-lg border border-outline-variant/20 focus:border-primary/50 focus:outline-none transition-colors text-sm" />
            </div>
          )}

          <button type="submit" disabled={loading}
            className="btn-primary w-full mt-2">
            {loading ? 'Creating Account...' : 'Create Account'}
          </button>
        </form>

        <p className="mt-6 text-sm text-slate-500 text-center">
          Already have an account?{' '}
          <Link to="/login" className="text-primary font-bold hover:underline">
            Sign In
          </Link>
        </p>
      </div>
    </div>
  );
}
