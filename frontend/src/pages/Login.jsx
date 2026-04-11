import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();

  const [form, setForm] = useState({ email: '', password: '' });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    const u = await login(form.email, form.password || 'password');
    setLoading(false);
    navigate(u.role === 'instructor' ? '/instructor' : '/student', { replace: true });
  };

  const setStudentMock = () => setForm({ email: 'student@learnlens.com', password: 'password123' });
  const setInstructorMock = () => setForm({ email: 'instructor@learnlens.com', password: 'password123' });

  return (
    <div className="min-h-screen bg-surface flex flex-col items-center justify-center relative">
      <div className="max-w-md w-full bg-surface-container-low p-8 rounded-2xl border border-outline-variant/20 shadow-2xl z-10 text-center">
        <h1 className="text-4xl font-headline font-black text-primary mb-8">LearnLens</h1>
        
        <h2 className="text-2xl font-bold mb-6">Sign In</h2>

        <div className="flex gap-4 mb-6">
          <button type="button" onClick={setStudentMock} className="flex-1 py-2 rounded-lg bg-surface-container border border-outline-variant/30 text-sm font-bold text-on-surface-variant hover:text-on-surface hover:border-outline-variant/60 transition-colors">
            Dev: Fill Student
          </button>
          <button type="button" onClick={setInstructorMock} className="flex-1 py-2 rounded-lg bg-surface-container border border-outline-variant/30 text-sm font-bold text-on-surface-variant hover:text-on-surface hover:border-outline-variant/60 transition-colors">
            Dev: Fill Teacher
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <input 
            type="email" 
            placeholder="Email Address" 
            value={form.email}
            onChange={e => setForm({...form, email: e.target.value})} 
            className="w-full bg-surface-container px-4 py-3 rounded-xl border border-outline-variant/20 focus:border-primary/50 focus:outline-none transition-colors" 
            required 
          />
          <input 
            type="password" 
            placeholder="Password" 
            value={form.password}
            onChange={e => setForm({...form, password: e.target.value})} 
            className="w-full bg-surface-container px-4 py-3 rounded-xl border border-outline-variant/20 focus:border-primary/50 focus:outline-none transition-colors" 
            required 
          />
          
          <button type="submit" disabled={loading} className="btn-primary w-full mt-4">
            {loading ? 'Authenticating...' : 'Sign In'}
          </button>
        </form>
      </div>
    </div>
  );
}
