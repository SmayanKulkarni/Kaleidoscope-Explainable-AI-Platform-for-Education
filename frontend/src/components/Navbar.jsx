import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Navbar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  if (!user) return null;

  return (
    <nav className="fixed top-0 w-full z-50 bg-surface/80 backdrop-blur-xl flex justify-between items-center px-8 h-16 max-w-full shadow-sm">
      <div className="flex items-center gap-8">
        <span className="text-2xl font-black tracking-tight text-blue-900 font-headline">LearnLens</span>
        <div className="hidden md:flex gap-6 h-full items-center">
          {user.role === 'student' ? (
            <>
              <button 
                onClick={() => navigate('/student')} 
                className={`font-label text-sm uppercase tracking-wider h-full flex items-center transition-colors ${location.pathname === '/student' ? 'text-blue-700 font-bold border-b-2 border-blue-700' : 'text-slate-500 hover:text-blue-600'}`}
              >
                Student View
              </button>
            </>
          ) : (
            <>
              <button 
                onClick={() => navigate('/instructor')} 
                className={`font-label text-sm uppercase tracking-wider h-full flex items-center transition-colors ${location.pathname === '/instructor' ? 'text-blue-700 font-bold border-b-2 border-blue-700' : 'text-slate-500 hover:text-blue-600'}`}
              >
                Instructor View
              </button>
            </>
          )}
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="hidden lg:flex flex-col items-end">
          <span className="text-xs font-label text-slate-500">Current Role</span>
          <span className="text-sm font-bold text-primary capitalize">{user.role} Role</span>
        </div>
        <button className="material-symbols-outlined text-slate-600 hover:bg-blue-50/50 p-2 rounded-full transition-colors">
          notifications
        </button>
        <button onClick={logout} title="Logout" className="material-symbols-outlined text-error hover:bg-error/10 p-2 rounded-full transition-colors">
          logout
        </button>
        <div className="w-10 h-10 rounded-full bg-slate-200 overflow-hidden ring-2 ring-primary/10">
          <img 
            alt="User Avatar" 
            className="w-full h-full object-cover" 
            src={user.role === 'student' 
              ? "https://lh3.googleusercontent.com/aida-public/AB6AXuDiDnUBSwYnGf5KM0sdkEQ2DIcLurA9cqT8TUD3MZg2f3VtZ3tCgcj1LZb0nw_p8XdOvwZW6FoSsdNXnlJTLNTcAiycvpEZPIW4_ovmMH1TvA9NUCicYocez0Dya6q8b47mKBQALpkN9ksKPSeMoCgDe283X_fMwdA3EnHHfa_L0aY7csvSOlKkFKGNxIO_vcu0zOEF5-ox8k06kI_xf33ofS6MJeSpEfzvdhA6ROF0gpEz0DrmDzr2MyQZqs9vz9G8vmMQx0uUwoE" 
              : "https://lh3.googleusercontent.com/aida-public/AB6AXuAcMt0aTAQvZJ0niRph4aXYsYONeeeyr8ONzByeqDU0aGxo6BbqdKGlvdkUYdwg3iY3AUtajBhNuWNmY-4tUPlJF-ERq_3KY9XbU-FcYDfCVCDrBzQu-xSPBIKuKmmdln9sRYOQIGnDipKQU-BzQUVN600jybrhsWQ3Xv9KverrLZ0rKG1M8m74DuF7tE8wMxOy4kQxeNgMf3ouxBIXfOesHja76B4o3Wwa5QK64uGIK5tNyYPXu4G94QcHj54YAV03mDduPOYHZfA"
            }
          />
        </div>
      </div>
    </nav>
  );
}
