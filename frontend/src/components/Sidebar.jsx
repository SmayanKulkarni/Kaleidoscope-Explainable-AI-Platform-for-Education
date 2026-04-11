import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Sidebar() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const { pathname } = useLocation();

  if (!user) return null;

  const studentLinks = [
    { to: '/student', icon: 'dashboard', label: 'Overview' },
    { to: '/action-plan', icon: 'assignment_turned_in', label: 'Action Plan' },
    { to: '#history', icon: 'history', label: 'History' },
  ];

  const instructorLinks = [
    { to: '/instructor', icon: 'dashboard', label: 'Overview' },
    { to: '/what-if', icon: 'science', label: 'What-If Explorer' },
    { to: '#history', icon: 'history', label: 'History' },
  ];

  const links = user.role === 'instructor' ? instructorLinks : studentLinks;

  return (
    <aside className="h-screen w-64 fixed left-0 top-16 bg-surface-container-low flex flex-col pt-4 hidden md:flex border-r border-outline-variant/5">
      <div className="px-6 py-6 mb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl overflow-hidden bg-primary-container ring-2 ring-primary/10">
            <img 
              alt="Profile Picture" 
              className="w-full h-full object-cover" 
              src={user.role === 'student'
                 ? "https://lh3.googleusercontent.com/aida-public/AB6AXuBpgsdRNMZgwLb0SjVMPPrbei3A3painjeRBU8A0NMx1E4MohVFu-LLHzne6lHEq2eXPwT2ecZqHfDwIHmI5sJytYzaupAgrk_Fv3pAL91uJs2xWqpKERu0FYIlC1MU70__oboeN-_GAB6OWE-x_xBlTdchD4kvNXc-5PyEFp1ww6-MtPwaQ0wnfkh-h2hc8XZcXIzbSQLSEIxP58-xDuhisi61DgO8aAqyvF-doWnvlTH6kKHtr85W8VFQSB751q2tgMlldd345-0"
                 : "https://lh3.googleusercontent.com/aida-public/AB6AXuAfvkBGxakSQ1IbSsmxOs8mqCuwuNoDZkMlgJZV9TFE9M-xcH5aNXY8Bcd0CiyPB_k3C2YuOGssbMKy4wMJIPAQYGAkQT_3wscOVI2uOhMtT_pLjI-vD26Eu-GAjuVL-gS0nx4URrs1oveb31viLqKOUbBV5j9kuJChFFPNQrfVgju2R1q6UeG6E-fvl4NLOM51Ia-AfYc_cx-fCc8p0bw_n3cpRdz36bo-_VWnguNNH637QGTEwEG9uzOHU25IQwxYOCZ8HkftvDk"
              }
            />
          </div>
          <div>
            <h3 className="font-headline font-bold text-sm leading-tight text-blue-800">{user.name || 'User'}</h3>
            <p className="font-label text-[10px] text-primary/70 uppercase tracking-tighter">{user.role === 'student' ? 'AI Scholar' : 'Instructor'}</p>
          </div>
        </div>
      </div>
      
      <nav className="flex-1">
        {links.map((v) => {
          const isActive = pathname === v.to;
          return (
            <button
              key={v.to}
              onClick={() => { if(!v.locked) navigate(v.to); }}
              className={`w-full text-left py-3 px-6 flex items-center gap-3 font-label text-sm transition-all ${
                isActive 
                  ? 'text-blue-800 font-bold bg-surface-container-lowest rounded-r-full' 
                  : `text-slate-600 hover:bg-blue-50 ${v.locked ? 'opacity-50 cursor-not-allowed' : ''}`
              }`}
            >
              <span className={`material-symbols-outlined ${isActive ? 'fill-current' : ''}`} style={{ fontVariationSettings: isActive ? "'FILL' 1" : "'FILL' 0" }}>{v.icon}</span> 
              {v.label} {v.locked && <span className="material-symbols-outlined text-[14px]">lock</span>}
            </button>
          )
        })}
      </nav>

      <div className="px-6 py-8">
        <button className="w-full py-3 bg-gradient-to-r from-primary to-primary-container text-on-primary rounded-xl font-bold text-sm shadow-sm hover:shadow-md transition-all flex items-center justify-center gap-2">
          {user.role === 'student' ? <span className="material-symbols-outlined text-sm">bolt</span> : null}
          Get AI Help
        </button>
      </div>

      <div className="border-t border-outline-variant/10 mt-auto pb-8">
        <a className="text-slate-600 hover:bg-blue-50 py-3 px-6 flex items-center gap-3 font-label text-sm transition-all" href="#">
            <span className="material-symbols-outlined">settings</span> Settings
        </a>
        <a className="text-slate-600 hover:bg-blue-50 py-3 px-6 flex items-center gap-3 font-label text-sm transition-all" href="#">
            <span className="material-symbols-outlined">help_outline</span> Support
        </a>
      </div>
    </aside>
  );
}
