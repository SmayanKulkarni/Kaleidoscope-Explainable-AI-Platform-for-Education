import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import { getStudentData } from '../services/xaiService';
import { useAuth } from '../context/AuthContext';
import { getInstructorStudents } from '../api/recommend';

export default function ActionPlan() {
  const { user } = useAuth();
  const isInstructor = user?.role === 'instructor';
  const [data, setData] = useState(null);
  const [selectedStudentId, setSelectedStudentId] = useState(null);

  const { data: roster } = useQuery({
    queryKey: ['instructor-students'],
    queryFn: getInstructorStudents,
    enabled: isInstructor,
    staleTime: 60_000,
  });
  const students = roster?.students ?? [];

  useEffect(() => { getStudentData().then(setData); }, []);

  const selectedStudent = isInstructor && selectedStudentId
    ? students.find(s => s.learner_id === selectedStudentId)
    : null;

  if (!data) return <div className="min-h-screen bg-surface flex justify-center items-center">Loading Action Plan...</div>;

  return (
    <div className="bg-surface font-body text-on-surface min-h-screen">
      <Navbar />
      <Sidebar />
      <main className="md:ml-64 pt-24 pb-12 px-8 min-h-screen">
        <div className="max-w-4xl mx-auto">
          
          <div className="mb-12">
            <div className="flex items-center gap-3 mb-4">
              <span className="material-symbols-outlined text-primary bg-primary/10 p-2 rounded-lg">assignment_turned_in</span>
              <span className="font-label text-sm uppercase tracking-widest text-primary font-bold">Guided Path</span>
            </div>
            <h1 className="text-4xl font-headline font-extrabold tracking-tight text-on-surface mb-4">
              {isInstructor ? 'Student Action Plan' : 'Your Action Plan'}
            </h1>
            <p className="text-xl text-slate-500 font-body leading-relaxed max-w-2xl">
              Based on the AI's analysis, these targeted actions have the highest probability of improving the learning trajectory.
            </p>
            {isInstructor && (
              <div className="mt-6 bg-surface-container-low rounded-xl border border-outline-variant/10 p-4 flex items-center gap-4">
                <label className="font-label text-xs uppercase tracking-widest text-slate-500 shrink-0">Viewing student:</label>
                <select
                  value={selectedStudentId ?? ''}
                  onChange={e => setSelectedStudentId(e.target.value || null)}
                  className="border border-outline-variant/30 rounded-lg px-3 py-2 text-sm font-label bg-surface w-full sm:w-72 focus:outline-none focus:ring-1 focus:ring-primary"
                >
                  <option value="">— Select a student —</option>
                  {students.map(s => (
                    <option key={s.learner_id} value={s.learner_id}>
                      {s.display_name} ({((s.dropout_risk_score ?? 0)*100).toFixed(0)}% risk)
                    </option>
                  ))}
                </select>
                {selectedStudent && (
                  <span className="text-xs font-label text-slate-500">
                    Module: {selectedStudent.current_module ?? '—'} · Week {selectedStudent.features?.current_week_in_course ?? '?'}
                  </span>
                )}
              </div>
            )}
          </div>

          <div className="space-y-6">
            
            <div className="bg-gradient-to-br from-primary to-primary-container rounded-2xl p-8 shadow-lg text-on-primary relative overflow-hidden group hover:shadow-xl transition-shadow scale-[1.02]">
              <div className="absolute right-0 top-0 w-64 h-64 bg-white/10 rounded-full blur-3xl -mr-10 -mt-10 group-hover:bg-white/20 transition-colors"></div>
              
              <div className="relative z-10 flex flex-col md:flex-row gap-8 items-start">
                <div className="w-16 h-16 bg-white/20 rounded-2xl flex items-center justify-center shrink-0 backdrop-blur-sm">
                  <span className="material-symbols-outlined text-4xl text-white">schedule</span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="font-label text-[10px] uppercase font-bold tracking-widest bg-white/20 px-2 py-1 rounded text-white">Top Priority</span>
                    <span className="font-label text-xs font-bold text-primary-fixed bg-on-primary/40 px-2 py-1 rounded">-12% Risk Drop</span>
                  </div>
                  <h3 className="text-2xl font-headline font-bold mb-3 text-white">Attend TA Office Hours Before Assignment #4</h3>
                  <p className="text-on-primary/80 mb-6 leading-relaxed">
                    The model identifies a strong correlation (0.84) between attending office hours this week and higher success rates on Assignment 4.
                  </p>
                  
                  <div className="flex gap-4">
                    {user.role === 'student' ? (
                      <button className="bg-white text-primary px-6 py-3 rounded-xl font-bold font-label text-sm hover:scale-105 transition-transform flex items-center gap-2 shadow-sm">
                        Schedule Session <span className="material-symbols-outlined text-sm">arrow_forward</span>
                      </button>
                    ) : (
                      <div className="bg-on-primary/40 text-white px-6 py-3 rounded-xl font-bold font-label text-sm flex items-center gap-2">
                        <span className="material-symbols-outlined text-sm">lock</span> Student Action Only
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-surface-container-lowest rounded-2xl p-8 border border-outline-variant/10 hover:border-primary/30 transition-colors flex flex-col md:flex-row gap-8 items-start">
               <div className="w-16 h-16 bg-surface-container rounded-2xl flex items-center justify-center shrink-0 text-slate-400">
                  <span className="material-symbols-outlined text-4xl">forum</span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="font-label text-[10px] uppercase font-bold tracking-widest text-slate-500">Secondary Action</span>
                    <span className="font-label text-xs font-bold text-primary bg-primary/10 px-2 py-1 rounded">-5% Risk Drop</span>
                  </div>
                  <h3 className="text-xl font-headline font-bold mb-3">Post 1 Question in the Weekly Forum</h3>
                  <p className="text-slate-500 mb-6 leading-relaxed">
                    Engagement in the forum strongly boosts concept retention for Module 3 topics.
                  </p>
                  <button className="border-2 border-primary text-primary px-6 py-2.5 rounded-xl font-bold font-label text-sm hover:bg-primary/5 transition-colors">
                    Go to Forums
                  </button>
                </div>
            </div>

          </div>

        </div>
      </main>
    </div>
  );
}
