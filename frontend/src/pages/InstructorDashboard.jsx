import { useState, useMemo, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import { getInstructorStudents } from '../api/recommend';
import { track } from '../services/eventTracker';
import { PAGE_VIEW, INSTRUCTOR_STUDENT_SELECTED } from '../constants/eventTypes';

const RISK_COLORS = {
  High:    { bg: 'bg-red-100',   text: 'text-red-700',   bar: 'bg-red-500'   },
  Medium:  { bg: 'bg-amber-100', text: 'text-amber-700', bar: 'bg-amber-500' },
  Low:     { bg: 'bg-green-100', text: 'text-green-700', bar: 'bg-green-500' },
  Unknown: { bg: 'bg-slate-100', text: 'text-slate-500', bar: 'bg-slate-400' },
};

const TRAJ_ICON = {
  improving:      { icon: 'trending_up',   color: 'text-green-600' },
  stable:         { icon: 'trending_flat', color: 'text-slate-500' },
  deteriorating:  { icon: 'trending_down', color: 'text-red-600'   },
  at_risk_stable: { icon: 'warning',       color: 'text-amber-600' },
};

export default function InstructorDashboard() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [sort, setSort] = useState('risk_desc');

  useEffect(() => { track(PAGE_VIEW, { page: '/instructor' }); }, []);

  const { data: roster, isLoading, isError } = useQuery({
    queryKey: ['instructor-students'],
    queryFn: getInstructorStudents,
    staleTime: 60_000,
  });

  const students = roster?.students ?? [];

  const cohortStats = useMemo(() => {
    if (!students.length) return null;
    const risks = students.map(s => s.dropout_risk_score ?? 0);
    const avg = risks.reduce((a, b) => a + b, 0) / risks.length;
    const high   = students.filter(s => (s.dropout_risk_score ?? 0) >= 0.6).length;
    const medium = students.filter(s => (s.dropout_risk_score ?? 0) >= 0.35 && (s.dropout_risk_score ?? 0) < 0.6).length;
    const low    = students.filter(s => (s.dropout_risk_score ?? 0) < 0.35).length;
    return { avg, high, medium, low, total: students.length };
  }, [students]);

  const filtered = useMemo(() => {
    let list = students.filter(s =>
      (s.display_name ?? '').toLowerCase().includes(search.toLowerCase()) ||
      (s.learner_id   ?? '').toLowerCase().includes(search.toLowerCase())
    );
    if (sort === 'risk_desc') list = [...list].sort((a, b) => (b.dropout_risk_score ?? 0) - (a.dropout_risk_score ?? 0));
    if (sort === 'risk_asc')  list = [...list].sort((a, b) => (a.dropout_risk_score ?? 0) - (b.dropout_risk_score ?? 0));
    if (sort === 'name')      list = [...list].sort((a, b) => (a.display_name ?? '').localeCompare(b.display_name ?? ''));
    return list;
  }, [students, search, sort]);

  return (
    <div className="bg-surface font-body text-on-surface min-h-screen">
      <Navbar />
      <Sidebar />
      <main className="md:ml-64 pt-20 pb-12 px-6 min-h-screen">
        <div className="max-w-7xl mx-auto">

          <div className="mb-8">
            <div className="flex items-center gap-2 mb-1">
              <span className="material-symbols-outlined text-primary text-lg">groups</span>
              <span className="font-label text-xs uppercase tracking-widest text-primary font-bold">Instructor Dashboard</span>
            </div>
            <h1 className="text-3xl font-headline font-extrabold">Cohort Risk Overview</h1>
            <p className="text-slate-500 text-sm mt-1">Real-time dropout risk analysis for all enrolled students.</p>
          </div>

          {isLoading && (
            <div className="flex items-center justify-center py-24">
              <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mr-3" />
              <span className="text-slate-500 font-label">Loading cohort data...</span>
            </div>
          )}

          {isError && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-red-700 text-sm font-label">
              Failed to load student roster. Make sure the backend is running and you are logged in as instructor.
            </div>
          )}

          {!isLoading && !isError && cohortStats && (
            <>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                <div className="bg-surface-container-low rounded-xl p-5 border border-outline-variant/10">
                  <span className="font-label text-xs uppercase tracking-widest text-slate-500">Total Students</span>
                  <div className="text-3xl font-headline font-bold text-on-surface mt-2">{cohortStats.total}</div>
                </div>
                <div className="bg-red-50 rounded-xl p-5 border border-red-100">
                  <span className="font-label text-xs uppercase tracking-widest text-red-600">High Risk</span>
                  <div className="text-3xl font-headline font-bold text-red-700 mt-2">{cohortStats.high}</div>
                  <div className="text-[10px] text-red-400 font-label">risk ≥ 60%</div>
                </div>
                <div className="bg-amber-50 rounded-xl p-5 border border-amber-100">
                  <span className="font-label text-xs uppercase tracking-widest text-amber-600">Medium Risk</span>
                  <div className="text-3xl font-headline font-bold text-amber-700 mt-2">{cohortStats.medium}</div>
                  <div className="text-[10px] text-amber-400 font-label">35–60%</div>
                </div>
                <div className="bg-green-50 rounded-xl p-5 border border-green-100">
                  <span className="font-label text-xs uppercase tracking-widest text-green-600">Low Risk</span>
                  <div className="text-3xl font-headline font-bold text-green-700 mt-2">{cohortStats.low}</div>
                  <div className="text-[10px] text-green-400 font-label">risk &lt; 35%</div>
                </div>
              </div>

              <div className="bg-surface-container-low rounded-xl border border-outline-variant/10 p-5 mb-8">
                <div className="flex items-center justify-between mb-3">
                  <span className="font-label text-xs uppercase tracking-widest text-slate-500">Average Cohort Risk</span>
                  <span className="font-headline font-bold text-lg text-on-surface">{(cohortStats.avg * 100).toFixed(1)}%</span>
                </div>
                <div className="h-3 bg-surface-container-high rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${cohortStats.avg >= 0.6 ? 'bg-red-500' : cohortStats.avg >= 0.35 ? 'bg-amber-500' : 'bg-green-500'}`}
                    style={{ width: `${cohortStats.avg * 100}%` }}
                  />
                </div>
              </div>

              <div className="bg-surface-container-low rounded-xl border border-outline-variant/10 overflow-hidden">
                <div className="px-6 py-4 border-b border-outline-variant/10 flex flex-col sm:flex-row sm:items-center gap-3">
                  <h2 className="font-headline font-bold text-lg flex-1">Student Roster</h2>
                  <input
                    type="text"
                    placeholder="Search by name or ID..."
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                    className="border border-outline-variant/30 rounded-lg px-3 py-1.5 text-sm font-label focus:outline-none focus:ring-1 focus:ring-primary bg-surface w-full sm:w-56"
                  />
                  <select
                    value={sort}
                    onChange={e => setSort(e.target.value)}
                    className="border border-outline-variant/30 rounded-lg px-3 py-1.5 text-sm font-label focus:outline-none focus:ring-1 focus:ring-primary bg-surface"
                  >
                    <option value="risk_desc">Highest Risk First</option>
                    <option value="risk_asc">Lowest Risk First</option>
                    <option value="name">Name A–Z</option>
                  </select>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left">
                    <thead className="bg-surface-container font-label text-[10px] uppercase tracking-widest text-slate-500">
                      <tr>
                        <th className="px-6 py-3">Student</th>
                        <th className="px-6 py-3">Risk Score</th>
                        <th className="px-6 py-3">Risk Level</th>
                        <th className="px-6 py-3">Trajectory</th>
                        <th className="px-6 py-3">Module</th>
                        <th className="px-6 py-3">Week</th>
                        <th className="px-6 py-3">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-outline-variant/5">
                      {filtered.map(s => {
                        const risk   = s.dropout_risk_score ?? 0;
                        const label  = s.risk_label ?? 'Unknown';
                        const colors = RISK_COLORS[label] ?? RISK_COLORS.Unknown;
                        const traj   = TRAJ_ICON[s.risk_trajectory] ?? TRAJ_ICON.stable;
                        return (
                          <tr key={s.learner_id} className="hover:bg-primary/5 transition-colors">
                            <td className="px-6 py-4">
                              <div className="flex items-center gap-3">
                                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                                  <span className="material-symbols-outlined text-primary text-sm">person</span>
                                </div>
                                <div>
                                  <div className="font-label font-bold text-sm">{s.display_name}</div>
                                  <div className="text-[10px] text-slate-400">{s.learner_id}</div>
                                </div>
                              </div>
                            </td>
                            <td className="px-6 py-4">
                              <div className="flex items-center gap-2">
                                <div className="w-24 h-2 bg-surface-container-high rounded-full overflow-hidden">
                                  <div className={`h-full ${colors.bar}`} style={{ width: `${risk * 100}%` }} />
                                </div>
                                <span className="font-label font-bold text-sm">{(risk * 100).toFixed(0)}%</span>
                              </div>
                            </td>
                            <td className="px-6 py-4">
                              <span className={`px-2 py-0.5 rounded text-[10px] font-label font-bold ${colors.bg} ${colors.text}`}>{label}</span>
                            </td>
                            <td className="px-6 py-4">
                              <span className={`material-symbols-outlined text-sm ${traj.color}`}>{traj.icon}</span>
                              <span className={`ml-1 text-xs font-label ${traj.color}`}>{(s.risk_trajectory ?? 'stable').replace('_', ' ')}</span>
                            </td>
                            <td className="px-6 py-4 text-sm font-label text-slate-500">{s.current_module ?? '—'}</td>
                            <td className="px-6 py-4 text-sm font-label text-slate-500">W{s.features?.current_week_in_course ?? '?'}</td>
                            <td className="px-6 py-4">
                              <button
                                onClick={() => navigate('/xai/instructor', { state: { learner_id: s.learner_id } })}
                                className="px-3 py-1.5 bg-primary text-on-primary rounded-lg text-xs font-label font-bold hover:bg-primary/90 transition-colors flex items-center gap-1"
                              >
                                <span className="material-symbols-outlined text-sm">analytics</span> Analyse
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  {filtered.length === 0 && (
                    <div className="text-center py-12 text-slate-400 font-label text-sm">No students match your search.</div>
                  )}
                </div>
              </div>
            </>
          )}

          {!isLoading && !isError && students.length === 0 && (
            <div className="text-center py-20 text-slate-400 font-label">
              <span className="material-symbols-outlined text-4xl block mb-3">group_off</span>
              No students enrolled yet. Ask an admin to assign students.
            </div>
          )}

        </div>
      </main>
    </div>
  );
}
