import { useState, useMemo } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import { useAuth } from '../context/AuthContext';
import { whatif, DEFAULT_FEATURES, MUTABLE_FEATURES } from '../api/dropout';
import { getInstructorStudents, instructorWhatIfForStudent } from '../api/recommend';

export default function WhatIfExplorer() {
  const { user } = useAuth();
  const isInstructor = user?.role === 'instructor';

  const { data: roster } = useQuery({
    queryKey: ['instructor-students'],
    queryFn: getInstructorStudents,
    enabled: isInstructor,
    staleTime: 60_000,
  });

  const students = roster?.students ?? [];
  const [selectedStudentId, setSelectedStudentId] = useState(null);

  const baseFeatures = useMemo(() => {
    if (!isInstructor || !selectedStudentId) return DEFAULT_FEATURES;
    const s = students.find(x => x.learner_id === selectedStudentId);
    return s?.features ?? DEFAULT_FEATURES;
  }, [isInstructor, selectedStudentId, students]);

  const [overrides, setOverrides] = useState({});
  const features = useMemo(() => ({ ...baseFeatures, ...overrides }), [baseFeatures, overrides]);

  const whatifMutation = useMutation({
    mutationFn: () =>
      isInstructor && selectedStudentId
        ? instructorWhatIfForStudent(selectedStudentId, overrides)
        : whatif(baseFeatures, overrides),
  });

  const handleSlider = (id, value) => {
    const newOv = { ...overrides, [id]: parseFloat(value) };
    setOverrides(newOv);
  };

  const handleRun = () => whatifMutation.mutate();

  const handleReset = () => {
    setOverrides({});
    whatifMutation.reset();
  };

  const result = whatifMutation.data;
  const baseRisk  = result?.base_risk  ?? null;
  const newRisk   = result?.new_risk   ?? null;
  const riskDelta = baseRisk !== null && newRisk !== null ? ((newRisk - baseRisk) * 100) : null;

  return (
    <div className="bg-surface font-body text-on-surface min-h-screen">
      <Navbar />
      <Sidebar />
      <main className="md:ml-64 pt-20 pb-12 px-6 min-h-screen">
        <div className="max-w-6xl mx-auto">

          <div className="mb-8">
            <div className="flex items-center gap-2 mb-1">
              <span className="material-symbols-outlined text-primary bg-primary/10 p-1.5 rounded-lg text-lg">science</span>
              <span className="font-label text-xs uppercase tracking-widest text-primary font-bold">Simulator</span>
            </div>
            <h1 className="text-3xl font-headline font-extrabold">What-If Explorer</h1>
            <p className="text-slate-500 text-sm mt-1">Adjust feature values to see how dropout risk changes.</p>
          </div>

          {isInstructor && (
            <div className="mb-6 bg-surface-container-low rounded-xl border border-outline-variant/10 p-4">
              <label className="font-label text-xs uppercase tracking-widest text-slate-500 block mb-2">Select Student</label>
              <select
                value={selectedStudentId ?? ''}
                onChange={e => { setSelectedStudentId(e.target.value || null); setOverrides({}); whatifMutation.reset(); }}
                className="border border-outline-variant/30 rounded-lg px-3 py-2 text-sm font-label bg-surface w-full sm:w-80 focus:outline-none focus:ring-1 focus:ring-primary"
              >
                <option value="">— My own profile (default) —</option>
                {students.map(s => (
                  <option key={s.learner_id} value={s.learner_id}>{s.display_name} ({(( s.dropout_risk_score ?? 0)*100).toFixed(0)}% risk)</option>
                ))}
              </select>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-5 space-y-4">
              <div className="bg-surface-container-low rounded-2xl p-6 border border-outline-variant/10">
                <h3 className="font-headline font-bold text-lg mb-5">Feature Sliders</h3>
                <div className="space-y-6">
                  {MUTABLE_FEATURES.map(f => {
                    const current = overrides[f.id] ?? baseFeatures[f.id] ?? f.min;
                    const changed = overrides[f.id] !== undefined;
                    return (
                      <div key={f.id}>
                        <div className="flex justify-between mb-2 items-end">
                          <label className={`font-label text-xs font-medium ${changed ? 'text-primary' : 'text-slate-500'}`}>
                            {f.label} {changed && <span className="text-[9px] bg-primary/10 text-primary px-1 rounded">modified</span>}
                          </label>
                          <span className="font-mono text-xs bg-surface-container px-2 py-0.5 rounded">{current}{f.unit}</span>
                        </div>
                        <input
                          type="range"
                          min={f.min} max={f.max} step={f.step}
                          value={current}
                          onChange={e => handleSlider(f.id, e.target.value)}
                          className="w-full h-2 rounded-lg appearance-none cursor-pointer accent-primary"
                        />
                      </div>
                    );
                  })}
                </div>
                <div className="mt-6 pt-5 border-t border-outline-variant/10 flex gap-3">
                  <button
                    onClick={handleRun}
                    disabled={whatifMutation.isPending}
                    className="flex-1 py-2.5 bg-primary text-on-primary rounded-xl text-sm font-bold hover:bg-primary/90 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                  >
                    {whatifMutation.isPending
                      ? <><div className="w-4 h-4 border-2 border-on-primary border-t-transparent rounded-full animate-spin" /> Running...</>
                      : <><span className="material-symbols-outlined text-sm">play_arrow</span> Run Simulation</>}
                  </button>
                  <button
                    onClick={handleReset}
                    className="px-4 py-2.5 bg-surface-container text-on-surface rounded-xl text-sm font-bold hover:bg-surface-container-high transition-colors"
                  >
                    Reset
                  </button>
                </div>
              </div>
            </div>

            <div className="lg:col-span-7 space-y-4">
              {!result && !whatifMutation.isPending && (
                <div className="flex items-center justify-center h-48 bg-surface-container-low rounded-2xl border border-outline-variant/10 text-slate-400 font-label text-sm">
                  <div className="text-center">
                    <span className="material-symbols-outlined text-3xl block mb-2">science</span>
                    Adjust sliders and click Run Simulation
                  </div>
                </div>
              )}

              {result && (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10 text-center">
                      <span className="font-label text-xs uppercase tracking-widest text-slate-500 block mb-2">Baseline Risk</span>
                      <div className="text-5xl font-headline font-black text-on-surface">{((baseRisk ?? 0)*100).toFixed(1)}%</div>
                    </div>
                    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10 text-center">
                      <span className="font-label text-xs uppercase tracking-widest text-slate-500 block mb-2">Projected Risk</span>
                      <div className={`text-5xl font-headline font-black ${riskDelta < 0 ? 'text-green-600' : riskDelta > 0 ? 'text-red-600' : 'text-on-surface'}`}>
                        {((newRisk ?? 0)*100).toFixed(1)}%
                      </div>
                      {riskDelta !== null && (
                        <div className={`mt-2 inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold ${riskDelta < 0 ? 'bg-green-100 text-green-700' : riskDelta > 0 ? 'bg-red-100 text-red-700' : 'bg-slate-100 text-slate-500'}`}>
                          <span className="material-symbols-outlined text-[12px]">{riskDelta < 0 ? 'arrow_downward' : riskDelta > 0 ? 'arrow_upward' : 'horizontal_rule'}</span>
                          {Math.abs(riskDelta).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  </div>

                  {result.feature_impacts && Object.keys(result.feature_impacts).length > 0 && (
                    <div className="bg-surface-container-low rounded-2xl p-6 border border-outline-variant/10">
                      <h3 className="font-headline font-bold text-base mb-4">Feature Impact Breakdown</h3>
                      <div className="space-y-3">
                        {Object.entries(result.feature_impacts)
                          .sort(([,a],[,b]) => Math.abs(b) - Math.abs(a))
                          .slice(0, 8)
                          .map(([feat, impact]) => (
                            <div key={feat} className="flex items-center gap-3">
                              <span className="font-label text-xs text-slate-500 w-40 truncate">{feat}</span>
                              <div className="flex-1 h-3 bg-surface-container-high rounded-full overflow-hidden relative">
                                <div className="absolute inset-y-0 left-1/2 w-px bg-slate-300" />
                                <div
                                  className={`absolute h-full ${impact > 0 ? 'bg-red-400 left-1/2' : 'bg-green-400 right-1/2'}`}
                                  style={{ width: `${Math.min(Math.abs(impact)*200, 50)}%` }}
                                />
                              </div>
                              <span className={`font-label text-xs font-bold w-14 text-right ${impact > 0 ? 'text-red-600' : 'text-green-600'}`}>
                                {impact > 0 ? '+' : ''}{(impact * 100).toFixed(2)}%
                              </span>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
