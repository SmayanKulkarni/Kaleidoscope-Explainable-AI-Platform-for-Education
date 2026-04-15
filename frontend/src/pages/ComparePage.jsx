import { lazy, useState, useMemo, useEffect } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useAuth } from '../context/AuthContext';
import { track } from '../services/eventTracker';
import { PAGE_VIEW, COMPARE_RUN } from '../constants/eventTypes';
import { useCompare } from '../hooks/useCompare';
import { DEFAULT_FEATURES } from '../api/dropout';
import { compareNarrate, getInstructorStudents } from '../api/recommend';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import RiskScoreCard from '../components/panels/RiskScoreCard';
import GraphCard from '../components/layout/GraphCard';
import { MODEL_COLORS } from '../lib/colors';

const ModelCompareGraph = lazy(() => import('../components/graphs/ModelCompareGraph'));

const GRAPH_LEGEND = [
  { label: 'GBM Feature',   color: MODEL_COLORS.gbm          },
  { label: 'LSTM Feature',  color: MODEL_COLORS.lstm         },
  { label: 'Shared',        color: MODEL_COLORS.shared       },
  { label: 'GBM Model',     color: MODEL_COLORS['model-gbm'] },
  { label: 'LSTM Model',    color: MODEL_COLORS['model-lstm']},
];

export default function ComparePage() {
  const { user } = useAuth();
  const isInstructor = user?.role === 'instructor' || user?.role === 'admin';
  const [selectedStudentId, setSelectedStudentId] = useState(null);
  const [narration, setNarration] = useState(null);

  useEffect(() => { track(PAGE_VIEW, { page: '/xai/compare' }); }, []);

  // Instructor/admin: load student roster for picker
  const { data: roster } = useQuery({
    queryKey: ['instructor-students'],
    queryFn: getInstructorStudents,
    enabled: isInstructor,
    staleTime: 60_000,
  });
  const students = roster?.students ?? [];

  // Determine which learner to compare
  const learner_id = isInstructor
    ? (selectedStudentId ?? null)
    : (user?.learner_id ?? user?.id ?? null);

  // Get that student's features from the roster (for instructors)
  const features = useMemo(() => {
    if (!isInstructor || !selectedStudentId) return DEFAULT_FEATURES;
    const s = students.find(x => x.learner_id === selectedStudentId);
    return s?.features ?? DEFAULT_FEATURES;
  }, [isInstructor, selectedStudentId, students]);

  const { data: compareResult, isLoading, error } = useCompare(
    { features, learner_id },
    { enabled: !!learner_id }
  );

  const narrateMutation = useMutation({
    mutationFn: () => compareNarrate(compareResult),
    onSuccess: setNarration,
  });

  const gbmScore  = compareResult?.gbm_score;
  const lstmScore = compareResult?.lstm_score;

  return (
    <div className="bg-surface font-body text-on-surface min-h-screen">
      <Navbar />
      <Sidebar />
      <main className="md:ml-64 pt-20 pb-12 px-6 min-h-screen">
        <div className="max-w-7xl mx-auto">

          <div className="mb-8">
            <div className="flex items-center gap-2 mb-1">
              <span className="material-symbols-outlined text-primary text-lg">compare</span>
              <span className="font-label text-xs uppercase tracking-widest text-primary font-bold">Model Comparison</span>
            </div>
            <h1 className="text-3xl font-headline font-extrabold">GBM vs LSTM</h1>
            {error && (
              <div className="mt-3 px-4 py-2 rounded-lg bg-error/10 text-error text-sm font-label border border-error/20">
                {error.response?.data?.detail ?? error.message}
              </div>
            )}
          </div>

          {/* Instructor/Admin: student picker */}
          {isInstructor && (
            <div className="mb-6 bg-surface-container-low rounded-xl border border-outline-variant/10 p-4">
              <label className="font-label text-xs uppercase tracking-widest text-slate-500 block mb-2">
                Select Student to Compare
              </label>
              <select
                value={selectedStudentId ?? ''}
                onChange={e => { setSelectedStudentId(e.target.value || null); setNarration(null); }}
                className="border border-outline-variant/30 rounded-lg px-3 py-2 text-sm font-label bg-surface w-full sm:w-80 focus:outline-none focus:ring-1 focus:ring-primary"
              >
                <option value="">— Select a student —</option>
                {students.map(s => (
                  <option key={s.learner_id} value={s.learner_id}>
                    {s.display_name} ({((s.dropout_risk_score ?? 0) * 100).toFixed(0)}% risk)
                  </option>
                ))}
              </select>
              {!selectedStudentId && (
                <p className="text-xs text-slate-400 font-label mt-2">Select a student to run the model comparison.</p>
              )}
            </div>
          )}

          {!isInstructor && !learner_id && (
            <div className="mb-6 px-4 py-3 rounded-xl bg-amber-50 border border-amber-200 text-amber-700 text-sm font-label">
              Could not determine your student profile. Please re-login.
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div>
              <p className="font-label text-xs uppercase text-slate-500 tracking-widest mb-2">GBM (Static)</p>
              <RiskScoreCard
                riskScore={gbmScore}
                riskLabel={compareResult?.gbm_risk_label}
                loading={isLoading}
              />
            </div>
            <div>
              <p className="font-label text-xs uppercase text-slate-500 tracking-widest mb-2">LSTM (Temporal)</p>
              <RiskScoreCard
                riskScore={lstmScore}
                riskLabel={compareResult?.lstm_risk_label}
                loading={isLoading}
              />
            </div>
          </div>

          {compareResult && (
            <div className={`mb-6 flex items-center gap-3 px-5 py-4 rounded-xl border text-sm font-label font-bold ${compareResult.disagreement_flag ? 'bg-amber-50 border-amber-200 text-amber-800' : 'bg-green-50 border-green-200 text-green-800'}`}>
              <span className="material-symbols-outlined text-xl">
                {compareResult.disagreement_flag ? 'warning' : 'check_circle'}
              </span>
              {compareResult.interpretation}
              {compareResult.score_delta != null && (
                <span className="ml-auto text-xs font-bold bg-white/60 px-2 py-1 rounded-full border">
                  Δ {(compareResult.score_delta * 100).toFixed(1)}pp
                </span>
              )}
            </div>
          )}

          <GraphCard
            title="GBM vs LSTM Feature Bipartite Graph"
            legend={GRAPH_LEGEND}
            height={480}
            action={
              compareResult && (
                <button
                  onClick={() => narrateMutation.mutate()}
                  disabled={narrateMutation.isPending}
                  className="px-3 py-1.5 bg-primary text-on-primary rounded-lg text-xs font-label font-bold hover:bg-primary/90 transition-colors flex items-center gap-1 disabled:opacity-50"
                >
                  {narrateMutation.isPending
                    ? <><div className="w-3 h-3 border border-on-primary border-t-transparent rounded-full animate-spin" /> Narrating...</>
                    : <><span className="material-symbols-outlined text-sm">auto_awesome</span> Explain Graph</>}
                </button>
              )
            }
          >
            <ModelCompareGraph compareResult={compareResult} height={480} />
          </GraphCard>

          {narration && (
            <div className="bg-surface-container-lowest rounded-2xl p-6 border border-primary/10 mt-6">
              <div className="flex items-center gap-2 mb-4">
                <span className="material-symbols-outlined text-primary text-lg">auto_awesome</span>
                <h3 className="font-headline font-bold text-base">AI Graph Narration</h3>
                <span className="ml-auto text-[10px] font-label text-slate-400">LLM grounded in bipartite data</span>
              </div>
              <p className="text-sm text-on-surface/80 leading-relaxed italic mb-4">"{narration.summary}"</p>
              {narration.key_insights?.length > 0 && (
                <ul className="space-y-2 mb-4">
                  {narration.key_insights.map((ins, i) => (
                    <li key={i} className="flex gap-2 text-sm text-on-surface/70">
                      <span className="text-primary shrink-0">▸</span> {ins}
                    </li>
                  ))}
                </ul>
              )}
              {narration.strongest_drivers?.length > 0 && (
                <div>
                  <span className="font-label text-xs uppercase tracking-widest text-slate-400 block mb-2">Strongest Drivers</span>
                  <div className="flex flex-wrap gap-2">
                    {narration.strongest_drivers.slice(0, 4).map((d, i) => (
                      <span key={i} className="bg-primary/5 border border-primary/10 text-primary px-3 py-1 rounded-full text-xs font-label font-bold">
                        {d.feature ?? d}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {compareResult?.feature_disagreement?.length > 0 && (
            <div className="mt-6 bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10">
              <h3 className="font-headline font-bold text-sm mb-3 text-amber-700">Feature Disagreement</h3>
              <p className="text-xs font-label text-slate-500 mb-3">
                These features are salient for GBM but not the most active temporal week in LSTM:
              </p>
              <div className="flex flex-wrap gap-2">
                {compareResult.feature_disagreement.map((f) => (
                  <span key={f} className="bg-amber-50 border border-amber-200 text-amber-800 px-3 py-1 rounded-full text-xs font-label font-bold">
                    {f}
                  </span>
                ))}
              </div>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}
