import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useAuth } from '../context/AuthContext';
import { useSimulate } from '../hooks/useSimulate';
import { DEFAULT_FEATURES } from '../api/dropout';
import { getInstructorStudents } from '../api/recommend';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import SimulationSetupPanel from '../components/panels/SimulationSetupPanel';
import OutcomeDistributionCard from '../components/panels/OutcomeDistributionCard';
import SimulationInterpretationCard from '../components/panels/SimulationInterpretationCard';
import FeatureProjectionGrid from '../components/panels/FeatureProjectionGrid';

export default function SimulatePage() {
  const { user } = useAuth();
  const isInstructor = user?.role === 'instructor' || user?.role === 'admin';

  const [selectedStudentId, setSelectedStudentId] = useState(null);
  const [outcome, setOutcome] = useState(null);
  const [targetWeek, setTargetWeek] = useState(null);
  const simulate = useSimulate();

  // Instructor: load student roster
  const { data: roster } = useQuery({
    queryKey: ['instructor-students'],
    queryFn: getInstructorStudents,
    enabled: isInstructor,
    staleTime: 60_000,
  });
  const students = roster?.students ?? [];

  // Derive active learner and their week/features
  const activeLearner = useMemo(() => {
    if (!isInstructor) {
      return {
        learner_id: user?.learner_id ?? null,
        current_week: user?.current_week ?? 6,
        features: DEFAULT_FEATURES,
      };
    }
    const s = selectedStudentId ? students.find(x => x.learner_id === selectedStudentId) : null;
    return {
      learner_id: s?.learner_id ?? null,
      current_week: s?.features?.current_week_in_course ?? 6,
      features: s?.features ?? DEFAULT_FEATURES,
    };
  }, [isInstructor, selectedStudentId, students, user]);

  const handleRun = ({ target_week, n_simulations }) => {
    if (!activeLearner.learner_id) return;
    setTargetWeek(target_week);
    setOutcome(null);
    simulate.mutate(
      {
        features: activeLearner.features,
        current_week: activeLearner.current_week,
        target_week,
        n_simulations,
      },
      {
        onSuccess: (res) => {
          // Backend wraps outcome inside outcome_distribution
          setOutcome({
            ...(res.outcome_distribution ?? {}),
            feature_distributions: res.feature_distributions,
          });
        },
      }
    );
  };

  return (
    <div className="bg-surface font-body text-on-surface min-h-screen">
      <Navbar />
      <Sidebar />
      <main className="md:ml-64 pt-24 pb-12 px-8 min-h-screen">
        <div className="max-w-5xl mx-auto space-y-8">

          <header>
            <div className="flex items-center gap-2 mb-1">
              <span className="material-symbols-outlined text-primary text-lg">timeline</span>
              <span className="font-label text-xs uppercase tracking-widest text-primary font-bold">Monte Carlo</span>
            </div>
            <h1 className="text-3xl font-headline font-extrabold">Simulate Future Risk</h1>
            <p className="text-sm text-slate-500 mt-1">
              Run thousands of simulated futures to see projected dropout risk distributions.
            </p>
          </header>

          {/* Instructor/Admin: student picker */}
          {isInstructor && (
            <div className="bg-surface-container-low rounded-xl border border-outline-variant/10 p-4">
              <label className="font-label text-xs uppercase tracking-widest text-slate-500 block mb-2">
                Select Student to Simulate
              </label>
              <select
                value={selectedStudentId ?? ''}
                onChange={e => { setSelectedStudentId(e.target.value || null); setOutcome(null); }}
                className="border border-outline-variant/30 rounded-lg px-3 py-2 text-sm font-label bg-surface w-full sm:w-80 focus:outline-none focus:ring-1 focus:ring-primary"
              >
                <option value="">— Select a student —</option>
                {students.map(s => (
                  <option key={s.learner_id} value={s.learner_id}>
                    {s.display_name} — Week {s.features?.current_week_in_course ?? '?'} ({((s.dropout_risk_score ?? 0) * 100).toFixed(0)}% risk)
                  </option>
                ))}
              </select>
              {selectedStudentId && (
                <p className="text-xs text-primary font-label mt-2">
                  Simulating for: <strong>{students.find(s => s.learner_id === selectedStudentId)?.display_name}</strong> · Current week: {activeLearner.current_week}
                </p>
              )}
            </div>
          )}

          {!isInstructor && !activeLearner.learner_id && (
            <div className="px-4 py-3 rounded-xl bg-amber-50 border border-amber-200 text-amber-700 text-sm font-label">
              Your student profile could not be loaded. Please re-login.
            </div>
          )}

          {(activeLearner.learner_id || !isInstructor) && (
            <SimulationSetupPanel
              currentWeek={activeLearner.current_week}
              onRun={handleRun}
              loading={simulate.isPending}
            />
          )}

          {simulate.isPending && (
            <div className="flex items-center gap-3 py-8 justify-center text-slate-400 font-label">
              <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
              Running simulation...
            </div>
          )}

          {simulate.error && (
            <div className="px-4 py-3 rounded-xl bg-red-50 border border-red-200 text-red-600 text-sm font-label">
              {simulate.error.response?.data?.detail ?? simulate.error.message}
            </div>
          )}

          {outcome && (
            <div className="space-y-6">
              <SimulationInterpretationCard outcome={outcome} />
              <OutcomeDistributionCard outcome={outcome} targetWeek={targetWeek} />
              {outcome.feature_distributions && (
                <FeatureProjectionGrid
                  featureDistributions={outcome.feature_distributions}
                  currentFeatures={activeLearner.features}
                />
              )}
            </div>
          )}

        </div>
      </main>
    </div>
  );
}
