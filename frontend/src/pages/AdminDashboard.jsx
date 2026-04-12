import { lazy, Suspense, useEffect, useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import { getHealth, getMlopsHealth, getMlopsMetrics, getMlopsDriftReport, triggerRetrain, triggerReload } from '../api/dropout';
import { recommendHealth, recommendStudent, explainFairness, getAdminEnrollments, getAdminInstructors, getAdminStudents, addAdminEnrollment, deleteAdminEnrollment } from '../api/recommend';
import FairnessAuditPanel from '../components/panels/FairnessAuditPanel';
import GraphCard from '../components/layout/GraphCard';

const CausalDagGraph = lazy(() => import('../components/graphs/CausalDagGraph'));

// Representative sample — includes protected attributes so the fairness audit has
// demographic data to bucket and compare across groups.
const ADMIN_SAMPLE_ITEMS = [
  { item_id: 'Module 4 Quiz', features: { difficulty: 0.6, time_required: 30, explicit_gender: 'M', explicit_age_band: '0-35', explicit_disability: 'N' } },
  { item_id: 'TA Office Hours', features: { difficulty: 0.2, time_required: 60, explicit_gender: 'F', explicit_age_band: '0-35', explicit_disability: 'N' } },
  { item_id: 'Forum Week 6', features: { difficulty: 0.1, time_required: 15, explicit_gender: 'M', explicit_age_band: '35-55', explicit_disability: 'N' } },
  { item_id: 'Practice Set A', features: { difficulty: 0.5, time_required: 45, explicit_gender: 'F', explicit_age_band: '35-55', explicit_disability: 'Y' } },
  { item_id: 'Video Lecture 7', features: { difficulty: 0.3, time_required: 20, explicit_gender: 'M', explicit_age_band: '55<=', explicit_disability: 'N' } },
  { item_id: 'Peer Review Task', features: { difficulty: 0.4, time_required: 35, explicit_gender: 'F', explicit_age_band: '0-35', explicit_disability: 'Y' } },
];

function HealthDot({ ok }) {
  return <span className={`w-2.5 h-2.5 rounded-full inline-block ${ok ? 'bg-green-500' : 'bg-red-400'}`} />;
}

function StatCard({ label, value, icon, sub }) {
  return (
    <div className="bg-surface-container-lowest rounded-xl p-6 border border-outline-variant/5 shadow-sm">
      <div className="flex items-center gap-2 mb-3">
        <span className="material-symbols-outlined text-primary text-lg">{icon}</span>
        <p className="text-xs font-bold text-slate-500 uppercase tracking-wider">{label}</p>
      </div>
      <p className="text-3xl font-headline font-bold text-on-surface">{value ?? '—'}</p>
      {sub && <p className="text-xs text-slate-400 mt-1">{sub}</p>}
    </div>
  );
}

export default function AdminDashboard() {
  const [canaryFraction, setCanaryFraction] = useState(0.1);
  const [runFairness, setRunFairness] = useState(false);
  const [fairnessExplanation, setFairnessExplanation] = useState(null);
  const [enrollForm, setEnrollForm] = useState({ learner_id: '', instructor_profile_id: '', course_id: 'COURSE-001' });
  const [enrollError, setEnrollError] = useState(null);
  const [selectedStudentId, setSelectedStudentId] = useState('');
  const [selectedInstructorId, setSelectedInstructorId] = useState('');

  const { data: fairnessSample, isLoading: fairnessLoading } = useQuery({
    queryKey: ['admin-fairness-sample'],
    queryFn: () => recommendStudent('admin_sample', ADMIN_SAMPLE_ITEMS, 6),
    enabled: runFairness,
    staleTime: 5 * 60 * 1000,
  });

  const explainMutation = useMutation({
    mutationFn: (report) => explainFairness(report),
    onSuccess: (data) => setFairnessExplanation(data),
  });

  const { data: health } = useQuery({ queryKey: ['health'], queryFn: getHealth, refetchInterval: 30_000 });
  const { data: mlopsHealth } = useQuery({ queryKey: ['mlops-health'], queryFn: getMlopsHealth, refetchInterval: 30_000 });
  const { data: mlopsMetrics } = useQuery({ queryKey: ['mlops-metrics'], queryFn: getMlopsMetrics });
  const { data: driftReport } = useQuery({ queryKey: ['drift-report'], queryFn: getMlopsDriftReport });
  const { data: recHealth } = useQuery({ queryKey: ['recommend-health'], queryFn: recommendHealth });

  const retrain = useMutation({ mutationFn: triggerRetrain });
  const reload = useMutation({ mutationFn: () => triggerReload(canaryFraction) });

  const { data: enrollments, refetch: refetchEnrollments } = useQuery({
    queryKey: ['admin-enrollments'],
    queryFn: getAdminEnrollments,
    staleTime: 30_000,
  });

  const { data: adminStudents } = useQuery({
    queryKey: ['admin-students'],
    queryFn: getAdminStudents,
    staleTime: 60_000,
  });

  const { data: instructors } = useQuery({
    queryKey: ['admin-instructors'],
    queryFn: getAdminInstructors,
    staleTime: 60_000,
  });

  const studentList = adminStudents?.students ?? [];
  const instructorList = instructors?.instructors ?? [];

  useEffect(() => {
    if (!selectedStudentId && studentList.length > 0) {
      setSelectedStudentId(studentList[0].learner_id);
    }
  }, [selectedStudentId, studentList]);

  useEffect(() => {
    if (!selectedInstructorId && instructorList.length > 0) {
      setSelectedInstructorId(instructorList[0].profile_id ?? instructorList[0].instructor_profile_id ?? '');
    }
  }, [selectedInstructorId, instructorList]);

  const selectedStudent = studentList.find((student) => student.learner_id === selectedStudentId) ?? null;
  const selectedInstructor = instructorList.find((instructor) => (instructor.profile_id ?? instructor.instructor_profile_id) === selectedInstructorId) ?? null;

  const addEnrollMutation = useMutation({
    mutationFn: addAdminEnrollment,
    onSuccess: () => { setEnrollForm({ learner_id: '', instructor_profile_id: '', course_id: 'COURSE-001' }); setEnrollError(null); refetchEnrollments(); },
    onError: (e) => setEnrollError(e.response?.data?.detail ?? 'Failed to enroll'),
  });

  const removeEnrollMutation = useMutation({
    mutationFn: deleteAdminEnrollment,
    onSuccess: () => refetchEnrollments(),
  });

  return (
    <div className="bg-surface font-body text-on-surface min-h-screen">
      <Navbar />
      <Sidebar />
      <main className="md:ml-64 pt-24 pb-12 px-8 min-h-screen">
        <div className="max-w-7xl mx-auto space-y-8">

          <header>
            <h1 className="text-4xl font-headline font-extrabold tracking-tight text-on-background mb-1">Admin Dashboard</h1>
            <p className="text-on-surface-variant font-body text-sm">System health, model metrics, drift monitoring, and MLOps controls.</p>
          </header>

          <section className="space-y-3">
            <h2 className="font-headline font-bold text-lg">Identity Explorer</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-surface-container-lowest rounded-xl p-6 border border-outline-variant/5 shadow-sm space-y-4">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-primary">school</span>
                  <h3 className="font-bold text-sm">Select Student</h3>
                </div>
                <select
                  value={selectedStudentId}
                  onChange={(e) => setSelectedStudentId(e.target.value)}
                  className="border border-outline-variant/30 rounded-lg px-3 py-2 text-sm font-label bg-surface w-full focus:outline-none focus:ring-1 focus:ring-primary"
                >
                  <option value="">— Select a student —</option>
                  {studentList.map((student) => (
                    <option key={student.learner_id} value={student.learner_id}>
                      {student.display_name ?? student.learner_id} ({student.learner_id})
                    </option>
                  ))}
                </select>
                {selectedStudent ? (
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="rounded-lg bg-surface p-3 border border-outline-variant/10">
                      <div className="text-slate-400 uppercase tracking-widest">Learner ID</div>
                      <div className="font-bold mt-1 break-all">{selectedStudent.learner_id}</div>
                    </div>
                    <div className="rounded-lg bg-surface p-3 border border-outline-variant/10">
                      <div className="text-slate-400 uppercase tracking-widest">Risk</div>
                      <div className="font-bold mt-1">{((selectedStudent.dropout_risk_score ?? 0) * 100).toFixed(0)}%</div>
                    </div>
                    <div className="rounded-lg bg-surface p-3 border border-outline-variant/10 col-span-2">
                      <div className="text-slate-400 uppercase tracking-widest">Current Snapshot</div>
                      <div className="font-bold mt-1">{selectedStudent.current_module ?? '—'} · {selectedStudent.current_presentation ?? '—'} · Week {selectedStudent.features?.current_week_in_course ?? '—'}</div>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-slate-400">Choose a learner to inspect snapshot-backed dashboard data.</p>
                )}
              </div>

              <div className="bg-surface-container-lowest rounded-xl p-6 border border-outline-variant/5 shadow-sm space-y-4">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-secondary">person</span>
                  <h3 className="font-bold text-sm">Select Instructor</h3>
                </div>
                <select
                  value={selectedInstructorId}
                  onChange={(e) => setSelectedInstructorId(e.target.value)}
                  className="border border-outline-variant/30 rounded-lg px-3 py-2 text-sm font-label bg-surface w-full focus:outline-none focus:ring-1 focus:ring-secondary"
                >
                  <option value="">— Select an instructor —</option>
                  {instructorList.map((instructor) => {
                    const profileId = instructor.profile_id ?? instructor.instructor_profile_id ?? '';
                    return (
                      <option key={profileId || instructor.user_id} value={profileId}>
                        {instructor.full_name ?? instructor.username ?? profileId} ({instructor.department ?? 'No department'})
                      </option>
                    );
                  })}
                </select>
                {selectedInstructor ? (
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="rounded-lg bg-surface p-3 border border-outline-variant/10">
                      <div className="text-slate-400 uppercase tracking-widest">Profile ID</div>
                      <div className="font-bold mt-1 break-all">{selectedInstructor.profile_id ?? selectedInstructor.instructor_profile_id}</div>
                    </div>
                    <div className="rounded-lg bg-surface p-3 border border-outline-variant/10">
                      <div className="text-slate-400 uppercase tracking-widest">Students</div>
                      <div className="font-bold mt-1">{selectedInstructor.student_count ?? 0}</div>
                    </div>
                    <div className="rounded-lg bg-surface p-3 border border-outline-variant/10 col-span-2">
                      <div className="text-slate-400 uppercase tracking-widest">Instructor</div>
                      <div className="font-bold mt-1">{selectedInstructor.full_name ?? selectedInstructor.username ?? '—'} · {selectedInstructor.department ?? '—'}</div>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-slate-400">Choose an instructor to inspect roster context and existing instructor records.</p>
                )}
              </div>
            </div>
          </section>

          {/* Health Status */}
          <section className="space-y-3">
            <h2 className="font-headline font-bold text-lg">System Health</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[
                {
                  label: 'API',
                  ok: health?.status === 'ok',
                  detail: health ? `GBM ${health.gbm_loaded ? '✓' : '✗'}  LSTM ${health.lstm_loaded ? '✓' : '✗'}` : 'Loading…',
                  sub: health?.model_version,
                },
                {
                  label: 'MLOps',
                  ok: mlopsHealth?.status === 'ok',
                  detail: mlopsHealth ? `${mlopsHealth.prediction_count ?? 0} predictions` : 'Loading…',
                  sub: mlopsHealth?.drift_status,
                },
                {
                  label: 'Recommender',
                  ok: recHealth?.student_ranker?.loaded && recHealth?.instructor_ranker?.loaded,
                  detail: recHealth
                    ? `Student: ${recHealth.student_ranker?.loaded ? '✓' : '✗'}  Instructor: ${recHealth.instructor_ranker?.loaded ? '✓' : '✗'}`
                    : 'Loading…',
                  sub: recHealth?.student_ranker?.n_features ? `${recHealth.student_ranker.n_features} features` : null,
                },
              ].map(({ label, ok, detail, sub }) => (
                <div key={label} className="bg-surface-container-lowest rounded-xl p-5 border border-outline-variant/5 shadow-sm flex items-start gap-3">
                  <HealthDot ok={ok} />
                  <div>
                    <p className="font-bold text-sm">{label}</p>
                    <p className="text-xs text-slate-500 mt-0.5">{detail}</p>
                    {sub && <p className="text-[10px] text-slate-400 mt-0.5">{sub}</p>}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Model Metrics */}
          <section className="space-y-3">
            <h2 className="font-headline font-bold text-lg">Model Metrics</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="AUC-ROC" value={mlopsMetrics?.training?.auc?.toFixed(4)} icon="show_chart" />
              <StatCard label="F1 Score" value={mlopsMetrics?.training?.f1?.toFixed(4)} icon="balance" />
              <StatCard label="Brier Score" value={mlopsMetrics?.training?.brier?.toFixed(4)} icon="straighten" sub="lower is better" />
              <StatCard label="SHAP Fidelity" value={mlopsMetrics?.training?.shap_fidelity?.toFixed(4)} icon="psychology" />
            </div>
            {mlopsMetrics?.model_version && (
              <p className="text-xs text-slate-400">Model version: <span className="font-bold">{mlopsMetrics.model_version}</span></p>
            )}
          </section>

          {/* Drift Report */}
          <section className="space-y-3">
            <h2 className="font-headline font-bold text-lg">Drift Report</h2>
            <div className="bg-surface-container-lowest rounded-xl p-6 border border-outline-variant/5 shadow-sm">
              {!driftReport ? (
                <p className="text-sm text-slate-400">Loading drift report…</p>
              ) : driftReport.message ? (
                <p className="text-sm text-amber-600">{driftReport.message}</p>
              ) : (
                <div className="flex items-center gap-6">
                  <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Dataset Drift</p>
                    <span className={`text-lg font-bold ${driftReport.dataset_drift ? 'text-red-500' : 'text-green-600'}`}>
                      {driftReport.dataset_drift ? 'Detected' : 'None'}
                    </span>
                  </div>
                  <div className="w-px h-10 bg-outline-variant/20" />
                  <div>
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">Drifted Features</p>
                    <span className="text-lg font-bold">{driftReport.n_drifted_features ?? 0}</span>
                  </div>
                </div>
              )}
            </div>
          </section>

          {/* Causal DAG */}
          <section className="space-y-3">
            <h2 className="font-headline font-bold text-lg">Causal Feature Graph</h2>
            <GraphCard title="Causal DAG" height={420}>
              <Suspense fallback={
                <div className="flex items-center justify-center h-full text-slate-400 text-sm font-label animate-pulse">Loading graph…</div>
              }>
                <CausalDagGraph height={420} />
              </Suspense>
            </GraphCard>
          </section>

          {/* Fairness Monitor */}
          <section className="space-y-3">
            <h2 className="font-headline font-bold text-lg">Fairness Monitor</h2>
            <div className="bg-surface-container-lowest rounded-xl p-6 border border-outline-variant/5 shadow-sm">
              {!runFairness ? (
                <div className="flex items-center justify-between">
                  <p className="text-sm text-slate-500">Run a sample recommendation call to check fairness across protected groups.</p>
                  <button
                    onClick={() => setRunFairness(true)}
                    className="px-4 py-2 bg-primary text-white rounded-xl font-bold text-sm hover:bg-primary/90 transition-colors"
                  >
                    Run Fairness Check
                  </button>
                </div>
              ) : fairnessLoading ? (
                <div className="animate-pulse h-32 bg-surface-container rounded-xl" />
              ) : fairnessSample?.fairness_audit ? (
                <FairnessAuditPanel
                  report={fairnessSample.fairness_audit}
                  explanation={fairnessExplanation}
                  explaining={explainMutation.isPending}
                  onExplain={() => explainMutation.mutate(fairnessSample.fairness_audit)}
                />
              ) : (
                <p className="text-sm text-slate-400">No fairness audit data returned — backend may not support it yet.</p>
              )}
            </div>
          </section>

          {/* MLOps Controls */}
          <section className="space-y-3">
            <h2 className="font-headline font-bold text-lg">MLOps Controls</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

              {/* Retrain */}
              <div className="bg-surface-container-lowest rounded-xl p-6 border border-outline-variant/5 shadow-sm space-y-4">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-primary">model_training</span>
                  <h3 className="font-bold text-sm">Trigger Retrain</h3>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Runs the full retraining pipeline. Hard-stops if AUC drop &gt; 0.02 or Brier rise &gt; 0.03.
                </p>

                {retrain.data && (
                  <div className={`text-xs p-3 rounded-lg ${retrain.data.success ? 'bg-green-50 text-green-700' : 'bg-amber-50 text-amber-700'}`}>
                    {retrain.data.success ? '✓ Retrain complete' : `⚠ ${retrain.data.reason ?? 'Queued'}`}
                  </div>
                )}
                {retrain.error && (
                  <div className="text-xs p-3 rounded-lg bg-red-50 text-red-600">
                    {retrain.error.response?.data?.detail ?? 'Retrain failed'}
                  </div>
                )}

                <button
                  onClick={() => retrain.mutate()}
                  disabled={retrain.isPending}
                  className="w-full py-2.5 bg-primary text-white rounded-xl font-bold text-sm hover:bg-primary/90 transition-colors disabled:opacity-60"
                >
                  {retrain.isPending ? 'Retraining…' : 'Run Retrain'}
                </button>
              </div>

              {/* Reload */}
              <div className="bg-surface-container-lowest rounded-xl p-6 border border-outline-variant/5 shadow-sm space-y-4">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-secondary">restart_alt</span>
                  <h3 className="font-bold text-sm">Hot Reload Model</h3>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Loads the latest model artifacts into the live process without a container restart.
                </p>

                <div>
                  <label className="flex justify-between text-xs font-bold text-slate-500 mb-1">
                    <span>Canary Fraction</span>
                    <span>{(canaryFraction * 100).toFixed(0)}%</span>
                  </label>
                  <input
                    type="range" min={0.0} max={1.0} step={0.05}
                    value={canaryFraction}
                    onChange={(e) => setCanaryFraction(Number(e.target.value))}
                    className="w-full accent-primary"
                  />
                </div>

                {reload.data && (
                  <div className={`text-xs p-3 rounded-lg ${reload.data.success ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-600'}`}>
                    {reload.data.success ? `✓ Reloaded — ${reload.data.model_version}` : 'Reload failed'}
                  </div>
                )}
                {reload.error && (
                  <div className="text-xs p-3 rounded-lg bg-red-50 text-red-600">
                    {reload.error.response?.data?.detail ?? 'Reload failed'}
                  </div>
                )}

                <button
                  onClick={() => reload.mutate()}
                  disabled={reload.isPending}
                  className="w-full py-2.5 bg-secondary text-white rounded-xl font-bold text-sm hover:bg-secondary/90 transition-colors disabled:opacity-60"
                >
                  {reload.isPending ? 'Reloading…' : 'Hot Reload'}
                </button>
              </div>
            </div>
          </section>

          {/* Enrollment Management */}
          <section className="space-y-3">
            <h2 className="font-headline font-bold text-lg">Enrollment Management</h2>
            <div className="bg-surface-container-lowest rounded-xl border border-outline-variant/5 shadow-sm overflow-hidden">
              <div className="px-6 py-5 border-b border-outline-variant/10">
                <h3 className="font-bold text-sm mb-4">Add Enrollment</h3>
                <div className="flex flex-col sm:flex-row gap-3">
                  <input
                    type="text"
                    placeholder="Learner ID (e.g. learner_001)"
                    value={enrollForm.learner_id}
                    onChange={e => setEnrollForm(f => ({ ...f, learner_id: e.target.value }))}
                    className="border border-outline-variant/30 rounded-lg px-3 py-2 text-sm font-label bg-surface flex-1 focus:outline-none focus:ring-1 focus:ring-primary"
                  />
                  <select
                    value={enrollForm.instructor_profile_id}
                    onChange={e => setEnrollForm(f => ({ ...f, instructor_profile_id: e.target.value }))}
                    className="border border-outline-variant/30 rounded-lg px-3 py-2 text-sm font-label bg-surface w-full sm:w-56 focus:outline-none focus:ring-1 focus:ring-primary"
                  >
                    <option value="">— Select Instructor —</option>
                    {(instructors?.instructors ?? []).map(ins => (
                      <option key={ins.profile_id ?? ins.instructor_profile_id} value={ins.profile_id ?? ins.instructor_profile_id}>{ins.full_name ?? ins.username ?? ins.profile_id ?? ins.instructor_profile_id}</option>
                    ))}
                  </select>
                  <input
                    type="text"
                    placeholder="Course ID"
                    value={enrollForm.course_id}
                    onChange={e => setEnrollForm(f => ({ ...f, course_id: e.target.value }))}
                    className="border border-outline-variant/30 rounded-lg px-3 py-2 text-sm font-label bg-surface w-32 focus:outline-none focus:ring-1 focus:ring-primary"
                  />
                  <button
                    onClick={() => addEnrollMutation.mutate(enrollForm)}
                    disabled={addEnrollMutation.isPending || !enrollForm.learner_id || !enrollForm.instructor_profile_id}
                    className="px-4 py-2 bg-primary text-white rounded-lg font-bold text-sm hover:bg-primary/90 transition-colors disabled:opacity-50 flex items-center gap-1 shrink-0"
                  >
                    <span className="material-symbols-outlined text-sm">person_add</span> Enroll
                  </button>
                </div>
                {enrollError && <p className="mt-2 text-xs text-red-600 font-label">{enrollError}</p>}
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead className="bg-surface-container font-label text-[10px] uppercase tracking-widest text-slate-500">
                    <tr>
                      <th className="px-6 py-3">Learner ID</th>
                      <th className="px-6 py-3">Instructor</th>
                      <th className="px-6 py-3">Course</th>
                      <th className="px-6 py-3">Enrolled At</th>
                      <th className="px-6 py-3">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-outline-variant/5">
                    {(enrollments?.enrollments ?? []).map(e => (
                      <tr key={e.id} className="hover:bg-primary/5 transition-colors">
                        <td className="px-6 py-3 font-label text-sm font-bold">{e.learner_id}</td>
                        <td className="px-6 py-3 font-label text-sm text-slate-600">{e.instructor_name ?? e.instructor_username ?? e.instructor_profile_id}</td>
                        <td className="px-6 py-3 font-label text-sm text-slate-500">{e.course_id}</td>
                        <td className="px-6 py-3 font-label text-xs text-slate-400">{e.enrolled_at ? new Date(e.enrolled_at).toLocaleDateString() : '—'}</td>
                        <td className="px-6 py-3">
                          <button
                            onClick={() => removeEnrollMutation.mutate(e.id)}
                            disabled={removeEnrollMutation.isPending}
                            className="text-red-500 hover:text-red-700 transition-colors"
                          >
                            <span className="material-symbols-outlined text-sm">person_remove</span>
                          </button>
                        </td>
                      </tr>
                    ))}
                    {(enrollments?.enrollments ?? []).length === 0 && (
                      <tr><td colSpan={5} className="px-6 py-8 text-center text-slate-400 font-label text-sm">No enrollments found.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </section>

        </div>
      </main>
    </div>
  );
}
