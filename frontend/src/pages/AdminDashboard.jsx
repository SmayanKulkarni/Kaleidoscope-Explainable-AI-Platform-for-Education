import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import { getHealth, getMlopsHealth, getMlopsMetrics, getMlopsDriftReport, triggerRetrain, triggerReload } from '../api/dropout';
import { recommendHealth } from '../api/recommend';

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

  const { data: health } = useQuery({ queryKey: ['health'], queryFn: getHealth, refetchInterval: 30_000 });
  const { data: mlopsHealth } = useQuery({ queryKey: ['mlops-health'], queryFn: getMlopsHealth, refetchInterval: 30_000 });
  const { data: mlopsMetrics } = useQuery({ queryKey: ['mlops-metrics'], queryFn: getMlopsMetrics });
  const { data: driftReport } = useQuery({ queryKey: ['drift-report'], queryFn: getMlopsDriftReport });
  const { data: recHealth } = useQuery({ queryKey: ['recommend-health'], queryFn: recommendHealth });

  const retrain = useMutation({ mutationFn: triggerRetrain });
  const reload = useMutation({ mutationFn: () => triggerReload(canaryFraction) });

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

        </div>
      </main>
    </div>
  );
}
