import { lazy } from 'react';
import { useAuth } from '../context/AuthContext';
import { useCompare } from '../hooks/useCompare';
import { DEFAULT_FEATURES } from '../api/dropout';
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
  const learner_id = user?.learner_id ?? user?.id ?? 'anonymous';

  const { data: compareResult, isLoading, error } =
    useCompare({ features: DEFAULT_FEATURES, learner_id });

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
          >
            <ModelCompareGraph compareResult={compareResult} height={480} />
          </GraphCard>

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
