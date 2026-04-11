import { lazy, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useExplain } from '../hooks/useExplain';
import { useRecommendStudent } from '../hooks/useRecommend';
import { DEFAULT_FEATURES } from '../api/dropout';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import RiskScoreCard from '../components/panels/RiskScoreCard';
import TopFeaturesBar from '../components/panels/TopFeaturesBar';
import AnchorRuleCard from '../components/panels/AnchorRuleCard';
import TrustScoreCard from '../components/panels/TrustScoreCard';
import WhatIfForm from '../components/panels/WhatIfForm';
import RecommendationList from '../components/panels/RecommendationList';
import GraphCard from '../components/layout/GraphCard';
import { CAUSAL_COLORS } from '../lib/colors';

const ShapInteractionGraph = lazy(() => import('../components/graphs/ShapInteractionGraph'));
const CausalDagGraph       = lazy(() => import('../components/graphs/CausalDagGraph'));
const RecommendPathGraph    = lazy(() => import('../components/graphs/RecommendPathGraph'));

const SAMPLE_ITEMS = [
  { item_id: 'Module 4 Quiz',      features: { difficulty: 0.6, time_required: 30 } },
  { item_id: 'TA Office Hours',    features: { difficulty: 0.2, time_required: 60 } },
  { item_id: 'Forum Week 6',       features: { difficulty: 0.1, time_required: 15 } },
  { item_id: 'Video Lecture 4.2',  features: { difficulty: 0.4, time_required: 25 } },
  { item_id: 'Practice Problems',  features: { difficulty: 0.7, time_required: 45 } },
];

const GRAPH_TABS = ['Interactions', 'Causal DAG'];
const CAUSAL_LEGEND = [
  { label: 'Causal',        color: CAUSAL_COLORS.causal        },
  { label: 'Correlational', color: CAUSAL_COLORS.correlational },
  { label: 'Unknown',       color: CAUSAL_COLORS.unknown       },
];

export default function StudentView() {
  const { user } = useAuth();
  const learner_id = user?.learner_id ?? user?.id ?? 'anonymous';

  const features = DEFAULT_FEATURES;

  const { data: explanation, isLoading: explainLoading, error: explainError } =
    useExplain({ features, learner_id, model: 'gbm', audience: 'learner' });

  const { data: recoData, isLoading: recoLoading } =
    useRecommendStudent({ learner_id, items: SAMPLE_ITEMS, top_k: 5 });

  const [graphTab, setGraphTab]         = useState('Interactions');
  const [selectedReco, setSelectedReco] = useState(null);

  const recommendations = recoData?.recommendations ?? [];

  return (
    <div className="bg-surface font-body text-on-surface min-h-screen">
      <Navbar />
      <Sidebar />
      <main className="md:ml-64 pt-20 pb-12 px-6 min-h-screen">
        <div className="max-w-7xl mx-auto">

          <div className="mb-8">
            <div className="flex items-center gap-2 mb-1">
              <span className="material-symbols-outlined text-primary text-lg">psychology</span>
              <span className="font-label text-xs uppercase tracking-widest text-primary font-bold">XAI Dashboard</span>
            </div>
            <h1 className="text-3xl font-headline font-extrabold">Student Risk Overview</h1>
            {explainError && (
              <div className="mt-3 px-4 py-2 rounded-lg bg-error/10 text-error text-sm font-label border border-error/20">
                {explainError.response?.data?.detail ?? explainError.message}
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div className="space-y-4">
              <RiskScoreCard
                riskScore={explanation?.risk_score}
                riskLabel={explanation?.risk_label}
                modelVersion={explanation?.model_version}
                loading={explainLoading}
              />
              <AnchorRuleCard
                anchorRule={explanation?.anchor_rule}
                precision={explanation?.anchor_rule?.precision}
                loading={explainLoading}
              />
              <TrustScoreCard trustScore={explanation?.trust_score} loading={explainLoading} />
            </div>

            <div className="space-y-4">
              <TopFeaturesBar
                shapValues={explanation?.shap_values}
                topFeatures={explanation?.top_features}
                loading={explainLoading}
              />
              <WhatIfForm
                baseFeatures={features}
                mode="dropout"
                learner_id={learner_id}
              />
            </div>
          </div>

          <div className="mb-6">
            <GraphCard
              title="Feature Explanation Graph"
              legend={CAUSAL_LEGEND}
              height={440}
            >
              <div className="absolute top-3 left-1/2 -translate-x-1/2 flex gap-1 z-10">
                {GRAPH_TABS.map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setGraphTab(tab)}
                    className={`px-3 py-1 rounded-full text-xs font-label font-bold transition-colors ${graphTab === tab ? 'bg-primary text-white' : 'bg-surface-container text-slate-500 hover:bg-surface-container-high'}`}
                  >
                    {tab}
                  </button>
                ))}
              </div>
              {graphTab === 'Interactions' ? (
                <ShapInteractionGraph explanation={explanation} height={440} />
              ) : (
                <CausalDagGraph explanation={explanation} height={440} />
              )}
            </GraphCard>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h2 className="font-headline font-bold text-lg mb-3">Recommendations</h2>
              <RecommendationList
                recommendations={recommendations}
                onSelect={setSelectedReco}
                loading={recoLoading}
              />
            </div>
            <div>
              <GraphCard title="Recommendation Path" height={340}>
                <RecommendPathGraph
                  recommendation={selectedReco}
                  learnerId={learner_id}
                  height={340}
                />
              </GraphCard>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
