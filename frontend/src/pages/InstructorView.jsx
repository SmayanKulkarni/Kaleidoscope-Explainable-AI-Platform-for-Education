import { lazy, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useRecommendInstructor } from '../hooks/useRecommend';
import { useMutation } from '@tanstack/react-query';
import { recommendInstructorExplain } from '../api/recommend';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import RecommendationList from '../components/panels/RecommendationList';
import AnchorRuleCard from '../components/panels/AnchorRuleCard';
import TrustScoreCard from '../components/panels/TrustScoreCard';
import NarrativeCard from '../components/panels/NarrativeCard';
import InterventionMetaCard from '../components/panels/InterventionMetaCard';
import FairnessAuditPanel from '../components/panels/FairnessAuditPanel';
import InstructorStudentCard from '../components/panels/InstructorStudentCard';
import WhatIfForm from '../components/panels/WhatIfForm';
import GraphCard from '../components/layout/GraphCard';
import { CAUSAL_COLORS } from '../lib/colors';

const ShapInteractionGraph = lazy(() => import('../components/graphs/ShapInteractionGraph'));

const INTERVENTION_ITEMS = [
  { item_id: 'Office Hours Nudge',      features: { urgency: 0.8, effort: 0.2, cohort_size: 0.6 } },
  { item_id: 'Forum Participation Push', features: { urgency: 0.5, effort: 0.3, cohort_size: 0.4 } },
  { item_id: 'Assignment Deadline Alert',features: { urgency: 0.9, effort: 0.1, cohort_size: 0.8 } },
  { item_id: 'Peer Study Group',         features: { urgency: 0.4, effort: 0.5, cohort_size: 0.3 } },
  { item_id: 'Video Re-watch Prompt',    features: { urgency: 0.3, effort: 0.2, cohort_size: 0.5 } },
];

const CAUSAL_LEGEND = [
  { label: 'Causal',        color: CAUSAL_COLORS.causal        },
  { label: 'Correlational', color: CAUSAL_COLORS.correlational },
  { label: 'Unknown',       color: CAUSAL_COLORS.unknown       },
];

export default function InstructorView() {
  const { user } = useAuth();
  const instructor_id = user?.id ?? 'anonymous';

  const { data: recoData, isLoading: recoLoading, error: recoError } =
    useRecommendInstructor({ instructor_id, items: INTERVENTION_ITEMS, top_k: 5 });

  const [selectedReco, setSelectedReco] = useState(null);
  const [explainData, setExplainData]   = useState(null);

  const explainMutation = useMutation({
    mutationFn: (item) =>
      recommendInstructorExplain(instructor_id, item.features ?? {}, item.item_id),
    onSuccess: setExplainData,
  });

  const handleSelect = (item) => {
    setSelectedReco(item);
    explainMutation.mutate(item);
  };

  const recommendations = recoData?.recommendations ?? [];
  const useCards = recommendations.some((r) => r.features?.student_dropout_risk_score != null);

  return (
    <div className="bg-surface font-body text-on-surface min-h-screen">
      <Navbar />
      <Sidebar />
      <main className="md:ml-64 pt-20 pb-12 px-6 min-h-screen">
        <div className="max-w-7xl mx-auto">

          <div className="mb-8">
            <div className="flex items-center gap-2 mb-1">
              <span className="material-symbols-outlined text-secondary text-lg">manage_accounts</span>
              <span className="font-label text-xs uppercase tracking-widest text-secondary font-bold">Instructor XAI</span>
            </div>
            <h1 className="text-3xl font-headline font-extrabold">Intervention Recommendations</h1>
            {recoError && (
              <div className="mt-3 px-4 py-2 rounded-lg bg-error/10 text-error text-sm font-label border border-error/20">
                {recoError.response?.data?.detail ?? recoError.message}
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div>
              <h2 className="font-headline font-bold text-lg mb-3">Ranked Interventions</h2>
              {useCards ? (
                <div className="space-y-3">
                  {recoLoading
                    ? [1, 2, 3].map((i) => (
                        <div key={i} className="bg-surface-container-lowest rounded-xl p-4 border border-outline-variant/10 animate-pulse h-28" />
                      ))
                    : recommendations.map((item) => (
                        <InstructorStudentCard
                          key={item.item_id}
                          item={item}
                          onExplain={() => handleSelect(item)}
                          onWhatIf={() => setSelectedReco(item)}
                        />
                      ))}
                </div>
              ) : (
                <RecommendationList
                  recommendations={recommendations}
                  onSelect={handleSelect}
                  loading={recoLoading}
                  diversityScore={recoData?.diversity_score}
                  diversityWarning={recoData?.diversity_warning}
                  fairnessAudit={recoData?.fairness_audit}
                />
              )}
              {!useCards && recoData?.fairness_audit && (
                <div className="mt-3">
                  <FairnessAuditPanel report={recoData.fairness_audit} />
                </div>
              )}
            </div>

            <div className="space-y-4">
              {selectedReco ? (
                <>
                  <div className="bg-surface-container-lowest rounded-2xl p-5 border border-outline-variant/10">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="material-symbols-outlined text-secondary text-base">info</span>
                      <h3 className="font-headline font-bold text-sm">{selectedReco.item_id}</h3>
                    </div>
                    <p className="text-sm text-slate-500 font-label">
                      Score: <span className="font-bold text-primary">{selectedReco.score?.toFixed(3)}</span>
                    </p>
                    {explainData?.plain_language && (
                      <p className="mt-3 text-sm italic text-on-surface/70">"{explainData.plain_language}"</p>
                    )}
                  </div>
                  {explainData?.anchor_rule && (
                    <AnchorRuleCard
                      anchorRule={explainData.anchor_rule}
                      precision={explainData.anchor_precision}
                    />
                  )}
                  <InterventionMetaCard
                    interventionType={explainData?.intervention_type}
                    interventionUrgency={explainData?.intervention_urgency}
                    contentType={explainData?.recommended_content_type}
                    effortHours={explainData?.estimated_effort_hours}
                    studentRiskScore={explainData?.student_dropout_risk_score}
                    studentTrajectory={explainData?.student_risk_trajectory}
                    cohortDropoutRate={explainData?.cohort_avg_dropout_rate}
                    instructorArchetype={explainData?.instructor_archetype}
                    teachingStyle={explainData?.instructor_teaching_style}
                    loading={explainMutation.isPending}
                  />
                  <TrustScoreCard
                    trustScore={explainData?.trust_score}
                    loading={explainMutation.isPending}
                  />
                  <NarrativeCard
                    narratives={explainData?.narratives}
                    audience="instructor"
                    isLoading={explainMutation.isPending}
                  />
                  <WhatIfForm
                    baseFeatures={selectedReco.features ?? {}}
                    mode="recommend"
                    learner_id={instructor_id}
                    itemFeatures={selectedReco.features ?? {}}
                  />
                </>
              ) : (
                <div className="flex items-center justify-center h-48 bg-surface-container-lowest rounded-2xl border border-outline-variant/10 text-slate-400 text-sm font-label">
                  Select an intervention to see its explanation
                </div>
              )}
            </div>
          </div>

          <GraphCard
            title="Intervention Feature Network"
            legend={CAUSAL_LEGEND}
            height={420}
          >
            <ShapInteractionGraph
              explanation={explainData}
              height={420}
            />
          </GraphCard>

        </div>
      </main>
    </div>
  );
}
