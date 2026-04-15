import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import AIHelpModal from '../components/AIHelpModal';
import { getStudentData } from '../services/xaiService';
import { useAuth } from '../context/AuthContext';
import { track } from '../services/eventTracker';
import { PAGE_VIEW, RECOMMENDATION_VIEWED } from '../constants/eventTypes';

export default function StudentDashboard() {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState('Prediction');
  const [aiModalOpen, setAiModalOpen] = useState(false);

  const learner_id = user?.learner_id ?? localStorage.getItem('ll_learner_id') ?? 'anonymous';

  useEffect(() => { track(PAGE_VIEW, { page: '/student' }); }, []);
  const { data, isLoading } = useQuery({
    queryKey: ['student-dashboard', learner_id],
    queryFn: () => getStudentData({ learner_id, audience: 'learner' }),
    enabled: !!learner_id,
    staleTime: 5 * 60 * 1000,
  });

  const explanation = data?.rawExplain;

  const buildFallbackItems = (items, emptyLabel) => {
    if (items.length > 0) return items;
    return [
      {
        name: emptyLabel,
        value: 1,
        direction: 'increases',
      },
    ];
  };

  const getTabData = () => {
    const fallbackSignals = (data?.shapFeatures ?? []).slice(0, 3).map((feature) => ({
      ...feature,
      type: 'fallback',
    }));

    const conceptFallbackSignals = buildFallbackItems(
      (data?.conceptAnalysis?.length ? data.conceptAnalysis : fallbackSignals),
      'No concept-level interaction data was returned for this learner yet.',
    );

    if (activeTab === 'Prediction') return data?.shapFeatures ?? [];
    if (activeTab === 'Feature') return (data?.featureContributions?.length ? data.featureContributions : fallbackSignals);
    return conceptFallbackSignals;
  };

  const getFallbackMessage = () => {
    if (activeTab === 'Feature' && !(data?.featureContributions?.length > 0)) {
      return 'No causal-annotation breakdown was returned for this learner, so the strongest prediction signals are shown here instead.';
    }
    if (activeTab === 'Concept' && !(data?.conceptAnalysis?.length > 0)) {
      return 'No interaction-level breakdown was returned for this learner, so the strongest available signals are shown here instead.';
    }
    return '';
  };

  if (isLoading || !data) return <div className="min-h-screen flex items-center justify-center bg-surface">Loading Dashboard...</div>;

  const renderTabContent = () => {
    const listData = getTabData();
    const maxMagnitude = Math.max(
      0,
      ...listData.map((item) => Math.abs(Number(item.value) || 0)),
    );

    return (
      <div className="relative pt-6">
        <div className="absolute left-1/2 top-0 bottom-0 w-[1px] bg-primary/20"></div>
        {getFallbackMessage() && (
          <div className="mb-5 rounded-xl border border-primary/10 bg-primary/5 px-4 py-3 text-sm text-primary leading-relaxed">
            {getFallbackMessage()}
          </div>
        )}
        {listData.map((f, index) => (
          <div key={f.name} className="relative flex items-center h-12 mb-6">
            {f.direction === 'increases' ? (
              <>
                <div className="w-1/2 pr-6 text-right"><span className="font-label text-xs text-slate-500">{f.name}</span></div>
                <div className="w-1/2 pl-0">
                   <div
                     className="h-6 bg-tertiary rounded-r-lg group relative overflow-hidden"
                     style={{ width: maxMagnitude > 0 ? `${Math.max(10, (Math.abs(Number(f.value) || 0) / maxMagnitude) * 100)}%` : `${Math.max(10, 100 - index * 15)}%` }}
                   ></div>
                </div>
              </>
            ) : (
              <>
                <div className="w-1/2 pr-0 flex justify-end">
                  <div
                    className="h-6 bg-secondary rounded-l-lg group relative overflow-hidden"
                    style={{ width: maxMagnitude > 0 ? `${Math.max(10, (Math.abs(Number(f.value) || 0) / maxMagnitude) * 100)}%` : `${Math.max(10, 100 - index * 15)}%` }}
                  ></div>
                </div>
                <div className="w-1/2 pl-6"><span className="font-label text-xs text-slate-500">{f.name}</span></div>
              </>
            )}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="bg-surface font-body text-on-surface">
      <Navbar />
      <Sidebar />
      <main className="md:ml-64 pt-24 pb-12 px-8 min-h-screen">
        <div className="max-w-7xl mx-auto space-y-8">
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-surface-container-lowest p-8 rounded-xl shadow-[0_12px_32px_rgba(7,30,39,0.04)] relative overflow-hidden group border border-outline-variant/5">
              <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                <span className="material-symbols-outlined text-6xl text-tertiary">trending_up</span>
              </div>
              <p className="font-label text-xs text-slate-500 uppercase tracking-widest mb-1">Dropout Risk Score</p>
              <div className="flex items-baseline gap-4 mb-2">
                <h2 className="font-headline font-extrabold text-5xl text-on-surface">{data.riskScore}%</h2>
                <span className="bg-tertiary-container/10 text-tertiary px-3 py-1 rounded-full text-xs font-bold font-label border border-tertiary/20">{data.riskLevel} Risk</span>
              </div>
              <p className="text-sm text-slate-500 leading-relaxed">Based on recent activity and grades</p>
            </div>

            <div className="bg-surface-container-lowest p-8 rounded-xl shadow-[0_12px_32px_rgba(7,30,39,0.04)] relative overflow-hidden group border border-outline-variant/5">
              <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                <span className="material-symbols-outlined text-6xl text-secondary">verified</span>
              </div>
              <p className="font-label text-xs text-slate-500 uppercase tracking-widest mb-1">Trust Score</p>
              <div className="flex items-baseline gap-4 mb-2">
                <h2 className="font-headline font-extrabold text-5xl text-on-surface">{data.trustScore}</h2>
                <span className="bg-secondary-container text-on-secondary-container px-3 py-1 rounded-full text-xs font-bold font-label">HIGH Confidence</span>
              </div>
              <p className="text-sm text-slate-500 leading-relaxed">Model stability and fidelity</p>
            </div>

            <div className="bg-gradient-to-br from-primary to-primary-container p-8 rounded-xl shadow-[0_12px_32px_rgba(7,30,39,0.08)] text-white relative group">
              <div className="absolute top-4 right-4">
                <span className="material-symbols-outlined text-white/30 text-4xl">assignment</span>
              </div>
              <p className="font-label text-xs text-white/70 uppercase tracking-widest mb-1">Top Recommended Action</p>
              <h3 className="font-headline font-bold text-xl mb-4">{data.topAction.label}</h3>
              <div className="flex items-center gap-2 bg-white/10 backdrop-blur-md w-fit px-3 py-1.5 rounded-lg border border-white/20">
                <span className="material-symbols-outlined text-secondary-fixed text-sm">expand_circle_down</span>
                <span className="text-xs font-bold font-label">-{data.topAction.riskReduction}% Risk Reduction</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
            <div className="lg:col-span-3 space-y-6">
              <div className="bg-surface-container-low p-8 rounded-xl min-h-[500px] flex flex-col border border-outline-variant/5">
                <h3 className="font-headline font-bold text-2xl mb-8">Why You're At Risk</h3>
                <div className="flex gap-8 border-b border-primary/5 mb-8">
                  <button onClick={() => setActiveTab('Prediction')} className={`pb-4 text-sm font-label transition-colors ${activeTab === 'Prediction' ? 'font-bold border-b-2 border-primary text-primary' : 'font-medium text-slate-400 hover:text-primary'}`}>Prediction</button>
                  <button onClick={() => setActiveTab('Feature')} className={`pb-4 text-sm font-label transition-colors ${activeTab === 'Feature' ? 'font-bold border-b-2 border-primary text-primary' : 'font-medium text-slate-400 hover:text-primary'}`}>Feature</button>
                  <button onClick={() => setActiveTab('Concept')} className={`pb-4 text-sm font-label transition-colors ${activeTab === 'Concept' ? 'font-bold border-b-2 border-primary text-primary' : 'font-medium text-slate-400 hover:text-primary'}`}>Concept</button>
                </div>
                
                <div className="space-y-10 flex-1">
                  {renderTabContent()}
                </div>
              </div>
            </div>

            <div className="lg:col-span-2 space-y-6">
              <div className="bg-surface-container-low p-8 rounded-xl h-full flex flex-col border border-outline-variant/5">
                <h3 className="font-headline font-bold text-2xl mb-6">Students Like You</h3>
                <div className="space-y-4 flex-1">
                  {data.lookalikes.map((lk, i) => (
                    <div key={i} className="bg-surface-container-lowest p-4 rounded-xl flex items-center justify-between border border-primary/5 hover:shadow-md transition-shadow">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-slate-100 flex items-center justify-center text-primary">
                          <span className="material-symbols-outlined">person</span>
                        </div>
                        <div>
                          <h4 className="font-label text-sm font-bold">Learner #{lk.id}</h4>
                          <p className="text-[10px] text-slate-400 font-label">{lk.match}% PATTERN MATCH</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <span className={`text-[10px] uppercase font-bold px-2 py-1 rounded ${lk.outcome === 'DROPPED OUT' ? 'text-tertiary bg-tertiary/5' : 'text-secondary bg-secondary/5'}`}>
                           {lk.outcome}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-8 p-4 bg-primary/5 rounded-xl border-t border-primary/10">
                  <p className="text-sm text-primary leading-relaxed flex gap-3">
                    <span className="material-symbols-outlined shrink-0 text-primary">info</span>
                    <span>Students with similar patterns often benefit from early intervention. Reach out to a tutor today.</span>
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="w-full">
            <div className="bg-surface-container-highest/50 backdrop-blur-md p-10 rounded-2xl border border-white/50 shadow-lg relative overflow-hidden">
              <div className="absolute -right-20 -bottom-20 w-64 h-64 bg-primary/5 rounded-full blur-3xl"></div>
              <div className="relative z-10 flex flex-col md:flex-row items-start gap-8">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-primary to-secondary flex items-center justify-center shrink-0 shadow-inner">
                  <span className="material-symbols-outlined text-white text-3xl" style={{fontVariationSettings: "'FILL' 1"}}>smart_toy</span>
                </div>
                <div className="flex-1 space-y-4">
                  <div className="flex items-center gap-3">
                    <h3 className="font-headline font-extrabold text-2xl">Your AI Coach Says</h3>
                    <span className="text-[10px] font-label bg-primary/10 px-2 py-0.5 rounded text-primary border border-primary/20 uppercase tracking-tighter">XAI Powered</span>
                  </div>
                  <p className="text-lg text-on-surface/80 leading-relaxed font-body italic">
                    "{data.aiNarrative}"
                  </p>
                  <div className="flex gap-4 pt-2">
                    <button
                      onClick={() => setAiModalOpen(true)}
                      className="text-sm font-bold text-primary flex items-center gap-2 group hover:underline"
                    >
                      Ask for study tips <span className="material-symbols-outlined text-sm group-hover:translate-x-1 transition-transform">arrow_forward</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

        </div>
      </main>
      <AIHelpModal
        isOpen={aiModalOpen}
        onClose={() => setAiModalOpen(false)}
        explanation={explanation}
      />
    </div>
  );
}
