import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import {
  getInstructorStudents,
  getInstructorRecoForStudent,
  explainInstructorRecoForStudent,
} from '../api/recommend';
import { track } from '../services/eventTracker';
import { PAGE_VIEW, INSTRUCTOR_STUDENT_SELECTED, INSTRUCTOR_RECO_VIEWED } from '../constants/eventTypes';
const RISK_BADGE = {
  High:    'bg-red-100 text-red-700',
  Medium:  'bg-amber-100 text-amber-700',
  Low:     'bg-green-100 text-green-700',
  Unknown: 'bg-slate-100 text-slate-500',
};

export default function InstructorView() {
  const location = useLocation();

  useEffect(() => { track(PAGE_VIEW, { page: '/xai/instructor' }); }, []);

  const { data: roster, isLoading: rosterLoading } = useQuery({
    queryKey: ['instructor-students'],
    queryFn: getInstructorStudents,
    staleTime: 60_000,
  });

  const students = roster?.students ?? [];
  const preselect = location.state?.learner_id ?? null;

  const [selectedId, setSelectedId] = useState(preselect);
  const [recoData, setRecoData]     = useState(null);
  const [explainData, setExplainData] = useState(null);
  const [selectedReco, setSelectedReco] = useState(null);

  useEffect(() => {
    if (preselect && !selectedId) setSelectedId(preselect);
  }, [preselect]);

  const recoMutation = useMutation({
    mutationFn: (id) => getInstructorRecoForStudent(id),
    onSuccess: (data) => { setRecoData(data); setExplainData(null); setSelectedReco(null); },
  });

  const explainMutation = useMutation({
    mutationFn: ({ learner_id, features, item_id }) =>
      explainInstructorRecoForStudent(learner_id, features, item_id),
    onSuccess: setExplainData,
  });

  const handleStudentSelect = (id) => {
    setSelectedId(id);
    setRecoData(null);
    setExplainData(null);
    setSelectedReco(null);
    recoMutation.mutate(id);
  };

  const handleRecoSelect = (reco) => {
    setSelectedReco(reco);
    explainMutation.mutate({
      learner_id: selectedId,
      features: reco.features ?? {},
      item_id: reco.item_id,
    });
  };

  const selectedStudent = students.find(s => s.learner_id === selectedId);
  const recommendations = recoData?.recommendations ?? [];

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
            <h1 className="text-3xl font-headline font-extrabold">Student Intervention Planner</h1>
            <p className="text-slate-500 text-sm mt-1">Select a student to see ranked intervention recommendations from the instructor model.</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1">
              <h2 className="font-headline font-bold text-sm uppercase tracking-widest text-slate-500 mb-3">Your Students</h2>
              {rosterLoading ? (
                <div className="space-y-2">
                  {[1,2,3,4].map(i => <div key={i} className="h-16 bg-surface-container-low rounded-xl animate-pulse" />)}
                </div>
              ) : (
                <div className="space-y-2 max-h-[70vh] overflow-y-auto pr-1">
                  {students.map(s => {
                    const risk  = s.dropout_risk_score ?? 0;
                    const label = s.risk_label ?? 'Unknown';
                    const active = s.learner_id === selectedId;
                    return (
                      <button
                        key={s.learner_id}
                        onClick={() => handleStudentSelect(s.learner_id)}
                        className={`w-full text-left p-4 rounded-xl border transition-all ${active ? 'border-primary bg-primary/5 shadow-sm' : 'border-outline-variant/10 bg-surface-container-low hover:bg-primary/5'}`}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-label font-bold text-sm">{s.display_name}</span>
                          <span className={`text-[10px] font-label font-bold px-1.5 py-0.5 rounded ${RISK_BADGE[label] ?? RISK_BADGE.Unknown}`}>{label}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-1.5 bg-surface-container-high rounded-full overflow-hidden">
                            <div
                              className={`h-full ${label === 'High' ? 'bg-red-500' : label === 'Medium' ? 'bg-amber-500' : 'bg-green-500'}`}
                              style={{ width: `${risk * 100}%` }}
                            />
                          </div>
                          <span className="text-[10px] font-label text-slate-400">{(risk * 100).toFixed(0)}%</span>
                        </div>
                      </button>
                    );
                  })}
                  {students.length === 0 && (
                    <div className="text-center py-10 text-slate-400 font-label text-sm">No students enrolled.</div>
                  )}
                </div>
              )}
            </div>

            <div className="lg:col-span-2 space-y-6">
              {!selectedId && (
                <div className="flex items-center justify-center h-64 bg-surface-container-low rounded-xl border border-outline-variant/10 text-slate-400 font-label text-sm">
                  <div className="text-center">
                    <span className="material-symbols-outlined text-3xl block mb-2">person_search</span>
                    Select a student to view recommendations
                  </div>
                </div>
              )}

              {selectedStudent && (
                <div className="bg-surface-container-low rounded-xl p-5 border border-outline-variant/10 flex items-center gap-4">
                  <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
                    <span className="material-symbols-outlined text-primary">person</span>
                  </div>
                  <div>
                    <div className="font-headline font-bold text-lg">{selectedStudent.display_name}</div>
                    <div className="text-xs font-label text-slate-500">{selectedStudent.learner_id} · Module: {selectedStudent.current_module ?? '—'} · Week {selectedStudent.features?.current_week_in_course ?? '?'}</div>
                  </div>
                  <div className="ml-auto text-right">
                    <div className="font-headline font-bold text-2xl text-on-surface">{((selectedStudent.dropout_risk_score ?? 0) * 100).toFixed(0)}%</div>
                    <div className="text-xs text-slate-400 font-label">dropout risk</div>
                  </div>
                </div>
              )}

              {recoMutation.isPending && (
                <div className="flex items-center gap-3 py-8 justify-center text-slate-400 font-label">
                  <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                  Generating recommendations...
                </div>
              )}

              {recommendations.length > 0 && (
                <div>
                  <h2 className="font-headline font-bold text-base mb-3">Ranked Interventions</h2>
                  <div className="space-y-3">
                    {recommendations.map((reco, i) => {
                      const active = selectedReco?.item_id === reco.item_id;
                      return (
                        <div
                          key={reco.item_id}
                          className={`rounded-xl border p-4 cursor-pointer transition-all ${active ? 'border-primary bg-primary/5' : 'border-outline-variant/10 bg-surface-container-low hover:border-primary/30'}`}
                          onClick={() => handleRecoSelect(reco)}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-3">
                              <span className="w-6 h-6 rounded-full bg-primary text-on-primary flex items-center justify-center text-xs font-bold shrink-0">{i + 1}</span>
                              <span className="font-label font-bold text-sm">{reco.item_id}</span>
                            </div>
                            <span className="font-headline font-bold text-primary text-sm">{reco.score?.toFixed(3)}</span>
                          </div>
                          {reco.shap_values && (
                            <div className="mt-3 flex flex-wrap gap-2">
                              {Object.entries(reco.shap_values).slice(0, 3).map(([k, v]) => (
                                <span key={k} className={`text-[10px] font-label px-2 py-0.5 rounded ${v > 0 ? 'bg-primary/10 text-primary' : 'bg-red-100 text-red-600'}`}>
                                  {k}: {v > 0 ? '+' : ''}{(+v).toFixed(3)}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {selectedReco && (
                <div className="bg-surface-container-low rounded-xl border border-outline-variant/10 p-5 space-y-4">
                  <h3 className="font-headline font-bold text-base flex items-center gap-2">
                    <span className="material-symbols-outlined text-secondary text-base">psychology</span>
                    Explanation: {selectedReco.item_id}
                  </h3>
                  {explainMutation.isPending && (
                    <div className="flex items-center gap-2 text-slate-400 font-label text-sm">
                      <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                      Loading explanation...
                    </div>
                  )}
                  {explainData && (
                    <>
                      {explainData.plain_language && (
                        <p className="text-sm italic text-on-surface/80 bg-primary/5 rounded-lg px-4 py-3">"{explainData.plain_language}"</p>
                      )}
                      {explainData.anchor_rule?.conditions?.length > 0 && (
                        <div className="bg-secondary/5 rounded-lg p-4 border border-secondary/10">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="material-symbols-outlined text-secondary text-sm">policy</span>
                            <span className="font-label font-bold text-sm text-secondary">Anchor Rule</span>
                            <span className="ml-auto text-[10px] font-label text-slate-400">precision {((explainData.anchor_precision ?? 0) * 100).toFixed(0)}%</span>
                          </div>
                          <ul className="space-y-1">
                            {explainData.anchor_rule.conditions.map((c, i) => (
                              <li key={i} className="text-xs font-label text-on-surface/70 flex items-start gap-2">
                                <span className="text-secondary shrink-0">IF</span> {c}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {explainData.narratives?.instructor && (
                        <div className="bg-surface-container-lowest rounded-lg p-4 border border-outline-variant/10">
                          <span className="font-label text-xs uppercase tracking-widest text-slate-400 block mb-2">LLM Narrative</span>
                          <p className="text-sm text-on-surface/80 leading-relaxed">{explainData.narratives.instructor}</p>
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
