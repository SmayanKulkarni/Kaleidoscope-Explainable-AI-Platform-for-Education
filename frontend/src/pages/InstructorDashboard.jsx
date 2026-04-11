import { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import { getInstructorData } from '../services/xaiService';

export default function InstructorDashboard() {
  const [data, setData] = useState(null);

  useEffect(() => { getInstructorData().then(setData); }, []);

  if (!data) return <div className="min-h-screen flex items-center justify-center bg-surface">Loading Dashboard...</div>;

  return (
    <div className="bg-surface font-body text-on-surface min-h-screen">
      <Navbar />
      <Sidebar />
      <main className="ml-64 mt-16 p-8 min-h-screen">
        <div className="max-w-7xl mx-auto">
          <header className="mb-10">
            <h1 className="text-4xl font-headline font-extrabold tracking-tight text-on-background mb-2">Class Performance Curator</h1>
            <p className="text-on-surface-variant font-body">Synthesizing predictive analytics across the current semester cohort.</p>
          </header>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-surface-container-lowest rounded-xl p-6 shadow-sm ring-1 ring-black/5 flex flex-col justify-between">
              <span className="font-label text-xs uppercase tracking-widest text-on-surface-variant mb-4">Risk Score (Aggregated)</span>
              <div className="flex items-end justify-between">
                <span className="text-3xl font-headline font-bold text-tertiary">{data.classStats.riskScore}%</span>
                <div className="h-10 w-20 bg-tertiary/10 rounded overflow-hidden">
                  <div className="h-full bg-tertiary" style={{width: `${data.classStats.riskScore}%`}}></div>
                </div>
              </div>
            </div>
            
            <div className="bg-surface-container-lowest rounded-xl p-6 shadow-sm ring-1 ring-black/5 flex flex-col justify-between">
              <span className="font-label text-xs uppercase tracking-widest text-on-surface-variant mb-4">Model Confidence (AUC-ROC)</span>
              <div className="flex items-end justify-between">
                <span className="text-3xl font-headline font-bold text-primary">{data.classStats.modelConfidence}</span>
                <span className="material-symbols-outlined text-secondary">verified_user</span>
              </div>
            </div>

            <div className="bg-surface-container-lowest rounded-xl p-6 shadow-sm ring-1 ring-black/5 flex flex-col justify-between">
              <span className="font-label text-xs uppercase tracking-widest text-on-surface-variant mb-4">Stability Score</span>
              <div className="flex items-end justify-between">
                <span className="text-3xl font-headline font-bold text-on-background">{data.classStats.stabilityScore}</span>
                <span className="material-symbols-outlined text-on-surface-variant">auto_graph</span>
              </div>
            </div>

            <div className="bg-surface-container-lowest rounded-xl p-6 shadow-sm ring-1 ring-black/5 flex flex-col justify-between">
              <span className="font-label text-xs uppercase tracking-widest text-on-surface-variant mb-4">Fidelity Score</span>
              <div className="flex items-end justify-between">
                <span className="text-3xl font-headline font-bold text-on-background">{data.classStats.fidelityScore}</span>
                <span className="material-symbols-outlined text-on-surface-variant">psychology</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-10 gap-8 mb-8">
            <div className="lg:col-span-6 bg-surface-container-low rounded-xl p-8 shadow-sm">
              <div className="flex justify-between items-center mb-8">
                <h3 className="font-headline font-bold text-xl text-on-background">SHAP Feature Attribution</h3>
                <div className="flex gap-4">
                  <div className="flex items-center gap-1 text-[10px] font-label text-on-surface-variant"><span className="material-symbols-outlined text-sm">link</span> Causal</div>
                  <div className="flex items-center gap-1 text-[10px] font-label text-on-surface-variant"><span className="material-symbols-outlined text-sm">waves</span> Correlated</div>
                </div>
              </div>
              <div className="space-y-4">
                {data.shapFeatures.map(f => (
                   <div className="grid grid-cols-12 items-center gap-4" key={f.name}>
                     <div className="col-span-3 font-label text-xs">{f.name}</div>
                     <div className={`col-span-8 h-4 bg-surface-container-highest rounded-full flex overflow-hidden ${f.direction==='negative'?'justify-end':''}`}>
                        <div className={`h-full ${f.direction === 'negative' ? 'bg-tertiary' : 'bg-primary'}`} style={{width: `${Math.abs(f.value)*100}%`}}></div>
                     </div>
                     <div className={`col-span-1 text-right font-label text-xs ${f.direction === 'negative' ? 'text-tertiary' : 'text-primary'}`}>
                        {f.value > 0 ? '+' : ''}{f.value} <span className="material-symbols-outlined text-xs align-sub">{f.type === 'causal' ? 'link' : 'waves'}</span>
                     </div>
                   </div>
                ))}
              </div>
            </div>

            <div className="lg:col-span-4 flex flex-col gap-6">
              <div className="bg-surface-container-highest rounded-xl p-6 border-l-4 border-primary">
                <div className="flex items-center gap-2 mb-3">
                  <span className="material-symbols-outlined text-primary">policy</span>
                  <h4 className="font-headline font-bold text-on-background">Anchor Rule</h4>
                </div>
                <p className="font-label text-sm leading-relaxed text-on-surface-variant">
                    {data.anchorRule}
                </p>
              </div>

              <div className="bg-surface-container-lowest rounded-xl p-6 shadow-sm flex-1">
                <h4 className="font-headline font-bold text-on-background text-sm mb-6">Trust Metrics Breakdown</h4>
                <div className="space-y-6">
                  <div>
                    <div className="flex justify-between text-xs font-label mb-2"><span>Fidelity</span><span className="font-bold">{data.trustMetrics.fidelity}%</span></div>
                    <div className="h-1.5 w-full bg-surface-container-high rounded-full overflow-hidden"><div className="bg-secondary h-full" style={{width: `${data.trustMetrics.fidelity}%`}}></div></div>
                  </div>
                  <div>
                    <div className="flex justify-between text-xs font-label mb-2"><span>Stability</span><span className="font-bold">{data.trustMetrics.stability}%</span></div>
                    <div className="h-1.5 w-full bg-surface-container-high rounded-full overflow-hidden"><div className="bg-secondary h-full" style={{width: `${data.trustMetrics.stability}%`}}></div></div>
                  </div>
                  <div>
                    <div className="flex justify-between text-xs font-label mb-2"><span>Completeness</span><span className="font-bold">{data.trustMetrics.completeness}%</span></div>
                    <div className="h-1.5 w-full bg-surface-container-high rounded-full overflow-hidden"><div className="bg-secondary h-full" style={{width: `${data.trustMetrics.completeness}%`}}></div></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-surface-container-lowest rounded-xl shadow-sm overflow-hidden ring-1 ring-black/5">
            <div className="px-8 py-6 border-b-0">
              <h3 className="font-headline font-bold text-xl text-on-background">Strategic Intervention Catalog</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead className="bg-surface-container-low font-label text-xs uppercase tracking-wider text-on-surface-variant">
                  <tr>
                    <th className="px-8 py-4">Feature</th>
                    <th className="px-8 py-4">Current Value</th>
                    <th className="px-8 py-4">Target Value</th>
                    <th className="px-8 py-4">Estimated Impact</th>
                    <th className="px-8 py-4">Factor Type</th>
                    <th className="px-8 py-4">Priority</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-surface-container-low">
                  {data.interventions.map((inv, idx) => (
                    <tr key={idx} className="hover:bg-blue-50/10 transition-colors">
                      <td className="px-8 py-5 font-semibold text-sm">{inv.feature}</td>
                      <td className="px-8 py-5 text-sm">{inv.current}</td>
                      <td className="px-8 py-5 text-sm">{inv.target}</td>
                      <td className="px-8 py-5 text-sm text-primary font-bold">{inv.impact} Success</td>
                      <td className="px-8 py-5"><span className="bg-secondary/10 text-secondary px-2 py-1 rounded text-[10px] font-label font-bold">{inv.type}</span></td>
                      <td className="px-8 py-5"><span className="bg-tertiary/10 text-tertiary px-2 py-1 rounded text-[10px] font-label font-bold">{inv.priority}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
