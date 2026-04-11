import { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import { getInstructorData, simulateWhatIf } from '../services/xaiService';
import { useAuth } from '../context/AuthContext';

export default function WhatIfExplorer() {
  const { user } = useAuth();
  const [data, setData] = useState(null);
  const [sliders, setSliders] = useState({});
  const [simulationParams, setSimulationParams] = useState(null);

  useEffect(() => {
    getInstructorData().then(d => {
      setData(d);
      const initialSliders = {
        'Assignments Completed': 8,
        'Days Since Last Login': 12,
        'Forum Posts': 2
      };
      setSliders(initialSliders);
      simulateWhatIf(initialSliders).then(setSimulationParams);
    });
  }, []);

  const handleSliderChange = async (name, value) => {
    const newSliders = { ...sliders, [name]: parseInt(value, 10) };
    setSliders(newSliders);
    const newSim = await simulateWhatIf(newSliders);
    setSimulationParams(newSim);
  };

  if (!data || !simulationParams) return <div className="min-h-screen bg-surface flex justify-center items-center">Loading Explorer...</div>;

  const baselineRisk = data.classStats.riskScore;
  const riskDelta = Math.round(simulationParams.newRisk - baselineRisk);

  return (
    <div className="bg-surface font-body text-on-surface min-h-screen">
      <Navbar />
      <Sidebar />
      <main className="md:ml-64 pt-24 pb-12 px-8 min-h-screen">
        <div className="max-w-7xl mx-auto">
          
          <div className="flex justify-between items-end mb-10">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <span className="material-symbols-outlined text-primary bg-primary/10 p-2 rounded-lg">science</span>
                <span className="font-label text-sm uppercase tracking-widest text-primary font-bold">Simulator</span>
              </div>
              <h1 className="text-4xl font-headline font-extrabold tracking-tight text-on-surface">What-If Explorer</h1>
              <p className="text-on-surface-variant font-body mt-2">Adjust feature values to observe predicted changes in risk.</p>
            </div>
            <div className="text-right">
              <span className="font-label text-xs uppercase tracking-widest text-slate-500 block mb-1">Target Learner</span>
              <span className="bg-surface-container-low px-4 py-2 rounded-lg font-bold text-sm border border-outline/10">ID: STU-8492</span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            <div className="lg:col-span-5 space-y-6">
              <div className="bg-surface-container-low rounded-2xl p-8 shadow-sm border border-outline-variant/10">
                <h3 className="font-headline font-bold text-xl mb-6">Feature Interventions</h3>
                <div className="space-y-8">
                  {Object.entries(sliders).map(([name, val]) => {
                    const max = name === 'Days Since Last Login' ? 30 : 20;
                    return (
                      <div key={name}>
                        <div className="flex justify-between mb-3 items-end">
                          <label className="font-label text-sm font-medium">{name}</label>
                          <span className="font-mono bg-on-surface/5 px-2 py-1 rounded text-xs">{val}</span>
                        </div>
                        <input 
                          type="range" 
                          min="0" max={max} 
                          value={val} 
                          onChange={(e) => handleSliderChange(name, e.target.value)}
                          className="w-full h-2 bg-surface-container-highest rounded-lg appearance-none cursor-pointer accent-primary hover:accent-primary-container transition-all" 
                        />
                      </div>
                    )
                  })}
                </div>
                <div className="mt-8 pt-6 border-t border-outline-variant/10">
                  <button onClick={() => {
                        const initialSliders = {'Assignments Completed': 8, 'Days Since Last Login': 12, 'Forum Posts': 2};
                        setSliders(initialSliders);
                        simulateWhatIf(initialSliders).then(setSimulationParams);
                    }} 
                    className="w-full py-3 bg-surface-container hover:bg-surface-container-highest rounded-xl text-sm font-bold transition-colors">
                     Reset to Baseline
                  </button>
                </div>
              </div>
            </div>

            <div className="lg:col-span-7 space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div className="bg-surface-container-lowest rounded-2xl p-8 shadow-sm border border-outline-variant/10 text-center relative overflow-hidden group">
                  <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                  <span className="font-label text-xs uppercase tracking-widest text-slate-500 mb-2 block">Projected Risk</span>
                  <div className="text-6xl font-headline font-black text-on-surface mb-2">{simulationParams.newRisk}%</div>
                  <div className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-bold ${riskDelta < 0 ? 'bg-primary/10 text-primary' : riskDelta > 0 ? 'bg-tertiary/10 text-tertiary' : 'bg-surface-container text-slate-500'}`}>
                    <span className="material-symbols-outlined text-[14px]">
                      {riskDelta < 0 ? 'arrow_downward' : riskDelta > 0 ? 'arrow_upward' : 'horizontal_rule'}
                    </span>
                    {Math.abs(riskDelta)}% {riskDelta === 0 ? 'Change' : riskDelta > 0 ? 'Increase' : 'Decrease'}
                  </div>
                </div>

                <div className="bg-surface-container-lowest rounded-2xl p-8 shadow-sm border border-outline-variant/10 flex flex-col justify-center">
                  <span className="font-label text-xs uppercase tracking-widest text-slate-500 mb-4">Model Confidence</span>
                  <div className="flex items-center gap-4">
                    <span className="text-4xl font-headline font-black text-secondary">0.94</span>
                    <div className="flex-1 space-y-1">
                      <div className="flex justify-between text-[10px] uppercase font-label"><span>Fidelity</span><span>High</span></div>
                      <div className="h-1 bg-surface-container-highest rounded-full"><div className="h-full bg-secondary w-11/12 rounded-full"></div></div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-surface-container-low rounded-2xl p-8 shadow-sm border border-outline-variant/10">
                <h3 className="font-headline font-bold text-xl mb-6">Impact Trajectory</h3>
                <div className="space-y-4">
                  {simulationParams.impactFactors.map(factor => (
                    <div key={factor.name} className="flex items-center gap-4 group">
                      <div className="w-1/3 text-right">
                        <span className="font-label text-xs font-medium text-slate-400 group-hover:text-on-surface transition-colors">{factor.name}</span>
                      </div>
                      <div className="w-2/3 h-8 bg-surface-container-highest rounded-lg flex items-center relative overflow-hidden">
                        <div className={`h-full ${factor.impact > 0 ? 'bg-tertiary/80 right-1/2 origin-right' : 'bg-primary/80 left-1/2 origin-left'} absolute`} 
                             style={{width: `${Math.abs(factor.impact) * 20}%`}}></div>
                        <div className="absolute w-[2px] h-full bg-on-surface/20 left-1/2 z-10"></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
