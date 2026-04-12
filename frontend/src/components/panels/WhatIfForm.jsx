import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { whatif, MUTABLE_FEATURES } from '../../api/dropout';
import { recommendStudentWhatif } from '../../api/recommend';

export default function WhatIfForm({ baseFeatures, mode = 'dropout', learner_id, itemFeatures }) {
  const [overrides, setOverrides]   = useState({});
  const [result, setResult]         = useState(null);

  const mutation = useMutation({
    mutationFn: (ovr) => {
      if (mode === 'recommend') {
        return recommendStudentWhatif(learner_id, { ...itemFeatures, ...ovr }, ovr);
      }
      return whatif({ ...baseFeatures, ...ovr }, ovr);
    },
    onSuccess: setResult,
  });

  const handleSlider = (id, raw, step) => {
    const val = step >= 1 ? parseInt(raw, 10) : parseFloat(raw);
    const newOvr = { ...overrides, [id]: val };
    setOverrides(newOvr);
    mutation.mutate(newOvr);
  };

  const handleReset = () => {
    setOverrides({});
    setResult(null);
  };

  const baseRisk = result?.base_risk ?? result?.baseRisk ?? null;
  const newRisk = result?.new_risk ?? result?.risk_score ?? result?.modified_score ?? null;
  const delta = baseRisk != null && newRisk != null
    ? newRisk - baseRisk
    : (result?.risk_delta ?? result?.score_delta ?? null);

  return (
    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10">
      <div className="flex items-center justify-between mb-6">
        <span className="font-label text-xs uppercase tracking-widest text-slate-500">What-If Simulator</span>
        <button onClick={handleReset} className="text-[11px] font-label text-primary hover:underline">Reset</button>
      </div>

      <div className="space-y-6">
        {MUTABLE_FEATURES.map(({ id, label, min, max, step, unit }) => {
          const val = overrides[id] ?? baseFeatures?.[id] ?? min;
          return (
            <div key={id}>
              <div className="flex justify-between mb-2 items-end">
                <label className="font-label text-xs font-medium text-on-surface">{label}</label>
                <span className="font-mono text-xs bg-on-surface/5 px-2 py-0.5 rounded">{val} {unit}</span>
              </div>
              <input
                type="range"
                min={min} max={max} step={step}
                value={val}
                onChange={(e) => handleSlider(id, e.target.value, step)}
                className="w-full h-1.5 rounded-full accent-primary cursor-pointer"
              />
            </div>
          );
        })}
      </div>

      {result && (
        <div className="mt-6 pt-5 border-t border-outline-variant/10 flex items-center gap-4">
          {newRisk != null && (
            <div className="text-center">
              <p className="text-[10px] font-label text-slate-500 uppercase tracking-wide">New Risk</p>
              <p className="text-2xl font-headline font-black text-on-surface">{Math.round(newRisk * 100)}%</p>
            </div>
          )}
          {delta != null && (
            <div className={`flex-1 text-center px-4 py-2 rounded-xl text-sm font-bold font-label ${delta < 0 ? 'bg-green-50 text-green-700' : delta > 0 ? 'bg-red-50 text-red-700' : 'bg-slate-100 text-slate-500'}`}>
              <span className="material-symbols-outlined text-sm align-sub mr-1">
                {delta < 0 ? 'arrow_downward' : delta > 0 ? 'arrow_upward' : 'horizontal_rule'}
              </span>
              {Math.abs(delta * 100).toFixed(2)}% {delta === 0 ? 'no change' : delta < 0 ? 'decrease' : 'increase'}
            </div>
          )}
          {mutation.isPending && (
            <span className="text-xs text-slate-400 font-label animate-pulse">Recalculating…</span>
          )}
        </div>
      )}
    </div>
  );
}
