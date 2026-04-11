function RangeBar({ current, q10, q50, q90, max }) {
  const scale = (v) => Math.min(100, Math.max(0, (v / (max || 1)) * 100));

  return (
    <div className="relative h-4 w-full bg-surface-container rounded-full overflow-hidden mt-1">
      <div
        className="absolute h-full bg-green-200 rounded-full"
        style={{ left: `${scale(q10)}%`, width: `${scale(q90) - scale(q10)}%` }}
      />
      <div
        className="absolute h-full w-1 bg-amber-400 rounded-full"
        style={{ left: `${scale(q50)}%` }}
      />
      <div
        className="absolute h-full w-1.5 bg-primary rounded-full z-10"
        style={{ left: `${scale(current)}%` }}
        title={`Current: ${current.toFixed(2)}`}
      />
    </div>
  );
}

export default function FeatureProjectionGrid({ featureDistributions, currentFeatures, topFeatures }) {
  if (!featureDistributions) return null;

  const entries = Object.entries(featureDistributions);
  const sorted = topFeatures
    ? [
        ...topFeatures.filter((f) => featureDistributions[f]),
        ...entries.map(([k]) => k).filter((k) => !topFeatures.includes(k)),
      ]
    : entries.map(([k]) => k);

  return (
    <div className="p-6 rounded-xl bg-surface-container-low border border-outline-variant/10">
      <h3 className="font-headline font-bold text-base mb-4">Feature Projections</h3>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {sorted.map((feature) => {
          const dist = featureDistributions[feature];
          if (!dist) return null;
          const current = currentFeatures?.[feature] ?? dist.mean;
          const maxVal = Math.max(current, dist.q90) * 1.1 || 1;

          return (
            <div key={feature} className="p-3 rounded-lg bg-surface-container">
              <div className="flex justify-between items-baseline mb-1">
                <span className="text-xs font-bold text-slate-600 truncate max-w-[60%]">{feature}</span>
                <span className="text-[10px] text-slate-400">
                  now: <span className="font-bold text-on-surface">{current.toFixed(2)}</span>
                </span>
              </div>
              <RangeBar
                current={current}
                q10={dist.q10}
                q50={dist.q50}
                q90={dist.q90}
                max={maxVal}
              />
              <div className="flex justify-between text-[9px] text-slate-400 mt-0.5">
                <span>{dist.q10.toFixed(1)}</span>
                <span className="text-amber-500">{dist.q50.toFixed(1)}</span>
                <span>{dist.q90.toFixed(1)}</span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="flex items-center gap-4 mt-4 text-[10px] text-slate-400">
        <span className="flex items-center gap-1"><span className="w-3 h-1.5 bg-primary rounded-full inline-block" /> Current</span>
        <span className="flex items-center gap-1"><span className="w-3 h-1.5 bg-amber-400 rounded-full inline-block" /> Q50</span>
        <span className="flex items-center gap-1"><span className="w-3 h-1.5 bg-green-200 rounded-full inline-block" /> Q10–Q90 range</span>
      </div>
    </div>
  );
}
