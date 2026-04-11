export default function TrustScoreCard({ trustScore, loading }) {
  if (loading) return (
    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10 animate-pulse h-48" />
  );

  const ts     = trustScore ?? {};
  const score  = ts.trust_score ?? ts.stability ?? 0;
  const band   = ts.label ?? (score < 0.5 ? 'Low' : score < 0.75 ? 'Medium' : 'High');
  const pct    = Math.round(score * 100);

  const meters = [
    { label: 'Fidelity',     val: ts.fidelity     ?? 0, color: '#6366f1' },
    { label: 'Stability',    val: ts.stability     ?? 0, color: '#3b82f6' },
    { label: 'Completeness', val: ts.completeness  ?? 0, color: '#06b6d4' },
  ];

  return (
    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10">
      <div className="flex justify-between items-center mb-4">
        <span className="font-label text-xs uppercase tracking-widest text-slate-500">Trust Score</span>
        <span className="text-2xl font-headline font-black text-primary">{pct}%</span>
      </div>
      <div className="mb-4">
        <span className="text-xs font-bold uppercase text-slate-400 font-label">{band} Confidence</span>
      </div>
      <div className="space-y-3">
        {meters.map(({ label, val, color }) => (
          <div key={label}>
            <div className="flex justify-between text-[11px] font-label mb-1">
              <span className="text-slate-500">{label}</span>
              <span className="font-bold" style={{ color }}>{Math.round(val * 100)}%</span>
            </div>
            <div className="h-1.5 bg-surface-container-high rounded-full overflow-hidden">
              <div className="h-full rounded-full transition-all" style={{ width: `${Math.round(val * 100)}%`, background: color }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
