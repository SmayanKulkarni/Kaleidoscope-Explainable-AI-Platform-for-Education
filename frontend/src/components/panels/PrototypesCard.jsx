const OUTCOME_STYLES = {
  completed:  'bg-green-100 text-green-700',
  dropped_out: 'bg-red-100 text-red-700',
};

export default function PrototypesCard({ prototypes = [], title = 'Similar Learners', loading }) {
  if (loading) return (
    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10 animate-pulse h-48" />
  );

  return (
    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10">
      <div className="flex items-center gap-2 mb-4">
        <span className="material-symbols-outlined text-primary text-lg">group</span>
        <span className="font-label text-xs uppercase tracking-widest text-slate-500">{title}</span>
      </div>

      {prototypes.length === 0 ? (
        <p className="text-sm text-slate-400 font-label italic text-center py-4">No similar profiles found.</p>
      ) : (
        <div className="space-y-4">
          {prototypes.map((proto, i) => {
            const outcome = proto.outcome ?? 'unknown';
            const outcomeStyle = OUTCOME_STYLES[outcome] ?? 'bg-slate-100 text-slate-500';
            const simPct = Math.round((proto.similarity ?? 0) * 100);

            return (
              <div key={proto.item_id ?? i} className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
                  <span className="material-symbols-outlined text-primary text-sm">person</span>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-label text-xs font-bold truncate">{proto.item_id}</span>
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ml-2 shrink-0 ${outcomeStyle}`}>
                      {outcome.replace('_', ' ')}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-1.5 bg-surface-container-high rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full bg-primary"
                        style={{ width: `${simPct}%` }}
                      />
                    </div>
                    <span className="text-[10px] font-label text-slate-400 w-10 text-right shrink-0">
                      {simPct}% sim
                    </span>
                    <span className="text-[10px] font-mono text-primary font-bold w-12 text-right shrink-0">
                      {(proto.score ?? 0).toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
