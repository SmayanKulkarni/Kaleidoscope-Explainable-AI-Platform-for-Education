import { useState } from 'react';

export default function FairnessAuditPanel({ report, loading }) {
  const [openGroup, setOpenGroup] = useState(null);

  if (loading) return (
    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10 animate-pulse h-40" />
  );

  if (!report) return null;

  const flaggedSet = new Set(
    (report.flagged_disparities ?? []).map((d) => `${d.feature}::${d.group}`)
  );

  const flaggedMap = Object.fromEntries(
    (report.flagged_disparities ?? []).map((d) => [`${d.feature}::${d.group}`, d])
  );

  return (
    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10">
      <div className="flex items-center gap-3 mb-4">
        <span className="material-symbols-outlined text-lg text-slate-500">balance</span>
        <span className="font-label text-xs uppercase tracking-widest text-slate-500">Fairness Audit</span>
        <span
          className={`ml-auto text-[10px] font-bold px-2.5 py-0.5 rounded-full ${
            report.overall_fair
              ? 'bg-green-100 text-green-700'
              : 'bg-red-100 text-red-700'
          }`}
        >
          {report.overall_fair ? '✓ Fair' : '⚠ Disparities Detected'}
        </span>
      </div>

      <div className="space-y-2">
        {Object.entries(report.group_scores ?? {}).map(([feature, groups]) => {
          const isOpen = openGroup === feature;
          const hasFlagged = Object.keys(groups).some((g) => flaggedSet.has(`${feature}::${g}`));

          return (
            <div key={feature} className="border border-outline-variant/10 rounded-xl overflow-hidden">
              <button
                className="w-full flex items-center justify-between px-4 py-2.5 text-left hover:bg-surface-container/50 transition-colors"
                onClick={() => setOpenGroup(isOpen ? null : feature)}
              >
                <span className="font-label text-xs font-bold text-slate-700">{feature}</span>
                <div className="flex items-center gap-2">
                  {hasFlagged && (
                    <span className="text-[10px] font-bold text-red-500">⚑ flagged</span>
                  )}
                  <span className="material-symbols-outlined text-slate-400 text-base">
                    {isOpen ? 'expand_less' : 'expand_more'}
                  </span>
                </div>
              </button>

              {isOpen && (
                <div className="px-4 pb-3 space-y-2 border-t border-outline-variant/10 pt-3">
                  {Object.entries(groups).map(([group, score]) => {
                    const key = `${feature}::${group}`;
                    const isFlagged = flaggedSet.has(key);
                    const disparity = flaggedMap[key];
                    const pct = Math.round(score * 100);

                    return (
                      <div key={group} className="flex items-center gap-2">
                        <span className={`font-label text-[11px] w-24 truncate ${isFlagged ? 'text-red-600 font-bold' : 'text-slate-500'}`}>
                          {group}
                        </span>
                        <div className="flex-1 h-1.5 bg-surface-container-high rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${isFlagged ? 'bg-red-400' : 'bg-primary'}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className={`text-[11px] font-label font-bold w-10 text-right ${isFlagged ? 'text-red-600' : 'text-slate-600'}`}>
                          {score.toFixed(2)}
                        </span>
                        {isFlagged && disparity && (
                          <span className="text-[10px] text-red-500 font-label shrink-0">
                            {disparity.direction === 'below' ? '▼' : '▲'}{disparity.deviation_pct.toFixed(0)}%
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
