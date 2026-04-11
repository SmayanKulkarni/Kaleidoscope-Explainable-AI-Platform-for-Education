import { useState } from 'react';

const RANK_COLORS = ['#f59e0b', '#94a3b8', '#cd7f32'];

export default function RecommendationList({ recommendations, onSelect, loading }) {
  const [expanded, setExpanded] = useState(null);

  if (loading) return (
    <div className="space-y-3">
      {[1, 2, 3].map((i) => (
        <div key={i} className="bg-surface-container-lowest rounded-xl p-4 border border-outline-variant/10 animate-pulse h-16" />
      ))}
    </div>
  );

  const items = recommendations ?? [];

  if (items.length === 0) return (
    <div className="text-slate-400 text-sm font-label text-center py-8">No recommendations available.</div>
  );

  return (
    <div className="space-y-3">
      {items.map((item, idx) => {
        const isOpen = expanded === idx;
        const score  = item.score ?? 0;
        const topFeats = item.top_features ?? [];

        return (
          <div
            key={item.item_id ?? idx}
            className="bg-surface-container-lowest rounded-xl border border-outline-variant/10 overflow-hidden hover:border-primary/30 transition-colors"
          >
            <button
              className="w-full text-left px-5 py-4 flex items-center gap-4"
              onClick={() => {
                setExpanded(isOpen ? null : idx);
                onSelect?.(item);
              }}
            >
              <span
                className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-black text-white shrink-0"
                style={{ background: RANK_COLORS[idx] ?? '#6366f1' }}
              >
                {idx + 1}
              </span>
              <div className="flex-1 min-w-0">
                <p className="font-label text-sm font-bold truncate">{item.item_id}</p>
                <div className="flex gap-2 mt-1 flex-wrap">
                  {topFeats.slice(0, 3).map((f) => (
                    <span key={f} className="text-[10px] bg-primary/8 text-primary px-2 py-0.5 rounded-full font-label">{f}</span>
                  ))}
                </div>
              </div>
              <div className="text-right shrink-0">
                <p className="text-lg font-headline font-black text-primary">{score.toFixed(3)}</p>
                <p className="text-[10px] font-label text-slate-400">score</p>
              </div>
              <span className="material-symbols-outlined text-slate-400 text-lg shrink-0">
                {isOpen ? 'expand_less' : 'expand_more'}
              </span>
            </button>

            {isOpen && item.shap_values && (
              <div className="px-5 pb-5 border-t border-outline-variant/10">
                <p className="text-[11px] font-label text-slate-500 uppercase tracking-wider mt-3 mb-2">SHAP contributions</p>
                <div className="space-y-2">
                  {Object.entries(item.shap_values)
                    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                    .slice(0, 6)
                    .map(([feat, val]) => (
                      <div key={feat} className="flex items-center gap-2">
                        <span className="text-[11px] font-label text-slate-500 w-40 truncate">{feat}</span>
                        <div className="flex-1 h-2 bg-surface-container-high rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${Math.min(100, Math.abs(val) * 200)}%`,
                              background: val >= 0 ? '#6366f1' : '#f87171',
                            }}
                          />
                        </div>
                        <span className={`text-[11px] font-mono w-14 text-right ${val >= 0 ? 'text-indigo-600' : 'text-red-500'}`}>
                          {val >= 0 ? '+' : ''}{val.toFixed(3)}
                        </span>
                      </div>
                    ))}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
