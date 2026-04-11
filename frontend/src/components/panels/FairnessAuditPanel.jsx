import { useState } from 'react';

function MetaChip({ label, value }) {
  return (
    <span className="inline-flex items-center gap-1 text-[10px] font-label bg-surface-container px-2 py-0.5 rounded-full text-slate-500">
      <span className="font-bold text-slate-600">{label}</span>
      {value}
    </span>
  );
}

function NarrationCard({ explanation, explaining, onExplain }) {
  if (explaining) {
    return (
      <div className="mt-4 rounded-xl border border-outline-variant/10 bg-surface-container p-4 space-y-2 animate-pulse">
        <div className="h-3 w-1/3 bg-slate-200 rounded" />
        <div className="h-2 w-full bg-slate-100 rounded" />
        <div className="h-2 w-5/6 bg-slate-100 rounded" />
        <div className="h-2 w-3/4 bg-slate-100 rounded" />
      </div>
    );
  }

  if (!explanation) {
    return (
      <div className="mt-4">
        <button
          onClick={onExplain}
          className="flex items-center gap-2 px-4 py-2 rounded-xl border border-primary/30 text-primary text-xs font-bold hover:bg-primary/5 transition-colors"
        >
          <span className="material-symbols-outlined text-base">auto_awesome</span>
          Explain this check with AI
        </button>
      </div>
    );
  }

  const sourceLabel = explanation.source === 'groq' ? 'Groq llama-3.3' : 'Static fallback';
  const sections = [
    { key: 'methodology', icon: 'settings',    label: 'Methodology' },
    { key: 'findings',    icon: 'search',       label: 'Findings' },
    { key: 'verdict',     icon: 'gavel',        label: 'Verdict' },
  ];

  return (
    <div className="mt-4 rounded-xl border border-primary/15 bg-gradient-to-br from-primary/5 to-transparent p-4 space-y-3">
      <div className="flex items-center gap-2 mb-1">
        <span className="material-symbols-outlined text-primary text-base">auto_awesome</span>
        <span className="font-label text-xs font-bold text-primary uppercase tracking-wider">AI Explanation</span>
        <span className="ml-auto text-[9px] text-slate-400 font-label">{sourceLabel}</span>
      </div>

      {sections.map(({ key, icon, label }) =>
        explanation[key] ? (
          <div key={key} className="flex gap-2.5">
            <span className="material-symbols-outlined text-slate-400 text-sm mt-0.5 shrink-0">{icon}</span>
            <div>
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-0.5">{label}</p>
              <p className="text-xs text-on-surface leading-relaxed">{explanation[key]}</p>
            </div>
          </div>
        ) : null
      )}

      {Array.isArray(explanation.action_items) && explanation.action_items.length > 0 && (
        <div className="flex gap-2.5">
          <span className="material-symbols-outlined text-amber-500 text-sm mt-0.5 shrink-0">checklist</span>
          <div>
            <p className="text-[10px] font-bold text-slate-500 uppercase tracking-wider mb-1">Action Items</p>
            <ul className="space-y-1">
              {explanation.action_items.map((item, i) => (
                <li key={i} className="text-xs text-on-surface leading-relaxed flex gap-1.5">
                  <span className="text-amber-500 shrink-0">•</span>
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      <div className="pt-1 border-t border-outline-variant/10">
        <button
          onClick={onExplain}
          className="text-[10px] text-slate-400 hover:text-primary transition-colors font-label flex items-center gap-1"
        >
          <span className="material-symbols-outlined text-xs">refresh</span>
          Re-run explanation
        </button>
      </div>
    </div>
  );
}

export default function FairnessAuditPanel({ report, loading, explanation, explaining, onExplain }) {
  const [openGroup, setOpenGroup] = useState(null);

  if (loading) return (
    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10 animate-pulse h-40" />
  );

  if (!report) return null;

  const meta     = report.metadata ?? {};
  const flaggedSet = new Set(
    (report.flagged_disparities ?? []).map((d) => `${d.feature}::${d.group}`)
  );
  const flaggedMap = Object.fromEntries(
    (report.flagged_disparities ?? []).map((d) => [`${d.feature}::${d.group}`, d])
  );

  const hasGroupData = Object.keys(report.group_scores ?? {}).length > 0;
  const missingFeatures = (meta.protected_features_checked ?? []).filter(
    (f) => !(meta.features_with_data ?? []).includes(f)
  );

  return (
    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10 space-y-4">

      {/* Header */}
      <div className="flex items-center gap-3">
        <span className="material-symbols-outlined text-lg text-slate-500">balance</span>
        <span className="font-label text-xs uppercase tracking-widest text-slate-500">Fairness Audit</span>
        <span
          className={`ml-auto text-[10px] font-bold px-2.5 py-0.5 rounded-full ${
            report.overall_fair
              ? 'bg-green-100 text-green-700'
              : 'bg-red-100 text-red-700'
          }`}
        >
          {report.overall_fair ? '✓ Fair' : `⚠ ${report.flagged_disparities?.length ?? 0} Disparit${(report.flagged_disparities?.length ?? 0) === 1 ? 'y' : 'ies'} Detected`}
        </span>
      </div>

      {/* Metadata chips */}
      {(meta.item_count || meta.disparity_threshold_pct) && (
        <div className="flex flex-wrap gap-1.5">
          {meta.item_count > 0 && <MetaChip label="Items scored" value={meta.item_count} />}
          {meta.overall_mean > 0 && <MetaChip label="Mean score" value={meta.overall_mean?.toFixed(3)} />}
          {meta.disparity_threshold_pct && <MetaChip label="Threshold" value={`${meta.disparity_threshold_pct?.toFixed(0)}% deviation`} />}
          {(meta.protected_features_checked ?? []).map((f) => (
            <MetaChip key={f} label="Protected" value={f.replace('explicit_', '')} />
          ))}
        </div>
      )}

      {/* Warning when no demographic data found */}
      {!hasGroupData && (
        <div className="flex items-start gap-2 rounded-xl bg-amber-50 border border-amber-200 p-3">
          <span className="material-symbols-outlined text-amber-500 text-sm shrink-0 mt-0.5">warning</span>
          <div>
            <p className="text-xs font-bold text-amber-700 mb-0.5">No demographic data in scored items</p>
            <p className="text-[11px] text-amber-600 leading-relaxed">
              Protected features{missingFeatures.length > 0 ? ` (${missingFeatures.join(', ')})` : ''} were absent from the scored items.
              The "Fair" verdict means no disparity <em>was found</em>, not that the system <em>is</em> globally fair.
              Add protected attributes to scored items for a meaningful audit.
            </p>
          </div>
        </div>
      )}

      {/* Group score breakdown */}
      {hasGroupData && (
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
                  <span className="font-label text-xs font-bold text-slate-700">
                    {feature.replace('explicit_', '')}
                  </span>
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
                    {/* Reference line: overall mean */}
                    {meta.overall_mean > 0 && (
                      <p className="text-[10px] text-slate-400 font-label mb-1">
                        Overall mean: <span className="font-bold text-slate-500">{meta.overall_mean?.toFixed(3)}</span>
                        {' · '}±{meta.disparity_threshold_pct?.toFixed(0)}% threshold
                      </p>
                    )}
                    {Object.entries(groups).map(([group, score]) => {
                      const key = `${feature}::${group}`;
                      const isFlagged = flaggedSet.has(key);
                      const disparity = flaggedMap[key];
                      const pct = Math.min(100, Math.round(score * 100));

                      return (
                        <div key={group} className="flex items-center gap-2">
                          <span className={`font-label text-[11px] w-24 truncate ${isFlagged ? 'text-red-600 font-bold' : 'text-slate-500'}`}>
                            {group}
                          </span>
                          <div className="flex-1 h-1.5 bg-surface-container-high rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all ${isFlagged ? 'bg-red-400' : 'bg-primary'}`}
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                          <span className={`text-[11px] font-label font-bold w-10 text-right ${isFlagged ? 'text-red-600' : 'text-slate-600'}`}>
                            {score.toFixed(3)}
                          </span>
                          {isFlagged && disparity && (
                            <span className="text-[10px] text-red-500 font-label shrink-0">
                              {disparity.direction === 'below_mean' ? '▼' : '▲'}{Math.abs(disparity.deviation_pct).toFixed(0)}%
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
      )}

      {/* LLM Narration */}
      <NarrationCard explanation={explanation} explaining={explaining} onExplain={onExplain} />
    </div>
  );
}
