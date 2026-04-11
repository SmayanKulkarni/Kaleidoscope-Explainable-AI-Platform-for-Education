export default function CounterfactualCard({ counterfactual, rankedActions }) {
  const actions = rankedActions ?? counterfactual?.actions ?? [];
  if (!actions.length) return null;

  return (
    <div className="p-6 rounded-xl bg-surface-container-low border border-outline-variant/10">
      <div className="flex items-center gap-2 mb-4">
        <span className="material-symbols-outlined text-primary" style={{ fontVariationSettings: "'FILL' 1" }}>
          auto_fix_high
        </span>
        <h3 className="font-headline font-bold text-base">Counterfactual Actions</h3>
        <span className="text-[10px] font-bold bg-primary/10 text-primary px-2 py-0.5 rounded-full ml-auto">
          DiCE
        </span>
      </div>

      <p className="text-xs text-slate-500 mb-4 leading-relaxed">
        If you make the following changes, the model predicts your dropout risk will drop below the threshold.
      </p>

      <div className="space-y-3">
        {actions.slice(0, 5).map((action, i) => (
          <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-surface-container border border-outline-variant/5 hover:border-primary/20 transition-colors">
            <div className="flex items-center gap-3">
              <span className="w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-bold flex items-center justify-center shrink-0">
                {i + 1}
              </span>
              <div>
                <p className="text-sm font-bold text-on-surface">{action.feature ?? action.action}</p>
                {action.current_value !== undefined && (
                  <p className="text-[10px] text-slate-400">
                    {Number(action.current_value).toFixed(2)} → {Number(action.target_value ?? action.suggested_value).toFixed(2)}
                  </p>
                )}
              </div>
            </div>
            {action.impact !== undefined && (
              <span className="text-xs font-bold text-green-600 bg-green-50 px-2 py-1 rounded-lg shrink-0">
                {action.impact > 0 ? '+' : ''}{(action.impact * 100).toFixed(1)}% success
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
