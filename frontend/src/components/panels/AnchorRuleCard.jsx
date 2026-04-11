export default function AnchorRuleCard({ anchorRule, precision, loading }) {
  if (loading) return (
    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10 animate-pulse h-32" />
  );

  const raw = anchorRule?.human_readable ?? anchorRule ?? '';

  const parts = raw
    .replace(/^IF\s+/i, '')
    .replace(/\s+THEN\s+.*$/i, '')
    .split(/\s+AND\s+/i)
    .map((s) => s.trim())
    .filter(Boolean);

  const conclusion = raw.match(/THEN\s+(.*)/i)?.[1] ?? '';

  return (
    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-l-4 border-primary">
      <div className="flex items-center gap-2 mb-4">
        <span className="material-symbols-outlined text-primary text-lg">policy</span>
        <span className="font-label text-xs uppercase tracking-widest text-slate-500">Anchor Rule</span>
        {precision != null && (
          <span className="ml-auto text-[10px] font-bold bg-primary/10 text-primary px-2 py-0.5 rounded-full">
            {Math.round(precision * 100)}% precision
          </span>
        )}
      </div>
      {parts.length === 0 ? (
        <p className="text-slate-400 text-sm font-label italic">No rule generated yet.</p>
      ) : (
        <div className="flex flex-wrap gap-2">
          {parts.map((cond, i) => (
            <span key={i} className="bg-surface-container px-3 py-1 rounded-full text-xs font-label font-medium border border-outline-variant/20">
              {cond}
            </span>
          ))}
          {conclusion && (
            <>
              <span className="px-2 py-1 text-xs font-label text-slate-400">→</span>
              <span className="bg-primary/10 text-primary px-3 py-1 rounded-full text-xs font-label font-bold border border-primary/20">
                {conclusion}
              </span>
            </>
          )}
        </div>
      )}
    </div>
  );
}
