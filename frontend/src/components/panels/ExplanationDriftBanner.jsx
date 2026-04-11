export default function ExplanationDriftBanner({ drift }) {
  if (!drift || !drift.drift_detected) return null;

  return (
    <div className="bg-amber-50 border border-amber-200 rounded-2xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <span className="material-symbols-outlined text-amber-600 text-lg">bolt</span>
        <span className="font-label text-xs font-bold text-amber-700 uppercase tracking-wider">
          Explanation changed since last session
        </span>
        {drift.jsd_score != null && (
          <span className="ml-auto text-[10px] bg-amber-100 text-amber-600 font-bold px-2 py-0.5 rounded-full font-label">
            JSD {drift.jsd_score.toFixed(3)}
          </span>
        )}
      </div>

      <div className="space-y-2">
        {drift.previous_top3?.length > 0 && (
          <div className="flex items-start gap-2">
            <span className="font-label text-[10px] text-slate-500 uppercase tracking-wider w-16 shrink-0 pt-0.5">Previous</span>
            <div className="flex flex-wrap gap-1">
              {drift.previous_top3.map((f) => (
                <span key={f} className="text-[10px] font-label bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full">
                  {f}
                </span>
              ))}
            </div>
          </div>
        )}
        {drift.current_top3?.length > 0 && (
          <div className="flex items-start gap-2">
            <span className="font-label text-[10px] text-slate-500 uppercase tracking-wider w-16 shrink-0 pt-0.5">Current</span>
            <div className="flex flex-wrap gap-1">
              {drift.current_top3.map((f) => (
                <span key={f} className="text-[10px] font-label bg-primary/10 text-primary px-2 py-0.5 rounded-full">
                  {f}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
