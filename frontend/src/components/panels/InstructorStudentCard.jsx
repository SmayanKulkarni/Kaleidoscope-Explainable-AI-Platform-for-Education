const URGENCY_STYLES = {
  high:   'bg-red-100 text-red-700',
  medium: 'bg-amber-100 text-amber-700',
  low:    'bg-green-100 text-green-700',
};

export default function InstructorStudentCard({ item, onExplain, onWhatIf }) {
  const features = item?.features ?? {};
  const riskScore    = features.student_dropout_risk_score ?? null;
  const trajectory   = features.student_risk_trajectory ?? null;
  const urgency      = features.intervention_urgency ?? null;
  const intType      = features.intervention_type ?? null;
  const contentType  = features.recommended_content_type ?? null;
  const effortHours  = features.estimated_effort_hours ?? null;
  const quizAvg      = features.student_quiz_avg_score ?? null;
  const daysInactive = features.student_days_inactive ?? null;
  const score        = item?.score ?? 0;

  const riskPct   = riskScore != null ? Math.round(riskScore * 100) : null;
  const riskColor = riskScore > 0.6 ? 'bg-red-100 text-red-700' : riskScore > 0.3 ? 'bg-amber-100 text-amber-700' : 'bg-green-100 text-green-700';
  const urgencyStyle = URGENCY_STYLES[urgency?.toLowerCase()] ?? 'bg-slate-100 text-slate-500';

  return (
    <div className="bg-surface-container-lowest rounded-xl border border-outline-variant/10 p-5 hover:border-primary/30 transition-colors">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0">
            <span className="material-symbols-outlined text-primary text-sm">person</span>
          </div>
          <span className="font-label text-sm font-bold truncate">{item?.item_id}</span>
        </div>
        <div className="text-right shrink-0">
          <p className="text-base font-headline font-black text-primary">{score.toFixed(3)}</p>
          <p className="text-[10px] font-label text-slate-400">priority</p>
        </div>
      </div>

      <div className="flex flex-wrap gap-1.5 mb-3">
        {riskPct != null && (
          <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${riskColor}`}>
            Risk {riskPct}%
          </span>
        )}
        {trajectory && (
          <span className="text-[10px] font-label bg-slate-100 text-slate-600 px-2 py-0.5 rounded-full">
            {trajectory === 'declining' ? '↘' : trajectory === 'improving' ? '↗' : '→'} {trajectory}
          </span>
        )}
        {urgency && (
          <span className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded-full ${urgencyStyle}`}>
            {urgency}
          </span>
        )}
      </div>

      <div className="flex flex-wrap gap-3 text-[11px] font-label text-slate-500 mb-3">
        {quizAvg != null && <span>Quiz avg: <strong>{quizAvg.toFixed(1)}</strong></span>}
        {daysInactive != null && <span>Inactive: <strong>{daysInactive}d</strong></span>}
        {intType && <span>Type: <strong className="capitalize">{intType.replace('_', ' ')}</strong></span>}
        {contentType && <span>{contentType.replace('_', ' ')}</span>}
        {effortHours != null && <span>{effortHours}h</span>}
      </div>

      <div className="flex gap-2">
        <button
          onClick={onExplain}
          className="flex-1 py-1.5 bg-primary/10 text-primary rounded-lg text-xs font-bold font-label hover:bg-primary/20 transition-colors"
        >
          Explain ↗
        </button>
        <button
          onClick={onWhatIf}
          className="flex-1 py-1.5 bg-surface-container text-slate-600 rounded-lg text-xs font-bold font-label hover:bg-surface-container-high transition-colors"
        >
          What-If ↗
        </button>
      </div>
    </div>
  );
}
