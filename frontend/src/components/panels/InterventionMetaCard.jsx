const URGENCY_STYLES = {
  high:   'bg-red-100 text-red-700',
  medium: 'bg-amber-100 text-amber-700',
  low:    'bg-green-100 text-green-700',
};

const TRAJECTORY_ICONS = {
  declining: '↘',
  stable:    '→',
  improving: '↗',
};

export default function InterventionMetaCard({
  interventionType,
  interventionUrgency,
  contentType,
  effortHours,
  studentRiskScore,
  studentTrajectory,
  cohortDropoutRate,
  instructorArchetype,
  teachingStyle,
  loading,
}) {
  if (loading) return (
    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10 animate-pulse h-48" />
  );

  if (!interventionType && studentRiskScore == null) return null;

  const urgencyStyle = URGENCY_STYLES[interventionUrgency?.toLowerCase()] ?? 'bg-slate-100 text-slate-500';
  const riskPct = studentRiskScore != null ? Math.round(studentRiskScore * 100) : null;
  const riskColor = studentRiskScore > 0.6 ? '#f87171' : studentRiskScore > 0.3 ? '#fbbf24' : '#34d399';
  const trajectoryIcon = TRAJECTORY_ICONS[studentTrajectory] ?? '→';

  return (
    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10">
      <div className="flex items-center gap-2 mb-4">
        <span className="material-symbols-outlined text-primary text-lg">emergency</span>
        <span className="font-label text-xs uppercase tracking-widest text-slate-500">Intervention Context</span>
      </div>

      <div className="space-y-3">
        {(interventionType || interventionUrgency) && (
          <div className="flex items-center gap-2 flex-wrap">
            {interventionType && (
              <span className="text-xs font-label font-bold bg-primary/10 text-primary px-2.5 py-1 rounded-full capitalize">
                {interventionType.replace('_', ' ')}
              </span>
            )}
            {interventionUrgency && (
              <span className={`text-[10px] font-bold uppercase px-2.5 py-1 rounded-full ${urgencyStyle}`}>
                {interventionUrgency} urgency
              </span>
            )}
          </div>
        )}

        {(contentType || effortHours != null) && (
          <div className="flex items-center gap-3 text-xs font-label text-slate-600">
            {contentType && (
              <span className="flex items-center gap-1">
                <span className="material-symbols-outlined text-sm text-slate-400">play_circle</span>
                {contentType.replace('_', ' ')}
              </span>
            )}
            {effortHours != null && (
              <span className="flex items-center gap-1">
                <span className="material-symbols-outlined text-sm text-slate-400">schedule</span>
                {effortHours}h effort
              </span>
            )}
          </div>
        )}

        {riskPct != null && (
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-[11px] font-label text-slate-500">Student Risk</span>
              <div className="flex items-center gap-1">
                <span className="text-sm font-bold font-mono" style={{ color: riskColor }}>{riskPct}%</span>
                {studentTrajectory && (
                  <span className="text-xs font-label text-slate-400">{trajectoryIcon} {studentTrajectory}</span>
                )}
              </div>
            </div>
            <div className="h-1.5 bg-surface-container-high rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{ width: `${riskPct}%`, background: riskColor }}
              />
            </div>
          </div>
        )}

        {cohortDropoutRate != null && (
          <div className="flex items-center justify-between text-xs font-label">
            <span className="text-slate-500">Cohort Dropout Rate</span>
            <span className="font-bold text-slate-700">{Math.round(cohortDropoutRate * 100)}%</span>
          </div>
        )}

        {(instructorArchetype || teachingStyle) && (
          <div className="pt-2 border-t border-outline-variant/10 flex items-center gap-2 text-[11px] font-label text-slate-500">
            <span className="material-symbols-outlined text-sm">school</span>
            {instructorArchetype && <span className="font-bold">{instructorArchetype}</span>}
            {teachingStyle && <span>({teachingStyle})</span>}
          </div>
        )}
      </div>
    </div>
  );
}
