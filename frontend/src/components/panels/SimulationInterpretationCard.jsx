export default function SimulationInterpretationCard({ outcome }) {
  if (!outcome) return null;

  const { dropout_prob_q10, dropout_prob_q50, dropout_prob_q90, dropout_rate } = outcome;

  let icon, title, body, colorClass;

  if (dropout_prob_q90 > 0.7) {
    icon = 'warning';
    colorClass = 'border-red-300 bg-red-50';
    title = 'High Risk in Pessimistic Scenarios';
    body = `In the most pessimistic simulated futures your risk climbs above ${(dropout_prob_q90 * 100).toFixed(0)}%. Taking action now — especially around attendance and assignment deadlines — can shift you toward the optimistic path (${(dropout_prob_q10 * 100).toFixed(0)}%).`;
  } else if (dropout_prob_q10 < 0.3 && dropout_prob_q90 < 0.6) {
    icon = 'check_circle';
    colorClass = 'border-green-300 bg-green-50';
    title = 'Risk Stays Manageable';
    body = `Your risk stays below 60% across most simulated paths. Median projection: ${(dropout_prob_q50 * 100).toFixed(0)}%. Keep up current engagement patterns to stay on track.`;
  } else {
    icon = 'info';
    colorClass = 'border-amber-300 bg-amber-50';
    title = 'Mixed Outlook';
    body = `${(dropout_rate * 100).toFixed(0)}% of simulated futures predict dropout. Median risk is ${(dropout_prob_q50 * 100).toFixed(0)}%. The spread between best (${(dropout_prob_q10 * 100).toFixed(0)}%) and worst (${(dropout_prob_q90 * 100).toFixed(0)}%) case is large — small behavior changes now have an outsized effect on outcomes.`;
  }

  return (
    <div className={`p-5 rounded-xl border ${colorClass} flex gap-4 items-start`}>
      <span className="material-symbols-outlined text-2xl mt-0.5 shrink-0"
        style={{ fontVariationSettings: "'FILL' 1" }}>
        {icon}
      </span>
      <div>
        <p className="font-bold text-sm mb-1">{title}</p>
        <p className="text-sm text-slate-600 leading-relaxed">{body}</p>
      </div>
    </div>
  );
}
