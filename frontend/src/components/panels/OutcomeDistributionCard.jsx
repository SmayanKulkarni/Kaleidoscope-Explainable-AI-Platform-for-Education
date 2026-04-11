import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { getRiskColor } from '../../lib/colors';

function pct(v) {
  return `${(v * 100).toFixed(0)}%`;
}

export default function OutcomeDistributionCard({ outcome, targetWeek }) {
  if (!outcome) return null;

  const {
    dropout_prob_mean,
    dropout_prob_q10,
    dropout_prob_q50,
    dropout_prob_q90,
    dropout_rate,
    dropout_prob_std,
  } = outcome;

  const confidence =
    dropout_prob_std < 0.08 ? 'High' : dropout_prob_std < 0.16 ? 'Medium' : 'Low';

  const chartData = [
    { label: 'Best case (Q10)', value: dropout_prob_q10, fill: '#22c55e' },
    { label: 'Median (Q50)', value: dropout_prob_q50, fill: '#f59e0b' },
    { label: 'Worst case (Q90)', value: dropout_prob_q90, fill: '#ef4444' },
  ];

  const riskColor = getRiskColor(dropout_prob_mean);

  return (
    <div className="p-6 rounded-xl bg-surface-container-low border border-outline-variant/10 space-y-5">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-bold text-slate-500 uppercase tracking-wider">
            Projected Risk at Week {targetWeek}
          </p>
          <div className="flex items-baseline gap-3 mt-1">
            <span className={`text-4xl font-headline font-extrabold ${riskColor}`}>
              {pct(dropout_prob_mean)}
            </span>
            <span className="text-sm text-slate-500">mean dropout prob.</span>
          </div>
        </div>
        <span className="text-xs font-bold px-2.5 py-1 rounded-full bg-slate-100 text-slate-600">
          Confidence: {confidence}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-3 text-center">
        {[
          { label: 'Best case', value: dropout_prob_q10, color: 'text-green-600' },
          { label: 'Median', value: dropout_prob_q50, color: 'text-amber-500' },
          { label: 'Worst case', value: dropout_prob_q90, color: 'text-red-500' },
        ].map(({ label, value, color }) => (
          <div key={label} className="p-3 rounded-lg bg-surface-container">
            <p className="text-[10px] text-slate-400 font-bold uppercase tracking-wide mb-1">{label}</p>
            <p className={`text-xl font-headline font-bold ${color}`}>{pct(value)}</p>
          </div>
        ))}
      </div>

      <div className="h-24">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={[
            { x: 'Q10', risk: dropout_prob_q10 },
            { x: 'Q50', risk: dropout_prob_q50 },
            { x: 'Q90', risk: dropout_prob_q90 },
          ]}>
            <defs>
              <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="x" tick={{ fontSize: 10 }} />
            <YAxis domain={[0, 1]} hide />
            <Tooltip formatter={(v) => pct(v)} />
            <ReferenceLine y={0.5} stroke="#ef4444" strokeDasharray="3 3" />
            <Area type="monotone" dataKey="risk" stroke="#6366f1" fill="url(#riskGrad)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <p className="text-sm text-slate-500 text-center">
        <span className="font-bold text-on-surface">{pct(dropout_rate)}</span> of simulated futures predict dropout
      </p>
    </div>
  );
}
