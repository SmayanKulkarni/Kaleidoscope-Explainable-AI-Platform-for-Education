import { RadialBarChart, RadialBar, ResponsiveContainer } from 'recharts';
import { riskColor } from '../../lib/colors';

export default function RiskScoreCard({ riskScore, riskLabel, modelVersion, loading }) {
  const score = riskScore ?? 0;
  const color = riskColor(score);
  const pct   = Math.round(score * 100);
  const label = riskLabel ?? (score < 0.3 ? 'low' : score < 0.6 ? 'medium' : 'high');

  if (loading) return (
    <div className="bg-surface-container-lowest rounded-2xl p-8 border border-outline-variant/10 animate-pulse h-52" />
  );

  return (
    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10 flex flex-col gap-4">
      <span className="font-label text-xs uppercase tracking-widest text-slate-500">Dropout Risk</span>
      <div className="flex items-center gap-6">
        <div className="relative w-28 h-28 shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <RadialBarChart
              cx="50%" cy="50%"
              innerRadius="60%" outerRadius="100%"
              startAngle={220} endAngle={-40}
              data={[{ value: pct, fill: color }]}
            >
              <RadialBar dataKey="value" cornerRadius={6} background={{ fill: '#e2e8f0' }} />
            </RadialBarChart>
          </ResponsiveContainer>
          <span
            className="absolute inset-0 flex items-center justify-center text-2xl font-headline font-black"
            style={{ color }}
          >
            {pct}%
          </span>
        </div>
        <div className="flex flex-col gap-1">
          <span
            className="text-sm font-bold uppercase px-3 py-1 rounded-full border"
            style={{ color, borderColor: color, background: `${color}15` }}
          >
            {label} risk
          </span>
          {modelVersion && (
            <span className="text-[10px] font-label text-slate-400">Model: {modelVersion}</span>
          )}
        </div>
      </div>
    </div>
  );
}
