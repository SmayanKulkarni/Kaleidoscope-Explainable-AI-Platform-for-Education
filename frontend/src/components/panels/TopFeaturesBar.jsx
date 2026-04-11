import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer, ReferenceLine,
} from 'recharts';

const POS_COLOR = '#f87171';
const NEG_COLOR = '#34d399';

export default function TopFeaturesBar({ shapValues, topFeatures, loading }) {
  if (loading) return (
    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10 animate-pulse h-72" />
  );

  const features = topFeatures ?? [];
  const sv       = shapValues  ?? {};

  const data = features
    .slice(0, 8)
    .map((f) => {
      const name = f.name ?? f;
      const val  = sv[name] ?? f.shap ?? 0;
      return { name, value: parseFloat(val.toFixed(4)) };
    })
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  return (
    <div className="bg-surface-container-lowest rounded-2xl p-6 border border-outline-variant/10">
      <div className="flex justify-between items-center mb-4">
        <span className="font-label text-xs uppercase tracking-widest text-slate-500">SHAP Feature Attribution</span>
        <div className="flex gap-3 text-[10px] font-label">
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-sm" style={{ background: POS_COLOR }} />Increases risk</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-sm" style={{ background: NEG_COLOR }} />Reduces risk</span>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 32, top: 4, bottom: 4 }}>
          <XAxis type="number" tickFormatter={(v) => v.toFixed(2)} tick={{ fontSize: 10 }} />
          <YAxis type="category" dataKey="name" width={160} tick={{ fontSize: 10 }} />
          <Tooltip formatter={(v) => v.toFixed(4)} />
          <ReferenceLine x={0} stroke="#94a3b8" strokeWidth={1} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((d) => (
              <Cell key={d.name} fill={d.value >= 0 ? POS_COLOR : NEG_COLOR} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
