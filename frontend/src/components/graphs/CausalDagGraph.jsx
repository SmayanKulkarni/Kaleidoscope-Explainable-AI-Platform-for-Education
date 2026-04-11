import { useRef, useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import ForceGraph3D from 'react-force-graph-3d';
import { causalDagToGraph } from '../../lib/graphTransforms';
import { useCausalGraph } from '../../hooks/useCausalGraph';
import { explainCausalGraph } from '../../api/causal';

const COLOR_RE = /^(#(?:[0-9a-fA-F]{3,8})|rgba?\([^\)]+\)|hsla?\([^\)]+\))$/;
const safeColor = (value, fallback) => {
  if (typeof value !== 'string') return fallback;
  const normalized = value.trim();
  return COLOR_RE.test(normalized) ? normalized : fallback;
};

const GROUP_COLORS = {
  confounder:  '#94a3b8',
  engagement:  '#60a5fa',
  performance: '#34d399',
  risk_signal: '#f87171',
  outcome:     '#fbbf24',
};

const LEGEND_ENTRIES = Object.entries(GROUP_COLORS);

const DIRECTION_ICONS = {
  positive:  '↑',
  negative:  '↓',
  neutral:   '→',
};

export default function CausalDagGraph({ explanation, width, height = 400 }) {
  const fgRef = useRef();
  const { data: apiGraph, isLoading, error } = useCausalGraph();

  // Auto-load the LLM explanation
  const { data: dagExplanation, isLoading: explainLoading } = useQuery({
    queryKey: ['causal-dag-explanation'],
    queryFn: explainCausalGraph,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  });

  const graphData = useMemo(() => {
    if (apiGraph?.nodes?.length) {
      return {
        nodes: apiGraph.nodes.map((n) => ({
          id:    n.id,
          name:  n.label ?? n.id,
          val:   n.is_causal ? Math.abs(n.ate ?? 0) * 300 + 10 : 6,
          color: safeColor(GROUP_COLORS[n.group], '#94a3b8'),
          group: n.group,
        })),
        links: apiGraph.edges.map((e) => ({
          source: e.from,
          target: e.to,
          color:  '#6b7280',
        })),
      };
    }
    return causalDagToGraph(explanation);
  }, [apiGraph, explanation]);

  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.d3Force('charge')?.strength(-100);
    }
  }, [graphData]);

  if (isLoading) return (
    <div className="flex items-center justify-center h-full text-slate-400 text-sm font-label animate-pulse">
      Loading causal graph…
    </div>
  );

  if (error && !graphData.nodes.length) return (
    <div className="flex items-center justify-center h-full text-slate-400 text-sm font-label">
      Causal graph unavailable.
    </div>
  );

  if (!graphData.nodes.length) return (
    <div className="flex items-center justify-center h-full text-slate-400 text-sm font-label">
      No causal data available.
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="relative w-full" style={{ height }}>
        <ForceGraph3D
          ref={fgRef}
          graphData={graphData}
          width={width}
          height={height}
          backgroundColor="rgba(0,0,0,0)"
          nodeLabel="name"
          nodeVal="val"
          nodeColor={(n) => safeColor(n?.color, '#94a3b8')}
          linkColor={(l) => safeColor(l?.color, '#6b7280')}
          linkWidth={1.5}
          linkDirectionalArrowLength={4}
          linkDirectionalArrowRelPos={1}
          linkLineDash={(l) => (l.dashed ? [2, 3] : null)}
          controlType="orbit"
        />
        {apiGraph?.nodes?.length > 0 && (
          <div className="absolute top-2 right-2 bg-surface-container-lowest/80 backdrop-blur-sm rounded-xl p-2 space-y-1 text-[10px] font-label border border-outline-variant/10">
            {LEGEND_ENTRIES.map(([group, color]) => (
              <div key={group} className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: color }} />
                <span className="text-slate-500 capitalize">{group.replace('_', ' ')}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Auto-loaded DAG explanation */}
      {explainLoading ? (
        <div className="animate-pulse bg-surface-container rounded-xl p-5 space-y-3">
          <div className="h-3 bg-slate-200 rounded w-3/4" />
          <div className="h-3 bg-slate-200 rounded w-1/2" />
          <div className="h-3 bg-slate-200 rounded w-2/3" />
        </div>
      ) : dagExplanation ? (
        <div className="bg-surface-container-lowest rounded-xl p-5 border border-outline-variant/10 space-y-4">
          <div className="flex items-center gap-2 mb-1">
            <span className="material-symbols-outlined text-primary text-lg">auto_awesome</span>
            <h3 className="font-headline font-bold text-sm text-on-surface">AI Causal Insight</h3>
            {dagExplanation.source && (
              <span className="text-[9px] font-label text-slate-400 uppercase tracking-wider ml-auto px-2 py-0.5 rounded-full bg-surface-container">
                {dagExplanation.source === 'groq' ? '✨ LLM' : 'Rule-based'}
              </span>
            )}
          </div>

          {dagExplanation.summary && (
            <p className="text-sm text-on-surface-variant leading-relaxed">
              {dagExplanation.summary}
            </p>
          )}

          {dagExplanation.key_insights?.length > 0 && (
            <div className="space-y-1.5">
              <p className="text-[10px] font-label text-slate-500 uppercase tracking-wider">Key Insights</p>
              <ul className="space-y-1">
                {dagExplanation.key_insights.map((insight, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-on-surface-variant">
                    <span className="text-primary font-bold mt-0.5 shrink-0">•</span>
                    <span>{insight}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {dagExplanation.strongest_drivers?.length > 0 && (
            <div className="space-y-2">
              <p className="text-[10px] font-label text-slate-500 uppercase tracking-wider">Strongest Causal Drivers</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {dagExplanation.strongest_drivers.map((d, i) => (
                  <div key={i} className="flex items-center gap-2 p-2 rounded-lg bg-surface-container/50 text-xs">
                    <span className={`text-sm font-bold ${
                      d.direction === 'positive' ? 'text-red-500' :
                      d.direction === 'negative' ? 'text-green-500' : 'text-slate-400'
                    }`}>
                      {DIRECTION_ICONS[d.direction] || '→'}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="font-bold text-on-surface truncate">{d.feature}</p>
                      {d.explanation && (
                        <p className="text-[10px] text-slate-500 leading-tight mt-0.5">{d.explanation}</p>
                      )}
                    </div>
                    <span className="text-[9px] font-mono text-slate-400 shrink-0">
                      ATE: {typeof d.ate === 'number' ? d.ate.toFixed(4) : d.ate}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}
